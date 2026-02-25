# 🚀 LBM CFD 數據預處理筆記：HDF5 to Zarr 架構設計

## 1. 核心目標與痛點解決

- **痛點：** 訓練神經細胞自動機 (NCA) 需要進行高頻率的「局部時空隨機抽樣 (Spacetime Jittering)」。如果直接從巨大的 HDF5 檔案讀取，會因為「邏輯切片與物理存儲塊不對稱」導致嚴重的 **I/O 放大 (I/O Amplification)**，拖垮 DataLoader。
- **解法：** 將連續的 HDF5 轉換為 Zarr 格式，並透過「精心設計的 Chunk Size」與「獨立檔案儲存」來最大化 HDD 的隨機讀取效能。

## 2. 物理分塊 (Chunking) 策略設計

為了平衡記憶體佔用與讀取效率，我們制定了以下黃金法則：

- **時間維度 (T)：採用「大於需求原則 (Over-Sizing)」**
- 若訓練目標 $T_{box} = 100$，則 Zarr 的 $T_{chunk} = 200$。
- **目的：** 降低隨機抽樣時跨越兩個實體檔案的機率，減少硬碟磁頭跳轉。

- **空間維度 (H, W)：固定高度，寬度切分**
- 若模擬場域為 $104 \times 512$ 或未來的 $256 \times 1260$，設定 $W_{chunk} = 256$。
- **尾部對齊策略 (Overlapping Alignment)：** 當寬度無法整除時（如 1260），**不使用**補零 (Padding) 或非對稱剩餘塊，而是讓最後一塊**往回靠攏**（例如 `[1004:1260]`），確保所有 Chunk 都是統一的形狀（如 $256 \times 256$）。這能讓 L2 讀取器的代碼保持極度簡潔。

## 3. 數值與型態安全 (Float16 + Normalization)

LBM 的非平衡態矩（如應力張量）數值極小，直接轉 `float16` 會導致 Underflow (下溢位變成 0)，破壞物理特徵。

- **兩階段處理 (Two-Pass Pipeline)：**

1. **Pass 1 (計算全域統計)：** 掃描所有 `status == "Success"` 的案例，計算 9 個通道的 `Global_Mean` 與 `Global_Std`，並避開數值爆炸的 Failed Cases。
2. **Pass 2 (正規化與儲存)：** 讀取數據 $\rightarrow$ Channel-wise 正規化 $(x - \mu) / \sigma$ $\rightarrow$ 轉型為 `float16` $\rightarrow$ 寫入 Zarr。

- **防呆機制：** 計算變異數 (Variance) 時需加上 `np.maximum(var, 1e-10)`，防止穩態通道除以零。

## 4. Zarr v3 實作規範 (API 更新)

在最新的 Zarr v3 架構中，必須遵循新的 API 規範以確保相容性與最高效能：

- **建立陣列：** 棄用 `require_dataset`，全面改用 **`require_array`**。
- **壓縮器設定：** 棄用 `numcodecs` 傳入 `compressor`，改用 Zarr 內建的 `BloscCodec` 傳入列表 **`compressors`**。

```python
from zarr.codecs import BloscCodec
compressors = [BloscCodec(cname='zstd', clevel=5, shuffle='bitshuffle')]
root.require_array('turbulence', shape=(...), chunks=(...), dtype=np.float16, compressors=compressors)

```

## 5. 最終檔案結構 (L1 Archive)

採用「一案一目錄」結構，並將全域統計值獨立拉出，供 L2 DataLoader 初始化時讀取：

```text
L1_Zarr_Archive/
├── global_stats.json             # 【必備】全域通道的 Mean/Std 與成功案例清單
├── L200_0001.zarr/               # 單一案子目錄
│   ├── turbulence/               # 核心物理場，被切分成多個 200x9x256x256 的二進制小檔
│   ├── static_mask/              # 靜態邊界遮罩 (保留原始型態)
│   └── mean_vel_field/           # 時間平均流場 (經全域正規化 + Float16)
└── L200_0003.zarr/

```
