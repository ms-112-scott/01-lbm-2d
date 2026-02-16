# Gemini Code Companion: lbm_2d (Taichi MRT-LBM Solver)

## 專案概述 (Project Overview)

`lbm_2d` 是一個基於 **Taichi** 語言的高效能二維 **Lattice Boltzmann Method (LBM)** 計算流體力學求解器。

此專案專為 **AI 代理模型 (Surrogate Model) 訓練數據生成** 而設計，核心目標是批量生產高保真的流場數據（速度、密度、渦度、MRT矩）。它採用 **MRT (Multiple Relaxation Time)** 碰撞模型結合 **LES (Large Eddy Simulation)** 湍流模型，確保在高雷諾數下的數值穩定性。

**核心特性:**

- **GPU 加速:** 利用 Taichi 的大規模並行計算能力。
- **MRT-LBM:** 多重鬆弛時間模型，比傳統 LBM (SRT/BGK) 更穩定且準確。
- **LES 湍流模型:** Smagorinsky 次網格模型，用於模擬高雷諾數流動。
- **批量自動化:** 支援單一配置批量跑圖，或 Config-Mask 一對一的高級批量模式。
- **物理約束生成:** 內建遮罩生成器，可產生符合物理限制 (阻塞率、間距) 的隨機障礙物。
- **數據導向:** 輸出 HDF5 (`.h5`) 格式，包含完整的物理場與 MRT 矩 (Moments)，便於深度學習使用。

---

## 專案結構 (Project Structure)

專案已採用模組化架構，將配置、文檔、輸出與源碼分離。

```text
C:\Users\GAI\Desktop\NCA_workspace\01-lbm-2d\
├── config.yaml                     # [核心] 預設全域配置文件
├── requirements.txt                # Python 依賴庫
├── configs/                        # [配置] 實驗與基準測試配置
│   ├── templates/                  # 模板 (config_template.yaml)
│   ├── experiments/                # 批量實驗 (Hyper Configs)
│   └── benchmarks/                 # 基準測試
├── docs/                           # [文件] 專案文檔與筆記 (原 my_docs)
├── notebooks/                      # [實驗] Jupyter Notebooks
├── outputs/                        # [輸出] 模擬結果與日誌
│   ├── simulation_data/            # HDF5 數據 (原 output/dataset)
│   ├── benchmarks/                 # 基準測試結果
│   └── visualization/              # 圖片與影片
├── src/
│   ├── tools/                      # [工具] 地圖與遮罩生成 (原 generators)
│   │   ├── mask_rect_gen.py        # 隨機矩形生成
│   │   ├── hybrid_map_gen.py       # 混合地圖生成
│   │   └── config_batch_gen.py     # 批量配置生成
│   ├── analysis/                   # [分析] 後處理與統計 (原 post_process)
│   └── lbm_mrt_les/                # [核心] LBM 求解器
│       ├── engine/                 # 物理運算
│       ├── runners/                # 執行入口
│       └── io/                     # 數據讀寫
└── archive/                        # [歸檔] 舊版代碼 (原 old)
```

---

## 快速開始 (Quick Start)

### 1. 安裝依賴

確保已安裝 Python 3.8 ~ 3.13 與 CUDA (若使用 GPU)。

```bash
pip install -r requirements.txt
```

### 2. 生成遮罩數據 (Optional)

如果沒有現成的幾何遮罩，可使用生成器產生一批隨機障礙物：

```bash
python src/tools/mask_rect_gen.py
```

- 輸出: `src/tools/rect_masks/*.png`

### 3. 執行模擬

#### 模式 A: 標準批量 (單一 Config -> 多個 Masks)

適用於固定物理條件 (如固定 Re)，跑多個不同幾何形狀。

```bash
python -m src.lbm_mrt_les.runners.run_one_case --config configs/templates/config_template.yaml --mask_dir src/tools/rect_masks
```

#### 模式 B: 進階批量 (Config-Mask 一對一)

適用於產生多樣化物理條件 (不同 Re, 不同解析度) 的數據集。需確保 Config 檔案數量與 Mask 檔案數量一致且排序對應。

```bash
python -m src.lbm_mrt_les.runners.run_multi_case --config_dir configs/experiments --mask_dir src/tools/hybrid_maps
```

---

## 模擬組態 (`config.yaml`)

設定檔採 YAML 格式，支援 "Strict Mode" (缺鍵即報錯)。

**關鍵參數:**

```yaml
simulation:
  name: "Building_Wind_Sim_multichannel"
  nu: 0.03 # LBM 黏滯係數 (控制 Re)
  nx: 512 # 網格寬度
  ny: 256 # 網格高度
  max_steps: 10000 # 最大模擬步數 (工程限制)
  save_npy: true # (Legacy) 是否存 npy
  steps_per_batch: 100 # GUI/Console 刷新頻率

boundaries:
  # 0: Velocity Inlet, 1: Outflow, 2: Free Slip
  types: [0, 0, 1, 0] # Left, Top, Right, Bottom
  values:
    - [0.1, 0.0] # Inlet Velocity (LBM units)

obstacle:
  use_mask: true
  mask_dir: "outputs/simulation_data/masks"

outputs:
  dataset:
    enable: true
    folder: "outputs/simulation_data" # HDF5 輸出位置
  video:
    enable: true
    fps: 30
```

---

## 輸出數據格式 (Output Data Format)

系統主要輸出 **HDF5 (`.h5`)** 檔案，專為深度學習設計。

**檔案路徑:** `outputs/simulation_data/h5_SimData/<CaseName>.h5`

**數據結構 (HDF5 Group/Dataset):**

- **Global Attributes:**
  - `Re`: 雷諾數
  - `nx`, `ny`: 解析度
  - `dt`: 時間步長
- **Datasets (Time-series):**
  - `velocity_x`: (T, H, W) - X 方向速度
  - `velocity_y`: (T, H, W) - Y 方向速度
  - `density`: (T, H, W) - 流體密度 Rho
  - `vorticity`: (T, H, W) - 渦度 (Curl of Velocity)
  - `mask`: (H, W) - 幾何遮罩 (0=Fluid, 1=Solid)
  - **MRT Moments (Advanced):**
    - 包含 `rho`, `e`, `eps`, `jx`, `qx`, `jy`, `qy`, `pxx`, `pxy` 等 9 個 MRT 矩分量，用於高階物理學習。

---

## 開發者指南 (Developer Notes)

- **坐標系:** Taichi 使用 `(i, j)` 索引，對應 `(x, y)`。注意 Numpy 與 Taichi 在維度順序上的潛在差異 (通常是 `[x, y]` vs `[row, col]`)，但在導出時 `LBM_solver.py` 已做處理。
- **邊界處理:** 採用 Ghost Node 能夠處理 Velocity Inlet (拋物線/均勻), Outflow (Neumann), Free-Slip 等條件。
- **Sponge Layer:** 在出口與上下邊界設有隱式阻尼層 (Sponge Layer)，防止壓力波反射影響流場。
- **數據驗證:** 專案包含 `DFG Benchmark` 的驗證邏輯 (參見 `get_force` 與 `compute_force_on_obstacle`)，可用於計算阻力/升力係數。
