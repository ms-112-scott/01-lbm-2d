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

專案已採用 `src/` 為核心的架構，並將求解器模組化分類。

```text
C:\Users\GAI\Desktop\NCA_workspace\01-lbm-2d\
├── config.yaml                     # [核心] 預設全域配置文件
├── requirements.txt                # Python 依賴庫
├── src/
│   ├── configs/                    # 配置檔模板與變體
│   │   ├── config_template.yaml
│   │   └── hyper_configs/          # 用於批量實驗的參數化配置
│   ├── GenMask/                    # [工具] 幾何遮罩生成
│   │   ├── gen_rects_numpy.py      # 生成隨機矩形障礙物 Masks
│   │   └── rect_masks/             # 預設 Mask 輸出目錄
│   └── lbm_mrt_les/                # [核心] LBM 求解器源碼
│       ├── engine/                 # [核心] 物理運算與模擬邏輯
│       │   ├── LBM_solver.py       # 主要求解器 (LBM2D_MRT_LES Class)
│       │   └── simulation_ops.py   # 模擬主迴圈 (Loop & Stability Check)
│       ├── runners/                # [執行] 程式入口點
│       │   ├── run_one_case.py     # 單一 Config 批量執行入口
│       │   └── run_multi_case.py   # Config-Mask 一對一批量執行入口
│       ├── utils/                  # [輔助] 工具與配置生成
│       │   └── utils.py            # 通用輔助函式 (Config讀取, 物理計算)
│       └── io/                     # [輸出] 視覺化與數據寫入
│           ├── LBMCaseWriter.py    # HDF5 數據寫入器
│           ├── Taichi_Gui_Viz.py   # Taichi GUI 視覺化
│           └── VideoRecorder.py    # 影片錄製工具
├── output/                         # 模擬結果輸出 (HDF5, Video, JSON)
└── my_docs/                        # 文件與研究筆記
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
python src/GenMask/gen_rects_numpy.py
```

- 輸出: `src/GenMask/rect_masks/*.png`

### 3. 執行模擬

#### 模式 A: 標準批量 (單一 Config -> 多個 Masks)

適用於固定物理條件 (如固定 Re)，跑多個不同幾何形狀。

```bash
python -m src.lbm_mrt_les.runners.run_one_case --config src/configs/config_template.yaml --mask_dir src/GenMask/rect_masks
```

#### 模式 B: 進階批量 (Config-Mask 一對一)

適用於產生多樣化物理條件 (不同 Re, 不同解析度) 的數據集。需確保 Config 檔案數量與 Mask 檔案數量一致且排序對應。

```bash
python -m src.lbm_mrt_les.runners.run_multi_case --config_dir src/configs/hyper_configs --mask_dir src/GenMask/generated_maps_advanced
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
  mask_dir: "output/masks"

outputs:
  dataset:
    enable: true
    folder: "output/dataset" # HDF5 輸出位置
  video:
    enable: true
    fps: 30
```

---

## 輸出數據格式 (Output Data Format)

系統主要輸出 **HDF5 (`.h5`)** 檔案，專為深度學習設計。

**檔案路徑:** `output/dataset/h5_SimData/<CaseName>.h5`

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
