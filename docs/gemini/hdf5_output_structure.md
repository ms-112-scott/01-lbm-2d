# LBM 模擬數據輸出格式說明 (HDF5)

本文件詳細說明 `lbm_2d` 專案所產生的 HDF5 (`.h5`) 數據結構。此格式專為深度學習模型訓練設計，包含了時序物理場、幾何遮罩、SDF (Signed Distance Field) 以及累積渦度統計。

## 1. 檔案位置與命名

- **預設路徑**: `outputs/simulation_data/h5_SimData/`
- **命名規則**: `<SimulationName>_<MaskName>.h5`

## 2. HDF5 數據結構概覽

HDF5 檔案內部採用扁平化結構，主要包含四個核心 Dataset 與一組全域屬性 (Attributes)。

```text
root (/)
├── turbulence       (Dataset) [T, 9, H, W] - 主要時序物理場 (MRT Moments)
├── static_mask      (Dataset) [2, H, W]    - 靜態幾何資訊 (Mask + SDF)
├── sum_vor          (Dataset) [H, W]       - 累積絕對渦度 (Accumulated |Vorticity|)
├── mean_field       (Dataset) [9, H, W]    - 時序平均場
└── (Attributes)     - 全域元數據 (Config, Stats)
```

---

## 3. Dataset 詳細說明

### 3.1 `turbulence` (主要數據)

儲存模擬過程中的瞬時物理場數據。原名為 `snapshots`。

- **Shape**: `(T, 9, H, W)`
  - `T`: 時間步數 (Frames)。
  - `9`: 通道數 (Channels)，對應 MRT LBM 的 9 個矩 (Moments)。
  - `H, W`: 儲存解析度。
- **Data Type**: `float32`

#### 通道定義 (Channel Mapping)

| Index | 代號 | 物理意義 | 說明 |
| :--- | :--- | :--- | :--- |
| **0** | `rho` | **密度 (Density)** | 流體密度波動。 |
| **1** | `e` | **能量 (Energy)** | 與壓力相關。 |
| **2** | `eps` | **能量平方 (Epsilon)** | 高階修正。 |
| **3** | `jx` | **動量 X (Momentum X)** | $j_x = \rho u_x$。 |
| **4** | `qx` | **熱通流 X (Heat Flux X)** | 高階速度修正。 |
| **5** | `jy` | **動量 Y (Momentum Y)** | $j_y = \rho u_y$。 |
| **6** | `qy` | **熱通流 Y (Heat Flux Y)** | 高階速度修正。 |
| **7** | `pxx` | **正應力 (Normal Stress)** | 黏滯與應力。 |
| **8** | `pxy` | **切應力 (Shear Stress)** | 渦旋相關。 |

---

### 3.2 `static_mask` (幾何資訊)

- **Shape**: `(2, H, W)`
- **通道定義**:
  - **Index 0**: `binary_mask` (0: 流體, 1: 固體)
  - **Index 1**: `sdf` (符號距離場，流體為正，固體為負)

---

### 3.3 `sum_vor` (累積渦度)

儲存整個模擬時段內，每個網格點上 **絕對渦度 (|Vorticity|)** 的累加值。這對於識別流場中的高湍流區域或常駐渦流非常有用。

- **Shape**: `(H, W)`
- **計算方式**: $\sum_{t=1}^{T} | \nabla \times \mathbf{u}_t |$
- **用途**: 評估結構受風影響的劇烈程度。

---

### 3.4 `mean_field` (統計場)

- **Shape**: `(9, H, W)`
- **內容**: `turbulence` 在時間維度上的平均值。

---

## 4. 全域屬性 (Global Attributes)

- **`config_json`**: 模擬參數 JSON 字串。
- **`stats_min / max / mean`**: 每個通道的全域統計值。

---

## 5. Python 讀取範例

```python
import h5py
import numpy as np

f = h5py.File("path/to/data.h5", "r")

# 1. 讀取時序數據 (例如讀取 jx)
jx = f["turbulence"][:, 3, :, :] # (T, H, W)

# 2. 讀取 SDF
sdf = f["static_mask"][1, :, :]  # (H, W)

# 3. 讀取累積渦度
sum_vor = f["sum_vor"][:]        # (H, W)

# 4. 讀取 Config
import json
config = json.loads(f.attrs["config_json"])
print(f"Reynolds Number: {config['simulation']['nu']}")

f.close()
```