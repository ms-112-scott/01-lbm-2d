# LBM MRT-LES Solver 技術文檔

## 1. 模組概述 (Overview)

`LBM_MRT_LES` 是一個基於 **Taichi** 實作的高效能二維晶格波茲曼 (Lattice Boltzmann Method, LBM) 求解器。

本求解器專為 **高雷諾數 (High Reynolds Number)** 與 **非定常流 (Unsteady Flow)** 設計，結合了多重鬆弛時間 (MRT) 碰撞模型與 Smagorinsky 次網格 (Sub-grid Scale) 湍流模型，以在保持數值穩定性的同時，捕捉流場的動態特徵。

**核心特性：**

- **D2Q9 模型**: 使用標準的二維九速晶格。
- **MRT 碰撞算子**: 透過矩空間 (Moment Space) 的獨立鬆弛，克服傳統 BGK 模型在高雷諾數下的不穩定性。
- **LES 湍流模型**: 動態調整局部黏滯係數，模擬小尺度的湍流耗散。
- **Sponge Layer (阻尼層)**: 在邊界處施加隱式阻尼，防止壓力波反射干擾流場。
- **Ghost Node 邊界處理**: 支援多種邊界條件 (Inlet, Outlet, Free-slip, No-slip)。
- **數值熔斷機制**: 實時監控速度與受力，防止數值爆炸導致的無效數據生成。

---

## 2. 物理模型 (Physics Model)

### 2.1 晶格結構 (D2Q9)

使用 D2Q9 離散速度模型，包含 9 個離散方向 $\mathbf{e}_i$ 與對應權重 $w_i$：

| Index | Direction | Vector $(e_x, e_y)$      | Weight $w_i$ |
| :---- | :-------- | :----------------------- | :----------- |
| 0     | Center    | $(0, 0)$                 | $4/9$        |
| 1-4   | Axis      | $(\pm 1, 0), (0, \pm 1)$ | $1/9$        |
| 5-8   | Diagonal  | $(\pm 1, \pm 1)$         | $1/36$       |

### 2.2 MRT 碰撞算子 (Multiple Relaxation Time)

不同於 SRT (BGK) 使用單一鬆弛時間 $ au$，MRT 將分布函數 $f$ 轉換到矩空間 $\mathbf{m}$ 進行碰撞：

$$
\mathbf{m} = \mathbf{M} \mathbf{f}
$$

$$
\mathbf{m}^* = \mathbf{m} - \mathbf{S} (\mathbf{m} - \mathbf{m}^{eq})
$$

$$
\mathbf{f}^{new} = \mathbf{M}^{-1} \mathbf{m}^*
$$

其中 $\mathbf{M}$ 為轉換矩陣 (Lallemand & Luo formulation)，$\mathbf{S}$ 為對角鬆弛矩陣。

- **$s_7, s_8$**: 控制剪切黏滯性 (Shear Viscosity)，與物理黏滯係數 $
u$ 相關。
- **$s_1, s_2, s_4, s_6$**: ghost moments，通常設為略大於 1 的值 (如 1.1) 以增強穩定性。

### 2.3 LES 湍流模型 (Large Eddy Simulation)

採用 Smagorinsky 模型，根據局部應變率張量 (Strain Rate Tensor) 動態增加有效鬆弛時間：

$$
	au_{total} = 	au_0 + 	au_{eddy}
$$

$$
	au_{eddy} = \frac{1}{2} \left( \sqrt{	au_0^2 + 18 C_s^2 |S|} - 	au_0
ight)
$$

其中 $|S|$ 為非平衡矩 (Non-equilibrium moments) 計算出的應變率強度。此機制在剪切強烈的區域 (如物體後方尾流) 自動增加數值黏滯性，模擬湍流耗散。

### 2.4 Sponge Layer (阻尼層)

為了模擬開放邊界並減少反射，求解器在 **出口 (Outlet)** 與 **上下邊界 (Top/Bottom)** 設有阻尼層。

- **機制**: 在邊界附近的區域，強制增加鬆弛時間 $ au$。
- **效果**: 進入阻尼層的渦流會被迅速耗散，防止其撞擊邊界後反彈回計算域。
- **實現**: 使用二次函數 (Quadratic profile) 平滑過渡阻尼強度。

---

## 3. 程式架構 (Implementation Details)

### 3.1 核心類別: `LBM2D_MRT_LES` (`src/lbm_mrt_les/LBM_solver.py`)

#### 主要資料結構 (Taichi Fields)

- `rho` (Scalar): 巨觀密度場。
- `vel` (Vector2): 巨觀速度場。
- `f_old`, `f_new` (Vector9): 分布函數 (雙緩衝區 Double Buffering)。
- `mask` (Scalar): 幾何遮罩 (0=流體, 1=固體)。
- `moments_field` (Vector9): **(輸出用)** 儲存 MRT 矩空間變數，供 AI 訓練使用。

#### 關鍵核心函數 (Kernels)

1.  `collide_and_stream()`:
    - 執行 Streaming (移位)。
    - 執行 MRT 碰撞 (Matrix multiply -> Relaxation -> Inverse Matrix)。
    - 計算 LES 渦黏性與 Sponge Layer 阻尼。
2.  `update_macro_var()`:
    - 計算 `rho`, `vel`。
    - 交換 Time step (更新 `f_old`)。
3.  `apply_bc()`:
    - 處理 Inlet (拋物線/均勻流)。
    - 處理 Outlet (Neumann)。
    - 處理 Wall (Bounce-back / Free-slip)。
    - **Ghost Node**: 利用外插法 (Extrapolation) 修正邊界節點的分布函數。

### 3.2 模擬迴圈: `run_simulation_loop` (`src/lbm_mrt_les/simulation_ops.py`)

負責協調求解器、視覺化與 I/O，並包含關鍵的**穩定性檢查**。

#### 穩定性熔斷機制 (Stability Fuses)

為確保自動化批量生產的數據品質，系統會監控以下指標，一旦異常立即終止模擬：

1.  **NaN/Inf Check**: 檢查全場是否有非數值 (爆掉)。
2.  **Force Explosion**: 檢查障礙物受力是否超過閾值 (通常發生在壓力場震盪時)。
3.  **Velocity Limit**: 檢查最大流速是否超過 LBM 極限 (Mach number < 0.577)。設有 `warmup_steps` 容許初始震盪。

---

## 4. 邊界條件設定 (Boundary Conditions)

在 `config.yaml` 中透過 `boundaries` 區塊設定：

```yaml
boundaries:
  types: [0, 0, 1, 0] # 左, 上, 右, 下
  values:
    - [0.1, 0.0] # 左側入口速度
```

| Type Code | 描述           | 物理意義   | 實現細節                                                                            |
| :-------- | :------------- | :--------- | :---------------------------------------------------------------------------------- |
| **0**     | Velocity Inlet | 速度入口   | Dirichlet 條件。若為左邊界，預設為 **Parabolic Profile** (拋物線)，其餘為 Uniform。 |
| **1**     | Outflow        | 壓力出口   | Neumann 條件 (零梯度)。速度與密度直接複製鄰居節點。                                 |
| **2**     | Free Slip      | 滑移壁面   | 對稱邊界。切向速度保留，法向速度歸零。                                              |
| **3**     | No Slip        | 無滑移壁面 | 固壁。速度強制為 0 (標準 Bounce-back)。                                             |

---

## 5. 數據輸出 (Data Output)

求解器支援輸出 HDF5 格式的訓練數據，特別是 **MRT Moments**。

### MRT Moments 通道定義

為了讓 AI 模型學習更豐富的物理特徵，我們不只輸出速度，還輸出 MRT 矩空間的原始分量：

| Channel | Symbol     | Description   | Physics Relevance       |
| :------ | :--------- | :------------ | :---------------------- |
| 0       | $          |
| ho$     | Density    | 質量守恆      |
| 1       | $e$        | Energy        | 與壓力相關              |
| 2       | $\epsilon$ | Energy Square | 高階能量修正            |
| 3       | $j_x$      | Momentum X    | 動量守恆 ($x$)          |
| 4       | $q_x$      | Heat Flux X   | 熱通流 ($x$)            |
| 5       | $j_y$      | Momentum Y    | 動量守恆 ($y$)          |
| 6       | $q_y$      | Heat Flux Y   | 熱通流 ($y$)            |
| 7       | $p_{xx}$   | Normal Stress | 正應力 (與黏性有關)     |
| 8       | $p_{xy}$   | Shear Stress  | 切應力 (與渦旋生成有關) |

這些通道完整描述了流體在介觀尺度 (Mesoscopic scale) 的狀態。
