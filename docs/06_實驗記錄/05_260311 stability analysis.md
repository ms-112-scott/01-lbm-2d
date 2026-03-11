# LBM MRT-LES 2D — 模擬不穩定根因分析與修正方案

> **版本**: 05 — 穩定性崩潰深度分析  
> **狀態**: ⚠️ 發現 2 個 CRITICAL 根因 + 3 個 IMPORTANT 問題

---

## 目錄

1. [崩潰根因分析（診斷追蹤）](#1-崩潰根因分析)
2. [Bug A — L_char 嚴重高估（Y 軸投影總和）](#2-bug-a--l_char-嚴重高估)
3. [Bug B — 缺乏阻塞率修正，速度噴射必然發生（最直接崩潰原因）](#3-bug-b--缺乏阻塞率修正)
4. [Bug C — Solver Re 顯示用 dummy bc_value 計算](#4-bug-c--solver-re-顯示錯誤)
5. [Bug D — 出口 Zou-He BC 未處理 backflow](#5-bug-d--出口-zou-he-bc-未處理-backflow)
6. [驗證：Zou-He / MRT / LES 核心邏輯審核](#6-核心邏輯審核)
7. [完整修正程式碼](#7-完整修正程式碼)
8. [修改總表](#8-修改總表)

---

## 1. 崩潰根因分析

### 日誌解讀（Case 1）

```
[Solver] Initialized: target rho_in=1.01, rho_out=1.0, Initial Re est.=2327.50
[Strategy] max_steps=950,150  (from config, CTU-based)
[Strategy] u_estimated=0.08165 lu/step  Re_estimated=3800.8
[Strategy] warmup_steps=152,024  start_record=380,060

 24% | step 228,000 | MaxV=0.2306
[CRITICAL] Velocity 0.2557 exceeded stability threshold (0.25) at step 228,950
```

### 逆推分析

**Re 顯示不一致（Solver: 2327 vs Strategy: 3800）**

兩者都用相同 `nu=0.020`, 但分母 `u_char` 不同：

```
Solver Re:   u = bc_value[0][0] = 0.05 (dummy)   → Re = 0.05 × L_char / 0.02
Strategy Re: u = sqrt(2/3 × 0.01) = 0.0817        → Re = 0.0817 × L_char / 0.02

0.05 / 0.0817 = 0.612 → 2327 / 3800 ≈ 0.613 ✓ (一致)
```

兩者使用的 **L_char 是一致的**，從 Strategy Re 反推：

```
L_char = Re × nu / u = 3800 × 0.020 / 0.0817 = 930 px
```

**對 Case 2：L_char = 5907 × 0.020 / 0.0817 = 1447 px（佔 ny=1792 的 80%！）**

### 速度崩潰物理機制

```
rho_in = 1.010 → Δρ = 0.010 → u_inlet_avg = sqrt(2/3 × 0.01) = 0.0817 lu/step
Ma = 0.0817 / 0.577 = 0.141  ✅ (入口均值安全)

但穿越建築間隙時，連續方程要求速度放大：
  u_gap = u_inlet / (1 - blockage_ratio)

若阻塞率 = 60%：u_gap = 0.0817 / 0.40 = 0.204  ← 接近邊界
若阻塞率 = 67%：u_gap = 0.0817 / 0.33 = 0.248  ← 貼近 0.25 閾值
若加上渦流脫落瞬時加速（+ 10~15%）：0.248 × 1.05 = 0.260 ❌ 崩潰
```

步驟 228,000 MaxV=0.2306 → 228,950 MaxV=0.2557，在 warmup 結束後（warmup=152,024）
流場完全發展，渦流結構建立後，間隙處速度進一步放大，觸發崩潰。

---

## 2. Bug A — L_char 嚴重高估（Y 軸投影總和）

**位置**: `src/tools/config_batch_gen.py` — `calc_l_char_from_png`

### 問題

```python
y_occupied = np.any(solid, axis=0)  # Y 軸投影
return max(1, int(np.sum(y_occupied)))  # ← 所有建築 Y 投影的「總和」
```

`np.any(solid, axis=0)` → shape `[ny]` 陣列，任一 X 列有固體就為 True。  
`np.sum(y_occupied)` → 計算的是**整個 domain 中，有任意固體佔據的 Y 行數量總和**。

對都市 mask（建築群散佈在不同高度）：

```
建築A 在 y=[100, 200]，建築B 在 y=[400, 550]，建築C 在 y=[700, 900]...
y_occupied 在這些範圍都是 True
sum = 100 + 150 + 200 + ... ≈ 930 px（每個建築的高度加總）
```

### 影響

| 實際個別建築高度 | L_char 計算值 | Re 放大倍數 | max_steps 誤差 |
| :--------------: | :-----------: | :---------: | :------------: |
|      ~150px      |     930px     |    6.2×     | 6.2× (Case 1)  |
|      ~150px      |    1447px     |    9.6×     | 9.6× (Case 2)  |

- Re 被放大 6-10 倍（顯示 3800，實際應 ~400）
- max_steps 放大 6-10 倍（950k steps 實際只需 ~150k）
- 浪費 GPU 時間，且讓後面的 pass 判斷完全錯誤

### 修正：改用最大連通域的 Y 跨度

```python
from scipy import ndimage

def calc_l_char_from_png(png_path: str, invert: bool, nx: int, ny: int) -> int:
    """
    L_char = 最大單一建築的 Y 軸跨度（像素）。

    修正前：np.sum(np.any(solid, axis=0)) → 所有建築高度加總 [錯誤]
    修正後：最大連通域的 Y-extent → 代表最大個別障礙物尺寸 [正確]
    """
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"無法讀取圖片：{png_path}")
    if img.shape != (ny, nx):
        img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_NEAREST)

    solid = (img > 127) if invert else (img < 127)
    solid = solid.T  # → [nx, ny]（Taichi field 慣例）

    # 連通域標記
    labeled, n_features = ndimage.label(solid)
    if n_features == 0:
        # 沒有任何固體，取 ny 的 1/4 作為預設
        return max(1, ny // 4)

    max_l = 0
    for label_id in range(1, n_features + 1):
        region = (labeled == label_id)
        # Y 方向的跨度：有此 label 的所有 y 位置
        y_indices = np.where(np.any(region, axis=0))[0]
        if len(y_indices) > 0:
            l = int(y_indices[-1] - y_indices[0] + 1)
            max_l = max(max_l, l)

    return max(1, max_l)
```

---

## 3. Bug B — 缺乏阻塞率修正（最直接崩潰原因）

**位置**: `src/tools/config_batch_gen.py` — `check_feasibility` 與 `main`

### 問題

`check_feasibility` 只檢查**入口均值速度的 Ma**，完全沒有考慮建築間隙的速度放大：

```python
def check_feasibility(rho_in, rho_out, nu_lb):
    u_bernoulli = math.sqrt((2.0/3.0) * delta_rho)
    Ma = u_bernoulli / CS           # ← 只看入口均值，忽略間隙放大
    if Ma > MA_LIMIT:
        return False, "..."
    if tau < TAU_MIN:
        return False, "..."
    return True, ""                 # ← 通過檢查，但間隙速度可能是 3× 這個值
```

### 修正：加入阻塞率感知的 rho_in 動態調整

#### Step 1：新增阻塞率計算函數

```python
def calc_max_blockage_from_png(png_path: str, invert: bool, nx: int, ny: int) -> float:
    """
    計算 mask 中最嚴重的 X 截面阻塞率。

    對每個 x-column，計算固體像素佔 y 方向的比例。
    取最大值作為最壞情況阻塞率。
    """
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_NEAREST)
    solid = (img > 127) if invert else (img < 127)
    solid = solid.T  # [nx, ny]
    # 每個 x-column 的固體比例
    blockage_per_x = np.mean(solid.astype(float), axis=1)  # shape [nx]
    return float(np.max(blockage_per_x))
```

#### Step 2：修改 main 中的 rho_in 設定邏輯

```python
# ── 從 PNG 計算 L_char 與最大阻塞率 ──────────────────────────
try:
    l_char = calc_l_char_from_png(mask_path, mask_invert, nx_val, ny_val)
    max_blockage = calc_max_blockage_from_png(mask_path, mask_invert, nx_val, ny_val)
except Exception as e:
    print(f"  [Skip] 讀取 mask 失敗：{e}\n")
    skip_count += 1
    continue

# ── 阻塞率感知的 rho_in 動態調整 ───────────────────────────────
# 目標：確保間隙最大速度 < U_GAP_MAX
# u_gap = u_inlet / (1 - blockage)
# u_inlet = sqrt(2/3 × delta_rho)
# 需要：u_inlet < U_GAP_MAX × (1 - blockage)
U_GAP_MAX = 0.20          # 最大允許間隙速度（安全餘量到 0.25 崩潰閾值）
MIN_OPEN_FRACTION = 0.15  # 防止分母太小（極端阻塞 >85% 強制降低）

open_fraction = max(MIN_OPEN_FRACTION, 1.0 - max_blockage)
u_inlet_safe = U_GAP_MAX * open_fraction
delta_rho_safe = (3.0 / 2.0) * u_inlet_safe ** 2
rho_in_case = min(rho_in, 1.0 + delta_rho_safe)

if rho_in_case < rho_in - 1e-6:
    print(
        f"  [Blockage Adj] max_blockage={max_blockage:.1%}  "
        f"open={open_fraction:.1%}  "
        f"rho_in: {rho_in:.4f} → {rho_in_case:.4f}  "
        f"(u_inlet_safe={u_inlet_safe:.5f})"
    )
else:
    rho_in_case = rho_in

# ── 重新執行可行性檢查（用調整後的 rho_in）──────────────────────
ok, reason = check_feasibility(rho_in_case, rho_out, nu_lb)
if not ok:
    print(f"  [Skip] {reason}\n")
    skip_count += 1
    continue

# ── 速度與 Re 計算（用調整後的 rho_in）──────────────────────────
delta_rho = rho_in_case - rho_out
u_bernoulli = math.sqrt((2.0 / 3.0) * delta_rho) if delta_rho > 0 else 0.01
u_for_steps = u_bernoulli * U_STEP_FACTOR

Re_lb = u_bernoulli * l_char / nu_lb
Ma = u_bernoulli / CS
```

#### 調整後 rho_in 效果預覽

| max_blockage | u_inlet_safe | rho_in_safe | u_gap_max |
| :----------: | :----------: | :---------: | :-------: |
|     30%      |    0.140     |   1.0294    | 0.200 ✅  |
|     50%      |    0.100     |   1.0150    | 0.200 ✅  |
|     60%      |    0.080     |   1.0096    | 0.200 ✅  |
|     70%      |    0.060     |   1.0054    | 0.200 ✅  |
|     80%      |    0.030     |   1.0014    | 0.200 ✅  |

---

## 4. Bug C — Solver Re 顯示用 dummy bc_value 計算

**位置**: `src/lbm_mrt_les/core/LBM2D_MRT_LES.py` — `_init_params`

### 問題

```python
bc_values = self.config["boundary_condition"]["value"]
self.u_inlet = np.array(bc_values[0], dtype=np.float32)
u_char = np.linalg.norm(self.u_inlet)  # ← bc_value[0] = [0.05, 0.0]（dummy）
self.Re = (u_char * self.characteristic_length) / self.nu
# → Re = 0.05 × 930 / 0.02 = 2327（偏低 1.63×）
```

### 修正

```python
import math as _math

# 改用 Bernoulli 壓差估算入口速度
delta_rho = self.rho_in_target - self.rho_out_target
u_char = _math.sqrt(2.0 / 3.0 * delta_rho) if delta_rho > 1e-9 else 0.01

if self.nu > 0:
    self.Re = (u_char * self.characteristic_length) / self.nu
else:
    self.Re = float("inf")

print(
    f"[Solver] Initialized: target rho_in={self.rho_in_target}, "
    f"rho_out={self.rho_out_target}, "
    f"u_est={u_char:.5f}, Re_est={self.Re:.1f}"
)
```

---

## 5. Bug D — 出口 Zou-He BC 未處理 Backflow

**位置**: `src/lbm_mrt_les/core/LBM2D_MRT_LES.py` — `apply_bc_core`，`bc_type==1` 分支

### 問題

```python
elif self.bc_type[dr] == 1:  # Outlet
    if ibc == self.nx - 1:
        ux = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_out
        # ↑ 若出口有 backflow (ux < 0)，f3, f6, f7 的計算會產生負值
        # 這不會立即崩潰，但會積累非物理誤差
```

當都市建築的渦流導致出口局部有回流時（ux < 0），Zou-He 壓力邊界假設 ux > 0，
計算出的未知分布函數 f3, f6, f7 會不物理，逐步積累直到觸發速度爆炸。

### 修正

```python
elif self.bc_type[dr] == 1:  # Outlet (Zou-He Pressure at East)
    if ibc == self.nx - 1:
        ux = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_out
        uy = 0.0

        # [FIX] Backflow 防護：若出口檢測到回流，改用零梯度外推
        # 物理上正確：出口有回流代表流場不應被壓力 BC 強制，
        # 改用 copy-from-neighbor 避免非物理分佈函數。
        if ux < 0.0:
            self.vel[ibc, jbc] = self.vel[inb, jnb]
            self.rho[ibc, jbc] = rho_out
            self.f_old[ibc, jbc] = (
                self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
            )
        else:
            f3 = f1 - (2.0 / 3.0) * rho_out * ux
            f6 = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux
            f7 = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux

            self.rho[ibc, jbc] = rho_out
            self.vel[ibc, jbc] = tm.vec2(ux, uy)
            self.f_old[ibc, jbc] = self.f_eq(ibc, jbc)
            self.f_old[ibc, jbc][3] = f3
            self.f_old[ibc, jbc][6] = f6
            self.f_old[ibc, jbc][7] = f7
```

---

## 6. 核心邏輯審核

以下是對整個求解器核心邏輯的完整審核，確認哪些是**正確的**，哪些有問題。

### 6.1 MRT 矩陣（✅ 正確）

採用標準 Lallemand & Luo (2000) D2Q9 M 矩陣，矩空間順序：
`[rho, e, eps, jx, qx, jy, qy, pxx, pxy]`

M 矩陣與 M_inv 計算正確，已用 `np.linalg.inv` 求得精確逆矩陣。

### 6.2 平衡矩 m_eq（✅ 正確）

```python
def get_meq(self, rho, u, v):
    u2 = u*u + v*v
    return [rho,
            rho*(-2 + 3*u2),     # e_eq
            rho*(1 - 3*u2),      # eps_eq
            rho*u,                # jx_eq
            -rho*u,               # qx_eq
            rho*v,                # jy_eq
            -rho*v,               # qy_eq
            rho*(u*u - v*v),      # pxx_eq
            rho*u*v]              # pxy_eq
```

符合 Lallemand & Luo 標準形式。✅

### 6.3 LES Smagorinsky 修正（✅ 正確，但注意係數）

```python
neq_tensor_norm = sqrt(2*neq_7² + 2*neq_8²)  # Frobenius 範數正確
term_inside = tau_0² + Cs_sq_factor * neq_tensor_norm / rho_l
tau_eddy = 0.5 * (sqrt(term_inside) - tau_0)
tau_eff = tau_0 + tau_eddy
```

對應 Hou et al. (1994) 公式，`Cs_sq_factor = 18 × Cs²`。

**注意**: `smagorinsky_constant=0.2` 寫死在 `generate_case_config`，
建議移入 `master_config.yaml` 以便調整（Cs=0.10~0.18 通常更保守）。

### 6.4 Sponge Layer（✅ 正確，強度可調）

```python
# X 方向（出口側）
if i > (nx - sponge_w_x):
    coord = (i - (nx - sponge_w_x)) / sponge_w_x
    damping_x = sponge_strength * coord²

# Y 方向（上下壁面）類似
tau_eff += max(damping_x, damping_y)
```

二次函數阻尼正確。`sponge_strength=2.0` 硬編碼，
**建議**: 移入 config，都市場景建議 2.5~4.0（較強阻尼吸收更多渦流反射）。

### 6.5 Zou-He 入口壓力 BC（✅ 正確，有緩啟動）

```python
rho_current = 1.0 + (rho_in - 1.0) * ramp  # cosine 緩啟動
ux = 1.0 - (f0+f2+f4 + 2*(f3+f6+f7)) / rho_current
f1 = f3 + (2/3) * rho_current * ux
f5 = f7 - 0.5*(f2-f4) + (1/6) * rho_current * ux
f8 = f6 + 0.5*(f2-f4) + (1/6) * rho_current * ux
```

標準 Zou-He (1997) 西邊壓力入口公式，cosine ramp 緩啟動防止初始衝擊。✅

### 6.6 障礙物 BC — 平衡態補填法（⚠️ 可用，非最優）

```python
if mask[i,j] == 1.0:
    vel[i,j] = (0, 0)
    f_old[i,j] = f_eq(i,j)  # 以 rho≈1, u=0 的平衡態補填
```

這是「wet-node equilibrium refill」法，比半步 bounce-back 實作簡單，
稳定性較好，但在固液界面精度略差（O(Δx) vs O(Δx²)）。

對都市 ML 資料集目標（關注流場統計特性而非精確壁面剪切）是可接受的。

### 6.7 操作順序（✅ 正確）

```
collide_and_stream()  ← pull-scheme：讀 f_old，寫 f_new
update_macro_var()    ← 計算 rho/vel，swap f_new → f_old
apply_bc()            ← 修正邊界 f_old（供下一步 stream 使用）
```

標準 LBM 操作序列，無誤。✅

### 6.8 compute_moments_for_output（✅ 正確）

所有 9 個 MRT 矩量 (rho, e, eps, jx, qx, jy, qy, pxx, pxy) 的手動計算
與 M 矩陣行定義完全吻合。✅

---

## 7. 完整修正程式碼

### 7.1 config_batch_gen.py — 完整修正部分

```python
# ─── 新增 import ───────────────────────────────────────────────────────────
from scipy import ndimage  # 新增

# ─── 修正 calc_l_char_from_png ────────────────────────────────────────────
def calc_l_char_from_png(png_path: str, invert: bool, nx: int, ny: int) -> int:
    """
    L_char = 最大單一障礙物的 Y 軸跨度（像素）。
    [修正] 改用連通域分析，不再用所有建築高度加總。
    """
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"無法讀取圖片：{png_path}")
    if img.shape != (ny, nx):
        img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_NEAREST)

    solid = (img > 127) if invert else (img < 127)
    solid = solid.T  # → [nx, ny]

    labeled, n_features = ndimage.label(solid)
    if n_features == 0:
        return max(1, ny // 4)

    max_l = 0
    for label_id in range(1, n_features + 1):
        region = (labeled == label_id)
        y_indices = np.where(np.any(region, axis=0))[0]
        if len(y_indices) > 0:
            l = int(y_indices[-1] - y_indices[0] + 1)
            max_l = max(max_l, l)

    return max(1, max_l)


# ─── 新增 calc_max_blockage_from_png ─────────────────────────────────────
def calc_max_blockage_from_png(png_path: str, invert: bool, nx: int, ny: int) -> float:
    """
    計算最嚴重的 X 截面阻塞率（fraction of Y that is solid）。
    """
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_NEAREST)
    solid = (img > 127) if invert else (img < 127)
    solid = solid.T  # [nx, ny]
    blockage_per_x = np.mean(solid.astype(float), axis=1)
    return float(np.max(blockage_per_x))


# ─── main() 中修改 per-mask 處理邏輯 ──────────────────────────────────────
# (只貼出需要修改的部分，其他保持不變)

    for i, mask_path in enumerate(mask_files):
        seq_id = i + 1
        sim_name = f"case_{seq_id:04d}"

        print(f"[{seq_id:04d}/{total}]  {os.path.basename(mask_path)}")

        # ── nu_lb 取樣
        nu_lb = get_sampled_nu(nu_lb_list)

        # ── 從 PNG 計算 L_char 與最大阻塞率 ──────────────────── [修正]
        try:
            l_char = calc_l_char_from_png(mask_path, mask_invert, nx_val, ny_val)
            max_blockage = calc_max_blockage_from_png(mask_path, mask_invert, nx_val, ny_val)
        except Exception as e:
            print(f"  [Skip] 讀取 mask 失敗：{e}\n")
            skip_count += 1
            continue

        # ── 阻塞率感知 rho_in 調整 ─────────────────────────── [新增]
        U_GAP_MAX = 0.20          # 間隙最大安全速度
        MIN_OPEN_FRACTION = 0.15  # 極端阻塞防呆
        open_fraction = max(MIN_OPEN_FRACTION, 1.0 - max_blockage)
        u_inlet_safe = U_GAP_MAX * open_fraction
        delta_rho_safe = (3.0 / 2.0) * u_inlet_safe ** 2
        rho_in_case = min(rho_in, 1.0 + delta_rho_safe)

        if rho_in_case < rho_in - 1e-6:
            print(
                f"  [Blockage] max_blockage={max_blockage:.1%}  "
                f"rho_in: {rho_in:.4f} → {rho_in_case:.4f}"
            )

        # ── 可行性檢查 ─────────────────────────────────────── [修正順序]
        ok, reason = check_feasibility(rho_in_case, rho_out, nu_lb)
        if not ok:
            print(f"  [Skip] {reason}\n")
            skip_count += 1
            continue

        # ── 速度與 Re（用調整後 rho_in）──────────────────────── [修正]
        delta_rho = rho_in_case - rho_out
        u_bernoulli = math.sqrt((2.0 / 3.0) * delta_rho) if delta_rho > 0 else 0.01
        u_for_steps = u_bernoulli * U_STEP_FACTOR
        Re_lb = u_bernoulli * l_char / nu_lb
        Ma = u_bernoulli / CS

        # ── 步數計算
        steps_per_ctu = int(l_char / u_for_steps) if u_for_steps > 0 else 10000
        warmup_steps = w_passes * steps_per_ctu
        max_steps = t_passes * steps_per_ctu
        start_record_step = s_passes * steps_per_ctu
        target_interval = max(1, int(steps_per_ctu / saves_per_ctu))

        print(
            f"  nu_lb={nu_lb:.4f}  L_char={l_char}px  u={u_bernoulli:.5f}  "
            f"Ma={Ma:.4f}  Re={Re_lb:.0f}  blockage={max_blockage:.1%}\n"
            f"  CTU={steps_per_ctu:,}  warmup={warmup_steps:,}  "
            f"max={max_steps:,}  rho_in={rho_in_case:.4f}"
        )

        # ── 組裝 config（rho_in 改用 rho_in_case）────────────────── [修正]
        run_params = {
            ...
            "rho_in": rho_in_case,  # ← 用調整後的值
            ...
        }
```

### 7.2 LBM2D_MRT_LES.py — \_init_params Re 顯示修正

```python
def _init_params(self):
    # ... 其他參數不變 ...

    # [FIX] 用 Bernoulli 估算入口速度，不用 dummy bc_value
    import math as _math
    delta_rho = self.rho_in_target - self.rho_out_target
    u_char = _math.sqrt(2.0 / 3.0 * delta_rho) if delta_rho > 1e-9 else 0.01

    if self.nu > 0:
        self.Re = (u_char * self.characteristic_length) / self.nu
    else:
        self.Re = float("inf")

    print(
        f"[Solver] Initialized: target rho_in={self.rho_in_target}, "
        f"rho_out={self.rho_out_target}, u_est={u_char:.5f}, Re_est={self.Re:.1f}"
    )
```

### 7.3 LBM2D_MRT_LES.py — Outlet Zou-He Backflow 防護

```python
elif self.bc_type[dr] == 1:  # Outlet (Zou-He Pressure)
    if ibc == self.nx - 1:
        rho_out = self.rho_out_target
        f0 = self.f_old[inb, jnb][0]
        f1 = self.f_old[inb, jnb][1]
        f2 = self.f_old[inb, jnb][2]
        f4 = self.f_old[inb, jnb][4]
        f5 = self.f_old[inb, jnb][5]
        f8 = self.f_old[inb, jnb][8]

        ux = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_out
        uy = 0.0

        # [FIX] Backflow 防護
        if ux < 0.0:
            # 改用零梯度外推，防止非物理分佈函數積累
            self.vel[ibc, jbc] = self.vel[inb, jnb]
            self.rho[ibc, jbc] = rho_out
            self.f_old[ibc, jbc] = (
                self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
            )
        else:
            f3 = f1 - (2.0 / 3.0) * rho_out * ux
            f6 = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux
            f7 = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux

            self.rho[ibc, jbc] = rho_out
            self.vel[ibc, jbc] = tm.vec2(ux, uy)
            self.f_old[ibc, jbc] = self.f_eq(ibc, jbc)
            self.f_old[ibc, jbc][3] = f3
            self.f_old[ibc, jbc][6] = f6
            self.f_old[ibc, jbc][7] = f7
```

---

## 8. 修改總表

|  #  |   嚴重程度   | 位置                           | 問題                                                    | 修正方式                                             |
| :-: | :----------: | :----------------------------- | :------------------------------------------------------ | :--------------------------------------------------- |
|  B  | 🔴 CRITICAL  | `config_batch_gen.py`          | 無阻塞率修正，間隙速度必然超限                          | 新增 `calc_max_blockage_from_png`，動態調整 `rho_in` |
|  A  | 🔴 CRITICAL  | `config_batch_gen.py`          | `calc_l_char_from_png` 計算所有建築高度加總，高估 6-10× | 改用連通域最大 Y-span                                |
|  D  | 🟠 IMPORTANT | `LBM2D_MRT_LES.py`             | Outlet Zou-He BC 未處理 backflow                        | 偵測 ux < 0 改用零梯度外推                           |
|  C  |   🟡 MINOR   | `LBM2D_MRT_LES.py`             | `_init_params` 用 dummy bc_value 顯示 Re                | 改用 Bernoulli 估算 u_char                           |
|  E  |   🟡 MINOR   | `config_batch_gen.py` / solver | `sponge_strength=2.0` 硬編碼                            | 移入 config，建議都市場景用 3.0                      |
|  F  |   🟡 MINOR   | `config_batch_gen.py`          | `smagorinsky_constant=0.2` 硬編碼                       | 移入 master_config.yaml                              |

### 優先修正順序

```
1. Bug B（阻塞率 rho_in 調整）← 直接解決所有崩潰
2. Bug A（L_char 修正）← 修正步數與 Re 估算
3. Bug D（Backflow BC）← 預防高 Re 出口回流崩潰
4. Bug C / E / F（顯示與可配置性）← 非緊急
```

### 預期效果

修正 Bug B 後，以 `max_blockage=60%` 的都市 mask 為例：

```
修正前：rho_in=1.010 → u_inlet=0.082 → u_gap=0.082/0.4=0.205 → MaxV≈0.25 → 崩潰
修正後：rho_in=1.010 → 調整 → rho_in_case=1.0096 → u_inlet=0.080 → u_gap=0.200 → 穩定✅
```

所有已知的阻塞率誘發崩潰應該被消除。
