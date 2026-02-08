import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  # 用於擬合完美波形
import cv2


def load_config(path="config.yaml"):
    """讀取 YAML 設定檔"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading config: {e}")
        sys.exit(1)


def print_reynolds_info(u_char, l_char, nu, shape_name="Characteristic Length"):
    """
    計算並列印雷諾數資訊
    Re = (U * L) / nu
    """
    re = (u_char * l_char) / nu

    print("=" * 40)
    print(f"   REYNOLDS NUMBER CALCULATION")
    print("=" * 40)
    print(f"Characteristic Velocity (U) : {u_char:.6f} (Lattice Speed)")
    print(f"Characteristic Length   (L) : {l_char:.2f}   ({shape_name})")
    print(f"Kinematic Viscosity     (nu): {nu:.6f}")
    print("-" * 40)
    print(f"LBM Reynolds Number (Re)    : {re:.2f}")
    print(f"Physical Reynolds Number    : {re:.2f} (Dimensionless)")
    print("=" * 40)

    return re


def plot_mask(mask):
    """視覺化 Mask"""
    plt.figure(figsize=(10, 5))
    plt.imshow(mask.T, cmap="gray_r", origin="lower")  # .T 是為了轉置讓 x 軸橫向
    plt.title("Two Rooms Layout (White=Wall, Black=Air)")
    plt.colorbar()
    plt.show()


def _create_cylinder_mask(nx, ny, cx, cy, r):
    """產生圓柱障礙物遮罩 (Mask)"""
    # 建立網格座標矩陣
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    # 計算每個點到圓心的距離平方
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    # 產生 Mask (圓內為 1, 圓外為 0)
    mask = np.where(dist_sq <= r**2, 1.0, 0.0)
    return mask


def _create_rect_mask(nx, ny, cx, cy, r):
    """
    產生矩形障礙物遮罩 (Mask)
    x0, y0: 矩形左上角 (或起始點) 座標
    w, h: 矩形的寬度與高度
    """
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    # 判斷點是否在矩形範圍內： x0 <= x < x0+w 且 y0 <= y < y0+h
    mask = np.where(
        (x >= cx - r / 2) & (x < cx + r / 2) & (y >= cy - r / 2) & (y < cy + r / 2),
        1.0,
        0.0,
    )
    return mask


def _create_two_rooms_mask(nx, ny, params):
    mask = np.zeros((nx, ny))

    # --- 1. 原始參數設定 (保持不變) ---
    shift = params.get("shift_left", 50)
    angle = params.get("angle_deg", 20)
    w = params.get("w", 6)
    d_half = params.get("d_half", 8)
    margin_lr = params.get("margin_lr", 350)
    margin_td = params.get("margin_td", 100)

    # 原始邊距計算
    x_start = margin_lr - shift
    x_end = (nx - margin_lr) - shift
    y_start = margin_td
    y_end = ny - margin_td

    room_width = x_end - x_start
    x_mid = x_start + int(room_width / 3)
    y_mid = ny // 2

    # --- 2. 旋轉參數設定 (新增) ---
    # 將角度轉為弧度
    theta = np.radians(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 設定旋轉中心 (Pivot Point)
    # 建議設在「房間的中心」，這樣旋轉時房間才不會跑出畫面
    cx = (x_start + x_end) / 2
    cy = (y_start + y_end) / 2

    # --- 3. 迴圈檢查 (加入座標旋轉) ---
    for i in range(nx):
        for j in range(ny):

            # [核心修改]: 座標逆向旋轉
            # 我們要檢查畫布上的點 (i, j)，在旋轉前的原始空間是對應到哪裡
            dx = i - cx
            dy = j - cy

            # 旋轉公式 (Inverse Rotation):
            # 如果要把物體逆時針轉 20 度，等於座標軸順時針轉 20 度
            # x' = dx * cos + dy * sin
            # y' = -dx * sin + dy * cos
            local_x = dx * cos_t + dy * sin_t + cx
            local_y = -dx * sin_t + dy * cos_t + cy

            # --- 接下來的邏輯完全不變，只是把 i, j 換成 local_x, local_y ---

            # 1. 定義牆壁位置 (使用 local 座標)
            is_left_wall = x_start <= local_x < x_start + w
            is_right_wall = x_end - w <= local_x < x_end

            # 中牆位置
            is_mid_wall = x_mid - w // 2 <= local_x < x_mid + w // 2

            is_top_wall = y_end - w <= local_y < y_end
            is_bottom_wall = y_start <= local_y < y_start + w

            # 2. 定義開口位置
            is_opening_zone = y_mid - d_half <= local_y < y_mid + d_half

            # 用來限制左右牆與中牆的高度範圍
            in_y_range = y_start <= local_y < y_end
            # 用來限制上下牆的寬度範圍
            in_x_range = x_start <= local_x < x_end

            # 3. 繪製邏輯
            # 上下牆
            if (is_top_wall or is_bottom_wall) and in_x_range:
                mask[i, j] = 1.0

            # 垂直牆 (左、右、中) + 避開開口
            if (
                (is_left_wall or is_right_wall or is_mid_wall)
                and (not is_opening_zone)
                and in_y_range
            ):
                mask[i, j] = 1.0

    return mask


def create_mask(config):
    mask_cfg = config.get("mask", {})
    mask = None
    nx = config["simulation"]["nx"]
    ny = config["simulation"]["ny"]

    if mask_cfg.get("enable"):
        m_type = mask_cfg["type"]
        print(f"Generating Mask: {m_type}")

        # --- 更新：Mask 生成邏輯分支 ---
        if m_type == "cylinder":
            p = mask_cfg["params"]
            mask = _create_cylinder_mask(nx, ny, p["cx"], p["cy"], p["r"])

        elif m_type == "rect":  # 新增 rect 判斷
            p = mask_cfg["params"]
            mask = _create_rect_mask(nx, ny, p["cx"], p["cy"], p["r"])

        elif m_type == "room":
            p = mask_cfg["params"]
            mask = _create_two_rooms_mask(nx, ny, p)

    return mask


# ==========================================
# Helper Functions: 物理計算與數據處理
# ==========================================


def get_simulation_strategy(solver, u_inlet):
    """計算模擬策略：總步數與 Pass 數"""
    if u_inlet < 1e-6:
        print("[Warning] Inlet velocity is nearly zero. Defaulting to 100,000 steps.")
        return 100000, 0

    # 計算流體穿過一次場域需要的步數 (Flow-through time)
    steps_per_pass = int(solver.nx / u_inlet)
    target_passes = 5  # DFG 建議跑久一點
    total_steps = steps_per_pass * target_passes

    print("=" * 40)
    print(f"   SIMULATION STRATEGY: {target_passes} PASSES")
    print(f"   (1 Pass ~ {steps_per_pass} steps)")
    print("=" * 40)
    print(f"Domain Length (nx) : {solver.nx}")
    print(f"Inlet Velocity (U) : {u_inlet:.4f}")
    print(f"Target Total Steps : {total_steps}")
    print("=" * 40)

    return total_steps, steps_per_pass


def compute_coefficients(fx_arr, fy_arr, u_max, D, rho=1.0):
    """
    將 Lattice Force 轉換為無因次係數 Cd (阻力) 和 Cl (升力)
    DFG Benchmark 定義: U_mean = 2/3 * U_max (拋物線入口)
    """
    # 拋物線入口的平均速度
    u_mean = (2.0 / 3.0) * u_max

    # 動壓 * 特徵長度 (分母) -> 0.5 * rho * U_mean^2 * D
    denominator = 0.5 * rho * (u_mean**2) * D

    cd_arr = fx_arr / denominator
    cl_arr = fy_arr / denominator

    return cd_arr, cl_arr, u_mean


def fit_sine_wave(t, signal):
    """
    嘗試對信號進行正弦波擬合： y = A * sin(omega * t + phi) + offset
    用於驗證升力是否為完美的卡門渦街震盪
    """

    def sine_func(t, A, omega, phi, offset):
        return A * np.sin(omega * t + phi) + offset

    # 初始猜測
    guess_amp = (np.max(signal) - np.min(signal)) / 2
    guess_offset = np.mean(signal)

    # 透過 FFT 猜測頻率 (簡單估算)
    fft_vals = np.fft.rfft(signal - guess_offset)
    fft_freqs = np.fft.rfftfreq(len(signal))
    guess_freq_idx = np.argmax(np.abs(fft_vals))
    guess_omega = 2 * np.pi * fft_freqs[guess_freq_idx]

    try:
        # 進行曲線擬合
        popt, _ = curve_fit(
            sine_func,
            t,
            signal,
            p0=[guess_amp, guess_omega, 0, guess_offset],
            maxfev=10000,
        )
        fitted_curve = sine_func(t, *popt)
        return fitted_curve, popt  # popt = [A, omega, phi, offset]
    except:
        print("[Warn] Sine wave fitting failed.")
        return None, None


# ==========================================
# Helper Functions: 繪圖與驗證
# ==========================================


def plot_verification_results(out_dir, steps, cd, cl, re_num, u_mean, D):
    """
    繪製包含「完美波對照」的驗證圖
    """
    # 只取後 50% 的數據進行穩定性分析與繪圖 (避免初始震盪影響)
    start_idx = int(len(steps) * 0.5)

    t_stable = steps[start_idx:]
    cd_stable = cd[start_idx:]
    cl_stable = cl[start_idx:]

    # --- 統計數據 ---
    cd_mean = np.mean(cd_stable)
    cl_max = np.max(np.abs(cl_stable))  # 取絕對值最大值作為幅值

    # --- 擬合正弦波 (Perfect Wave) ---
    # 為了擬合方便，我們將 t 歸零
    t_local = np.arange(len(cl_stable))
    fitted_wave, popt = fit_sine_wave(t_local, cl_stable)

    # 計算 Strouhal Number (St)
    st_num = 0.0
    if popt is not None:
        omega = popt[1]  # 角頻率
        freq = omega / (2 * np.pi)  # 週期/step
        # St = f * D / U_mean
        steps_per_sample = 10  # 務必確認這與你模擬時的設定一致
        freq_per_step = (omega / (2 * np.pi)) / steps_per_sample

        # St = f * D / U_mean
        st_num = freq_per_step * D / u_mean

    # --- 開始繪圖 ---
    plt.figure(figsize=(14, 10))
    plt.suptitle(f"DFG 2D Benchmark Validation (Re={int(re_num)})", fontsize=16)

    # 1. 阻力係數 (Drag Coefficient)
    plt.subplot(2, 1, 1)
    plt.plot(steps, cd, label="Simulated $C_D$", color="tab:blue", linewidth=1.5)
    plt.axhline(
        cd_mean,
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"Mean $C_D$ = {cd_mean:.4f}",
    )
    plt.ylabel("Drag Coefficient ($C_D$)")
    plt.title(f"Drag Coefficient History (Ref: 3.22 ~ 3.24)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    # 2. 升力係數 (Lift Coefficient) + 完美波
    plt.subplot(2, 1, 2)
    plt.plot(
        t_stable,
        cl_stable,
        label="Simulated $C_L$ (Stable Region)",
        color="tab:red",
        linewidth=2,
        alpha=0.8,
    )

    if fitted_wave is not None:
        plt.plot(
            t_stable,
            fitted_wave,
            label="Perfect Sine Wave Fit",
            color="lime",
            linestyle="--",
            linewidth=1.5,
        )
        info_text = (
            f"Analysis Result:\n"
            f"Max $C_L$: {cl_max:.4f} (Ref: ~1.0)\n"
            f"Strouhal (St): {st_num:.4f} (Ref: 0.29~0.30)"
        )
        # 在圖表上標註數據
        plt.text(
            0.02,
            0.95,
            info_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.xlabel("Simulation Steps")
    plt.ylabel("Lift Coefficient ($C_L$)")
    plt.title("Lift Coefficient vs Perfect Sine Wave")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(out_dir, f"Validation_Re{int(re_num)}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n[Validation] Plot saved to: {save_path}")
    print(f"[Result] Mean Cd: {cd_mean:.4f}")
    print(f"[Result] Max Cl:  {cl_max:.4f}")
    if popt is not None:
        print(f"[Result] Strouhal Number (St): {st_num:.4f}")


def calcu_gui_size(raw_w, raw_h, max_display_size=None):
    """
    計算最終顯示尺寸
    Returns: (target_w, target_h) 均為整數，且保證 >= 1
    """
    # 預設為原始尺寸
    target_w, target_h = raw_w, raw_h

    # 執行縮放計算
    if max_display_size and max_display_size > 0:
        max_side = max(raw_w, raw_h)
        if max_side > max_display_size:
            scale_ratio = max_display_size / max_side
            target_w = int(raw_w * scale_ratio)
            target_h = int(raw_h * scale_ratio)

    # 防呆機制：確保不會出現 0x0 導致 FFmpeg 崩潰
    target_w = max(1, target_w)
    target_h = max(1, target_h)

    return target_w, target_h * 2


def apply_resize(img, target_w, target_h):
    """
    執行影像縮放
    只有在尺寸不同時才呼叫 cv2.resize，節省效能
    """
    current_h, current_w = img.shape[:2]

    if current_w == target_w and current_h == target_h:
        return img

    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


# ==========================================
# 區域定義 (Region of Interest & Sponge)
# ==========================================
def get_zone_config(config):
    """
    定義阻尼層與安全區的物理座標
    """
    nx = config["simulation"]["nx"]
    ny = config["simulation"]["ny"]
    zone_config = config["display"].get("zone", {})

    # 阻尼層配置
    sponge_y = zone_config.get("sponge_y", 40)  # 上下阻尼厚度
    sponge_x = zone_config.get("sponge_x", 200)  # 左右阻尼厚度

    # 安全區 (ROI) 配置：切除阻尼層 + 額外緩衝
    buffer = zone_config.get("buffer", 50)
    inlet_buffer = zone_config.get("inlet_buffer", 100)

    roi_x_start = inlet_buffer
    roi_x_end = nx - sponge_x - buffer
    roi_y_start = sponge_y + buffer
    roi_y_end = ny - sponge_y - buffer

    return {
        "sponge_y": sponge_y,
        "sponge_x": sponge_x,
        "roi_x_start": roi_x_start,
        "roi_x_end": roi_x_end,
        "roi_y_start": roi_y_start,
        "roi_y_end": roi_y_end,
        "nx": nx,
        "ny": ny,
    }


# ==========================================
# 2. GUI 繪圖工具 (處理視窗比例轉換)
# ==========================================
def draw_zone_overlay(gui, zones, split_ratio=0.5, y_offset=0.0):
    """
    在 Taichi GUI 上繪製區域框線，支援上下拼接的畫面

    Args:
        gui: Taichi GUI 物件
        zones: get_zone_config 回傳的字典
        split_ratio: 如果畫面高度是原本的 2 倍，這裡填 0.5 (表示物理場只佔視窗的一半)
        y_offset: 如果物理場在視窗的上半部，填 0.5；如果在下半部，填 0.0
    """
    nx, ny = zones["nx"], zones["ny"]

    # --- 座標轉換 Helper ---
    # 將物理座標 (x, y) 轉換為 GUI 歸一化座標 (0.0 ~ 1.0)
    def to_gui(x, y):
        u = x / nx
        # y 軸先歸一化到 (0~1)，然後縮放到視窗佔比 (split_ratio)，再加上偏移 (offset)
        v = (y / ny) * split_ratio + y_offset
        return [u, v]

    # --- 1. 繪製阻尼層 (Sponge Layer) - 綠色 ---
    sp_x = zones["sponge_x"]
    sp_y = zones["sponge_y"]
    color_sponge = 0x00FF00

    # 右側阻尼分界線 (垂直)
    # 從 (nx-sp_x, 0) 到 (nx-sp_x, ny)
    gui.line(
        begin=to_gui(nx - sp_x, 0),
        end=to_gui(nx - sp_x, ny),
        color=color_sponge,
        radius=2,
    )

    # 下阻尼分界線 (水平)
    gui.line(begin=to_gui(0, sp_y), end=to_gui(nx, sp_y), color=color_sponge, radius=2)

    # 上阻尼分界線 (水平)
    gui.line(
        begin=to_gui(0, ny - sp_y),
        end=to_gui(nx, ny - sp_y),
        color=color_sponge,
        radius=2,
    )

    # --- 2. 繪製安全區 (ROI) - 紅色 ---
    color_roi = 0xFF0000

    # 左下角與右上角
    p1 = to_gui(zones["roi_x_start"], zones["roi_y_start"])
    p2 = to_gui(zones["roi_x_end"], zones["roi_y_end"])

    # Taichi 的 rect 需要 top-left 和 bottom-right (但在某些版本是 p1, p2 對角)
    # 為了安全起見，我們用 lines 畫矩形，確保在各種 Taichi 版本都正確，且可以控制線條順序
    x0, y0 = p1
    x1, y1 = p2

    # 畫四個邊
    gui.line([x0, y0], [x1, y0], color=color_roi, radius=3)  # 下
    gui.line([x1, y0], [x1, y1], color=color_roi, radius=3)  # 右
    gui.line([x1, y1], [x0, y1], color=color_roi, radius=3)  # 上
    gui.line([x0, y1], [x0, y0], color=color_roi, radius=3)  # 左

    # --- 3. 加入文字標籤 ---
    gui.text("Dataset ROI", pos=[x0, y1 + 0.02], color=color_roi, font_size=20)
    gui.text(
        "Sponge",
        pos=[to_gui(nx - sp_x + 10, ny / 2)[0], to_gui(nx - sp_x + 10, ny / 2)[1]],
        color=color_sponge,
        font_size=20,
    )
