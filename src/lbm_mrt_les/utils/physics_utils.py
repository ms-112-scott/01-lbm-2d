import numpy as np
from scipy.optimize import curve_fit

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

def calculate_characteristic_length(mask):
    """
    計算流場的特徵長度 L。
    定義：Y 軸上的總投影長度 (Total Projected Length)。
    物理意義：這是流體必須繞過的「有效障礙物寬度」，直接決定了
             狹縫處的加速效應 (Venturi effect) 和雷諾數的尺度。
    """
    # 1. 取得 Y 軸投影 (Axis 1 = X軸方向壓縮 -> 得到 Y 軸分佈)
    # Mask: 255=Fluid, 0=Object
    # np.min: 如果一行中有任何黑色像素(0)，該行結果就是 0
    y_projection = np.min(mask, axis=1)

    # 2. 統計被佔用的像素總數 (即特徵長度 L)
    L_char = np.sum(y_projection == 0)

    return int(L_char)

def calculate_simulation_time_scale(config, print_console=False):
    """
    計算並印出 LBM 的物理時間尺度 (Characteristic Time Units)
    公式: Steps per Time Unit = L_char / U_lb
    """
    try:
        # 1. 抓取特徵長度 L (通常是障礙物直徑或流場高度)
        l_char = config["simulation"].get("characteristic_length", 0)

        # 2. 抓取入口速度 U (從邊界條件中解析)
        # 假設 Inlet 設在 boundary_condition 的第一個位置
        # 結構: boundary_condition -> value -> [ [u_x, u_y], ... ]
        u_lb = config["boundary_condition"]["value"][0][0]

        if u_lb == 0 or l_char == 0:
            print(
                "[TimeScale] Warning: U_lb or L_char is 0. Cannot calculate time scale."
            )
            return

        # 3. 計算轉換率
        # 這就是「物理模擬中的 1 秒」對應的 iters
        steps_per_time_unit = l_char / u_lb

        # 4. 取得總模擬步數
        max_steps = config["simulation"]["max_steps"]
        total_physical_time = max_steps / steps_per_time_unit
        if print_console:
            print(f"--- [Physics Time Scale] ---")
            print(f"   L_char (Length) : {l_char} cells")
            print(f"   U_lb   (Velocity): {u_lb} lattice-speed")
            print(f"   -----------------------------------------")
            print(
                f"   \033[96m1 Physical Second (CTU) = {steps_per_time_unit:.1f} iters\033[0m"
            )
            print(
                f"   Total Simulation Time   = {total_physical_time:.2f} Physical Seconds"
            )
            print(f"   -----------------------------------------")

        return steps_per_time_unit

    except Exception as e:
        print(f"[TimeScale] Error parsing config: {e}")
        return 0

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
