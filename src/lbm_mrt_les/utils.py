import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit  # 用於擬合完美波形
import cv2
import shutil
import random
import time

import json
from datetime import datetime


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


def _create_from_png(nx, ny, config, png_path):
    """
    從 PNG 讀取 Mask
    """

    if not png_path or not os.path.exists(png_path):
        raise FileNotFoundError(f"[Error] Mask file not found: {png_path}")

    # 1. 以灰階模式讀取
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"[Error] Failed to load image: {png_path}")

    # 2. 強制縮放到模擬網格大小 (nx, ny)
    # cv2.resize 接受 (width, height) -> (nx, ny)
    # resize 後的 img numpy array 形狀會是 (height, width) -> (ny, nx)
    if img.shape != (ny, nx):
        print(f"  -> Resizing mask from {img.shape[::-1]} to ({nx}, {ny})")
        img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_NEAREST)

    # 3. 二值化轉換
    threshold = 127
    inverse = config["mask"]["invert"]

    if inverse:
        mask = img > threshold
    else:
        mask = img < threshold

    # 4. [關鍵修正] 轉置矩陣 (Transpose)
    # Numpy/OpenCV 是 [y, x] (1024, 2048)
    # Taichi Solver 是 [x, y] (2048, 1024)
    # 必須使用 .T 將其轉置，否則無法塞入 taichi field
    mask = mask.T

    return mask.astype(bool)


def create_mask(config, png_path):
    mask_cfg = config["mask"]
    mask = None
    nx = config["simulation"]["nx"]
    ny = config["simulation"]["ny"]

    if config["mask"]["enable"]:

        if config["mask"]["type"] == "png":
            mask = _create_from_png(nx, ny, config, png_path=png_path)

    # 如果沒有 mask 生成 (或 type 不對)，建立一個全 False (全流體) 的空 mask
    if mask is None:
        mask = np.zeros((ny, nx), dtype=bool)

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
    zone_config = config["domain_zones"]

    # 阻尼層配置
    sponge_y = zone_config["sponge_y"]  # 上下阻尼厚度
    sponge_x = zone_config["sponge_x"]  # 左右阻尼厚度

    # 安全區 (ROI) 配置：切除阻尼層 + 額外緩衝
    buffer = zone_config["buffer"]
    inlet_buffer = zone_config["inlet_buffer"]

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


def get_random_png_path(folder_path):
    """
    從指定資料夾中隨機挑選一張 PNG 圖片，並回傳完整路徑。
    """
    # 1. 檢查資料夾是否存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"[Error] Folder not found: {folder_path}")

    # 2. 列出所有檔案並篩選 .png (不分大小寫)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]

    # 3. 檢查是否有圖
    if not files:
        raise ValueError(f"[Error] No PNG files found in: {folder_path}")

    # 4. 隨機挑選
    selected_file = random.choice(files)

    # 5. 組合完整路徑 (跨平台相容)
    full_path = os.path.join(folder_path, selected_file)

    return full_path


def force_clean_cache():
    """
    [System] 強制清理 Taichi 快取
    解決 Windows 下 'Lock file failed' 的問題
    """
    # 這是 Taichi 在 Windows 的預設路徑，根據你的報錯訊息設定
    cache_path = "C:/taichi_cache/ticache"

    if os.path.exists(cache_path):
        try:
            print(f"[System] Cleaning Taichi cache at: {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)
            # 稍微等待一下 I/O 釋放，避免 Race Condition
            time.sleep(0.5)
        except Exception as e:
            print(f"[Warn] Failed to clean cache: {e}")
    else:
        print("[System] Cache directory not found (Clean start).")


def save_case_metadata(json_path, case_id, metadata):
    """
    [IO] 將單一 Case 的 Metadata 更新到總表 JSON 中 (無 Class 版本)

    Args:
        json_path (str): 總表路徑 (e.g., './output/summary.json')
        case_id (str): 該 Case 的唯一 ID (通常是檔名)
        metadata (dict): 要寫入的數據字典
    """

    # -------------------------------------------------
    # 1. 準備 Numpy 轉換器 (閉包或是直接定義)
    # -------------------------------------------------
    def convert_numpy(obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")

    # -------------------------------------------------
    # 2. 讀取現有的 JSON (Read)
    # -------------------------------------------------
    full_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                full_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"[Warn] JSON {json_path} corrupted or empty. Creating new.")
            full_data = {}

    # -------------------------------------------------
    # 3. 更新數據 (Update)
    # -------------------------------------------------
    # 加上時間戳記
    metadata["_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 使用 case_id 作為 Key (例如 'rect_001.png')
    full_data[case_id] = metadata

    # -------------------------------------------------
    # 4. 寫回檔案 (Write)
    # -------------------------------------------------
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            # 使用 default=convert_numpy 處理所有數值型別
            json.dump(full_data, f, default=convert_numpy, indent=4, ensure_ascii=False)
        print(f"[Metadata] Updated entry '{case_id}' in {os.path.basename(json_path)}")
    except Exception as e:
        print(f"[Error] Failed to save JSON metadata: {e}")


# ==========================================
# 新增：計算特徵長度 (Characteristic Length)
# ==========================================
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
