import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_mask(mask):
    """視覺化 Mask"""
    plt.figure(figsize=(10, 5))
    plt.imshow(mask.T, cmap="gray_r", origin="lower")  # .T 是為了轉置讓 x 軸橫向
    plt.title("Two Rooms Layout (White=Wall, Black=Air)")
    plt.colorbar()
    plt.show()

def plot_verification_results(out_dir, steps, cd, cl, re_num, u_mean, D):
    """
    繪製包含「完美波對照」的驗證圖
    """
    from .physics_utils import fit_sine_wave

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
            f"Analysis Result:
"
            f"Max $C_L$: {cl_max:.4f} (Ref: ~1.0)
"
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

    print(f"
[Validation] Plot saved to: {save_path}")
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

def draw_zone_overlay(gui, zones, split_ratio=0.5, y_offset=0.0):
    """
    在 Taichi GUI 上繪製區域框線，支援上下拼接的畫面
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
