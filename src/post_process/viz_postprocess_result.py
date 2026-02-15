import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import cm

# ================= Configuration =================
POSTPROCESS_ROOT = "output_postprocess"  # 指向你的後處理資料夾
TARGET_BATCH = "Hyper"  # 你要畫的 Batch 名稱
# =============================================


def read_h5(path, key=None):
    """讀取 H5 並自動找 Key"""
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return None
    with h5py.File(path, "r") as f:
        if key is None:
            key = list(f.keys())[0]
        data = f[key][:]
    return data


def flow_to_rgb(u, v, scale=1.0):
    """
    將 2D 速度場 (u, v) 轉換為 RGB 圖像以便視覺化。
    R: U component (normalized)
    G: V component (normalized)
    B: Magnitude (Optional, here 0.5 for contrast)
    """
    h, w = u.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # 計算 Magnitude 用於 Normalize
    mag = np.sqrt(u**2 + v**2)
    max_val = np.percentile(mag, 99) if np.max(mag) > 0 else 1.0  # 抗 Outlier

    # Normalize 到 -1 ~ 1 區間 (視需求調整)
    u_norm = np.clip(u / (max_val * scale), -1, 1)
    v_norm = np.clip(v / (max_val * scale), -1, 1)

    # Mapping: 0 在 0.5 的位置 (灰色是靜止)
    # R channel = U flow (0.5 + u/2)
    # G channel = V flow (0.5 + v/2)
    rgb[..., 0] = 0.5 + 0.5 * u_norm
    rgb[..., 1] = 0.5 + 0.5 * v_norm
    rgb[..., 2] = 0.5 + 0.5 * (mag / max_val)  # Blue 為強度

    return np.clip(rgb, 0, 1)


def viz_batch(batch_name):
    base_dir = os.path.join(POSTPROCESS_ROOT, batch_name)
    fluc_dir = os.path.join(base_dir, "fluctuations")

    # 1. Load Static Data (RANS & SumMag)
    rans_path = os.path.join(base_dir, "rans.h5")
    sum_mag_path = os.path.join(base_dir, "sum_mag.h5")

    u_mean = read_h5(rans_path, "mean_velocity")  # (H, W, 2)
    sum_mag = read_h5(
        sum_mag_path, "sum_magnitude"
    )  # (H, W) or (H, W, 2) -> check shape

    if u_mean is None:
        return

    # 處理 Sum Mag 如果它是向量
    if sum_mag.ndim == 3:
        sum_mag = np.linalg.norm(sum_mag, axis=-1)

    # 2. Get Fluctuation Files
    fluc_files = sorted(glob.glob(os.path.join(fluc_dir, "*.h5")))
    num_files = len(fluc_files)

    if num_files == 0:
        print("No fluctuation files found.")
        return

    # ==========================================
    # Task 1: Turbulent Time Series (2x5)
    # ==========================================
    print("Generating Time Series Plot...")
    fig1, axes1 = plt.subplots(2, 5, figsize=(20, 7))
    fig1.suptitle(f"{batch_name} - Turbulent Fluctuations (RGB: U/V/Mag)", fontsize=16)

    # 選取 10 個等間距的時間點
    indices = np.linspace(0, num_files - 1, 10, dtype=int)

    for i, idx in enumerate(indices):
        ax = axes1.flat[i]

        # 讀取脈衝場 u'
        u_prime = read_h5(fluc_files[idx])  # (H, W, 2)

        # 轉換為 RGB 顯示 (突顯結構)
        rgb_img = flow_to_rgb(u_prime[..., 0], u_prime[..., 1], scale=1.5)

        ax.imshow(rgb_img, origin="lower")  # LBM 通常 origin 在左下
        ax.set_title(f"T={idx}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    # ==========================================
    # Task 2: Comparison (Raw vs RANS vs SumMag)
    # ==========================================
    print("Generating Comparison Plot...")

    # 重建最後一幀的 Raw Data: U_raw = U_mean + u'_last
    u_prime_last = read_h5(fluc_files[-1])
    u_last_raw = u_mean + u_prime_last

    # 計算 Magnitude 用於畫圖
    mag_last_raw = np.linalg.norm(u_last_raw, axis=-1)
    mag_mean = np.linalg.norm(u_mean, axis=-1)

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Last Frame Raw Magnitude
    im1 = axes2[0].imshow(mag_last_raw, cmap="magma", origin="lower")
    axes2[0].set_title("Last Frame (Raw Magnitude)")
    plt.colorbar(im1, ax=axes2[0], fraction=0.046, pad=0.04)

    # Plot 2: RANS Mean Magnitude
    im2 = axes2[1].imshow(mag_mean, cmap="magma", origin="lower")
    axes2[1].set_title("Time-Averaged Mean (RANS)")
    plt.colorbar(im2, ax=axes2[1], fraction=0.046, pad=0.04)

    # Plot 3: Sum Magnitude (Accumulated Energy)
    im3 = axes2[2].imshow(sum_mag, cmap="viridis", origin="lower")
    axes2[2].set_title("Sum Magnitude (Total Energy)")
    plt.colorbar(im3, ax=axes2[2], fraction=0.046, pad=0.04)

    for ax in axes2:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 如果你想自動掃描，可以用 os.listdir
    # 這裡示範指定單一 Batch
    viz_batch(TARGET_BATCH)
