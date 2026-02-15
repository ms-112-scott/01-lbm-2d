import h5py
import numpy as np
import os
import glob
from tqdm import tqdm

# ================= Configuration =================
SOURCE_ROOT = "output"  # 原始模擬資料根目錄
POSTPROCESS_ROOT = "output_postprocess"  # 後處理結果輸出目錄
DATASET_KEY = "velocity"  # H5 內的 dataset 名稱 (例如 'velocity', 'u', 'flow')
# =================================================


def read_h5_data(filepath, key):
    """讀取 H5 檔案並轉為 float64"""
    with h5py.File(filepath, "r") as f:
        if key not in f:
            # 自動嘗試抓取第一個 key
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"Empty h5 file: {filepath}")
            key = keys[0]
        data = f[key][:]
    return data.astype(np.float64)


def save_h5_data(filepath, data, key_name, attrs=None):
    """封裝存檔邏輯"""
    with h5py.File(filepath, "w") as f:
        dset = f.create_dataset(key_name, data=data, compression="gzip")
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v


def process_batch(batch_name):
    """
    處理單一 Batch：
    1. Pass 1: 計算 RANS Mean & Sum Magnitude
    2. Pass 2: 計算 Fluctuations (u' = u - u_mean)
    """

    # 1. 定義路徑
    src_dir = os.path.join(SOURCE_ROOT, batch_name, "h5_SimData")
    dest_dir = os.path.join(POSTPROCESS_ROOT, batch_name)
    fluc_dir = os.path.join(dest_dir, "fluctuations")  # 專門放脈衝場的資料夾

    if not os.path.exists(src_dir):
        print(f"[Skip] Source dir not found: {src_dir}")
        return

    # 建立輸出目錄
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(fluc_dir, exist_ok=True)

    # 獲取檔案列表
    h5_files = sorted(glob.glob(os.path.join(src_dir, "*.h5")))
    if not h5_files:
        print(f"[Skip] No h5 files in {batch_name}")
        return

    print(f"\n🚀 Processing Batch: {batch_name}")
    print(f"   📂 Input: {src_dir}")
    print(f"   📂 Output: {dest_dir}")

    # ==========================================
    # Pass 1: 計算時間平均 (Time Averaging)
    # ==========================================
    velocity_sum = None
    mag_sum = None
    count = 0

    print("   👉 Pass 1: Calculating RANS Mean...")
    for fpath in tqdm(h5_files, unit="frame", ncols=80):
        try:
            # 讀取
            u_inst = read_h5_data(fpath, DATASET_KEY)

            # 計算 Magnitude (假設最後一維是 u,v 分量)
            # 若 shape 為 (H, W, 2) -> axis=-1; 若 (2, H, W) -> axis=0
            axis_dim = -1 if u_inst.shape[-1] == 2 else 0
            u_mag = np.linalg.norm(u_inst, axis=axis_dim)

            # 初始化 Accumulator
            if velocity_sum is None:
                velocity_sum = np.zeros_like(u_inst)
                mag_sum = np.zeros_like(u_mag)

            # 累加
            velocity_sum += u_inst
            mag_sum += u_mag
            count += 1

        except Exception as e:
            print(f"      [Warn] Error reading {os.path.basename(fpath)}: {e}")

    if count == 0:
        print("      [Error] No valid frames found.")
        return

    # 計算平均值
    u_mean = velocity_sum / count
    mag_sum_final = mag_sum  # 根據需求，這是 sum 不是 mean

    # 存檔 RANS & Sum Mag
    save_h5_data(
        os.path.join(dest_dir, "rans.h5"),
        u_mean,
        "mean_velocity",
        {"description": "Time-Averaged Velocity Field (RANS)", "frames": count},
    )

    save_h5_data(
        os.path.join(dest_dir, "sum_mag.h5"),
        mag_sum_final,
        "sum_magnitude",
        {"description": "Accumulated Velocity Magnitude", "frames": count},
    )

    print(f"      ✅ Saved rans.h5 & sum_mag.h5")

    # ==========================================
    # Pass 2: 計算湍流脈衝 (Fluctuations)
    # u' = u_raw - u_mean
    # ==========================================
    print("   👉 Pass 2: Calculating Fluctuations (u' = u - u_mean)...")

    # 為了節省記憶體，我們必須再讀一次檔案，而不是把所有 frames 存在 RAM
    for fpath in tqdm(h5_files, unit="frame", ncols=80):
        try:
            # 讀取原始瞬時場
            u_inst = read_h5_data(fpath, DATASET_KEY)

            # 計算脈衝 (Broadcasting: (H,W,2) - (H,W,2))
            u_prime = u_inst - u_mean

            # 構建輸出檔名 (保持原始檔名，加上 prefix 或放在資料夾內)
            fname = os.path.basename(fpath)
            out_name = f"fluc_{fname}"
            out_path = os.path.join(fluc_dir, out_name)

            # 存檔
            save_h5_data(
                out_path,
                u_prime,
                "fluctuation",
                {"description": "Instantaneous Turbulent Fluctuation (u - u_mean)"},
            )

        except Exception as e:
            print(f"      [Warn] Error processing fluctuation for {fname}: {e}")

    print(f"      ✅ Saved {count} fluctuation fields to /fluctuations/")


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source directory '{SOURCE_ROOT}' does not exist.")
        exit()

    # 掃描 output/ 下的所有資料夾
    subdirs = [
        d
        for d in os.listdir(SOURCE_ROOT)
        if os.path.isdir(os.path.join(SOURCE_ROOT, d))
    ]

    print(f"--- LBM Post-Processing: RANS & Fluctuations ---")
    print(f"Total Batches Found: {len(subdirs)}\n")

    for batch in subdirs:
        process_batch(batch)

    print("\n--- All tasks completed ---")
