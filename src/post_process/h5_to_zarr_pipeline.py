import os
import json
import h5py
import zarr
import numpy as np
from tqdm import tqdm
from zarr.codecs import BloscCodec
from typing import List, Dict, Tuple

# ==========================================
# 專案工程師設定區 (Configuration)
# ==========================================
JSON_PATH = "E:/Scott/outputs/Hyper-1/plots/all_cases_summary.json"
RAW_DIR = "E:/Scott/outputs/Hyper-1/raw"
OUTPUT_DIR = "E:/Scott/outputs/L1_Zarr"

# Zarr Chunking 策略 (T, C, H, W)
# H 設定為 None 代表不切分高度 (適應 104 或 256)
CHUNK_T = 200
CHUNK_W = 256


def get_successful_cases(json_path: str) -> List[Dict]:
    """讀取 JSON 並過濾出所有 Success 的案例"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 只保留 Success 且包含 h5_file 資訊的案例
    return [c for c in data if c.get("status") == "Success" and "run_summary" in c]


def pass1_calculate_global_stats(
    cases: List[Dict], raw_dir: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    第一階段：計算全域通道的 Mean 與 Std
    採用 Welford 在線演算法或總和累加法以節省記憶體
    """
    print("/n[Pass 1] 開始計算全域統計值 (Global Mean & Std)...")

    n_channels = 9
    sum_x = np.zeros(n_channels, dtype=np.float64)
    sum_x2 = np.zeros(n_channels, dtype=np.float64)
    total_pixels = 0

    for case in tqdm(cases, desc="掃描 H5 檔案"):
        h5_filename = case["run_summary"]["h5_file"]
        h5_path = os.path.join(raw_dir, h5_filename)

        if not os.path.exists(h5_path):
            print(f"警告：找不到檔案 {h5_path}，跳過。")
            continue

        with h5py.File(h5_path, "r") as f:
            # turbulence shape: [T, 9, H, W]
            turb = f["turbulence"]
            T, C, H, W = turb.shape

            # 為了避免 RAM 爆炸，分批讀取時間步
            for t in range(0, T, CHUNK_T):
                t_end = min(t + CHUNK_T, T)
                data_chunk = turb[t:t_end]  # [t_chunk, 9, H, W]

                # 將每個通道展平計算
                for c in range(C):
                    channel_data = data_chunk[:, c, :, :].astype(np.float64)
                    sum_x[c] += np.sum(channel_data)
                    sum_x2[c] += np.sum(channel_data**2)

                total_pixels += (t_end - t) * H * W

    mean = sum_x / total_pixels
    variance = (sum_x2 / total_pixels) - (mean**2)
    # 避免數值誤差導致 variance 為負
    variance = np.maximum(variance, 1e-10)
    std = np.sqrt(variance)

    print(f"-> 全域 Mean: {mean}")
    print(f"-> 全域 Std : {std}")
    return mean, std


def pass2_convert_to_zarr(
    cases: List[Dict], raw_dir: str, output_dir: str, mean: np.ndarray, std: np.ndarray
):
    """
    第二階段：正規化並寫入 Zarr (採用 Float16 與高效壓縮) - Zarr v3 完美相容版
    """
    print("\n[Pass 2] 開始轉換為 Zarr 格式...")
    os.makedirs(output_dir, exist_ok=True)

    # 【核心修正 1】使用 Zarr v3 內建的 BloscCodec，並放進 List 中
    compressors = [BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")]

    # 將 mean/std 轉為 reshape 方便 Broadcasting 計算
    mean_bc = mean.reshape(1, 9, 1, 1).astype(np.float32)
    std_bc = std.reshape(1, 9, 1, 1).astype(np.float32)

    for case in tqdm(cases, desc="轉換進度"):
        h5_filename = case["run_summary"]["h5_file"]
        h5_path = os.path.join(raw_dir, h5_filename)
        case_name = case["case_name"]

        if not os.path.exists(h5_path):
            continue

        zarr_path = os.path.join(output_dir, f"{case_name}.zarr")

        with h5py.File(h5_path, "r") as h5f:
            root = zarr.open_group(zarr_path, mode="w")

            # 寫入專案 MetaData
            root.attrs["case_name"] = case_name
            if "physical_scaled" in case:
                root.attrs["reynolds_number"] = case["physical_scaled"][
                    "reynolds_number_calculated"
                ]

            # 1. 處理 Turbulence
            turb_h5 = h5f["turbulence"]
            T, C, H, W = turb_h5.shape

            # 【核心修正 2】參數改為 compressors=compressors
            turb_zarr = root.require_array(
                "turbulence",
                shape=(T, C, H, W),
                chunks=(CHUNK_T, C, H, CHUNK_W),
                dtype=np.float16,
                compressors=compressors,
            )

            for t in range(0, T, CHUNK_T):
                t_end = min(t + CHUNK_T, T)
                data_chunk = turb_h5[t:t_end].astype(np.float32)

                # Channel-wise 正規化
                norm_chunk = (data_chunk - mean_bc) / std_bc

                turb_zarr[t:t_end] = norm_chunk.astype(np.float16)

            # 2. 處理 Static Mask
            mask_h5 = h5f["static_mask"][:]
            mask_zarr = root.require_array(
                "static_mask",
                shape=mask_h5.shape,
                chunks=(2, H, CHUNK_W),
                dtype=mask_h5.dtype,
                compressors=compressors,
            )
            mask_zarr[:] = mask_h5

            # 3. 處理 Mean Fields
            mean_vel = h5f["mean_vel_field"][:].astype(np.float32)
            mean_vel_norm = (mean_vel - mean.reshape(9, 1, 1)) / std.reshape(9, 1, 1)
            mean_zarr = root.require_array(
                "mean_vel_field",
                shape=mean_vel_norm.shape,
                chunks=(9, H, CHUNK_W),
                dtype=np.float16,
                compressors=compressors,
            )
            mean_zarr[:] = mean_vel_norm.astype(np.float16)

            # 4. 處理 Mean Vel Sq Field
            vel_sq = h5f["mean_vel_sq_field"][:].astype(np.float16)
            sq_zarr = root.require_array(
                "mean_vel_sq_field",
                shape=vel_sq.shape,
                chunks=(H, CHUNK_W),
                dtype=np.float16,
                compressors=compressors,
            )
            sq_zarr[:] = vel_sq


if __name__ == "__main__":
    # 1. 取得有效案例
    successful_cases = get_successful_cases(JSON_PATH)
    print(f"共找到 {len(successful_cases)} 個 Success 案例準備處理。")

    # 2. 計算全域統計
    global_mean, global_std = pass1_calculate_global_stats(successful_cases, RAW_DIR)

    # 3. 儲存全域統計供後續 PyTorch DataLoader 使用
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stats_dict = {
        "mean": global_mean.tolist(),
        "std": global_std.tolist(),
        "success_cases": [c["case_name"] for c in successful_cases],
    }
    with open(os.path.join(OUTPUT_DIR, "global_stats.json"), "w") as f:
        json.dump(stats_dict, f, indent=4)

    # 4. 執行轉換
    pass2_convert_to_zarr(
        successful_cases, RAW_DIR, OUTPUT_DIR, global_mean, global_std
    )

    print("/n✅ 所有 HDF5 檔案已成功轉換並壓縮至 L1_Zarr_Archive/ !")
