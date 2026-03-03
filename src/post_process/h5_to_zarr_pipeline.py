import json
import h5py
import zarr
import numpy as np
from pathlib import Path
from tqdm import tqdm
from numcodecs import Blosc
from typing import List, Dict, Tuple

# --- Configuration ---
JSON_PATH = Path("C:/Users/User/Desktop/NCA_workspace/01-lbm-2d/outputs/Hyper-1/plots/all_cases_summary.json")
RAW_DIR = Path("C:/Users/User/Desktop/NCA_workspace/01-lbm-2d/outputs/Hyper-1/raw")
OUTPUT_DIR = Path("C:/Users/User/Desktop/NCA_workspace/sim_dataset_zarr64")

CHUNK_T, CHUNK_W = 100, 64
V2_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def get_successful_cases(json_path: Path) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [c for c in data if c.get("status") == "Success" and "run_summary" in c]


def pass1_calculate_global_stats(
    cases: List[Dict], raw_dir: Path
) -> Tuple[np.ndarray, np.ndarray]:
    print("\n[Pass 1] Calculating Global Statistics...")
    sum_x = np.zeros(9, dtype=np.float64)
    sum_x2 = np.zeros(9, dtype=np.float64)
    total_pixels = 0

    for case in tqdm(cases, desc="Scanning H5"):
        h5_path = raw_dir / case["run_summary"]["h5_file"]
        if not h5_path.exists():
            continue

        with h5py.File(h5_path, "r") as f:
            turb = f["turbulence"]
            T, C, H, W = turb.shape
            for t in range(0, T, CHUNK_T):
                chunk = turb[t : min(t + CHUNK_T, T)]
                for c in range(C):
                    data = chunk[:, c].astype(np.float64)
                    sum_x[c] += np.sum(data)
                    sum_x2[c] += np.sum(data**2)
                total_pixels += chunk.shape[0] * H * W

    mean = sum_x / total_pixels
    std = np.sqrt(np.maximum((sum_x2 / total_pixels) - (mean**2), 1e-10))
    return mean, std


def pass2_convert_to_zarr(
    cases: List[Dict],
    raw_dir: Path,
    output_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
):
    print("\n[Pass 2] Converting to Zarr with Precomputed Weights...")
    output_dir.mkdir(parents=True, exist_ok=True)

    m_bc, s_bc = [v.reshape(1, 9, 1, 1).astype(np.float32) for v in (mean, std)]

    for case in tqdm(cases, desc="Processing Cases"):
        h5_path = raw_dir / case["run_summary"]["h5_file"]
        if not h5_path.exists():
            continue

        with h5py.File(h5_path, "r") as h5f:
            zarr_path = output_dir / f"{case['case_name']}.zarr"
            root = zarr.group(store=zarr.DirectoryStore(str(zarr_path)), overwrite=True)

            root.attrs.update(
                {
                    "case_name": case["case_name"],
                    "reynolds_number": case.get("physical_scaled", {}).get(
                        "reynolds_number_calculated"
                    ),
                }
            )

            # --- 1. 原有的物理場轉換 ---
            T, C, H, W = h5f["turbulence"].shape

            # Turbulence
            z_turb = root.require_dataset(
                "turbulence",
                shape=(T, C, H, W),
                chunks=(CHUNK_T, C, H, CHUNK_W),
                dtype="f2",
                compressor=V2_COMPRESSOR,
            )
            for t in range(0, T, CHUNK_T):
                t_e = min(t + CHUNK_T, T)
                z_turb[t:t_e] = (
                    (h5f["turbulence"][t:t_e].astype("f4") - m_bc) / s_bc
                ).astype("f2")

            # Static Mask (0: binary, 1: SDF)
            mask = h5f["static_mask"][:]
            root.require_dataset(
                "static_mask",
                shape=mask.shape,
                chunks=(2, H, CHUNK_W),
                dtype=mask.dtype,
                compressor=V2_COMPRESSOR,
            )[:] = mask

            # Mean Fields
            m_vel = h5f["mean_vel_field"][:].astype("f4")
            m_vel_norm = (m_vel - mean.reshape(9, 1, 1)) / std.reshape(9, 1, 1)
            root.require_dataset(
                "mean_vel_field",
                shape=m_vel_norm.shape,
                chunks=(9, H, CHUNK_W),
                dtype="f2",
                compressor=V2_COMPRESSOR,
            )[:] = m_vel_norm.astype("f2")

            sq = h5f["mean_vel_sq_field"][:]
            root.require_dataset(
                "mean_vel_sq_field",
                shape=sq.shape,
                chunks=(H, CHUNK_W),
                dtype="f2",
                compressor=V2_COMPRESSOR,
            )[:] = sq.astype("f2")

            # --- 2. 關鍵新增：預算抽樣權重圖 (Precomputed Weights) ---
            # 建立一個子組專門存放權重
            weight_grp = root.create_group("sampling_weights")

            # (A) Vor Mode: 使用平均速度平方和作為波動代理
            vor_w = sq.astype("f4")
            vor_w = (vor_w - vor_w.min()) / (vor_w.max() - vor_w.min() + 1e-6)

            # (B) SDF Mode: 反轉 SDF，越近邊界權重越高
            sdf = np.abs(mask[1]).astype("f4")
            sdf_w = np.exp(-sdf / 5.0)  # Sigma=5.0

            # (C) Mix Mode: 兩者混合
            mix_w = 0.5 * vor_w + 0.5 * sdf_w

            # 儲存權重 (使用 f4 以保證抽樣精度，且權重圖很小，不佔空間)
            for name, data in [("vor", vor_w), ("sdf", sdf_w), ("mix", mix_w)]:
                weight_grp.require_dataset(
                    name,
                    shape=data.shape,
                    chunks=(H, CHUNK_W),
                    dtype="f4",
                    compressor=V2_COMPRESSOR,
                )[:] = data


if __name__ == "__main__":
    cases = get_successful_cases(JSON_PATH)
    if not cases:
        exit("No successful cases found.")

    # 1. 計算全局統計資訊
    g_mean, g_std = pass1_calculate_global_stats(cases, RAW_DIR)

    # 2. 關鍵修正：在寫入 JSON 之前，先確保輸出資料夾存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. 儲存統計資訊
    stats_path = OUTPUT_DIR / "global_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": g_mean.tolist(),
                "std": g_std.tolist(),
                "cases": [c["case_name"] for c in cases],
            },
            f,
            indent=4,
        )

    # 4. 執行第二階段轉換
    pass2_convert_to_zarr(cases, RAW_DIR, OUTPUT_DIR, g_mean, g_std)
    print(f"\n🎉 Task Completed. Stats saved to: {stats_path}")
