import os
import time
import h5py
import zarr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random

# 設定
H5_DIR = "E:/Scott/outputs/Hyper-1/raw"
ZARR_DIR = "E:/Scott/outputs/L1_Zarr"

T_LIST = [1, 50, 100, 150, 200, 250]
HW_LIST = [16, 32, 48, 64, 80, 96]
N_TRIALS = 10  # 每個組合測試 10 次取平均以節省時間


def get_target_case():
    with open("E:/Scott/outputs/Hyper-1/plots/all_cases_summary.json", "r") as f:
        cases = json.load(f)
    test_case = next(c for c in cases if c["status"] == "Success")
    return test_case["case_name"], test_case["run_summary"]["h5_file"]


def timed_read(mode, case_name, h5_file, t_size, hw_size):
    if mode == "h5":
        path = os.path.join(H5_DIR, h5_file)
        with h5py.File(path, "r") as f:
            data = f["turbulence"]
            t_s = random.randint(0, data.shape[0] - t_size)
            h_s = random.randint(0, data.shape[2] - hw_size)
            w_s = random.randint(0, data.shape[3] - hw_size)
            start = time.perf_counter()
            _ = data[t_s : t_s + t_size, :, h_s : h_s + hw_size, w_s : w_s + hw_size]
            return time.perf_counter() - start
    else:
        path = os.path.join(ZARR_DIR, f"{case_name}.zarr")
        root = zarr.open(path, mode="r")
        data = root["turbulence"]
        t_s = random.randint(0, data.shape[0] - t_size)
        h_s = random.randint(0, data.shape[2] - hw_size)
        w_s = random.randint(0, data.shape[3] - hw_size)
        start = time.perf_counter()
        _ = data[t_s : t_s + t_size, :, h_s : h_s + hw_size, w_s : w_s + hw_size]
        return time.perf_counter() - start


def run_multi_benchmark():
    case_name, h5_file = get_target_case()
    results = []

    print(f"🚀 開始大規模 Benchmark... 案子: {case_name}")

    for hw in HW_LIST:
        for t in T_LIST:
            h5_ts = [
                timed_read("h5", case_name, h5_file, t, hw) for _ in range(N_TRIALS)
            ]
            zarr_ts = [
                timed_read("zarr", case_name, h5_file, t, hw) for _ in range(N_TRIALS)
            ]

            results.append(
                {
                    "HW_Size": hw,
                    "T_Size": t,
                    "H5_Time": np.mean(h5_ts),
                    "Zarr_Time": np.mean(zarr_ts),
                    "Speedup": np.mean(h5_ts) / np.mean(zarr_ts),
                }
            )
            print(f"HW={hw}, T={t} | Speedup: {results[-1]['Speedup']:.2f}x")

    df = pd.DataFrame(results)

    # 繪圖
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 圖 1: Speedup Heatmap
    pivot_df = df.pivot(index="HW_Size", columns="T_Size", values="Speedup")
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Speedup Factor (H5_Time / Zarr_Time)\nHigher is Better for Zarr")

    # 圖 2: 趨勢圖 (以 HW=64 為例)
    for hw in HW_LIST:
        subset = df[df["HW_Size"] == hw]
        axes[1].plot(subset["T_Size"], subset["Speedup"], marker="o", label=f"HW={hw}")

    axes[1].set_xlabel("Time Window Size (BOX_T)")
    axes[1].set_ylabel("Speedup Factor")
    axes[1].set_title("Speedup Trend by Window Size")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("h5_vs_zarr_benchmark.png")
    plt.show()


if __name__ == "__main__":
    run_multi_benchmark()
