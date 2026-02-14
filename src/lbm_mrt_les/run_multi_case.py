import argparse
import os
import sys
import time
import yaml  # 需要 pip install pyyaml
import gc  # Garbage Collection
import traceback

from src.lbm_mrt_les.run_one_case import main


def run_batch_simulation():
    parser = argparse.ArgumentParser(
        description="One-to-One Batch Runner (Config[i] -> Mask[i])"
    )

    # 1. Config 資料夾
    parser.add_argument(
        "--config_dir",
        type=str,
        default="src/configs/hyper_configs",
        help="Directory containing YAML config files",
    )

    # 2. Mask 資料夾
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="src/GenMask/generated_maps_advanced",
        help="Directory containing PNG mask files",
    )

    args = parser.parse_args()

    # --- 檢查路徑是否存在 ---
    if not os.path.exists(args.config_dir):
        print(f"[Error] Config directory not found: {args.config_dir}")
        sys.exit(1)

    if not os.path.exists(args.mask_dir):
        print(f"[Error] Mask directory not found: {args.mask_dir}")
        sys.exit(1)

    # --- 搜尋並排序檔案 (關鍵：必須排序以確保一對一順序正確) ---
    config_files = sorted(
        [
            f
            for f in os.listdir(args.config_dir)
            if f.endswith(".yaml") or f.endswith(".yml")
        ]
    )

    mask_files = sorted(
        [f for f in os.listdir(args.mask_dir) if f.lower().endswith(".png")]
    )

    # ==========================================
    # [新增] 嚴格檢查數量是否一致
    # ==========================================
    num_configs = len(config_files)
    num_masks = len(mask_files)

    if num_configs == 0 or num_masks == 0:
        print("[Error] No config or mask files found.")
        sys.exit(1)

    if num_configs != num_masks:
        print("\n" + "=" * 60)
        print(f"\033[91m[CRITICAL ERROR] Count Mismatch!\033[0m")
        print(f"Config files: {num_configs}")
        print(f"Mask files  : {num_masks}")
        print(
            "To run in one-to-one mode, the number of files must be exactly the same."
        )
        print("Please check your directories.")
        print("=" * 60)
        sys.exit(1)

    print(f"Found {num_configs} pairs of (Config + Mask).")
    print("Mode: One-to-One Sequential Execution")
    print("=" * 60)

    # ==========================================
    # 單層迴圈: 使用 zip 同時遍歷兩個列表
    # ==========================================
    # zip 會將 (config_files[0], mask_files[0]), (config_files[1], mask_files[1])... 配對
    for i, (cfg_file, mask_file) in enumerate(zip(config_files, mask_files)):

        job_id = i + 1
        full_config_path = os.path.join(args.config_dir, cfg_file)
        full_mask_path = os.path.join(args.mask_dir, mask_file)

        # 取得檔名 (不含副檔名)
        cfg_name = os.path.splitext(cfg_file)[0]
        mask_name = os.path.splitext(mask_file)[0]

        print(f"\n[Job {job_id}/{num_configs}]")
        print(f"   Config: {cfg_name}")
        print(f"   Mask  : {mask_name}")
        print("-" * 40)

        # --- 記憶體清理 ---
        gc.collect()

        try:
            # --- 呼叫主程式 ---
            # 這裡呼叫你的 main 函數
            # 確保 main.py 內部邏輯能處理這兩個參數
            main(full_config_path, full_mask_path)

            print(f"   >>> \033[92mSuccess\033[0m.")

        except KeyboardInterrupt:
            print("\n[User Abort] Stopping batch run...")
            sys.exit(0)
        except Exception as e:
            print(f"   >>> \033[91m[Error] Failed: {e}\033[0m")
            # 這裡建議把錯誤寫入 log 檔，然後繼續跑下一個
            traceback.print_exc()
            continue

        # 稍作休息，讓 GPU 降溫
        time.sleep(1.0)

    print("\n" + "=" * 60)
    print("All batch jobs finished.")


if __name__ == "__main__":
    run_batch_simulation()
