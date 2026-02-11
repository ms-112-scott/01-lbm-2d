import argparse
import os
import sys
import time
import yaml  # 需要 pip install pyyaml
import gc  # Garbage Collection
import traceback

from main import main


def run_batch_simulation():
    parser = argparse.ArgumentParser(
        description="Multi-Config x Multi-Mask Batch Runner"
    )

    # 1. Config 資料夾 (存放 Re1000.yaml, Re2000.yaml...)
    parser.add_argument(
        "--config_dir",
        type=str,
        default="src/configs/hyper_configs",
        help="Directory containing YAML config files",
    )

    # 2. Mask 資料夾 (存放 mask_01.png, mask_02.png...)
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

    # --- 搜尋並排序檔案 (確保執行順序固定) ---
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

    if not config_files or not mask_files:
        print("[Error] No config or mask files found.")
        sys.exit(1)

    total_jobs = len(config_files) * len(mask_files)
    print(f"Found {len(config_files)} configs and {len(mask_files)} masks.")
    print(f"Total Simulation Jobs: {total_jobs}")
    print("=" * 60)

    job_counter = 0

    # ==========================================
    # 雙層迴圈: Outer Loop (Config) x Inner Loop (Mask)
    # ==========================================
    for cfg_file in config_files:
        for mask_file in mask_files:
            job_counter += 1

            full_config_path = os.path.join(args.config_dir, cfg_file)
            full_mask_path = os.path.join(args.mask_dir, mask_file)

            # 取得檔名 (不含副檔名)，用於建立獨立的輸出資料夾
            cfg_name = os.path.splitext(cfg_file)[0]  # e.g., "sim_config_Re1000"
            mask_name = os.path.splitext(mask_file)[0]  # e.g., "hybrid_adv_0000"

            print(f"\n[Job {job_counter}/{total_jobs}]")
            print(f"   Config: {cfg_name}")
            print(f"   Mask  : {mask_name}")
            print("-" * 40)

            # --- 記憶體清理 (重要！) ---
            # 強制回收 Python 垃圾，防止記憶體洩漏
            gc.collect()

            try:
                # --- [關鍵修改] 呼叫主程式 ---
                # 這裡假設你的 main 函數接收 (config_path, mask_path)
                # 並且在 main 內部會處理路徑

                # 選項 A: 直接呼叫 (如果 main 寫得好，支援參數傳入)
                # 這裡建議傳入 override_output_folder 參數，
                # 讓結果存到: output/Re1000/hybrid_adv_0000/ 避免覆蓋

                # 為了避免改動你的 main.py 太多，這裡示範最安全的「修改 Config 物件」邏輯
                # 但因為 main 通常只收路徑，我們可以用一個小技巧：
                # 在 main.py 裡面加入動態修改輸出路徑的程式碼 (見下方說明)

                main(full_config_path, full_mask_path)

                print(f"   >>> Success.")

            except KeyboardInterrupt:
                print("\n[User Abort] Stopping batch run...")
                sys.exit(0)
            except Exception as e:
                print(f"   >>> [Error] Failed: {e}")
                traceback.print_exc()  # 印出詳細錯誤，方便除錯
                # 失敗後繼續跑下一個 Case，不要停
                continue

            # 稍作休息，讓 GPU 降溫
            time.sleep(1.0)

    print("\n" + "=" * 60)
    print("All batch jobs finished.")


if __name__ == "__main__":
    run_batch_simulation()
