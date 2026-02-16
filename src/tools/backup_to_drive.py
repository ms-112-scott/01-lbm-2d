import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import traceback


def backup_artifacts(config_dir, mask_dir, output_dir, dest_root, sim_name="LBM_Sim"):
    """
    備份 Config, Mask, Summary (JSON) 與 Data (H5) 到指定路徑
    """
    config_path = Path(config_dir)
    mask_path = Path(mask_dir)
    output_path = Path(output_dir)
    dest_root_path = Path(dest_root)

    # 1. 檢查來源是否存在
    if not config_path.exists():
        print(f"\033[91m[Error]\033[0m Config directory not found: {config_path}")
        return
    if not mask_path.exists():
        print(f"\033[91m[Error]\033[0m Mask directory not found: {mask_path}")
        return

    # 2. 產生備份資料夾名稱
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{sim_name}_{timestamp}"
    dest_path = dest_root_path / folder_name

    try:
        # 3. 建立目標資料夾
        dest_path.mkdir(parents=True, exist_ok=True)
        print(f"\n[Backup] Starting backup to: {dest_path}")

        # 4. 複製 Configs (完整資料夾)
        dest_config = dest_path / "configs"
        shutil.copytree(config_path, dest_config, dirs_exist_ok=True)
        print(f"  ├─ [Configs] Copied from {config_path}")

        # 5. 複製 Masks (完整資料夾)
        dest_mask = dest_path / "masks"
        shutil.copytree(mask_path, dest_mask, dirs_exist_ok=True)
        print(f"  ├─ [Masks]   Copied from {mask_path}")

        # 6. 複製輸出檔案 (Summary JSON 與 H5 Files)
        if output_path.exists():
            # 使用 rglob 遞迴搜尋子資料夾中的所有相關檔案
            summary_files = list(output_path.rglob("*summary.json"))
            h5_files = list(output_path.rglob("*.h5"))

            all_files = summary_files + h5_files

            if all_files:
                dest_output = dest_path / "data_outputs"
                dest_output.mkdir(exist_ok=True)
                for f_path in all_files:
                    # 保持簡單平鋪到 data_outputs 或是可以保留目錄結構
                    # 這裡選擇平鋪，但加上檔名前綴避免衝突（如果有同名的話）
                    shutil.copy2(f_path, dest_output / f_path.name)

                print(
                    f"  ├─ [Data]    Copied {len(summary_files)} JSONs and {len(h5_files)} H5 files"
                )
            else:
                print(f"  ├─ [Data]    No summary or h5 files found in {output_path}")
        else:
            print(f"  ├─ [Error]   Output path does not exist: {output_path}")

        print(f"[Backup] \033[92mSuccess!\033[0m All artifacts secured.\n")

    except Exception as e:
        print(f"[Backup] \033[91mFailed:\033[0m {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup LBM Simulation Artifacts")

    # 1. 來源預設路徑 (設定為包含所有數據的根目錄)
    default_conf = "configs/Hyper"
    default_mask = "mask/Hyper"
    default_out = "outputs/simulation_data"

    # 2. 目的地預設路徑 (你的 Google Drive 路徑)
    default_dest = "G:/我的雲端硬碟/01_碩班/00_個人研究/NCA_workspace/lbm_sim_dataset"

    parser.add_argument("--config_dir", type=str, default=default_conf)
    parser.add_argument("--mask_dir", type=str, default=default_mask)
    parser.add_argument("--output_dir", type=str, default=default_out)

    parser.add_argument(
        "--dest",
        type=str,
        default=default_dest,
        help=f"Destination Root Path (Default: {default_dest})",
    )
    parser.add_argument("--name", type=str, default="LBM_Batch_Hyper")

    args = parser.parse_args()

    backup_artifacts(
        args.config_dir, args.mask_dir, args.output_dir, args.dest, args.name
    )
