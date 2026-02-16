import os
import shutil
import argparse
from datetime import datetime
from pathlib import Path  # 現代化路徑處理
import traceback


def backup_artifacts(config_dir, mask_dir, output_dir, dest_root, sim_name="LBM_Sim"):
    """
    備份 Config, Mask 與 Summary 到指定路徑 (e.g. Google Drive)
    """
    # 將字串轉換為 Path 物件
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
    dest_path = dest_root_path / folder_name  # 使用 / 符號組合路徑，自動處理斜線

    try:
        # 3. 建立目標資料夾
        dest_path.mkdir(parents=True, exist_ok=True)
        print(f"\n[Backup] Created backup folder: {dest_path}")

        # 4. 複製 Configs
        dest_config = dest_path / "configs"
        shutil.copytree(config_path, dest_config, dirs_exist_ok=True)
        print(f"  ├─ Copied configs: {config_path} -> {dest_config}")

        # 5. 複製 Masks
        dest_mask = dest_path / "masks"
        shutil.copytree(mask_path, dest_mask, dirs_exist_ok=True)
        print(f"  ├─ Copied masks:   {mask_path} -> {dest_mask}")

        # 6. 複製 Summary JSON
        if output_path.exists():
            summary_files = list(output_path.glob("*summary.json"))  # 使用 glob 搜尋
            if summary_files:
                dest_summary = dest_path / "summary"
                dest_summary.mkdir(exist_ok=True)
                for f_path in summary_files:
                    shutil.copy2(f_path, dest_summary / f_path.name)
                print(f"  ├─ Copied {len(summary_files)} summaries from {output_path}")

        print(f"[Backup] \033[92mSuccess!\033[0m All files backed up to {dest_path}\n")

    except Exception as e:
        print(f"[Backup] \033[91mFailed:\033[0m {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup LBM Simulation Artifacts")

    # 預設路徑
    default_conf = "configs/Hyper"
    default_mask = "mask/Hyper"
    default_out = "outputs/simulation_data/Hyper"

    parser.add_argument("--config_dir", type=str, default=default_conf)
    parser.add_argument("--mask_dir", type=str, default=default_mask)
    parser.add_argument("--output_dir", type=str, default=default_out)
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="e.g. 'G:/我的雲端硬碟/01_碩班/00_個人研究/NCA_workspace/lbm_sim_dataset'",
    )
    parser.add_argument("--name", type=str, default="LBM_Batch_Hyper")

    args = parser.parse_args()

    backup_artifacts(
        args.config_dir, args.mask_dir, args.output_dir, args.dest, args.name
    )
