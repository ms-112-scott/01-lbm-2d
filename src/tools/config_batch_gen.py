import yaml
import os
import sys
import re
import glob
import copy
import argparse


def load_yaml(path):
    """讀取 YAML 設定檔"""
    if not os.path.exists(path):
        print(f"[Error] Config file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_case_config(base_template, run_params):
    """
    根據模板與運算出的參數，產生單一 Case 的 Config
    """
    # 深拷貝模板，避免修改到原始讀取的資料
    config = copy.deepcopy(base_template)

    # 解構參數
    sim_name = run_params["sim_name"]
    nu_value = run_params["nu"]
    l_char = run_params["l_char"]
    re_value = run_params["re"]
    u_lb = run_params["u_lb"]
    interval = run_params["interval"]
    mask_path = run_params["mask_path"]
    data_save_root = run_params["data_save_root"]

    # 1. Simulation Section
    config["simulation"]["name"] = sim_name
    config["simulation"]["nu"] = float(f"{nu_value:.6f}")
    config["simulation"]["characteristic_length"] = float(l_char)
    config["simulation"]["compute_step_size"] = interval

    # 2. Outputs Section
    # Update GUI
    config["outputs"]["gui"]["interval_steps"] = interval

    # Update Video
    config["outputs"]["video"]["interval_steps"] = interval
    config["outputs"]["video"]["filename"] = f"video_{sim_name}.mp4"

    # Update Dataset
    config["outputs"]["dataset"]["interval_steps"] = interval
    # 組合輸出路徑 (可選：是否要依 Re 分資料夾，這裡維持原本邏輯)
    config["outputs"]["dataset"]["folder"] = data_save_root

    # 3. Boundary Condition (動態注入 U_LB)
    # 假設 value 格式是 [ [u, v], ... ]，更新第一個入口速度
    config["boundary_condition"]["value"] = [
        [float(u_lb), 0.0],  # West
        [0.0, 0.0],  # East
        [0.0, 0.0],  # South
        [0.0, 0.0],  # North
    ]

    # 4. Mask
    config["mask"]["path"] = mask_path

    return config


def main():
    # 1. 解析命令列參數
    parser = argparse.ArgumentParser(
        description="Generate LBM configs from a master YAML."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="master_config.yaml",
        help="Path to the master config file",
    )
    args = parser.parse_args()

    # 2. 讀取外部設定
    print(f"Loading master config: {args.config}")
    master_cfg = load_yaml(args.config)

    # 提取設定區塊
    settings = master_cfg["settings"]
    physics = master_cfg["physics_control"]
    base_template = master_cfg["template"]

    # 提取路徑與參數
    mask_dir = settings["mask_dir"]
    output_dir = settings["output_dir"]
    data_save_root = settings["data_save_root"]

    re_list = physics["re_list"]
    u_lb = physics["u_lb"]
    saves_per_phys_sec = physics["saves_per_physical_second"]

    # 3. 準備輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    # 4. 搜尋 Mask 檔案
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    if not mask_files:
        print(f"[Error] No PNG files found in {mask_dir}")
        return

    mask_files.sort()
    print(f"--- Found {len(mask_files)} maps. Output dir: {output_dir} ---")
    print(f"--- Strategy: Cyclic Re list {re_list} ---")

    valid_count = 0

    # 5. 核心生成迴圈
    for mask_path in mask_files:
        filename = os.path.basename(mask_path)
        filename_stem = os.path.splitext(filename)[0]

        # 提取特徵長度 L
        match = re.search(r"_L(\d+)", filename)
        if match:
            l_char = float(match.group(1))
        else:
            print(f"[Warning] No L found in {filename}, skipping.")
            continue

        # 決定 Re (循環)
        re_index = valid_count % len(re_list)
        re_val = re_list[re_index]

        # 物理計算
        nu = (u_lb * l_char) / re_val
        steps_per_phys_sec = l_char / u_lb

        target_interval = int(steps_per_phys_sec / saves_per_phys_sec)
        if target_interval < 1:
            target_interval = 1

        # 準備傳遞給生成器的參數包
        run_params = {
            "sim_name": f"{filename_stem}_Re{re_val}",
            "nu": nu,
            "l_char": l_char,
            "re": re_val,
            "u_lb": u_lb,
            "interval": target_interval,
            "mask_path": mask_path,
            "data_save_root": data_save_root,
        }

        # 生成最終 Config 字典
        final_config = generate_case_config(base_template, run_params)

        # 寫入檔案
        config_filename = f"cfg_{filename_stem}_Re{re_val}.yaml"
        full_config_path = os.path.join(output_dir, config_filename)

        with open(full_config_path, "w") as f:
            yaml.dump(final_config, f, sort_keys=False, default_flow_style=None)

        print(
            f"[{valid_count+1:03d}] {filename} -> Re={re_val}, nu={nu:.6f} | Saved: {config_filename}"
        )

        valid_count += 1

    print(f"\n[Done] Generated {valid_count} configs.")


if __name__ == "__main__":
    main()
