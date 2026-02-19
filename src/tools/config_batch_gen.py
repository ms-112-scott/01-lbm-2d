import yaml
import os
import re
import glob
import copy
import argparse
from pathlib import Path

def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_case_config(base_template, run_params):
    config = copy.deepcopy(base_template)
    
    sim_name = run_params["sim_name"]
    nu_value = run_params["nu"]
    l_char = run_params["l_char"]
    re_value = run_params["re"]
    u_lb = run_params["u_lb"]
    interval = run_params["interval"]
    mask_path = run_params["mask_path"]
    data_save_root = run_params["data_save_root"]
    project_name = run_params["project_name"]

    config["simulation"]["name"] = sim_name
    config["simulation"]["nu"] = float(f"{nu_value:.6f}")
    config["simulation"]["characteristic_length"] = float(l_char)
    config["simulation"]["compute_step_size"] = interval

    # 【策略干預】：低 Re 關閉 LES，避免過度耗散抹除自然渦流
    if re_value < 1000:
        config["simulation"]["smagorinsky_constant"] = 0.0
    else:
        config["simulation"]["smagorinsky_constant"] = 0.2

    config["outputs"]["project_name"] = project_name
    config["outputs"]["data_save_root"] = data_save_root
    config["outputs"]["target_re"] = int(re_value)
    
    config["outputs"]["gui"]["interval_steps"] = interval
    config["outputs"]["video"]["interval_steps"] = interval
    config["outputs"]["video"]["filename"] = f"{sim_name}.mp4"
    config["outputs"]["dataset"]["interval_steps"] = interval

    if "folder" in config["outputs"]["dataset"]:
        del config["outputs"]["dataset"]["folder"]

    config["boundary_condition"]["value"] = [[float(f"{u_lb:.4f}"), 0.0]] + [[0.0, 0.0]] * 3
    config["mask"]["path"] = mask_path

    return config

def main():
    parser = argparse.ArgumentParser(description="Generate LBM configs from a master YAML.")
    parser.add_argument("-c", "--config", default="master_config.yaml", help="Path to the master config file")
    args = parser.parse_args()

    master_cfg = load_yaml(args.config)
    settings = master_cfg["settings"]
    physics = master_cfg["physics_control"]
    base_template = master_cfg["template"]
    re_list = master_cfg["physics_control"]["re_list"]

    project_name = settings["project_name"]
    project_dir = os.path.join("SimCases", project_name)
    mask_dir = os.path.join(project_dir, "masks")
    output_dir = os.path.join(project_dir, "configs")
    data_save_root = os.path.join("outputs", project_name) 

    base_u_lb = physics["u_lb"]
    saves_per_phys_sec = physics["saves_per_physical_second"]

    os.makedirs(output_dir, exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not mask_files:
        print(f"[Error] No PNG files found in the project mask directory: {mask_dir}")
        return

    print(f"--- Found {len(mask_files)} maps. Start generating configs... ---")
    success_count = 0

    for i, mask_path in enumerate(mask_files):
        filename_stem = os.path.splitext(os.path.basename(mask_path))[0]
        
        # 【修改點】：依照順序從 master_config 取出目標 Re
        target_re = re_list[i % len(re_list)]
        
        # 【修改點】：修復 args.num_maps 未定義的問題，改用 len(mask_files)
        print(f"Generating map {i+1}/{len(mask_files)} (Target Re: {target_re})...")
        
        # 僅需從檔名抽取特徵長度 L
        match_l = re.search(r"L(\d+)", filename_stem)

        if not match_l:
            print(f"[Warning] Could not parse L from filename: {filename_stem}, skipping.")
            continue
        
        # 【修改點】：直接將目標 Re 賦予 re_val，取代錯誤的 match_re.group(1)
        re_val = int(target_re)
        l_char = float(match_l.group(1))

        # ==========================================
        # 【數值干預】：自動微調 u_lb 以滿足 Tau 穩定性
        # ==========================================
        current_u_lb = base_u_lb
        nu = (current_u_lb * l_char) / re_val
        tau = 3.0 * nu + 0.5
        
        # 當 tau 太低，且 u_lb 還沒頂到馬赫數極限 (0.15) 時，微調 u_lb
        while tau < 0.505 and current_u_lb < 0.15:
            current_u_lb += 0.005  # 每次微調 0.005
            nu = (current_u_lb * l_char) / re_val
            tau = 3.0 * nu + 0.5
            
        # 檢查防呆極限
        if tau < 0.505:
            print(f"[拒絕生成] 案例 {filename_stem}:")
            print(f"  -> 致命錯誤：為了達到 Re={re_val}，即使 u_lb 逼近極限 0.15，Tau ({tau:.4f}) 依然低於 0.505。")
            print(f"  -> 策略建議：此地圖解析度 (L={l_char}) 太低，無法支撐如此高的雷諾數。請捨棄此案例或提高 Map 解析度。\n")
            continue # 直接跳過這個案例，不生成 YAML

        if current_u_lb > base_u_lb:
            print(f"[數值微調] 案例 {filename_stem}: 為維持穩定性，u_lb 自動從 {base_u_lb} 升至 {current_u_lb:.3f} (Tau={tau:.4f})")
        # ==========================================

        # 使用調整後的 current_u_lb 重新計算採樣間隔
        steps_per_phys_sec = l_char / current_u_lb
        target_interval = max(1, int(steps_per_phys_sec / saves_per_phys_sec))

        run_params = {
            "sim_name": filename_stem, # 使用乾淨的檔名作為 sim_name
            "nu": nu, "l_char": l_char, "re": re_val, "u_lb": current_u_lb,
            "interval": target_interval, "mask_path": mask_path,
            "data_save_root": data_save_root, "project_name": project_name,
        }

        final_config = generate_case_config(base_template, run_params)
        config_filename = f"Re{re_val}_{filename_stem}.yaml"
        full_config_path = os.path.join(output_dir, config_filename)

        with open(full_config_path, "w") as f:
            yaml.dump(final_config, f, sort_keys=False, default_flow_style=None)

        print(f"[{success_count+1:03d}] {filename_stem} | Re={re_val} | 保存成功")
        success_count += 1

    print(f"\n[Done] 嘗試生成 {len(mask_files)} 個，成功生成 {success_count} 個 Configs。")

if __name__ == "__main__":
    main()