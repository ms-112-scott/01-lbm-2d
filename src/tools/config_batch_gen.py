import yaml
import os
import sys
import re
import glob
import copy
import argparse
import random
from pathlib import Path
from config_utils import get_sampled_value


def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_case_config(base_template, run_params, physical_constants):
    config = copy.deepcopy(base_template)
    config["physical_constants"] = physical_constants
    
    sim_name = run_params["sim_name"]
    nu_value = run_params["nu"]
    l_char = run_params["l_char"]
    re_value = run_params["re"]
    u_lb = run_params["u_lb"]
    interval = run_params["interval"]
    mask_path = run_params["mask_path"]
    data_save_root = run_params["data_save_root"]
    project_name = run_params["project_name"]
    warmup_steps = run_params["warmup_steps"]

    config["simulation"]["name"] = sim_name
    config["simulation"]["nu"] = float(f"{nu_value:.6f}")
    config["simulation"]["characteristic_length"] = float(l_char)
    config["simulation"]["compute_step_size"] = interval
    config["simulation"]["warmup_steps"] = warmup_steps  # Set dynamic warmup steps

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
    physical_constants = master_cfg["physical_constants"]
    re_list_config = master_cfg["physics_control"]["re_list"]

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
        
        target_re = get_sampled_value(re_list_config)
        if target_re is None:
            print(f"  [Warning] Could not sample a valid Re value from 're_list' in config. Skipping.")
            continue

        print(f"Generating config for {filename_stem} (Target Re: {int(target_re)})...")
        
        match_l = re.search(r"L(\d+)", filename_stem)
        if not match_l:
            print(f"  [Warning] Could not parse L from filename, skipping.")
            continue
        
        re_val = int(target_re)
        l_char = float(match_l.group(1))

        current_u_lb = base_u_lb
        nu = (current_u_lb * l_char) / re_val
        tau = 3.0 * nu + 0.5
        
        while tau < 0.505 and current_u_lb < 0.15:
            current_u_lb += 0.005
            nu = (current_u_lb * l_char) / re_val
            tau = 3.0 * nu + 0.5
            
        if tau < 0.505:
            print(f"  [Error] Cannot generate stable config for Re={re_val} with L={l_char}. Tau ({tau:.4f}) is too low. Skipping.")
            continue

        if current_u_lb > base_u_lb:
            print(f"  [Info] u_lb adjusted to {current_u_lb:.3f} to maintain stability (Tau={tau:.4f})")

        # Dynamically calculate warmup_steps as 1 pass
        nx_val = base_template["simulation"]["nx"]
        warmup_steps = int(nx_val / current_u_lb) if current_u_lb > 0 else 0
        
        steps_per_phys_sec = l_char / current_u_lb if current_u_lb > 0 else 0
        target_interval = max(1, int(steps_per_phys_sec / saves_per_phys_sec)) if steps_per_phys_sec > 0 else 1

        run_params = {
            "sim_name": filename_stem,
            "nu": nu, "l_char": l_char, "re": re_val, "u_lb": current_u_lb,
            "interval": target_interval, "mask_path": mask_path,
            "data_save_root": data_save_root, "project_name": project_name,
            "warmup_steps": warmup_steps
        }

        final_config = generate_case_config(base_template, run_params, physical_constants)
        config_filename = f"{filename_stem}_Re{re_val}.yaml"
        full_config_path = os.path.join(output_dir, config_filename)

        with open(full_config_path, "w") as f:
            yaml.dump(final_config, f, sort_keys=False, default_flow_style=None)

        print(f"  -> Config saved successfully.")
        success_count += 1

    print(f"\n[Done] Successfully generated {success_count} of {len(mask_files)} configs.")

if __name__ == "__main__":
    main()
