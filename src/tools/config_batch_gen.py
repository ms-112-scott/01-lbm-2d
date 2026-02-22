import yaml
import os
import sys
import re
import glob
import copy
import argparse
import random
import math
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
    rho_in = run_params["rho_in"]
    rho_out = run_params["rho_out"]
    interval = run_params["interval"]
    mask_path = run_params["mask_path"]
    data_save_root = run_params["data_save_root"]
    project_name = run_params["project_name"]
    warmup_steps = run_params["warmup_steps"]

    config["simulation"]["name"] = sim_name
    config["simulation"]["nu"] = float(f"{nu_value:.6f}")
    config["simulation"]["characteristic_length"] = float(l_char)
    config["simulation"]["rho_in"] = float(rho_in)
    config["simulation"]["rho_out"] = float(rho_out)
    config["simulation"]["compute_step_size"] = interval
    config["simulation"]["warmup_steps"] = warmup_steps  # Set dynamic warmup steps

    # Fixed smagorinsky constant for all runs, or you can add logic if needed
    config["simulation"]["smagorinsky_constant"] = 0.2

    config["outputs"]["project_name"] = project_name
    config["outputs"]["data_save_root"] = data_save_root
    
    # Optional target parameter, mostly for tracking/naming in output
    config["outputs"]["target_rho_in"] = float(rho_in)
    
    config["outputs"]["gui"]["interval_steps"] = interval
    config["outputs"]["video"]["interval_steps"] = interval
    config["outputs"]["video"]["filename"] = f"{sim_name}.mp4"
    config["outputs"]["dataset"]["interval_steps"] = interval

    if "folder" in config["outputs"]["dataset"]:
        del config["outputs"]["dataset"]["folder"]

    # Dummy boundary values since Zou-He boundaries use density
    config["boundary_condition"]["value"] = [[0.05, 0.0]] + [[0.0, 0.0]] * 3
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
    rho_in_list = physics["rho_in_list"]

    project_name = settings["project_name"]
    project_dir = os.path.join("SimCases", project_name)
    mask_dir = os.path.join(project_dir, "masks")
    output_dir = os.path.join(project_dir, "configs")
    data_save_root = os.path.join("outputs", project_name) 

    base_nu = physics["nu"]
    base_rho_out = physics["rho_out"]
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
        
        target_rho_in = get_sampled_value(rho_in_list)
        if target_rho_in is None:
            print(f"  [Warning] Could not sample a valid rho_in value from 'rho_in_list' in config. Skipping.")
            continue

        print(f"Generating config for {filename_stem} (Target rho_in: {target_rho_in})...")
        
        match_l = re.search(r"L(\d+)", filename_stem)
        if not match_l:
            print(f"  [Warning] Could not parse L from filename, skipping.")
            continue
        
        l_char = float(match_l.group(1))

        # Check tau stability
        tau = 3.0 * base_nu + 0.5
        if tau < 0.505:
            print(f"  [Error] Cannot generate stable config with nu={base_nu}. Tau ({tau:.4f}) is too low. Skipping.")
            continue

        # Approximate u to determine warmup steps and physical seconds
        # u ~ sqrt(2/3 * (rho_in - rho_out)) for small density diff
        estimated_u = math.sqrt((2.0 / 3.0) * (target_rho_in - base_rho_out)) if target_rho_in > base_rho_out else 0.01

        nx_val = base_template["simulation"]["nx"]
        warmup_steps = int(nx_val / estimated_u) if estimated_u > 0 else 81920
        
        steps_per_phys_sec = l_char / estimated_u if estimated_u > 0 else 0
        target_interval = max(1, int(steps_per_phys_sec / saves_per_phys_sec)) if steps_per_phys_sec > 0 else 160

        run_params = {
            "sim_name": filename_stem,
            "nu": base_nu, "l_char": l_char, 
            "rho_in": target_rho_in, "rho_out": base_rho_out,
            "interval": target_interval, "mask_path": mask_path,
            "data_save_root": data_save_root, "project_name": project_name,
            "warmup_steps": warmup_steps
        }

        final_config = generate_case_config(base_template, run_params, physical_constants)
        
        # Format rho_in for filename (e.g. 1.005 -> 1_005)
        rho_in_str = f"{target_rho_in:.3f}".replace('.', '-')
        config_filename = f"{filename_stem}_Rho{rho_in_str}.yaml"
        full_config_path = os.path.join(output_dir, config_filename)

        with open(full_config_path, "w") as f:
            yaml.dump(final_config, f, sort_keys=False, default_flow_style=None)

        print(f"  -> Config saved successfully.")
        success_count += 1

    print(f"\n[Done] Successfully generated {success_count} of {len(mask_files)} configs.")

if __name__ == "__main__":
    main()
