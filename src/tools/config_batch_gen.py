import yaml
import os
import sys
import re
import glob
import copy
import argparse

def load_yaml(path):
    """Loads a YAML configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_case_config(base_template, run_params):
    """Generates a single case configuration."""
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

    config["outputs"]["project_name"] = project_name
    config["outputs"]["data_save_root"] = data_save_root
    config["outputs"]["target_re"] = int(re_value)
    
    config["outputs"]["gui"]["interval_steps"] = interval
    config["outputs"]["video"]["interval_steps"] = interval
    config["outputs"]["video"]["filename"] = f"{sim_name}.mp4"
    config["outputs"]["dataset"]["interval_steps"] = interval

    if "folder" in config["outputs"]["dataset"]:
        del config["outputs"]["dataset"]["folder"]

    config["boundary_condition"]["value"] = [[float(u_lb), 0.0]] + [[0.0, 0.0]] * 3
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

    project_name = settings["project_name"]
    
    # MODIFIED: Paths are now inside the project directory within 'SimCases'
    project_dir = os.path.join("SimCases", project_name)
    mask_dir = os.path.join(project_dir, "masks")
    output_dir = os.path.join(project_dir, "configs")
    # data_save_root remains pointing to the outputs folder, as outputs go there
    data_save_root = os.path.join("outputs", project_name) 

    re_list = physics["re_list"]
    u_lb = physics["u_lb"]
    saves_per_phys_sec = physics["saves_per_physical_second"]

    os.makedirs(output_dir, exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not mask_files:
        print(f"[Error] No PNG files found in the project mask directory: {mask_dir}")
        print("Please generate masks first using hybrid_map_gen.py.")
        return

    print(f"--- Found {len(mask_files)} maps in {mask_dir}. Outputting configs to: {output_dir} ---")

    for i, mask_path in enumerate(mask_files):
        filename_stem = os.path.splitext(os.path.basename(mask_path))[0]
        
        match_re = re.search(r"Re(\d+)", filename_stem)
        match_l = re.search(r"_L(\d+)", filename_stem)

        if not match_re or not match_l:
            print(f"[Warning] Could not parse Re and L from filename: {filename_stem}, skipping.")
            continue
        
        re_val = int(match_re.group(1))
        l_char = float(match_l.group(1))

        nu = (u_lb * l_char) / re_val
        steps_per_phys_sec = l_char / u_lb
        target_interval = max(1, int(steps_per_phys_sec / saves_per_phys_sec))

        run_params = {
            "sim_name": f"Case{i:04d}_{filename_stem}",
            "nu": nu, "l_char": l_char, "re": re_val, "u_lb": u_lb,
            "interval": target_interval, "mask_path": mask_path,
            "data_save_root": data_save_root, "project_name": project_name,
        }

        final_config = generate_case_config(base_template, run_params)
        
        config_filename = f"cfg_{filename_stem}.yaml"
        full_config_path = os.path.join(output_dir, config_filename)

        with open(full_config_path, "w") as f:
            yaml.dump(final_config, f, sort_keys=False, default_flow_style=None)

        print(f"[{i+1:03d}] {filename_stem} -> Re={re_val} | Saved: {config_filename}")

    print(f"\n[Done] Generated {len(mask_files)} configs.")

if __name__ == "__main__":
    main()
