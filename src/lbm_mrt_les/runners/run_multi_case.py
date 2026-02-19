import argparse
import os
import sys
import yaml
import gc
import json
import traceback
from typing import List, Dict, Any

from .. import utils
from .run_one_case import main as run_one_case_main

def setup_directories(base_path: str) -> Dict[str, str]:
    """Creates the standardized directory structure for a project."""
    paths = {
        "base": base_path,
        "raw": os.path.join(base_path, "raw"),
        "vis": os.path.join(base_path, "vis"),
        "plots": os.path.join(base_path, "plots"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def run_batch_simulation():
    parser = argparse.ArgumentParser(description="Multi-case batch runner for LBM simulations.")
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="The project name (e.g., 'Hyper') to find configs and store outputs.",
    )
    args = parser.parse_args()

    # MODIFIED: All paths are now derived from the project_name within the 'SimCases' directory
    project_base_dir = os.path.join("SimCases", args.project_name)
    config_dir = os.path.join(project_base_dir, "configs")
    mask_dir = os.path.join(project_base_dir, "masks") # New: define mask_dir here

    if not os.path.isdir(config_dir):
        print(f"[Error] Config directory not found for project '{args.project_name}': {config_dir}")
        print("Please ensure configs are generated and located at this path.")
        sys.exit(1)

    # It's good practice to also check for mask_dir if we're expecting masks there
    if not os.path.isdir(mask_dir):
        print(f"[Error] Mask directory not found for project '{args.project_name}': {mask_dir}")
        print("Please ensure masks are generated and located at this path.")
        sys.exit(1)

    config_files = sorted([f for f in os.listdir(config_dir) if f.endswith(".yaml")])
    if not config_files:
        print(f"[Error] No YAML config files found in {config_dir}")
        sys.exit(1)

    print(f"Found {len(config_files)} simulation cases for project '{args.project_name}'.")
    print("=" * 60)

    all_cases_metadata = []
    # The output structure is still within 'outputs' as previously defined
    output_project_base_path = os.path.join("outputs", args.project_name)
    dir_paths = setup_directories(output_project_base_path)

    for i, cfg_file in enumerate(config_files):
        job_id = i + 1
        full_config_path = os.path.join(config_dir, cfg_file)

        print(f"\n[Job {job_id}/{len(config_files)}]")
        print(f"    Config: {cfg_file}")
        print("-" * 40)

        gc.collect()

        try:
            config = utils.load_config(full_config_path)
            outputs_cfg = config.get("outputs", {})
            sim_cfg = config.get("simulation", {})
            mask_cfg = config.get("mask", {})

            sim_name = sim_cfg.get("name", f"Case_{job_id:04d}")
            target_re = outputs_cfg.get("target_re", 0)
            
            # MODIFIED: Get mask filename from the config and construct full path with the new mask_dir
            mask_filename = os.path.basename(mask_cfg.get("path", "")) # config_batch_gen now writes the full path, but we need to reconstruct for the new structure
            if not mask_filename:
                 print(f"   >>> \033[91m[Error] Mask filename not found in config for {cfg_file}\033[0m")
                 continue
            mask_path = os.path.join(mask_dir, mask_filename)

            if not os.path.exists(mask_path):
                 print(f"   >>> \033[91m[Error] Mask file not found: {mask_path}\033[0m")
                 continue

            base_filename = f"{sim_name}_Re{target_re}"
            h5_path = os.path.join(dir_paths["raw"], f"{base_filename}.h5")
            video_path = os.path.join(dir_paths["vis"], f"{base_filename}.mp4")

            case_metadata = run_one_case_main(full_config_path, mask_path, h5_path, video_path)
            
            case_metadata["case_name"] = sim_name
            case_metadata["config_file"] = cfg_file
            case_metadata["mask_file"] = os.path.basename(mask_path)
            all_cases_metadata.append(case_metadata)

            if case_metadata.get("status") == "Success":
                 print(f"   >>> \032[92mSuccess\032[0m.")
            else:
                 print(f"   >>> \032[91m[Error] Failed: {case_metadata.get('reason')}\032[0m")

        except KeyboardInterrupt:
            print("\n[User Abort] Stopping batch run...")
            break
        except Exception as e:
            print(f"   >>> \032[91m[Critical Error] Job failed: {e}\032[0m")
            traceback.print_exc()
            all_cases_metadata.append({"case_name": cfg_file, "status": "Failed", "reason": str(e)})
            continue

    summary_path = os.path.join(dir_paths["plots"], "all_cases_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_cases_metadata, f, indent=4)
        print(f"\n[Done] Saved batch summary to: {summary_path}")
    except Exception as e:
        print(f"\n[Error] Could not save summary file: {e}")

    print("\n" + "=" * 60)
    print("All batch jobs finished.")

if __name__ == "__main__":
    run_batch_simulation()
