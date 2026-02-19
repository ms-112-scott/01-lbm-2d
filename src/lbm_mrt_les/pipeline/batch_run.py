import argparse
import os
import sys
import gc
from typing import List, Dict

from . import paths
from . import case_executor
from ..io import batch_io

def find_config_files(config_dir: str) -> List[str]:
    """Finds and sorts all YAML configuration files in a directory."""
    if not os.path.isdir(config_dir):
        print(f"[Error] Config directory not found: {config_dir}")
        sys.exit(1)
        
    config_files = sorted([f for f in os.listdir(config_dir) if f.endswith(".yaml")])
    
    if not config_files:
        print(f"[Error] No YAML config files found in {config_dir}")
        sys.exit(1)
        
    return config_files

def main():
    parser = argparse.ArgumentParser(description="Multi-case batch runner for LBM simulations.")
    parser.add_argument("--project_name", type=str, required=True, help="The project name to run.")
    args = parser.parse_args()

    # 1. Get all project-related paths
    project_paths = paths.get_project_paths(args.project_name)
    
    # 2. Find configuration files
    config_files = find_config_files(project_paths["configs"])
    print(f"Found {len(config_files)} cases for project '{args.project_name}'.")

    # 3. Set up the output directory structure
    output_dirs = paths.setup_output_directories(project_paths["outputs"])
    summary_path = os.path.join(output_dirs["plots"], "all_cases_summary.json")
    batch_io.init_summary_file(summary_path)

    # 4. Main execution loop
    for i, cfg_file in enumerate(config_files):
        job_id = i + 1
        full_config_path = os.path.join(project_paths["configs"], cfg_file)
        
        print(f"--- Running Job {job_id}/{len(config_files)}: {cfg_file} ---")
        
        # Load config early to get pre-calculable info
        try:
            config = case_executor.utils.load_config(full_config_path)
            sim_name = config.get("simulation", {}).get("name", cfg_file)
            target_re = config.get("outputs", {}).get("target_re", "Unknown")
            nx = config.get("simulation", {}).get("nx")
            ny = config.get("simulation", {}).get("ny")
            
            # Pre-write "Running" status with pre-calculable info
            pre_summary = {
                "case_name": sim_name,
                "status": "Running",
                "job_id": job_id,
                "parameters": {
                    "lattice": {
                        "target_re": target_re,
                        "resolution_px": [nx, ny]
                    }
                },
                "source_files": {
                    "config_file": cfg_file,
                    "mask_file": os.path.basename(config.get("mask", {}).get("path", "N/A"))
                }
            }
            batch_io.update_summary_file(pre_summary, summary_path)
        except Exception as e:
            print(f"  [Warning] Could not pre-calculate info for {cfg_file}: {e}")

        gc.collect()

        summary_entry = case_executor.execute_case(full_config_path, project_paths, output_dirs, job_id)
        
        # Update with final result
        batch_io.update_summary_file(summary_entry, summary_path)

    print(f"\n[Finished] All cases processed. Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
