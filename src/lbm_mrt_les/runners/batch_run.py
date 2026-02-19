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

    # 4. Main execution loop
    all_cases_summary = []
    for i, cfg_file in enumerate(config_files):
        job_id = i + 1
        full_config_path = os.path.join(project_paths["configs"], cfg_file)
        
        print(f"--- Running Job {job_id}/{len(config_files)}: {cfg_file} ---")
        gc.collect()

        summary_entry = case_executor.execute_case(full_config_path, project_paths, output_dirs, job_id)
        all_cases_summary.append(summary_entry)

    # 5. Save the final summary file
    summary_path = os.path.join(output_dirs["plots"], "all_cases_summary.json")
    batch_io.save_summary_file(all_cases_summary, summary_path)

if __name__ == "__main__":
    main()

