import os
from typing import Dict

from .. import utils
from ..utils import physics_scaling
from . import summary_builder
from .run_one_case import main as run_one_case_main

def execute_case(full_config_path: str, project_paths: Dict, output_dirs: Dict, job_id: int) -> Dict:
    """
    Executes a single simulation case and returns its summary dictionary.
    
    This function orchestrates loading the config, constructing paths, running the simulation,
    calculating physical parameters, and building the summary entry.
    """
    try:
        config = utils.load_config(full_config_path)
        
        # --- 1. Extract info and construct paths ---
        mask_path_from_cfg = config.get("mask", {}).get("path")
        sim_name = config.get("simulation", {}).get("name")
        target_re = config.get("outputs", {}).get("target_re")
        
        mask_path = os.path.join(project_paths["masks"], os.path.basename(mask_path_from_cfg))
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        base_filename = f"Case{job_id:02d}_{sim_name}_Re{target_re}"
        h5_path = os.path.join(output_dirs["raw"], f"{base_filename}.h5")
        video_path = os.path.join(output_dirs["vis"], f"{base_filename}.mp4")

        # --- 2. Run the core simulation ---
        lattice_metadata = run_one_case_main(full_config_path, mask_path, h5_path, video_path)
        
        if lattice_metadata.get("status") != "Success":
            raise RuntimeError(f"Simulation failed: {lattice_metadata.get('reason')}")

        # --- 3. Perform post-processing and data structuring ---
        physical_params = physics_scaling.calculate_physical_params(config, lattice_metadata)
        
        source_files = {
            "config_file": os.path.basename(full_config_path),
            "mask_file": os.path.basename(mask_path)
        }
        
        summary_entry = summary_builder.build_summary_entry(
            config, lattice_metadata, physical_params, source_files
        )
        
        print(f"  [Success] Finished case {sim_name}.")
        return summary_entry

    except Exception as e:
        print(f"  [Error] Case failed: {e}")
        # Return a failure summary entry
        return {
            "case_name": os.path.basename(full_config_path), 
            "status": "Failed", 
            "reason": str(e)
        }
