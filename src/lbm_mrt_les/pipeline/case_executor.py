"""
src/lbm_mrt_les/pipeline/case_executor.py

Orchestrates a single simulation case:
  1. Resolves paths
  2. Runs the simulation (run_one_case.main)
  3. Calculates physical scaling parameters
  4. Builds the structured summary entry

On failure: cleans up any partial output files in raw/ and vis/ to save disk
space, then returns a failure summary dict.
"""

import os
import glob
from typing import Dict

from .. import utils
from ..utils import physics_scaling
from . import summary_builder
from .run_one_case import main as run_one_case_main


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup helper
# ─────────────────────────────────────────────────────────────────────────────


def _cleanup_failed_outputs(h5_path: str, video_path: str) -> None:
    """
    Removes the .h5 and .mp4 files produced by a failed simulation run so
    that partial / corrupt data does not accumulate on disk.

    Glob patterns are used so that any temp-suffixed variants (e.g. .h5.tmp)
    are also caught.
    """
    targets = [h5_path, video_path]

    for path in targets:
        if not path:
            continue

        # Match the exact file plus any partial-write variants (*.tmp, *.part)
        candidates = [path] + glob.glob(path + ".*")

        for fpath in candidates:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    print(f"  [Cleanup] Removed failed output: {fpath}")
                except OSError as e:
                    print(f"  [Cleanup] Warning – could not remove {fpath}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main executor
# ─────────────────────────────────────────────────────────────────────────────


def execute_case(
    full_config_path: str,
    project_paths: Dict,
    output_dirs: Dict,
    job_id: int,
) -> Dict:
    """
    Execute a single simulation case and return its summary dictionary.

    Returns a Success or Failed summary dict; never raises.
    """
    # Initialise path variables at the top-level scope so the except block can
    # reference them for cleanup even if the error occurred mid-construction.
    h5_path = ""
    video_path = ""
    sim_name = os.path.basename(full_config_path)  # fallback display name

    try:
        config = utils.load_config(full_config_path)

        # ── 1. Extract info and construct output paths ────────────────────
        mask_path_from_cfg = config.get("mask", {}).get("path")
        sim_name = config.get("simulation", {}).get("name", sim_name)

        mask_path = os.path.join(
            project_paths["masks"], os.path.basename(mask_path_from_cfg)
        )
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        h5_path = os.path.join(output_dirs["raw"], f"{sim_name}.h5")
        video_path = os.path.join(output_dirs["vis"], f"{sim_name}.mp4")

        # ── 2. Run the core simulation ────────────────────────────────────
        lattice_metadata = run_one_case_main(
            full_config_path, mask_path, h5_path, video_path
        )

        if lattice_metadata.get("status") != "Success":
            raise RuntimeError(f"Simulation failed: {lattice_metadata.get('reason')}")

        # ── 3. Physical scaling & summary ────────────────────────────────
        physical_params = physics_scaling.calculate_physical_params(
            config, lattice_metadata
        )

        source_files = {
            "config_file": os.path.basename(full_config_path),
            "mask_file": os.path.basename(mask_path),
        }

        summary_entry = summary_builder.build_summary_entry(
            config, lattice_metadata, physical_params, source_files
        )

        print(f"  [Success] Finished case {sim_name}.")
        return summary_entry

    except Exception as e:
        print(f"  [Error] Case '{sim_name}' failed: {e}")

        # ── Cleanup: remove partial output files to save disk space ──────
        if h5_path or video_path:
            print(f"  [Cleanup] Removing partial outputs for failed case '{sim_name}'…")
            _cleanup_failed_outputs(h5_path, video_path)

        return {
            "case_name": sim_name,
            "status": "Failed",
            "reason": str(e),
        }
