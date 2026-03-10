"""
src/lbm_mrt_les/pipeline/batch_run.py

Entry point for multi-case batch simulation.

Usage:
    python -m src.lbm_mrt_les.pipeline.batch_run --project_name <name>

Output files written to outputs/<project_name>/plots/:
    all_cases_summary.json   – full structured summary (updated after every case)
    all_cases_vectors.npz    – flat numeric feature matrix for ML downstream use
"""

import argparse
import os
import sys
import gc
from typing import List, Dict

from . import paths
from . import case_executor
from ..io import batch_io
from ..io.case_vector_builder import build_npz


def find_config_files(config_dir: str) -> List[str]:
    """Finds and sorts all YAML configuration files in a directory."""
    if not os.path.isdir(config_dir):
        print(f"[Error] Config directory not found: {config_dir}")
        sys.exit(1)

    config_files = sorted(
        [f for f in os.listdir(config_dir) if f.endswith(".yaml")]
    )

    if not config_files:
        print(f"[Error] No YAML config files found in {config_dir}")
        sys.exit(1)

    return config_files


def main():
    parser = argparse.ArgumentParser(
        description="Multi-case batch runner for LBM simulations."
    )
    parser.add_argument(
        "--project_name", type=str, required=True, help="The project name to run."
    )
    args = parser.parse_args()

    # ── 1. Resolve paths ────────────────────────────────────────────────────
    project_paths = paths.get_project_paths(args.project_name)

    # ── 2. Find configuration files ─────────────────────────────────────────
    config_files = find_config_files(project_paths["configs"])
    print(f"Found {len(config_files)} cases for project '{args.project_name}'.")

    # ── 3. Set up output directory structure ────────────────────────────────
    output_dirs  = paths.setup_output_directories(project_paths["outputs"])
    summary_path = os.path.join(output_dirs["plots"], "all_cases_summary.json")
    npz_path     = os.path.join(output_dirs["plots"], "all_cases_vectors.npz")
    batch_io.init_summary_file(summary_path)

    # ── 4. Main execution loop ───────────────────────────────────────────────
    success_count = 0

    for i, cfg_file in enumerate(config_files):
        job_id = i + 1
        full_config_path = os.path.join(project_paths["configs"], cfg_file)

        print(f"\n--- Running Job {job_id}/{len(config_files)}: {cfg_file} ---")
        gc.collect()

        # Pre-write "Running" status so the JSON is never stale mid-batch
        try:
            config   = case_executor.utils.load_config(full_config_path)
            sim_name = config.get("simulation", {}).get("name", cfg_file)
            nx       = config.get("simulation", {}).get("nx")
            ny       = config.get("simulation", {}).get("ny")

            pre_summary = {
                "case_name": sim_name,
                "status":    "Running",
                "job_id":    job_id,
                "parameters": {"lattice": {"resolution_px": [nx, ny]}},
                "source_files": {
                    "config_file": cfg_file,
                    "mask_file": os.path.basename(
                        config.get("mask", {}).get("path", "N/A")
                    ),
                },
            }
            batch_io.update_summary_file(pre_summary, summary_path)
        except Exception as e:
            print(f"  [Warning] Could not pre-write status for {cfg_file}: {e}")

        gc.collect()

        # Run the case (cleanup of failed outputs happens inside execute_case)
        summary_entry = case_executor.execute_case(
            full_config_path, project_paths, output_dirs, job_id
        )

        # Persist the result immediately (crash-safe: every case is flushed)
        batch_io.update_summary_file(summary_entry, summary_path)

        if summary_entry["status"] == "Success":
            success_count += 1

    # ── 5. Post-batch: build NPZ feature matrix ──────────────────────────────
    print(f"\n[Batch] {success_count}/{len(config_files)} cases succeeded.")
    print(f"[Batch] Building ML feature vectors from summary …")

    try:
        build_npz(summary_path, npz_path)
    except Exception as e:
        print(f"[Warning] NPZ build failed (summary JSON still valid): {e}")

    print(f"\n[Finished] All cases processed.")
    print(f"  Summary  → {summary_path}")
    print(f"  Vectors  → {npz_path}")


if __name__ == "__main__":
    main()
