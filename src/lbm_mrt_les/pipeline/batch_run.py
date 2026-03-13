"""
src/lbm_mrt_les/pipeline/batch_run.py

Entry point for multi-case batch simulation.

Usage:
    python -m src.lbm_mrt_les.pipeline.batch_run --project_name <n>
    python -m src.lbm_mrt_les.pipeline.batch_run --project_name <n> --max_success 20

Output files written to outputs/<project_name>/plots/:
    all_cases_summary.json   – full structured summary (updated after every case)
    all_cases_vectors.npz    – flat numeric feature matrix for ML downstream use

Resume / skip logic:
    - 若 all_cases_summary.json 已存在，啟動時讀取它
    - case_name 已有 "Success" 或 "Failed" → 直接跳過（不重跑）
    - case_name 狀態為 "Running" → 視為上次中斷，重新執行
    - case_name 不在 summary → 正常執行
    - 已完成的 Success 數量從 summary 讀取，計入 max_success 統計
"""

import argparse
import os
import sys
import gc
import json
from typing import List, Dict, Set, Tuple

from . import paths
from . import case_executor
from ..io import batch_io
from ..io.case_vector_builder import build_npz


# ─────────────────────────────────────────────────────────────────────────────
# Resume helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_existing_summary(summary_path: str) -> Dict[str, str]:
    """
    讀取現有 summary JSON，回傳 {case_name: status} 的 dict。
    若 JSON 不存在或損毀則回傳空 dict。
    """
    if not os.path.exists(summary_path):
        return {}
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        result = {}
        for entry in entries:
            name = entry.get("case_name")
            status = entry.get("status", "Unknown")
            if name:
                result[name] = status
        return result
    except Exception as e:
        print(f"[Warning] Could not read existing summary ({e}). Starting fresh.")
        return {}


def _get_sim_name_from_config(config_path: str) -> str:
    """
    從 config YAML 快速讀取 simulation.name。
    失敗時回傳空字串。
    """
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("simulation", {}).get("name", "")
    except Exception:
        return ""


def _print_resume_plan(
    config_files: List[str],
    existing: Dict[str, str],
    config_dir: str,
) -> Tuple[int, Set[str]]:
    """
    掃描所有 config，印出 resume 計畫。
    回傳 (already_success_count, skip_set)。
    skip_set 包含所有「已完成（Success 或 Failed）」的 case_name，不會重跑。
    """
    skip_names: Set[str] = set()
    already_success = 0

    if not existing:
        print("[Resume] No existing summary found. Starting from scratch.")
        return 0, skip_names

    print("[Resume] Existing summary found. Scanning for completed cases...")
    for cfg_file in config_files:
        full_path = os.path.join(config_dir, cfg_file)
        sim_name = _get_sim_name_from_config(full_path)
        if not sim_name:
            continue
        status = existing.get(sim_name)
        if status == "Success":
            skip_names.add(sim_name)
            already_success += 1
            print(f"  [Skip ✅] {sim_name}  ({cfg_file})  → already succeeded")
        elif status == "Failed":
            skip_names.add(sim_name)
            print(
                f"  [Skip ❌] {sim_name}  ({cfg_file})  → already failed, will not retry"
            )
        elif status == "Running":
            print(
                f"  [Retry ↩] {sim_name}  ({cfg_file})  → interrupted last time, will re-run"
            )
        # status is None (not in summary) → will run normally, no print needed

    print(
        f"\n[Resume] {len(skip_names)} case(s) will be skipped "
        f"({already_success} previously successful).\n"
    )
    return already_success, skip_names


# ─────────────────────────────────────────────────────────────────────────────
# Config discovery
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Multi-case batch runner for LBM simulations."
    )
    parser.add_argument(
        "--project_name", type=str, required=True, help="The project name to run."
    )
    parser.add_argument(
        "--max_success",
        type=int,
        default=None,
        help="Stop after this many TOTAL successful cases (including previous runs). "
        "Failed cases do not count. Omit to run all configs.",
    )
    args = parser.parse_args()

    max_success = args.max_success

    # -- 1. Resolve paths ---------------------------------------------------
    project_paths = paths.get_project_paths(args.project_name)

    # -- 2. Find configuration files ----------------------------------------
    config_files = find_config_files(project_paths["configs"])
    print(f"Found {len(config_files)} configs for project '{args.project_name}'.")

    # -- 3. Set up output directory structure --------------------------------
    output_dirs = paths.setup_output_directories(project_paths["outputs"])
    summary_path = os.path.join(output_dirs["plots"], "all_cases_summary.json")
    npz_path = os.path.join(output_dirs["plots"], "all_cases_vectors.npz")

    # -- 4. Resume: read existing summary and determine skip set -------------
    existing_summary = _load_existing_summary(summary_path)
    already_success, skip_names = _print_resume_plan(
        config_files, existing_summary, project_paths["configs"]
    )

    # 若 summary 不存在才初始化（避免覆蓋既有資料）
    if not os.path.exists(summary_path):
        batch_io.init_summary_file(summary_path)

    # 檢查 max_success 是否已達成
    if max_success is not None:
        remaining_needed = max_success - already_success
        if remaining_needed <= 0:
            print(
                f"[Batch] Already reached max_success={max_success} "
                f"from previous runs ({already_success} successful). Nothing to do."
            )
            return
        print(
            f"[Batch] max_success={max_success}  "
            f"(already={already_success}, still need={remaining_needed} more)"
        )

    # -- 5. Main execution loop ----------------------------------------------
    new_success_count = 0  # 本次 session 新增的成功數
    skipped_count = 0  # 因已完成而跳過的數量

    for i, cfg_file in enumerate(config_files):
        job_id = i + 1
        full_config_path = os.path.join(project_paths["configs"], cfg_file)

        # 快速取得 sim_name 用於 skip 判斷
        sim_name_quick = _get_sim_name_from_config(full_config_path)

        # ── Skip 判斷（Success 或 Failed 都跳過）───────────────────────
        if sim_name_quick and sim_name_quick in skip_names:
            status_str = existing_summary.get(sim_name_quick, "?")
            icon = "✅" if status_str == "Success" else "❌"
            print(
                f"--- [{icon} Skip {job_id}/{len(config_files)}] "
                f"{cfg_file}  ({status_str})"
            )
            skipped_count += 1
            continue

        # ── max_success 達標提前結束 ─────────────────────────────────────
        total_success_so_far = already_success + new_success_count
        if max_success is not None and total_success_so_far >= max_success:
            remaining = len(config_files) - i
            print(
                f"\n[Batch] Reached max_success={max_success} "
                f"(previous={already_success}, new this session={new_success_count}). "
                f"Stopping. {remaining} config(s) not run."
            )
            break

        # ── 執行 ──────────────────────────────────────────────────────────
        total_success_so_far = already_success + new_success_count
        progress = (
            f"[success: {total_success_so_far}/{max_success}]"
            if max_success is not None
            else f"[success so far: {total_success_so_far}]"
        )
        print(f"\n--- Running Job {job_id}/{len(config_files)}: {cfg_file} {progress}")
        gc.collect()

        # Pre-write "Running" status（crash-safe：重啟時可識別中斷點）
        try:
            config = case_executor.utils.load_config(full_config_path)
            sim_name = config.get("simulation", {}).get("name", cfg_file)
            nx = config.get("simulation", {}).get("nx")
            ny = config.get("simulation", {}).get("ny")

            pre_summary = {
                "case_name": sim_name,
                "status": "Running",
                "job_id": job_id,
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

        # Run the case
        summary_entry = case_executor.execute_case(
            full_config_path, project_paths, output_dirs, job_id
        )

        # Persist result immediately（每個 case 完成都立刻寫入，崩潰安全）
        batch_io.update_summary_file(summary_entry, summary_path)

        if summary_entry["status"] == "Success":
            new_success_count += 1

    # -- 6. Post-batch summary -----------------------------------------------
    total_success = already_success + new_success_count

    print(f"\n{'='*60}")
    print(f"[Batch] Session complete.")
    print(f"  Previously successful : {already_success}")
    print(f"  New this session      : {new_success_count}")
    print(f"  Total successful      : {total_success}")
    print(f"  Skipped (done)        : {skipped_count}")
    print(f"{'='*60}")

    print(f"\n[Batch] Building ML feature vectors from summary ...")
    try:
        build_npz(summary_path, npz_path)
    except Exception as e:
        print(f"[Warning] NPZ build failed (summary JSON still valid): {e}")

    print(f"\n[Finished] All cases processed.")
    print(f"  Summary  -> {summary_path}")
    print(f"  Vectors  -> {npz_path}")


if __name__ == "__main__":
    main()
