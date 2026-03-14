"""
src/lbm_mrt_les/pipeline/batch_run.py

Entry point for multi-case batch simulation.

Usage:
    python -m src.lbm_mrt_les.pipeline.batch_run --project_name <n>
    python -m src.lbm_mrt_les.pipeline.batch_run --project_name <n> --max_success 20

輸出檔案：
    outputs/{project}/plots/all_cases_summary.json  原有 summary（保留相容性）
    outputs/{project}/plots/all_cases_vectors.npz   ML 特徵矩陣
    outputs/{project}/plots/sim_results.json         新版三層尺度完整結果

Resume / skip 邏輯：
    - 唯一識別鍵：config 檔名（config_filename），不再用 case_name
      原因：同一 mask 現在有多個 nu → 多個 config → 多個 case，case_name 不唯一
    - sim_results.json 中 status="Success" 或 "Failed" → 跳過
    - status="Running" → 上次中斷，重新執行
    - 不在 sim_results.json → 正常執行

config_meta.json 整合：
    - 啟動時從 SimCases/{project}/config_meta.json 讀入預算參數
    - 若 sim_results.json 不存在，自動以 config_meta 初始化（複製結構）
    - 模擬完成後只填入 simulation_outputs + wall_time_s，不重算物理尺度
"""

import argparse
import os
import sys
import gc
import json
import time
from typing import Dict, List, Set, Tuple

from . import paths
from . import case_executor
from ..io import batch_io
from ..io import sim_results_io
from ..io.case_vector_builder import build_npz


# ─────────────────────────────────────────────────────────────────────────────
# region Config 檔掃描
# ─────────────────────────────────────────────────────────────────────────────


def find_config_files(config_dir: str) -> List[str]:
    """找出並排序 config 目錄中所有 YAML 檔名（不含路徑）。"""
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


def _read_sim_name_from_config(config_path: str) -> str:
    """快速從 YAML 讀取 simulation.name；失敗時回傳空字串。"""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("simulation", {}).get("name", "")
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# region Resume helpers（以 config_filename 為鍵）
# ─────────────────────────────────────────────────────────────────────────────


def _build_resume_plan(
    config_files: List[str],
    status_map: Dict[str, str],
) -> Tuple[int, Set[str]]:
    """
    根據 sim_results.json 的 status_map 決定哪些 config 跳過。

    Args:
        config_files: YAML 檔名清單（不含路徑）
        status_map:   {config_filename: status}，來自 sim_results_io.get_status_map()

    Returns:
        (already_success_count, skip_filenames)
        skip_filenames: Success 或 Failed 的 config 檔名集合
    """
    if not status_map:
        print("[Resume] 無現有 sim_results 記錄，從頭開始。")
        return 0, set()

    print("[Resume] 掃描已完成的 case...")
    skip_set: Set[str] = set()
    already_success = 0

    for cfg_file in config_files:
        status = status_map.get(cfg_file)
        if status == "Success":
            skip_set.add(cfg_file)
            already_success += 1
            print(f"  [Skip OK ] {cfg_file}")
        elif status == "Failed":
            skip_set.add(cfg_file)
            print(f"  [Skip ERR] {cfg_file}  （上次失敗，不重試）")
        elif status == "Running":
            print(f"  [Retry   ] {cfg_file}  （上次中斷，重新執行）")

    print(
        f"\n[Resume] 跳過 {len(skip_set)} 個（其中 {already_success} 個已成功）。\n"
    )
    return already_success, skip_set


# ─────────────────────────────────────────────────────────────────────────────
# region 舊版 all_cases_summary 相容層（保留 NPZ builder 可用）
# ─────────────────────────────────────────────────────────────────────────────


def _load_legacy_summary(summary_path: str) -> Dict[str, str]:
    """讀取舊版 all_cases_summary.json → {case_name: status}（相容性用）。"""
    if not os.path.exists(summary_path):
        return {}
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        return {
            e["case_name"]: e.get("status", "Unknown")
            for e in entries
            if "case_name" in e
        }
    except Exception as e:
        print(f"[Warning] 讀取舊版 summary 失敗：{e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# region Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-case batch runner for LBM simulations."
    )
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="模擬專案名稱（對應 SimCases/{name} 目錄）",
    )
    parser.add_argument(
        "--max_success",
        type=int,
        default=None,
        help="達到此成功數後停止（含先前 session）；省略則跑完全部 config",
    )
    args = parser.parse_args()

    # ── 1. 路徑初始化 ────────────────────────────────────────────────────────
    project_paths = paths.get_project_paths(args.project_name)
    output_dirs = paths.setup_output_directories(project_paths["outputs"])

    config_meta_path = os.path.join(
        "SimCases", args.project_name, "config_meta.json"
    )
    sim_results_path = os.path.join(output_dirs["plots"], "sim_results.json")
    legacy_summary_path = os.path.join(output_dirs["plots"], "all_cases_summary.json")
    npz_path = os.path.join(output_dirs["plots"], "all_cases_vectors.npz")

    # ── 2. 載入 config_meta 並初始化 sim_results ─────────────────────────────
    if not os.path.exists(config_meta_path):
        print(
            f"[Warning] config_meta.json 不存在：{config_meta_path}\n"
            f"  請先執行 config_batch_gen_all_nu.py 生成預算參數。\n"
            f"  本次仍可執行，但 sim_results.json 的 Tier 1/2/3 欄位將為空。"
        )
        config_meta: dict[str, dict] = {}
    else:
        config_meta = sim_results_io.load_config_meta(config_meta_path)
        sim_results_io.init_sim_results(config_meta, sim_results_path)

    # ── 3. 掃描 config 檔案 ─────────────────────────────────────────────────
    config_files = find_config_files(project_paths["configs"])
    print(
        f"[Batch] 專案 '{args.project_name}'：找到 {len(config_files)} 個 config。"
    )

    # ── 4. Resume 計畫（以 config_filename 為鍵）────────────────────────────
    status_map = sim_results_io.get_status_map(sim_results_path)
    already_success, skip_filenames = _build_resume_plan(config_files, status_map)

    # 舊版 summary 初始化（保留 NPZ builder 相容性）
    if not os.path.exists(legacy_summary_path):
        batch_io.init_summary_file(legacy_summary_path)

    # max_success 預檢
    if args.max_success is not None:
        remaining = args.max_success - already_success
        if remaining <= 0:
            print(
                f"[Batch] 已達 max_success={args.max_success}（先前 session 累計 "
                f"{already_success} 個）。無需執行。"
            )
            return
        print(
            f"[Batch] max_success={args.max_success}  "
            f"（已有={already_success}，本次還需={remaining}）"
        )

    # ── 5. 主迴圈 ────────────────────────────────────────────────────────────
    new_success = 0
    new_skip = 0

    for i, cfg_file in enumerate(config_files):
        full_config_path = os.path.join(project_paths["configs"], cfg_file)
        job_id = i + 1

        # ── Skip 判斷 ───────────────────────────────────────────────────────
        if cfg_file in skip_filenames:
            s = status_map.get(cfg_file, "?")
            tag = "OK " if s == "Success" else "ERR"
            print(
                f"--- [Skip {tag} {job_id}/{len(config_files)}] {cfg_file}"
            )
            new_skip += 1
            continue

        # ── max_success 提前結束 ─────────────────────────────────────────────
        total_so_far = already_success + new_success
        if args.max_success is not None and total_so_far >= args.max_success:
            remaining_cfgs = len(config_files) - i
            print(
                f"\n[Batch] 達到 max_success={args.max_success}，停止。"
                f"剩餘 {remaining_cfgs} 個 config 未執行。"
            )
            break

        # ── 執行 ──────────────────────────────────────────────────────────
        progress = (
            f"[{already_success + new_success}/{args.max_success}]"
            if args.max_success
            else f"[已成功 {already_success + new_success}]"
        )
        print(f"\n--- Job {job_id}/{len(config_files)}: {cfg_file} {progress}")
        gc.collect()

        # 預寫 Running（崩潰安全）
        sim_results_io.set_status(
            config_filename=cfg_file,
            status="Running",
            sim_results_path=sim_results_path,
        )
        # 舊版 summary 預寫（相容性）
        try:
            import yaml
            with open(full_config_path, "r", encoding="utf-8") as _f:
                _cfg = yaml.safe_load(_f)
            _sim_cfg = _cfg.get("simulation", {})
            batch_io.update_summary_file(
                {
                    "case_name": _sim_cfg.get("name", cfg_file),
                    "status": "Running",
                    "job_id": job_id,
                    "parameters": {
                        "lattice": {
                            "resolution_px": [
                                _sim_cfg.get("nx"),
                                _sim_cfg.get("ny"),
                            ]
                        }
                    },
                    "source_files": {
                        "config_file": cfg_file,
                        "mask_file": os.path.basename(
                            _cfg.get("mask", {}).get("path", "N/A")
                        ),
                    },
                },
                legacy_summary_path,
            )
        except Exception as _e:
            print(f"  [Warning] 無法預寫舊版 summary：{_e}")

        gc.collect()

        # ── 執行模擬（含計時）─────────────────────────────────────────────
        wall_t0 = time.perf_counter()

        summary_entry = case_executor.execute_case(
            full_config_path, project_paths, output_dirs, job_id
        )

        wall_time_s = time.perf_counter() - wall_t0
        summary_entry["wall_time_s"] = round(wall_time_s, 2)

        # ── 寫入結果 ─────────────────────────────────────────────────────
        is_success = summary_entry.get("status") == "Success"

        if is_success:
            # 提取模擬實際輸出量（不含 Tier1/2/3，它們已在 config_meta 預算）
            sim_params = summary_entry.get("parameters", {})
            actual_outputs = sim_params.get("simulation_outputs", {})

            # summary_builder 可能仍使用舊 key physical_scaled；統一取出
            if not actual_outputs and "simulation_outputs" not in sim_params:
                actual_outputs = {
                    "actual_reynolds_number": sim_params.get(
                        "simulation_outputs", {}
                    ).get("actual_reynolds_number"),
                    "total_steps_executed": sim_params.get(
                        "simulation_outputs", {}
                    ).get("total_steps_executed"),
                    "tensor_shapes": sim_params.get(
                        "simulation_outputs", {}
                    ).get("tensor_shapes"),
                }

            run_summary = summary_entry.get("run_summary", {})

            sim_results_io.fill_simulation_outputs(
                config_filename=cfg_file,
                simulation_outputs=actual_outputs,
                run_summary=run_summary,
                wall_time_s=wall_time_s,
                sim_results_path=sim_results_path,
            )
            new_success += 1
        else:
            # 失敗：寫入 Failed + reason
            sim_results_io.set_status(
                config_filename=cfg_file,
                status="Failed",
                sim_results_path=sim_results_path,
                extra_fields={
                    "wall_time_s": round(wall_time_s, 2),
                    "reason": summary_entry.get("reason", "Unknown"),
                },
            )

        # 舊版 summary 更新（相容性）
        batch_io.update_summary_file(summary_entry, legacy_summary_path)

        print(
            f"  [{'OK' if is_success else 'FAIL'}] {cfg_file}  "
            f"wall_time={wall_time_s:.1f}s"
        )

    # ── 6. 批次結束統計 ──────────────────────────────────────────────────────
    total_success = already_success + new_success
    sep = "=" * 60
    print(f"\n{sep}")
    print("[Batch] Session 完成。")
    print(f"  先前已成功    : {already_success}")
    print(f"  本次新增成功  : {new_success}")
    print(f"  累計總成功    : {total_success}")
    print(f"  本次跳過      : {new_skip}")
    print(f"{sep}")

    # ── 7. NPZ 特徵矩陣（從舊版 summary 讀，保持相容性）────────────────────
    print("\n[Batch] 建立 ML 特徵向量...")
    try:
        build_npz(legacy_summary_path, npz_path)
    except Exception as e:
        print(f"[Warning] NPZ 建立失敗（sim_results.json 仍有效）：{e}")

    print(f"\n[Finished]")
    print(f"  sim_results  -> {sim_results_path}")
    print(f"  legacy summary -> {legacy_summary_path}")
    print(f"  vectors      -> {npz_path}")


if __name__ == "__main__":
    main()
