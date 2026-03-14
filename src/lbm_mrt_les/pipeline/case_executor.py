"""
src/lbm_mrt_les/pipeline/case_executor.py

Orchestrates a single simulation case:
  1. Resolves paths
  2. Runs the simulation (run_one_case.main)
  3. Builds the structured summary entry

與舊版差異：
  - 新增 wall_time_s 計時（在 batch_run 層做，executor 不重計）
  - simulation_outputs 使用標準鍵名，與 sim_results_io 對齊
  - 不再呼叫 physics_scaling.calculate_physical_params（物理尺度在 config_meta 已預算）
  - summary_builder 仍呼叫以維持舊版 all_cases_summary.json 格式相容性

On failure: cleans up partial output files in raw/ and vis/ to save disk space.
"""

import os
import glob
from typing import Dict

from .. import utils
from ..utils import physics_scaling
from . import summary_builder
from .run_one_case import main as run_one_case_main


# ─────────────────────────────────────────────────────────────────────────────
# region Cleanup helper
# ─────────────────────────────────────────────────────────────────────────────


def _cleanup_failed_outputs(h5_path: str, video_path: str) -> None:
    """
    移除失敗模擬產生的不完整 .h5 / .mp4 檔，避免殘檔累積。
    Glob pattern 同時清除 .tmp / .part 等臨時後綴。
    """
    for path in [h5_path, video_path]:
        if not path:
            continue
        for fpath in [path] + glob.glob(path + ".*"):
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    print(f"  [Cleanup] 移除不完整輸出：{fpath}")
                except OSError as e:
                    print(f"  [Cleanup] 無法移除 {fpath}：{e}")


# ─────────────────────────────────────────────────────────────────────────────
# region Main executor
# ─────────────────────────────────────────────────────────────────────────────


def execute_case(
    full_config_path: str,
    project_paths: Dict,
    output_dirs: Dict,
    job_id: int,
) -> Dict:
    """
    執行單一模擬 case，回傳結構化 summary dict。

    回傳的 dict 包含：
      status          : "Success" | "Failed"
      parameters:
        lattice_inputs     : 格子輸入量（來自 config）
        simulation_outputs : 實際量測值（actual Re, steps, tensor shapes）
        physical_scaled    : Tier 2 風洞尺度（保留舊版相容性）
      run_summary:
        h5_file, video_file
      source_files:
        config_file, mask_file
      wall_time_s       : 由 batch_run 在外層計時後填入（此處不計）

    此函式內部不計時，wall_time_s 由 batch_run 在呼叫前後計算。
    Never raises：任何例外均轉為 Failed 狀態回傳。
    """
    h5_path = ""
    video_path = ""
    sim_name = os.path.basename(full_config_path)  # fallback

    try:
        config = utils.load_config(full_config_path)

        # ── 1. 路徑解析 ──────────────────────────────────────────────────────
        mask_path_from_cfg = config.get("mask", {}).get("path", "")
        sim_name = config.get("simulation", {}).get("name", sim_name)
        cfg_filename = os.path.basename(full_config_path)

        mask_path = os.path.join(
            project_paths["masks"], os.path.basename(mask_path_from_cfg)
        )
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        h5_path = os.path.join(output_dirs["raw"], f"{sim_name}.h5")
        video_path = os.path.join(output_dirs["vis"], f"{sim_name}.mp4")

        # ── 2. 執行模擬 ──────────────────────────────────────────────────────
        lattice_metadata = run_one_case_main(
            full_config_path, mask_path, h5_path, video_path
        )

        if lattice_metadata.get("status") != "Success":
            raise RuntimeError(
                f"Simulation failed: {lattice_metadata.get('reason')}"
            )

        # ── 3. 組裝 simulation_outputs（標準鍵名，供 sim_results_io 使用）──
        sim_out = {
            "actual_reynolds_number": round(
                lattice_metadata.get("reynolds_number_lattice_actual", 0.0), 4
            ),
            "total_steps_executed": lattice_metadata.get("total_steps_executed"),
            "tensor_shapes": {
                "static_mask": lattice_metadata.get("tensor_shape_static_mask"),
                "turbulence": lattice_metadata.get("tensor_shape_turbulence"),
            },
        }

        # ── 4. 物理尺度（Tier 2，保留舊版 summary 相容性）─────────────────
        physical_params = physics_scaling.calculate_physical_params(
            config, lattice_metadata
        )

        source_files = {
            "config_file": cfg_filename,
            "mask_file": os.path.basename(mask_path),
        }

        # ── 5. 組裝完整 summary entry ────────────────────────────────────────
        # summary_builder 產生舊格式（all_cases_summary.json 相容）
        legacy_entry = summary_builder.build_summary_entry(
            config, lattice_metadata, physical_params, source_files
        )

        # 覆寫 simulation_outputs 為標準格式（讓 batch_run 的 sim_results_io 可直接取用）
        legacy_entry.setdefault("parameters", {})["simulation_outputs"] = sim_out

        # 加入 config_filename 方便 batch_run 傳給 sim_results_io
        legacy_entry["config_filename"] = cfg_filename

        print(
            f"  [Success] {sim_name}  "
            f"Re={sim_out['actual_reynolds_number']:.2f}  "
            f"steps={sim_out['total_steps_executed']:,}"
        )
        return legacy_entry

    except Exception as e:
        print(f"  [Error] Case '{sim_name}' failed: {e}")
        if h5_path or video_path:
            _cleanup_failed_outputs(h5_path, video_path)
        return {
            "case_name": sim_name,
            "config_filename": os.path.basename(full_config_path),
            "status": "Failed",
            "reason": str(e),
        }
