"""
src/lbm_mrt_les/io/sim_results_io.py

config_meta.json  →  sim_results.json 的橋接 IO 模組。

設計原則：
  - config_meta.json 是唯讀來源（由 config_batch_gen_all_nu.py 生成）
  - sim_results.json 是本模組唯一寫入目標
  - 三層物理尺度（Tier 1/2/3）不在此重算，直接從 config_meta 複製
  - 模擬完成後只填入 simulation_outputs + run_summary + wall_time_s
  - 唯一識別鍵：source_files.config_file（config 檔名），而非 case_name

sim_results.json 結構（每個 entry）：
  {
    "case_name":       str        mask_XX（同一 mask 多個 nu 會重複）
    "config_filename": str        唯一識別鍵，e.g. "mask_00_cfg_Nu0-0300.yaml"
    "status":          str        "Pending" | "Running" | "Success" | "Failed"
    "wall_time_s":     float|null 實際模擬用時（秒）
    "parameters": {
      "lattice_inputs":           dict  從 config_meta 複製（Tier 1）
      "simulation_outputs":       dict  模擬後填入（實際 Re、步數、tensor shape）
      "wind_tunnel_model_scale":  dict  從 config_meta 複製（Tier 2）
      "real_world_urban_scale":   dict  從 config_meta 複製（Tier 3，若存在）
    }
    "run_summary":     dict       h5_file, video_file（模擬後填入）
    "source_files":    dict       config_file, mask_file
  }
"""

import copy
import json
import os
from typing import Any

from .NumpySafeJSONEncoder import NumpySafeJSONEncoder


# ─────────────────────────────────────────────────────────────────────────────
# region 基礎 IO
# ─────────────────────────────────────────────────────────────────────────────


def _read_json(path: str) -> list[dict]:
    """讀取 JSON 清單；不存在或損毀時回傳空清單。"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warning] 無法讀取 {path}：{e}，以空清單替代。")
        return []


def _write_json(data: list[dict], path: str) -> None:
    """原子寫入：先寫 .tmp 再 rename，避免部分寫入損毀。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpySafeJSONEncoder)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[Error] 寫入 {path} 失敗：{e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# region 公開 API
# ─────────────────────────────────────────────────────────────────────────────


def load_config_meta(config_meta_path: str) -> dict[str, dict]:
    """
    載入 config_meta.json，回傳 {config_filename: entry} dict。

    Args:
        config_meta_path: SimCases/{project}/config_meta.json

    Returns:
        以 config_filename 為鍵的 dict；若檔案不存在回傳空 dict。
    """
    entries = _read_json(config_meta_path)
    result: dict[str, dict] = {}
    for entry in entries:
        key = entry.get("config_filename")
        if key:
            result[key] = entry
        else:
            print(f"[Warning] config_meta entry 缺少 config_filename：{entry}")
    print(f"[SimResults] 載入 config_meta：{config_meta_path}（{len(result)} 筆）")
    return result


def init_sim_results(
    config_meta: dict[str, dict],
    sim_results_path: str,
) -> None:
    """
    若 sim_results.json 不存在，將 config_meta 的所有 entry 複製過去並寫入。
    若已存在則不覆蓋（保留已有的模擬結果）。

    Args:
        config_meta:      load_config_meta() 的回傳值
        sim_results_path: 寫入目標路徑
    """
    if os.path.exists(sim_results_path):
        print(f"[SimResults] sim_results.json 已存在，不覆寫：{sim_results_path}")
        return

    entries = list(config_meta.values())
    _write_json(entries, sim_results_path)
    print(f"[SimResults] 初始化完成：{sim_results_path}（{len(entries)} 個 case）")


def get_status_map(sim_results_path: str) -> dict[str, str]:
    """
    讀取 sim_results.json，回傳 {config_filename: status} dict。
    供 batch_run 判斷哪些 case 可以跳過。

    Returns:
        空 dict 代表尚無任何記錄。
    """
    entries = _read_json(sim_results_path)
    return {
        e["config_filename"]: e.get("status", "Unknown")
        for e in entries
        if "config_filename" in e
    }


def set_status(
    config_filename: str,
    status: str,
    sim_results_path: str,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """
    原地更新單一 entry 的 status 欄位（與可選的額外欄位）。

    用於：
      - 預寫 "Running"（crash-safe 中斷識別）
      - 寫入 "Failed"（含 reason）

    Args:
        config_filename: 唯一識別鍵
        status:          "Running" | "Failed" | "Pending"
        sim_results_path: sim_results.json 路徑
        extra_fields:    要額外合併到 entry 頂層的欄位（如 reason, wall_time_s）
    """
    entries = _read_json(sim_results_path)
    found = False
    for entry in entries:
        if entry.get("config_filename") == config_filename:
            entry["status"] = status
            if extra_fields:
                entry.update(extra_fields)
            found = True
            break
    if not found:
        # config_meta 未包含此 config（異常情況），建立最簡 entry
        new_entry: dict[str, Any] = {
            "config_filename": config_filename,
            "status": status,
        }
        if extra_fields:
            new_entry.update(extra_fields)
        entries.append(new_entry)
        print(f"[Warning] {config_filename} 不在 config_meta，已新增最簡 entry。")

    _write_json(entries, sim_results_path)


def fill_simulation_outputs(
    config_filename: str,
    simulation_outputs: dict[str, Any],
    run_summary: dict[str, str],
    wall_time_s: float,
    sim_results_path: str,
) -> None:
    """
    模擬成功後，將實際模擬結果填入對應 entry 的 simulation_outputs 欄位。

    不重算 Tier 1/2/3 參數，只填入：
      - simulation_outputs.actual_reynolds_number
      - simulation_outputs.total_steps_executed
      - simulation_outputs.tensor_shapes
      - run_summary.h5_file / video_file
      - wall_time_s
      - status → "Success"

    Args:
        config_filename:    唯一識別鍵
        simulation_outputs: 來自 run_one_case 的實際量測值
        run_summary:        {"h5_file": ..., "video_file": ...}
        wall_time_s:        實際 wall-clock 模擬用時（秒）
        sim_results_path:   sim_results.json 路徑
    """
    entries = _read_json(sim_results_path)
    found = False
    for entry in entries:
        if entry.get("config_filename") == config_filename:
            entry["status"] = "Success"
            entry["wall_time_s"] = round(wall_time_s, 2)

            # 填入 simulation_outputs（保留 _note，只更新實際量）
            existing_sim_out = entry.get("parameters", {}).get(
                "simulation_outputs", {}
            )
            existing_sim_out.update(
                {
                    "actual_reynolds_number": simulation_outputs.get(
                        "actual_reynolds_number"
                    ),
                    "total_steps_executed": simulation_outputs.get(
                        "total_steps_executed"
                    ),
                    "tensor_shapes": simulation_outputs.get("tensor_shapes"),
                }
            )
            existing_sim_out.pop("_note", None)
            entry.setdefault("parameters", {})["simulation_outputs"] = existing_sim_out

            # 填入 run_summary
            entry["run_summary"] = run_summary

            found = True
            break

    if not found:
        print(f"[Warning] fill_simulation_outputs：找不到 {config_filename}，略過。")
        return

    _write_json(entries, sim_results_path)
    print(f"[SimResults] 已填入模擬結果：{config_filename}（{wall_time_s:.1f} s）")
