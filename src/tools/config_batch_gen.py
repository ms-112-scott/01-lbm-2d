"""
src/tools/config_batch_gen_all_nu.py

全 nu 展開版本的批次 config 生成器。
每個 mask 展開所有可行 nu，各自生成一個 YAML config 檔，
並同時將三層物理尺度的預算參數寫入 config_meta.json。

資料流：
  master_config.yaml
    → build_sim_context()         → sim_ctx（全批次共享）
    for each mask:
      metadata.json entry
        → build_mask_context()    → mask_ctx
        → fill_geometry()         → mask_ctx["l_char"], ["max_blockage"]
      → fill_blockage_adj()       → case_result["rho_in_case", ...]
      → _get_all_feasible_nu()    → [(nu, Re), ...]（全部可行）
        for each nu:
          case_result["nu_lb"] = nu
          → fill_physics_and_steps() → 步數、Ma、Re、dx_mm
          → build_config()           → (yaml_dict, output_path)
          → save_yaml()
          → _build_case_meta()       → meta_entry（三層尺度）
    → save config_meta.json        → SimCases/{project}/config_meta.json

三層物理尺度：
  Tier 1 lattice_inputs         : 無因次格子量（直接來自 config 參數）
  Tier 2 wind_tunnel_model_scale: 等效縮尺風洞量（Re 相似性推導 dx/dt）
  Tier 3 real_world_urban_scale : 真實城市量（GIS metadata m_per_px 換算）
"""

import copy
import glob
import json
import math
import os
import sys
import argparse
from typing import Any

import numpy as np
import yaml

from config_utils import (
    build_sim_context,
    build_mask_context,
    fill_geometry,
    fill_blockage_adj,
    fill_physics_and_steps,
    build_config,
    calc_l_char,
    print_re_preview,
    print_summary,
)
from config_utils.feasibility import check_feasibility
from config_utils.constants import CS


# ─────────────────────────────────────────────────────────────────────────────
# region I/O 輔助
# ─────────────────────────────────────────────────────────────────────────────


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到設定檔：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(config: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=None)


def load_mask_metadata(mask_dir: str) -> dict:
    """載入 metadata.json，回傳 {file_name: entry} dict。"""
    json_path = os.path.join(mask_dir, "metadata.json")
    if not os.path.exists(json_path):
        print(f"[Warning] metadata.json 不存在：{json_path}")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    result = {e["file_name"]: e for e in entries}
    print(f"[Info] 載入 metadata：{json_path}（{len(result)} 筆）")
    return result


def save_meta_json(meta_list: list[dict], output_path: str) -> None:
    """將 config_meta 清單序列化為 JSON。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2, ensure_ascii=False)
    print(f"[Meta] 已儲存：{output_path}（{len(meta_list)} 個 case）")


# ─────────────────────────────────────────────────────────────────────────────
# region 驗證
# ─────────────────────────────────────────────────────────────────────────────


def validate_passes(sim_ctx: dict) -> None:
    w, s, t = (
        sim_ctx["warmup_passes"],
        sim_ctx["start_record_passes"],
        sim_ctx["total_passes"],
    )
    if not (w < s < t):
        print(
            f"[Error] passes 設定不合理：\n"
            f"  warmup={w}, start_record={s}, total={t}\n"
            f"  必須滿足 warmup < start_record < total。"
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# region Pre-scan
# ─────────────────────────────────────────────────────────────────────────────


def prescan_l_char(
    mask_files: list[str],
    sim_ctx: dict,
    mask_meta: dict,
) -> list[int]:
    """預掃描所有 mask 的 L_char，用於 Re 預覽表。"""
    results: list[int] = []
    for mp in mask_files:
        fname = os.path.basename(mp)
        entry = mask_meta.get(fname)
        if entry is None:
            continue
        try:
            nx = int(entry["domain_W_total"])
            ny = int(entry["domain_H_total"])
            lc = calc_l_char(mp, sim_ctx["mask_invert"], nx, ny)
            results.append(lc)
        except Exception as e:
            print(f"  [Warning] {fname}：{e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# region 全 nu 展開
# ─────────────────────────────────────────────────────────────────────────────


def _get_all_feasible_nu(
    rho_in_case: float,
    rho_out: float,
    nu_lb_list: list[float],
    l_char: int,
) -> list[tuple[float, float]]:
    """
    過濾所有可行 nu，回傳 [(nu, Re_estimated), ...] 清單（nu 由大到小）。

    Re 以 Bernoulli 速度估算，與 fill_nu_sample 邏輯一致。
    """
    delta_rho = rho_in_case - rho_out
    u_b = math.sqrt((2.0 / 3.0) * delta_rho) if delta_rho > 1e-9 else 0.01

    feasible: list[tuple[float, float]] = []
    for nu in sorted(nu_lb_list, reverse=True):
        ok, reason = check_feasibility(rho_in_case, rho_out, nu, l_char)
        if ok:
            feasible.append((nu, u_b * l_char / nu))
        else:
            print(f"    [Skip nu={nu:.4f}] {reason}")
    return feasible


# ─────────────────────────────────────────────────────────────────────────────
# region 三層物理尺度計算（config_meta 核心）
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_sci(value: float, digits: int = 4) -> str:
    """將 float 格式化為科學記號字串，與 all_cases_summary_v2 格式一致。"""
    return f"{value:.{digits}e}"


def _build_tier1(
    case_result: dict,
    mask_ctx: dict,
    sim_ctx: dict,
) -> dict:
    """
    Tier 1：格子單位（Lattice Units），所有量為無因次。
    """
    return {
        "target_rho_in": case_result["rho_in_case"],
        "rho_in": case_result["rho_in_case"],
        "rho_out": sim_ctx["rho_out"],
        "characteristic_length_px": float(mask_ctx["l_char"]),
        "inlet_velocity_lu": round(case_result["u_bernoulli"], 6),
        "kinematic_viscosity_lu": case_result["nu_lb"],
        "resolution_px": [mask_ctx["nx"], mask_ctx["ny"]],
    }


def _build_tier2(
    case_result: dict,
    mask_ctx: dict,
    sim_ctx: dict,
) -> dict:
    """
    Tier 2：等效縮尺風洞模型（Re 相似性推導 dx/dt）。

    dx      = nu_air * u_lu / (nu_lu * u_ref)
    dt      = u_lu * dx / u_ref
    L_model = L_px * dx
    Re      = u_lu * L_px / nu_lu

    total_simulation_time_s_estimated：以 max_steps 預算，實際模擬後更新。
    """
    u_lu: float = case_result["u_bernoulli"]
    nu_lu: float = case_result["nu_lb"]
    nu_air: float = sim_ctx["nu_air"]
    u_ref: float = sim_ctx["U_phys"]
    l_px: int = mask_ctx["l_char"]
    max_steps: int = case_result["max_steps"]

    dx: float = nu_air * u_lu / (nu_lu * u_ref)
    dt: float = u_lu * dx / u_ref
    l_model: float = l_px * dx
    re: float = u_lu * l_px / nu_lu

    return {
        "_note": (
            "等效縮尺風洞實驗尺度。"
            "空間單位 dx = nu_air * u_lu / (nu_lu * u_ref)，與真實城市幾何無關。"
        ),
        "reference_inlet_velocity_ms": u_ref,
        "reynolds_number_calculated": round(re, 4),
        "characteristic_length_m": _fmt_sci(l_model),
        "kinematic_viscosity_air_m2_s": _fmt_sci(nu_air),
        "cell_size_m": _fmt_sci(dx),
        "time_step_s": _fmt_sci(dt),
        "steps_per_physical_second": _fmt_sci(1.0 / dt),
        "total_simulation_time_s_estimated": _fmt_sci(max_steps * dt),
    }


def _build_tier3(
    case_result: dict,
    mask_ctx: dict,
    sim_ctx: dict,
    meta_entry: dict,
) -> dict | None:
    """
    Tier 3：真實城市尺度（GIS metadata m_per_px 換算）。

    空間縮放：dx_real = m_per_px
    時間縮放（對流相似）：
      scale = (L_real / L_model) * (u_model / u_real)
      dt_real = dt_model * scale
    Re_real = u_real * L_real / nu_air

    若 metadata 無 m_per_px 欄位則回傳 None，僅生成 Tier 1/2。
    """
    m_per_px: float | None = meta_entry.get("m_per_px")
    if m_per_px is None:
        return None

    u_lu: float = case_result["u_bernoulli"]
    nu_lu: float = case_result["nu_lb"]
    nu_air: float = sim_ctx["nu_air"]
    u_ref: float = sim_ctx["U_phys"]
    l_px: int = mask_ctx["l_char"]
    max_steps: int = case_result["max_steps"]

    # Tier 2 中間量（用於時間縮放基準）
    dx_model: float = nu_air * u_lu / (nu_lu * u_ref)
    dt_model: float = u_lu * dx_model / u_ref
    l_model: float = l_px * dx_model

    # Tier 3 空間固定量
    dx_real: float = m_per_px
    l_real: float = l_px * m_per_px

    # 空間縮放比（約 40,000 倍）
    spatial_scale: float = l_real / l_model

    # 各風速案例
    raw_speeds: list[float] | float = sim_ctx["physical_constants"]["inlet_velocity_ms"]
    speeds: list[float] = raw_speeds if isinstance(raw_speeds, list) else [raw_speeds]

    wind_speed_cases: dict[str, dict] = {}
    for u_real in speeds:
        # 對流時間縮放：時間縮放 = 空間縮放 * 速度縮放
        time_scale: float = spatial_scale * (u_ref / u_real)
        dt_real: float = dt_model * time_scale
        re_real: float = u_real * l_real / nu_air
        key: str = f"{u_real:.1f}_ms"
        wind_speed_cases[key] = {
            "inlet_velocity_ms": u_real,
            "reynolds_number": round(re_real, 0),
            "cell_size_m": _fmt_sci(dx_real, 3),
            "time_step_s": _fmt_sci(dt_real),
            "steps_per_physical_second": _fmt_sci(1.0 / dt_real),
            "total_simulation_time_s_estimated": _fmt_sci(max_steps * dt_real),
        }

    return {
        "_note": (
            "真實城市幾何尺度，由 GIS 元數據 m_per_px 換算。"
            "時間依對流相似性縮放：dt_real = dt_model * (L_real/L_model) * (u_model/u_real)。"
            "Re_real 反映真實都市風場（完全湍流）。"
        ),
        "cell_size_m": _fmt_sci(dx_real, 3),
        "characteristic_length_m": _fmt_sci(l_real),
        "m_per_px": m_per_px,
        "kinematic_viscosity_air_m2_s": _fmt_sci(nu_air),
        "wind_speed_cases": wind_speed_cases,
    }


def _build_case_meta(
    case_result: dict,
    mask_ctx: dict,
    sim_ctx: dict,
    meta_entry: dict,
) -> dict:
    """
    組裝單一 case 的三層尺度 meta dict（對應 all_cases_summary_v2 格式）。

    simulation_outputs 欄位以 None 占位，
    待實際模擬完成後由後處理腳本回填。
    """
    tier2 = _build_tier2(case_result, mask_ctx, sim_ctx)
    tier3 = _build_tier3(case_result, mask_ctx, sim_ctx, meta_entry)

    parameters: dict[str, Any] = {
        "lattice_inputs": _build_tier1(case_result, mask_ctx, sim_ctx),
        "simulation_outputs": {
            "_note": "待模擬完成後由後處理腳本回填。",
            "actual_reynolds_number": None,
            "total_steps_executed": None,
            "tensor_shapes": None,
        },
        "wind_tunnel_model_scale": tier2,
    }
    if tier3 is not None:
        parameters["real_world_urban_scale"] = tier3

    return {
        "case_name": case_result["sim_name"],
        "config_filename": case_result["config_filename"],
        "status": "Pending",
        "parameters": parameters,
        "source_files": {
            "config_file": case_result["config_filename"],
            "mask_file": os.path.basename(mask_ctx["mask_path"]),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# region 單一 mask 處理
# ─────────────────────────────────────────────────────────────────────────────


def process_mask_all_nu(
    mask_path: str,
    meta_entry: dict,
    sim_ctx: dict,
) -> tuple[int, int, list[dict]]:
    """
    對單一 mask 展開所有可行 nu，各自生成 YAML + meta_entry。

    Returns:
        (成功數, skip 數, case_meta 清單)
    """
    mask_ctx = build_mask_context(mask_path, meta_entry)

    try:
        fill_geometry(mask_ctx, sim_ctx)
    except Exception as e:
        print(f"  [Skip] 讀取 mask 失敗：{e}\n")
        return 0, 1, []

    print(
        f"  [Mask] {mask_ctx['mask_stem']}  "
        f"nx={mask_ctx['nx']}  ny={mask_ctx['ny']}  "
        f"L_char={mask_ctx['l_char']} px"
    )

    case_base: dict = {}
    fill_blockage_adj(case_base, mask_ctx, sim_ctx)
    rho_in_case: float = case_base["rho_in_case"]
    print(
        f"  [BlockageAdj] rho_in_case={rho_in_case:.5f}  "
        f"open_fraction={case_base.get('open_fraction', 0):.0%}"
    )

    feasible = _get_all_feasible_nu(
        rho_in_case,
        sim_ctx["rho_out"],
        sim_ctx["nu_lb_list"],
        mask_ctx["l_char"],
    )

    if not feasible:
        print(f"  [Skip] {mask_ctx['mask_stem']}：無可行 nu，略過。\n")
        return 0, 1, []

    print(f"  [NuAll] 可行選項 {len(feasible)} 個：")
    for nu, re in feasible:
        print(f"    nu={nu:.4f}  tau={3*nu+0.5:.4f}  Re≈{re:.0f}")

    success_count = 0
    case_metas: list[dict] = []

    for nu, _ in feasible:
        case_result: dict = copy.deepcopy(case_base)
        case_result["nu_lb"] = nu
        case_result["nu_re_pairs"] = feasible

        fill_physics_and_steps(case_result, mask_ctx, sim_ctx)

        config, out_path = build_config(case_result, mask_ctx, sim_ctx)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_yaml(config, out_path)

        print(
            f"    -> YAML：{case_result['config_filename']}"
            f"  Re≈{case_result['Re']:.0f}"
            f"  Ma={case_result['Ma']:.4f}"
            f"  tau={case_result['tau']:.4f}"
        )

        case_meta = _build_case_meta(case_result, mask_ctx, sim_ctx, meta_entry)
        case_metas.append(case_meta)
        success_count += 1

    print()
    return success_count, 0, case_metas


# ─────────────────────────────────────────────────────────────────────────────
# region Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "批次生成所有 mask × 所有可行 nu 的 LBM config YAML，"
            "並同步輸出三層物理尺度的 config_meta.json。"
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        default="master_config.yaml",
        help="master_config YAML 路徑（預設：master_config.yaml）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="僅列印 Re 預覽表，不實際生成任何檔案",
    )
    args = parser.parse_args()

    master_cfg = load_yaml(args.config)
    sim_ctx = build_sim_context(master_cfg)

    validate_passes(sim_ctx)
    os.makedirs(sim_ctx["output_dir"], exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(sim_ctx["mask_dir"], "*.png")))
    if not mask_files:
        print(f"[Error] 在 {sim_ctx['mask_dir']} 找不到任何 PNG 檔。")
        return

    mask_meta = load_mask_metadata(sim_ctx["mask_meta_dir"])

    l_char_all = prescan_l_char(mask_files, sim_ctx, mask_meta)
    if not l_char_all:
        print("[Error] 無法從任何 mask 計算 L_char，請確認路徑與格式。")
        return

    l_min, l_max = min(l_char_all), max(l_char_all)
    print_re_preview(sim_ctx, (l_min, l_max))

    if args.dry_run:
        print("[Dry-run] 結束，未生成任何檔案。")
        return

    # 主迴圈：mask × nu 全展開
    total_success = 0
    total_skipped = 0
    all_case_metas: list[dict] = []

    for mask_path in mask_files:
        fname = os.path.basename(mask_path)
        entry = mask_meta.get(fname)
        if entry is None:
            print(f"[Skip] {fname} 不在 metadata.json，略過。\n")
            total_skipped += 1
            continue

        success, skipped, metas = process_mask_all_nu(mask_path, entry, sim_ctx)
        total_success += success
        total_skipped += skipped
        all_case_metas.extend(metas)

    # 寫出 config_meta.json
    meta_json_path = os.path.join(
        "SimCases",
        sim_ctx["project_name"],
        "config_meta.json",
    )
    save_meta_json(all_case_metas, meta_json_path)

    print_summary(sim_ctx, total_success, total_skipped, l_min, l_max)


if __name__ == "__main__":
    main()
