"""
src/tools/config_batch_gen.py

從 master_config.yaml 批量生成每個 case 的 YAML 設定檔。

資料流：
  master_config.yaml
    → build_sim_context()  → sim_ctx  (全域共享)
    ┌── for each mask:
    │     metadata.json entry
    │       → build_mask_context()   → mask_ctx
    │       → fill_geometry()        → mask_ctx["l_char"], ["max_blockage"]
    │     {}
    │       → fill_blockage_adj()    → case_result["rho_in_case", ...]
    │       → fill_nu_sample()       → case_result["nu_lb", "nu_re_pairs"]
    │       → fill_physics_and_steps()→ case_result["Re", "steps_per_ctu", ...]
    │       → build_config()         → (yaml_dict, output_path)
    │       → save YAML
    └──
"""

import yaml
import os
import sys
import glob
import argparse
import json

import numpy as np

from config_utils import (
    build_sim_context,
    build_mask_context,
    fill_geometry,
    fill_blockage_adj,
    fill_nu_sample,
    fill_physics_and_steps,
    build_config,
    calc_l_char,
    print_re_preview,
    print_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# region I/O helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(config: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=None)


def load_mask_metadata(mask_dir: str) -> dict:
    """載入 metadata.json → {file_name: entry} dict。"""
    json_path = os.path.join(mask_dir, "metadata.json")
    if not os.path.exists(json_path):
        print(f"[Warning] metadata.json 不存在：{json_path}，將使用 template 預設值。")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    result = {e["file_name"]: e for e in entries}
    print(f"[Info] Loaded mask metadata: {json_path} ({len(result)} entries)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# region Validation
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
            f"  必須滿足 warmup < start_record < total。\n"
            f"  若 start_record ≥ total，HDF5 內將完全沒有資料。"
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# region Pre-scan
# ─────────────────────────────────────────────────────────────────────────────


def prescan_l_char(mask_files: list, sim_ctx: dict, mask_meta: dict) -> list:
    results = []
    for mp in mask_files:
        fname = os.path.basename(mp)
        entry = mask_meta.get(fname)
        if entry is None:
            print(f"  [Pre-scan Skip] {fname} 不在 metadata.json")
            continue
        try:
            nx = int(entry["domain_W_total"])
            ny = int(entry["domain_H_total"])
            l = calc_l_char(mp, sim_ctx["mask_invert"], nx, ny)
            results.append(l)
        except (KeyError, Exception) as e:
            print(f"  [Warning] {fname}: {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# region Per-mask 處理
# ─────────────────────────────────────────────────────────────────────────────


def process_mask(mask_path: str, meta_entry: dict, sim_ctx: dict) -> bool:
    """
    處理單一 mask，生成對應的 YAML config 檔。

    流程：build_mask_context → fill_geometry → fill_blockage_adj
          → fill_nu_sample → fill_physics_and_steps → build_config → save

    Returns:
        True → 成功，False → skip
    """
    # 建立 mask_ctx（含 nx/ny/pad_* 來自 metadata）
    mask_ctx = build_mask_context(mask_path, meta_entry)

    # 幾何計算（l_char, max_blockage）
    try:
        fill_geometry(mask_ctx, sim_ctx)
    except Exception as e:
        print(f"  [Skip] 讀取 mask 失敗：{e}\n")
        return False

    _log_mask(mask_ctx)

    # case_result：各步驟依序填入
    case_result = {}

    fill_blockage_adj(case_result, mask_ctx, sim_ctx)
    _log_blockage(case_result, sim_ctx)

    if not fill_nu_sample(case_result, mask_ctx, sim_ctx):
        return False
    _log_nu(case_result)

    fill_physics_and_steps(case_result, mask_ctx, sim_ctx)
    _log_physics(case_result)

    config, out_path = build_config(case_result, mask_ctx, sim_ctx)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_yaml(config, out_path)
    print(
        f"  -> Saved: {case_result['config_filename']}  (Re≈{case_result['Re']:.0f})\n"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# region 日誌輔助
# ─────────────────────────────────────────────────────────────────────────────


def _log_mask(mask_ctx: dict) -> None:
    print(f"  [Mask] {mask_ctx['mask_stem']}  nx={mask_ctx['nx']}  ny={mask_ctx['ny']}")


def _log_blockage(case_result: dict, sim_ctx: dict) -> None:
    from config_utils.constants import U_GAP_MAX

    rho_in = sim_ctx["rho_in"]
    rho_in_case = case_result["rho_in_case"]
    print(
        f"  [Geometry]   L_char=?px  blockage={case_result.get('open_fraction', 0):.0%}"
    )
    if rho_in_case < rho_in - 1e-6:
        print(
            f"  [BlockageAdj] rho_in {rho_in:.5f} → {rho_in_case:.5f}  "
            f"(u_safe={case_result['u_inlet_safe']:.5f}  u_gap_max={U_GAP_MAX:.2f})"
        )
    else:
        print(f"  [BlockageAdj] rho_in={rho_in_case:.5f} (無需調整)")


def _log_nu(case_result: dict) -> None:
    pairs_str = ", ".join(
        f"nu={n:.3f}→Re={re:.0f}" for n, re in case_result["nu_re_pairs"]
    )
    print(
        f"  [NuSample]   可用 {len(case_result['nu_re_pairs'])} 個選項 "
        f"[{pairs_str}]  → 選 nu={case_result['nu_lb']:.4f}"
    )


def _log_physics(case_result: dict) -> None:
    print(
        f"  tau={case_result['tau']:.4f}  u={case_result['u_bernoulli']:.5f}  "
        f"Ma={case_result['Ma']:.4f}  Re={case_result['Re']:.0f}  "
        f"dx={case_result['dx_mm']:.3f}mm\n"
        f"  CTU={case_result['steps_per_ctu']:,}  "
        f"warmup={case_result['warmup_steps']:,}  "
        f"max={case_result['max_steps']:,}  "
        f"start_rec={case_result['start_record_step']:,}  "
        f"interval={case_result['interval']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# region Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    MASTER_CONFIG_PATH = "master_config.yaml"

    master_cfg = load_yaml(MASTER_CONFIG_PATH)
    sim_ctx = build_sim_context(master_cfg)

    validate_passes(sim_ctx)
    os.makedirs(sim_ctx["output_dir"], exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(sim_ctx["mask_dir"], "*.png")))
    if not mask_files:
        print(f"[Error] No PNG files found in: {sim_ctx['mask_dir']}")
        return

    mask_meta = load_mask_metadata(sim_ctx["mask_meta_dir"])

    # Pre-scan → Re 預覽表
    l_char_all = prescan_l_char(mask_files, sim_ctx, mask_meta)
    if not l_char_all:
        print("[Error] 無法讀取任何 mask，請確認路徑與格式。")
        return

    l_min, l_max = min(l_char_all), max(l_char_all)
    print(
        f"[Pre-scan] L_char 範圍：{l_min} ~ {l_max} px  "
        f"(平均 {int(np.mean(l_char_all))} px，共 {len(l_char_all)} 個有效 mask)"
    )
    print_re_preview(sim_ctx, (l_min, l_max))

    # 批次生成
    print(f"--- Found {len(mask_files)} masks. Generating configs... ---\n")
    success, skipped = 0, 0

    for mask_path in mask_files:
        fname = os.path.basename(mask_path)
        meta_entry = mask_meta.get(fname)
        if meta_entry is None:
            print(f"  [Skip] {fname} 不在 metadata.json 中\n")
            skipped += 1
            continue

        if process_mask(mask_path, meta_entry, sim_ctx):
            success += 1
        else:
            skipped += 1

    print_summary(sim_ctx, success, skipped, l_min, l_max)


if __name__ == "__main__":
    main()
