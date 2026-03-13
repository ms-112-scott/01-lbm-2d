"""
config_utils/config_assembler.py

從 sim_ctx、mask_ctx、case_result 組裝完整的 YAML config dict，
並決定輸出檔名。

職責：
  - deep copy base_template
  - 填入 case 專屬參數
  - 回傳 (config_dict, output_path) tuple
  - 呼叫端負責寫入磁碟
"""

import copy
import os


def build_config(case_result: dict, mask_ctx: dict, sim_ctx: dict) -> tuple[dict, str]:
    """
    組裝單一 case 的 YAML config dict 並決定輸出路徑。

    Args:
        case_result : 須含 rho_in_case, nu_lb, l_char（由前序步驟寫入）
                      以及步數欄位（由 fill_physics_and_steps 寫入）
        mask_ctx    : 須含 mask_stem, mask_path
        sim_ctx     : 須含 base_template, physical_constants, rho_out,
                      project_name, data_save_root, output_dir, c_smag

    Returns:
        (config_dict, full_output_path)
    """
    config = copy.deepcopy(sim_ctx["base_template"])
    config["physical_constants"] = sim_ctx["physical_constants"]

    nu_lb = case_result["nu_lb"]
    rho_in = case_result["rho_in_case"]
    mask_stem = mask_ctx["mask_stem"]

    # sim_name = mask 前兩段（e.g. "mask_01"）
    sim_name = "_".join(mask_stem.split("_")[:2])

    # simulation
    sim = config["simulation"]
    sim["name"] = sim_name
    sim["nu"] = float(f"{nu_lb:.6f}")
    sim["characteristic_length"] = float(mask_ctx["l_char"])
    sim["rho_in"] = float(rho_in)
    sim["rho_out"] = float(sim_ctx["rho_out"])
    sim["compute_step_size"] = case_result["interval"]
    sim["warmup_steps"] = case_result["warmup_steps"]
    sim["max_steps"] = case_result["max_steps"]
    sim["smagorinsky_constant"] = sim_ctx["c_smag"]

    # nx / ny 從 mask_ctx 填入（來源：metadata.json）
    print(mask_ctx)
    sim["nx"] = mask_ctx["nx"]
    sim["ny"] = mask_ctx["ny"]

    # outputs
    out = config["outputs"]
    out["project_name"] = sim_ctx["project_name"]
    out["data_save_root"] = sim_ctx["data_save_root"]
    out["target_rho_in"] = float(rho_in)
    out["start_record_step"] = case_result["start_record_step"]
    out["gui"]["interval_steps"] = case_result["interval"]
    out["video"]["interval_steps"] = case_result["interval"]
    out["video"]["filename"] = f"{sim_name}.mp4"
    out["dataset"]["interval_steps"] = case_result["interval"]
    out["dataset"].pop("folder", None)

    # domain_zones — 從 mask_ctx 的 pad_* 填入
    buffer = sim_ctx.get("blockage_buffer", 128)
    dz = config.get("domain_zones", {})
    dz["sponge_top"] = max(1, mask_ctx["pad_top"] - buffer)  # 上阻尼寬度
    dz["sponge_bot"] = max(1, mask_ctx["pad_bot"] - buffer)  # 下阻尼寬度
    dz["sponge_out"] = max(1, mask_ctx["pad_right"] - buffer)  # 右側出口阻尼寬度
    dz["sponge_in"] = max(1, mask_ctx["pad_left"] - buffer)  # 左側入口阻尼寬度
    dz["buffer"] = buffer
    # 移除舊 key 避免殘留干擾
    dz.pop("sponge_x", None)
    dz.pop("sponge_y", None)
    dz.pop("inlet_buffer", None)
    dz.pop("sponge_inlet", None)
    config["domain_zones"] = dz

    # boundary_condition：Zou-He 壓力邊界 dummy velocity
    config["boundary_condition"]["value"] = [[0.05, 0.0]] + [[0.0, 0.0]] * 3

    # mask 路徑
    config["mask"]["path"] = mask_ctx["mask_path"]

    # 輸出路徑
    nu_str = f"{nu_lb:.4f}".replace(".", "-")
    config_filename = f"{mask_stem}_cfg_Nu{nu_str}.yaml"
    full_path = os.path.join(sim_ctx["output_dir"], config_filename)

    # 寫回 case_result 供日誌使用
    case_result["config_filename"] = config_filename
    case_result["sim_name"] = sim_name

    return config, full_path
