"""
config_utils/blockage_adjuster.py

依據阻塞率動態調整 rho_in，結果寫入 case_result dict。

【物理推導】
  u_gap = u_inlet / (1 - blockage)
  需要：u_gap < U_GAP_MAX
  → delta_rho_safe = (3/2) × (U_GAP_MAX × open_fraction)²
  → rho_in_case = min(rho_in, rho_out + delta_rho_safe)
"""

from .constants import U_GAP_MAX, MIN_OPEN


def fill_blockage_adj(case_result: dict, mask_ctx: dict, sim_ctx: dict) -> None:
    """
    計算調整後的 rho_in，結果寫入 case_result。

    Writes:
        case_result["rho_in_case"]   float
        case_result["u_inlet_safe"]  float
        case_result["open_fraction"] float
    """
    open_fraction = max(MIN_OPEN, 1.0 - mask_ctx["max_blockage"])
    u_inlet_safe = U_GAP_MAX * open_fraction
    delta_rho_safe = (3.0 / 2.0) * u_inlet_safe ** 2
    case_result["rho_in_case"] = min(sim_ctx["rho_in"], sim_ctx["rho_out"] + delta_rho_safe)
    case_result["u_inlet_safe"] = u_inlet_safe
    case_result["open_fraction"] = open_fraction
