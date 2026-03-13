"""
config_utils/steps_calc.py

衍生物理量與步數計算，結果寫入 case_result dict。

所有步數以 CTU（Convective Time Unit = L_char / u_conservative）為單位，
確保不同 mask / 不同 nu 的 case 在物理時間上具有一致性。
"""

import math
from .constants import U_STEP_FACTOR, CS


def fill_physics_and_steps(case_result: dict, mask_ctx: dict, sim_ctx: dict) -> None:
    """
    計算衍生物理量（u, Ma, Re, tau, dx_mm）與步數，結果寫入 case_result。

    Args:
        case_result : 須含 rho_in_case, nu_lb（由前序步驟寫入）
        mask_ctx    : 須含 l_char
        sim_ctx     : 須含 rho_out, warmup_passes, total_passes,
                      start_record_passes, saves_per_ctu, U_phys, nu_air

    Writes:
        case_result["u_bernoulli"]      float
        case_result["Ma"]               float
        case_result["Re"]               float
        case_result["tau"]              float
        case_result["dx_mm"]            float  (僅供顯示)
        case_result["steps_per_ctu"]    int
        case_result["warmup_steps"]     int
        case_result["max_steps"]        int
        case_result["start_record_step"] int
        case_result["interval"]         int
    """
    rho_in  = case_result["rho_in_case"]
    rho_out = sim_ctx["rho_out"]
    nu_lb   = case_result["nu_lb"]
    l_char  = mask_ctx["l_char"]
    U_phys  = sim_ctx["U_phys"]
    nu_air  = sim_ctx["nu_air"]

    delta_rho   = rho_in - rho_out
    u_bernoulli = math.sqrt((2.0 / 3.0) * delta_rho) if delta_rho > 1e-9 else 0.01
    Ma          = u_bernoulli / CS
    tau         = 3.0 * nu_lb + 0.5
    Re          = u_bernoulli * l_char / nu_lb

    vel_scale = U_phys / u_bernoulli if u_bernoulli > 1e-9 else 0
    dx_mm = (nu_air / (vel_scale * nu_lb)) * 1000 if (vel_scale * nu_lb) > 1e-9 else 0

    # 步數（以 CTU 倍率計）
    u_conservative  = u_bernoulli * U_STEP_FACTOR
    steps_per_ctu   = max(1, int(l_char / u_conservative))
    saves_per_ctu   = sim_ctx["saves_per_ctu"]

    case_result.update({
        "u_bernoulli":       u_bernoulli,
        "Ma":                Ma,
        "Re":                Re,
        "tau":               tau,
        "dx_mm":             dx_mm,
        "steps_per_ctu":     steps_per_ctu,
        "warmup_steps":      int(sim_ctx["warmup_passes"]       * steps_per_ctu),
        "max_steps":         int(sim_ctx["total_passes"]        * steps_per_ctu),
        "start_record_step": int(sim_ctx["start_record_passes"] * steps_per_ctu),
        "interval":          max(1, int(steps_per_ctu / saves_per_ctu)),
    })
