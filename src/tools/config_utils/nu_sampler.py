"""
config_utils/nu_sampler.py

可行 nu 過濾與加權隨機取樣，結果寫入 case_result dict。

【取樣策略】
  1. 過濾：對所有 nu 候選做三重可行性檢查（Ma + tau + Re）
  2. 加權：weight = 1/Re，補償低 Re 選項天生較多的偏差
  3. 取樣：加權隨機取樣一個 nu
"""

import math
import random

from .feasibility import check_feasibility


def fill_nu_sample(case_result: dict, mask_ctx: dict, sim_ctx: dict) -> bool:
    """
    過濾可行 nu 並加權取樣，結果寫入 case_result。

    Args:
        case_result : 須含 rho_in_case（由 fill_blockage_adj 寫入）
        mask_ctx    : 須含 l_char
        sim_ctx     : 須含 nu_lb_list, rho_out

    Writes:
        case_result["nu_lb"]       float  選出的 nu 值
        case_result["nu_re_pairs"] list   [(nu, Re), ...] 可行清單

    Returns:
        True  → 取樣成功
        False → 無可行 nu，case_result 不寫入
    """
    rho_in_case = case_result["rho_in_case"]
    rho_out     = sim_ctx["rho_out"]
    l_char      = mask_ctx["l_char"]

    feasible = [
        nu for nu in sorted(sim_ctx["nu_lb_list"])
        if check_feasibility(rho_in_case, rho_out, nu, l_char)[0]
    ]

    if not feasible:
        _, reason = check_feasibility(rho_in_case, rho_out, max(sim_ctx["nu_lb_list"]), l_char)
        print(f"  [Skip] 無可用 nu（所有選項均不可行）。最大 nu 原因：{reason}\n")
        return False

    delta_rho = rho_in_case - rho_out
    u = math.sqrt(2.0 / 3.0 * delta_rho) if delta_rho > 1e-9 else 0.01
    re_values = [u * l_char / nu for nu in feasible]

    # 1/Re 加權取樣
    weights = [1.0 / re for re in re_values]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    r = random.random()
    cumulative = 0.0
    chosen = feasible[-1]
    for nu, prob in zip(feasible, probs):
        cumulative += prob
        if r <= cumulative:
            chosen = nu
            break

    case_result["nu_lb"] = chosen
    case_result["nu_re_pairs"] = list(zip(feasible, re_values))
    return True
