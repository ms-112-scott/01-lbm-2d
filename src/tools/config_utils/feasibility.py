"""
config_utils/feasibility.py

LBM 模擬參數的物理可行性檢查。

三重保護（按嚴重程度排序）：
  1. Ma 上限：確保低馬赫假設成立
  2. tau 下限：確保 LBM 鬆弛格式數值穩定
  3. Re 上限：確保 2D 都市場景不崩潰

設計原則：
  - 純函式，無副作用
  - 回傳 (ok, reason) tuple，由呼叫端決定如何處理
  - 不直接呼叫 sys.exit 或 print
"""

import math
from .constants import CS, MA_LIMIT, TAU_MIN, RE_MAX


def check_feasibility(
    rho_in: float,
    rho_out: float,
    nu_lb: float,
    l_char: int,
) -> tuple[bool, str]:
    """
    檢查一組 (rho_in, rho_out, nu_lb, l_char) 是否物理可行。

    Args:
        rho_in  : 入口密度（格子單位）
        rho_out : 出口密度（格子單位）
        nu_lb   : 格子運動黏度
        l_char  : 特徵長度（像素）

    Returns:
        (True, "")          → 可行
        (False, reason_str) → 不可行，reason_str 說明原因與建議
    """
    delta_rho = rho_in - rho_out
    u_bernoulli = math.sqrt((2.0 / 3.0) * delta_rho) if delta_rho > 0 else 0.0
    Ma = u_bernoulli / CS
    tau = 3.0 * nu_lb + 0.5
    Re = u_bernoulli * l_char / nu_lb if nu_lb > 0 else float("inf")

    if Ma > MA_LIMIT:
        from .constants import CS2
        max_safe_drho = 1.5 * CS2 * MA_LIMIT ** 2
        return False, (
            f"Ma={Ma:.4f} > {MA_LIMIT}  "
            f"(u={u_bernoulli:.5f} lu/step, Δρ={delta_rho:.5f})。"
            f"建議 rho_in ≤ {rho_out + max_safe_drho:.5f}"
        )

    if tau < TAU_MIN:
        return False, (
            f"tau={tau:.4f} < {TAU_MIN}  (nu_lb={nu_lb:.5f})。"
            f"請將 nu_lb ≥ {(TAU_MIN - 0.5) / 3.0:.5f}"
        )

    if Re > RE_MAX:
        return False, (
            f"Re={Re:.0f} > RE_MAX={RE_MAX}  "
            f"(nu={nu_lb:.4f}, L_char={l_char}px, u={u_bernoulli:.5f})。"
            f"此組合在 2D LBM 都市場景數值不穩定，"
            f"請增大 nu_lb 或等待更小 L_char 的 mask"
        )

    return True, ""
