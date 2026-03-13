"""
config_utils/preview.py

Re 範圍預覽表列印（執行前）與完成統計（執行後）。

所有函式接受 sim_ctx dict，不需要展開引數。
"""

import math
from .constants import CS, MA_LIMIT, RE_MAX, U_GAP_MAX, TAU_MIN


def print_re_preview(sim_ctx: dict, l_char_range: tuple) -> None:
    """
    列印 nu_lb × L_char 的 Re 預覽表。

    Args:
        sim_ctx      : SimContext，須含 rho_in, rho_out, nu_lb_list, U_phys, nu_air
        l_char_range : (l_min, l_max)
    """
    rho_in  = sim_ctx["rho_in"]
    rho_out = sim_ctx["rho_out"]
    nu_list = sorted(sim_ctx["nu_lb_list"], reverse=True)
    U_phys  = sim_ctx["U_phys"]
    nu_air  = sim_ctx["nu_air"]

    delta_rho = rho_in - rho_out
    u_lb = math.sqrt(2 / 3 * delta_rho) if delta_rho > 0 else 0.01
    Ma   = u_lb / CS
    l_min, l_max = l_char_range
    l_samples = _sample_l(l_min, l_max)

    sep = "=" * 90
    print(sep)
    print("  Re 可達範圍預覽  (❌ 超過 RE_MAX 的組合會被自動 skip)")
    print(sep)
    print(
        f"  固定 rho_in={rho_in} → u_lb={u_lb:.5f}  Ma={Ma:.4f}  "
        f"{'✅ 安全' if Ma <= MA_LIMIT else '❌ 危險'}"
    )
    print(f"  RE_MAX={RE_MAX}  U_GAP_MAX={U_GAP_MAX}  TAU_MIN={TAU_MIN}")
    print(f"  物理常數：U_phys={U_phys} m/s,  nu_air={nu_air:.2e} m²/s")
    print(f"  mask L_char 預估範圍：{l_min} ~ {l_max} px\n")

    _print_table(nu_list, l_samples, u_lb, show_dx=False, U_phys=U_phys, nu_air=nu_air)
    _print_table(nu_list, l_samples, u_lb, show_dx=True,  U_phys=U_phys, nu_air=nu_air)

    print("  ⚠️  重要提示：")
    print("     - rho_in 不影響 Re！改 rho_in 只改 dx（物理解析度），Re 不變。")
    print("     - Re 多樣性的唯一正確旋鈕：nu_lb_list。")
    print(f"     - Re > {RE_MAX} 的組合在 2D 都市場景會崩潰，已自動 skip。")
    print(f"     - 阻塞率高時 rho_in 會自動降低（間隙速度 < {U_GAP_MAX}）。")
    print(sep + "\n")


def print_summary(sim_ctx: dict, success: int, skipped: int, l_min: int, l_max: int) -> None:
    """
    列印批次完成後的 Re 分佈統計。

    Args:
        sim_ctx  : SimContext，須含 rho_in, rho_out, nu_lb_list
        success  : 成功生成數
        skipped  : skip 數
        l_min/max: pre-scan 得到的 L_char 範圍
    """
    print("=" * 60)
    print(f"[Done] Generated {success} configs, skipped {skipped}.")
    if success == 0:
        print("=" * 60)
        return

    rho_in  = sim_ctx["rho_in"]
    rho_out = sim_ctx["rho_out"]
    nu_list = sim_ctx["nu_lb_list"]
    u_ref   = math.sqrt(2 / 3 * (rho_in - rho_out))

    print(f"\n[Re 分佈統計]  (rho_in={rho_in}，u_lb≈{u_ref:.5f})")
    print(f"  nu_lb 選項：{sorted(nu_list)}")
    print(
        f"\n  {'nu_lb':>8}  {'tau':>6}  "
        f"{'Re @ L_min={:d}px'.format(l_min):>20}  "
        f"{'Re @ L_max={:d}px'.format(l_max):>20}"
    )
    print("  " + "-" * 58)
    for nu in sorted(nu_list):
        tau = 3.0 * nu + 0.5
        re_min = u_ref * l_min / nu
        re_max = u_ref * l_max / nu
        print(
            f"  {nu:>8.4f}  {tau:>6.4f}  "
            f"{'❌' if re_min > RE_MAX else '  '}{re_min:>18.0f}  "
            f"{'❌' if re_max > RE_MAX else '  '}{re_max:>18.0f}"
        )
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# region 內部輔助
# ─────────────────────────────────────────────────────────────────────────────

def _sample_l(l_min: int, l_max: int, n: int = 5) -> list:
    if l_min == l_max:
        return [l_min]
    step = max(1, (l_max - l_min) // (n - 1))
    samples = list(range(l_min, l_max, step))
    if l_max not in samples:
        samples.append(l_max)
    return samples[:n]


def _print_table(nu_list, l_samples, u_lb, show_dx, U_phys, nu_air):
    if show_dx:
        print("  【物理 Re = 格子 Re（不變性）】  dx = nu_air / (U_phys/u_lb × nu_lb)")
        header = f"  {'nu_lb':>8}  {'dx (mm)':>9}"
    else:
        print("  【格子 Re】  Re_lb = u_lb × L_char / nu_lb   (❌ = 超過 RE_MAX)")
        header = f"  {'nu_lb':>8}  {'tau':>6}  {'穩定':>4}"

    for l in l_samples:
        header += f"  L={l:>4}px"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for nu in nu_list:
        tau = 3.0 * nu + 0.5
        re_vals = [u_lb * l / nu for l in l_samples]

        if show_dx:
            vel_scale = U_phys / u_lb if u_lb > 1e-9 else 0
            dx = nu_air / (vel_scale * nu) if (vel_scale * nu) > 1e-9 else 0
            row = f"  {nu:>8.4f}  {dx * 1000:>9.4f}"
        else:
            row = f"  {nu:>8.4f}  {tau:>6.4f}  {'✅' if tau >= TAU_MIN else '⚠️ '}"

        for re in re_vals:
            row += f"  {'❌' if re > RE_MAX else '  '}{re:>6.0f}"
        print(row)
    print()
