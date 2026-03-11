"""
src/tools/config_batch_gen.py

從 master_config.yaml 批量生成每個 case 的 YAML 設定檔。

設計原則：
  - mask 檔名僅作為排序依據（流水號），不從檔名傳遞任何物理資訊
  - L_char 直接從 PNG 像素計算（與 physics_utils.calculate_characteristic_length 一致）
  - Re 多樣性由 nu_lb_list 控制，每個 case 隨機取樣不同 nu 達到多種 Re
  - 執行前列印完整的 Re 可達範圍表，方便調參

【Re 的核心公式】
  Re = u_lb × L_char / nu_lb         (格子空間，dimensionless)
     = U_phys × L_phys / nu_air      (物理空間，完全相等)

  → rho_in 不影響 Re，只影響 dx（物理解析度）和 dt（時間步）
  → Re 的唯一有效旋鈕是 nu_lb 和 L_char
  → 想要 Re 多樣性，必須讓 nu_lb 跨越多個數量級

【master_config.yaml 建議新增】
  physics_control:
    nu_lb_list:
      - 0.050   # Re 低 (~130 @ L=80px)   層流
      - 0.020   # Re 中 (~327)             低湍流
      - 0.010   # Re 中高 (~653)           中湍流
      - 0.007   # Re 高 (~933)             高湍流
    rho_in: 1.010     # 固定單一值，不做多樣性
"""

import yaml
import os
import sys
import glob
import copy
import argparse
import math
import random

import cv2
import numpy as np
from config_utils import get_sampled_value


# ─────────────────────────────────────────────────────────────────────────────
# 物理常數與穩定性閾值
# ─────────────────────────────────────────────────────────────────────────────

CS2 = 1.0 / 3.0
CS = math.sqrt(CS2)  # ≈ 0.5774 lu/step

MA_LIMIT = 0.17  # 對應 u_max ≈ 0.098 lu/step，最大安全 Δρ ≈ 0.015
TAU_MIN = 0.52  # nu_min ≈ 0.0067
U_STEP_FACTOR = 0.6  # Bernoulli 高估修正：保守速度 = u_bernoulli × 0.6


# ─────────────────────────────────────────────────────────────────────────────
# Re 範圍預覽（執行前印出，方便調參）
# ─────────────────────────────────────────────────────────────────────────────


def print_re_range_preview(
    rho_in: float,
    rho_out: float,
    nu_lb_list: list,
    l_char_range: tuple,
    U_phys: float,
    nu_air: float,
):
    """
    列印所有 nu_lb × L_char 組合的 Re 範圍預覽表。
    讓使用者在跑批次之前確認 Re 分佈是否達到目標多樣性。

    Re 不變性說明：
      Re_lattice = u_lb × L_char(px) / nu_lb
      Re_physical = U_phys × L_phys(m) / nu_air
      兩者永遠相等，因為 dx 的定義保證了這一點。
    """
    delta_rho = rho_in - rho_out
    u_lb = math.sqrt(2 / 3 * delta_rho) if delta_rho > 0 else 0.01
    Ma = u_lb / CS
    l_min, l_max = l_char_range
    # 展示用的代表 L_char 值
    l_samples = []
    step = max(1, (l_max - l_min) // 4)
    v = l_min
    while v <= l_max:
        l_samples.append(v)
        v += step
    if l_max not in l_samples:
        l_samples.append(l_max)

    sep = "=" * 80

    print(sep)
    print("  Re 可達範圍預覽")
    print(sep)
    print(
        f"  固定 rho_in={rho_in} → u_lb={u_lb:.5f}  Ma={Ma:.4f}  "
        f"{'✅ 安全' if Ma <= MA_LIMIT else '❌ 危險'}"
    )
    print(f"  物理常數：U_phys={U_phys} m/s,  nu_air={nu_air:.2e} m²/s")
    print(f"  mask L_char 預估範圍：{l_min} ~ {l_max} px")
    print()

    # ── 格子 Re 表 ─────────────────────────────────────────────────────────
    print("  【格子 Re】  Re_lb = u_lb × L_char / nu_lb")
    header = f"  {'nu_lb':>8}  {'tau':>6}  {'穩定':>4}"
    for l in l_samples:
        header += f"  L={l:>4}px"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for nu_lb in nu_lb_list:
        tau = 3.0 * nu_lb + 0.5
        ok = "✅" if tau >= TAU_MIN else "⚠️ "
        row = f"  {nu_lb:>8.4f}  {tau:>6.4f}  {ok}"
        for l in l_samples:
            re = u_lb * l / nu_lb
            row += f"  {re:>8.0f}"
        print(row)

    print()

    # ── 物理 Re 表（與格子 Re 完全相同，但同時列印 dx 幫助理解物理意義）─
    print("  【物理 Re】  Re_phys = U_phys × L_phys / nu_air  （= 格子 Re，Re 不變性）")
    print("              dx = nu_air / (U_phys/u_lb × nu_lb)  每格對應的公尺數")
    header2 = f"  {'nu_lb':>8}  {'dx (mm)':>9}"
    for l in l_samples:
        header2 += f"  L={l:>4}px"
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for nu_lb in nu_lb_list:
        vel_scale = U_phys / u_lb if u_lb > 1e-9 else 0
        dx = nu_air / (vel_scale * nu_lb) if (vel_scale * nu_lb) > 1e-9 else 0
        row = f"  {nu_lb:>8.4f}  {dx*1000:>9.4f}"
        for l in l_samples:
            re_phys = u_lb * l / nu_lb  # Re 不變性，與格子 Re 完全相等
            row += f"  {re_phys:>8.0f}"
        print(row)

    print()

    # ── 重要提示 ────────────────────────────────────────────────────────────
    print("  ⚠️  重要提示：")
    print("     - rho_in 不影響 Re！改 rho_in 只改 dx（物理解析度），Re 不變。")
    print("     - 達到 Re 多樣性的唯一正確方式：在 nu_lb_list 中放多個不同 nu 值。")
    print("     - 建議 nu_lb 跨越 0.007 ~ 0.050，覆蓋層流到高湍流。")
    print(sep)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 從 PNG 計算 L_char（像素）
# ─────────────────────────────────────────────────────────────────────────────


def calc_l_char_from_png(png_path: str, invert: bool, nx: int, ny: int) -> int:
    """
    直接從 mask PNG 計算 L_char（像素數），不依賴檔名。

    定義與 physics_utils.calculate_characteristic_length 完全一致：
      Y 軸方向上，被任意固體像素佔據的總投影長度。

    讀取流程與 mask_utils._create_from_png 完全鏡像：
      灰階讀取 → resize(nx, ny) → threshold 127 → invert 旗標 → transpose
      invert=False → pixel < 127 是固體（黑色建築）
      invert=True  → pixel > 127 是固體（白色建築）
    """
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"無法讀取圖片：{png_path}")

    if img.shape != (ny, nx):
        img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_NEAREST)

    solid = (img > 127) if invert else (img < 127)
    solid = solid.T  # → [nx, ny]（Taichi field 慣例）
    y_occupied = np.any(solid, axis=0)  # Y 軸投影
    return max(1, int(np.sum(y_occupied)))


# ─────────────────────────────────────────────────────────────────────────────
# 物理可行性檢查
# ─────────────────────────────────────────────────────────────────────────────


def check_feasibility(rho_in: float, rho_out: float, nu_lb: float) -> tuple:
    """回傳 (ok: bool, reason: str)"""
    delta_rho = rho_in - rho_out
    u_bernoulli = math.sqrt((2.0 / 3.0) * delta_rho) if delta_rho > 0 else 0.0
    Ma = u_bernoulli / CS
    tau = 3.0 * nu_lb + 0.5

    if Ma > MA_LIMIT:
        max_safe_drho = 1.5 * CS2 * MA_LIMIT**2
        return False, (
            f"Ma={Ma:.4f} > {MA_LIMIT}  (u={u_bernoulli:.5f} lu/step, Δρ={delta_rho:.4f})。"
            f"建議 rho_in ≤ {rho_out + max_safe_drho:.4f}。"
        )
    if tau < TAU_MIN:
        return False, (
            f"tau={tau:.4f} < {TAU_MIN}  (nu_lb={nu_lb:.5f})。"
            f"請將 nu_lb 提高至 ≥ {(TAU_MIN - 0.5) / 3.0:.5f}。"
        )
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Config 組裝
# ─────────────────────────────────────────────────────────────────────────────


def generate_case_config(
    base_template: dict, run_params: dict, physical_constants: dict
) -> dict:
    config = copy.deepcopy(base_template)
    config["physical_constants"] = physical_constants

    config["simulation"]["name"] = run_params["sim_name"]
    config["simulation"]["nu"] = float(f'{run_params["nu_lb"]:.6f}')
    config["simulation"]["characteristic_length"] = float(run_params["l_char"])
    config["simulation"]["rho_in"] = float(run_params["rho_in"])
    config["simulation"]["rho_out"] = float(run_params["rho_out"])
    config["simulation"]["compute_step_size"] = run_params["interval"]
    config["simulation"]["warmup_steps"] = run_params["warmup_steps"]
    config["simulation"]["max_steps"] = run_params["max_steps"]
    config["simulation"]["smagorinsky_constant"] = 0.2

    config["outputs"]["project_name"] = run_params["project_name"]
    config["outputs"]["data_save_root"] = run_params["data_save_root"]
    config["outputs"]["target_rho_in"] = float(run_params["rho_in"])
    config["outputs"]["start_record_step"] = run_params["start_record_step"]
    config["outputs"]["gui"]["interval_steps"] = run_params["interval"]
    config["outputs"]["video"]["interval_steps"] = run_params["interval"]
    config["outputs"]["video"]["filename"] = f'{run_params["sim_name"]}.mp4'
    config["outputs"]["dataset"]["interval_steps"] = run_params["interval"]

    if "folder" in config["outputs"]["dataset"]:
        del config["outputs"]["dataset"]["folder"]

    # Zou-He 壓力邊界 dummy velocity，實際速度由 rho_in/rho_out 壓差自決
    config["boundary_condition"]["value"] = [[0.05, 0.0]] + [[0.0, 0.0]] * 3
    config["mask"]["path"] = run_params["mask_path"]

    return config


# ─────────────────────────────────────────────────────────────────────────────
# YAML helper
# ─────────────────────────────────────────────────────────────────────────────


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_sampled_nu(nu_lb_list) -> float:
    """
    從 nu_lb_list 取樣一個 nu 值。
    支援與 get_sampled_value 相同的格式（單值、list、range dict）。
    若 nu_lb_list 是普通 Python list，直接 random.choice。
    """
    if isinstance(nu_lb_list, list):
        return random.choice(nu_lb_list)
    # 若是 dict（range 格式），復用 get_sampled_value
    return get_sampled_value(nu_lb_list)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate LBM configs from a master YAML."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="master_config.yaml",
        help="Path to the master config file",
    )
    args = parser.parse_args()

    master_cfg = load_yaml(args.config)
    settings = master_cfg["settings"]
    physics = master_cfg["physics_control"]
    base_template = master_cfg["template"]
    physical_constants = master_cfg["physical_constants"]

    project_name = settings["project_name"]
    project_dir = os.path.join("SimCases", project_name)
    mask_dir = os.path.join(project_dir, "masks")
    output_dir = os.path.join(project_dir, "configs")
    data_save_root = os.path.join("outputs", project_name)

    # ── physics_control 參數讀取 ──────────────────────────────────────────
    rho_in = physics["rho_in"]  # 單一固定值（不再是 list）
    rho_out = physics["rho_out"]
    saves_per_ctu = physics["saves_per_physical_second"]
    w_passes = physics["warmup_passes"]
    t_passes = physics["total_passes"]
    s_passes = physics["start_record_passes"]

    # nu_lb_list：Re 多樣性的唯一旋鈕
    # 若 master_config 仍使用舊的單一 "nu" 欄位，自動包裝成 list 保持相容
    nu_lb_list = physics.get("nu_lb_list", None)
    if nu_lb_list is None:
        nu_single = physics.get("nu", 0.010)
        nu_lb_list = [nu_single]
        print(
            f"[Info] 未找到 nu_lb_list，使用單一 nu={nu_single}。"
            f"建議在 master_config.yaml 新增 nu_lb_list 以獲得 Re 多樣性。"
        )

    # 物理常數（用於 Re 預覽表）
    phys_const = master_cfg.get("physical_constants", {})
    U_phys_raw = phys_const.get("inlet_velocity_ms", 5.0)
    U_phys = U_phys_raw[0] if isinstance(U_phys_raw, list) else U_phys_raw
    nu_air = phys_const.get("kinematic_viscosity_air_m2_s", 1.5e-5)

    # passes 邏輯防呆
    if not (w_passes < s_passes < t_passes):
        print(
            f"[Error] passes 設定不合理：\n"
            f"  warmup={w_passes}, start_record={s_passes}, total={t_passes}\n"
            f"  必須滿足 warmup < start_record < total。\n"
            f"  若 start_record ≥ total，HDF5 內將完全沒有資料。"
        )
        sys.exit(1)

    mask_invert = base_template.get("mask", {}).get("invert", False)
    nx_val = base_template["simulation"]["nx"]
    ny_val = base_template["simulation"]["ny"]

    os.makedirs(output_dir, exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not mask_files:
        print(f"[Error] No PNG files found in: {mask_dir}")
        return

    # ── 執行前：掃描所有 mask 的 L_char，印出完整 Re 預覽表 ───────────────
    print(f"\n[Pre-scan] 讀取所有 {len(mask_files)} 個 mask 計算 L_char 範圍...")
    l_char_all = []
    for mp in mask_files:
        try:
            l = calc_l_char_from_png(mp, mask_invert, nx_val, ny_val)
            l_char_all.append(l)
        except Exception:
            pass

    l_min = min(l_char_all) if l_char_all else nx_val // 4
    l_max = max(l_char_all) if l_char_all else nx_val // 2
    print(
        f"[Pre-scan] L_char 範圍：{l_min} ~ {l_max} px  (共 {len(l_char_all)} 個有效 mask)"
    )

    # Re 範圍預覽表
    print_re_range_preview(
        rho_in=rho_in,
        rho_out=rho_out,
        nu_lb_list=sorted(nu_lb_list, reverse=True),
        l_char_range=(l_min, l_max),
        U_phys=U_phys,
        nu_air=nu_air,
    )

    # ── 批次生成 ──────────────────────────────────────────────────────────
    total = len(mask_files)
    success_count = 0
    skip_count = 0

    print(f"--- Found {total} masks. Generating configs... ---\n")

    for i, mask_path in enumerate(mask_files):
        seq_id = i + 1
        sim_name = f"case_{seq_id:04d}"

        print(f"[{seq_id:04d}/{total}]  {os.path.basename(mask_path)}")

        # ── nu_lb 取樣（Re 多樣性的來源）─────────────────────────────────
        nu_lb = get_sampled_nu(nu_lb_list)

        # ── 物理可行性檢查 ─────────────────────────────────────────────────
        ok, reason = check_feasibility(rho_in, rho_out, nu_lb)
        if not ok:
            print(f"  [Skip] {reason}\n")
            skip_count += 1
            continue

        # ── 從 PNG 像素計算 L_char ─────────────────────────────────────────
        try:
            l_char = calc_l_char_from_png(mask_path, mask_invert, nx_val, ny_val)
        except Exception as e:
            print(f"  [Skip] 讀取 mask 失敗：{e}\n")
            skip_count += 1
            continue

        # ── 速度與 Re 計算 ──────────────────────────────────────────────────
        delta_rho = rho_in - rho_out
        u_bernoulli = math.sqrt((2.0 / 3.0) * delta_rho) if delta_rho > 0 else 0.01
        u_for_steps = u_bernoulli * U_STEP_FACTOR  # 保守版本，步數偏大

        Re_lb = u_bernoulli * l_char / nu_lb  # = Re_phys（不變性）
        Ma = u_bernoulli / CS

        # dx for display only（物理解析度，不影響模擬）
        vel_scale = U_phys / u_bernoulli if u_bernoulli > 1e-9 else 0
        dx_mm = (
            (nu_air / (vel_scale * nu_lb)) * 1000 if (vel_scale * nu_lb) > 1e-9 else 0
        )

        # ── 步數計算（以 L_char / u_for_steps 的 CTU 為單位）──────────────
        steps_per_ctu = int(l_char / u_for_steps) if u_for_steps > 0 else 10000
        warmup_steps = w_passes * steps_per_ctu
        max_steps = t_passes * steps_per_ctu
        start_record_step = s_passes * steps_per_ctu
        target_interval = max(1, int(steps_per_ctu / saves_per_ctu))

        print(
            f"  nu_lb={nu_lb:.4f}  L_char={l_char}px  u={u_bernoulli:.5f}  "
            f"Ma={Ma:.4f}  Re={Re_lb:.0f}  dx={dx_mm:.4f}mm\n"
            f"  CTU={steps_per_ctu:,}  warmup={warmup_steps:,}  "
            f"max={max_steps:,}  start_rec={start_record_step:,}  interval={target_interval}"
        )

        # ── 組裝並寫出 YAML ────────────────────────────────────────────────
        run_params = {
            "sim_name": sim_name,
            "nu_lb": nu_lb,
            "l_char": l_char,
            "rho_in": rho_in,
            "rho_out": rho_out,
            "interval": target_interval,
            "mask_path": mask_path,
            "data_save_root": data_save_root,
            "project_name": project_name,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "start_record_step": start_record_step,
        }

        final_config = generate_case_config(
            base_template, run_params, physical_constants
        )

        nu_str = f"{nu_lb:.4f}".replace(".", "-")
        config_filename = f"{sim_name}_Nu{nu_str}.yaml"
        full_config_path = os.path.join(output_dir, config_filename)

        with open(full_config_path, "w", encoding="utf-8") as f:
            yaml.dump(final_config, f, sort_keys=False, default_flow_style=None)

        print(f"  -> Saved: {config_filename}  (Re≈{Re_lb:.0f})\n")
        success_count += 1

    # ── 完成統計 ──────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"[Done] Generated {success_count} configs, skipped {skip_count}.")
    if success_count > 0:
        # 印出最終 Re 分佈統計
        print(f"\n[Re 分佈統計]")
        print(f"  rho_in={rho_in}  →  u_lb={math.sqrt(2/3*(rho_in-rho_out)):.5f}")
        print(f"  nu_lb 選項：{sorted(nu_lb_list)}")
        u_ref = math.sqrt(2 / 3 * (rho_in - rho_out))
        print(
            f"\n  {'nu_lb':>8}  {'Re @ L_min={:d}px'.format(l_min):>18}  {'Re @ L_max={:d}px'.format(l_max):>18}"
        )
        print("  " + "-" * 48)
        for nu_v in sorted(nu_lb_list):
            re_min = u_ref * l_min / nu_v
            re_max = u_ref * l_max / nu_v
            print(f"  {nu_v:>8.4f}  {re_min:>18.0f}  {re_max:>18.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
