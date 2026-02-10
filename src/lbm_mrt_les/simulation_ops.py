import time
import numpy as np
import taichi as ti
from tqdm import tqdm
import utils


def check_stability(
    forces,
    velocity_field,
    step_count,
    v_threshold=0.5,
    f_threshold=1e6,
    warmup_step=1000,
):
    """
    [數值監控] 檢查模擬是否穩定 (數值熔斷機制)

    用於檢測 LBM 是否發生數值爆炸 (Explosion) 或出現 NaN/Inf。
    若檢測到異常，應立即停止模擬以保護數據集品質。

    Args:
        forces (list/array): [fx, fy] 當前總受力
        velocity_field (numpy array): 速度場 (nx, ny, 2)
        step_count (int): 當前模擬步數
        v_threshold (float): 速度熔斷閾值 (LBM 理論極限約 0.577, 保守設 0.5)
        f_threshold (float): 受力熔斷閾值 (防止受力計算溢位)
        warmup_step (int): 預熱步數 (在此期間忽略速度過大的檢查，允許初始震盪)

    Returns:
        (bool, str): (是否穩定, 錯誤訊息)
    """
    # --- 1. Force Check (NaN/Inf 檢查永遠執行) ---
    fx, fy = forces[0], forces[1]

    # 檢查 NaN (Not a Number) 或 Inf (Infinity)
    if np.isnan(fx) or np.isnan(fy) or np.isinf(fx) or np.isinf(fy):
        return False, f"Force becomes NaN/Inf at step {step_count} (Fx={fx}, Fy={fy})"

    # 檢查數值爆炸 (Explosion)
    # 受力過大通常代表壓力場求解發散
    if abs(fx) > f_threshold or abs(fy) > f_threshold:
        return (
            False,
            f"Force exploded (> {f_threshold:.1e}) at step {step_count} (Fx={fx:.2e}, Fy={fy:.2e})",
        )

    # --- 2. Velocity Check (預熱後執行) ---
    # 計算全場速度量值 (L2 Norm) -> shape: (nx, ny)
    v_norm = np.linalg.norm(velocity_field, axis=-1)
    max_v = np.max(v_norm)

    # NaN/Inf 檢查 (永遠執行)
    if np.isnan(max_v) or np.isinf(max_v):
        return False, f"Velocity field contains NaN/Inf at step {step_count}"

    # 數值大小檢查 (給予緩衝期 Warmup)
    if step_count > warmup_step:
        # LBM 的馬赫數限制：|u| < c_s (約 0.577)
        if max_v > v_threshold:
            return (
                False,
                f"Velocity {max_v:.4f} exceeded stability threshold ({v_threshold}) at step {step_count}",
            )

    return True, ""


def run_simulation_loop(config, solver, viz, recorder, gui, writer, max_steps):
    """
    [核心迴圈] LBM 模擬主流程

    已移除 History 紀錄，專注於 HDF5 數據集生成與即時監控。
    """
    # --- 1. 讀取設定與初始化 ---
    sim_cfg = config["simulation"]
    out_cfg = config["outputs"]
    zones = utils.get_zone_config(config)

    # 運算批次 (compute_step_size)
    compute_step_size = sim_cfg["compute_step_size"]

    # 取得各項輸出的間隔 (Interval)
    gui_interval = out_cfg["gui"]["interval_steps"]
    vid_interval = out_cfg["video"]["interval_steps"]
    data_interval = out_cfg["dataset"]["interval_steps"]

    current_steps = 0

    # 進度條
    pbar = tqdm(total=max_steps, unit="step")

    # [Profile] 初始化計時器
    timings = {
        "compute": 0.0,
        "force": 0.0,
        "field_fetch": 0.0,
        "stability": 0.0,
        "viz_proc": 0.0,
        "gui_draw": 0.0,
        "video_io": 0.0,
        "moment_fetch": 0.0,
        "hdf5_io": 0.0,
    }

    try:
        while current_steps < max_steps:
            t_loop_start = time.perf_counter()

            if gui and not gui.running:
                print("\n[Info] GUI closed by user.")
                break

            # =========================================================
            # 1. 核心運算 (LBM Compute)
            # =========================================================
            t0 = time.perf_counter()
            solver.run_step(compute_step_size)
            ti.sync()
            current_steps += compute_step_size
            t1 = time.perf_counter()
            timings["compute"] = (t1 - t0) * 1000

            # =========================================================
            # 2. 數據採樣 (Force Calculation)
            # =========================================================
            t0 = time.perf_counter()
            # 依然需要計算 Force 用於穩定性檢查與進度條顯示
            forces = solver.get_force()
            t1 = time.perf_counter()
            timings["force"] = (t1 - t0) * 1000

            # =========================================================
            # 3. 穩定性檢查 (Stability Check)
            # =========================================================
            t0 = time.perf_counter()
            vel_field, mask_data = solver.get_physical_fields()
            warmup = sim_cfg["warmup_steps"]
            is_stable, reason = check_stability(
                forces, vel_field, current_steps, warmup_step=warmup
            )

            if not is_stable and current_steps > warmup:
                print(f"\n\033[91m[CRITICAL] {reason}\033[0m")
                print("Aborting simulation loop early...")
                break

            t1 = time.perf_counter()
            timings["stability"] = (t1 - t0) * 1000

            # 更新 CLI 進度條
            pbar.set_postfix(Fx=f"{forces[0]:.3e}", Fy=f"{forces[1]:.3e}")
            pbar.update(compute_step_size)

            # =========================================================
            # 4. 視覺化處理 (Visualization Pipeline)
            # =========================================================
            t0 = time.perf_counter()
            need_viz = False
            is_gui_frame = out_cfg["gui"]["enable"] and (
                current_steps % gui_interval == 0
            )
            is_vid_frame = out_cfg["video"]["enable"] and (
                current_steps % vid_interval == 0
            )

            img = None
            if is_gui_frame or is_vid_frame:
                img = viz.process_frame(vel_field, mask_data)
                need_viz = True
            t1 = time.perf_counter()
            timings["viz_proc"] = (t1 - t0) * 1000 if need_viz else 0.0

            # --- 4.1 GUI ---
            t0 = time.perf_counter()
            if is_gui_frame and gui:
                gui.set_image(img)
                if out_cfg["gui"]["show_zone_overlay"]:
                    utils.draw_zone_overlay(gui, zones, y_offset=0.0)
                    utils.draw_zone_overlay(gui, zones, y_offset=0.5)
                gui.show()
            t1 = time.perf_counter()
            timings["gui_draw"] = (t1 - t0) * 1000 if is_gui_frame else 0.0

            # --- 4.2 Video ---
            t0 = time.perf_counter()
            if is_vid_frame and recorder:
                recorder.write_frame(np.transpose(img, (1, 0, 2)))
            t1 = time.perf_counter()
            timings["video_io"] = (t1 - t0) * 1000 if is_vid_frame else 0.0

            # =========================================================
            # 5. HDF5 數據集寫入
            # =========================================================
            t0_fetch = time.perf_counter()
            is_data_step = out_cfg["dataset"]["enable"] and (
                current_steps % data_interval == 0
            )

            if is_data_step and writer:
                moments_raw = solver.get_moments_numpy()
                t1_fetch = time.perf_counter()
                timings["moment_fetch"] = (t1_fetch - t0_fetch) * 1000

                t0_io = time.perf_counter()
                writer.append(moments_raw)
                t1_io = time.perf_counter()
                timings["hdf5_io"] = (t1_io - t0_io) * 1000
            else:
                timings["moment_fetch"] = 0.0
                timings["hdf5_io"] = 0.0

            # =========================================================
            # [Profile Report Console]
            # =========================================================
            total_time = (time.perf_counter() - t_loop_start) * 1000
            if (current_steps // compute_step_size) % 10 == 0 and config["outputs"][
                "enable_profiling"
            ]:
                print(f"\n[Profile] Step {current_steps} | Loop: {total_time:.1f}ms")
                print(f"  ├─ LBM Compute:  {timings['compute']:.1f} ms")
                if need_viz:
                    print(
                        f"  ├─ Vis Pipeline: {timings['viz_proc'] + timings['gui_draw'] + timings['video_io']:.1f} ms"
                    )
                if is_data_step:
                    print(
                        f"  └─ \033[92m[SAVE]\033[0m HDF5 I/O:    {timings['moment_fetch'] + timings['hdf5_io']:.1f} ms"
                    )
                else:
                    print(f"  └─ [SKIP] HDF5 I/O:    0.0 ms")

    except KeyboardInterrupt:
        print("\n[Info] Simulation interrupted by user.")

    pbar.close()

    # 只回傳 Metadata
    metadata = {
        "re_val": float(solver.Re),
        "u_max": float(np.linalg.norm(solver.u_inlet)),
        "D": float(config["simulation"]["characteristic_length"]),
        "data_interval": data_interval,
        "compute_step_size": compute_step_size,
    }

    return metadata
