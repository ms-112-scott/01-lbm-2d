import os
import numpy as np
from tqdm import tqdm
import utils


def check_stability(
    forces,
    velocity_field,
    step_count,
    v_threshold=0.5,  # [數值策略] 一般 LBM 極限是 0.577 (c_s * sqrt(3))，保守設 0.5
    f_threshold=1e6,  # [數值策略] 受力閾值不宜過大，1e10 太寬鬆，建議改 1e6
    warmup_step=1000,  # [應用策略] 給予 1000 步的緩衝期讓壓力波消散
):
    """
    檢查模擬是否穩定 (數值熔斷機制)

    Args:
        forces (list/array): [fx, fy] 當前受力
        velocity_field (numpy array): 速度場 (nx, ny, 2)
        step_count (int): 當前模擬步數
        v_threshold (float): 速度熔斷閾值 (LBM 理論極限約 0.58)
        f_threshold (float): 受力熔斷閾值
        warmup_step (int): 忽略速度檢查的初始步數 (避免初始震盪誤判)

    Returns:
        (bool, str): (是否穩定, 錯誤訊息)
    """
    # 1. Force Check (NaN/Inf 檢查永遠執行)
    fx, fy = forces[0], forces[1]

    # 檢查 NaN/Inf
    if np.isnan(fx) or np.isnan(fy) or np.isinf(fx) or np.isinf(fy):
        return False, f"Force becomes NaN/Inf at step {step_count} (Fx={fx}, Fy={fy})"

    # 檢查數值爆炸 (Explosion)
    # 注意：受力通常與 Re 和網格解析度有關，這裡設一個絕對巨大的值作為最後防線
    if abs(fx) > f_threshold or abs(fy) > f_threshold:
        return (
            False,
            f"Force exploded (> {f_threshold:.1e}) at step {step_count} (Fx={fx:.2e}, Fy={fy:.2e})",
        )

    # 2. Velocity Check (只在 warmup 之後嚴格檢查數值大小)
    # 計算全場速度量值 (L2 Norm)
    # velocity_field shape: (nx, ny, 2) -> norm -> (nx, ny)
    v_norm = np.linalg.norm(velocity_field, axis=-1)
    max_v = np.max(v_norm)

    # NaN/Inf 檢查永遠執行 (即使在 warmup 期間，出現 NaN 也是不對的)
    if np.isnan(max_v) or np.isinf(max_v):
        return False, f"Velocity field contains NaN/Inf at step {step_count}"

    # 數值大小檢查 (給予緩衝期)
    if step_count > warmup_step:
        if max_v > v_threshold:
            # [策略提示] LBM 的馬赫數 Ma = u / cs。當 u > 0.577 時 Ma > 1，算法必崩潰。
            return (
                False,
                f"Velocity {max_v:.4f} exceeded stability threshold ({v_threshold}) at step {step_count}",
            )

    return True, ""


def run_simulation_loop(config, solver, viz, recorder, gui, max_steps):
    """
    [New] 模擬主迴圈：負責執行運算、更新畫面、收集數據
    移入 ops 以便被不同的 main script 重用
    """
    # 1. 取得區域設定
    zones = utils.get_zone_config(config)

    # 容器初始化
    history = {"steps": [], "fx": [], "fy": []}
    current_steps = 0

    # 進度條
    pbar = tqdm(total=max_steps, unit="step")

    try:
        while current_steps < max_steps:
            # GUI 關閉檢查
            if gui and not gui.running:
                print("\n[Info] GUI closed by user.")
                break

            # 1. 核心運算
            solver.run_step(solver.steps_per_frame)
            current_steps += solver.steps_per_frame

            # 2. 數據採樣
            forces = solver.get_force()

            # 3. [熔斷檢查] & 視覺化數據準備
            # 只有在需要繪圖或檢查穩定性時才從 GPU 拉回大量數據 (優化效能)
            # 這裡假設每幀都檢查
            vel_field, mask_data = solver.get_physical_fields()
            warmup_step = config["simulation"].get("warmup_steps", 100)
            is_stable, reason = check_stability(
                forces, vel_field, current_steps, warmup_step=warmup_step
            )

            if not is_stable:
                print(f"\n\033[91m[CRITICAL] {reason}\033[0m")
                print("Aborting simulation loop early...")
                break

            # 4. 記錄數據
            history["fx"].append(forces[0])
            history["fy"].append(forces[1])
            history["steps"].append(current_steps)

            # 5. 更新顯示
            pbar.set_postfix(Fx=f"{forces[0]:.3e}", Fy=f"{forces[1]:.3e}")
            pbar.update(solver.steps_per_frame)

            # 6. 渲染與錄影
            img = viz.process_frame(vel_field, mask_data)

            if gui:
                gui.set_image(img)
                show_zone_overlay = config["display"].get("show_zone_overlay", True)
                if show_zone_overlay:
                    utils.draw_zone_overlay(
                        gui,
                        zones,
                        y_offset=0.0,  # <--- 如果框框跑到底下的圖，請改成 0.5
                    )
                    utils.draw_zone_overlay(
                        gui,
                        zones,
                        y_offset=0.5,  # <--- 如果框框跑到底下的圖，請改成 0.5
                    )

                gui.show()

            if recorder:
                recorder.write_frame(np.transpose(img, (1, 0, 2)))

    except KeyboardInterrupt:
        print("\n[Info] Simulation interrupted by user.")

    pbar.close()

    # 準備 Metadata
    metadata = {
        "re_val": float(solver.Re),
        "u_max": float(np.linalg.norm(solver.u_inlet)),
        "D": float(solver.config["simulation"].get("characteristic_length", 1.0)),
    }

    return history, metadata


def save_simulation_data(path, history, metadata):
    """儲存模擬數據 (.npz)"""
    print(f"[Save] Saving data to {path}...")
    np.savez(
        path,
        steps=np.array(history["steps"]),
        fx=np.array(history["fx"]),
        fy=np.array(history["fy"]),
        **metadata,
    )


def load_simulation_data(path, config_fallback=None):
    """讀取模擬數據 (.npz)"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    print(f"[Load] Loading data from {path}...")
    data = np.load(path)
    history = {"steps": data["steps"], "fx": data["fx"], "fy": data["fy"]}

    metadata = {}
    if "re_val" in data:
        metadata["re_val"] = float(data["re_val"])
        metadata["u_max"] = float(data["u_max"])
        metadata["D"] = float(data["D"])
    elif config_fallback:
        metadata = config_fallback
    else:
        metadata = {"re_val": 0.0, "u_max": 0.0, "D": 1.0}

    return history, metadata


def perform_post_processing(out_dir, history, metadata):
    """執行後處理計算與繪圖"""
    # [Fix] 將 List 轉換為 Numpy Array 以進行向量運算
    steps = np.array(history["steps"])
    fx = np.array(history["fx"])
    fy = np.array(history["fy"])

    if len(steps) == 0:
        print("[Error] No data to process.")
        return

    # 確保 metadata 存在
    re_val = metadata.get("re_val", 0.0)
    u_max = metadata.get("u_max", 0.1)
    D = metadata.get("D", 1.0)

    print("\n--- Analyzing Data ---")

    # 這裡傳入的 fx, fy 現在已經是 numpy array，不會再報 TypeError
    cd_arr, cl_arr, u_mean = utils.compute_coefficients(fx, fy, u_max, D)

    utils.plot_verification_results(out_dir, steps, cd_arr, cl_arr, re_val, u_mean, D)
