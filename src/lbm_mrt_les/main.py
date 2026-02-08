import os
import sys
import numpy as np
import taichi as ti

# 確保可以 import 同目錄下的模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils
import simulation_ops as ops
from lbm2d_mrt_les import LBM2D_MRT_LES
from visualization import LBMVisualizer
from VideoRecorder import VideoRecorder

ti.init(arch=ti.gpu)


def init_simulation_env(config, out_dir):
    """
    初始化模擬環境：Solver, Mask, Visualizer, GUI, Recorder
    """
    # 1. Create Mask
    mask = utils.create_mask(config)

    gui_w, gui_h = utils.calcu_gui_size(
        raw_w=config["simulation"].get("nx", 400),
        raw_h=config["simulation"].get("ny", 300),
        max_display_size=config["display"].get("max_size", 256),
    )  # 預計算 GUI 尺寸以供 Visualizer 使用

    print(f"calcu size: W{gui_w}xH{gui_h}")

    # 2. Init Solver
    solver = LBM2D_MRT_LES(config, mask_data=mask)
    solver.init()

    # 3. Init Visualization
    viz = LBMVisualizer(
        gui_w,
        gui_h,
        viz_sigma=config["simulation"].get("visualization_gaussian_sigma", 1.0),
    )

    # 4. Init GUI (Optional)
    gui = None
    if config["display"].get("gui_show", True):
        gui = ti.GUI("LBM Simulation", res=(gui_w, gui_h))

    # 5. Init Recorder (Optional)
    recorder = None
    try:
        # 錄影檔名包含 Re 數
        v_path = os.path.join(out_dir, f"Re{int(solver.Re)}_Sim.mp4")
        # 這裡使用 viz.width/height 確保與 FFmpeg 兼容 (如果你之前的 Visualizer code 有修好的話)
        recorder = VideoRecorder(v_path, width=viz.width, height=viz.height, fps=30)
        recorder.start()
    except Exception as e:
        print(f"[Warn] Video recorder failed to start: {e}")

    return solver, viz, gui, recorder


def main_run_sim_case(config_path):
    print(f"\n\n=== Starting Simulation Case: {config_path} ===")

    # 1. Load Config & Setup Paths
    config = utils.load_config(config_path)
    out_dir = config.get("foler_paths", {}).get("output", "./output")
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, "simulation_data.npz")

    # 2. Initialize Environment
    solver, viz, gui, recorder = init_simulation_env(config, out_dir)

    # 計算模擬策略 (總步數)
    u_max = np.linalg.norm(solver.u_inlet)
    max_steps, _ = utils.get_simulation_strategy(solver, u_max)

    # 3. Run Simulation Loop (Delegate to ops)
    print(f">>> Running Simulation (Target Steps: {max_steps})")
    history, metadata = ops.run_simulation_loop(
        config, solver, viz, recorder, gui, max_steps
    )

    # 4. Cleanup Resources
    if recorder:
        recorder.stop()
    if gui:
        gui.close()

    # 5. Save & Post-Process
    ops.save_simulation_data(data_path, history, metadata)
    ops.perform_post_processing(out_dir, history, metadata)

    print(f"=== Case Completed. Results in {out_dir} ===\n")


if __name__ == "__main__":
    # 指定設定檔路徑
    cfg_name = "config_re4000_room.yaml"
    path = os.path.join("src/configs", cfg_name)

    if os.path.exists(path):
        main_run_sim_case(path)
    else:
        print(f"Error: Config file not found at {path}")
