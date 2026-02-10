import os
import sys
import numpy as np
import taichi as ti
import argparse

# 確保可以 import 同目錄下的模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils
import simulation_ops as ops
from LBM2D_MRT_LES import LBM2D_MRT_LES
from Taichi_Gui_Viz import Taichi_Gui_Viz
from VideoRecorder import VideoRecorder

# 初始化 Taichi，建議使用 GPU
ti.init(arch=ti.gpu, device_memory_fraction=0.8)  # 預留顯存給 Torch 或其他程序


def init_simulation_env(config, out_dir):
    """
    初始化模擬環境：Solver, Mask, Visualizer, GUI, Recorder
    對應新的 Config 結構 (outputs 下分層)
    """
    # 1. Setup Mask
    search_dir = "src/GenMask/rect_masks"
    png_path = utils.get_random_png_path(search_dir)
    mask = utils.create_mask(config, png_path)

    # 2. GUI / Visualizer Size Calculation
    # 從 outputs.gui 讀取參數，若無則給默認值
    gui_cfg = config.get("outputs", {}).get("gui", {})

    gui_w, gui_h = utils.calcu_gui_size(
        raw_w=config["simulation"].get("nx", 1024),
        raw_h=config["simulation"].get("ny", 512),
        max_display_size=gui_cfg.get("max_size", 960),
    )
    print(f"[Setup] Viewport Size: {gui_w}x{gui_h}")

    # 3. Init Solver
    print("[Setup] Initializing LBM Solver...")
    solver = LBM2D_MRT_LES(config, mask_data=mask)
    solver.init()

    # 4. Init Visualization Logic (Color mapping, curl calculation etc.)
    viz = Taichi_Gui_Viz(
        gui_w,
        gui_h,
        viz_sigma=gui_cfg.get("gaussian_sigma", 1.0),
    )

    # 5. Init GUI Window (Optional)
    gui = None
    if gui_cfg.get("enable", True):
        gui = ti.GUI(
            "Taichi LBM - Strategy Monitor",
            res=(gui_w, gui_h),
            background_color=0xFFFFFF,
        )

    # 6. Init Video Recorder (Optional)
    recorder = None
    vid_cfg = config.get("outputs", {}).get("video", {})

    if vid_cfg.get("enable", False):
        try:
            # 檔名包含 Re 以便辨識
            fname = vid_cfg.get("filename", "simulation.mp4")
            v_path = os.path.join(out_dir, f"Re{int(solver.Re)}_{fname}")

            # 使用 viz.width/height 確保與 FFmpeg 兼容
            recorder = VideoRecorder(
                v_path, width=viz.width, height=viz.height, fps=vid_cfg.get("fps", 30)
            )
            recorder.start()
            print(f"[Setup] Video recording to: {v_path}")
        except Exception as e:
            print(f"[Warn] Video recorder failed to start: {e}")

    return solver, viz, gui, recorder


def main(config_path):
    print(f"\n{'='*50}")
    print(f"=== LBM Simulation Strategy: {os.path.basename(config_path)} ===")
    print(f"{'='*50}\n")

    # 1. Load Config
    if not os.path.exists(config_path):
        print(f"[Error] Config file not found: {config_path}")
        return

    config = utils.load_config(config_path)

    # Setup Output Directory
    # 優先使用 config 定義的路徑，若無則預設 output
    out_dir = config.get("outputs", {}).get("dataset", {}).get("folder", "./output")
    # 如果是 dataset 模式，可能路徑在 config['outputs']['dataset']['folder']
    # 這裡做一個通用 fallback
    if not out_dir:
        out_dir = "./output"

    os.makedirs(out_dir, exist_ok=True)

    # 2. Initialize Environment
    solver, viz, gui, recorder = init_simulation_env(config, out_dir)

    # 3. Determine Strategy (Duration)
    # 計算特徵速度 U_inlet (L2 norm)
    u_inlet_mag = np.linalg.norm(solver.u_inlet)

    # 如果 config 指定了 max_steps 則使用，否則自動計算 flow-through time
    flow_pass_steps, _ = utils.get_simulation_strategy(solver, u_inlet_mag)
    config_max_steps = config["simulation"]["max_steps"]
    max_steps = min(config_max_steps, flow_pass_steps)
    if flow_pass_steps > config_max_steps:
        strategy_source = "Config Defined"
    else:
        strategy_source = "Auto Calculated (Flow-through)"

    print(f"\n>>> Strategy Check:")
    print(f"    - Reynolds Number: {solver.Re:.2f}")
    print(f"    - Grid Resolution: {solver.nx} x {solver.ny}")
    print(f"    - Target Steps:    {max_steps} ({strategy_source})")
    print(f"    - Video Output:    {'Enabled' if recorder else 'Disabled'}")
    print(f"    - GUI Display:     {'Enabled' if gui else 'Disabled'}\n")

    # 4. Run Simulation Loop
    writer = None
    try:
        history, metadata = ops.run_simulation_loop(
            config, solver, viz, recorder, gui, writer, max_steps
        )

        # 5. Save & Post-Process
        data_path = os.path.join(out_dir, "simulation_data.npz")
        ops.save_simulation_data(data_path, history, metadata)

        # 僅在有紀錄歷史數據時才進行後處理
        if history:
            ops.perform_post_processing(out_dir, history, metadata)

        print(f"\n=== Simulation Completed Successfully. ===")
        print(f"=== Results saved in: {out_dir} ===\n")

    except KeyboardInterrupt:
        print("\n[Interrupt] Simulation stopped by user.")
    except Exception as e:
        print(f"\n[Error] Simulation failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 6. Cleanup Resources
        if recorder:
            recorder.stop()
        if gui:
            gui.close()


if __name__ == "__main__":
    # 使用 argparse 讓命令行調用更靈活
    parser = argparse.ArgumentParser(description="Taichi LBM Simulation Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/config_template.yaml",
        help="Path to the configuration YAML file",
    )

    args = parser.parse_args()

    main(args.config)
