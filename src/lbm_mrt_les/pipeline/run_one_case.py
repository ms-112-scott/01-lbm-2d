import os
import sys
import numpy as np
import taichi as ti
import traceback
from typing import Dict, Any

from .. import utils
from ..core import simulation_ops as ops
from ..core.LBM2D_MRT_LES import LBM2D_MRT_LES
from ..io.LBM_Writer import AsyncLBMCaseWriter
from ..visualization.Taichi_Gui_Viz import Taichi_Gui_Viz
from ..io.Video_Recorder import Video_Recorder

ti.init(arch=ti.gpu, device_memory_fraction=0.8, log_level=ti.INFO)


def init_simulation_env(
    config: Dict[str, Any],
    mask_path: str,
    h5_output_path: str,
    video_output_path: str,
):
    """
    Initializes the simulation environment for a single case.
    This function now receives final, pre-constructed paths for its outputs.
    """
    sim_cfg = config["simulation"]
    gui_cfg = config["outputs"]["gui"]
    vid_cfg = config["outputs"]["video"]
    data_cfg = config["outputs"]["dataset"]

    # 1. Load and prepare the obstacle mask
    mask = utils.create_mask(config, mask_path)
    # The characteristic_length from the config is the single source of truth.
    # Do NOT recalculate it from the mask at runtime.

    # 2. Setup GUI and visualizer
    gui_w, gui_h = utils.calcu_gui_size(
        raw_w=sim_cfg["nx"],
        raw_h=sim_cfg["ny"],
        max_display_size=gui_cfg["max_size"],
    )
    viz = Taichi_Gui_Viz(gui_w, gui_h, viz_sigma=gui_cfg["gaussian_sigma"])
    gui = ti.GUI("Taichi LBM", res=(gui_w, gui_h)) if gui_cfg["enable"] else None

    # 3. Initialize the LBM solver
    solver = LBM2D_MRT_LES(config, mask_data=mask)
    solver.init()
    print(f"[Setup] Solver initialized for Re={solver.Re:.2f}")

    # 4. Initialize the Video Recorder
    recorder = None
    if vid_cfg["enable"] and video_output_path:
        os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
        recorder = Video_Recorder(
            video_output_path, width=viz.width, height=viz.height, fps=vid_cfg["fps"]
        )
        recorder.start()

    # 5. Initialize the HDF5 Dataset Writer
    writer = None
    if data_cfg["enable"] and h5_output_path:
        writer = AsyncLBMCaseWriter(
            h5_output_path, config, solver.nx, solver.ny, mask_data=mask
        )

    return solver, viz, gui, recorder, writer


def main(
    config_path: str,
    mask_path: str,
    h5_output_path: str,
    video_output_path: str,
) -> Dict[str, Any]:
    """
    Main function to run a single simulation case.
    It returns a metadata dictionary for the summary.
    """
    print(f"\n{'='*60}")
    print(f"=== Running LBM Simulation ===")
    print(f"    Config: {os.path.basename(config_path)}")
    print(f"    Mask:   {os.path.basename(mask_path)}")
    print(f"{'='*60}\n")

    utils.force_clean_cache()
    metadata = {"status": "Failed", "reason": "Unknown error"}
    solver, viz, gui, recorder, writer = None, None, None, None, None

    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = utils.load_config(config_path)

        # 1. Initialize environment
        solver, viz, gui, recorder, writer = init_simulation_env(
            config, mask_path, h5_output_path, video_output_path
        )

        # 2. max_steps 直接從 config 讀取，不再呼叫 get_simulation_strategy。
        #
        # 原本的邏輯：
        #   u_inlet_mag = np.linalg.norm(solver.u_inlet)   ← 讀到 dummy 0.05
        #   phys_steps  = nx / 0.05 × 5 = 409600           ← 硬算出錯誤步數
        #   max_steps   = min(cfg_max_steps, 409600)        ← 吃掉 config 設定
        #
        # solver.u_inlet 是從 boundary_condition.value[0] 讀的，
        # 而 Zou-He 壓力邊界的 value 只是 dummy [0.05, 0.0]，不代表真實速度。
        # 真實入口速度由 rho_in - rho_out 的壓差在執行時自動決定。
        # 因此 get_simulation_strategy 的結果完全不可信，應直接使用 config。
        #
        # 同時修正 solver 的 Re 顯示：用 Bernoulli 估算取代 dummy u_inlet
        sim_cfg = config["simulation"]
        rho_in = sim_cfg["rho_in"]
        rho_out = sim_cfg.get("rho_out", 1.0)
        nu = sim_cfg["nu"]
        l_char = sim_cfg["characteristic_length"]
        delta_rho = rho_in - rho_out
        u_estimated = (((2.0 / 3.0) * delta_rho) ** 0.5) if delta_rho > 0 else 0.01
        re_estimated = u_estimated * l_char / nu if nu > 0 else float("inf")

        max_steps = int(sim_cfg["max_steps"])

        print(f"[Strategy] max_steps={max_steps:,}  (from config, CTU-based)")
        print(
            f"[Strategy] u_estimated={u_estimated:.5f} lu/step  Re_estimated={re_estimated:.1f}"
        )
        print(
            f"[Strategy] warmup_steps={sim_cfg.get('warmup_steps', 0):,}  "
            f"start_record={config['outputs'].get('start_record_step', 0):,}"
        )

        # 3. Run the main simulation loop
        loop_metadata = ops.run_simulation_loop(
            config, solver, viz, recorder, gui, writer, max_steps
        )

        # 4. Collect metadata for summary
        metadata.update(loop_metadata)

        if metadata.get("status") == "Success":
            metadata["reason"] = "Completed successfully"

            # [FIX] 統一使用 inlet_u 這一個變數名，消除 measured_u NameError。
            #
            # 取入口面 (x=1 列) 的 y 方向平均 x 速度作為入口速度代表值。
            # 使用 x=1 而非 x=0，因為 x=0 是邊界節點，
            # collide_and_stream 的迴圈範圍是 1..nx-2，
            # 邊界節點的分佈函數在每步由 apply_bc 更新但不參與碰撞，
            # 取 x=1 的速度更能代表流入流場的真實速度。
            vel_np = solver.vel.to_numpy()  # shape: [nx, ny, 2]
            inlet_u = float(np.mean(vel_np[1, 1:-1, 0]))  # x 方向速度，排除上下牆

            l_char = config["simulation"]["characteristic_length"]
            nu = config["simulation"]["nu"]
            actual_re = (inlet_u * l_char) / nu if nu > 0 else float("inf")

            # metadata 只寫一次，不重複覆蓋
            metadata["u_inlet_lattice_lu"] = inlet_u
            metadata["reynolds_number_lattice_actual"] = actual_re
            metadata["l_char_lattice_px"] = l_char
            metadata["nu_lattice_lu"] = nu
            metadata["nx"] = solver.nx
            metadata["ny"] = solver.ny
            metadata["total_steps_executed"] = metadata.get("final_steps", 0)

            metadata["h5_file"] = (
                os.path.basename(h5_output_path) if h5_output_path else "N/A"
            )
            metadata["video_file"] = (
                os.path.basename(video_output_path) if video_output_path else "N/A"
            )

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Simulation Failed: {e}")
        traceback.print_exc()
        metadata["reason"] = str(e)

    finally:
        print("\n[System] Cleaning up resources...")
        if recorder:
            recorder.stop()
        if writer:
            try:
                if hasattr(writer, "writer") and metadata.get("status") == "Success":
                    h = writer.writer.target_h
                    w = writer.writer.target_w
                    c = writer.writer.channels
                    count = writer.writer.running_count
                    metadata["tensor_shape_static_mask"] = [2, h, w]
                    metadata["tensor_shape_turbulence"] = [count, c, h, w]
            except Exception as e:
                print(f"[Warning] Failed to read tensor shapes: {e}")
            writer.close()
        if gui:
            gui.close()
        print("[System] Done.\n")

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test runner for a single LBM case.")
    parser.add_argument(
        "--config", required=True, help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--mask", required=True, help="Path to the obstacle mask PNG file."
    )
    args = parser.parse_args()

    test_h5_path = "outputs/test_run/test_case.h5"
    test_video_path = "outputs/test_run/test_case.mp4"
    main(args.config, args.mask, test_h5_path, test_video_path)
