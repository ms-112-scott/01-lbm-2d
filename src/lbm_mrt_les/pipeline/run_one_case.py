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
        # The runner is responsible for creating the directory
        os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
        recorder = Video_Recorder(
            video_output_path, width=viz.width, height=viz.height, fps=vid_cfg["fps"]
        )
        recorder.start()

    # 5. Initialize the HDF5 Dataset Writer
    writer = None
    if data_cfg["enable"] and h5_output_path:
        # The runner is responsible for creating the directory
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

        # 2. Determine simulation strategy
        u_inlet_mag = np.linalg.norm(solver.u_inlet)
        phys_suggested_steps, _ = utils.get_simulation_strategy(solver, u_inlet_mag)
        cfg_max_steps = config["simulation"]["max_steps"]
        max_steps = int(min(cfg_max_steps, phys_suggested_steps))

        # 3. Run the main simulation loop
        loop_metadata = ops.run_simulation_loop(
            config, solver, viz, recorder, gui, writer, max_steps
        )

        # 4. Collect metadata for summary
        metadata.update(loop_metadata)
        metadata["status"] = "Success"
        metadata["reason"] = "Completed successfully"
        
        # Add detailed lattice-level outputs for physical scaling
        metadata["reynolds_number_lattice_actual"] = solver.Re
        metadata["l_char_lattice_px"] = config["simulation"]["characteristic_length"]
        metadata["u_inlet_lattice_lu"] = config["boundary_condition"]["value"][0][0]
        metadata["nu_lattice_lu"] = config["simulation"]["nu"]
        metadata["nx"] = solver.nx
        metadata["ny"] = solver.ny
        metadata["total_steps_executed"] = max_steps

        # File info
        metadata["h5_file"] = os.path.basename(h5_output_path) if h5_output_path else "N/A"
        metadata["video_file"] = os.path.basename(video_output_path) if video_output_path else "N/A"

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Simulation Failed: {e}")
        traceback.print_exc()
        metadata["reason"] = str(e)

    finally:
        # 4. Cleanup resources
        print("\n[System] Cleaning up resources...")
        if recorder:
            recorder.stop()
        if writer:
            writer.close()
        if gui:
            gui.close()
        print("[System] Done.\n")

    return metadata

if __name__ == "__main__":
    # This block is now for testing a single case, not for batch processing.
    parser = argparse.ArgumentParser(description="Test runner for a single LBM case.")
    parser.add_argument("--config", required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--mask", required=True, help="Path to the obstacle mask PNG file.")
    args = parser.parse_args()

    # For testing, we can define some dummy output paths.
    test_h5_path = "outputs/test_run/test_case.h5"
    test_video_path = "outputs/test_run/test_case.mp4"

    main(args.config, args.mask, test_h5_path, test_video_path)
