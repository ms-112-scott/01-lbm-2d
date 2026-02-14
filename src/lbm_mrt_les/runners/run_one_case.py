import os
import sys
import numpy as np
import taichi as ti
import argparse
import traceback
import time

from .. import utils
from ..engine import simulation_ops as ops
from ..io.lbm_writer import AsyncLBMCaseWriter
from ..engine.lbm_solver import LBM2D_MRT_LES
from ..io.visualizer import Taichi_Gui_Viz
from ..io.video_recorder import VideoRecorder

# ---------------------------------------------------------
# [System] Taichi Initialization
# ---------------------------------------------------------
ti.init(arch=ti.gpu, device_memory_fraction=0.8)


def init_simulation_env(config, out_dir, png_path):
    """
    [Factory] 初始化模擬環境 (Strict Mode)
    嚴格讀取 Config，若鍵值不存在直接報錯 (KeyError)
    """

    # --- 1. Mask Injection ---
    # 嚴格讀取 mask 設定區塊
    mask_cfg = config["mask"]

    # 若有外部傳入路徑，強制覆寫 (Runtime Override)
    if png_path:
        mask_cfg["path"] = png_path
        print(f"[Setup] Overriding mask path: {png_path}")

    # 確保 mask 被正確建立
    mask = utils.create_mask(config, png_path)

    # 計算CL並且複寫
    config["simulation"]["characteristic_length"] = (
        utils.calculate_characteristic_length(mask)
    )

    # --- 2. GUI / Viewport Sizing ---
    # 嚴格讀取 simulation 與 outputs.gui 區塊
    sim_cfg = config["simulation"]
    gui_cfg = config["outputs"]["gui"]

    # 計算視窗大小 (強制要求 config 內有 max_size)
    gui_w, gui_h = utils.calcu_gui_size(
        raw_w=sim_cfg["nx"],
        raw_h=sim_cfg["ny"],
        max_display_size=gui_cfg["max_size"],
    )
    print(f"[Setup] Viewport Size: {gui_w}x{gui_h}")

    # --- 3. Solver Initialization ---
    print("[Setup] Initializing LBM Solver (MRT-LES)...")
    # Solver 內部也應確保讀取 config 時不缺漏，這裡傳入完整 config
    solver = LBM2D_MRT_LES(config, mask_data=mask)
    solver.init()

    # --- 4. Visualizer Setup ---
    # 嚴格讀取 visualization_gaussian_sigma (若 Config 沒寫，這裡會 Crash，這是預期的)
    viz = Taichi_Gui_Viz(
        gui_w, gui_h, viz_sigma=config["outputs"]["gui"]["gaussian_sigma"]
    )

    # --- 5. GUI Window ---
    gui = None
    if gui_cfg["enable"]:
        gui = ti.GUI(
            "Taichi LBM - Strategy Monitor",
            res=(gui_w, gui_h),
            background_color=0xFFFFFF,
        )

    # --- 6. Video Recorder ---
    recorder = None
    vid_cfg = config["outputs"]["video"]

    if vid_cfg["enable"]:
        try:
            # 嚴格讀取 fps 與 filename
            base_name = (
                os.path.splitext(os.path.basename(png_path))[0]
                if png_path
                else "default"
            )
            fname = f"Re{int(solver.Re)}_{base_name}.mp4"
            v_path = os.path.join(out_dir, "video", fname)
            os.makedirs(os.path.join(out_dir, "video"), exist_ok=True)
            recorder = VideoRecorder(
                v_path,
                width=viz.width,
                height=viz.height,
                fps=vid_cfg["fps"],  # 強制要求 yaml 定義 fps
            )
            recorder.start()
            print(f"[Setup] Video recording started: {v_path}")
        except Exception as e:
            print(f"[Warn] Video recorder failed to start: {e}")
            # 若 Config 正確但 FFmpeg 失敗，這裡還是要報錯嗎？
            # 策略建議：Config 錯導致 Crash 是好的，但 FFmpeg 環境問題可保留為 Warn
            raise e

    # --- 7. Dataset Writer ---
    writer = None
    data_cfg = config["outputs"]["dataset"]

    if data_cfg["enable"]:
        # 嚴格讀取 folder 設定，若無則依賴外部傳入的 out_dir
        # 但既然是嚴格模式，建議 YAML 必須寫清楚
        h5_folder = data_cfg["folder"]
        h5_folder = os.path.join(h5_folder, "h5_SimData")
        os.makedirs(h5_folder, exist_ok=True)

        case_name = sim_cfg["name"]
        h5_filename = f"{case_name}_{os.path.basename(png_path).replace('.png', '')}.h5"
        h5_path = os.path.join(h5_folder, h5_filename)

        print(f"[Setup] Initializing Async HDF5 Writer: {h5_path}")
        writer = AsyncLBMCaseWriter(h5_path, config, solver.nx, solver.ny)

    return solver, viz, gui, recorder, writer


def main(config_path, png_path):
    print(f"\n{'='*60}")
    print(f"=== LBM Simulation Strategy Start (Strict Mode) ===")
    print(f"=== Config: {os.path.basename(config_path)}")
    print(f"=== Mask:   {os.path.basename(png_path) if png_path else 'None'}")
    print(f"{'='*60}\n")

    # =========================================================
    # Phase 1: Configuration & Environment Setup
    # =========================================================
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[Error] Config file not found: {config_path}")

    config = utils.load_config(config_path)

    # [Strict] 輸出路徑必須在 YAML 中定義，或明確指定
    # 這裡我們假設 dataset folder 是主要輸出地
    out_dir = config["outputs"]["dataset"]["folder"]
    os.makedirs(out_dir, exist_ok=True)

    # 初始化 (若 Config 缺漏 key，將在此處拋出 KeyError)
    solver, viz, gui, recorder, writer = init_simulation_env(config, out_dir, png_path)

    # 定義總表路徑 (建議放在 dataset 資料夾下)
    summary_json_path = os.path.join(out_dir, "all_cases_summary.json")
    # 取檔名當作 ID
    case_id = os.path.basename(png_path)

    # =========================================================
    # Phase 2: Strategy Determination
    # =========================================================
    u_inlet_mag = np.linalg.norm(solver.u_inlet)

    # 計算物理建議
    phys_suggested_steps, strategy_desc = utils.get_simulation_strategy(
        solver, u_inlet_mag
    )

    # [Strict] 讀取 simulation.max_steps
    cfg_max_steps = config["simulation"]["max_steps"]

    # 策略決策
    max_steps = int(min(cfg_max_steps, phys_suggested_steps))

    if max_steps == int(cfg_max_steps):
        strategy_source = "Config Defined (Engineering Limit)"
    else:
        strategy_source = "Auto Calculated (Physical Flow-through)"

    print(f"\n>>> Strategy Decision:")
    print(f"    - Reynolds Number:   {solver.Re:.2f}")
    print(f"    - Resolution:        {solver.nx} x {solver.ny}")
    print(f"    - Physics Need:      {phys_suggested_steps} steps ({strategy_desc})")
    print(f"    - Config Limit:      {cfg_max_steps} steps")
    print(f"    - FINAL TARGET:      {max_steps} steps [{strategy_source}]")
    print(
        f"    - Outputs:           Video={bool(recorder)}, GUI={bool(gui)}, H5={bool(writer)}\n"
    )

    # =========================================================
    # Phase 3: Execution Loop
    # =========================================================
    try:
        # 執行主迴圈
        metadata = ops.run_simulation_loop(
            config, solver, viz, recorder, gui, writer, max_steps
        )

        # [Strategy] 存檔 Metadata 到總表  補充額外資訊
        if "status" not in metadata or not metadata["status"]:
            metadata["status"] = "Success"
            if "reason" not in metadata:
                metadata["reason"] = "Completed successfully (Default)"
        metadata["final_steps"] = max_steps
        metadata["config_used"] = config_path

        # 呼叫 utils 函式存檔
        utils.save_case_metadata(summary_json_path, case_id, metadata)

    except KeyError as e:
        print(f"\n[STRICT CONFIG ERROR] Missing key in configuration: {e}")
        print("Please check your YAML file structure against the code requirements.")
        traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Interrupt] Simulation stopped by user manually.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Simulation Failed: {e}")
        traceback.print_exc()

    finally:
        # =====================================================
        # Phase 4: Resource Cleanup
        # =====================================================
        print("\n[System] Cleaning up resources...")
        if recorder:
            recorder.stop()
        if writer:
            writer.close()
        if gui:
            gui.close()
        print("[System] Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Taichi LBM Simulation Runner (Strict)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/config_template.yaml",
        help="Path to the configuration YAML file",
    )

    parser.add_argument(
        "--mask_dir",
        type=str,
        default="src/generators/rect_masks",
        help="Directory containing PNG mask files",
    )

    args = parser.parse_args()

    if not os.path.exists(args.mask_dir):
        print(f"[Error] Mask directory not found: {args.mask_dir}")
        sys.exit(1)

    files = [f for f in os.listdir(args.mask_dir) if f.lower().endswith(".png")]
    files.sort()

    print(f"Found {len(files)} masks in {args.mask_dir}")

    # --- [Strategy] 執行迴圈 ---
    for i, filename in enumerate(files):
        print(f"\n{'#'*60}")
        print(f"### Processing Case {i+1}/{len(files)}: {filename} ###")
        print(f"{'#'*60}")

        # 1. 在每次模擬開始前，先炸掉 Cache
        utils.force_clean_cache()

        # 2. 組合完整路徑
        full_png_path = os.path.join(args.mask_dir, filename)

        # 3. 執行主程式
        # try:
        main(args.config, full_png_path)
        # except Exception as e:
        #     print(f"[Loop Error] Case {filename} failed: {e}")
        #     # 這裡可以選擇 continue 或 break，目前設為繼續跑下一個
        #     continue

        # 4. (Optional) 跑完後稍微休息，讓 GPU/CPU 降溫或釋放資源
        time.sleep(1.0)
