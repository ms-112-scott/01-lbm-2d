import taichi as ti
import numpy as np
import os
import sys
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
from lbm2d_mrt_les import LBM2D_MRT_LES
from visualization import LBMVisualizer
from VideoRecorder import VideoRecorder

ti.init(arch=ti.gpu)


def main(config_path):
    # ------------------------------------------------
    # 1. 初始化 Solver
    # ------------------------------------------------
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = utils.load_config(config_path)
    mask = utils.create_mask(config)

    solver = LBM2D_MRT_LES(config, mask_data=mask)
    solver.init()
    solver.Re = solver.check_re()

    # ------------------------------------------------
    # [策略監督] 計算 Flow-through Time (FTT)
    # ------------------------------------------------
    # 1. 取得入口速度 (假設邊界條件 index 0 是左側入口)
    # config["boundary_condition"]["value"] 格式為 [[ux, uy], ...]
    inlet_vel_vec = config["boundary_condition"]["value"][0]
    u_inlet = np.linalg.norm(inlet_vel_vec)

    # 防呆：如果速度為 0，避免除以零錯誤
    if u_inlet < 1e-6:
        print("[Warning] Inlet velocity is nearly zero. Defaulting to 100,000 steps.")
        max_simulation_steps = 100000
    else:
        # 2. 計算由左至右穿過一次所需的步數
        # T = L / U
        steps_per_pass = int(solver.nx / u_inlet)

        # 3. 設定目標 Pass 數
        target_passes = 5

        # 4. 總目標步數
        max_simulation_steps = steps_per_pass * target_passes

        print("=" * 40)
        print(f"   SIMULATION STRATEGY: {target_passes} PASSES")
        print("=" * 40)
        print(f"Domain Length (nx) : {solver.nx}")
        print(f"Inlet Velocity (U) : {u_inlet:.4f}")
        print(f"Steps per Pass     : {steps_per_pass}")
        print(f"Target Total Steps : {max_simulation_steps}")
        print("=" * 40)

    # ------------------------------------------------
    # 2. 初始化 Visualizer & GUI
    # ------------------------------------------------
    max_size = config.get("display", {}).get("max_size", 1024)
    viz = LBMVisualizer(
        nx=solver.nx,
        ny=solver.ny,
        viz_sigma=config["simulation"].get("visualization_gaussian_sigma", 1.0),
        max_display_size=max_size,
    )
    display_res = viz.get_display_size()

    gui = ti.GUI("Room Jet Flow", res=display_res)

    # 錄影設定 (保持不變)
    recorder = None
    try:
        if "foler_paths" in config and "output" in config["foler_paths"]:
            out_dir = config["foler_paths"]["output"]
        else:
            out_dir = config.get("output", {}).get("video_dir", "./output")

        os.makedirs(out_dir, exist_ok=True)
        video_path = os.path.join(
            out_dir, f"Re{int(solver.Re)}_nx{solver.nx}_{target_passes}Passes.mp4"
        )

        recorder = VideoRecorder(
            video_path, width=display_res[0], height=display_res[1], fps=30
        )
        recorder.start()
        print(f"[Video] Saving to: {video_path}")
    except Exception as e:
        print(f"[Warning] VideoRecorder init failed: {e}")
        recorder = None

    print("--- Simulation Started ---")

    # ------------------------------------------------
    # 3. 主迴圈 (使用 tqdm 監控進度)
    # ------------------------------------------------
    current_steps = 0

    # 使用 with 語法自動管理進度條的開啟與關閉
    # total: 總步數
    # unit: 單位名稱 (顯示為 x steps/s)
    # desc: 進度條左側的描述文字
    with tqdm(total=max_simulation_steps, unit="step", desc="LBM Progress") as pbar:

        # 迴圈條件：視窗開著 且 步數未達上限
        while gui.running and current_steps < max_simulation_steps:

            # A. 物理步進
            # 這裡計算了 steps_per_frame 步
            solver.run_step(solver.steps_per_frame)

            # 更新計數器
            current_steps += solver.steps_per_frame

            # --- 關鍵修改：手動更新 tqdm ---
            # 告訴進度條我們剛剛完成了多少步
            pbar.update(solver.steps_per_frame)

            # B. 獲取數據 & 渲染
            vel, mask_data = solver.get_physical_fields()
            img_gui = viz.process_frame(vel, mask_data)

            gui.set_image(img_gui)
            gui.show()

            if recorder:
                img_video = np.transpose(img_gui, (1, 0, 2))
                recorder.write_frame(img_video)

    # ------------------------------------------------
    # 4. 結束處理
    # ------------------------------------------------
    if recorder:
        recorder.stop()

    print(f"--- Simulation Completed ({target_passes} Passes) ---")

    # 選擇性：模擬結束後是否要自動關閉視窗？
    # 如果想保留最後一幀畫面給使用者看，可以把這行註解掉
    gui.close()


if __name__ == "__main__":
    configs = ["config_re100.yaml", "config_re2000.yaml", "config_re4000.yaml"]
    path_prefix = "src/lbm_mrt_les/configs"

    for config_name in configs:
        main(f"{path_prefix}/{config_name}")
