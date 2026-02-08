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

# ==========================================
# 全局控制開關
# ==========================================
# True:  執行完整 LBM 模擬 -> 存檔 -> 後處理
# False: 直接讀取上次的 .npz 檔 -> 後處理 (不需重新計算)
RUN_SIMULATION = False


# ==========================================
# Main Execution
# ==========================================


def main(config_path):
    # 1. 設定與路徑準備
    config = utils.load_config(config_path)

    # 輸出路徑 (提早定義以便讀取)
    out_dir = config.get("foler_paths", {}).get("output", "./output")
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, "simulation_data.npz")

    # 預先定義變數，確保後處理區塊有值可用
    steps_arr = None
    fx_arr = None
    fy_arr = None
    re_val = 0.0
    u_max = 0.0
    D = 73.0

    # ==========================================
    # 分支 A: 執行模擬 (Simulation Phase)
    # ==========================================
    if RUN_SIMULATION:
        print(">>> Mode: FULL SIMULATION")
        mask = utils.create_mask(config)

        solver = LBM2D_MRT_LES(config, mask_data=mask)
        solver.init()
        solver.Re = solver.check_re()

        # 紀錄物理參數
        re_val = solver.Re
        D = config["simulation"].get("characteristic_length", 73.0)
        u_inlet_vec = config["boundary_condition"]["value"][0]
        u_max = np.linalg.norm(u_inlet_vec)

        # 設定模擬步數
        max_steps = 100000  # 或使用 utils.get_simulation_strategy

        # 視覺化設置
        viz = LBMVisualizer(
            nx=solver.nx,
            ny=solver.ny,
            viz_sigma=config["simulation"].get("visualization_gaussian_sigma", 1.0),
        )
        gui = ti.GUI("DFG Benchmark Validator", res=viz.get_display_size())

        # 錄影
        recorder = None
        try:
            v_path = os.path.join(out_dir, f"Re{int(solver.Re)}_Sim.mp4")
            recorder = VideoRecorder(v_path, width=viz.width, height=viz.height, fps=30)
            recorder.start()
        except Exception as e:
            print(f"[Warn] Video init failed: {e}")

        # 數據容器
        history_fx = []
        history_fy = []
        history_steps = []

        print(f"--- Simulation Started (Re={solver.Re:.2f}) ---")

        # 主迴圈
        current_steps = 0
        with tqdm(total=max_steps, unit="step") as pbar:
            while gui.running and current_steps < max_steps:
                # A. 運算
                solver.run_step(solver.steps_per_frame)
                current_steps += solver.steps_per_frame

                # B. 採樣受力
                forces = solver.get_force()  # returns [fx, fy]
                history_fx.append(forces[0])
                history_fy.append(forces[1])
                history_steps.append(current_steps)

                # C. 顯示與進度條
                pbar.set_postfix(Fx=f"{forces[0]:.3e}", Fy=f"{forces[1]:.3e}")
                pbar.update(solver.steps_per_frame)

                # D. 渲染畫面
                vel, mask_data = solver.get_physical_fields()
                img = viz.process_frame(vel, mask_data)
                gui.set_image(img)
                gui.show()

                if recorder:
                    recorder.write_frame(np.transpose(img, (1, 0, 2)))

        # 收尾
        if recorder:
            recorder.stop()
        gui.close()

        # 轉換為 Numpy
        steps_arr = np.array(history_steps)
        fx_arr = np.array(history_fx)
        fy_arr = np.array(history_fy)

        # 儲存原始數據 (包含物理參數 metadata，這很重要！)
        print(f"\n[Save] Saving data to {data_path}...")
        np.savez(
            data_path,
            steps=steps_arr,
            fx=fx_arr,
            fy=fy_arr,
            # 額外儲存 metadata，讓後處理模式知道當時的物理條件
            re_val=re_val,
            u_max=u_max,
            D=D,
        )

    # ==========================================
    # 分支 B: 僅讀取數據 (Load Only Phase)
    # ==========================================
    else:
        print(">>> Mode: POST-PROCESSING ONLY (Loading Data)")
        if not os.path.exists(data_path):
            print(f"[Error] Data file not found: {data_path}")
            print("Please set RUN_SIMULATION = True to generate data first.")
            return

        # 讀取檔案
        data = np.load(data_path)
        steps_arr = data["steps"]
        fx_arr = data["fx"]
        fy_arr = data["fy"]

        # 嘗試讀取 metadata (兼容舊檔案的防呆)
        if "re_val" in data:
            re_val = float(data["re_val"])
            u_max = float(data["u_max"])
            D = float(data["D"])
            print(f"[Load] Metadata loaded: Re={re_val:.2f}, U_max={u_max}, D={D}")
        else:
            # 如果讀到舊版資料，嘗試從 config 抓取 (備用方案)
            print("[Warn] Metadata not found in .npz, estimating from Config...")
            u_inlet_vec = config["boundary_condition"]["value"][0]
            u_max = np.linalg.norm(u_inlet_vec)
            D = config["simulation"].get("characteristic_length", 73.0)
            # Re 只能估算或手動指定
            re_val = 100.0

    # ==========================================
    # 通用後處理 (Post-Processing Phase)
    # ==========================================
    print("\n--- Analyzing Data ---")

    # 無論數據來自模擬還是檔案，都重新計算係數
    # 這樣你可以隨時調整 compute_coefficients 的公式而不用重跑模擬
    cd_arr, cl_arr, u_mean = utils.compute_coefficients(fx_arr, fy_arr, u_max, D)

    # 繪製驗證圖
    utils.plot_verification_results(
        out_dir, steps_arr, cd_arr, cl_arr, re_val, u_mean, D
    )


if __name__ == "__main__":
    # 簡單的參數處理
    cfg_path = "src/lbm_mrt_les/configs/config_DFGBenchmark2D-1.yaml"
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]

    if os.path.exists(cfg_path):
        main(cfg_path)
    else:
        print(f"Error: Config file not found: {cfg_path}")
