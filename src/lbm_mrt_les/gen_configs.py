import yaml
import os
import sys

# ---------------------------------------------------------
# [System] Path Setup & Imports
# ---------------------------------------------------------
# 確保能找到同目錄下的 utils.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utils

# ==========================================
# 1. 物理參數設定
# ==========================================
RE_LIST = [1000, 2000, 3000, 4000, 5000]
U_LB = 0.05  # 入口速度
L_CHAR = 160.0  # 特徵長度

# 設定目標：每「物理秒」存幾次檔？
SAVES_PER_PHYSICAL_SECOND = 10


# ==========================================
# 2. 基礎模板 (Base Template)
# ==========================================
def get_base_config(re_value, nu_value):
    return {
        "simulation": {
            "name": f"Re{re_value}_hybrid_urban",
            "nx": 4096,
            "ny": 1024,
            "nu": float(f"{nu_value:.6f}"),
            "characteristic_length": L_CHAR,
            "compute_step_size": 100,
            "max_steps": 300000,  # 稍微加長一點，確保能跑足夠的物理秒數
            "warmup_steps": 2000,
            "smagorinsky_constant": 0.2,
            "ghost_moments_s": 1.05,
        },
        "outputs": {
            "enable_profiling": False,
            "gui": {
                "enable": False,
                "interval_steps": 100,  # 這個可以是預設值，等等會被覆蓋
                "max_size": 1024,
                "show_zone_overlay": True,
                "gaussian_sigma": 1.0,
            },
            "video": {
                "enable": False,
                "interval_steps": 500,  # 預設值，稍後覆蓋
                "fps": 30,
                "filename": f"video_Re{re_value}.mp4",
            },
            "dataset": {
                "enable": True,
                "interval_steps": 100,  # 預設值，稍後覆蓋
                "folder": f"output/Hyper/Re{re_value}_data/",
                "compression": "lzf",
                "save_resolution": 512,
            },
        },
        "boundary_condition": {
            "type": [0, 2, 1, 2],
            "value": [
                [U_LB, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        },
        "mask": {
            "enable": True,
            "type": "png",
            "invert": False,
            "path": "src/GenMask/generated_maps_advanced/hybrid_adv_0000.png",
        },
        "domain_zones": {
            "sponge_y": 64,
            "sponge_x": 128,
            "buffer": 64,
            "inlet_buffer": 128,
        },
    }


# ==========================================
# 3. 生成 YAML 檔案
# ==========================================
output_dir = "src/configs/hyper_configs"
os.makedirs(output_dir, exist_ok=True)

print(f"--- Generating Configs for Re: {RE_LIST} ---")
print(f"Base Parameters: U_lb={U_LB}, L_char={L_CHAR}")
print(f"Target Sampling Rate: {SAVES_PER_PHYSICAL_SECOND} saves / physical second")

for re in RE_LIST:
    # 計算 nu
    nu = (U_LB * L_CHAR) / re

    # 1. 取得基礎 Config
    config_data = get_base_config(re, nu)

    # ---------------------------------------------------------
    # [關鍵修改] 使用 Utils 計算時間尺度並自動設定 Interval
    # ---------------------------------------------------------

    # 呼叫 utils 計算：1 物理秒 = 多少 steps
    # 注意：假設你的 utils 函數回傳 (float) steps_per_unit
    # 如果你的 utils 只會 print，你需要修改 utils 讓它 return 數值
    steps_per_phys_sec = utils.calculate_simulation_time_scale(config_data)

    # 若 utils 回傳 None 或 0 (防呆)，手動算一次: L / U
    if not steps_per_phys_sec:
        steps_per_phys_sec = L_CHAR / U_LB

    # 計算目標 interval: (Steps per Sec) / (Saves per Sec)
    target_interval = int(steps_per_phys_sec / SAVES_PER_PHYSICAL_SECOND)
    # print(f"steps_per_phys_sec {steps_per_phys_sec} target_interval {target_interval}")

    # 防呆：interval 至少要是 1，且建議是 compute_step_size 的倍數(選擇性)
    if target_interval < 1:
        target_interval = 1

    # 將計算結果寫回 Config
    config_data["outputs"]["dataset"]["interval_steps"] = target_interval
    config_data["outputs"]["gui"]["interval_steps"] = target_interval
    config_data["outputs"]["video"]["interval_steps"] = target_interval

    print(
        f"   [Re={re}] 1 Phys Sec = {int(steps_per_phys_sec)} steps. "
        f"Save Interval = {target_interval} steps (10Hz)."
    )

    # ---------------------------------------------------------

    # 存檔
    filename = os.path.join(output_dir, f"sim_config_Re{re}.yaml")
    with open(filename, "w") as f:
        yaml.dump(config_data, f, sort_keys=False, default_flow_style=None)

print("\nDone! YAML files are ready in 'configs/' folder.")
