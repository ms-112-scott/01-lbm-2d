import yaml
import os

# ==========================================
# 1. 物理參數設定
# ==========================================
RE_LIST = [1000, 2000, 3000, 4000, 5000]
U_LB = 0.05  # 入口速度 (需與 Boundary Condition 一致)
L_CHAR = 160.0  # 特徵長度 (與 simulation.characteristic_length 一致)


# ==========================================
# 2. 基礎模板 (Base Template)
#    (已修正 nx=4096 以匹配 HybridMapGenerator)
# ==========================================
def get_base_config(re_value, nu_value):
    return {
        "simulation": {
            "name": f"Re{re_value}_hybrid_urban",
            "nx": 4096,  # [修正] 必須匹配 HybridMapGenerator 的 width
            "ny": 1024,  # [修正] 必須匹配 HybridMapGenerator 的 height
            # --- 自動計算的物理參數 ---
            "nu": float(f"{nu_value:.6f}"),  # 控制 Re 的關鍵
            "characteristic_length": L_CHAR,
            "compute_step_size": 100,
            "max_steps": 200000,  # 視情況調整
            "warmup_steps": 2000,
            # --- 湍流模型 ---
            "smagorinsky_constant": 0.2,
            "ghost_moments_s": 1.05,
        },
        "outputs": {
            "enable_profiling": False,
            "gui": {
                "enable": False,  # 批量跑通常關閉 GUI，若要看請改 True
                "interval_steps": 100,
                "max_size": 1024,  # 視窗寬一點
                "show_zone_overlay": True,
                "gaussian_sigma": 1.0,
            },
            "video": {
                "enable": False,
                "interval_steps": 500,
                "fps": 30,
                "filename": f"video_Re{re_value}.mp4",
            },
            "dataset": {
                "enable": True,
                "interval_steps": 100,
                "folder": f"output/Hyper/Re{re_value}_data/",
                "compression": "lzf",
                "save_resolution": 512,
            },
        },
        "boundary_condition": {
            "type": [0, 2, 1, 2],  # Inlet, Free, Outlet, Free
            "value": [
                [U_LB, 0.0],  # Inlet Velocity
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        },
        "mask": {
            "enable": True,
            "type": "png",
            "invert": False,
            # [修正] 路徑指向 HybridMapGenerator 的輸出目錄
            # 這裡預設讀取第 0 張圖，實際跑的時候可能需要外部 Script 迴圈替換
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

for re in RE_LIST:
    # 計算 nu
    # nu = (U * L) / Re
    nu = (U_LB * L_CHAR) / re

    # 建立 Config 字典
    config_data = get_base_config(re, nu)

    # 存檔
    filename = os.path.join(output_dir, f"sim_config_Re{re}.yaml")
    with open(filename, "w") as f:
        yaml.dump(config_data, f, sort_keys=False, default_flow_style=None)

    print(f"Generated: {filename} | Re={re}, nu={nu:.6f}")

print("\nDone! YAML files are ready in 'configs/' folder.")
