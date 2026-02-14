import yaml
import os
import sys
import re  # 用於解析檔名
import glob  # 用於搜尋檔案

# ==========================================
# 1. 物理參數設定
# ==========================================
# 當地圖數量超過 8 張時，第 9 張會回到 1000，第 10 張是 2000，依此類推
RE_LIST = [1000, 2000, 3000, 4000, 100, 200, 500, 800]
U_LB = 0.05  # 入口速度
SAVES_PER_PHYSICAL_SECOND = 25  # 目標採樣率

# 路徑設定
MASK_DIR = "src/generators/hybrid_maps"
OUTPUT_DIR = "src/configs/hyper_configs"


# ==========================================
# 2. 基礎模板 (Base Template)
# ==========================================
def get_base_config(re_value, nu_value, l_char, mask_path, filename_stem):
    """
    re_value: 雷諾數
    nu_value: 黏滯係數
    l_char: 特徵長度
    mask_path: 遮罩圖片的完整路徑
    filename_stem: 遮罩檔名(不含副檔名)，用於命名輸出資料夾
    """

    # 建立唯一的實驗名稱 (包含 Re 以便識別)
    sim_name = f"{filename_stem}_Re{re_value}"

    return {
        "simulation": {
            "name": sim_name,
            "nx": 4224,
            "ny": 1280,
            "nu": float(f"{nu_value:.6f}"),
            "characteristic_length": float(l_char),
            "compute_step_size": 100,  # 預設 會覆蓋
            "max_steps": 1000000,
            "warmup_steps": 2000,
            "smagorinsky_constant": 0.2,
            "ghost_moments_s": 1.05,
        },
        "outputs": {
            "enable_profiling": False,
            "gui": {
                "enable": True,  # 批量跑通常關閉 GUI
                "interval_steps": 100,  # 預設 會覆蓋
                "max_size": 1024,
                "show_zone_overlay": True,
                "gaussian_sigma": 1.0,
            },
            "video": {
                "enable": True,
                "interval_steps": 500,  # 預設 會覆蓋
                "fps": 30,
                "filename": f"video_{sim_name}.mp4",
            },
            "dataset": {
                "enable": True,
                "interval_steps": 100,  # 預設 會覆蓋
                "folder": f"output/Hyper/",
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
            "path": mask_path,
        },
        "domain_zones": {
            "sponge_y": 64,
            "sponge_x": 128,
            "buffer": 192,
            "inlet_buffer": 128,
        },
    }


# ==========================================
# 3. 主執行邏輯
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 搜尋所有 png 檔案
    mask_files = glob.glob(os.path.join(MASK_DIR, "*.png"))
    if not mask_files:
        print(f"Error: No PNG files found in {MASK_DIR}")
        return

    # [新增] 排序檔案，確保每次執行分配順序一致
    mask_files.sort()

    print(f"--- Found {len(mask_files)} maps. Assigning Re from list (Cyclic) ---")
    print(f"Re List: {RE_LIST}")

    valid_count = 0  # 用來計算有效生成的數量，並作為 Re 的索引

    # 2. 遍歷每一張地圖 (一對一模式)
    for mask_path in mask_files:
        # 取得檔名
        filename = os.path.basename(mask_path)
        filename_stem = os.path.splitext(filename)[0]

        # 提取特徵長度
        match = re.search(r"_L(\d+)", filename)
        if match:
            l_char = float(match.group(1))
        else:
            print(f"[Warning] No L found in {filename}, skipping.")
            continue

        # ==========================================
        # [核心修改] 決定這張圖要用哪個 Re
        # ==========================================
        # 使用餘數運算 (%) 來實現循環讀取 RE_LIST
        re_index = valid_count % len(RE_LIST)
        re_val = RE_LIST[re_index]

        # --- 物理計算 ---
        nu = (U_LB * l_char) / re_val
        steps_per_phys_sec = l_char / U_LB

        target_interval = int(steps_per_phys_sec / SAVES_PER_PHYSICAL_SECOND)
        if target_interval < 1:
            target_interval = 1

        # --- 生成 Config ---
        config_data = get_base_config(re_val, nu, l_char, mask_path, filename_stem)

        # 更新 Interval
        config_data["outputs"]["dataset"]["interval_steps"] = target_interval
        config_data["outputs"]["gui"]["interval_steps"] = target_interval
        config_data["outputs"]["video"]["interval_steps"] = target_interval
        config_data["simulation"]["compute_step_size"] = target_interval

        # --- 存檔 ---
        config_filename = f"cfg_{filename_stem}_Re{re_val}.yaml"
        full_config_path = os.path.join(OUTPUT_DIR, config_filename)

        with open(full_config_path, "w") as f:
            yaml.dump(config_data, f, sort_keys=False, default_flow_style=None)

        print(
            f"[{valid_count+1:03d}] {filename} -> Re={re_val} (Interval={target_interval}) | Saved: {config_filename}"
        )

        valid_count += 1

    print(f"\n[Done] Total {valid_count} config files generated in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
