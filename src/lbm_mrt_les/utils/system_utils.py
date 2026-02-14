import os
import shutil
import time
import random

def force_clean_cache():
    """
    [System] 強制清理 Taichi 快取
    解決 Windows 下 'Lock file failed' 的問題
    """
    # 這是 Taichi 在 Windows 的預設路徑，根據你的報錯訊息設定
    cache_path = "C:/taichi_cache/ticache"

    if os.path.exists(cache_path):
        try:
            print(f"[System] Cleaning Taichi cache at: {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)
            # 稍微等待一下 I/O 釋放，避免 Race Condition
            time.sleep(0.5)
        except Exception as e:
            print(f"[Warn] Failed to clean cache: {e}")
    else:
        print("[System] Cache directory not found (Clean start).")

def get_random_png_path(folder_path):
    """
    從指定資料夾中隨機挑選一張 PNG 圖片，並回傳完整路徑。
    """
    # 1. 檢查資料夾是否存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"[Error] Folder not found: {folder_path}")

    # 2. 列出所有檔案並篩選 .png (不分大小寫)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]

    # 3. 檢查是否有圖
    if not files:
        raise ValueError(f"[Error] No PNG files found in: {folder_path}")

    # 4. 隨機挑選
    selected_file = random.choice(files)

    # 5. 組合完整路徑 (跨平台相容)
    full_path = os.path.join(folder_path, selected_file)

    return full_path
