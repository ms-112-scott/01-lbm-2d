import os
import sys
import yaml
import json
import numpy as np
from datetime import datetime


def load_config(path="config.yaml"):
    """讀取 YAML 設定檔"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading config: {e}")
        sys.exit(1)


def get_zone_config(config):
    nx = config["simulation"]["nx"]
    ny = config["simulation"]["ny"]
    zone_config = config["domain_zones"]

    sponge_in = zone_config["sponge_in"]
    sponge_out = zone_config["sponge_out"]
    sponge_top = zone_config["sponge_top"]
    sponge_bot = zone_config["sponge_bot"]
    buffer = zone_config["buffer"]

    roi_x_start = sponge_in + buffer
    roi_x_end = nx - sponge_out - buffer
    roi_y_start = sponge_bot + buffer
    roi_y_end = ny - sponge_top - buffer

    return {
        "sponge_in": sponge_in,
        "sponge_out": sponge_out,
        "sponge_top": sponge_top,
        "sponge_bot": sponge_bot,
        "roi_x_start": roi_x_start,
        "roi_x_end": roi_x_end,
        "roi_y_start": roi_y_start,
        "roi_y_end": roi_y_end,
        "nx": nx,
        "ny": ny,
    }


def save_case_metadata(json_path, case_id, metadata):
    """
    [IO] 將單一 Case 的 Metadata 更新到總表 JSON 中 (無 Class 版本)

    Args:
        json_path (str): 總表路徑 (e.g., './output/summary.json')
        case_id (str): 該 Case 的唯一 ID (通常是檔名)
        metadata (dict): 要寫入的數據字典
    """

    # -------------------------------------------------
    # 1. 準備 Numpy 轉換器 (閉包或是直接定義)
    # -------------------------------------------------
    def convert_numpy(obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")

    # -------------------------------------------------
    # 2. 讀取現有的 JSON (Read)
    # -------------------------------------------------
    full_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                full_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"[Warn] JSON {json_path} corrupted or empty. Creating new.")
            full_data = {}

    # -------------------------------------------------
    # 3. 更新數據 (Update)
    # -------------------------------------------------
    # 加上時間戳記
    metadata["_updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 使用 case_id 作為 Key (例如 'rect_001.png')
    full_data[case_id] = metadata

    # -------------------------------------------------
    # 4. 寫回檔案 (Write)
    # -------------------------------------------------
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            # 使用 default=convert_numpy 處理所有數值型別
            json.dump(full_data, f, default=convert_numpy, indent=4, ensure_ascii=False)
        print(f"[Metadata] Updated entry '{case_id}' in {os.path.basename(json_path)}")
    except Exception as e:
        print(f"[Error] Failed to save JSON metadata: {e}")
