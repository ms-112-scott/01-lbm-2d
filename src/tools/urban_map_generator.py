import os
import json
import yaml
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 假設這些是你的既有工具組，若環境中沒有，請確保路徑正確
# 這裡保留你的 import 慣例
try:
    from config_utils import get_sampled_value
    from map_gen.shapes import add_circle, add_rotated_rect, add_triangle
    from map_gen.validators import check_sdf_validity, check_blockage_ratio
except ImportError:
    # 這裡提供 Mock 或基本實作以防報錯
    def get_sampled_value(val):
        if isinstance(val, list):
            return np.random.uniform(val[0], val[1])
        return val

    def add_rotated_rect(mask, cx, cy, w, h, angle):
        rect = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        cv2.drawContours(mask, [box], 0, 1, -1)

    def check_sdf_validity(mask, box, dist):
        return True

    def check_blockage_ratio(mask, box, ratio):
        return True


class UrbanMapGenerator:
    def __init__(self, config):
        self.H = config["domain"]["height"]
        self.W = config["domain"]["width"]
        self.config = config
        self.grid = np.zeros((self.H, self.W), dtype=np.uint8)

    def reset(self):
        self.grid.fill(0)

    def _get_random_rotated_rect(self, bounds, size_cfg, angle_range):
        max_w = size_cfg["w"][1] if isinstance(size_cfg["w"], list) else size_cfg["w"]
        max_h = size_cfg["h"][1] if isinstance(size_cfg["h"], list) else size_cfg["h"]
        max_diag = np.sqrt(max_w**2 + max_h**2)

        margin = int(max_diag / 2) + 2
        safe_x_min = bounds["min_x"] + margin
        safe_x_max = bounds["max_x"] - margin
        safe_y_min = bounds["min_y"] + margin
        safe_y_max = bounds["max_y"] - margin

        cx = get_sampled_value([safe_x_min, max(safe_x_min, safe_x_max)])
        cy = get_sampled_value([safe_y_min, max(safe_y_min, safe_y_max)])
        w = get_sampled_value(size_cfg["w"])
        h = get_sampled_value(size_cfg["h"])
        angle = get_sampled_value(angle_range)

        rect_def = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect_def)
        return np.int64(box), w

    def generate_pure_urban(self):
        """
        核心任務：生成僅包含後段建築區塊的 Mask。
        跳過 Pinball, Tube Bank 與 Step Section。
        """
        self.reset()
        cfg = self.config["step_urban"]

        # 定義建築區域邊界：完全無視前段 Step，從配置的 block_start 開始
        block_start_x = int(self.W * get_sampled_value(cfg["block_start_ratio"]))
        block_end_x = int(self.W * get_sampled_value(cfg["block_end_ratio"]))

        urban_bounds = {
            "min_x": block_start_x,
            "max_x": block_end_x,
            "min_y": 0,
            "max_y": self.H,
        }

        rect_count = get_sampled_value(cfg["rect_count"])
        angle_range = get_sampled_value(cfg["rotate_angle_max"])
        max_attempts = cfg.get("max_attempts", 200)

        placed_widths = []
        for _ in range(max_attempts):
            if len(placed_widths) >= rect_count:
                break

            box_points, w_val = self._get_random_rotated_rect(
                urban_bounds, cfg["rect_size"], angle_range
            )

            min_dist = get_sampled_value(cfg["min_distance"])
            max_blockage = get_sampled_value(cfg["max_blockage_ratio"])

            # 數值安全性檢查：重疊與阻塞率
            if check_sdf_validity(
                self.grid, box_points, min_dist
            ) and check_blockage_ratio(self.grid, box_points, max_blockage):
                cv2.drawContours(self.grid, [box_points], 0, 1, -1)
                placed_widths.append(w_val)

        # 邊界清洗：確保 Inlet/Outlet 緩衝區絕對乾淨，避免 Zou-He 邊界點噴射報錯
        buffer = self.config["validation"]["boundary_buffer"]
        self.grid[:, :buffer] = 0
        self.grid[:, -buffer:] = 0

        max_feature_length = np.max(placed_widths) if placed_widths else 1.0
        return float(max_feature_length)

    def save_map(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # 依照配置決定是否反轉顏色 (1為障礙物 vs 0為障礙物)
        output_grid = (
            1 - self.grid if self.config["output"]["invert_values"] else self.grid
        )
        plt.imsave(filename, output_grid, cmap="gray", vmin=0, vmax=1)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="urban_config.yaml")
    args = parser.parse_args()

    master_config = load_yaml(args.config)
    gen = UrbanMapGenerator(master_config["map_generator"])

    output_dir = os.path.join(
        "SimCases", master_config["settings"]["project_name"], "masks"
    )
    os.makedirs(output_dir, exist_ok=True)

    num_maps = 1
    print(f"--- 啟動純建築區塊生成模式：預計生成 {num_maps} 組 ---")

    for i in range(num_maps):
        l_char = gen.generate_pure_urban()
        filename = os.path.join(output_dir, f"Urban_{int(l_char)}_{i:04d}.png")
        gen.save_map(filename)
        print(f"  [Success] Map {i:04d} | 特徵長度 L: {l_char:.1f}")

    print(f"\n[任務完成] 檔案存放於: {output_dir}")
