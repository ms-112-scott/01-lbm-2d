import sys

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
import yaml
import argparse
from config_utils import get_sampled_value


def load_yaml(path):
    """Loads a YAML configuration file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class HybridMapGenerator:
    def __init__(self, config):
        self.H = config["domain"]["height"]
        self.W = config["domain"]["width"]
        self.config = config
        self.grid = np.zeros((self.H, self.W), dtype=np.uint8)

    def reset(self):
        self.grid.fill(0)

    def _add_cylinder(self, cx, cy, r):
        y, x = np.ogrid[: self.H, : self.W]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r**2
        self.grid[mask] = 1

    def _add_box(self, x, y, w, h):
        y1, y2 = max(0, y), min(self.H, y + h)
        x1, x2 = max(0, x), min(self.W, x + w)
        self.grid[y1:y2, x1:x2] = 1

    def _generate_pinball_section(self):
        cfg = self.config["pinball"]
        center_x = int(self.W * get_sampled_value(cfg["center_x_ratio"]))
        center_y = int(self.H * get_sampled_value(cfg["center_y_ratio"]))
        r = int(self.H * get_sampled_value(cfg["radius_ratio"]))
        spacing = int(r * get_sampled_value(cfg["spacing_factor"]))
        self._add_cylinder(center_x - spacing, center_y, r)
        self._add_cylinder(center_x + spacing, center_y + spacing, r)
        self._add_cylinder(center_x + spacing, center_y - spacing, r)

    def _generate_tube_bank_section(self):
        cfg = self.config["tube_bank"]
        start_x = int(self.W * get_sampled_value(cfg["start_x_ratio"]))
        end_x = int(self.W * get_sampled_value(cfg["end_x_ratio"]))
        r = int(self.H * get_sampled_value(cfg["radius_ratio"]))
        cols = get_sampled_value(cfg["num_cols"])
        rows = get_sampled_value(cfg["num_rows"])
        col_spacing = (end_x - start_x) // cols
        row_spacing = self.H // (rows + 1)
        jitter_range = cfg.get("jitter_range", [0, 0])

        for c in range(cols):
            offset_y = (row_spacing // 2) if (c % 2 == 1) else 0
            for r_idx in range(rows):
                cx = start_x + c * col_spacing
                cy = row_spacing * (r_idx + 1) + offset_y
                jx = get_sampled_value(jitter_range)
                jy = get_sampled_value(jitter_range)
                if r < cy < self.H - r:
                    self._add_cylinder(cx + jx, cy + jy, r)

    def _get_random_rotated_rect(self, bounds, size_cfg, angle_range):
        # Using a simple diagonal for margin calculation
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

    def _check_sdf_validity(self, new_box, min_dist):
        if np.sum(self.grid) == 0: return True
        inv_grid = (1 - self.grid).astype(np.uint8)
        sdf = cv2.distanceTransform(inv_grid, cv2.DIST_L2, 5)
        new_mask = np.zeros_like(self.grid)
        cv2.drawContours(new_mask, [new_box], 0, 1, -1)
        covered_sdf = sdf[new_mask == 1]
        return len(covered_sdf) == 0 or np.min(covered_sdf) >= min_dist

    def _check_blockage_ratio(self, new_box, max_ratio):
        temp_grid = self.grid.copy()
        cv2.drawContours(temp_grid, [new_box], 0, 1, -1)
        blocked_height = np.sum(np.max(temp_grid, axis=1))
        return (blocked_height / self.H) <= max_ratio

    def _generate_step_urban_section(self):
        cfg = self.config["step_urban"]
        # 第一個特徵：初始的階梯方塊寬度
        step_x = int(self.W * get_sampled_value(cfg["step_start_ratio"]))
        step_h = int(self.H * get_sampled_value(cfg["step_height_ratio"]))
        step_w = int(self.W * get_sampled_value(cfg["step_width_ratio"]))
        self._add_box(step_x, 0, step_w, step_h)

        block_start_x = int(self.W * get_sampled_value(cfg["block_start_ratio"]))
        urban_bounds = {
            "min_x": max(block_start_x, step_x + step_w + 20),
            "max_x": int(self.W * get_sampled_value(cfg["block_end_ratio"])),
            "min_y": 0, "max_y": self.H
        }
        
        rect_count = get_sampled_value(cfg["rect_count"])
        angle_range = get_sampled_value(cfg["rotate_angle_max"])

        placed_widths = []
        for _ in range(cfg["max_attempts"]):
            if len(placed_widths) >= rect_count: break
            
            box_points, w_val = self._get_random_rotated_rect(
                urban_bounds, cfg["rect_size"], angle_range
            )
            
            min_dist = get_sampled_value(cfg["min_distance"])
            max_blockage = get_sampled_value(cfg["max_blockage_ratio"])

            if self._check_sdf_validity(box_points, min_dist) and \
               self._check_blockage_ratio(box_points, max_blockage):
                cv2.drawContours(self.grid, [box_points], 0, 1, -1)
                placed_widths.append(w_val)
                
        # 【修改點】：取生成的隨機方塊與初始階梯方塊中的「最大值」
        max_placed_w = np.max(placed_widths) if placed_widths else 0
        max_feature_length = max(step_w, max_placed_w)
        return float(max_feature_length)

    def generate(self):
        self.reset()
        self._generate_pinball_section()
        self._generate_tube_bank_section()
        
        # 【修改點】：獲取真正的最大特徵長度
        max_feature_length = self._generate_step_urban_section()
        
        buffer = self.config["validation"]["boundary_buffer"]
        self.grid[:, :buffer] = self.grid[:, -buffer:] = 0
        
        # 【修改點】：直接回傳浮點數最大寬度，移除 /20 * 20 的階梯化邏輯
        base_factor = 50
        max_feature_length = int(max_feature_length/base_factor) * base_factor
        return max_feature_length

    def save_map(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        output_grid = 1 - self.grid if self.config["output"]["invert_values"] else self.grid
        plt.imsave(filename, output_grid, cmap="gray", vmin=0, vmax=1)
        print(f"Saved: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hybrid maps using a master config.")
    parser.add_argument(
        "-c", "--config", default="master_config.yaml",
        help="Path to the master config file."
    )
    args = parser.parse_args()

    master_config = load_yaml(args.config)
    map_gen_config = master_config["map_generator"]
    
    re_list = master_config["physics_control"]["re_list"]
    
    project_name = master_config["settings"]["project_name"]
    output_dir = os.path.join("SimCases", project_name, "masks")
    
    generator = HybridMapGenerator(map_gen_config)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "map_gen_config.json"), "w") as f:
        json.dump(map_gen_config, f, indent=4)

    # If re_list is a range, we need to decide how many maps to generate.
    # We'll default to 10 or a specified number.
    num_maps_to_generate = 10 
    if isinstance(re_list, list) and len(re_list) != 2:
        num_maps_to_generate = len(re_list)

    print(f"--- Generating {num_maps_to_generate} maps... ---")
    for i in range(num_maps_to_generate):

        l_char = generator.generate()
        
        # 檔名保留整數以維持乾淨，但內部運算是精確的 float
        filename = os.path.join(output_dir, f"L{int(l_char)}_{i:04d}.png")
        
        generator.save_map(filename)
        print(f"  -> Characteristic Length (L): {l_char:.1f}")

    print(f"\n[Done] Generated {num_maps_to_generate} maps in '{output_dir}'.")