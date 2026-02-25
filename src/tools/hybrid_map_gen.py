import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2
import yaml
import argparse
from config_utils import get_sampled_value
from map_gen.shapes import add_circle, add_rotated_rect, add_triangle
from map_gen.validators import check_sdf_validity, check_blockage_ratio


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

    def _generate_pinball_section(self):
        cfg = self.config["pinball"]
        shape_type = get_sampled_value(cfg["shape"])
        center_x = int(self.W * get_sampled_value(cfg["center_x_ratio"]))
        center_y = int(self.H * get_sampled_value(cfg["center_y_ratio"]))
        size = int(self.H * get_sampled_value(cfg["size_ratio"]))
        spacing = int(size * get_sampled_value(cfg["spacing_factor"]))

        positions = [
            (center_x - spacing, center_y),
            (center_x + spacing, center_y + spacing),
            (center_x + spacing, center_y - spacing),
        ]

        for cx, cy in positions:
            if shape_type == "circle":
                add_circle(self.grid, cx, cy, size)
            else:
                angle = get_sampled_value(cfg["rotation_angle"])
                if shape_type == "square":
                    side_length = size * 2
                    add_rotated_rect(self.grid, cx, cy, side_length, side_length, angle)
                elif shape_type == "triangle":
                    orientation = get_sampled_value(cfg["triangle_orientation"])
                    add_triangle(self.grid, cx, cy, size, angle, orientation)

    def _generate_tube_bank_section(self):
        cfg = self.config["tube_bank"]
        shape_type = get_sampled_value(cfg["shape"])
        layout_type = get_sampled_value(cfg["layout"])
        start_x = int(self.W * get_sampled_value(cfg["start_x_ratio"]))
        end_x = int(self.W * get_sampled_value(cfg["end_x_ratio"]))
        size = int(self.H * get_sampled_value(cfg["size_ratio"]))
        cols = get_sampled_value(cfg["num_cols"])
        rows = get_sampled_value(cfg["num_rows"])
        col_spacing = (end_x - start_x) // cols if cols > 0 else 0
        row_spacing = self.H // (rows + 1) if rows > 0 else 0
        jitter = cfg.get("jitter_amount", [0, 0])

        for c in range(cols):
            offset_y = 0
            if layout_type == "staggered" and c % 2 == 1:
                offset_y = row_spacing // 2
            
            for r_idx in range(rows):
                cx = start_x + c * col_spacing
                cy = row_spacing * (r_idx + 1) + offset_y
                jx = get_sampled_value(jitter)
                jy = get_sampled_value(jitter)
                
                final_cx, final_cy = cx + jx, cy + jy
                
                if not (size < final_cy < self.H - size):
                    continue

                if shape_type == "circle":
                    add_circle(self.grid, final_cx, final_cy, size)
                else:
                    angle = get_sampled_value(cfg["rotation_angle"])
                    if shape_type == "square":
                        side_length = size * 2
                        add_rotated_rect(self.grid, final_cx, final_cy, side_length, side_length, angle)
                    elif shape_type == "triangle":
                        orientation = get_sampled_value(cfg["triangle_orientation"])
                        add_triangle(self.grid, final_cx, final_cy, size, angle, orientation)

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

    def _generate_step_urban_section(self):
        cfg = self.config["step_urban"]
        step_x = int(self.W * get_sampled_value(cfg["step_start_ratio"]))
        step_h = int(self.H * get_sampled_value(cfg["step_height_ratio"]))
        step_w = int(self.W * get_sampled_value(cfg["step_width_ratio"]))
        # Use the helper for drawing a simple box (unrotated rectangle)
        add_rotated_rect(self.grid, step_x + step_w / 2, step_h / 2, step_w, step_h, 0)


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

            if check_sdf_validity(self.grid, box_points, min_dist) and \
               check_blockage_ratio(self.grid, box_points, max_blockage):
                cv2.drawContours(self.grid, [box_points], 0, 1, -1)
                placed_widths.append(w_val)
                
        max_placed_w = np.max(placed_widths) if placed_widths else 0
        max_feature_length = max(step_w, max_placed_w)
        return float(max_feature_length)

    def generate(self):
        self.reset()
        self._generate_pinball_section()
        self._generate_tube_bank_section()
        
        max_feature_length = self._generate_step_urban_section()
        
        buffer = self.config["validation"]["boundary_buffer"]
        self.grid[:, :buffer] = self.grid[:, -buffer:] = 0
        
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
    
    rho_in_list = master_config["physics_control"]["rho_in_list"]
    
    project_name = master_config["settings"]["project_name"]
    output_dir = os.path.join("SimCases", project_name, "masks")
    
    generator = HybridMapGenerator(map_gen_config)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "map_gen_config.json"), "w") as f:
        json.dump(map_gen_config, f, indent=4)

    num_maps_to_generate = 20
    if isinstance(rho_in_list, list) and len(rho_in_list) > 1:
        num_maps_to_generate = max(20, len(rho_in_list))

    print(f"--- Generating {num_maps_to_generate} maps... ---")
    for i in range(num_maps_to_generate):
        l_char = generator.generate()
        filename = os.path.join(output_dir, f"L{int(l_char)}_{i:04d}.png")
        generator.save_map(filename)
        print(f"  -> Characteristic Length (L): {l_char:.1f}")

    print(f"\n[Done] Generated {num_maps_to_generate} maps in '{output_dir}'.")