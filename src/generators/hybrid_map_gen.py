import numpy as np
import matplotlib.pyplot as plt
import os
import json
import cv2  # 需要安裝 opencv-python


class HybridMapGenerator:
    def __init__(self, config):
        self.H = config["domain"]["height"]
        self.W = config["domain"]["width"]
        self.config = config

        # 內部邏輯: 0 = Fluid, 1 = Obstacle
        self.grid = np.zeros((self.H, self.W), dtype=np.uint8)

    def reset(self):
        self.grid = np.zeros((self.H, self.W), dtype=np.uint8)

    def _add_cylinder(self, cx, cy, r):
        y, x = np.ogrid[: self.H, : self.W]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r**2
        self.grid[mask] = 1

    def _add_box(self, x, y, w, h):
        y1, y2 = max(0, y), min(self.H, y + h)
        x1, x2 = max(0, x), min(self.W, x + w)
        self.grid[y1:y2, x1:x2] = 1

    # ==========================================
    # Section I & II (保持原樣)
    # ==========================================

    def _generate_pinball_section(self):
        cfg = self.config["pinball"]
        center_x = int(self.W * cfg["center_x_ratio"])
        center_y = int(self.H * cfg["center_y_ratio"])
        r = int(self.H * cfg["radius_ratio"])
        spacing = int(r * cfg["spacing_factor"])

        self._add_cylinder(center_x - spacing, center_y, r)
        self._add_cylinder(center_x + spacing, center_y + spacing, r)
        self._add_cylinder(center_x + spacing, center_y - spacing, r)

    def _generate_tube_bank_section(self):
        cfg = self.config["tube_bank"]
        start_x = int(self.W * cfg["start_x_ratio"])
        end_x = int(self.W * cfg["end_x_ratio"])
        r = int(self.H * cfg["radius_ratio"])
        cols = cfg["num_cols"]
        rows = cfg["num_rows"]

        col_spacing = (end_x - start_x) // cols
        row_spacing = self.H // (rows + 1)

        for c in range(cols):
            offset_y = (row_spacing // 2) if (c % 2 == 1) else 0
            for r_idx in range(rows):
                cx = start_x + c * col_spacing
                cy = row_spacing * (r_idx + 1) + offset_y
                jitter = cfg["jitter_amount"]
                jx = np.random.randint(-jitter, jitter + 1)
                jy = np.random.randint(-jitter, jitter + 1)

                if r < cy < self.H - r:
                    self._add_cylinder(cx + jx, cy + jy, r)

    # ==========================================
    # Section III: Advanced Urban Generation
    # ==========================================

    def _get_random_rotated_rect(self, bounds, size_cfg, angle_max):
        # ... (保持原樣) ...
        max_diag = np.sqrt(size_cfg["max_w"] ** 2 + size_cfg["max_h"] ** 2)
        margin = int(max_diag / 2) + 2

        safe_x_min = bounds["min_x"] + margin
        safe_x_max = bounds["max_x"] - margin
        safe_y_min = bounds["min_y"] + margin
        safe_y_max = bounds["max_y"] - margin

        if safe_x_max <= safe_x_min:
            safe_x_max = safe_x_min + 1
        if safe_y_max <= safe_y_min:
            safe_y_max = safe_y_min + 1

        cx = np.random.randint(safe_x_min, safe_x_max)
        cy = np.random.randint(safe_y_min, safe_y_max)
        w = np.random.randint(size_cfg["min_w"], size_cfg["max_w"])
        h = np.random.randint(size_cfg["min_h"], size_cfg["max_h"])
        angle = np.random.uniform(-angle_max, angle_max)

        # [關鍵] 回傳 w 以便統計
        rect_def = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect_def)
        return np.int64(box), w

    def _check_sdf_validity(self, new_box, min_dist):
        if np.sum(self.grid) == 0:
            return True
        inv_grid = (1 - self.grid).astype(np.uint8)
        sdf = cv2.distanceTransform(inv_grid, cv2.DIST_L2, 5)
        new_mask = np.zeros_like(self.grid)
        cv2.drawContours(new_mask, [new_box], 0, 1, -1)
        covered_sdf = sdf[new_mask == 1]
        if len(covered_sdf) == 0:
            return True
        if np.min(covered_sdf) < min_dist:
            return False
        return True

    def _check_blockage_ratio(self, new_box, max_ratio):
        temp_grid = self.grid.copy()
        cv2.drawContours(temp_grid, [new_box], 0, 1, -1)
        y_occupancy = np.max(temp_grid, axis=1)
        blocked_height = np.sum(y_occupancy)
        current_ratio = blocked_height / self.H
        return current_ratio <= max_ratio

    def _generate_step_urban_section(self):
        """
        生成隨機方塊並回傳 [平均寬度]
        """
        cfg = self.config["step_urban"]

        # 1. 建立固定的大階梯 (Step) - 這個不計入平均寬度，因為它是邊界條件
        step_x = int(self.W * cfg["step_start_ratio"])
        step_h = int(self.H * cfg["step_height_ratio"])
        step_w = int(self.W * cfg["step_width_ratio"])
        self._add_box(step_x, 0, step_w, step_h)

        # 2. 定義區域
        block_start_x = int(self.W * cfg["block_start_ratio"])
        block_end_x = int(self.W * cfg["block_end_ratio"])
        step_end_x = step_x + step_w
        if block_start_x < step_end_x + 20:
            block_start_x = step_end_x + 20

        urban_bounds = {
            "min_x": block_start_x,
            "max_x": block_end_x,
            "min_y": 0,
            "max_y": self.H,
        }

        # 3. 放置方塊並統計寬度
        count = 0
        attempts = 0
        target_count = cfg["rect_count"]
        max_attempts = cfg["max_attempts"]

        # [新增] 用來記錄所有成功放置的方塊寬度
        placed_widths = []

        while count < target_count and attempts < max_attempts:
            attempts += 1

            # 注意：_get_random_rotated_rect 現在回傳 (box, w)
            box_points, w_val = self._get_random_rotated_rect(
                urban_bounds, cfg["rect_size"], cfg["rotate_angle_max"]
            )

            if not self._check_sdf_validity(box_points, cfg["min_distance"]):
                continue

            if not self._check_blockage_ratio(box_points, cfg["max_blockage_ratio"]):
                continue

            cv2.drawContours(self.grid, [box_points], 0, 1, -1)

            # [新增] 記錄寬度
            placed_widths.append(w_val)
            count += 1

        print(
            f"  [Urban] Placed {count}/{target_count} blocks. Widths: {placed_widths}"
        )

        # [新增] 計算並回傳平均寬度 (若是空的則回傳預設值 0)
        if not placed_widths:
            return 0.0
        return np.mean(placed_widths)

    # ==========================================
    # Main Generation & Saving
    # ==========================================

    def generate(self):
        """
        執行生成並回傳計算出的特徵長度 (L_char)
        """
        self.reset()
        self._generate_pinball_section()
        self._generate_tube_bank_section()

        # 取得 Urban 區域的平均寬度作為 L_char
        avg_urban_width = self._generate_step_urban_section()

        # 邊界清理
        buffer = self.config["validation"]["boundary_buffer"]
        self.grid[:, :buffer] = 0
        self.grid[:, -buffer:] = 0
        base = 20
        avg_urban_width = int(avg_urban_width / base) * base

        return avg_urban_width

    def save_map(self, filename):
        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.config["output"]["invert_values"]:
            output_grid = 1 - self.grid
        else:
            output_grid = self.grid

        plt.imsave(filename, output_grid, cmap="gray", vmin=0, vmax=1)
        print(f"Saved: {filename}")


# ==========================================
# Config (保持你的設定)
# ==========================================
HYBRID_CONFIG = {
    "domain": {"height": 1280, "width": 4224},
    "pinball": {
        "center_x_ratio": 0.12,
        "center_y_ratio": 0.5,
        "radius_ratio": 0.05,
        "spacing_factor": 2.5,
    },
    "tube_bank": {
        "start_x_ratio": 0.30,
        "end_x_ratio": 0.55,
        "radius_ratio": 0.03,
        "num_rows": 5,
        "num_cols": 6,
        "jitter_amount": 3,
    },
    "step_urban": {
        "step_start_ratio": 0.55,
        "step_width_ratio": 0.05,
        "step_height_ratio": 0.35,
        "block_start_ratio": 0.70,
        "block_end_ratio": 0.85,
        "rect_count": 8,
        "max_attempts": 200,
        "min_distance": 64,
        "max_blockage_ratio": 1.0,
        "rotate_angle_max": 5,
        "rect_size": {
            "min_w": 50,
            "max_w": 200,
            "min_h": 50,
            "max_h": 200,
        },
    },
    "validation": {"boundary_buffer": 138},
    "output": {
        "save_dir": "src/generators/hybrid_maps",
        "prefix": "hybrid_adv",
        "invert_values": True,
    },
}

if __name__ == "__main__":
    generator = HybridMapGenerator(HYBRID_CONFIG)

    os.makedirs(HYBRID_CONFIG["output"]["save_dir"], exist_ok=True)
    with open(
        os.path.join(HYBRID_CONFIG["output"]["save_dir"], "config.json"), "w"
    ) as f:
        json.dump(HYBRID_CONFIG, f, indent=4)

    for i in range(5):
        print(f"Generating {i}...")

        # [關鍵修改] 接收 generate 回傳的特徵長度
        l_char = generator.generate()

        # [關鍵修改] 將 L_char 寫入檔名 (格式: prefix_L{整數長度}_{編號}.png)
        # 例如: hybrid_adv_L125_0000.png
        filename = os.path.join(
            HYBRID_CONFIG["output"]["save_dir"],
            f"{HYBRID_CONFIG['output']['prefix']}_L{int(l_char)}_{i:04d}.png",
        )

        generator.save_map(filename)
        print(f" -> L_char (Avg Width): {l_char:.2f}")
