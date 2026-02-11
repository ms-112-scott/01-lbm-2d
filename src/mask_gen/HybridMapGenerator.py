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
        """
        生成隨機旋轉矩形的頂點
        bounds: {'min_x', 'max_x', 'min_y', 'max_y'}
        """
        # 1. 計算安全邊距 (基於對角線)
        max_diag = np.sqrt(size_cfg["max_w"] ** 2 + size_cfg["max_h"] ** 2)
        margin = int(max_diag / 2) + 2

        safe_x_min = bounds["min_x"] + margin
        safe_x_max = bounds["max_x"] - margin
        safe_y_min = bounds["min_y"] + margin
        safe_y_max = bounds["max_y"] - margin

        # 防止邊界無效 (若區間太小，至少保證不報錯，雖然可能生不出東西)
        if safe_x_max <= safe_x_min:
            safe_x_max = safe_x_min + 1
        if safe_y_max <= safe_y_min:
            safe_y_max = safe_y_min + 1

        # 2. 隨機生成參數
        cx = np.random.randint(safe_x_min, safe_x_max)
        cy = np.random.randint(safe_y_min, safe_y_max)
        w = np.random.randint(size_cfg["min_w"], size_cfg["max_w"])
        h = np.random.randint(size_cfg["min_h"], size_cfg["max_h"])
        angle = np.random.uniform(-angle_max, angle_max)

        # 3. 獲取頂點 (cv2.boxPoints 返回 float32，需轉 int)
        rect_def = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect_def)
        return np.int64(box)

    def _check_sdf_validity(self, new_box, min_dist):
        """利用 Distance Transform 檢查新物體是否離現有障礙物太近"""
        if np.sum(self.grid) == 0:
            return True

        # 輸入圖：障礙物=0, 流體=1
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
        """檢查垂直截面阻塞率"""
        temp_grid = self.grid.copy()
        cv2.drawContours(temp_grid, [new_box], 0, 1, -1)

        y_occupancy = np.max(temp_grid, axis=1)
        blocked_height = np.sum(y_occupancy)
        current_ratio = blocked_height / self.H

        return current_ratio <= max_ratio

    def _generate_step_urban_section(self):
        cfg = self.config["step_urban"]

        # 1. 建立固定的大階梯 (Step)
        step_x = int(self.W * cfg["step_start_ratio"])
        step_h = int(self.H * cfg["step_height_ratio"])
        step_w = int(self.W * cfg["step_width_ratio"])
        self._add_box(step_x, 0, step_w, step_h)

        # 2. 定義 Urban 區域邊界 (使用 Config 中的新參數)
        # block_start_ratio: 控制左側留白 (階梯後到第一個方塊的距離)
        # block_end_ratio: 控制右側邊界

        block_start_x = int(self.W * cfg["block_start_ratio"])
        block_end_x = int(self.W * cfg["block_end_ratio"])

        # 安全檢查：確保 block_start 至少在 step 之後
        step_end_x = step_x + step_w
        if block_start_x < step_end_x + 20:
            # 如果設定太靠左，強制推到階梯後方 20px
            block_start_x = step_end_x + 20

        urban_bounds = {
            "min_x": block_start_x,
            "max_x": block_end_x,
            "min_y": 0,
            "max_y": self.H,
        }

        # 3. 嘗試放置旋轉方塊
        count = 0
        attempts = 0
        target_count = cfg["rect_count"]
        max_attempts = cfg["max_attempts"]

        while count < target_count and attempts < max_attempts:
            attempts += 1

            # 生成候選方塊
            box_points = self._get_random_rotated_rect(
                urban_bounds, cfg["rect_size"], cfg["rotate_angle_max"]
            )

            # 驗證 1: SDF 距離檢查
            if not self._check_sdf_validity(box_points, cfg["min_distance"]):
                continue

            # 驗證 2: 阻塞率檢查
            if not self._check_blockage_ratio(box_points, cfg["max_blockage_ratio"]):
                continue

            # 通過驗證，寫入 Grid
            cv2.drawContours(self.grid, [box_points], 0, 1, -1)
            count += 1

        print(
            f"  [Urban] Placed {count}/{target_count} blocks in range X[{block_start_x}:{block_end_x}]"
        )

    # ==========================================
    # Main Generation & Saving
    # ==========================================

    def generate(self):
        self.reset()
        self._generate_pinball_section()
        self._generate_tube_bank_section()
        self._generate_step_urban_section()

        # 邊界清理
        buffer = self.config["validation"]["boundary_buffer"]
        self.grid[:, :buffer] = 0
        self.grid[:, -buffer:] = 0

    def save_map(self, filename):
        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.config["output"]["invert_values"]:
            # 輸出：1=流體(白), 0=障礙物(黑)
            output_grid = 1 - self.grid
        else:
            # 輸出：0=流體(黑), 1=障礙物(白)
            output_grid = self.grid

        plt.imsave(filename, output_grid, cmap="gray", vmin=0, vmax=1)
        print(f"Saved: {filename}")


# ==========================================
# Updated Configuration
# ==========================================

HYBRID_CONFIG = {
    "domain": {"height": 1024, "width": 4096},
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
        # 1. 階梯設定 (BFS)
        "step_start_ratio": 0.55,  # 階梯開始
        "step_width_ratio": 0.05,  # 階梯厚度 -> 階梯結束於 0.60
        "step_height_ratio": 0.35,
        # 2. 方塊分佈設定 (控制留白)
        # 這裡設定 0.70，代表從 0.60 到 0.70 是空的 (大約 400px 的留白)
        "block_start_ratio": 0.70,
        "block_end_ratio": 0.85,
        # 3. 方塊生成參數
        "rect_count": 8,
        "max_attempts": 200,
        "min_distance": 20,
        "max_blockage_ratio": 1.0,
        "rotate_angle_max": 5,
        "rect_size": {
            "min_w": 50,
            "max_w": 200,
            "min_h": 50,
            "max_h": 200,
        },
    },
    "validation": {"boundary_buffer": 10},
    "output": {
        "save_dir": "src/GenMask/generated_maps_advanced",
        "prefix": "hybrid_adv",
        "invert_values": True,
    },
}

if __name__ == "__main__":
    generator = HybridMapGenerator(HYBRID_CONFIG)

    # 存下 Config 備份
    os.makedirs(HYBRID_CONFIG["output"]["save_dir"], exist_ok=True)
    with open(
        os.path.join(HYBRID_CONFIG["output"]["save_dir"], "config.json"), "w"
    ) as f:
        json.dump(HYBRID_CONFIG, f, indent=4)

    for i in range(5):
        print(f"Generating {i}...")
        generator.generate()
        filename = os.path.join(
            HYBRID_CONFIG["output"]["save_dir"],
            f"{HYBRID_CONFIG['output']['prefix']}_{i:04d}.png",
        )
        generator.save_map(filename)
