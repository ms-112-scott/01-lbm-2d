import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_dilation
import os
import random


class UrbanMapGenerator:
    """
    針對 LBM/NCA 訓練設計的都市微氣候流場地圖生成器。
    輸出：二值化矩陣 (0 = 流體 Fluid, 1 = 障礙物 Obstacle)
    """

    def __init__(self, height=128, width=256, min_gap=3):
        self.H = height
        self.W = width
        self.min_gap = min_gap  # LBM 數值穩定所需的最小間隙
        self.grid = np.zeros((self.H, self.W), dtype=np.int8)

    def reset(self):
        self.grid = np.zeros((self.H, self.W), dtype=np.int8)

    # ==========================================
    # 核心生成邏輯 (The 3 Blueprints)
    # ==========================================

    def generate_canyon_city(self, density=0.3):
        """
        藍圖 A：街道峽谷 (Street Canyon)
        特徵：筆直的長通道，模擬文丘里效應與十字路口剪切流。
        """
        self.reset()
        # 1. 建立基礎網格 (Block Layout)
        num_blocks_y = np.random.randint(2, 5)
        num_blocks_x = np.random.randint(4, 8)

        # 定義街道寬度 (隨機變化，但保證最小間隙)
        street_w_x = [
            np.random.randint(self.min_gap + 2, 15) for _ in range(num_blocks_x + 1)
        ]
        street_w_y = [
            np.random.randint(self.min_gap + 2, 15) for _ in range(num_blocks_y + 1)
        ]

        # 計算建築區塊大小
        total_street_x = sum(street_w_x)
        total_street_y = sum(street_w_y)
        block_w = (self.W - total_street_x) // num_blocks_x
        block_h = (self.H - total_street_y) // num_blocks_y

        # 2. 填充建築物
        current_y = street_w_y[0]
        for i in range(num_blocks_y):
            current_x = street_w_x[0]
            for j in range(num_blocks_x):
                # 隨機決定這個區塊是否要蓋房子 (創造空地)
                if np.random.random() < density * 1.5:
                    # 建築物內部微擾 (不一定是填滿的矩形)
                    bw = block_w + np.random.randint(-2, 3)
                    bh = block_h + np.random.randint(-2, 3)

                    # 邊界檢查
                    y_start = max(0, current_y)
                    y_end = min(self.H, current_y + bh)
                    x_start = max(0, current_x)
                    x_end = min(self.W, current_x + bw)

                    self.grid[y_start:y_end, x_start:x_end] = 1

                current_x += block_w + street_w_x[j + 1]
            current_y += block_h + street_w_y[i + 1]

    def generate_staggered_array(self, rows=5, cols=8):
        """
        藍圖 B：交錯陣列 (Staggered Array)
        特徵：規則中的變異，模擬住宅區或柱狀陣列的尾流干擾。
        """
        self.reset()
        cell_h = self.H // rows
        cell_w = self.W // cols

        for r in range(rows):
            for c in range(cols):
                # 交錯位移 (奇數行平移)
                offset_x = (cell_w // 2) if (r % 2 == 1) else 0

                # 基礎中心點
                center_y = r * cell_h + cell_h // 2
                center_x = c * cell_w + cell_w // 2 + offset_x

                # 隨機抖動 (Jitter)
                jitter_y = np.random.randint(-cell_h // 4, cell_h // 4)
                jitter_x = np.random.randint(-cell_w // 4, cell_w // 4)

                # 隨機尺寸
                obj_h = np.random.randint(cell_h // 4, cell_h // 2)
                obj_w = np.random.randint(cell_w // 4, cell_w // 2)

                # 繪製障礙物
                y1 = max(0, center_y + jitter_y - obj_h // 2)
                y2 = min(self.H, center_y + jitter_y + obj_h // 2)
                x1 = max(0, center_x + jitter_x - obj_w // 2)
                x2 = min(self.W, center_x + jitter_x + obj_w // 2)

                if x2 > x1 and y2 > y1:
                    self.grid[y1:y2, x1:x2] = 1

    def generate_plaza_highrise(self):
        """
        藍圖 C：孤島與廣場 (Plaza & High-rise)
        特徵：少數巨大障礙物 + 周圍細碎干擾，模擬大尺度渦旋脫落。
        """
        self.reset()

        # 1. 放置 1-3 個巨大地標 (High-rise)
        num_big = np.random.randint(1, 4)
        for _ in range(num_big):
            w = np.random.randint(20, 40)
            h = np.random.randint(20, 60)
            x = np.random.randint(self.W // 4, self.W * 3 // 4)  # 集中在中間
            y = np.random.randint(self.H // 4, self.H * 3 // 4)

            y1, y2 = max(0, y - h // 2), min(self.H, y + h // 2)
            x1, x2 = max(0, x - w // 2), min(self.W, x + w // 2)
            self.grid[y1:y2, x1:x2] = 1

        # 2. 散佈細碎障礙物 (Trees / Kiosks)
        num_small = np.random.randint(10, 30)
        for _ in range(num_small):
            s = np.random.randint(3, 8)  # 小尺寸
            x = np.random.randint(10, self.W - 10)
            y = np.random.randint(10, self.H - 10)

            # 檢查是否與大樓重疊 (簡單檢查)
            if self.grid[y, x] == 0:
                self.grid[y : y + s, x : x + s] = 1

    # ==========================================
    # 驗證與守門員 (Validation & Quality Control)
    # ==========================================

    def validate_map(self):
        """
        檢查地圖是否適合做 CFD 模擬。
        回傳: (is_valid, reason)
        """
        # 1. 阻塞率檢查 (Blockage Ratio)
        # 對於都市流場，我們希望 BR 在 5% ~ 30% 之間
        br = np.sum(self.grid) / (self.H * self.W)
        if br < 0.05:
            return False, f"Too Empty (BR={br:.2f})"
        if br > 0.35:
            return False, f"Too Dense (BR={br:.2f})"

        # 2. 連通性檢查 (Connectivity) - 這是最關鍵的
        # 我們檢查左邊界 (Inlet) 是否能連通到右邊界 (Outlet)
        fluid_mask = 1 - self.grid
        labeled_array, num_features = label(fluid_mask)

        # 取得左邊界和右邊界的所有 label ID
        left_labels = np.unique(labeled_array[:, 0])
        right_labels = np.unique(labeled_array[:, -1])

        # 移除 0 (背景/牆壁)
        left_labels = left_labels[left_labels != 0]
        right_labels = right_labels[right_labels != 0]

        # 檢查是否有交集 (即同一團流體橫跨左右)
        intersect = np.intersect1d(left_labels, right_labels)

        if len(intersect) == 0:
            return False, "Path Blocked (No Inlet-Outlet connection)"

        return True, "Valid"

    def clean_artifacts(self):
        """
        清理過小的間隙 (Optional)
        如果兩個障礙物距離 < min_gap，就把它們連起來，避免數值不穩定。
        (這裡使用簡單的膨脹後腐蝕邏輯，或直接保留現狀)
        """
        # 簡單實作：確保入口和出口沒有障礙物
        buffer = 5
        self.grid[:, :buffer] = 0  # Inlet buffer
        self.grid[:, -buffer:] = 0  # Outlet buffer

    # ==========================================
    # 主流程接口
    # ==========================================

    def generate_random_valid_map(self):
        """
        嘗試生成一張合法的地圖，直到成功為止 (Rejection Sampling)。
        """
        max_attempts = 50
        for i in range(max_attempts):
            mode = np.random.choice(["canyon", "staggered", "plaza"], p=[0.4, 0.4, 0.2])

            if mode == "canyon":
                self.generate_canyon_city()
            elif mode == "staggered":
                self.generate_staggered_array()
            else:
                self.generate_plaza_highrise()

            self.clean_artifacts()
            is_valid, reason = self.validate_map()

            if is_valid:
                return self.grid, mode

            # print(f"Attempt {i}: Failed ({reason})") # Debug use

        raise RuntimeError("Could not generate a valid map after multiple attempts.")


# ==========================================
# 測試與視覺化腳本
# ==========================================

if __name__ == "__main__":
    generator = UrbanMapGenerator(height=128, width=256)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    modes = ["canyon", "staggered", "plaza"]

    print("Generating sample maps...")

    for ax, mode in zip(axes, modes):
        # 強制指定模式進行測試
        valid = False
        while not valid:
            if mode == "canyon":
                generator.generate_canyon_city()
            elif mode == "staggered":
                generator.generate_staggered_array()
            elif mode == "plaza":
                generator.generate_plaza_highrise()

            generator.clean_artifacts()
            valid, reason = generator.validate_map()

        # 繪圖
        ax.imshow(generator.grid, cmap="gray_r")  # 0(白)=Fluid, 1(黑)=Wall
        ax.set_title(f"Mode: {mode.capitalize()} (Valid: {valid})")
        ax.axis("off")

        # 計算阻塞率
        br = np.sum(generator.grid) / generator.grid.size
        ax.text(5, 10, f"BR: {br:.2%}", color="red", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()
    print("Done! Copy this logic to your dataset pipeline.")
