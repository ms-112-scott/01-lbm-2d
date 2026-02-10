import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import apply_resize
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
import cv2


class Taichi_Gui_Viz:
    def __init__(
        self,
        width,
        height,
        viz_sigma=1.0,
        u_norm_max=0.15,
        vorticity_range=0.03,
        max_display_size=1024,
    ):
        self.width = width
        self.height = height
        self.viz_sigma = viz_sigma
        self.u_norm_max = u_norm_max
        self.vorticity_range = vorticity_range

        self._init_render_resources()

    def get_display_size(self):
        return self.display_shape

    def _init_render_resources(self):
        # ... (這部分保持不變) ...
        colors = [
            (1, 1, 0),
            (0.953, 0.490, 0.016),
            (0, 0, 0),
            (0.176, 0.976, 0.529),
            (0, 1, 1),
        ]
        self.vor_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "vorticity_cmap", colors
        )
        self.vor_cmap.set_bad(color="grey")

        self.vor_norm = matplotlib.colors.Normalize(
            vmin=-self.vorticity_range, vmax=self.vorticity_range
        )

    def process_frame(self, vel_raw, mask_np):
        # 1. 高斯模糊
        if self.viz_sigma > 0:
            vel_x = gaussian_filter(vel_raw[:, :, 0], sigma=self.viz_sigma)
            vel_y = gaussian_filter(vel_raw[:, :, 1], sigma=self.viz_sigma)
        else:
            vel_x = vel_raw[:, :, 0]
            vel_y = vel_raw[:, :, 1]

        # 2. 計算物理量
        vel_mag = np.sqrt(vel_x**2 + vel_y**2)
        ugrad = np.gradient(vel_x)
        vgrad = np.gradient(vel_y)
        vor = ugrad[1] - vgrad[0]

        # 3. 處理遮罩
        vor[mask_np > 0] = np.nan
        mask_indices = mask_np == 1
        obstacle_color = 0.5

        # 4. 著色
        vor_mapper = cm.ScalarMappable(norm=self.vor_norm, cmap=self.vor_cmap)
        vor_img = vor_mapper.to_rgba(vor)[:, :, :3]
        vor_img[mask_indices] = obstacle_color

        vel_norm_val = np.clip(vel_mag / self.u_norm_max, 0, 1)
        vel_img = cm.plasma(vel_norm_val)[:, :, :3]
        vel_img[mask_indices] = obstacle_color

        # 5. 拼接
        combined_img = np.concatenate((vel_img, vor_img), axis=1)
        # print(f"combined_img shape: {combined_img.shape}")
        resize = apply_resize(combined_img, self.height, self.width)
        # print(f"Resized image to: {resize.shape}")

        # 6. [呼叫外部函式] 執行縮放並回傳
        return resize
