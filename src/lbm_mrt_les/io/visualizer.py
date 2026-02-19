from ..utils import apply_resize
from ..utils.colorMapper import (
    apply_resize, 
    get_vorticity_mapper, 
    apply_color_mapping, 
    apply_velocity_coloring
)
import numpy as np
from scipy.ndimage import gaussian_filter
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

        # 初始化 Color Mapper
        self.vor_mapper = get_vorticity_mapper(self.vorticity_range)

    def process_frame(self, vel_raw, mask_np):
        # 1. 濾波處理
        if self.viz_sigma > 0:
            vel_x = gaussian_filter(vel_raw[:, :, 0], sigma=self.viz_sigma)
            vel_y = gaussian_filter(vel_raw[:, :, 1], sigma=self.viz_sigma)
        else:
            vel_x, vel_y = vel_raw[:, :, 0], vel_raw[:, :, 1]

        # 2. 計算物理量
        vel_mag = np.sqrt(vel_x**2 + vel_y**2)
        ugrad = np.gradient(vel_x)
        vgrad = np.gradient(vel_y)
        vor = ugrad[1] - vgrad[0]

        # 3. 處理遮罩與著色 (使用抽離後的 utils)
        # 處理渦度圖 (含 NaN 處理)
        vor_field = vor.copy()
        vor_field[mask_np > 0] = np.nan
        
        vor_img = apply_color_mapping(
            vor_field, self.vor_mapper, mask=mask_np, obstacle_color=0.5
        )

        # 處理速度圖
        vel_img = apply_velocity_coloring(
            vel_mag, self.u_norm_max, mask=mask_np, obstacle_color=0.5
        )

        # 4. 拼接與縮放
        combined_img = np.concatenate((vel_img, vor_img), axis=1)
        return apply_resize(combined_img, self.height, self.width)