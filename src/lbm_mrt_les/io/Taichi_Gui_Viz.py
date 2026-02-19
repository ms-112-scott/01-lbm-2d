from ..utils import apply_resize
from ..utils.color_utils import *
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

    def process_frame(self, vel_raw: NDArray, mask_np: NDArray) -> NDArray:
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

        # 2. 針對不同物理場呼叫對應函數
        vel_img = colorize_velocity(
            vel_mag, 
            u_norm_max=self.u_norm_max, 
            mask=mask_np
        )

        vor_img = colorize_vorticity(
            vor, 
            vorticity_range=self.vorticity_range, 
            mask=mask_np
        )

        # 3. 拼接與縮放
        combined = np.concatenate((vel_img, vor_img), axis=1)
        return apply_resize(combined, self.height, self.width)