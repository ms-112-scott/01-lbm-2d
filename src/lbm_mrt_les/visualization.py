import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter
import cv2


class LBMVisualizer:
    """
    LBM 視覺化渲染器 (CPU/Matplotlib + OpenCV)
    """

    def __init__(
        self,
        nx,
        ny,
        viz_sigma=1.0,
        u_norm_max=0.15,
        vorticity_range=0.03,
        max_display_size=1024,
    ):
        """
        初始化渲染器資源，並計算顯示尺寸

        Args:
            nx, ny (int): 模擬網格解析度
            max_display_size (int): 輸出的最大邊長 (像素)
        """
        self.nx = nx
        self.ny = ny
        self.viz_sigma = viz_sigma
        self.u_norm_max = u_norm_max
        self.vorticity_range = vorticity_range
        self.max_display_size = max_display_size

        # --- 核心修改：預先計算顯示尺寸 ---
        # 原始組合後的形狀 (因為是左右拼接，假設 axis=1) -> (nx, ny * 2)
        # 註：Taichi 的 GUI 座標系通常第一維是寬(X)，第二維是高(Y)
        raw_w = nx
        raw_h = ny * 2

        self.display_shape = (raw_w, raw_h)  # 預設為原始尺寸
        self.need_resize = False

        if max_display_size and max_display_size > 0:
            max_side = max(raw_w, raw_h)
            if max_side > max_display_size:
                self.need_resize = True
                scale_ratio = max_display_size / max_side
                # 計算新尺寸 (保持整數)
                new_w = int(raw_w * scale_ratio)
                new_h = int(raw_h * scale_ratio)
                self.display_shape = (new_w, new_h)

                print(
                    f"[Visualizer] Resize enabled: ({raw_w}, {raw_h}) -> {self.display_shape}"
                )

        # 初始化顏色映射表
        self._init_render_resources()

    def get_display_size(self):
        """回傳計算好的顯示尺寸 (width, height)，供 GUI 初始化使用"""
        return self.display_shape

    def _init_render_resources(self):
        """建立靜態繪圖資源"""
        # 1. 渦度 Colormap
        colors = [
            (1, 1, 0),  # Yellow
            (0.953, 0.490, 0.016),  # Orange
            (0, 0, 0),  # Black
            (0.176, 0.976, 0.529),  # Green
            (0, 1, 1),  # Cyan
        ]
        self.vor_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "vorticity_cmap", colors
        )
        self.vor_cmap.set_bad(color="grey")

        self.vor_norm = matplotlib.colors.Normalize(
            vmin=-self.vorticity_range, vmax=self.vorticity_range
        )
        self.vel_cmap_name = "plasma"

    def process_frame(self, vel_raw, mask_np):
        """渲染並 Resize"""
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

        # 5. 拼接 (原始尺寸)
        combined_img = np.concatenate((vel_img, vor_img), axis=1)

        # 6. Resize (使用預計算的尺寸)
        if self.need_resize:
            # cv2.resize 接受的 dsize 是 (dim1_len, dim0_len) 即 (col, row)
            # 我們的 display_shape 是 (nx, ny*2) 即 (row, col) 在 numpy 意義上，
            # 但在 Taichi GUI 意義上是 (Width, Height)。
            # OpenCV 的 resize dsize 參數需要 (width, height)。
            # 在 numpy array中，shape是 (width, height, 3) (因為 taichi field 是 x,y)
            # 所以我們要 resize 到 (self.display_shape[1], self.display_shape[0])
            # 這是因為 cv2 dsize 是 (columns, rows)。

            # 這裡有點繞，因為 Taichi 的 Field 形狀定義是 (X, Y)，對應 Numpy 是 (Dim0, Dim1)。
            # 若 X 是寬，Y 是高。Numpy shape = (Width, Height, 3)。
            # OpenCV 視角：Dim0 是 Row(Height), Dim1 是 Col(Width)。
            # 所以這裡 OpenCV 會認為這張圖是被轉置的 (Width x Height)。
            # 為了保持一致，我們直接指定目標形狀：

            target_w = self.display_shape[1]  # 對應 array axis 1
            target_h = self.display_shape[0]  # 對應 array axis 0

            combined_img = cv2.resize(
                combined_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        return combined_img
