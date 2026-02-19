import numpy as np
import matplotlib
from matplotlib import cm

def get_vorticity_mapper(vorticity_range):
    """
    建立渦度（Vorticity）專用的 Colormap 與 Mapper
    """
    colors = [
        (1, 1, 0),             # 黃
        (0.953, 0.490, 0.016), # 橘
        (0, 0, 0),             # 黑
        (0.176, 0.976, 0.529), # 綠
        (0, 1, 1),             # 青
    ]
    vor_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "vorticity_cmap", colors
    )
    vor_cmap.set_bad(color="grey")
    
    vor_norm = matplotlib.colors.Normalize(
        vmin=-vorticity_range, vmax=vorticity_range
    )
    return cm.ScalarMappable(norm=vor_norm, cmap=vor_cmap)

def apply_color_mapping(data, mapper, mask=None, obstacle_color=0.5):
    """
    將數值資料轉換為 RGB 影像，並套用遮罩顏色
    """
    # 轉換為 RGBA 並取前三通道 (RGB)
    img = mapper.to_rgba(data)[:, :, :3]
    
    if mask is not None:
        img[mask == 1] = obstacle_color
        
    return img

def apply_velocity_coloring(vel_mag, u_norm_max, mask=None, obstacle_color=0.5):
    """
    針對速度場的 plasma 著色處理
    """
    vel_norm_val = np.clip(vel_mag / u_norm_max, 0, 1)
    vel_img = cm.plasma(vel_norm_val)[:, :, :3]
    
    if mask is not None:
        vel_img[mask == 1] = obstacle_color
        
    return vel_img