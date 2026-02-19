import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from typing import Optional, List, Tuple
from numpy.typing import NDArray

# --- 私有輔助工具 ---

def _create_vorticity_cmap() -> LinearSegmentedColormap:
    """建立專用的渦度 Colormap (黃-橘-黑-綠-青)"""
    colors = [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0), 
              (0.176, 0.976, 0.529), (0, 1, 1)]
    cmap = LinearSegmentedColormap.from_list("vorticity_cmap", colors)
    cmap.set_bad(color="grey")
    return cmap

def _apply_colormap(
    data: NDArray[np.float32],
    cmap: matplotlib.colors.Colormap,
    vmin: float,
    vmax: float,
    mask: Optional[NDArray] = None,
    obstacle_color: float = 0.5
) -> NDArray[np.float32]:
    """通用顏色映射邏輯"""
    norm = Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # 處理遮罩 (設為 NaN 讓 cmap.set_bad 起作用)
    plot_data = data.copy()
    if mask is not None:
        plot_data[mask > 0] = np.nan
        
    img_rgb = mapper.to_rgba(plot_data)[:, :, :3]
    
    # 強制覆蓋障礙物顏色 (確保邊界銳利)
    if mask is not None:
        img_rgb[mask == 1] = obstacle_color
        
    return img_rgb.astype(np.float32)

# --- 公開場量轉換函數 ---

def colorize_velocity(
    vel_mag: NDArray[np.float32],
    u_norm_max: float,
    mask: Optional[NDArray] = None,
    cmap_name: str = "plasma"
) -> NDArray[np.float32]:
    """將速度量值轉換為 RGB 影像"""
    cmap = cm.get_cmap(cmap_name)
    return _apply_colormap(vel_mag, cmap, vmin=0, vmax=u_norm_max, mask=mask)

def colorize_vorticity(
    vorticity: NDArray[np.float32],
    vorticity_range: float,
    mask: Optional[NDArray] = None
) -> NDArray[np.float32]:
    """將渦度場轉換為 RGB 影像"""
    cmap = _create_vorticity_cmap()
    return _apply_colormap(vorticity, cmap, vmin=-vorticity_range, vmax=vorticity_range, mask=mask)

def colorize_pressure(
    pressure: NDArray[np.float32],
    p_min: float,
    p_max: float,
    mask: Optional[NDArray] = None
) -> NDArray[np.float32]:
    """(預留未來擴充) 將壓力場轉換為 RGB 影像"""
    cmap = cm.get_cmap("RdBu_r") # 壓力常用紅藍對比
    return _apply_colormap(pressure, cmap, vmin=p_min, vmax=p_max, mask=mask)