"""
config_utils/geometry.py

從 MaskContext 計算幾何特性，結果寫回同一個 dict：
  - l_char      : 最大單一建築的等效特徵尺寸（px）
  - max_blockage: 最嚴重 X 截面的阻塞率

公開函式：
  fill_geometry(mask_ctx, sim_ctx) -> None
    直接修改 mask_ctx["l_char"] 與 mask_ctx["max_blockage"]

  calc_l_char / calc_max_blockage
    保留為獨立函式供 pre-scan 使用（pre-scan 只需 l_char，不建整包 MaskContext）
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion

from .mask_io import load_solid_mask

_EROSION_ITER = 3
_AREA_FRAC_MAX = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# region 公開入口：填充 mask_ctx
# ─────────────────────────────────────────────────────────────────────────────

def fill_geometry(mask_ctx: dict, sim_ctx: dict) -> None:
    """
    計算 L_char 與 max_blockage，結果寫入 mask_ctx。

    Args:
        mask_ctx : MaskContext，須含 mask_path, nx, ny, pad_right
        sim_ctx  : SimContext，須含 mask_invert, blockage_buffer

    Raises:
        ValueError / OSError: 無法讀取 PNG 時向上拋出
    """
    mask_ctx["l_char"] = calc_l_char(
        png_path=mask_ctx["mask_path"],
        invert=sim_ctx["mask_invert"],
        nx=mask_ctx["nx"],
        ny=mask_ctx["ny"],
    )
    mask_ctx["max_blockage"] = calc_max_blockage(
        png_path=mask_ctx["mask_path"],
        invert=sim_ctx["mask_invert"],
        nx=mask_ctx["nx"],
        ny=mask_ctx["ny"],
        pad_right=mask_ctx["pad_right"],
        buffer=sim_ctx["blockage_buffer"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# region 低階計算（供 pre-scan 直接呼叫）
# ─────────────────────────────────────────────────────────────────────────────

def calc_l_char(png_path: str, invert: bool, nx: int, ny: int) -> int:
    """
    計算最大單一建築的等效特徵尺寸 L_char（像素）。

    演算法（v4）：
      1. morphological erosion（3px）分離黏連建築
      2. 連通域分析
      3. 面積過濾：排除 > domain * 5% 的大連通域
      4. 每個合格連通域取 min(X-span, Y-span)
      5. 取最大值
    """
    solid = load_solid_mask(png_path, invert, nx, ny)
    solid_eroded = binary_erosion(solid, iterations=_EROSION_ITER)
    labeled, n_features = ndimage.label(solid_eroded)

    if n_features == 0:
        labeled, n_features = ndimage.label(solid)
        if n_features == 0:
            return max(1, ny // 8)

    area_max = int(nx * ny * _AREA_FRAC_MAX)
    max_l = 0

    for label_id in range(1, n_features + 1):
        region = labeled == label_id
        if int(np.sum(region)) > area_max:
            continue

        x_idx = np.where(np.any(region, axis=1))[0]
        y_idx = np.where(np.any(region, axis=0))[0]
        if len(x_idx) == 0 or len(y_idx) == 0:
            continue

        x0 = max(0, x_idx[0] - _EROSION_ITER)
        x1 = min(nx - 1, x_idx[-1] + _EROSION_ITER)
        y0 = max(0, y_idx[0] - _EROSION_ITER)
        y1 = min(ny - 1, y_idx[-1] + _EROSION_ITER)
        roi = solid[x0:x1 + 1, y0:y1 + 1]

        x_real = np.where(np.any(roi, axis=1))[0]
        y_real = np.where(np.any(roi, axis=0))[0]
        if len(x_real) == 0 or len(y_real) == 0:
            continue

        max_l = max(max_l, min(
            int(x_real[-1] - x_real[0] + 1),
            int(y_real[-1] - y_real[0] + 1),
        ))

    if max_l == 0:
        min_area, fallback_l = nx * ny, 1
        for label_id in range(1, n_features + 1):
            region = labeled == label_id
            area = int(np.sum(region))
            if area < min_area:
                x_idx = np.where(np.any(region, axis=1))[0]
                y_idx = np.where(np.any(region, axis=0))[0]
                if len(x_idx) > 0 and len(y_idx) > 0:
                    min_area = area
                    fallback_l = min(
                        int(x_idx[-1] - x_idx[0] + 1),
                        int(y_idx[-1] - y_idx[0] + 1),
                    )
        max_l = fallback_l

    return max(1, max_l)


def calc_max_blockage(
    png_path: str,
    invert: bool,
    nx: int,
    ny: int,
    pad_right: int = 512,
    buffer: int = 128,
) -> float:
    """
    計算最嚴重 X 截面的阻塞率（5px 滾動平均後取最大值）。

    排除範圍：
      左側：前 5%（inlet BC 節點）
      右側：nx - pad_right - buffer（sponge + ROI buffer）
    """
    solid = load_solid_mask(png_path, invert, nx, ny)
    x_start = max(1, int(nx * 0.05))
    x_end = min(nx - 1, nx - pad_right - buffer)
    roi = solid[x_start:x_end, :]

    if roi.shape[0] == 0:
        return 0.0

    blockage_per_x = np.mean(roi.astype(np.float32), axis=1)
    window = 5
    if len(blockage_per_x) >= window:
        kernel = np.ones(window, dtype=np.float32) / window
        smoothed = np.convolve(blockage_per_x, kernel, mode="valid")
    else:
        smoothed = blockage_per_x

    return float(np.max(smoothed))
