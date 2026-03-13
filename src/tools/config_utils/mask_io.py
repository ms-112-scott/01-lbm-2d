"""
config_utils/mask_io.py

Mask PNG 讀取的統一入口。

確保與 solver mask_utils 讀取邏輯完全一致：
  灰階讀取 → cv2.resize(width=nx, height=ny) → threshold 127
  → invert 旗標 → .T 轉置為 [nx, ny]

注意事項：
  - cv2.resize 參數順序是 (width, height) = (nx, ny)
  - 讀入後 img.shape = (ny, nx)，.T 後 = (nx, ny)，與 Taichi field 慣例一致
  - 此模組僅負責 I/O，不做任何幾何計算
"""

import cv2
import numpy as np


def load_solid_mask(png_path: str, invert: bool, nx: int, ny: int) -> np.ndarray:
    """
    讀取建物遮罩 PNG，返回 solid[nx, ny] (bool，True=固體)。

    Args:
        png_path : PNG 檔案路徑
        invert   : True 時以亮色為固體（黑底白建築），False 時以暗色為固體
        nx       : domain X 方向格點數（水平）
        ny       : domain Y 方向格點數（垂直）

    Returns:
        shape=(nx, ny) 的 bool 陣列，True 代表固體格點

    Raises:
        ValueError: 無法讀取圖片時
    """
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"無法讀取圖片：{png_path}")

    # cv2.resize(src, (width, height)) → img.shape = (ny, nx)
    if img.shape != (ny, nx):
        img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_NEAREST)

    solid_yx = (img > 127) if invert else (img < 127)  # shape: (ny, nx)
    return solid_yx.T  # → (nx, ny)，與 Taichi field 慣例一致
