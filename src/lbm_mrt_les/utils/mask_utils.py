import os
import cv2
import numpy as np

def _create_from_png(nx, ny, config, png_path):
    """
    從 PNG 讀取 Mask
    """

    if not png_path or not os.path.exists(png_path):
        raise FileNotFoundError(f"[Error] Mask file not found: {png_path}")

    # 1. 以灰階模式讀取
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"[Error] Failed to load image: {png_path}")

    # 2. 強制縮放到模擬網格大小 (nx, ny)
    # cv2.resize 接受 (width, height) -> (nx, ny)
    # resize 後的 img numpy array 形狀會是 (height, width) -> (ny, nx)
    if img.shape != (ny, nx):
        print(f"  -> Resizing mask from {img.shape[::-1]} to ({nx}, {ny})")
        img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_NEAREST)

    # 3. 二值化轉換
    threshold = 127
    inverse = config["mask"]["invert"]

    if inverse:
        mask = img > threshold
    else:
        mask = img < threshold

    # 4. [關鍵修正] 轉置矩陣 (Transpose)
    # Numpy/OpenCV 是 [y, x] (1024, 2048)
    # Taichi Solver 是 [x, y] (2048, 1024)
    # 必須使用 .T 將其轉置，否則無法塞入 taichi field
    mask = mask.T

    return mask.astype(bool)

def create_mask(config, png_path):
    mask_cfg = config["mask"]
    mask = None
    nx = config["simulation"]["nx"]
    ny = config["simulation"]["ny"]

    if config["mask"]["enable"]:

        if config["mask"]["type"] == "png":
            mask = _create_from_png(nx, ny, config, png_path=png_path)

    # 如果沒有 mask 生成 (或 type 不對)，建立一個全 False (全流體) 的空 mask
    if mask is None:
        mask = np.zeros((ny, nx), dtype=bool)

    return mask
