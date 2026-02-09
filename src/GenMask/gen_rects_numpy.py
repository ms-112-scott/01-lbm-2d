import numpy as np
import cv2
import os
import random

# ==========================================
# 參數設定 (Configuration)
# ==========================================
CONFIG = {
    "NX": 2048,
    "NY": 1024,
    # 批次設定
    "RECT_COUNT": [2, 4, 6],
    "NUM_SAMPLES": [20, 20, 10],
    # === 物理限制 1: 幾何間距 ===
    "MIN_DISTANCE": 30,  # 建築物間最小空隙 (px)
    # === 物理限制 2: 阻塞率 (Blockage Ratio) ===
    # 限制所有建築物在 Y 軸上的投影總和，不能超過畫布高度的多少比例
    # 建議 < 0.4 (40%) 以避免流速過快炸裂
    "MAX_BLOCKAGE_RATIO": 0.4,
    "ROTATE_ANGLE_MAX": 15,
    "OUTPUT_DIR": "src/GenMask/rect_masks",
    "MAX_ATTEMPTS": 200,
    "VAL_BACKGROUND": 255,
    "VAL_OBJECT": 0,
    "BUFFER": {"TOP": 128, "BOTTOM": 128, "LEFT": 128, "RIGHT": 512},
    "RECT_SIZE": {"MIN_W": 50, "MAX_W": 400, "MIN_H": 50, "MAX_H": 400},
}


def get_random_rotated_rect_constrained(canvas_shape, buffers):
    """生成單個旋轉矩形參數"""
    H, W = canvas_shape
    max_diag = np.sqrt(
        CONFIG["RECT_SIZE"]["MAX_W"] ** 2 + CONFIG["RECT_SIZE"]["MAX_H"] ** 2
    )
    margin = int(max_diag / 2) + 10

    safe_x_min = buffers["LEFT"] + margin
    safe_x_max = W - buffers["RIGHT"] - margin
    safe_y_min = buffers["TOP"] + margin
    safe_y_max = H - buffers["BOTTOM"] - margin

    if safe_x_max <= safe_x_min or safe_y_max <= safe_y_min:
        raise ValueError("Buffer settings are too large!")

    cx = random.randint(safe_x_min, safe_x_max)
    cy = random.randint(safe_y_min, safe_y_max)
    w = random.randint(CONFIG["RECT_SIZE"]["MIN_W"], CONFIG["RECT_SIZE"]["MAX_W"])
    h = random.randint(CONFIG["RECT_SIZE"]["MIN_H"], CONFIG["RECT_SIZE"]["MAX_H"])
    angle = random.uniform(-CONFIG["ROTATE_ANGLE_MAX"], CONFIG["ROTATE_ANGLE_MAX"])

    rect_def = ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect_def)
    box = np.int64(box)
    return box


def check_valid_placement_sdf(current_mask, new_box_points, min_dist):
    """檢查 SDF 距離 (避免黏在一起)"""
    if np.all(current_mask == CONFIG["VAL_BACKGROUND"]):
        return True

    # 0 is Object, 255 is Background.
    # distanceTransform calculates distance to the nearest ZERO pixel.
    # So we invert the mask for calculation: we want dist to nearest WALL.
    # Actually, if we use the mask as is (bg=255, wall=0), distTransform calculates distance to wall.
    # Perfect.
    sdf = cv2.distanceTransform(current_mask, cv2.DIST_L2, 5)

    # 模擬畫上新矩形
    new_rect_mask = np.zeros_like(current_mask)
    cv2.drawContours(new_rect_mask, [new_box_points], 0, 255, -1)

    # 檢查新矩形區域內的 SDF 值
    covered_sdf_values = sdf[new_rect_mask > 0]

    if len(covered_sdf_values) == 0:
        return True
    if np.min(covered_sdf_values) < min_dist:
        return False

    return True


def check_blockage_ratio(current_mask, new_box_points, max_ratio):
    """
    檢查 Y 軸投影阻塞率 (避免風道堵死)
    """
    H, W = current_mask.shape

    # 1. 建立一個暫時的 Mask，將新矩形畫上去
    temp_mask = current_mask.copy()
    cv2.drawContours(temp_mask, [new_box_points], 0, CONFIG["VAL_OBJECT"], -1)

    # 2. 計算 Y 軸投影
    # 我們要看每一行(Row)是否有物體(0)
    # axis=1 代表沿著 X 軸檢查：如果這一行有任何一個像素是黑色的，這行就算被佔用
    # np.min(temp_mask, axis=1) 會得到一個長度為 H 的陣列
    # 如果某行有黑色(0)，min 就是 0；如果是全白(255)，min 就是 255
    y_projection = np.min(temp_mask, axis=1)

    # 3. 統計被佔用的像素行數 (數值為 0 的行數)
    blocked_pixels = np.sum(y_projection == 0)

    # 4. 計算比例
    current_ratio = blocked_pixels / H

    # print(f"Debug: Blockage Ratio: {current_ratio:.2f}") # Debug用

    if current_ratio > max_ratio:
        return False  # 阻塞率太高，拒絕

    return True


def generate_sample(n_rects, sample_id):
    """生成單張 Mask (包含 SDF 和 Blockage 檢查)"""
    mask = np.full(
        (CONFIG["NY"], CONFIG["NX"]), CONFIG["VAL_BACKGROUND"], dtype=np.uint8
    )

    added_count = 0
    attempts = 0

    while added_count < n_rects:
        attempts += 1
        if attempts > CONFIG["MAX_ATTEMPTS"]:
            # 如果嘗試太多次都失敗，就放棄這張圖剩下的矩形，避免無窮迴圈
            # print(f"  -> Sample {sample_id}: Stops at {added_count}/{n_rects} rects (Constraints too tight)")
            break

        # 1. 生成
        box_points = get_random_rotated_rect_constrained(mask.shape, CONFIG["BUFFER"])

        # 2. 檢查 SDF (距離)
        if not check_valid_placement_sdf(mask, box_points, CONFIG["MIN_DISTANCE"]):
            continue

        # 3. 檢查 Blockage Ratio (物理阻塞)
        if not check_blockage_ratio(mask, box_points, CONFIG["MAX_BLOCKAGE_RATIO"]):
            continue

        # 4. 通過所有檢查 -> 正式寫入
        cv2.drawContours(mask, [box_points], 0, CONFIG["VAL_OBJECT"], -1)
        added_count += 1

    return mask


def main():
    if not os.path.exists(CONFIG["OUTPUT_DIR"]):
        os.makedirs(CONFIG["OUTPUT_DIR"])

    print(f"Start Physics-Constrained Generation.")
    print(
        f"Constraints: Min Dist = {CONFIG['MIN_DISTANCE']}px, Max Blockage = {CONFIG['MAX_BLOCKAGE_RATIO']*100}%"
    )

    total_files = 0
    for r_count, n_samples in zip(CONFIG["RECT_COUNT"], CONFIG["NUM_SAMPLES"]):
        print(f"--- Batch: Target {r_count} rects, {n_samples} images ---")
        for i in range(n_samples):
            mask = generate_sample(r_count, i)

            # 檔名
            filename = f"mask_phys_r{r_count}_{i:04d}.png"
            filepath = os.path.join(CONFIG["OUTPUT_DIR"], filename)
            cv2.imwrite(filepath, mask)
            total_files += 1

    print(f"Done! Saved {total_files} images to {CONFIG['OUTPUT_DIR']}")


if __name__ == "__main__":
    main()
