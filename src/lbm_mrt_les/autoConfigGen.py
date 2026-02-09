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
    "RECT_COUNT": [2, 4, 6],
    "NUM_SAMPLES": [2, 2, 2],
    "MIN_DISTANCE": 12,
    "MAX_BLOCKAGE_RATIO": 0.4,
    "ROTATE_ANGLE_MAX": 15,
    "OUTPUT_DIR": "src/GenMask/rect_masks",  # 修改輸出目錄以便區分
    "MAX_ATTEMPTS": 200,
    "VAL_BACKGROUND": 255,
    "VAL_OBJECT": 0,
    "BUFFER": {"TOP": 128, "BOTTOM": 128, "LEFT": 128, "RIGHT": 960},
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
    """檢查 SDF 距離"""
    if np.all(current_mask == CONFIG["VAL_BACKGROUND"]):
        return True
    sdf = cv2.distanceTransform(current_mask, cv2.DIST_L2, 5)
    new_rect_mask = np.zeros_like(current_mask)
    cv2.drawContours(new_rect_mask, [new_box_points], 0, 255, -1)
    covered_sdf_values = sdf[new_rect_mask > 0]
    if len(covered_sdf_values) == 0:
        return True
    if np.min(covered_sdf_values) < min_dist:
        return False
    return True


def check_blockage_ratio(current_mask, new_box_points, max_ratio):
    """檢查阻塞率"""
    H, W = current_mask.shape
    temp_mask = current_mask.copy()
    cv2.drawContours(temp_mask, [new_box_points], 0, CONFIG["VAL_OBJECT"], -1)

    # 計算 Y 軸投影 (背景=255, 物體=0)
    # 只要該行有一個像素是 0，min 就是 0
    y_projection = np.min(temp_mask, axis=1)
    blocked_pixels = np.sum(y_projection == 0)

    current_ratio = blocked_pixels / H
    if current_ratio > max_ratio:
        return False
    return True


def generate_sample(n_rects, sample_id):
    """生成單張 Mask"""
    mask = np.full(
        (CONFIG["NY"], CONFIG["NX"]), CONFIG["VAL_BACKGROUND"], dtype=np.uint8
    )
    added_count = 0
    attempts = 0

    while added_count < n_rects:
        attempts += 1
        if attempts > CONFIG["MAX_ATTEMPTS"]:
            break

        box_points = get_random_rotated_rect_constrained(mask.shape, CONFIG["BUFFER"])

        if not check_valid_placement_sdf(mask, box_points, CONFIG["MIN_DISTANCE"]):
            continue
        if not check_blockage_ratio(mask, box_points, CONFIG["MAX_BLOCKAGE_RATIO"]):
            continue

        cv2.drawContours(mask, [box_points], 0, CONFIG["VAL_OBJECT"], -1)
        added_count += 1

    return mask


# ==========================================
# 新增：計算特徵長度 (Characteristic Length)
# ==========================================
def calculate_characteristic_length(mask):
    """
    計算流場的特徵長度 L。
    定義：Y 軸上的總投影長度 (Total Projected Length)。
    物理意義：這是流體必須繞過的「有效障礙物寬度」，直接決定了
             狹縫處的加速效應 (Venturi effect) 和雷諾數的尺度。
    """
    # 1. 取得 Y 軸投影 (Axis 1 = X軸方向壓縮 -> 得到 Y 軸分佈)
    # Mask: 255=Fluid, 0=Object
    # np.min: 如果一行中有任何黑色像素(0)，該行結果就是 0
    y_projection = np.min(mask, axis=1)

    # 2. 統計被佔用的像素總數 (即特徵長度 L)
    L_char = np.sum(y_projection == 0)

    return int(L_char)


def main():
    if not os.path.exists(CONFIG["OUTPUT_DIR"]):
        os.makedirs(CONFIG["OUTPUT_DIR"])

    print(f"Start Generation with Characteristic Length in filename.")

    total_files = 0
    for r_count, n_samples in zip(CONFIG["RECT_COUNT"], CONFIG["NUM_SAMPLES"]):
        print(f"--- Batch: {r_count} rects, {n_samples} images ---")
        for i in range(n_samples):
            # 1. 生成 Mask
            mask = generate_sample(r_count, i)

            # 2. 計算特徵長度
            L_char = calculate_characteristic_length(mask)

            # 3. 檔名包含特徵長度 (L)
            # 格式: mask_phys_r{矩形數}_{編號}_L{特徵長度}.png
            filename = f"mask_phys_r{r_count}_{i:04d}_L{L_char}.png"
            filepath = os.path.join(CONFIG["OUTPUT_DIR"], filename)

            cv2.imwrite(filepath, mask)
            total_files += 1

    print(f"Done! Saved {total_files} images to {CONFIG['OUTPUT_DIR']}")


if __name__ == "__main__":
    main()
