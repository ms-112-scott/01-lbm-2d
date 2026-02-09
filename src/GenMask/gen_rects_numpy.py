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
    "MAX_BLOCKAGE_RATIO": 0.4,
    "ROTATE_ANGLE_MAX": 15,
    "OUTPUT_DIR": "src/GenMask/rect_masks",
    "MAX_ATTEMPTS": 200,
    "VAL_BACKGROUND": 255,
    "VAL_OBJECT": 0,
    # === 關鍵修改：這裡定義左邊第一個像素必須出現的位置 ===
    "BUFFER": {"TOP": 128, "BOTTOM": 128, "LEFT": 128, "RIGHT": 512},
    "RECT_SIZE": {"MIN_W": 50, "MAX_W": 400, "MIN_H": 50, "MAX_H": 400},
}


def get_random_rotated_rect_constrained(canvas_shape, buffers):
    """生成單個旋轉矩形參數"""
    H, W = canvas_shape
    # 這裡的隨機範圍可以寬鬆一點，因為最後我們會強制左移
    # 但為了保證不會超出右邊界，還是保留 buffer
    max_diag = np.sqrt(
        CONFIG["RECT_SIZE"]["MAX_W"] ** 2 + CONFIG["RECT_SIZE"]["MAX_H"] ** 2
    )
    margin = int(max_diag / 2) + 10

    safe_x_min = buffers["LEFT"] + margin
    safe_x_max = W - buffers["RIGHT"] - margin
    safe_y_min = buffers["TOP"] + margin
    safe_y_max = H - buffers["BOTTOM"] - margin

    if safe_x_max <= safe_x_min or safe_y_max <= safe_y_min:
        # 如果 buffer 太大導致無法生成，稍微容錯處理或報錯
        # 這裡為了不卡死，可以動態調整，但目前先維持報錯
        raise ValueError("Buffer settings are too large for the object size!")

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

    # 背景是255，物體是0。distanceTransform 計算到最近的0的距離
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
    """檢查 Y 軸投影阻塞率"""
    H, W = current_mask.shape
    temp_mask = current_mask.copy()
    cv2.drawContours(temp_mask, [new_box_points], 0, CONFIG["VAL_OBJECT"], -1)

    # 投影檢查：若該行有任一黑點(0)，則該行算被佔據
    # min(axis=1) 為 0 代表該 row 有障礙物
    y_projection = np.min(temp_mask, axis=1)
    blocked_pixels = np.sum(y_projection == 0)
    current_ratio = blocked_pixels / H

    if current_ratio > max_ratio:
        return False
    return True


def align_objects_to_left_buffer(mask, target_buffer_x):
    """
    [新增功能]
    將 mask 上所有的物體整體向左平移，
    使得最左邊的物體像素 (Pixel value = 0) 正好落在 target_buffer_x 的位置。
    """
    # 1. 找出所有物體 (數值為 0) 的座標
    # np.where 回傳 (rows, cols)
    object_pixels = np.where(mask == CONFIG["VAL_OBJECT"])

    # 如果圖上完全沒有物體，直接回傳原圖
    if len(object_pixels[0]) == 0:
        return mask

    # 2. 找到目前最左邊的 X 座標
    current_min_x = np.min(object_pixels[1])

    # 3. 計算需要移動的距離
    # 如果 current_min_x 是 200，target 是 128，則 shift = 200 - 128 = 72 (向左移 72)
    # 如果 shift 為負值，代表物體已經在 buffer 左邊了 (這在隨機生成時應該被 buffer 擋住，但以防萬一)
    shift_x = -(current_min_x - target_buffer_x)

    if shift_x == 0:
        return mask

    # 4. 建立平移矩陣 [1, 0, tx], [0, 1, ty]
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])

    # 5. 執行平移
    # borderValue=CONFIG["VAL_BACKGROUND"] 很重要，因為平移後右邊空出來的地方要填白色 (255)
    shifted_mask = cv2.warpAffine(
        mask,
        M,
        (mask.shape[1], mask.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=CONFIG["VAL_BACKGROUND"],
    )

    return shifted_mask


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

        # 生成時先用一般的 Buffer 邏輯隨機撒
        box_points = get_random_rotated_rect_constrained(mask.shape, CONFIG["BUFFER"])

        if not check_valid_placement_sdf(mask, box_points, CONFIG["MIN_DISTANCE"]):
            continue

        if not check_blockage_ratio(mask, box_points, CONFIG["MAX_BLOCKAGE_RATIO"]):
            continue

        # 畫上去
        cv2.drawContours(mask, [box_points], 0, CONFIG["VAL_OBJECT"], -1)
        added_count += 1

    # ==========================================
    # 關鍵修改：生成完畢後，強制執行左對齊
    # ==========================================
    if added_count > 0:
        mask = align_objects_to_left_buffer(mask, CONFIG["BUFFER"]["LEFT"])

    return mask


def main():
    if not os.path.exists(CONFIG["OUTPUT_DIR"]):
        os.makedirs(CONFIG["OUTPUT_DIR"])

    print(f"Start Physics-Constrained Generation with Left Alignment.")
    print(
        f"Constraints: Min Dist = {CONFIG['MIN_DISTANCE']}px, "
        f"First Object X = {CONFIG['BUFFER']['LEFT']}px"
    )

    total_files = 0
    for r_count, n_samples in zip(CONFIG["RECT_COUNT"], CONFIG["NUM_SAMPLES"]):
        print(f"--- Batch: Target {r_count} rects, {n_samples} images ---")
        for i in range(n_samples):
            mask = generate_sample(r_count, i)

            # 存檔
            filename = f"mask_phys_r{r_count}_{i:04d}.png"
            filepath = os.path.join(CONFIG["OUTPUT_DIR"], filename)
            cv2.imwrite(filepath, mask)
            total_files += 1

    print(f"Done! Saved {total_files} images to {CONFIG['OUTPUT_DIR']}")


if __name__ == "__main__":
    main()
