import cv2
import numpy as np
import os
import sys


def save_last_frame_from_video(video_path, output_img_path):
    """
    [應用策略]: 提取模擬的最後一幀。
    這通常用於觀察模擬結束時的瞬時渦流結構 (Instantaneous Flow)。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 將指針移至最後一幀
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_img_path, frame)
        print(f"--- [Success] Last frame saved to: {output_img_path} ---")
    else:
        print(f"[Error] Failed to extract last frame from {video_path}")

    cap.release()
    return ret


def calculate_temporal_average_from_video(
    video_path, output_img_path, show_progress=True
):
    """
    [數值層面]: 計算影片影格平均值。
    注意：此為像素平均，僅供視覺分析使用，非物理場時間平均。
    """
    if not os.path.exists(video_path):
        print(f"[Error] Video file not found: {video_path}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    avg_accumulator = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_float = frame.astype(np.float32)
        if avg_accumulator is None:
            avg_accumulator = frame_float
        else:
            avg_accumulator += frame_float

        frame_count += 1
        if show_progress and frame_count % 50 == 0:
            sys.stdout.write(
                f"\rProcessing Average: {(frame_count / total_frames) * 100:.1f}%"
            )
            sys.stdout.flush()

    if frame_count > 0:
        avg_img = (avg_accumulator / frame_count).astype(np.uint8)
        cv2.imwrite(output_img_path, avg_img)
        print(f"\n--- [Success] Average image saved: {output_img_path} ---")

    cap.release()
    return True


# ==========================================
# 自動化批次處理：加入檢查機制
# ==========================================
if __name__ == "__main__":
    root_dir = "src/lbm_mrt_les/output"
    print(f"--- Starting Batch Post-Processing in: {root_dir} ---")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        video_files = [f for f in filenames if f.endswith(".mp4")]

        for video_file in video_files:
            input_path = os.path.join(dirpath, video_file)
            base_name = os.path.splitext(video_file)[0]

            output_avg = os.path.join(dirpath, f"{base_name}_AVG.png")
            output_last = os.path.join(dirpath, f"{base_name}_LAST.png")

            print(f"\n[Target]: {input_path}")

            # --- 策略性跳過判斷 ---

            # 處理 Average Image
            if os.path.exists(output_avg):
                print(
                    f"  >> Skipping Average: {os.path.basename(output_avg)} already exists."
                )
            else:
                calculate_temporal_average_from_video(input_path, output_avg)

            # 處理 Last Frame
            if os.path.exists(output_last):
                print(
                    f"  >> Skipping Last Frame: {os.path.basename(output_last)} already exists."
                )
            else:
                save_last_frame_from_video(input_path, output_last)

    print("\n--- All tasks completed ---")
