import cv2
import numpy as np
import os
import sys


def save_last_frame_from_video(video_path, output_img_path):
    """
    提取模擬最後一幀：用於觀察瞬時流場結構。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_img_path, frame)
        print(f"  [Success] Last frame -> {os.path.basename(output_img_path)}")

    cap.release()
    return ret


def calculate_temporal_average_from_video(
    video_path, output_img_path, show_progress=True
):
    """
    計算時間平均：用於觀察穩定風速分布或弱風區。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
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
                f"\r    Processing Average: {(frame_count / total_frames) * 100:.1f}%"
            )
            sys.stdout.flush()

    if frame_count > 0:
        avg_img = (avg_accumulator / frame_count).astype(np.uint8)
        cv2.imwrite(output_img_path, avg_img)
        if show_progress:
            print()  # Line break
        print(f"  [Success] Average img -> {os.path.basename(output_img_path)}")

    cap.release()
    return True


# ==========================================
# 執行核心： os.walk 遞迴處理
# ==========================================
if __name__ == "__main__":
    root_dir = "src/lbm_mrt_les/output"
    print(f"--- LBM Post-Processing Task Start ---")
    print(f"Scanning directory: {root_dir}\n")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 找出該資料夾下所有 mp4
        video_files = [f for f in filenames if f.endswith(".mp4")]

        if not video_files:
            continue

        print(f"\nChecking directory: {dirpath}")

        for video_file in video_files:
            # 完整路徑確保同名檔案不會衝突
            input_path = os.path.join(dirpath, video_file)
            base_name = os.path.splitext(video_file)[0]

            output_avg = os.path.join(dirpath, f"{base_name}_AVG.png")
            output_last = os.path.join(dirpath, f"{base_name}_LAST.png")

            # --- 執行與跳過機制 ---

            # 1. 處理 Last Frame (優先執行，速度快)
            if not os.path.exists(output_last):
                save_last_frame_from_video(input_path, output_last)
            else:
                print(f"  [Skip] Last frame exists for {video_file}")

            # 2. 處理 Average
            if not os.path.exists(output_avg):
                calculate_temporal_average_from_video(input_path, output_avg)
            else:
                print(f"  [Skip] Average exists for {video_file}")

    print("\n--- All LBM visualization tasks completed ---")
