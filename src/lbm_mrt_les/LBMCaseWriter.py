import h5py
import numpy as np
import json
import os
import cv2  # 必須安裝 opencv-python
import threading
import queue


class LBMCaseWriter:
    def __init__(self, file_path, config, nx, ny, channels=9):
        # 確保目錄存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        self.file_path = file_path
        self.config = config
        self.nx = nx
        self.ny = ny
        self.channels = channels
        self.is_closed = False

        # ---------------------------------------------------------
        # [Strict Config] 1. 嚴格讀取裁剪參數
        # ---------------------------------------------------------
        zones = config["domain_zones"]
        sponge_y = zones["sponge_y"]
        sponge_x = zones["sponge_x"]
        buffer = zones["buffer"]
        inlet_buffer = zones["inlet_buffer"]

        # ---------------------------------------------------------
        # [Slice Definition] 2. 定義切片範圍
        # ---------------------------------------------------------
        # User Logic: simulation_np[inlet_buffer : nx- sponge_x-buffer , sponge_y+ buffer: ny-buffer-sponge_y]
        self.slice_x = slice(inlet_buffer, nx - sponge_x - buffer)
        self.slice_y = slice(sponge_y + buffer, ny - buffer - sponge_y)

        # 計算裁剪後的原始尺寸
        self.crop_w = (nx - sponge_x - buffer) - inlet_buffer
        self.crop_h = (ny - buffer - sponge_y) - (sponge_y + buffer)

        if self.crop_w <= 0 or self.crop_h <= 0:
            raise ValueError(
                f"[Error] Crop area is invalid! "
                f"W={self.crop_w}, H={self.crop_h}. Check your domain_zones config."
            )

        # ---------------------------------------------------------
        # [Resize Logic] 3. 計算目標縮放尺寸
        # ---------------------------------------------------------
        save_res = config["outputs"]["dataset"]["save_resolution"]

        long_side = max(self.crop_w, self.crop_h)
        scale = save_res / long_side

        self.target_w = int(self.crop_w * scale)
        self.target_h = int(self.crop_h * scale)

        print(
            f"[H5 Init] Crop: {self.crop_w}x{self.crop_h} -> "
            f"Resize: {self.target_w}x{self.target_h} "
            f"(Scale: {scale:.4f}, Method: Average Pooling)"
        )

        # ---------------------------------------------------------
        # [HDF5 Setup] 4. 初始化檔案
        # ---------------------------------------------------------
        self.f = h5py.File(file_path, "w", libver="latest")

        self.dset_snapshots = self.f.create_dataset(
            "snapshots",
            shape=(0, channels, self.target_h, self.target_w),
            maxshape=(None, channels, self.target_h, self.target_w),
            dtype="f4",
            compression=config["outputs"]["dataset"]["compression"],
            chunks=(1, channels, self.target_h, self.target_w),
        )

        # 統計變數
        self.running_sum = np.zeros(
            (channels, self.target_h, self.target_w), dtype=np.float64
        )
        self.running_count = 0
        self.global_min = np.full(channels, np.inf)
        self.global_max = np.full(channels, -np.inf)

    def append(self, moment_data):
        """
        moment_data: (nx, ny, 9) - 來自 Solver 的原始數據
        """
        if self.is_closed:
            return

        # 1. [Crop] 裁剪安全區
        # moment_data shape: (NX, NY, C) -> (Width, Height, Channels)
        cropped_data = moment_data[self.slice_x, self.slice_y, :]

        # 2. [Prepare for OpenCV]
        # (Width, Height, Channels) -> (Height, Width, Channels)
        img_hwc = cropped_data.transpose(1, 0, 2)

        # 3. [Resize with Loop]
        # [Fix] OpenCV resize 不支援 > 4 channels，所以必須拆開處理
        resized_channels = []
        for i in range(self.channels):
            # 取出單一 Channel: (H, W)
            channel_data = img_hwc[:, :, i]

            # 單獨 Resize
            resized_ch = cv2.resize(
                channel_data,
                (self.target_w, self.target_h),
                interpolation=cv2.INTER_AREA,
            )
            resized_channels.append(resized_ch)

        # 將 List 堆疊回 Numpy Array -> (H, W, C)
        resized_hwc = np.stack(resized_channels, axis=2)

        # 4. [Format for HDF5]
        # 目前 resized_hwc 是 (H, W, C) -> 轉成 (C, H, W)
        data_final = resized_hwc.transpose(2, 0, 1)

        # 5. [Write]
        current_len = self.dset_snapshots.shape[0]
        self.dset_snapshots.resize(current_len + 1, axis=0)
        self.dset_snapshots[current_len] = data_final

        # 6. [Stats]
        self.running_sum += data_final
        self.running_count += 1

        frame_min = np.min(data_final, axis=(1, 2))
        frame_max = np.max(data_final, axis=(1, 2))
        self.global_min = np.minimum(self.global_min, frame_min)
        self.global_max = np.maximum(self.global_max, frame_max)

    def finalize(self):
        if self.is_closed:
            return

        if self.running_count == 0:
            self.f.close()
            self.is_closed = True
            return

        print(f"[H5] Finalizing stats for {self.file_path}...")

        mean_field = (self.running_sum / self.running_count).astype(np.float32)
        self.f.create_dataset("mean_field", data=mean_field)

        try:
            meta_config = self.config.copy()
            meta_config["_dataset_info"] = {
                "original_crop": [self.crop_w, self.crop_h],
                "saved_resolution": [self.target_w, self.target_h],
                "resize_algo": "cv2.INTER_AREA (Per-Channel Loop)",
            }
            self.f.attrs["config_json"] = json.dumps(meta_config, default=str)
        except Exception:
            pass

        self.f.attrs["stats_min"] = self.global_min
        self.f.attrs["stats_max"] = self.global_max
        self.f.attrs["stats_mean"] = np.mean(mean_field, axis=(1, 2))

        self.f.close()
        self.is_closed = True
        print("[H5] Closed.")

    def close(self):
        self.finalize()


# ------------------------------------------------------------------
# [Async Wrapper] 保持不變，直接複製使用即可
# ------------------------------------------------------------------
class AsyncLBMCaseWriter:
    def __init__(self, *args, **kwargs):
        self.writer = LBMCaseWriter(*args, **kwargs)
        self.queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                data = self.queue.get(timeout=1.0)
                if data is None:
                    break
                self.writer.append(data)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncWriter Error] {e}")

    def append(self, moment_data):
        self.queue.put(moment_data)

    def finalize(self):
        print("[AsyncH5] Waiting for background writes to finish...")
        self.stop_event.set()
        self.thread.join()
        self.writer.finalize()

    def close(self):
        self.finalize()
