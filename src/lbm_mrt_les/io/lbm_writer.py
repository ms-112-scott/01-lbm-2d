import h5py
import numpy as np
import json
import os
import cv2
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

        # 儲存瞬時場 (Fluctuations / Raw Data)
        self.dset_fluctuations = self.f.create_dataset(
            "fluctuations",
            shape=(0, channels, self.target_h, self.target_w),
            maxshape=(None, channels, self.target_h, self.target_w),
            dtype="f4",
            compression=config["outputs"]["dataset"]["compression"],
            chunks=(1, channels, self.target_h, self.target_w),
        )

        # --- 統計變數初始化 ---
        # 1. 用於 RANS (Mean Field)
        self.running_sum = np.zeros(
            (channels, self.target_h, self.target_w), dtype=np.float64
        )

        # 2. [新增] 用於累積絕對渦度 (Sum Abs Vorticity)
        # 形狀為 (H, W)，因為渦度是純量
        self.running_sum_abs_vor = np.zeros(
            (self.target_h, self.target_w), dtype=np.float64
        )

        self.running_count = 0
        self.global_min = np.full(channels, np.inf)
        self.global_max = np.full(channels, -np.inf)

    def append(self, moment_data):
        """
        moment_data: (nx, ny, channels) - 來自 Solver 的原始數據
        """
        if self.is_closed:
            return

        # 1. [Crop] 裁剪安全區
        # (Width, Height, Channels)
        cropped_data = moment_data[self.slice_x, self.slice_y, :]

        # 2. [Prepare for OpenCV]
        # (Width, Height, Channels) -> (Height, Width, Channels)
        img_hwc = cropped_data.transpose(1, 0, 2)

        # 3. [Resize with Loop]
        # OpenCV resize 不支援 > 4 channels，所以必須拆開處理
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

        # ---------------------------------------------------------
        # [New Feature] 計算渦度並累積
        # ---------------------------------------------------------
        # 假設: channel 1 = ux, channel 2 = uy (常見 D2Q9 輸出格式 [rho, ux, uy, ...])
        # 如果您的數據是純 Population (f0~f8)，請先在此處計算 Macroscopic Velocity
        try:
            ux = resized_hwc[:, :, 1]
            uy = resized_hwc[:, :, 2]

            # 計算梯度 (np.gradient 回傳 [gradient_axis_0, gradient_axis_1])
            # axis 0 是 y (row), axis 1 是 x (col)
            # du/dy, du/dx
            grad_ux = np.gradient(ux)
            du_dy = grad_ux[0]

            # dv/dy, dv/dx
            grad_uy = np.gradient(uy)
            dv_dx = grad_uy[1]

            # 2D Vorticity (Curl) = dv/dx - du/dy
            vorticity = dv_dx - du_dy

            # 累積絕對值
            self.running_sum_abs_vor += np.abs(vorticity)

        except IndexError:
            pass  # 如果 channel 不夠，跳過計算

        # ---------------------------------------------------------

        # 4. [Format for HDF5]
        # (H, W, C) -> (C, H, W)
        data_final = resized_hwc.transpose(2, 0, 1)

        # 5. [Write]
        current_len = self.dset_fluctuations.shape[0]
        self.dset_fluctuations.resize(current_len + 1, axis=0)
        self.dset_fluctuations[current_len] = data_final

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

        # 1. 儲存 Mean Field
        mean_field = (self.running_sum / self.running_count).astype(np.float32)
        self.f.create_dataset("mean_field", data=mean_field)

        # 2. [新增] 儲存 Sum Abs Vorticity
        # 這是所有時間步的 |ω| 總和，可用於視覺化湍流活躍區
        self.f.create_dataset(
            "sum_abs_vorticity", data=self.running_sum_abs_vor.astype(np.float32)
        )
        # 如果您也想要平均渦度場，可以除以 count:
        # self.f.create_dataset("mean_abs_vorticity", data=(self.running_sum_abs_vor / self.running_count).astype(np.float32))

        try:
            meta_config = self.config.copy()
            meta_config["_dataset_info"] = {
                "original_crop": [self.crop_w, self.crop_h],
                "saved_resolution": [self.target_w, self.target_h],
                "resize_algo": "cv2.INTER_AREA (Per-Channel Loop)",
                "extra_fields": ["sum_abs_vorticity"],
            }
            self.f.attrs["config_json"] = json.dumps(meta_config, default=str)
        except Exception:
            pass

        self.f.attrs["stats_min"] = self.global_min
        self.f.attrs["stats_max"] = self.global_max
        self.f.attrs["stats_mean"] = np.mean(mean_field, axis=(1, 2))

        # 記錄總幀數方便後續計算平均
        self.f.attrs["frame_count"] = self.running_count

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
