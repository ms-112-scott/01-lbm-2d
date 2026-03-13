import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_mask(mask):
    """視覺化 Mask"""
    plt.figure(figsize=(10, 5))
    plt.imshow(mask.T, cmap="gray_r", origin="lower")  # .T 是為了轉置讓 x 軸橫向
    plt.title("Two Rooms Layout (White=Wall, Black=Air)")
    plt.colorbar()
    plt.show()


def calcu_gui_size(raw_w, raw_h, max_display_size=None):
    """
    計算最終顯示尺寸
    Returns: (target_w, target_h) 均為整數，且保證 >= 1
    """
    # 預設為原始尺寸
    target_w, target_h = raw_w, raw_h

    # 執行縮放計算
    if max_display_size and max_display_size > 0:
        max_side = max(raw_w, raw_h)
        if max_side > max_display_size:
            scale_ratio = max_display_size / max_side
            target_w = int(raw_w * scale_ratio)
            target_h = int(raw_h * scale_ratio)

    # 防呆機制：確保不會出現 0x0 導致 FFmpeg 崩潰
    target_w = max(1, target_w)
    target_h = max(1, target_h)

    return target_w, target_h * 2


def apply_resize(img, target_w, target_h):
    """
    執行影像縮放
    只有在尺寸不同時才呼叫 cv2.resize，節省效能
    """
    current_h, current_w = img.shape[:2]

    if current_w == target_w and current_h == target_h:
        return img

    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def draw_zone_overlay(gui, zones, split_ratio=0.5, y_offset=0.0):
    nx, ny = zones["nx"], zones["ny"]

    def to_gui(x, y):
        u = x / nx
        v = (y / ny) * split_ratio + y_offset
        return [u, v]

    # --- 1. 阻尼層 - 綠色 ---
    sp_in = zones["sponge_in"]
    sp_out = zones["sponge_out"]
    sp_top = zones["sponge_top"]
    sp_bot = zones["sponge_bot"]
    color_sponge = 0x00FF00

    gui.line(
        begin=to_gui(sp_in, 0), end=to_gui(sp_in, ny), color=color_sponge, radius=2
    )
    gui.line(
        begin=to_gui(sp_out, 0), end=to_gui(sp_out, ny), color=color_sponge, radius=2
    )
    gui.line(
        begin=to_gui(0, sp_bot), end=to_gui(nx, sp_bot), color=color_sponge, radius=2
    )
    gui.line(
        begin=to_gui(0, sp_top), end=to_gui(nx, sp_top), color=color_sponge, radius=2
    )

    # --- 2. ROI - 紅色 ---
    color_roi = 0xFF0000
    x0, y0 = to_gui(zones["roi_x_start"], zones["roi_y_start"])
    x1, y1 = to_gui(zones["roi_x_end"], zones["roi_y_end"])

    gui.line([x0, y0], [x1, y0], color=color_roi, radius=3)
    gui.line([x1, y0], [x1, y1], color=color_roi, radius=3)
    gui.line([x1, y1], [x0, y1], color=color_roi, radius=3)
    gui.line([x0, y1], [x0, y0], color=color_roi, radius=3)

    # --- 3. 標籤 ---
    gui.text("Dataset ROI", pos=[x0, y1 + 0.02], color=color_roi, font_size=20)
    gui.text(
        "Sponge", pos=to_gui(sp_out + 10, ny / 2), color=color_sponge, font_size=20
    )
