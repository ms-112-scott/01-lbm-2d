### Day 1：資料池建置 (The Generator)

目標：不碰流體，只專注於幾何與參數，產出 1000 個靜態檔案。

- [ ] 實作 LHS 採樣器：整合我們討論過的 4 維抽樣與區間映射邏輯 (Rect/Complex/Urban/Indoor)。
- [ ] 實作 Mask 繪圖器：使用 OpenCV 繪製旋轉圖形，並實作室內窗戶挖洞邏輯。
- [ ] 執行 Sanity Check：加入 $\tau < 0.51$ 和 $D < 12$ 的剔除邏輯。
- [ ] 產出 Manifest：生成 dataset_pool_v1/，內含 1000 個子資料夾。
      檢核點：隨機打開 10 個 mask.npy，確認室內有窗戶，室外有邊界緩衝，且形狀沒有斷裂。

### Day 2：核心升級——MRT 矩空間與統計 (The Solver Core)

目標：修改 Taichi Kernel，讓它能「吐出」你需要的特殊數據。這是最困難的一天，需要修改 lbm_solver.py。

- [ ] 實作矩空間輸出 (Moment Output)：
      在 MRT 碰撞過程 ($m = M \cdot f$) 中，將碰撞後的 $m$ (Moment vector) 暫存下來。
      注意：不要存所有 timestep 的矩，會爆硬碟。通常是存 最後一幀 或 統計平均值。
- [ ] 實作線上統計 (On-the-fly Statistics)：
      我們不能存下每一秒的數據再來算平均。你需要定義新的 Taichi field：sum_u, sum_v, sum_u2, sum_v2, sum_uv。在 run_step 中，每隔固定步數 (e.g., 10 steps) 累加一次數值。
- [ ] 計算脈動量 (Fluctuation)：

利用公式：$\langle u'u' \rangle = \langle u^2 \rangle - \langle u \rangle^2$。存檔時再做這個減法計算。

### Day 3：邊界條件與場景切換 (The Scenario Switch)

目標：讓求解器變聰明，知道自己是在跑室內還是室外。

- [ ] 修改 apply_boundary_condition：
      讀取 config 中的 "scenario"。
      IF outdoor: 上下 wall 設為 Free-slip (copy velocity)。
      IF indoor: 上下 wall 設為 No-slip (bounce-back)，並讀取 window_info 設定 Inlet/outlet 位置。
- [ ] 測試跑通：手動跑一個室內 case 和一個室外 case，看流場是否合理（室內要有射流，室外要繞流）。

### Day 4：視覺化與 ROI 裁切 (The Visuals)

目標：確保看得到的和存下來的都是對的。

- [ ] 整合 GUI 畫框：把 utils.draw_zone_overlay 放進 main loop。
- [ ] 實作 ROI Cropping：

  存檔時，不只裁切 u, v，連同 Day 2 做出來的 moments, mean_u, reynolds_stress 全部都要根據紅框座標進行裁切。
  關鍵：確認裁切後的矩陣大小是固定的（這對 DL 很重要），或者紀錄裁切後的 shape。

### Day 5：自動化司機 (The Driver)

目標：串接所有環節。

- [ ] 撰寫 batch_manager.py：
      讀取 manifest.json。
      使用 subprocess 呼叫 main.py。
      加入 try...except 捕捉崩潰，確保腳本不會停。
- [ ] 磁碟空間估算：
      跑一個 Case，看 .npz 多大。
      假設一個 Case 50MB，1000 個就是 50GB。確認你的硬碟夠大。

### Day 6：飛行前演練 (Pilot Run)

目標：真實模擬，發現 Bug。

- [ ] 挑選 20 個代表性樣本：包含 5 個室內、5 個室外、5 個高 Re、5 個低 Re。
- [ ] 全流程執行：從生成 pool 到 batch run 全部跑一次。
- [ ] 數據驗收：

檢查 .npz 裡的 keys 是否包含 m_moments, mean_u, fluctuation_uv。
檢查數據有沒有 NaN 或 Inf。
檢查 ROI 裁切是否切到了障礙物（是否切太小？）。

### Day 7：緩衝與最終發射 (Launch Day)

- [ ] 修復 Bug：處理 Day 6 發現的問題。
- [ ] 清空暫存：刪除測試用的 data。
- [ ] 啟動：nohup python batch_manager.py > run.log 2>&1 & (如果你在 Linux 上) 或直接執行。
- [ ] 去喝杯咖啡：讓 GPU 為你工作。
