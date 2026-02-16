# 專案目錄結構重整建議 (Project Structure Reorganization Proposal)

## 1. 現狀分析 (Current Status Analysis)

目前的專案結構包含核心代碼 (`src`)、舊代碼備份 (`old`)、文件 (`my_docs`) 以及多個輸出目錄 (`output`, `output_benchmarks`, `output_postprocess`)。

**主要觀察:**
*   **文件目錄命名**: `my_docs` 較非標準，通常建議使用 `docs`。
*   **舊代碼**: `old` 目錄包含多個舊版本 (`lbm_srt`, `lbm_mrt` 等)，建議統一歸檔。
*   **輸出分散**: 模擬結果分散在 `output`、`output_benchmarks` 和 `output_postprocess`，建議整合。
*   **源碼結構**: `src` 內部結構尚可，但 `configs` 若能移至根目錄會更便於管理實驗配置。
*   **Notebooks 散落**: `.ipynb` 檔案散落在 `src` 與 `post_process` 中，建議集中管理。

---

## 2. 建議的新結構 (Proposed Structure)

```text
Project_Root/
├── archive/                     # [歸檔] 存放 old/ 及其他不再維護的舊代碼
│   ├── lbm_srt/
│   ├── lbm_mrt_legacy/
│   └── ...
├── configs/                     # [配置] 存放所有 .yaml 配置文件 (從 src/configs 移出)
│   ├── templates/               # 基礎模板 (config_template.yaml)
│   ├── experiments/             # 具體實驗配置 (hyper_configs)
│   └── benchmarks/              # 基準測試配置
├── docs/                        # [文件] (原 my_docs)
│   ├── development/             # 開發指引、研究規劃
│   ├── api/                     # API 文檔
│   └── references/              # 參考文檔
├── notebooks/                   # [實驗] 存放 Jupyter Notebooks (從 src 移出)
│   ├── exploration/             # 探索性分析
│   └── visualization/           # 繪圖與後處理演示
├── outputs/                     # [輸出] 統一輸出目錄
│   ├── simulation_data/         # (原 output/dataset)
│   ├── benchmarks/              # (原 output_benchmarks)
│   ├── visualization/           # 影片、圖片
│   └── logs/                    # 執行日誌
├── src/                         # [源碼] 核心 Python 包
│   ├── lbm_solver/              # (建議將 lbm_mrt_les 重命名為更通用的名稱，或保持)
│   │   ├── core/                # 核心運算 (engine)
│   │   ├── boundary/            # 邊界條件處理
│   │   └── utils/               # 求解器專用工具
│   ├── tools/                   # (原 generators) 地圖與遮罩生成工具
│   ├── analysis/                # (原 post_process) 後處理與分析工具
│   └── visualization/           # (原 io 中與視覺化相關的部分)
├── tests/                       # [測試] 單元測試與整合測試
├── scripts/                     # [執行] 頂層執行腳本 (如 run.py, batch_run.py)
├── .gitignore
├── requirements.txt
├── README.md
└── GEMINI.md
```

---

## 3. 詳細調整說明 (Detailed Changes)

### 3.1 核心目錄 (`src`)
*   **模組化**: 保持 `lbm_mrt_les` 作為核心包，但建議內部區分 `core` (運算核心), `io` (數據讀寫), `boundary` (邊界處理)。
*   **工具分離**: 將 `generators` 重命名為 `tools` 或 `preprocessing`，明確其作為預處理工具的定位。
*   **後處理**: `post_process` 重命名為 `analysis`，專注於數據分析邏輯，與視覺化 (`visualization`) 分離。

### 3.2 配置管理 (`configs`)
*   將 `src/configs` 移至根目錄下的 `configs/`。
*   **優點**: 使用者在執行實驗時，不需要進入源碼目錄修改配置，且路徑更直觀。

### 3.3 文檔與筆記 (`docs` & `notebooks`)
*   將 `my_docs` 重命名為 `docs`，符合開源專案慣例。
*   建立 `notebooks` 目錄，將散落在 `src` 中的 `.ipynb` 檔案（如 `LHS_sampling.ipynb`）移入，避免源碼目錄混雜實驗性腳本。

### 3.4 統一輸出 (`outputs`)
*   整合 `output`、`output_benchmarks`、`output_postprocess`。
*   建議結構：
    *   `outputs/data/`: 存放 `.h5` 或 `.npy` 原始數據。
    *   `outputs/figures/`: 存放生成的圖表。
    *   `outputs/videos/`: 存放生成的影片。
    *   `outputs/checkpoints/`: 存放模型或模擬的中斷點。

### 3.5 歸檔 (`archive`)
*   建立 `archive` 目錄，將 `old` 資料夾內容移入。
*   這能保持專案根目錄整潔，同時保留歷史代碼以供參考。

---

## 4. 執行步驟建議 (Action Plan)

1.  **建立新目錄**: `mkdir configs docs notebooks outputs archive tests scripts`
2.  **移動文件**:
    *   `mv my_docs/* docs/`
    *   `mv old/* archive/`
    *   `mv src/configs/* configs/`
    *   `mv src/*.ipynb notebooks/`
    *   `mv output* outputs/` (需小心合併子目錄)
3.  **重構源碼 (Optional)**:
    *   若移動了 `configs`，需更新 Python 腳本中的路徑讀取邏輯 (或是使用相對路徑/環境變數)。
    *   更新 `README.md` 與 `GEMINI.md` 中的路徑說明。
4.  **清理**: 刪除空的舊目錄。

---

## 5. 開發習慣建議

*   **路徑引用**: 在程式碼中盡量使用 `pathlib` 與相對專案根目錄的路徑，避免硬編碼 (Hard-coding)。
*   **版本控制**: 確保 `.gitignore` 包含 `outputs/`、`__pycache__/` 與 `.ipynb_checkpoints/`。
