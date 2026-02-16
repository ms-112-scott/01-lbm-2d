# Taichi LBM-2D: High-Fidelity CFD Dataset Generator

A high-performance, GPU-accelerated **Multiple Relaxation Time (MRT) Lattice Boltzmann Method (LBM)** solver implemented in Taichi. This project is specifically designed to generate large-scale, high-fidelity fluid dynamics datasets (velocity, pressure, MRT moments, SDF) for training AI surrogate models like Neural Cellular Automata (NCA) and Fourier Neural Operators (FNO).

## 🚀 Key Features

- **GPU Acceleration**: Leverages Taichi Lang for massively parallel computation on CUDA/Vulkan/Metal.
- **MRT-LES Model**: Combines Multiple Relaxation Time collision for stability and Smagorinsky LES for turbulence at high Reynolds numbers.
- **AI-Ready Output**: Exports HDF5 files containing 9-channel MRT moments, Signed Distance Fields (SDF), and accumulated statistics.
- **Automated Pipeline**: Includes procedural geometry generators and batch runners for unsupervised dataset expansion.

## 🛠 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ms-112-scott/01-lbm-2d.git
cd 01-lbm-2d

# Install dependencies
python3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Running a Simulation

```bash
python -m src.lbm_mrt_les.runners.run_one_case \
    --config configs/templates/config_template.yaml \
    --mask_dir src/tools/rect_masks

```

```bash
python -m src.lbm_mrt_les.runners.run_multi_case \
    --config_dir configs/Hyper \
    --mask_dir mask/Hyper

```

---

## 📖 Documentation

Detailed documentation is available in the `docs/` directory (mostly in Traditional Chinese).

### [00. Project Overview](./docs/00_專案總覽/00_文檔索引.md)

- [Project Introduction](./docs/00_專案總覽/01_專案簡介.md)
- [System Architecture & Core Modules](./docs/00_專案總覽/02_系統架構與核心模組.md)

### [01. Setup & Configuration](./docs/01_安裝與配置/01_模擬環境設定.md)

- [Environment Setup](./docs/01_安裝與配置/01_模擬環境設定.md)
- [Configuration Details (YAML)](./docs/01_安裝與配置/02_案例管理與配置詳解.md)

### [02. User Guide](./docs/02_操作指南/01_幾何場景準備.md)

- [Geometry Preparation (Masks)](./docs/02_操作指南/01_幾何場景準備.md)
- [Running Simulations & Visualization](./docs/02_操作指南/02_執行模擬與視覺化.md)
- [Advanced Batch Processing](./docs/02_操作指南/03_多通道模擬操作手冊.md)

### [03. Data & Outputs](./docs/03_數據結構與輸出/01_HDF5數據結構說明.md)

- [HDF5 Data Structure](./docs/03_數據結構與輸出/01_HDF5數據結構說明.md)
- [Data Pipeline Development](./docs/03_數據結構與輸出/04_數據管線開發進度.md)

### [04. Theory & Physics](./docs/04_理論基礎/01_LBM_MRT理論基礎.md)

- [LBM-MRT Theoretical Foundation](./docs/04_理論基礎/01_LBM_MRT理論基礎.md)
- [Numerical Stability Analysis](./docs/04_理論基礎/02_數值穩定性分析.md)

---

### Research Team

**Bo-Xuan Lu (呂博軒)** _M.S. Student (112)_ Graduate Institute of Architecture, National Yang Ming Chiao Tung University (NYCU), Taiwan  
[apc582nntscott@arch.nycu.edu.tw](mailto:apc582nntscott@arch.nycu.edu.tw)  
ORCID: [0009-0002-5308-4810](https://orcid.org/0009-0002-5308-4810)

**Assoc. Prof. June-Hao Hou (侯君昊)** _Advisor / Principal Investigator_ Graduate Institute of Architecture, National Yang Ming Chiao Tung University (NYCU), Taiwan  
[jhou@arch.nycu.edu.tw](mailto:jhou@arch.nycu.edu.tw)  
ORCID: [0000-0002-8362-7719](https://orcid.org/0000-0002-8362-7719)

### Related Publications

If you use this solver or the generated datasets, please cite:

- **[CAADRIA 2025]** _Neural Cellular Automata for Dynamic Ventilation in Architectural Spaces_ ([DOI: 10.52842/conf.caadria.2025.3.325](https://doi.org/10.52842/conf.caadria.2025.3.325))
- **[WIP]** _Modular Neural Cellular Automata (m-NCA): A Physics-Informed Framework for Real-Time Dynamic Simulation in Architectural Design_

### The Ecosystem (Polyrepo)

This project focuses on **Data Generation**. For the full AI pipeline, see:

1. **[01-lbm-2d](https://github.com/ms-112-scott/01-lbm-2d.git)**: Data Generation (This Repo)
2. **[02-nca-cfd](https://github.com/ms-112-scott/02-nca-cfd.git)**: Model Training (NCA)
3. **[03-gh-frontend](https://github.com/ms-112-scott/03-gh-frontend.git)**: Rhino/Grasshopper Integration

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
