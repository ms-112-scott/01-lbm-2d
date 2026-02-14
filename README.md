# Taichi LBM 2D: High-Fidelity Dataset Generator for AI Fluid Dynamics

A high-performance, GPU-accelerated **Multiple Relaxation Time (MRT) Lattice Boltzmann** solver. Engineered specifically for generating large-scale, ML-ready CFD datasets to train neural operators and surrogate models.

---

## 🔬 Academic Context & Attribution

This repository is a core component of a research framework for **AI-driven architectural wind environment simulation**. It is a customized and extended fork of the [LBM_Taichi](https://github.com/hietwll/LBM_Taichi.git) project.

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

## 🌟 Key Features

### 1. Advanced Physics Engine

- **MRT Collision Model**: Decoupled relaxation rates for enhanced numerical stability at high Reynolds numbers compared to BGK.
- **LES (Smagorinsky)**: Sub-grid scale turbulence modeling for capturing transient flow features.
- **Acoustic Sponge Layers**: Effective absorption of pressure wave reflections at boundaries to maintain domain integrity.

### 2. AI-Native Infrastructure

- **HDF5 Integration**: Optimized I/O for high-speed training access.
- **9-Component Moments**: Beyond primitive variables (), we export full MRT moments (Energy, Stress Tensors) for physics-informed learning.
- **Automated Batching**: Procedural mask generation and configuration pairing for unsupervised dataset expansion.

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/ms-112-scott/01-lbm-2d.git
cd 01-lbm-2d

# Install dependencies (Taichi, H5py, PyYAML, OpenCV)
pip install -r requirements.txt

```

---

## 🚀 Workflow Quick Start

### Step 1: Geometry Generation

Generate random rectangular obstacle masks to create structural diversity in your dataset:

```bash
python src/generators/mask_rect_gen.py

```

### Step 2: Production Run

Execute simulations for all masks in a directory using a template configuration:

```bash
python -m src.lbm_mrt_les.runners.run_one_case \
    --config src/configs/config_template.yaml \
    --mask_dir src/generators/rect_masks

```

### Step 3: Analytics & Labeling

Calculate Time-Averaged (RANS-like) fields for steady-state surrogate training:

```bash
python src/post_process/rans_calc.py

```

---

## 📊 Data Specification

Generated `.h5` files follow the `(Time, Channels, H, W)` tensor format:

| Channel | Description           | Symbol |
| ------- | --------------------- | ------ |
| 0       | Density               |        |
| 1 - 2   | Energy & Energy Sq.   |        |
| 3, 5    | Momentum              |        |
| 4, 6    | Heat Flux             |        |
| 7 - 8   | Normal & Shear Stress |        |

---

## 📂 Project Structure

```text
src/
├── lbm_mrt_les/         # Core Physics Engine
│   ├── engine/          # Taichi Kernels (Collision, Streaming, BCs)
│   ├── runners/         # Execution Logic (Batch/Single)
│   └── io/              # HDF5 & Visualization Handlers
├── generators/          # Dataset Synthesis Tools (Masks/Configs)
└── post_process/        # Statistical Analysis & Label Prep

```

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
