# Taichi LBM-2D: High-Fidelity CFD Dataset Generator

A high-performance, GPU-accelerated **Multiple Relaxation Time (MRT) Lattice Boltzmann Method (LBM)** solver implemented in Taichi. This project is specifically designed to generate large-scale, high-fidelity fluid dynamics datasets (velocity, pressure, MRT moments, SDF) for training AI surrogate models like Neural Cellular Automata (NCA) and Fourier Neural Operators (FNO).

## 🚀 Key Features

- **GPU Acceleration**: Leverages Taichi Lang for massively parallel computation on CUDA/Vulkan/Metal.
- **MRT-LES Model**: Combines Multiple Relaxation Time collision for stability and Smagorinsky LES for turbulence at high Reynolds numbers.
- **AI-Ready Output**: Exports HDF5 files containing 9-channel MRT moments, Signed Distance Fields (SDF), and accumulated statistics.
- **Automated Pipeline**: Includes procedural geometry generators and batch runners for unsupervised dataset expansion, organized by project.

## 🛠 New Workflow

The workflow is now organized by project. All inputs (masks, configs) and outputs (raw data, videos, plots) for a given experiment are centralized under a `project_name`.

### 1. Setup & Installation

First, clone the repository and install dependencies.

```bash
git clone https://github.com/ms-112-scott/01-lbm-2d.git
cd 01-lbm-2d

# Create a virtual environment and activate it
python -m venv .venv
source .venv/bin/activate  # For Linux/macOS
# .\.venv\Scripts\activate  # For Windows

pip install -r requirements.txt
```

### 2. Data Generation (Masks & Configs)

Define your project settings in `master_config.yaml`, then run the generation scripts.

```bash
# Step 1: Generate geometry masks for your project
# This creates SimCases/{project_name}/masks/
python src/tools/hybrid_map_gen.py

# Step 2: Generate simulation configs from the masks
# This creates SimCases/{project_name}/configs/
python src/tools/config_batch_gen.py
```
*   **Input Location**: `SimCases/{project_name}/`
*   **Configuration**: All settings are controlled by `master_config.yaml`.

### 3. Run Batch Simulation

Execute the main runner, referencing your `project_name`. The script will automatically find the inputs in `SimCases/` and save results to `outputs/`.

```bash
# Step 3: Run the full simulation batch for your project
python -m src.lbm_mrt_les.runners.run_multi_case --project_name Hyper
```
*   **Output Location**: `outputs/{project_name}/` (containing `raw`, `vis`, `plots`, and a summary JSON).

---

## 📖 Documentation

Detailed documentation is available in the `docs/` directory (mostly in Traditional Chinese).

### [00. Project Overview](./docs/00_專案總覽/00_文檔索引.md)
... (documentation links remain the same) ...

---

### Research Team
... (team info remains the same) ...

### The Ecosystem (Polyrepo)
... (ecosystem links remain the same) ...

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
