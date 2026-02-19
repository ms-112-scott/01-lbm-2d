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
python -m src.lbm_mrt_les.pipeline.batch_run --project_name Hyper-1
```
*   **Output Location**: `outputs/{project_name}/` (containing `raw`, `vis`, `plots`, and a summary JSON).

---

## Academic Context & Attribution

This repository serves as a core component of the research framework for **AI-driven architectural wind environment simulation**.

It is developed as a customized and extended fork of the [LBM_Taichi](https://github.com/hietwll/LBM_Taichi.git) project, originally authored by _hietwll_. Modifications focus on integrating architectural boundary conditions and optimizing data flow for deep learning applications.

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