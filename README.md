## 📝 Academic Attribution & Context

This project is a customized fork of the [LBM_Taichi](https://github.com/hietwll/LBM_Taichi.git), specifically optimized for **AI-driven fluid dynamics research**.

- **Primary Purpose**: Generation of high-fidelity synthetic datasets for training neural operators and surrogate models.
- **Academic Citation**: This solver is part of a larger research framework. If you use this dataset generator for your research for more detail please see papers:
  > [Paper DOI 1, e.g., "Neural Cellular Automata for Dynamic Ventilation in Architectural Spaces"](https://doi.org/10.52842/conf.caadria.2025.3.325)
  > [Paper DOI 2, e.g., "Modular Neural Cellular Automata (m-NCA):A Physics-Informed Framework for Real-Time Dynamic Simulation in Architectural Design"](https://doi.org/placeholder)
- **Polyrepo Integration**: This repository focuses on **Data Generation**. For model training and evaluation, please refer to our complementary repos:
  [01-lbm-2d](https://github.com/ms-112-scott/01-lbm-2d.git)
  [02-nca-cfd](https://github.com/ms-112-scott/02-nca-cfd.git)
  [03-gh-frontend](https://github.com/ms-112-scott/03-gh-frontend.git)

---

# lbm_2d: Taichi-based MRT-LBM Solver for AI Surrogate Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Taichi](https://img.shields.io/badge/Taichi-Enabled-orange.svg)](https://taichi-lang.org/)

**lbm_2d** is a high-performance, GPU-accelerated 2D Lattice Boltzmann Method (LBM) solver implemented in [Taichi Lang](https://taichi-lang.org/). It is specifically designed to generate large-scale, high-fidelity Computational Fluid Dynamics (CFD) datasets for training AI surrogate models (e.g., Deep Learning based fluid prediction).

## Key Features

- **Advanced Physics:**
  - **MRT (Multiple Relaxation Time)** collision model for enhanced stability at high Reynolds numbers.
  - **LES (Large Eddy Simulation)** with Smagorinsky sub-grid scale model for turbulence modeling.
  - **Sponge Layers** at boundaries to minimize pressure wave reflections.
- **High Performance:** Fully GPU-accelerated simulation using Taichi's parallel backend (CUDA/Vulkan/Metal).
- **AI-Ready Datasets:**
  - Direct export to **HDF5 (`.h5`)** format.
  - Channels include: Velocity (Vx, Vy), Density (Rho), Vorticity, and **9-component MRT Moments**.
- **Robustness:** Built-in stability monitoring (NaN checks, force explosion detection) for reliable batch processing.
- **Flexible Geometry:**
  - Import arbitrary geometries via PNG masks.
  - Built-in procedural generators for random obstacle arrays (rectangles, cylinders).

---

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ms-112-scott/01-lbm-2d.git
    cd lbm_2d
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    _Note: This project relies on `taichi`, `h5py`, `opencv-python`, and `pyyaml`._

---

## Quick Start

### 1. Generate Obstacle Masks (Optional)

If you don't have your own geometry files, you can generate a set of random rectangular obstacles.

```bash
python src/generators/mask_rect_gen.py
```

- **Output:** PNG masks will be saved to `src/generators/rect_masks/`.

### 2. Run Simulation

#### Mode A: Single Configuration (Standard Batch)

Run simulations for all masks in a directory using a single configuration file.

```bash
python -m src.lbm_mrt_les.runners.run_one_case
    --config src/configs/config_template.yaml
    --mask_dir src/generators/rect_masks
```

#### Mode B: Advanced One-to-One Batch

Run simulations where each mask has a specific corresponding configuration file (e.g., for varying Reynolds numbers).

1.  **Generate paired configs:**
    ```bash
    python src/generators/config_batch_gen.py
    ```
2.  **Run the batch runner:**
    ```bash
    python -m src.lbm_mrt_les.runners.run_multi_case
        --config_dir src/configs/hyper_configs
        --mask_dir src/generators/rect_masks
    ```

### 3. Post-Processing & Visualization

To generate time-averaged flow fields or extract the last frame from simulation videos:

```bash
python src/post_process/rans_calc.py
```

- **Input:** Scans `output/` for `.mp4` files.
- **Output:** Saves `_AVG.png` and `_LAST.png` visualizations next to the video files.

---

## Configuration (`config.yaml`)

The simulation is controlled via YAML configuration files. Key parameters include:

```yaml
simulation:
  name: "Simulation_Batch_001"
  nx: 512 # Domain width
  ny: 256 # Domain height
  nu: 0.03 # Kinematic viscosity (controls Reynolds number)
  max_steps: 10000 # Maximum simulation steps

boundaries:
  # 0: Velocity Inlet, 1: Outflow, 2: Free Slip, 3: No Slip
  types: [0, 0, 1, 0] # Left, Top, Right, Bottom
  values:
    - [0.1, 0.0] # Inlet velocity vector (LBM units)

outputs:
  dataset:
    enable: true
    folder: "output/dataset" # Path for HDF5 files
    save_resolution: 256 # Output resolution (can downsample)
```

---

## Project Structure

The codebase is organized into modular components under `src/`:

```text
src/
├── lbm_mrt_les/              # Core Solver Package
│   ├── engine/               # Physics engine (LBM solver, Simulation loop)
│   ├── runners/              # Execution scripts (Single/Multi-case)
│   ├── io/                   # Input/Output (HDF5 writer, Visualization)
│   └── utils/                # Utility functions (Config, Physics, Math)
│
├── generators/               # Procedural Generation Tools
│   ├── mask_rect_gen.py      # Generate random rectangle masks
│   ├── config_batch_gen.py   # Generate config files for batch runs
│   └── hybrid_map_gen.py     # Advanced hybrid map generation
│
└── post_process/             # Post-processing Tools
    └── rans_calc.py          # RANS (Time-Averaged) calculation
```

## Output Data Format (.h5)

The generated HDF5 files contain time-series data suitable for machine learning.

- **Group:** `snapshots`
- **Shape:** `(Time, Channels, Height, Width)`
- **Channels (9 total):** 0. `Density` ($
ho$) 1. `Energy` ($e$) 2. `Energy Square` ($\epsilon$) 3. `Momentum X` ($j_x$) 4. `Heat Flux X` ($q_x$) 5. `Momentum Y` ($j_y$) 6. `Heat Flux Y` ($q_y$) 7. `Normal Stress` ($p_{xx}$) 8. `Shear Stress` ($p_{xy}$)

_Note: Velocity ($u, v$) can be derived from Momentum and Density._

## License

This project is open-source and available under the MIT License.
