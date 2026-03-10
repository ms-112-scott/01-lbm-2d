"""
src/lbm_mrt_les/io/case_vector_builder.py

Reads the completed all_cases_summary.json and serialises every case into a
flat float32 feature vector, then saves the whole batch as a single .npz file.

Output arrays inside the .npz
──────────────────────────────
  vectors      : float32  (N, D)   – numeric feature matrix; NaN for failed cases
  case_names   : str      (N,)     – matches case_name in the JSON
  statuses     : str      (N,)     – "Success" | "Failed" | …
  feature_names: str      (D,)     – one label per column of vectors

Designed to drop straight into a PyTorch / sklearn Dataset.
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Feature schema
# Order is stable: add new features at the END to preserve backward compat.
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES: List[str] = [
    # ── lattice_inputs ──────────────────────────────────────────────────────
    "lat_rho_in",                   # target inlet density           [lu]
    "lat_rho_out",                  # outlet density                 [lu]
    "lat_characteristic_length_px", # L_char                         [px]
    "lat_inlet_velocity_lu",        # measured inlet velocity        [lu]
    "lat_kinematic_viscosity_lu",   # ν                              [lu]
    "lat_nx",                       # domain width                   [px]
    "lat_ny",                       # domain height                  [px]
    # ── simulation_outputs ──────────────────────────────────────────────────
    "sim_actual_reynolds_number",   # Re = U·L/ν (lattice)
    "sim_total_steps_executed",     # total LBM time steps
    "sim_tensor_T",                 # turbulence dataset: # frames
    "sim_tensor_C",                 # turbulence dataset: channels (9)
    "sim_tensor_H",                 # turbulence dataset: height     [px]
    "sim_tensor_W",                 # turbulence dataset: width      [px]
    # ── physical_scaled ─────────────────────────────────────────────────────
    "phys_reynolds_number",         # Re (physical, cross-check)
    "phys_characteristic_length_m", # L_char                         [m]
    "phys_inlet_velocity_ms",       # U_inlet                        [m/s]
    "phys_kinematic_viscosity_m2s", # ν_air                          [m²/s]
    "phys_cell_size_m",             # dx                             [m]
    "phys_time_step_s",             # dt                             [s]
    "phys_steps_per_second",        # 1/dt
    "phys_total_simulation_time_s", # total physical time            [s]
]

D = len(FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value, fallback: float = np.nan) -> float:
    """Convert a value (possibly a string like '1.23e-04') to float safely."""
    if value is None:
        return fallback
    try:
        return float(value)
    except (ValueError, TypeError):
        return fallback


def _extract_vector(entry: Dict) -> np.ndarray:
    """
    Extract a (D,) float32 vector from a single summary entry.
    Returns all-NaN for failed / incomplete cases.
    """
    vec = np.full(D, np.nan, dtype=np.float32)

    params = entry.get("parameters", {})
    lat   = params.get("lattice_inputs", {})
    sim   = params.get("simulation_outputs", {})
    phys  = params.get("physical_scaled", {})

    res = lat.get("resolution_px") or [np.nan, np.nan]

    turb_shape = (sim.get("tensor_shapes") or {}).get("turbulence") or [np.nan, np.nan, np.nan, np.nan]
    # turbulence shape is [T, C, H, W]
    t_T = _safe_float(turb_shape[0] if len(turb_shape) > 0 else np.nan)
    t_C = _safe_float(turb_shape[1] if len(turb_shape) > 1 else np.nan)
    t_H = _safe_float(turb_shape[2] if len(turb_shape) > 2 else np.nan)
    t_W = _safe_float(turb_shape[3] if len(turb_shape) > 3 else np.nan)

    values = [
        # lattice_inputs
        _safe_float(lat.get("rho_in")),
        _safe_float(lat.get("rho_out")),
        _safe_float(lat.get("characteristic_length_px")),
        _safe_float(lat.get("inlet_velocity_lu")),
        _safe_float(lat.get("kinematic_viscosity_lu")),
        _safe_float(res[0] if len(res) > 0 else np.nan),
        _safe_float(res[1] if len(res) > 1 else np.nan),
        # simulation_outputs
        _safe_float(sim.get("actual_reynolds_number")),
        _safe_float(sim.get("total_steps_executed")),
        t_T, t_C, t_H, t_W,
        # physical_scaled  (stored as e-notation strings → parse with _safe_float)
        _safe_float(phys.get("reynolds_number_calculated")),
        _safe_float(phys.get("characteristic_length_m")),
        _safe_float(phys.get("inlet_velocity_ms")),
        _safe_float(phys.get("kinematic_viscosity_air_m2_s")),
        _safe_float(phys.get("cell_size_m")),
        _safe_float(phys.get("time_step_s")),
        _safe_float(phys.get("steps_per_physical_second")),
        _safe_float(phys.get("total_simulation_time_s")),
    ]

    assert len(values) == D, f"Feature count mismatch: {len(values)} vs {D}"
    vec[:] = values
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_npz(summary_json_path: str, npz_output_path: str) -> str:
    """
    Read *summary_json_path*, build one feature vector per case, and write
    a .npz file to *npz_output_path*.

    Returns the path of the written file.
    """
    if not os.path.exists(summary_json_path):
        raise FileNotFoundError(f"[CaseVectorBuilder] Summary JSON not found: {summary_json_path}")

    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    if not summary_data:
        print("[CaseVectorBuilder] Warning: summary JSON is empty – no NPZ written.")
        return ""

    n = len(summary_data)
    vectors      = np.full((n, D), np.nan, dtype=np.float32)
    case_names   = np.empty(n, dtype=object)
    statuses     = np.empty(n, dtype=object)
    feature_names = np.array(FEATURE_NAMES, dtype=object)

    success_count = 0
    for idx, entry in enumerate(summary_data):
        case_names[idx] = entry.get("case_name", f"case_{idx:04d}")
        statuses[idx]   = entry.get("status", "Unknown")

        if statuses[idx] == "Success":
            vectors[idx] = _extract_vector(entry)
            success_count += 1
        # Failed cases keep NaN rows – intentional, keeps index alignment with JSON

    os.makedirs(os.path.dirname(npz_output_path), exist_ok=True)

    np.savez_compressed(
        npz_output_path,
        vectors       = vectors,
        case_names    = case_names,
        statuses      = statuses,
        feature_names = feature_names,
    )

    print(
        f"[CaseVectorBuilder] Saved {n} cases ({success_count} success / "
        f"{n - success_count} failed) → {npz_output_path}"
    )
    print(f"[CaseVectorBuilder] Vector shape: {vectors.shape}  "
          f"({D} features per case)")
    return npz_output_path
