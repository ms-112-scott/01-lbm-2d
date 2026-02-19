from typing import Dict

def build_summary_entry(config: Dict, lattice_metadata: Dict, physical_params: Dict, source_files: Dict) -> Dict:
    """
    Constructs the final, structured dictionary for a single simulation case summary.
    This version separates lattice inputs from simulation outputs for clarity and formats
    floating-point numbers for better readability.
    """
    sim_name = config.get("simulation", {}).get("name", "UnknownCase")
    
    # --- Format Lattice Inputs ---
    lat_in = {
        "target_re": config.get("outputs", {}).get("target_re"),
        "characteristic_length_px": config.get("simulation", {}).get("characteristic_length"),
        "inlet_velocity_lu": round(config.get("boundary_condition", {}).get("value", [[0,0]])[0][0], 4),
        "kinematic_viscosity_lu": round(config.get("simulation", {}).get("nu"), 6),
        "resolution_px": [config.get("simulation", {}).get("nx"), config.get("simulation", {}).get("ny")]
    }

    # --- Format Simulation Outputs ---
    sim_out = {
         "actual_reynolds_number": round(lattice_metadata.get("reynolds_number_lattice_actual", 0), 2),
         "total_steps_executed": lattice_metadata.get("total_steps_executed"),
    }

    # --- Format Physical Scaled Parameters ---
    phys_p = physical_params
    phys_scaled = {
        "reynolds_number_calculated": round(phys_p.get("reynolds_number_calculated", 0), 2),
        "characteristic_length_m": round(phys_p.get("characteristic_length_m", 0), 4),
        "inlet_velocity_ms": round(phys_p.get("inlet_velocity_ms", 0), 2),
        "kinematic_viscosity_air_m2_s": f'{phys_p.get("kinematic_viscosity_air_m2_s", 0):.2e}',
        "cell_size_m": f'{phys_p.get("cell_size_m", 0):.4e}',
        "time_step_s": f'{phys_p.get("time_step_s", 0):.4e}',
        "steps_per_physical_second": round(phys_p.get("steps_per_physical_second", 0), 1),
        "total_simulation_time_s": round(phys_p.get("total_simulation_time_s", 0), 2),
    }
    
    summary_entry = {
        "case_name": sim_name,
        "status": "Success",
        "parameters": {
            "lattice_inputs": lat_in,
            "simulation_outputs": sim_out,
            "physical_scaled": phys_scaled
        },
        "run_summary": {
            "h5_file": lattice_metadata.get("h5_file"),
            "video_file": lattice_metadata.get("video_file")
        },
        "source_files": source_files
    }
    return summary_entry
