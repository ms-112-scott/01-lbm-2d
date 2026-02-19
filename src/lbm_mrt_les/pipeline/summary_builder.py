from typing import Dict

def build_summary_entry(config: Dict, lattice_metadata: Dict, physical_params: Dict, source_files: Dict) -> Dict:
    """
    Constructs the final, structured dictionary for a single simulation case summary.
    
    Args:
        config: The configuration dictionary for the case.
        lattice_metadata: The dictionary of raw outputs from the lattice simulation.
        physical_params: The dictionary of calculated real-world physical parameters.
        source_files: A dictionary containing the names of the config and mask files.
        
    Returns:
        A comprehensive, structured dictionary for this case.
    """
    sim_name = config.get("simulation", {}).get("name", "UnknownCase")
    
    summary_entry = {
        "case_name": sim_name,
        "status": "Success",
        "parameters": {
            "lattice": {
                "reynolds_number": lattice_metadata.get("reynolds_number_lattice_actual"),
                "characteristic_length_px": lattice_metadata.get("l_char_lattice_px"),
                "inlet_velocity_lu": lattice_metadata.get("u_inlet_lattice_lu"),
                "kinematic_viscosity_lu": lattice_metadata.get("nu_lattice_lu"),
                "resolution_px": [lattice_metadata.get("nx"), lattice_metadata.get("ny")],
                "time_step_dt": 1.0,
                "cell_size_dx": 1.0
            },
            "physical": physical_params
        },
        "run_summary": {
            "total_steps_executed": lattice_metadata.get("total_steps_executed"),
            "h5_file": lattice_metadata.get("h5_file"),
            "video_file": lattice_metadata.get("video_file")
        },
        "source_files": source_files
    }
    return summary_entry
