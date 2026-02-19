import os
from typing import Dict

def get_project_paths(project_name: str) -> Dict[str, str]:
    """
    Defines and returns all essential paths for a given project name.
    """
    # Input paths are located in the SimCases directory
    project_base_dir = os.path.join("SimCases", project_name)
    config_dir = os.path.join(project_base_dir, "configs")
    mask_dir = os.path.join(project_base_dir, "masks")
    
    # Output paths are located in the outputs directory
    output_project_path = os.path.join("outputs", project_name)

    return {
        "project_base": project_base_dir,
        "configs": config_dir,
        "masks": mask_dir,
        "outputs": output_project_path,
    }

def setup_output_directories(base_output_path: str) -> Dict[str, str]:
    """
    Creates the standardized output directory structure (raw, vis, plots).
    """
    paths = {
        "base": base_output_path,
        "raw": os.path.join(base_output_path, "raw"),
        "vis": os.path.join(base_output_path, "vis"),
        "plots": os.path.join(base_output_path, "plots"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths
