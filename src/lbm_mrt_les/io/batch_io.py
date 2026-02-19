import json
from typing import List, Dict
from ..io.NumpySafeJSONEncoder import NumpySafeJSONEncoder

def save_summary_file(summary_data: List[Dict], output_path: str):
    """
    Saves the provided list of summary dictionaries to a JSON file.
    
    Args:
        summary_data: A list containing the summary dictionary for each case.
        output_path: The full path where the JSON file will be saved.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, cls=NumpySafeJSONEncoder)
        print(f"
[Done] Saved batch summary to: {output_path}")
    except Exception as e:
        print(f"
[Error] Could not save summary file: {e}")
