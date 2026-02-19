import json
from typing import List, Dict
from .NumpySafeJSONEncoder import NumpySafeJSONEncoder
import os

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
        print(f"[Done] Saved batch summary to: {output_path}")
    except Exception as e:
        print(f"[Error] Could not save summary file: {e}")

def init_summary_file(output_path: str):
    """Initializes an empty summary file."""
    save_summary_file([], output_path)

def update_summary_file(summary_entry: Dict, output_path: str):
    """
    Appends or updates a summary entry in the JSON file.
    If an entry with the same case_name exists, it updates it. Otherwise, it appends.
    """
    try:
        data = []
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        # Check if entry already exists (by case_name or source_files.config_file)
        target_name = summary_entry.get("case_name")
        found = False
        for i, entry in enumerate(data):
            if entry.get("case_name") == target_name:
                data[i] = summary_entry
                found = True
                break
        
        if not found:
            data.append(summary_entry)
            
        save_summary_file(data, output_path)
    except Exception as e:
        print(f"[Error] Could not update summary file: {e}")
