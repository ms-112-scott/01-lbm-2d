import json
import numpy as np

class NumpySafeJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that can safely handle common NumPy data types,
    preventing serialization errors when writing summary files.
    
    It converts NumPy types to their closest Python native equivalents.
    """
    def default(self, obj):
        # If the object is a NumPy integer (e.g., np.int32, np.int64), convert it to a Python int.
        if isinstance(obj, np.integer):
            return int(obj)
        
        # If the object is a NumPy float (e.g., np.float32, np.float64), convert it to a Python float.
        if isinstance(obj, np.floating):
            return float(obj)
            
        # If the object is a NumPy boolean (np.bool_), convert it to a Python bool.
        if isinstance(obj, np.bool_):
            return bool(obj)
            
        # If the object is a NumPy N-dimensional array, convert it to a nested Python list.
        if isinstance(obj, np.ndarray):
            return obj.tolist()
            
        # For any other types it doesn't know how to handle,
        # fall back to the default JSONEncoder behavior.
        return super(NumpySafeJSONEncoder, self).default(obj)
