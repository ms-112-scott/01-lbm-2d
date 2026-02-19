import random

def get_sampled_value(param_value):
    """
    Samples a value based on the configuration format.
    - If list of 2 [min, max]: uniform random sample.
    - If list of >2 [a, b, c]: random choice from the list.
    - Otherwise: return the value itself.
    """
    if not isinstance(param_value, list):
        return param_value

    if len(param_value) == 2:
        min_val, max_val = param_value
        # Ensure both are numbers and min_val <= max_val
        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)) and min_val <= max_val:
            # If both are integers, sample an integer. Otherwise, a float.
            if isinstance(min_val, int) and isinstance(max_val, int):
                return random.randint(min_val, max_val)
            else:
                return random.uniform(min_val, max_val)
        else:
            # Fallback for invalid range
            return None
    elif len(param_value) > 2:
        return random.choice(param_value)
    elif len(param_value) == 1:
        return param_value[0]
    else: # len == 0
        return None
