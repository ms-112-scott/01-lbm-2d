import numpy as np
import cv2

def check_sdf_validity(grid, new_box_points, min_dist):
    """
    Checks if placing a new shape violates the minimum distance requirement,
    using a Signed Distance Field.
    """
    if np.sum(grid) == 0:
        return True
    
    # Calculate SDF from existing obstacles
    inv_grid = (1 - grid).astype(np.uint8)
    sdf = cv2.distanceTransform(inv_grid, cv2.DIST_L2, 5)
    
    # Get the area of the new shape
    new_mask = np.zeros_like(grid)
    cv2.drawContours(new_mask, [new_box_points], 0, 1, -1)
    
    # Check the SDF values under the new shape's mask
    covered_sdf = sdf[new_mask == 1]
    
    # If there are no points (e.g., shape is off-grid) or the minimum distance
    # is met, it's valid.
    return len(covered_sdf) == 0 or np.min(covered_sdf) >= min_dist

def check_blockage_ratio(grid, new_box_points, max_ratio):
    """
    Checks if adding a new shape exceeds the maximum vertical blockage ratio.
    """
    temp_grid = grid.copy()
    cv2.drawContours(temp_grid, [new_box_points], 0, 1, -1)
    
    # Find the max value for each row. If any obstacle is present, it will be 1.
    blocked_height = np.sum(np.max(temp_grid, axis=1))
    
    grid_height = grid.shape[0]
    return (blocked_height / grid_height) <= max_ratio
