import numpy as np
import cv2

def add_circle(grid, cx, cy, r):
    """Adds a circle to the grid."""
    h, w = grid.shape
    y, x = np.ogrid[:h, :w]
    mask = (x - cx)**2 + (y - cy)**2 <= r**2
    grid[mask] = 1

def add_rotated_rect(grid, cx, cy, rect_w, rect_h, angle_deg):
    """Adds a rotated rectangle/square to the grid."""
    rect = ((cx, cy), (rect_w, rect_h), angle_deg)
    box = np.int64(cv2.boxPoints(rect))
    cv2.drawContours(grid, [box], 0, 1, -1)

def add_triangle(grid, cx, cy, size, angle_deg, orientation='vertex_left'):
    """
    Adds an equilateral triangle to the grid.

    Args:
        cx, cy: Center of the triangle.
        size: The distance from the center to a vertex.
        angle_deg: Rotation angle.
        orientation: 'vertex_left' or 'edge_left'.
    """
    # Base rotation to align orientation
    if orientation == 'vertex_left':
        base_angle_rad = np.deg2rad(-90)
    else:  # 'edge_left'
        base_angle_rad = np.deg2rad(90)

    total_angle_rad = base_angle_rad + np.deg2rad(angle_deg)
    
    # Vertices of an upward-pointing equilateral triangle centered at origin
    # The size is the distance from the center to a vertex (circumradius)
    p1 = np.array([0, -size])
    p2 = np.array([-size * np.sqrt(3) / 2, size / 2])
    p3 = np.array([size * np.sqrt(3) / 2, size / 2])

    # Rotation matrix
    c, s = np.cos(total_angle_rad), np.sin(total_angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]])

    # Rotate and translate points
    points = np.array([p1, p2, p3])
    rotated_points = points @ rot_matrix.T
    translated_points = np.int64(rotated_points + np.array([cx, cy]))

    cv2.drawContours(grid, [translated_points], 0, 1, -1)
