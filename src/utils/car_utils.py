import numpy as np
from scipy.interpolate import splprep, splev

def smooth_path_2d(waypoints, num_points):
    tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], s=0)
    u_fine = np.linspace(0, 1, num_points)
    smooth_points = splev(u_fine, tck)
    return smooth_points[0], smooth_points[1]

def rotation_matrix_2d(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

def transform_coordinates_2d(coords, rotation_matrix, translation):
    coords = np.array(coords)
    rotated_coords = np.dot(rotation_matrix, coords.T).T
    translated_coords = rotated_coords + translation
    return translated_coords.tolist()