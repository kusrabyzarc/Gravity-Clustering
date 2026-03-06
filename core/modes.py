import numpy as np
from scipy.ndimage import maximum_filter


def find_modes(field, X, Y, min_distance_px=20, threshold_rel=0.2):
    size = 2 * min_distance_px + 1
    max_mask = maximum_filter(field, size=size) == field
    threshold = threshold_rel * field.max()

    coords = np.argwhere(max_mask & (field >= threshold))

    modes = []
    for iy, ix in coords:
        modes.append((X[iy, ix], Y[iy, ix]))

    return np.array(modes)


def find_modes_nd(shifted_points):
    """
    In N-D mode-seeking we treat converged shifted points as mode candidates.
    """
    shifted_points = np.asarray(shifted_points, dtype=float)
    if shifted_points.ndim != 2:
        raise ValueError("shifted_points must be a 2D array: (n_samples, n_features)")
    return shifted_points
