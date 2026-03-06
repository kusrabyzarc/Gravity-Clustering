import numpy as np

from . import density, modes, merge, clustering


def cluster_nd(
    points,
    sigma=0.8,
    merge_radius=0.6,
    max_iter=50,
    tol=1e-3,
    batch_size=256,
):
    """
    N-dimensional gravity clustering pipeline:
    1) KDE ascent (mean-shift style)
    2) mode candidate extraction
    3) mode merging
    4) assignment to nearest merged mode
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2:
        raise ValueError("points must be a 2D array: (n_samples, n_features)")
    if points.shape[0] == 0:
        return {
            "labels": np.array([], dtype=int),
            "centers": np.empty((0, 0), dtype=float),
            "shifted_points": np.empty((0, 0), dtype=float),
            "converged": True,
            "iterations": 0,
        }

    shifted, converged, n_iter = density.shift_points_towards_density_modes(
        points=points,
        sigma=sigma,
        max_iter=max_iter,
        tol=tol,
        batch_size=batch_size,
    )

    mode_candidates = modes.find_modes_nd(shifted)
    merged_modes = merge.merge_close_modes(mode_candidates, merge_radius=merge_radius)

    if merged_modes.size == 0:
        merged_modes = mode_candidates[:1]

    labels = clustering.assign_points(points, merged_modes)

    return {
        "labels": labels,
        "centers": merged_modes,
        "shifted_points": shifted,
        "converged": converged,
        "iterations": n_iter,
    }
