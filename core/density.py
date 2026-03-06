import numpy as np


def compute_density(points, X, Y, sigma):
    field = np.zeros_like(X)
    for px, py in points:
        field += np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2 * sigma ** 2))

    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=1.0)
    except Exception:
        return field


def shift_points_towards_density_modes(
    points,
    sigma,
    max_iter=50,
    tol=1e-3,
    batch_size=256,
):
    """
    Mean-shift style ascent on Gaussian KDE in arbitrary dimensionality.

    Returns:
        shifted_points: np.ndarray (n_samples, n_features)
        converged: bool
        n_iter: int
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2:
        raise ValueError("points must be a 2D array: (n_samples, n_features)")
    if points.shape[0] == 0:
        return points.copy(), True, 0
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    shifted = points.copy()
    sigma2 = sigma * sigma
    n_samples = points.shape[0]

    for it in range(1, max_iter + 1):
        next_shifted = np.empty_like(shifted)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            chunk = shifted[start:end]

            diff = chunk[:, None, :] - points[None, :, :]
            sq_dist = np.sum(diff * diff, axis=2)
            weights = np.exp(-sq_dist / (2.0 * sigma2))

            numer = weights @ points
            denom = np.sum(weights, axis=1, keepdims=True) + 1e-12
            next_shifted[start:end] = numer / denom

        step = np.linalg.norm(next_shifted - shifted, axis=1)
        shifted = next_shifted

        if float(step.max(initial=0.0)) < tol:
            return shifted, True, it

    return shifted, False, max_iter
