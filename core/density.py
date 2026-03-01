import numpy as np
from scipy.ndimage import gaussian_filter

def compute_density(points, X, Y, sigma):
    field = np.zeros_like(X)
    for px, py in points:
        field += np.exp(-((X - px)**2 + (Y - py)**2) / (2 * sigma**2))
    return gaussian_filter(field, sigma=1.0)
