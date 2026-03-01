import numpy as np

def assign_points(points, centers):
    dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    return np.argmin(dists, axis=1)
