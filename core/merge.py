import numpy as np

def merge_close_modes(modes, merge_radius):
    merged = []
    for m in modes:
        if not merged:
            merged.append(m)
            continue
        dists = np.linalg.norm(np.array(merged) - m, axis=1)
        if np.all(dists > merge_radius):
            merged.append(m)
    return np.array(merged)
