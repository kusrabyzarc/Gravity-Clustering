import numpy as np

def make_grid(bounds=(-10, 10), grid_size=400):
    x = np.linspace(bounds[0], bounds[1], grid_size)
    y = np.linspace(bounds[0], bounds[1], grid_size)
    return np.meshgrid(x, y)
