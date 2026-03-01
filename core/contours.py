import numpy as np

def contour_mask(field, level):
    threshold = level * field.max()
    return field >= threshold
