from copy import deepcopy
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=5)
def limit_circle(height: int, width: int) -> np.ndarray:
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)

    mg_x, mg_y = np.meshgrid(x, y)
    lc = (mg_x ** 2 + mg_y ** 2) <= 1

    return lc

def normalize(arr: np.ndarray, remove_border: bool = True) -> np.ndarray:
    if remove_border:
        val = arr[limit_circle(arr.shape[0], arr.shape[1])]
    else:
        val = arr

    return (arr - np.min(val)) / (np.max(val) - np.min(val))
