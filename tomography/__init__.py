"""
All functions related to tomography and tomographic reconstruction
"""
import random
from typing import Callable

import numpy as np
from scipy.ndimage import rotate

from tomography.utils import limit_circle

"""
2 dimensional random scan 
"""


def radon_transform(image: np.ndarray, angles: np.ndarray) -> np.ndarray:
    sinogram = []

    for angle in angles:
        rotated = rotate(image, angle, reshape=False, order=1)
        projection = np.sum(rotated, axis=0)
        sinogram.append(projection)
    return np.array(sinogram).T  # shape: (N, n_angles)


def random_continuous_image(height: int,
                            width: int,

                            amount: int,

                            min_value: float = 0.0,
                            max_value: float = 1.0,

                            ellipse_param: Callable[[], float] = lambda: random.gauss(0.5, 0.1),
                            ellipse_position: Callable[[], float] = lambda: random.gauss(0, 0.25),
                            *,
                            alpha: float = 5.0,

                            ) -> np.ndarray:
    # Grille normalisée sur [-1, 1]
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    mg_x, mg_y = np.meshgrid(x, y)

    # Masque circulaire (optionnel)
    lc = limit_circle(height, width)

    image = np.zeros((height, width), dtype=float)

    def generate_gaussian():
        # Paramètres ellipse = variances en x et y
        a, b = ellipse_param(), ellipse_param()
        x0, y0 = ellipse_position(), ellipse_position()
        angle = random.uniform(0, 2 * np.pi)

        # Rotation du repère
        new_x = np.cos(angle) * (mg_x - x0) + np.sin(angle) * (mg_y - y0)
        new_y = -np.sin(angle) * (mg_x - x0) + np.cos(angle) * (mg_y - y0)

        new_x = new_x * alpha
        new_y = new_y * alpha

        # Fonction gaussienne 2D avec axes a, b
        gaussian = np.exp(- (new_x ** 2 / (2 * a ** 2) + new_y ** 2 / (2 * b ** 2)))
        return gaussian

    for _ in range(amount):
        gaussian = generate_gaussian()
        amplitude = np.random.uniform(min_value, max_value)
        image += amplitude * gaussian * lc  # on somme, au lieu d'écraser

    image = (image - np.min(image)) / np.max(image) * max_value + min_value

    return image


def random_discrete_image(height: int,
                          width: int,

                          amount: int,

                          min_value: float = 0.1,
                          max_value: float = 1.0,

                          ellipse_param: Callable[[], float] = lambda: random.gauss(0.5, 0.75),
                          ellipse_position: Callable[[], float] = lambda: random.gauss(0, 0.25)

                          ) -> np.ndarray:
    image = np.zeros((height, width))

    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)

    mg_x, mg_y = np.meshgrid(x, y)

    lc = limit_circle(height, width)

    def generate_ellipse():
        # generate ellipses that satisfy some properties
        a, b = ellipse_param(), ellipse_param()
        x0, y0 = ellipse_position(), ellipse_position()
        angle = random.uniform(- np.pi * 2, np.pi * 2)

        new_x = np.cos(angle) * (mg_x - x0) + np.sin(angle) * (mg_y - y0)
        new_y = - np.sin(angle) * (mg_x - x0) + np.cos(angle) * (mg_y - y0)

        return new_x ** 2 / a ** 2 + new_y ** 2 / b ** 2

    for _ in range(amount):
        ellipse = generate_ellipse()

        while np.any(ellipse[~lc] < 1):
            ellipse = generate_ellipse()

        value = np.random.uniform(min_value, max_value)

        image[(ellipse <= 1) & lc] = value

    return image


"""
3 dimensional random scan 
"""


def random_discrete_image_3d(height: int,
                             width: int,
                             depth: int,

                             amount: int,

                             min_value: float = 0.1,
                             max_value: float = 1.0,

                             ellipse_param: Callable[[], float] = lambda: random.gauss(0.5, 0.75),
                             ellipse_position: Callable[[], float] = lambda: random.gauss(0, 0.25)

                             ) -> np.ndarray:
    volume = np.zeros((height, width, depth))

    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    z = np.linspace(-1, 1, depth)

    mg_x, mg_y, mg_z = np.meshgrid(x, y, z)

    lc = (mg_x ** 2 + mg_y ** 2 + (mg_z / 5) ** 2) <= 1

    def generate_ellipse():
        # generate ellipses that satisfy some properties
        a, b, c = ellipse_param(), ellipse_param(), ellipse_param()
        x0, y0, z0 = ellipse_position(), ellipse_position(), ellipse_position()
        angle = random.uniform(- np.pi * 2, np.pi * 2)

        new_x = np.cos(angle) * (mg_x - x0) + np.sin(angle) * (mg_y - y0)
        new_y = - np.sin(angle) * (mg_x - x0) + np.cos(angle) * (mg_y - y0)

        return (new_x / a) ** 2 + (new_y / b) ** 2 + ((mg_z - z0) / c) ** 2

    for _ in range(amount):
        ellipse = generate_ellipse()

        while np.any(ellipse[~lc] < 1):
            ellipse = generate_ellipse()

        value = np.random.uniform(min_value, max_value)

        volume[(ellipse <= 1)] = value

    return volume


if __name__ == '__main__':
    """import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("TkAgg")

    V = random_discrete_image_3d(25, 25, 25, 20)

    for i in range(25):
        plt.matshow(V[i])

        plt.show()"""

    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("TkAgg")

    plt.matshow(random_discrete_image(100, 100, 10))

    plt.show()
