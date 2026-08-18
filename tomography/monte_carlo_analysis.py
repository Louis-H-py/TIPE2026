"""
Monte-Carlo statistical mesures of reconstruction 
"""
import os
from itertools import product
import time
from multiprocessing import cpu_count
from multiprocessing import Pool
from typing import Callable

import numpy as np

from tomography import random_discrete_image
from tomography.tomographic_reconstruction import radon_transform
from tomography.utils import normalize


# Importance du nombre d'itérations
# Importance nu nombre de prises de vue en angle
# 360° faire de 10 en 10 ?

# La grande mesure de ce TIPE est : Moins d'angle que possible pour le plus de qualité de reconstruction

# [ 10.,  35.,  60.,  85., 110., 135., 160., 185., 210., 235., 260.,
#        285., 310., 335., 360.]

def error_mesure(original: np.ndarray, predicted: np.ndarray) -> float:
    """
    This function mesure the error between the orginal image and the reconstructed image. 
    :param original
    :param predicted
    """
    return np.average(np.abs(original - predicted))


def monte_carlo_error_mesure(
        n_mesures: int,
        angles: np.ndarray,
        image_generator: Callable[[], np.ndarray],
        reconstruction: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> list[float]:
    """
    This function makes 'n_mesures' simulation of monte-carlo of sinogram reconstruction
    :param n_mesures: Amount of monte-carlo mesure to make
    :param angles: list of angles to capture
    :param image_generator: function that generate a scaner image
    :param reconstruction: a function that reconstruct the image based on sinogram and angle array
    :return: List of errors of each reconstruction
    """
    res = []

    for _ in range(n_mesures):
        image = image_generator()

        sinogram = radon_transform(image, angles)
        recon = reconstruction(sinogram, angles)
        err = error_mesure(normalize(image), normalize(recon))

        res.append(err)

    return res


def _worker_generate(args):
    k, angles = args
    image = random_discrete_image(100, 100, k)
    sinogram = radon_transform(image, angles)
    return image, sinogram.T


def generate_dataset(angles: np.ndarray,
                     amount: int,
                     file_path: str,
                     file_name: str,
                     k: int) -> None:
    params = [(k, angles) for _ in range(amount)]

    with Pool(cpu_count() - 1) as pool:
        results = pool.map(_worker_generate, params)

    data_images, data_sinograms = zip(*results)

    np.savez(
        file_path + file_name,
        x=np.array(data_sinograms, dtype=np.float32),
        y=np.array(data_images, dtype=np.float32)
    )


if __name__ == '__main__':
    """
    Create the test dataset
    """

    """parameters = [*product(
        [10, 35, 60, 110, 160, 210, 260, 310, 360],
        range(5, 25 + 1, 5),
        range(1)
    )]"""


    # Create the train dataset
    name = "training"
    parameters = [*product(
        [10, 35, 60, 110, 160, 210, 260, 310, 360],
        range(5, 25 + 1, 1),
        range(10)
    )]


    total_length = len(parameters)

    average_wait = []
    count = 0

    for angl, k, i in parameters:
        print(f"Objects: {k}\tID: {i}", end="\t")

        t = time.time()

        generate_dataset(
            np.linspace(0, 360, angl),
            amount=1000,
            file_path=f"C:/Users/louis/Documents/Python/ADVANCED/Tomographie/data/{name}/",
            file_name=f"data-{k}-{i}-{angl}",
            k=k
        )

        wait = time.time() - t
        average_wait.append(wait)
        count += 1

        print(f"Ended in {wait:.2f} seconds", end="\t")
        print(f"Remaining time: {np.average(average_wait) * (total_length - count):.2f} seconds")

    os.system("shutdown /s /t 60")

"""if __name__ == '__main__':

    from __init__ import *
    from tomographic_reconstruction import *

    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("TkAgg")

    img_gen = lambda: random_discrete_image(100, 100, 10)

    from tomography.neural_networks import TomographyLSTM
    import keras

    MODEL = TomographyLSTM()
    MODEL.model = keras.models.load_model(
        "C:/Users/louis/Documents/Python/ADVANCED/Tomographie/tomography/models/model_900.keras"
    )

    img_recon = lambda s, a: neural_network_reconstruction_many(s, a, MODEL)

    IMG1 = random_discrete_image(100, 100, 25)
    IMG2 = random_discrete_image(100, 100, 25)

    SINO1 = radon_transform(IMG1, np.linspace(0, 360, 360))
    SINO2 = radon_transform(IMG2, np.linspace(0, 360, 360))

    RECON1 = neural_network_reconstruction(np.random.uniform(0, 1, size=SINO1.shape), np.linspace(0, 360, 360), MODEL)
    RECON2 = neural_network_reconstruction(SINO2, np.linspace(0, 360, 360), MODEL)

    plt.matshow(RECON1)
    plt.show()
    plt.matshow(RECON2)
    plt.show()

    plt.matshow(RECON1 - RECON2, vmin=-1, vmax=1)
    plt.show()


    IMG = random_discrete_image(100, 100, 10)

    plt.matshow(IMG)
    plt.show()

    SINO = radon_transform(IMG, np.linspace(0, 360, 360))

    RECON = neural_network_reconstruction(SINO, np.linspace(0, 360, 360), MODEL)
    plt.matshow(RECON)
    plt.show()


    plt.matshow(np.abs(normalize(IMG) - normalize(RECON)), vmin=0, vmax=1)
    plt.show()

    '''l = monte_carlo_error_mesure(
        100,
        np.linspace(0, 360, 10_000),
        img_gen,
        img_recon)

    plt.hist(l, bins=51)

    plt.xlim((0, 1))
    plt.show()'''"""

"""if __name__ == "__main__":
    if __name__ == '__main__':
        from __init__ import *
        from tomographic_reconstruction import *

        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("TkAgg")

        IMG = random_discrete_image(100, 100, 10)


        plt.imshow(IMG)
        plt.show()
"""
