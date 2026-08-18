"""
Influence du nombre d'itérations pour la méthode de reconstruction itérative
"""
from multiprocessing import Pool
import sqlite3

import numpy as np

from tomography import random_discrete_image
from tomography.monte_carlo_analysis import monte_carlo_error_mesure
from tomography.tomographic_reconstruction import iterative_reconstruction

# parameters
IMG_SIZE: int = 100
ELLIPSIS_COUNT: int = 15
N_PROCESS: int = 22
N_ITER_MAX: int = 20

MESURE_COUNT: int = 1000
BATCH_SIZE: int = 25

img_gen = lambda: random_discrete_image(IMG_SIZE, IMG_SIZE, ELLIPSIS_COUNT)

# open/init the database
conn = sqlite3.connect(f"data/iterative_parameters.db", check_same_thread=False)
cursor = conn.cursor()

try:
    cursor.execute("""CREATE TABLE "results" ("n_iter" INTEGER NOT NULL, "error" REAL NOT NULL);""")
except Exception as e:
    print(e)


def mesure_process(param: tuple[int, int, int, int]) -> None:
    """
    Process to store the error of the iterative reconstruction in the database
    :param param: All parameters : id, batch_id, batch_size, n_iter
    :return: None
    """
    _id, batch_num, batch_size, n_iter = param

    adv = round(_id / (MESURE_COUNT * N_ITER_MAX) * BATCH_SIZE * 100, 2)
    print(f"Starting {_id}: {batch_num} {batch_size} {n_iter} ({adv}%)")

    img_recon = lambda s, a: iterative_reconstruction(s, a, n_iter=n_iter)

    res: list[float] = monte_carlo_error_mesure(
        batch_size,
        np.linspace(0, 360, 360),
        img_gen,
        img_recon
    )

    for val in res:
        conn.execute("INSERT INTO results (n_iter, error) VALUES (?, ?)", (n_iter, val))

    conn.commit()


if __name__ == '__main__':
    task = [(bnum, BATCH_SIZE, k) for k in range(1, N_ITER_MAX + 1) for bnum in range(MESURE_COUNT // BATCH_SIZE)]
    task = [*enumerate(task)]
    task = [*map(lambda c: (c[0],) + c[1], task)]

    with Pool(processes=N_PROCESS) as pool:
        map_result = pool.imap_unordered(mesure_process, [*task])
        for r in map_result:
            continue
