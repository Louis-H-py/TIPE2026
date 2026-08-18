"""
Compare all methods
"""
__import__("os").environ["CUDA_VISIBLE_DEVICES"] = ""

import os
import sqlite3
import time
from itertools import product

import numpy as np

from tomography.monte_carlo_analysis import error_mesure
from tomography.neural_networks.unet import CTUNet
from tomography.neural_networks.direct_reconstruction import CTReconstructionNetwork
from tomography.tomographic_reconstruction import back_projection
from tomography.tomographic_reconstruction import fourier_domain_reconstruction
from tomography.tomographic_reconstruction import iterative_reconstruction
from tomography.utils import normalize

METHODS_IDS: dict[str, int] = {n: i for i, n in enumerate(["bp", "fd", "ir", "nn", "unet_bp", "unet_fft"])}
METHODS_LABEL: dict[int, str] = {i: n for i, n in enumerate(["bp", "fd", "ir", "nn", "unet_bp", "unet_fft"])}

#
# TEST_PATH: str = "C:/Users/louis/Documents/Python/ADVANCED/Tomographie/data/testing"
TEST_PATH: str = "/mnt/c/Users/louis/Documents/Python/ADVANCED/Tomographie/data/testing"
MODEL = None

MODEL_BP = CTUNet.load(
    "/mnt/c/Users/louis/Documents/Python/ADVANCED/Tomographie/tomography/neural_networks/models/UNet_0010.keras")
MODEL_BP.bp_type = "bp"

MODEL_FFT = CTUNet.load(
    "/mnt/c/Users/louis/Documents/Python/ADVANCED/Tomographie/tomography/neural_networks/models/UNet_fft_0012.keras")
MODEL_FFT.bp_type = "fft"

MODEL_DIRECT = CTReconstructionNetwork.load(
    "/mnt/c/Users/louis/Documents/Python/ADVANCED/Tomographie/tomography/neural_networks/models/Direct_0011.keras")
MODEL_DIRECT.projection_dim = 128
MODEL_DIRECT.latent_dim = 256


def mesure_process(param: tuple[str, int, int, int]):
    method_label: str = param[0]
    n_shape = param[1]
    n_angle = param[2]
    _id = param[3]

    method_id = METHODS_IDS[method_label]

    print(f"Started: {method_label}\tshapes:{n_shape}\tangles:{n_angle}\tid:{_id}")
    t = time.time()

    data = np.load(os.path.join(TEST_PATH, f"data-{n_shape}-{_id}-{n_angle}.npz"))
    sinograms = data["x"]
    images = data["y"]
    angles = np.linspace(0, 360, n_angle)

    results = []  # (method_label, n_shape, n_angle, id, sub_id, error)

    if method_label == "unet_bp":
        preds = [p for p in MODEL_BP.predict(sinograms, angles_deg=angles)]

        for i, (sin, img, pred) in enumerate(zip(sinograms, images, preds)):
            err = np.average(np.abs(normalize(img) - normalize(pred)))
            results.append((method_label, n_shape, n_angle, _id, i, float(err)))

    elif method_label == "unet_fft":
        preds = [p for p in MODEL_FFT.predict(sinograms, angles_deg=angles)]

        for i, (sin, img, pred) in enumerate(zip(sinograms, images, preds)):
            err = np.average(np.abs(normalize(img) - normalize(pred)))
            results.append((method_label, n_shape, n_angle, _id, i, float(err)))


    elif method_label == "nn":
        preds = [p for p in[ MODEL_DIRECT.predict(s, angles_deg=angles) for s in sinograms]]

        for i, (sin, img, pred) in enumerate(zip(sinograms, images, preds)):
            err = np.average(np.abs(normalize(img) - normalize(pred)))
            results.append((method_label, n_shape, n_angle, _id, i, float(err)))

    else:
        if method_label == "bp":
            func = back_projection
        elif method_label == "fd":
            func = fourier_domain_reconstruction
        elif method_label == "ir":
            func = lambda s, a: iterative_reconstruction(s, a, 10)
        else:
            raise Exception("Unknown method")

        for i, (sin, img) in enumerate(zip(sinograms, images)):
            pred = func(sin.T, angles)
            err = error_mesure(normalize(img), normalize(pred))
            results.append((method_label, n_shape, n_angle, _id, i, float(err)))

    duration = time.time() - t
    print(f"Ended in {duration:.2f}s: {method_label}\tshapes:{n_shape}\tangles:{n_angle}\tid:{_id}")

    return method_id, n_shape, n_angle, _id, results


if __name__ == '__main__':
    """
    Test basics methods
    """

    # Create database
    try:
        conn = sqlite3.connect("data/methods_comparison.db")
        cursor = conn.cursor()

        # Create all tables for results
        for name in ["nn", "unet_bp", "unet_fft", "bp", "fd", "ir", ]:
            cursor.execute(f"""CREATE TABLE "results_{name}" (
                "n_shape" INTEGER NOT NULL, 
                "n_angle" INTEGER NOT NULL, 
                "id" INTEGER NOT NULL,
                "sub_id" INTEGER NOT NULL,
                "error" REAL NOT NULL
            );
            """)

        # Create main table to get the remaining mesure to do
        cursor.execute(f"""CREATE TABLE "total_results" (
                "method_id" INTEGER NOT NULL,
                "n_shape" INTEGER NOT NULL, 
                "n_angle" INTEGER NOT NULL, 
                "id" INTEGER NOT NULL
            );
            """)

        conn.close()

    except Exception as e:
        print(e)

    params = list(product(
        ["nn"],
        range(5, 25 + 1, 5),
        [10, 35, 60, 110, 160, 210, 260, 310, 360],
        [0]
    ))

    # [10, 35, 60, 110, 160, 210, 260, 310, 360], #  260, 310, 360

    conn = sqlite3.connect("data/methods_comparison.db")
    cursor = conn.cursor()

    '''
    #
    # Multiprocess version :
    #
    
    with Pool(1) as pool:
        for method_id, n_shape, n_angle, _id, res in pool.imap_unordered(mesure_process, params):

            # Skip if already done
            cursor.execute(
                "SELECT 1 FROM total_results WHERE method_id=? AND n_shape=? AND n_angle=? AND id=?",
                (method_id, n_shape, n_angle, _id)
            )
            if cursor.fetchone():
                print(f"Already done: {method_id} {n_shape} {n_angle} {_id}")
                continue

            # Cleanup partial runs
            method_label = [k for k, v in METHODS_IDS.items() if v == method_id][0]
            cursor.execute(
                f"DELETE FROM results_{method_label} WHERE n_shape=? AND n_angle=? AND id=?",
                (n_shape, n_angle, _id)
            )

            # Bulk insert (rapide)
            cursor.executemany(
                f"""
                    INSERT INTO results_{method_label}
                    (n_shape, n_angle, id, sub_id, error)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                [(n_shape, n_angle, _id, sub_id, err)
                 for _, n_shape, n_angle, _id, sub_id, err in res]
            )

            cursor.execute(
                "INSERT INTO total_results (method_id, n_shape, n_angle, id) VALUES (?, ?, ?, ?)",
                (method_id, n_shape, n_angle, _id)
            )

            conn.commit()

    conn.close()'''

    #
    # Single process version :
    #

    #
    # Version sans multiprocessing
    #

    for param in params:
        method_id, n_shape, n_angle, _id, res = mesure_process(param)

        # Skip if already done
        cursor.execute(
            "SELECT 1 FROM total_results WHERE method_id=? AND n_shape=? AND n_angle=? AND id=?",
            (method_id, n_shape, n_angle, _id)
        )
        if cursor.fetchone():
            print(f"Already done: {method_id} {n_shape} {n_angle} {_id}")
            continue

        # Cleanup partial runs
        method_label = [k for k, v in METHODS_IDS.items() if v == method_id][0]
        cursor.execute(
            f"DELETE FROM results_{method_label} WHERE n_shape=? AND n_angle=? AND id=?",
            (n_shape, n_angle, _id)
        )

        # Bulk insert
        cursor.executemany(
            f"""
            INSERT INTO results_{method_label}
            (n_shape, n_angle, id, sub_id, error)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (n_shape, n_angle, _id, sub_id, err)
                for _, n_shape, n_angle, _id, sub_id, err in res
            ]
        )

        cursor.execute(
            "INSERT INTO total_results (method_id, n_shape, n_angle, id) VALUES (?, ?, ?, ?)",
            (method_id, n_shape, n_angle, _id)
        )

        conn.commit()

    conn.close()
