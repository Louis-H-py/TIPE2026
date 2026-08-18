__import__("os").environ["KERAS_BACKEND"] = "jax"
__import__("os").environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import os
import random

import numpy as np


def train(
        model,
        name: str,
        folder: str,
        batch_size: int = 16,
        epochs: int = 10,
        models_dir: str = "models",
):


    os.makedirs(models_dir, exist_ok=True)

    files = os.listdir(folder)
    view = [*map(lambda x: int(x[:-4].split("-")[3]), files)]
    obj = [*map(lambda x: int(x[:-4].split("-")[1]), files)]

    dict_data = {x: [] for x in set(view)}

    for f, v, o in zip(files, view, obj):
        dict_data[v].append(f)

    for v in dict_data.keys():
        random.shuffle(dict_data[v])


    iteration = 0

    while sum([len(v) for v in dict_data.values()]) > 0:
        r = random.choice([*filter(lambda v: len(dict_data[v]) > 0, dict_data.keys())])

        x_train, y_train = [], []

        mem = 0.0
        n_angles = 0

        print("Iteration: ", iteration)
        print(r)

        while mem < 10000000000 and len(dict_data[r]) > 0:
            f_name = dict_data[r].pop()
            print(f"\r{f_name}\t {mem / 1_000_000}Mo", end="")

            data = np.load(os.path.join(folder, f_name))

            x_train.append(data["x"])
            y_train.append(data["y"])

            n_angles = data["x"].shape[1]

            mem += 2 * data["x"].nbytes + data["y"].nbytes

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        angles = np.linspace(0, 360, n_angles, endpoint=False, dtype=np.float32)

        model.train(
            sinograms=x_train,
            angles_deg=angles,
            images=y_train,
            batch_size=batch_size,
            epochs=epochs,
        )

        del x_train, y_train

        save_path = os.path.join(models_dir, f"{name}_{iteration:04d}.keras")
        model.save(save_path)

        iteration += 1