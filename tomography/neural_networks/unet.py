from os import name

__import__("os").environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from keras import layers, Model, ops
import random
import os
from numba import njit, prange
import math


@njit(parallel=True, cache=True)
def _fast_backprojection(sinogram: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    N = sinogram.shape[0]
    n_angles = sinogram.shape[1]
    det = sinogram.shape[2]
    cx = 50.0

    recon = np.zeros((N, 100, 100), dtype=np.float32)

    for n in prange(N):  # parallèle sur les samples
        for i in range(n_angles):
            rad = angles_deg[i] * (math.pi / 180.0)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            for row in range(100):
                for col in range(100):
                    x = col - cx
                    y = row - cx
                    t = x * cos_a + y * sin_a + cx
                    t_idx = int(t)
                    if t_idx < 0:
                        t_idx = 0
                    elif t_idx >= det:
                        t_idx = det - 1
                    recon[n, row, col] += sinogram[n, i, t_idx]

        for row in range(100):
            for col in range(100):
                recon[n, row, col] /= n_angles

    return recon


def _ramp_filter_fft(sinogram: np.ndarray) -> np.ndarray:
    det = sinogram.shape[2]
    freq = np.abs(np.fft.fftfreq(det)).astype(np.float32)  # (100,)

    # FFT sur l'axe détecteur (axis=2), multiplication, IFFT — tout vectorisé
    sino_fft = np.fft.fft(sinogram, axis=2)  # (N, n_angles, 100)
    sino_filtered = np.real(np.fft.ifft(sino_fft * freq, axis=2))
    return sino_filtered.astype(np.float32)


@njit(parallel=True, cache=True)
def _backproject_fft(sinogram: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    N = sinogram.shape[0]
    n_angles = sinogram.shape[1]
    det = sinogram.shape[2]
    cx = 50.0

    recon = np.zeros((N, 100, 100), dtype=np.float32)

    for n in prange(N):
        for i in range(n_angles):
            rad = angles_deg[i] * (math.pi / 180.0)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            for row in range(100):
                for col in range(100):
                    t = (col - cx) * cos_a + (row - cx) * sin_a + cx
                    t_idx = int(t)
                    if t_idx < 0:
                        t_idx = 0
                    elif t_idx >= det:
                        t_idx = det - 1
                    recon[n, row, col] += sinogram[n, i, t_idx]

        for row in range(100):
            for col in range(100):
                recon[n, row, col] /= n_angles

    return recon


def _fast_backprojection_fft(sinogram: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    return _backproject_fft(_ramp_filter_fft(sinogram), angles_deg)


_fast_backprojection(np.zeros((3, 100, 100)), np.linspace(0, 360, 100))
_fast_backprojection_fft(np.zeros((3, 100, 100)), np.linspace(0, 360, 100))


class CTUNet:
    def __init__(
            self,
            filters: int = 32,  # filtres de base (doublés à chaque niveau)
            depth: int = 4,  # nombre de niveaux encodeur/décodeur
            learning_rate: float = 1e-3,
            bp_type: str = "fft"
    ):
        self.filters = filters
        self.depth = depth
        self.learning_rate = learning_rate
        self.model: Model = self._build()
        self._compile()
        self.bp_type = bp_type

    @staticmethod
    def _conv_block(x, n_filters: int, name: str):
        x = layers.Conv2D(n_filters, 3, padding="same", name=f"{name}_c1")(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation("relu", name=f"{name}_r1")(x)
        x = layers.Conv2D(n_filters, 3, padding="same", name=f"{name}_c2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.Activation("relu", name=f"{name}_r2")(x)
        return x

    @staticmethod
    def _match_and_concat(x, skip, name: str):
        # shapes symboliques
        xs = ops.shape(x)
        ss = ops.shape(skip)
        # crop sur h et w si x est plus grand que skip
        dh = xs[1] - ss[1]
        dw = xs[2] - ss[2]
        if dh > 0 or dw > 0:
            x = layers.Cropping2D(
                cropping=((0, dh), (0, dw)), name=f"{name}_crop"
            )(x)
        return layers.Concatenate(name=name)([x, skip])

    def _build(self) -> Model:
        inp = keras.Input(shape=(100, 100, 1), name="bp_image")

        # Encodeur
        skips = []
        x = inp
        for d in range(self.depth):
            f = self.filters * (2 ** d)
            x = self._conv_block(x, f, name=f"enc{d}")
            skips.append(x)
            x = layers.MaxPooling2D(2, padding="same", name=f"pool{d}")(x)

        # Bottleneck
        x = self._conv_block(x, self.filters * (2 ** self.depth), name="bottleneck")

        # Décodeur
        for d in reversed(range(self.depth)):
            f = self.filters * (2 ** d)
            x = layers.Conv2DTranspose(f, 2, strides=2, padding="same",
                                       name=f"up{d}")(x)
            x = self._match_and_concat(x, skips[d], name=f"skip{d}")
            x = self._conv_block(x, f, name=f"dec{d}")

        # Sortie résiduelle : UNet prédit la correction
        correction = layers.Conv2D(1, 1, activation="tanh", name="correction")(x)
        output = layers.Add(name="output")([inp, correction])
        output = layers.Activation("sigmoid", name="clip")(output)

        return Model(inputs=inp, outputs=output, name="CTUNet")

    def _compile(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

    def summary(self):
        self.model.summary()

    def _prepare_inputs(
            self,
            sinograms: np.ndarray,  # (N, n_angles, 100)
            angles_deg: np.ndarray,  # (n_angles,)
    ) -> np.ndarray:
        if self.bp_type == "bp":
            bp = _fast_backprojection(sinograms, angles_deg)  # (N, 100, 100)

        elif self.bp_type == "fft":
            bp = _fast_backprojection_fft(sinograms, angles_deg)

        else:
            raise Exception()

        # Normalisation par sample
        vmax = bp.max(axis=(1, 2), keepdims=True)
        vmax = np.where(vmax == 0, 1.0, vmax)
        bp /= vmax
        return bp[..., np.newaxis]  # (N, 100, 100, 1)

    def train(
            self,
            sinograms: np.ndarray,  # (N, n_angles, 100)
            angles_deg: np.ndarray,  # (n_angles,)
            images: np.ndarray,  # (N, 100, 100)
            epochs: int = 50,
            batch_size: int = 16,
            validation_split: float = 0.1,
            vram_limit_gb: float = 10.0,
    ) -> None:
        print("Backprojection...", end=" ", flush=True)
        x = self._prepare_inputs(sinograms, angles_deg, )
        print("OK")

        # Normalisation images cibles
        y = images.astype(np.float32)
        vmax = y.max(axis=(1, 2), keepdims=True)
        vmax = np.where(vmax == 0, 1.0, vmax)
        y = (y / vmax)[..., np.newaxis]  # (N, 100, 100, 1)

        # Chunk pour ne pas dépasser vram_limit_gb
        bytes_per_sample = x[0].nbytes + y[0].nbytes
        chunk_size = max(batch_size, int(vram_limit_gb * 1e9 / bytes_per_sample))
        chunk_size = min(chunk_size, len(x))

        idx = np.random.permutation(len(x))
        x, y = x[idx], y[idx]

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, verbose=1
            ),
        ]

        for start in range(0, len(x), chunk_size):
            sl = slice(start, start + chunk_size)
            print(f"  chunk [{start}:{min(start + chunk_size, len(x))}] / {len(x)}")
            self.model.fit(
                x=x[sl], y=y[sl],
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
            )

    def predict(
            self,
            sinogram: np.ndarray,  # (n_angles, 100)  ou  (N, n_angles, 100)
            angles_deg: np.ndarray,  # (n_angles,)
    ) -> np.ndarray:
        single = sinogram.ndim == 2
        if single:
            sinogram = sinogram[np.newaxis]

        x = self._prepare_inputs(sinogram, angles_deg)  # (N, 100, 100, 1)
        pred = self.model.predict(x, verbose=0)  # (N, 100, 100, 1)
        pred = pred[..., 0]  # (N, 100, 100)
        return pred[0] if single else pred

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "CTUNet":
        obj = cls.__new__(cls)
        obj.model = keras.models.load_model(path)
        return obj


if __name__ == "__main__":
    BP_TYPE = "fft"

    from __init__ import train

    model = CTUNet(bp_type=BP_TYPE)

    train(
        model=model,
        name="UNet" + ("" if BP_TYPE == "bp" else "_fft"),
        folder=r"/mnt/c/Users/louis/Documents/Python/ADVANCED/Tomographie/data/training",
        batch_size=16,
        epochs=10,
        models_dir="models",
    )
