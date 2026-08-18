__import__("os").environ["KERAS_BACKEND"] = "jax"


import numpy as np
import keras
from keras import layers, Model, ops


class Deg2Rad(layers.Layer):
    def call(self, angles_deg):
        return angles_deg * (np.pi / 180.0)

    def compute_output_shape(self, input_shape):
        return input_shape


class AngleEncoding(layers.Layer):
    def call(self, angles_rad):
        return ops.stack([ops.sin(angles_rad), ops.cos(angles_rad)], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], 2


class CTReconstructionNetwork:
    def __init__(
            self,
            projection_dim: int = 128,  # taille embedding par projection
            latent_dim: int = 256,  # taille du vecteur global après pooling
            learning_rate: float = 1e-3,
    ):
        self.projection_dim = projection_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.model: Model = self._build()
        self._compile()

    def _build(self) -> Model:
        N = 100  # nb de détecteurs

        sinogram_in = keras.Input(shape=(None, N), name="sinogram")  # (B, n_angles, 100)
        angles_in = keras.Input(shape=(None,), name="angles_deg")  # (B, n_angles)

        # Encodage des angles
        angles_rad = Deg2Rad(name="deg2rad")(angles_in)  # (B, n_angles)

        angle_enc = AngleEncoding(name="angle_enc")(angles_rad)  # (B, n_angles, 2)

        # Concaténation projection + angle
        x = layers.Concatenate(axis=-1, name="proj_with_angle")(
            [sinogram_in, angle_enc]
        )  # (B, n_angles, 102)

        # MLP partagé sur chaque projection (TimeDistributed)
        x = layers.TimeDistributed(layers.Dense(256, activation="relu"),
                                   name="td_dense1")(x)  # (B, n_angles, 256)
        x = layers.TimeDistributed(layers.Dense(self.projection_dim, activation="relu"),
                                   name="td_dense2")(x)  # (B, n_angles, projection_dim)

        # Agrégation
        z = layers.GlobalAveragePooling1D(name="angle_pooling")(x)  # (B, projection_dim)

        # Couches denses de transition : espace latent
        z = layers.Dense(self.latent_dim, activation="relu", name="latent_dense1")(z)
        z = layers.Dense(self.latent_dim, activation="relu", name="latent_dense2")(z)

        # Décodeur CNN transposé 5 -> 10 -> 20 -> 50 -> 100
        z = layers.Dense(5 * 5 * 128, activation="relu", name="pre_reshape")(z)
        z = layers.Reshape((5, 5, 128), name="reshape")(z)

        z = layers.Conv2DTranspose(128, kernel_size=3, strides=2,
                                   padding="same", activation="relu",
                                   name="deconv1")(z)  # 10×10
        z = layers.Conv2DTranspose(64, kernel_size=3, strides=2,
                                   padding="same", activation="relu",
                                   name="deconv2")(z)  # 20×20
        z = layers.Conv2DTranspose(32, kernel_size=3, strides=2,
                                   padding="same", activation="relu",
                                   name="deconv3")(z)  # 40×40 (pad crop needed)
        z = layers.Conv2DTranspose(16, kernel_size=3, strides=2,
                                   padding="same", activation="relu",
                                   name="deconv4")(z)  # 80×80

        # 80 -> 100 : un Conv2DTranspose avec stride 1 + crop/resize
        z = layers.UpSampling2D(size=(2, 2), name="upsample")(z)  # 160×160
        z = layers.Conv2D(16, kernel_size=3, padding="same",
                          activation="relu", name="refine1")(z)
        z = layers.Cropping2D(cropping=30, name="crop_to_100")(z)  # 100×100

        output = layers.Conv2D(1, kernel_size=1, activation="sigmoid",
                               name="output")(z)  # (B, 100, 100, 1)

        return Model(inputs=[sinogram_in, angles_in],
                     outputs=output,
                     name="CTReconNet")

    def _compile(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

    def summary(self):
        self.model.summary()

    def train(
            self,
            sinograms: np.ndarray,  # (N_samples, n_angles, 100)
            angles_deg: np.ndarray,  # (n_angles,) ou (N_samples, n_angles)
            images: np.ndarray,  # (N_samples, 100, 100)
            epochs: int = 50,
            batch_size: int = 16,
            validation_split: float = 0.1,
            vram_limit_gb: float = 10.0,
    ) -> None:
        N_samples = sinograms.shape[0]

        # Broadcast angles si nécessaire
        if angles_deg.ndim == 1:
            angles_input = np.tile(angles_deg[np.newaxis, :], (N_samples, 1)).astype(np.float32)
        else:
            angles_input = angles_deg.astype(np.float32)

        # Normalisation sinogrammes
        x_sino = sinograms.astype(np.float32)
        smax = x_sino.max(axis=(1, 2), keepdims=True)
        smax = np.where(smax == 0, 1.0, smax)
        x_sino /= smax

        # Normalisation images [0, 1]
        y = images.astype(np.float32)
        vmax = y.max(axis=(1, 2), keepdims=True)
        vmax = np.where(vmax == 0, 1.0, vmax)
        y = (y / vmax)[..., np.newaxis]  # (N, 100, 100, 1)

        # Taille en octets d'un sample (sinogram + angles + image)
        bytes_per_sample = (
                x_sino[0].nbytes + angles_input[0].nbytes + y[0].nbytes
        )
        chunk_size = max(batch_size, int(vram_limit_gb * 1e9 / bytes_per_sample))
        chunk_size = min(chunk_size, N_samples)

        # Shuffle global
        idx = np.random.permutation(N_samples)
        x_sino = x_sino[idx]
        angles_input = angles_input[idx]
        y = y[idx]

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, verbose=1
            ),
        ]

        # Itération sur les chunks : model.fit() ne voit jamais plus de chunk_size samples
        for start in range(0, N_samples, chunk_size):
            sl = slice(start, start + chunk_size)
            print(f"  chunk [{start}:{start + chunk_size}] / {N_samples}")
            self.model.fit(
                x={"sinogram": x_sino[sl], "angles_deg": angles_input[sl]},
                y=y[sl],
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
            )

    def predict(
            self,
            sinogram: np.ndarray,  # (n_angles, 100)  ou  (B, n_angles, 100)
            angles_deg: np.ndarray,  # (n_angles,)       ou  (B, n_angles)
    ) -> np.ndarray:
        """
        Reconstruit une (ou plusieurs) image(s).

        Retourne : np.ndarray de shape (100, 100) si entrée 2D,
                   (B, 100, 100) si entrée 3D.
        """
        single = sinogram.ndim == 2
        if single:
            sinogram = sinogram[np.newaxis]  # (1, n_angles, 100)
            angles_deg = angles_deg[np.newaxis]  # (1, n_angles)

        # Normalisation
        x_sino = sinogram.astype(np.float32)
        smax = x_sino.max(axis=(1, 2), keepdims=True)
        smax = np.where(smax == 0, 1.0, smax)
        x_sino = x_sino / smax

        pred = self.model.predict(
            {"sinogram": x_sino, "angles_deg": angles_deg.astype(np.float32)},
            verbose=0,
        )  # (B, 100, 100, 1)

        pred = pred[..., 0]  # (B, 100, 100)
        return pred[0] if single else pred

    def save(self, path: str):
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "CTReconstructionNetwork":
        obj = cls.__new__(cls)
        obj.model = keras.models.load_model(
            path,
            custom_objects={"AngleEncoding": AngleEncoding, "Deg2Rad": Deg2Rad},
        )
        return obj


if __name__ == "__main__":
    from __init__ import train

    model = CTReconstructionNetwork(projection_dim=128, latent_dim=256)

    train(
        model=model,
        name="Direct",
        folder=r"/mnt/c/Users/louis/Documents/Python/ADVANCED/Tomographie/data/training",
        batch_size=16,
        epochs=10,
        models_dir="models",
    )