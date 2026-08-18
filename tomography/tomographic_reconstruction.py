"""
This Python libraries contains every function to make a tomography

Fourier-domain reconstruction algorithm
Back projection algorithm
Iterative reconstruction algorithm
"""
import numpy as np
from scipy.ndimage import rotate

from tomography import radon_transform


def back_projection(sinogram: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    This function realize the back projection transformation on a sinogram.
    :param sinogram: A sinogram
    :param angles: angles of each view in the sinogram
    :return: Reconstructed image
    """
    n = sinogram.shape[0]
    recon = np.zeros((n, n))

    for i, angle in enumerate(angles):
        proj = sinogram[:, i]

        back_proj = np.tile(proj, (n, 1))
        back_proj_rot = rotate(back_proj, -angle, reshape=False, order=1)
        recon += back_proj_rot

    recon /= len(angles)
    return recon


def fourier_domain_reconstruction(sinogram: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    This function realize the fourier domain reconstruction algorithm on a sinogram.
    :param sinogram: A sinogram
    :param angles: angles of each view in the sinogram
    :return: Reconstructed image
    """

    n = sinogram.shape[0]
    recon = np.zeros((n, n))
    freq = np.fft.fftfreq(n).reshape(-1, 1)

    for i, angle in enumerate(angles):
        proj = sinogram[:, i]
        proj_fft = np.fft.fft(proj)
        ramp = np.abs(freq).ravel()
        proj_fft_filtered = proj_fft * ramp
        proj = np.real(np.fft.ifft(proj_fft_filtered))

        back_proj = np.tile(proj, (n, 1))
        back_proj_rot = rotate(back_proj, -angle, reshape=False, order=1)
        recon += back_proj_rot

    recon /= len(angles)
    return recon


def iterative_reconstruction(sinogram: np.ndarray,
                             angles: np.ndarray,
                             n_iter: int = 10,
                             relaxation: float = 0.1) -> np.ndarray:
    """
    Iterative reconstruction with stability and normalization.
    """
    n = sinogram.shape[0]
    recon = np.zeros((n, n), dtype=np.float32)

    for it in range(n_iter):
        for i, angle in enumerate(angles):
            # Forward projection
            proj_current = radon_transform(recon, [angle])[:, 0]

            # Normalisation to avoid exploding values
            if np.max(proj_current) > 0:
                proj_current /= np.max(proj_current)
            target = sinogram[:, i]
            if np.max(target) > 0:
                target /= np.max(target)

            # Compute error
            error = target - proj_current

            # Back-propagate correction
            correction = np.tile(error, (n, 1))
            corr_rot = rotate(correction, -angle, reshape=False, order=1)

            # Update with relaxation factor
            recon += relaxation * corr_rot / len(angles)

        # Clip and sanitize to avoid NaNs
        recon = np.clip(recon, 0, None)
        recon = np.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)

    return recon


def neural_network_reconstruction(sinogram: np.ndarray,
                                  angles: np.ndarray,
                                  model,
                                  *args,
                                  **kwargs) -> np.ndarray:
    new_angle = np.reshape(angles, (len(angles), 1))

    return model(
        sinogram=np.array([sinogram.T]),
        angles=np.array([new_angle]),
        large_input=False,
        *args,
        **kwargs
    )[0]


def neural_network_reconstruction_many(sinograms: np.ndarray,
                                       angles: np.ndarray,
                                       model,
                                       *args,
                                       **kwargs) -> np.ndarray:
    new_angle = np.reshape(angles, (len(angles), 1))

    return model(
        sinogram=np.array([s for s in sinograms]),
        angles=np.array([new_angle for _ in sinograms]),
        large_input=True,
        *args,
        **kwargs
    )
