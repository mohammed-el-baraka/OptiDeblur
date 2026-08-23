"""
Core Wiener-Hunt Deconvolution Implementation
=============================================
Provides high-performance frequency-domain solvers for regularized image
restoration using the 2D Discrete Fourier Transform (DFT).
"""

from enum import Enum
from typing import Tuple, Optional, Union
import numpy as np


class RegularizationType(str, Enum):
    """Supported regularization operators."""
    IDENTITY = "identity"    # Tikhonov 0th order (energy minimization)
    GRADIENT = "gradient"    # 1st order finite differences (edge preserving)
    LAPLACIAN = "laplacian"  # 2nd order discrete Laplace (curvature penalty)


def get_regularization_kernel(reg_type: Union[str, RegularizationType] = RegularizationType.GRADIENT) -> np.ndarray:
    """
    Returns the spatial convolution kernel for the chosen regularization operator.

    Parameters
    ----------
    reg_type : str or RegularizationType
        One of 'identity', 'gradient', or 'laplacian'.

    Returns
    -------
    np.ndarray
        2D filter kernel matrix.
    """
    if isinstance(reg_type, str):
        reg_type = RegularizationType(reg_type.lower())

    if reg_type == RegularizationType.IDENTITY:
        return np.array([[0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0]], dtype=np.float64)
    elif reg_type == RegularizationType.GRADIENT:
        return np.array([[ 0.0, -1.0,  0.0],
                         [-1.0,  4.0, -1.0],
                         [ 0.0, -1.0,  0.0]], dtype=np.float64)
    elif reg_type == RegularizationType.LAPLACIAN:
        return np.array([[-1.0, -1.0, -1.0],
                         [-1.0,  8.0, -1.0],
                         [-1.0, -1.0, -1.0]], dtype=np.float64)
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")


def compute_otf(
    psf: np.ndarray,
    target_shape: Tuple[int, int],
    center_psf: bool = True
) -> np.ndarray:
    """
    Computes the Optical Transfer Function (OTF) from a spatial Point Spread Function (PSF).

    Parameters
    ----------
    psf : np.ndarray
        Spatial PSF matrix.
    target_shape : Tuple[int, int]
        (rows, cols) of the target image grid.
    center_psf : bool, default=True
        Whether to circularly shift the PSF center to (0, 0) to avoid phase delay.

    Returns
    -------
    np.ndarray
        2D complex Optical Transfer Function in Fourier domain.
    """
    rows, cols = target_shape
    p_rows, p_cols = psf.shape

    padded = np.zeros((rows, cols), dtype=np.float64)
    padded[:p_rows, :p_cols] = psf

    if center_psf:
        shift_r = -(p_rows // 2)
        shift_c = -(p_cols // 2)
        padded = np.roll(padded, (shift_r, shift_c), axis=(0, 1))

    return np.fft.fft2(padded)


def wiener_deconvolve(
    blurred: np.ndarray,
    psf: np.ndarray,
    reg_param: float = 1e-4,
    reg_type: Union[str, RegularizationType] = RegularizationType.GRADIENT,
    center_psf: bool = True,
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Wiener-Hunt regularized deconvolution in the 2D frequency domain.

    Solves the regularized least squares problem:
        min_x ||y - Hx||_2^2 + lambda ||Dx||_2^2

    Using the circulant matrix / FFT closed-form solution:
        X_hat(nu) = [ H*(nu) / (|H(nu)|^2 + lambda |D(nu)|^2 + eps) ] * Y(nu)

    Parameters
    ----------
    blurred : np.ndarray
        Observed blurred and noisy image (2D float matrix).
    psf : np.ndarray
        Point Spread Function (PSF) / impulse response kernel.
    reg_param : float, default=1e-4
        Regularization parameter lambda (trade-off between noise and sharpness).
    reg_type : str or RegularizationType, default='gradient'
        Regularization operator ('identity', 'gradient', 'laplacian').
    center_psf : bool, default=True
        If True, shifts PSF center to (0,0) to prevent spatial coordinate lag.
    eps : float, default=1e-12
        Numerical stability floor for frequency division.

    Returns
    -------
    restored : np.ndarray
        Deconvolved / reconstructed image (real-valued 2D matrix).
    transfer_function : np.ndarray
        2D complex transfer function G(nu) applied in frequency domain.
    """
    rows, cols = blurred.shape

    # 1. Optical Transfer Function H(nu)
    H = compute_otf(psf, (rows, cols), center_psf=center_psf)

    # 2. Regularization Operator Transfer Function D(nu)
    D_kernel = get_regularization_kernel(reg_type)
    D = compute_otf(D_kernel, (rows, cols), center_psf=center_psf)

    # 3. Frequency domain observation Y(nu)
    Y = np.fft.fft2(blurred)

    # 4. Wiener-Hunt Transfer Function G(nu)
    denom = np.abs(H)**2 + reg_param * (np.abs(D)**2) + eps
    G = np.conj(H) / denom

    # 5. Filtered Spectrum and Spatial Reconstruction
    X_hat = G * Y
    restored = np.real(np.fft.ifft2(X_hat))

    return restored, G
