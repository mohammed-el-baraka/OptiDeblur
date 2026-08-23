"""
Edge Tapering and Boundary Artifact Reduction
=============================================
Provides windowing algorithms to mitigate boundary wrap-around ringing artifacts
caused by circular convolution assumptions in 2D FFT deconvolution.
"""

import numpy as np


def apply_edge_taper(
    image: np.ndarray,
    psf: np.ndarray,
    taper_width: tuple = None
) -> np.ndarray:
    """
    Applies smooth cosine edge tapering to reduce boundary discontinuities.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image matrix.
    psf : np.ndarray
        Point Spread Function matrix (used to adaptively size taper width).
    taper_width : tuple of (int, int), optional
        Width of the tapering zone (rows, cols). If None, calculated from PSF dimensions.

    Returns
    -------
    np.ndarray
        Tapered image smoothly blended with its mean value at boundaries.
    """
    m, n = image.shape
    pm, pn = psf.shape

    if taper_width is None:
        wm = min(int(round(pm * 2)), int(round(m / 10)))
        wn = min(int(round(pn * 2)), int(round(n / 10)))
    else:
        wm, wn = taper_width

    alpha = np.ones((m, n), dtype=np.float64)

    # Vertical borders (top / bottom)
    if wm > 0:
        idx_m = np.arange(1, wm + 1)
        weight_m = 0.5 * (1.0 - np.cos(np.pi * idx_m / wm))
        for i in range(wm):
            w = weight_m[i]
            alpha[i, :] = np.minimum(alpha[i, :], w)
            alpha[m - 1 - i, :] = np.minimum(alpha[m - 1 - i, :], w)

    # Horizontal borders (left / right)
    if wn > 0:
        idx_n = np.arange(1, wn + 1)
        weight_n = 0.5 * (1.0 - np.cos(np.pi * idx_n / wn))
        for j in range(wn):
            w = weight_n[j]
            alpha[:, j] = np.minimum(alpha[:, j], w)
            alpha[:, n - 1 - j] = np.minimum(alpha[:, n - 1 - j], w)

    mean_val = float(np.mean(image))
    tapered = mean_val + alpha * (image - mean_val)

    return tapered
