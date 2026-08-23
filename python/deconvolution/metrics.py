"""
Image Quality and Error Metrics
===============================
Provides implementations for multi-norm distance metrics (L1, L2, Linf),
Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM).
"""

from typing import Dict
import numpy as np
from scipy.ndimage import uniform_filter


def compute_relative_l2(estimate: np.ndarray, ground_truth: np.ndarray) -> float:
    """Relative L2 error norm (normalized RMSE)."""
    diff_norm = np.linalg.norm(estimate.ravel() - ground_truth.ravel(), ord=2)
    truth_norm = np.linalg.norm(ground_truth.ravel(), ord=2)
    return float(diff_norm / (truth_norm + 1e-12))


def compute_relative_l1(estimate: np.ndarray, ground_truth: np.ndarray) -> float:
    """Relative L1 error norm (Mean Absolute Error normalized)."""
    diff_norm = np.linalg.norm(estimate.ravel() - ground_truth.ravel(), ord=1)
    truth_norm = np.linalg.norm(ground_truth.ravel(), ord=1)
    return float(diff_norm / (truth_norm + 1e-12))


def compute_relative_linf(estimate: np.ndarray, ground_truth: np.ndarray) -> float:
    """Relative L-infinity error norm (Chebyshev / maximum peak deviation)."""
    diff_max = np.max(np.abs(estimate.ravel() - ground_truth.ravel()))
    truth_max = np.max(np.abs(ground_truth.ravel()))
    return float(diff_max / (truth_max + 1e-12))


def compute_mse(estimate: np.ndarray, ground_truth: np.ndarray) -> float:
    """Mean Squared Error (MSE)."""
    return float(np.mean((estimate - ground_truth) ** 2))


def compute_psnr(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
    data_range: float = None
) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Parameters
    ----------
    estimate : np.ndarray
        Reconstructed image.
    ground_truth : np.ndarray
        Reference clean image.
    data_range : float, optional
        Dynamic range of pixels. If None, max(ground_truth) - min(ground_truth) is used.
    """
    mse = compute_mse(estimate, ground_truth)
    if mse < 1e-15:
        return float('inf')
    if data_range is None:
        data_range = float(np.max(ground_truth) - np.min(ground_truth))
        if data_range == 0:
            data_range = float(np.max(ground_truth))
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def compute_ssim(
    estimate: np.ndarray,
    ground_truth: np.ndarray,
    win_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = None
) -> float:
    """
    Computes the Mean Structural Similarity Index Measure (SSIM).

    Parameters
    ----------
    estimate : np.ndarray
        Reconstructed image (2D float).
    ground_truth : np.ndarray
        Reference clean image (2D float).
    win_size : int, default=7
        Sliding window size.
    k1, k2 : float
        Stability constants.
    data_range : float, optional
        Dynamic range. Defaults to max(truth) - min(truth).
    """
    if data_range is None:
        data_range = float(np.max(ground_truth) - np.min(ground_truth))
        if data_range == 0:
            data_range = 1.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Local means
    mu1 = uniform_filter(estimate, size=win_size)
    mu2 = uniform_filter(ground_truth, size=win_size)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Local variances and covariance
    sigma1_sq = uniform_filter(estimate * estimate, size=win_size) - mu1_sq
    sigma2_sq = uniform_filter(ground_truth * ground_truth, size=win_size) - mu2_sq
    sigma12 = uniform_filter(estimate * ground_truth, size=win_size) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-15)

    return float(np.mean(ssim_map))


def compute_all_metrics(estimate: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Computes all quality metrics at once and returns a clean dictionary."""
    return {
        "relative_l2": compute_relative_l2(estimate, ground_truth),
        "relative_l1": compute_relative_l1(estimate, ground_truth),
        "relative_linf": compute_relative_linf(estimate, ground_truth),
        "mse": compute_mse(estimate, ground_truth),
        "psnr_db": compute_psnr(estimate, ground_truth),
        "ssim": compute_ssim(estimate, ground_truth)
    }
