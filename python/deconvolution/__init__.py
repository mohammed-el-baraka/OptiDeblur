"""
Wiener-Hunt Image Deconvolution Package
========================================
A high-performance scientific library for 2D regularized image restoration
and ill-posed inverse problem analysis.
"""

from .core import (
    wiener_deconvolve,
    compute_otf,
    get_regularization_kernel,
    RegularizationType
)
from .metrics import (
    compute_relative_l2,
    compute_relative_l1,
    compute_relative_linf,
    compute_psnr,
    compute_ssim,
    compute_all_metrics
)
from .tapering import apply_edge_taper
from .io import load_dataset

__all__ = [
    "wiener_deconvolve",
    "compute_otf",
    "get_regularization_kernel",
    "RegularizationType",
    "compute_relative_l2",
    "compute_relative_l1",
    "compute_relative_linf",
    "compute_psnr",
    "compute_ssim",
    "compute_all_metrics",
    "apply_edge_taper",
    "load_dataset"
]
