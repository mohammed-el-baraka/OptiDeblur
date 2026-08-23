#!/usr/bin/env python3
"""
Wiener-Hunt Deconvolution CLI
=============================
Command-line interface to execute regularized image deconvolution,
parameter search, and metric evaluation.
"""

import os
import sys
from pathlib import Path

# Set cache dir for Matplotlib
os.environ["MPLCONFIGDIR"] = "/Users/med/.gemini/antigravity-ide/brain/1582d779-44e2-4505-9046-647d6b92a030/scratch/mpl"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local package import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deconvolution import (
    load_dataset,
    wiener_deconvolve,
    compute_all_metrics,
    apply_edge_taper,
    RegularizationType
)


def main():
    parser = argparse.ArgumentParser(
        description="Wiener-Hunt Regularized Image Deconvolution Solver"
    )
    parser.add_argument(
        "--dataset", "-d", choices=["1", "2"], default="1",
        help="Dataset selection (1: Gaussian blur, 2: Box/Motion blur)"
    )
    parser.add_argument(
        "--lambda-val", "-l", type=float, default=None,
        help="Regularization parameter lambda (if None, optimal is auto-discovered)"
    )
    parser.add_argument(
        "--prior", "-p", choices=["identity", "gradient", "laplacian"], default="gradient",
        help="Regularization prior operator"
    )
    parser.add_argument(
        "--taper", action="store_true",
        help="Apply edge tapering to mitigate boundary ringing"
    )
    parser.add_argument(
        "--no-center", action="store_true",
        help="Do not center PSF in frequency domain (legacy mode)"
    )
    parser.add_argument(
        "--save-output", "-s", type=str, default=None,
        help="Path to save the restored image (PNG/PDF)"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run logarithmic lambda sweep to find optimal parameter"
    )

    args = parser.parse_args()

    # Load data
    data = load_dataset(args.dataset)
    blurred = data["blurred"]
    psf = data["psf"]
    truth = data["ground_truth"]
    name = data["name"]

    print("=" * 60)
    print(f"  Wiener-Hunt Deconvolution Engine - {name}")
    print("=" * 60)
    print(f"Image Dimensions : {blurred.shape[0]} x {blurred.shape[1]}")
    print(f"PSF Dimensions   : {psf.shape[0]} x {psf.shape[1]}")
    print(f"Prior Type       : {args.prior}")
    print(f"Edge Tapering    : {'Enabled' if args.taper else 'Disabled'}")
    print(f"PSF Centering    : {'Disabled (Legacy)' if args.no_center else 'Enabled (Optimal)'}")

    input_img = apply_edge_taper(blurred, psf) if args.taper else blurred
    center = not args.no_center

    if args.sweep or args.lambda_val is None:
        print("\n--- Running Logarithmic Lambda Sweep ---")
        lambdas = np.logspace(-8, 4, 120)
        best_err = float("inf")
        best_lam = lambdas[0]

        for lam in lambdas:
            restored, _ = wiener_deconvolve(
                input_img, psf, reg_param=lam, reg_type=args.prior, center_psf=center
            )
            err = np.linalg.norm(restored - truth) / np.linalg.norm(truth)
            if err < best_err:
                best_err = err
                best_lam = lam

        print(f"Optimal Lambda (L2) : {best_lam:.4e}")
        print(f"Minimum Relative L2 : {best_err:.6f}")
        selected_lambda = best_lam
    else:
        selected_lambda = args.lambda_val

    # Perform restoration
    restored, _ = wiener_deconvolve(
        input_img, psf, reg_param=selected_lambda, reg_type=args.prior, center_psf=center
    )
    metrics = compute_all_metrics(restored, truth)

    print("\n--- Quantitative Reconstruction Quality ---")
    print(f"Applied Lambda   : {selected_lambda:.4e}")
    print(f"Relative L2 Norm : {metrics['relative_l2']:.6f}")
    print(f"Relative L1 Norm : {metrics['relative_l1']:.6f}")
    print(f"Relative L_inf   : {metrics['relative_linf']:.6f}")
    print(f"PSNR (dB)        : {metrics['psnr_db']:.2f} dB")
    print(f"SSIM             : {metrics['ssim']:.4f}")
    print("=" * 60)

    if args.save_output:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(truth, cmap="gray")
        axes[0].set_title("Ground Truth", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(blurred, cmap="gray")
        axes[1].set_title(f"Observed ({name})", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(restored, cmap="gray")
        axes[2].set_title(
            rf"Restored ($\lambda={selected_lambda:.2e}$, PSNR={metrics['psnr_db']:.1f}dB)",
            fontsize=12, fontweight="bold"
        )
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(args.save_output, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {args.save_output}")


if __name__ == "__main__":
    main()
