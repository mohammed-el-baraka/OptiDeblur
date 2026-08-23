#!/usr/bin/env python3
"""
Master Scientific Figure Generator
==================================
Generates publication-quality, high-resolution figures (300 DPI) for the
Wiener-Hunt Image Deconvolution technical report and GitHub repository.
"""

import os
import sys
from pathlib import Path

# Setup Matplotlib cache directory
os.environ["MPLCONFIGDIR"] = "/Users/med/.gemini/antigravity-ide/brain/1582d779-44e2-4505-9046-647d6b92a030/scratch/mpl"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set style for publication quality
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.autolayout": False,
    "mathtext.fontset": "cm"
})

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deconvolution import (
    load_dataset,
    wiener_deconvolve,
    compute_otf,
    compute_all_metrics,
    apply_edge_taper,
    get_regularization_kernel,
    RegularizationType
)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def figure1_spectral_analysis():
    """Figure 1: Observed Images and 2D Log-Magnitude Fourier Spectra."""
    print("Generating Figure 1: Spectral Analysis...")
    d1 = load_dataset(1)
    d2 = load_dataset(2)
    truth = d1["ground_truth"]

    N, M = truth.shape
    freq_x = np.fft.fftshift(np.fft.fftfreq(M))
    freq_y = np.fft.fftshift(np.fft.fftfreq(N))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=300)

    # Row 1: Spatial Images
    im0 = axes[0, 0].imshow(truth, cmap="gray")
    axes[0, 0].set_title("(a) Ground Truth ($x^*$)", fontweight="bold")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(d1["blurred"], cmap="gray")
    axes[0, 1].set_title("(b) Observed Data1 (Gaussian Blur)", fontweight="bold")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(d2["blurred"], cmap="gray")
    axes[0, 2].set_title("(c) Observed Data2 (Box/Motion Blur)", fontweight="bold")
    axes[0, 2].axis("off")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 2: 2D Spectra
    spec_truth = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(truth))) + 1.0)
    spec_d1 = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(d1["blurred"]))) + 1.0)
    spec_d2 = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(d2["blurred"]))) + 1.0)

    im3 = axes[1, 0].imshow(spec_truth, extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]], cmap="inferno")
    axes[1, 0].set_title(r"(d) Spectrum $\log_{10}|X^*(\nu)|$", fontweight="bold")
    axes[1, 0].set_xlabel(r"Frequency $\nu_x$")
    axes[1, 0].set_ylabel(r"Frequency $\nu_y$")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(spec_d1, extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]], cmap="inferno")
    axes[1, 1].set_title(r"(e) Spectrum $\log_{10}|Y_1(\nu)|$ (Isotropic)", fontweight="bold")
    axes[1, 1].set_xlabel(r"Frequency $\nu_x$")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(spec_d2, extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]], cmap="inferno")
    axes[1, 2].set_title(r"(f) Spectrum $\log_{10}|Y_2(\nu)|$ (Directional Nulls)", fontweight="bold")
    axes[1, 2].set_xlabel(r"Frequency $\nu_x$")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = FIG_DIR / "fig1_spectral_analysis.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def figure2_psf_transfer_functions():
    """Figure 2: 3D Point Spread Functions and Optical Transfer Functions with cross-sections."""
    print("Generating Figure 2: PSF and Optical Transfer Functions...")
    d1 = load_dataset(1)
    d2 = load_dataset(2)

    psf1 = d1["psf"]
    psf2 = d2["psf"]
    N, M = 256, 256

    H1 = compute_otf(psf1, (N, M), center_psf=True)
    H2 = compute_otf(psf2, (N, M), center_psf=True)

    H1_shift = np.fft.fftshift(H1)
    H2_shift = np.fft.fftshift(H2)

    freq_x = np.fft.fftshift(np.fft.fftfreq(M))
    freq_y = np.fft.fftshift(np.fft.fftfreq(N))
    center = N // 2

    fig = plt.figure(figsize=(15, 10), dpi=300)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1.0], width_ratios=[1, 1, 1.1])

    # 3D PSF Surface 1
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    X1, Y1 = np.meshgrid(np.arange(psf1.shape[1]), np.arange(psf1.shape[0]))
    ax1.plot_surface(X1, Y1, psf1, cmap="viridis", edgecolor="none", alpha=0.9)
    ax1.set_title("(a) PSF 1: Gaussian-like", fontweight="bold")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("Amplitude")
    ax1.view_init(elev=35, azim=45)

    # 3D PSF Surface 2
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    X2, Y2 = np.meshgrid(np.arange(psf2.shape[1]), np.arange(psf2.shape[0]))
    ax2.plot_surface(X2, Y2, psf2, cmap="plasma", edgecolor="none", alpha=0.9)
    ax2.set_title("(b) PSF 2: 7x7 Uniform Box", fontweight="bold")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("Amplitude")
    ax2.view_init(elev=35, azim=45)

    # 2D Transfer function map 1
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(np.log10(np.abs(H1_shift) + 1e-6), extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]], cmap="magma")
    ax3.set_title(r"(c) $|H_1(\nu)|$ (Log Scale)", fontweight="bold")
    ax3.set_xlabel(r"Frequency $\nu_x$")
    ax3.set_ylabel(r"Frequency $\nu_y$")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 2D Transfer function map 2 (with null lines)
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(np.log10(np.abs(H2_shift) + 1e-6), extent=[freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]], cmap="magma")
    ax4.set_title(r"(d) $|H_2(\nu)|$ (Log Scale, Zero Slices)", fontweight="bold")
    ax4.set_xlabel(r"Frequency $\nu_x$")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # 1D Cross section Comparison
    ax5 = fig.add_subplot(gs[:, 2])
    ax5.plot(freq_x, np.abs(H1_shift[center, :]), "b-", lw=2.2, label=r"$|H_1(\nu)|$ (Gaussian: Strictly $>0$)")
    ax5.plot(freq_x, np.abs(H2_shift[center, :]), "r--", lw=2.2, label=r"$|H_2(\nu)|$ (Box: Zero Crossings)")
    ax5.axhline(0, color="gray", linestyle=":", lw=1)
    ax5.set_title(r"(e) Frequency Cross-Section $\nu_y=0$", fontweight="bold")
    ax5.set_xlabel(r"Normalized Frequency $\nu_x$")
    ax5.set_ylabel(r"Magnitude $|H(\nu_x, 0)|$")
    ax5.grid(True, linestyle="--", alpha=0.6)
    ax5.legend(loc="upper right", frameon=True)

    # Annotate zeros
    zeros = freq_x[np.isclose(np.abs(H2_shift[center, :]), 0, atol=0.015)]
    if len(zeros) > 0:
        ax5.annotate("Zero Crossing\n(Irreversible Loss)",
                     xy=(zeros[len(zeros)//2], 0), xytext=(zeros[len(zeros)//2]+0.05, 0.25),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                     fontweight="bold", color="darkred")

    plt.tight_layout()
    out_path = FIG_DIR / "fig2_psf_transfer_functions.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def figure3_lambda_regimes():
    """Figure 3: Deconvolution visual gallery across 6 lambda regimes."""
    print("Generating Figure 3: Lambda Regimes Visual Gallery...")
    d1 = load_dataset(1)
    blurred = d1["blurred"]
    psf = d1["psf"]
    truth = d1["ground_truth"]

    test_lambdas = [0.0, 1e-8, 1e-5, 1e-2, 1.0, 100.0]
    titles = [
        r"$\lambda = 0$ (Inverse Filter)",
        r"$\lambda = 10^{-8}$ (Under-reg.)",
        r"$\lambda = 10^{-5}$ (Mild Under-reg.)",
        r"$\lambda = 10^{-2}$ (Optimal $\lambda^*$)",
        r"$\lambda = 1.0$ (Over-reg.)",
        r"$\lambda = 100$ (Severe Smoothing)"
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), dpi=300)
    axes = axes.ravel()

    # Ground truth & Observed
    axes[0].imshow(truth, cmap="gray")
    axes[0].set_title("(a) Ground Truth", fontweight="bold", color="darkgreen")
    axes[0].axis("off")

    axes[1].imshow(blurred, cmap="gray")
    metrics_obs = compute_all_metrics(blurred, truth)
    axes[1].set_title(f"(b) Observed\n(PSNR: {metrics_obs['psnr_db']:.1f} dB)", fontweight="bold", color="darkblue")
    axes[1].axis("off")

    for i, lam in enumerate(test_lambdas):
        restored, _ = wiener_deconvolve(blurred, psf, reg_param=lam, reg_type="gradient", center_psf=True)
        m = compute_all_metrics(restored, truth)
        ax = axes[i + 2]
        ax.imshow(restored, cmap="gray")
        color = "darkred" if i == 0 or i == 5 else ("darkgreen" if i == 3 else "black")
        ax.set_title(f"({chr(99+i)}) {titles[i]}\n(Err: {m['relative_l2']:.3f}, {m['psnr_db']:.1f} dB)",
                     fontweight="bold" if i == 3 else "normal", color=color)
        ax.axis("off")

    plt.tight_layout()
    out_path = FIG_DIR / "fig3_lambda_regimes.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def figure4_u_curve_optimization():
    """Figure 4: Convex U-curves (Error vs Lambda) for Data1 and Data2."""
    print("Generating Figure 4: U-Curve Optimization...")
    d1 = load_dataset(1)
    d2 = load_dataset(2)

    lambdas = np.logspace(-8, 4, 150)
    errs1 = []
    errs2 = []

    for lam in lambdas:
        r1, _ = wiener_deconvolve(d1["blurred"], d1["psf"], reg_param=lam, reg_type="gradient", center_psf=True)
        r2, _ = wiener_deconvolve(d2["blurred"], d2["psf"], reg_param=lam, reg_type="gradient", center_psf=True)
        errs1.append(compute_all_metrics(r1, d1["ground_truth"])["relative_l2"])
        errs2.append(compute_all_metrics(r2, d2["ground_truth"])["relative_l2"])

    errs1 = np.array(errs1)
    errs2 = np.array(errs2)

    idx_opt1 = np.argmin(errs1)
    idx_opt2 = np.argmin(errs2)
    opt_lam1 = lambdas[idx_opt1]
    opt_lam2 = lambdas[idx_opt2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Data1 U-Curve
    axes[0].semilogx(lambdas, errs1, "b-", lw=2.5, label=r"Reconstruction Error $\Delta_2(\lambda)$")
    axes[0].plot(opt_lam1, errs1[idx_opt1], "r*", markersize=14, markeredgewidth=1.5,
                 label=f"Optimal $\\lambda^* = {opt_lam1:.2e}$\n(Min Error = {errs1[idx_opt1]:.4f})")
    axes[0].set_title("(a) Data1: Gaussian Blur Regularization Curve", fontweight="bold")
    axes[0].set_xlabel(r"Regularization Parameter $\lambda$")
    axes[0].set_ylabel(r"Relative $L_2$ Error $\Delta_2$")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].annotate("Noise Amplification\n(High Variance)", xy=(1e-7, 0.30), xytext=(1e-7, 0.35),
                     arrowprops=dict(arrowstyle="->", color="blue"), fontsize=9, color="navy")
    axes[0].annotate("Over-smoothing\n(High Bias)", xy=(100, 0.30), xytext=(10, 0.35),
                     arrowprops=dict(arrowstyle="->", color="blue"), fontsize=9, color="navy")

    # Data2 U-Curve
    axes[1].semilogx(lambdas, errs2, "m-", lw=2.5, label=r"Reconstruction Error $\Delta_2(\lambda)$")
    axes[1].plot(opt_lam2, errs2[idx_opt2], "r*", markersize=14, markeredgewidth=1.5,
                 label=f"Optimal $\\lambda^* = {opt_lam2:.2e}$\n(Min Error = {errs2[idx_opt2]:.4f})")
    axes[1].set_title("(b) Data2: Box/Motion Blur Regularization Curve", fontweight="bold")
    axes[1].set_xlabel(r"Regularization Parameter $\lambda$")
    axes[1].set_ylabel(r"Relative $L_2$ Error $\Delta_2$")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].annotate("Severe Noise Instability\n(Transfer Function Zeros)", xy=(1e-7, 1.0), xytext=(1e-6, 1.4),
                     arrowprops=dict(arrowstyle="->", color="magenta"), fontsize=9, color="purple")

    plt.tight_layout()
    out_path = FIG_DIR / "fig4_u_curve_optimization.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def figure5_distance_metrics():
    """Figure 5: Multi-norm Concordance (L1, L2, Linf, PSNR, SSIM)."""
    print("Generating Figure 5: Multi-Norm Distance Concordance...")
    d2 = load_dataset(2)
    lambdas = np.logspace(-6, 3, 100)

    l2_list, l1_list, linf_list, psnr_list, ssim_list = [], [], [], [], []

    for lam in lambdas:
        restored, _ = wiener_deconvolve(d2["blurred"], d2["psf"], reg_param=lam, reg_type="gradient", center_psf=True)
        m = compute_all_metrics(restored, d2["ground_truth"])
        l2_list.append(m["relative_l2"])
        l1_list.append(m["relative_l1"])
        linf_list.append(m["relative_linf"])
        psnr_list.append(m["psnr_db"])
        ssim_list.append(m["ssim"])

    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)

    ax1.semilogx(lambdas, l2_list, "b-", lw=2.2, label=r"Relative $L_2$ Error ($\Delta_2$)")
    ax1.semilogx(lambdas, l1_list, "r--", lw=2.2, label=r"Relative $L_1$ Error ($\Delta_1$)")
    ax1.semilogx(lambdas, linf_list, "g-.", lw=2.2, label=r"Relative $L_\infty$ Error ($\Delta_\infty$)")
    ax1.set_xlabel(r"Regularization Parameter $\lambda$", fontsize=12)
    ax1.set_ylabel("Normalized Distance Metric", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Second axis for PSNR
    ax2 = ax1.twinx()
    ax2.semilogx(lambdas, psnr_list, "m:", lw=2.5, label="PSNR (dB)")
    ax2.set_ylabel("Peak Signal-to-Noise Ratio (dB)", color="purple", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="purple")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", frameon=True)
    ax1.set_title(r"Metric Concordance across Regularization Regimes ($\Delta_1, \Delta_2, \Delta_\infty, \mathrm{PSNR}$)", fontweight="bold")

    plt.tight_layout()
    out_path = FIG_DIR / "fig5_distance_metrics.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def figure6_prior_comparison():
    """Figure 6: Comparison of Priors (Tikhonov vs Gradient vs Laplacian)."""
    print("Generating Figure 6: Prior Regularization Comparison...")
    d1 = load_dataset(1)
    lambdas = np.logspace(-6, 2, 80)
    priors = ["identity", "gradient", "laplacian"]
    prior_names = ["Identity (Tikhonov)", "1st-Order Gradient", "2nd-Order Laplacian"]
    colors = ["teal", "crimson", "navy"]

    fig = plt.figure(figsize=(15, 8.5), dpi=300)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.1, 1.0])

    ax_curve = fig.add_subplot(gs[0, :])

    best_restorations = {}
    for p, name, col in zip(priors, prior_names, colors):
        errs = []
        for lam in lambdas:
            res, _ = wiener_deconvolve(d1["blurred"], d1["psf"], reg_param=lam, reg_type=p, center_psf=True)
            errs.append(compute_all_metrics(res, d1["ground_truth"])["relative_l2"])
        errs = np.array(errs)
        opt_idx = np.argmin(errs)
        opt_lam = lambdas[opt_idx]

        best_res, _ = wiener_deconvolve(d1["blurred"], d1["psf"], reg_param=opt_lam, reg_type=p, center_psf=True)
        best_restorations[p] = (best_res, opt_lam, errs[opt_idx])

        ax_curve.semilogx(lambdas, errs, color=col, lw=2.2, label=f"{name} (Opt $\\lambda={opt_lam:.2e}$, Min Err={errs[opt_idx]:.4f})")
        ax_curve.plot(opt_lam, errs[opt_idx], "o", color=col, markersize=8)

    ax_curve.set_title(r"(a) Error vs $\lambda$ for Different Regularization Priors", fontweight="bold")
    ax_curve.set_xlabel(r"Regularization Parameter $\lambda$")
    ax_curve.set_ylabel(r"Relative $L_2$ Error $\Delta_2$")
    ax_curve.grid(True, linestyle="--", alpha=0.5)
    ax_curve.legend(loc="upper right", frameon=True)

    # Subplots of optimal reconstructions
    for i, (p, name) in enumerate(zip(priors, prior_names)):
        ax = fig.add_subplot(gs[1, i])
        res, opt_lam, min_err = best_restorations[p]
        m = compute_all_metrics(res, d1["ground_truth"])
        ax.imshow(res, cmap="gray")
        ax.set_title(f"({chr(98+i)}) {name}\n$\\lambda^*={opt_lam:.2e}$ | PSNR={m['psnr_db']:.1f} dB", fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    out_path = FIG_DIR / "fig6_prior_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def figure7_error_spatial_maps():
    """Figure 7: Spatial Error Heatmaps (|Truth - Estimate|)."""
    print("Generating Figure 7: Spatial Error Heatmaps...")
    d1 = load_dataset(1)
    truth = d1["ground_truth"]
    blurred = d1["blurred"]
    psf = d1["psf"]

    res_inv, _ = wiener_deconvolve(blurred, psf, reg_param=0.0, reg_type="gradient", center_psf=True)
    res_opt, _ = wiener_deconvolve(blurred, psf, reg_param=1e-2, reg_type="gradient", center_psf=True)

    err_obs = np.abs(truth - blurred)
    err_inv = np.abs(truth - res_inv)
    err_opt = np.abs(truth - res_opt)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    im1 = axes[0].imshow(err_obs, cmap="magma")
    axes[0].set_title(f"(a) Error: Observed Image\n(Max Err: {np.max(err_obs):.2f})", fontweight="bold")
    axes[0].axis("off")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(err_inv, cmap="magma", vmax=np.percentile(err_inv, 99))
    axes[1].set_title(r"(b) Error: Inverse Filter ($\lambda=0$)" + "\n(Severe High-Freq. Noise)", fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = axes[2].imshow(err_opt, cmap="magma", vmax=np.max(err_obs))
    axes[2].set_title(r"(c) Error: Optimal Wiener-Hunt ($\lambda^*=10^{-2}$)" + "\n(Residuals Concentrated at Edges)", fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = FIG_DIR / "fig7_error_spatial_maps.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def figure8_edge_tapering():
    """Figure 8: Boundary Ringing Artifacts and Edge Tapering Demonstration."""
    print("Generating Figure 8: Edge Tapering Demonstration...")
    d2 = load_dataset(2)
    blurred = d2["blurred"]
    psf = d2["psf"]
    truth = d2["ground_truth"]

    # Without tapering
    res_notaper, _ = wiener_deconvolve(blurred, psf, reg_param=1e-3, reg_type="gradient", center_psf=True)

    # With tapering
    tapered_input = apply_edge_taper(blurred, psf)
    res_taper, _ = wiener_deconvolve(tapered_input, psf, reg_param=1e-3, reg_type="gradient", center_psf=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    axes[0].imshow(tapered_input, cmap="gray")
    axes[0].set_title(r"(a) Edge Tapered Observation $y_{\mathrm{taper}}$", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(res_notaper, cmap="gray")
    axes[1].set_title("(b) Deconvolution without Tapering\n(High Frequency Boundary Rings)", fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(res_taper, cmap="gray")
    axes[2].set_title("(c) Deconvolution with Smooth Tapering\n(Boundary Ringing Suppressed)", fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = FIG_DIR / "fig8_edge_tapering.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=== Generating All Scientific Figures ===")
    figure1_spectral_analysis()
    figure2_psf_transfer_functions()
    figure3_lambda_regimes()
    figure4_u_curve_optimization()
    figure5_distance_metrics()
    figure6_prior_comparison()
    figure7_error_spatial_maps()
    figure8_edge_tapering()
    print("=== All figures generated successfully! ===")


if __name__ == "__main__":
    main()
