#!/usr/bin/env python3
"""
Plot histogram and cumulative probability (CDF) for alignment results.

Implements visualization for isotropic baseline distributions as described
in §IV ("Results") of the 3I/ATLAS coherence paper.

Generates:
  • Normalized histogram of |θ| (deg)
  • Cumulative distribution function (CDF)
  • Markers for the observed θ₀
  • Dual output formats (.png and .pdf)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
#                      CORE PLOTTING FUNCTION
# ----------------------------------------------------------------------
def plot_angle_distribution(
    angles,
    theta0,
    plane_name,
    fig_dir,
    p_analytic=None,
    p_empirical=None,
):
    """
    Create histogram and CDF plots for absolute angular deviations.

    Parameters
    ----------
    angles : array-like
        Angular deviations (degrees).
    theta0 : float
        Observed alignment angle (degrees).
    plane_name : str
        Label for figure titles and filenames.
    fig_dir : str
        Output directory for figures.
    p_analytic : float, optional
        Analytic probability for |θ| ≤ θ₀ (for annotation).
    p_empirical : float, optional
        Monte Carlo probability for |θ| ≤ θ₀ (for annotation).
    """
    os.makedirs(fig_dir, exist_ok=True)
    abs_angles = np.abs(angles)

    # ---------------- Histogram ----------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(abs_angles, bins=180, density=True, color="RoyalBlue", alpha=0.7)
    ax.axvline(theta0, color="crimson", linestyle="--", lw=1.5, label=f"Observed θ₀ = {theta0:.2f}°")
    ax.set_xlabel(r"$|θ|$ (deg)")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Isotropic Plane-Angle Distribution — {plane_name}")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()

    hist_png = os.path.join(fig_dir, f"{plane_name}_alignment_hist.png")
    hist_pdf = os.path.join(fig_dir, f"{plane_name}_alignment_hist.pdf")
    plt.savefig(hist_png, dpi=300)
    plt.savefig(hist_pdf)
    plt.close(fig)

    # ---------------- Cumulative (CDF) ----------------
    sorted_angles = np.sort(abs_angles)
    cdf = np.arange(1, len(sorted_angles) + 1) / len(sorted_angles)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sorted_angles, cdf, color="RoyalBlue", lw=1.5)
    ax.axvline(theta0, color="crimson", linestyle="--", lw=1.2, label=f"θ₀ = {theta0:.2f}°")
    ax.axhline(y=np.interp(theta0, sorted_angles, cdf), color="gray", linestyle=":", lw=1)
    ax.set_xlabel(r"$|θ|$ (deg)")
    ax.set_ylabel(r"Cumulative $P(|θ|≤θ)$")
    ax.set_title(f"Cumulative Probability — {plane_name}")
    ax.grid(alpha=0.25)

    # Annotate probabilities if available
    text_y = 0.15
    if p_analytic is not None:
        ax.text(
            0.97,
            text_y,
            f"Analytic P(|θ|≤θ₀) = {p_analytic:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="black",
        )
        text_y += 0.07
    if p_empirical is not None:
        ax.text(
            0.97,
            text_y,
            f"Monte Carlo = {p_empirical:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="black",
        )

    ax.legend()
    plt.tight_layout()

    cdf_png = os.path.join(fig_dir, f"{plane_name}_alignment_cdf.png")
    cdf_pdf = os.path.join(fig_dir, f"{plane_name}_alignment_cdf.pdf")
    plt.savefig(cdf_png, dpi=300)
    plt.savefig(cdf_pdf)
    plt.close(fig)

    return {"hist_png": hist_png, "cdf_png": cdf_png}


# ----------------------------------------------------------------------
#                          CLI ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot histogram and CDF for angular alignment samples.")
    parser.add_argument("--angles", nargs="+", type=float, help="List of angular samples (deg).")
    parser.add_argument("--theta0", type=float, default=2.5, help="Observed threshold angle (deg).")
    parser.add_argument("--plane", type=str, default="TEST", help="Plane name label.")
    parser.add_argument("--out", type=str, default="./figures", help="Output directory for figures.")
    args = parser.parse_args()

    if args.angles:
        plot_angle_distribution(np.array(args.angles), args.theta0, args.plane, args.out)
        print(f"Plots saved under {args.out}")
