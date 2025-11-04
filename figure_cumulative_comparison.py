#!/usr/bin/env python3
"""
Figure 3 — Cumulative Probability Comparison

Integrated with run_all.py to visualize isotropic cumulative distributions
for the ecliptic and Jupiter-Laplace planes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_cumulative_comparison(
    angles_ecliptic: np.ndarray,
    angles_laplace: np.ndarray,
    theta0_ecl: float,
    theta0_lap: float,
    p_ecl: float,
    p_lap: float,
    out_dir: str = "figures",
):
    """Generate cumulative probability comparison figure (Figure 3)."""
    os.makedirs(out_dir, exist_ok=True)

    sorted_ecl = np.sort(np.abs(angles_ecliptic))
    sorted_lap = np.sort(np.abs(angles_laplace))
    cdf_ecl = np.arange(1, len(sorted_ecl) + 1) / len(sorted_ecl)
    cdf_lap = np.arange(1, len(sorted_lap) + 1) / len(sorted_lap)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sorted_ecl, cdf_ecl, color="RoyalBlue", lw=1.6, label="Ecliptic Plane")
    ax.plot(sorted_lap, cdf_lap, color="orange", lw=1.6, label="Jupiter Laplace Plane")

    # Markers
    ax.axvline(theta0_ecl, color="RoyalBlue", linestyle="--", lw=1.2)
    ax.axvline(theta0_lap, color="orange", linestyle="--", lw=1.2)

    # Labels
    ax.text(theta0_ecl + 1, p_ecl + 0.01, f"P={p_ecl:.3f}", color="RoyalBlue", fontsize=9)
    ax.text(theta0_lap + 1, p_lap + 0.01, f"P={p_lap:.3f}", color="orange", fontsize=9)

    # Aesthetics
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 0.2)
    ax.set_xlabel(r"$|\theta_{\mathrm{plane}}|$ (degrees)")
    ax.set_ylabel(r"Cumulative Probability $P(|\theta| \leq \theta_0)$")
    ax.set_title("Cumulative Alignment Probability — 3I/ATLAS")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    plt.tight_layout()

    out_png = os.path.join(out_dir, "Figure3_CumulativeProbabilityComparison.png")
    out_pdf = os.path.join(out_dir, "Figure3_CumulativeProbabilityComparison.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"✅ Saved Figure 3 → {out_png}")
