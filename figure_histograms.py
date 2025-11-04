#!/usr/bin/env python3
"""
Figure 2 — Angular Offsets Histogram

Visualizes the isotropic angular distribution of arrival directions
and overlays 3I/ATLAS’s measured offsets relative to the ecliptic and
Jupiter Laplace planes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ---------------------------------------------------------------------
def plot_angular_offsets_histogram(
    angles_ecliptic,
    angles_laplace,
    theta0_ecl,
    theta0_lap,
    p_ecl,
    p_lap,
    out_dir="figures",
):
    os.makedirs(out_dir, exist_ok=True)

    # Combined isotropic pool for smoother histogram
    all_angles = np.concatenate([angles_ecliptic, angles_laplace])
    bins = np.linspace(0, 90, 180)

    plt.figure(figsize=(7, 4.5))
    plt.hist(
        all_angles,
        bins=bins,
        density=True,
        color="lightgray",
        edgecolor="none",
        alpha=0.8,
        label="Isotropic Distribution",
    )

    # Ecliptic marker
    plt.axvline(
        theta0_ecl,
        color="RoyalBlue",
        linestyle="--",
        lw=1.6,
        label=fr"Ecliptic θ₀ = {theta0_ecl:.2f}°  (P={p_ecl:.3f})",
    )

    # Laplace marker
    plt.axvline(
        theta0_lap,
        color="orange",
        linestyle="--",
        lw=1.6,
        label=fr"Jupiter Laplace θ₀ = {theta0_lap:.2f}°  (P={p_lap:.3f})",
    )

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xlabel(r"$|\theta_{\mathrm{plane}}|$ (degrees)")
    plt.ylabel("Probability Density (%)")
    plt.title("Angular Offsets Under Isotropy — 3I/ATLAS")
    plt.grid(alpha=0.3)#!/usr/bin/env python3
"""
Figure 2 — Angular Offsets Histogram + Inset CDF

Visualizes the isotropic angular distribution of arrival directions
and overlays 3I/ATLAS’s measured offsets relative to:
  • the heliocentric ecliptic plane, and
  • Jupiter’s mean Laplace plane.

Inset CDF shows cumulative probability under isotropy.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_angular_offsets_histogram(
    angles_ecliptic,
    angles_laplace,
    theta0_ecl,
    theta0_lap,
    p_ecl,
    p_lap,
    out_dir="figures",
):
    os.makedirs(out_dir, exist_ok=True)

    # Combine isotropic samples for smoother histogram
    all_angles = np.concatenate([angles_ecliptic, angles_laplace])
    bins = np.linspace(0, 90, 180)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.hist(
        all_angles,
        bins=bins,
        density=True,
        color="lightgray",
        alpha=0.85,
        edgecolor="none",
        label="Isotropic Distribution",
    )

    # Vertical lines for observed angles
    ax.axvline(
        theta0_ecl,
        color="RoyalBlue",
        linestyle="--",
        lw=1.6,
        label=fr"Ecliptic θ₀ = {theta0_ecl:.2f}°  (P = {p_ecl:.3f})",
    )
    ax.axvline(
        theta0_lap,
        color="darkorange",
        linestyle="--",
        lw=1.6,
        label=fr"Jupiter Laplace θ₀ = {theta0_lap:.2f}°  (P = {p_lap:.3f})",
    )

    # Main axis formatting
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, 90)
    ax.set_xlabel(r"$|\theta_{\mathrm{plane}}|$ (degrees)")
    ax.set_ylabel("Probability Density (%)")
    ax.set_title("Angular Offsets Under Isotropy — 3I/ATLAS")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(frameon=False, fontsize=9)

    # --- Inset CDF ---
    inset = inset_axes(ax, width="33%", height="42%", loc="lower right",
                   bbox_to_anchor=(0.68, 0.48, 0.3, 0.5), bbox_transform=ax.transAxes)
    sorted_angles = np.sort(all_angles)
    cdf = np.arange(1, len(sorted_angles) + 1) / len(sorted_angles)
    inset.plot(sorted_angles, cdf, color="gray", lw=1.2)
    inset.axvline(theta0_ecl, color="RoyalBlue", linestyle="--", lw=1)
    inset.axvline(theta0_lap, color="darkorange", linestyle="--", lw=1)
    inset.set_xlim(0, 10)
    inset.set_ylim(0, 0.1)
    inset.set_xticks([0, 5, 10])
    inset.set_yticks([0, 0.05, 0.1])
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.2)
    inset.set_title("Cumulative P(|θ|≤θ₀)", fontsize=8, pad=2)

    fig.tight_layout()

    # Save both formats
    fname_base = os.path.join(out_dir, "Figure_2_AngularOffsetsHistogram")
    fig.savefig(fname_base + ".png", dpi=300)
    fig.savefig(fname_base + ".pdf")
    plt.close(fig)
    print(f"✅ Saved Figure 2 with inset CDF → {fname_base}.png")


if __name__ == "__main__":
    base = "results"
    out_dir = "figures"

    angles_ecliptic = np.load(os.path.join(base, "angles_ECLIPTIC.npy"))
    angles_laplace = np.load(os.path.join(base, "angles_JUPITER_LAPLACE.npy"))

    plot_angular_offsets_histogram(
        angles_ecliptic=angles_ecliptic,
        angles_laplace=angles_laplace,
        theta0_ecl=2.34,
        theta0_lap=2.63,
        p_ecl=0.0412,
        p_lap=0.0462,
        out_dir=out_dir,   # <-- pass it in here
    )

    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    path_png = os.path.join(out_dir, "Figure2_AngularOffsetsHistogram.png")
    path_pdf = os.path.join(out_dir, "Figure2_AngularOffsetsHistogram.pdf")
    plt.savefig(path_png, dpi=300)
    plt.savefig(path_pdf)
    plt.close()
    print(f"Saved Figure 2 → {path_png}")


