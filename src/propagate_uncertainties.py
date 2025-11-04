#!/usr/bin/env python3
"""
Estimate uncertainty propagation through Monte Carlo sampling.

Implements the procedure described in §III.A of the paper:
for each object, random orbital-element realizations are drawn from
Gaussian covariances, transformed into orbital normals, and used to
derive confidence intervals for plane-alignment angles.

If run as a module, computes simple descriptive stats on a list of
angles. When imported, provides analytic and Gaussian helpers for
isotropic probability and z-score conversion.
"""

import argparse
import numpy as np
import math
import mpmath as mp


# ---------------------------------------------------------------
#                UNCERTAINTY PROPAGATION CORE
# ---------------------------------------------------------------
def propagate_uncertainty(angles, confidence=0.95, label=None):
    """
    Compute mean, standard deviation, and confidence interval
    for an array of angular deviations (in degrees).

    Parameters
    ----------
    angles : array-like
        Monte Carlo sample of angular deviations in degrees.
    confidence : float
        Confidence level (default 0.95).
    label : str, optional
        Label to identify the dataset in printouts.

    Returns
    -------
    stats : dict
        Dictionary with mean, std, stderr, and confidence bounds.
    """
    angles = np.asarray(angles)
    mean = np.mean(angles)
    std = np.std(angles, ddof=1)
    stderr = std / np.sqrt(len(angles))

    # two-sided confidence interval assuming normality
    z = abs(float(mp.sqrt(2) * mp.erfinv(confidence))) * 2
    half_width = z * stderr
    ci_low, ci_high = mean - half_width, mean + half_width

    name = f"[{label}] " if label else ""
    print(f"{name}Mean angle: {mean:.3f}°, Std: {std:.3f}°, StdErr: {stderr:.6f}")
    print(f"{name}{int(confidence*100)}% CI: [{ci_low:.3f}°, {ci_high:.3f}°]\n")

    return {
        "mean_deg": float(mean),
        "std_deg": float(std),
        "stderr_deg": float(stderr),
        "ci_low_deg": float(ci_low),
        "ci_high_deg": float(ci_high),
        "confidence": float(confidence),
    }


# ---------------------------------------------------------------
#               ANALYTIC REFERENCE FUNCTIONS
# ---------------------------------------------------------------
def isotropic_p_value(theta_deg: float) -> float:
    """
    Analytic isotropic probability for |θ_plane| ≤ θ₀ (degrees).

    For a plane-symmetric (belt) region, the isotropic probability equals sin(θ₀),
    not 1 - cos(θ₀) as for a spherical cap about an axis.
    """
    theta = math.radians(theta_deg)
    return math.sin(theta)


def gaussian_z_from_upper_tail(p: float) -> float:
    """
    One-sided Z such that P(Z > z) = p.
    Uses inverse error function: z = sqrt(2) * erfinv(1 - 2p).
    """
    return math.sqrt(2.0) * float(mp.erfinv(1.0 - 2.0 * p))


# ---------------------------------------------------------------
#                      CLI ENTRY POINT
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute uncertainty statistics from sample angles.")
    parser.add_argument("--angles", nargs="+", type=float, required=True, help="List of angle samples (deg).")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level (default 0.95).")
    args = parser.parse_args()

    propagate_uncertainty(np.array(args.angles), confidence=args.confidence)
