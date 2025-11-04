#!/usr/bin/env python3
"""
Monte Carlo isotropic vector generator.

Implements the isotropic baseline sampling described in §III.B
of *"Preferential Plane Alignment of Interstellar Object 3I/ATLAS" (2025)*.

Generates uniformly distributed unit vectors over the unit sphere S².
Intended for use as the isotropic null model for arrival directions.

Usage (standalone):
  $ python montecarlo_isotropic.py --N 1e6 --seed 42

Returns or prints basic diagnostics; when imported, the function
`isotropic_vectors(N, seed)` provides reproducible 3×N arrays.
"""

import argparse
import numpy as np
from numpy.random import default_rng


# ----------------------------------------------------------------------
#                        CORE FUNCTION
# ----------------------------------------------------------------------
def isotropic_vectors(N: int, seed: int | None = None, dtype=np.float64) -> np.ndarray:
    """
    Generate N isotropic unit vectors on the sphere S².

    Parameters
    ----------
    N : int
        Number of random unit vectors to generate.
    seed : int or None
        Random seed for deterministic reproducibility.
    dtype : np.dtype
        Floating-point precision (default: float64).

    Returns
    -------
    vectors : (N, 3) ndarray
        Array of shape (N, 3) with unit-magnitude vectors.
    """
    rng = default_rng(seed)
    u = rng.random(N)
    v = rng.random(N)
    theta = np.arccos(1 - 2 * u)          # polar angle
    phi = 2 * np.pi * v                   # azimuthal angle

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    vectors = np.vstack((x, y, z)).T.astype(dtype, copy=False)
    return vectors


# ----------------------------------------------------------------------
#                       OPTIONAL VALIDATION
# ----------------------------------------------------------------------
def _validate_isotropy(vectors: np.ndarray, bins: int = 20):
    """
    Quick isotropy diagnostic: prints mean vector length and angular dispersion.
    Not needed for production runs; useful for testing.
    """
    norms = np.linalg.norm(vectors, axis=1)
    mean_norm = np.mean(norms)
    polar_angles = np.degrees(np.arccos(vectors[:, 2]))
    hist, _ = np.histogram(polar_angles, bins=bins, range=(0, 180), density=True)
    print(f"Mean |v| = {mean_norm:.6f} (should be 1.0)")
    print(f"Polar angle mean = {np.mean(polar_angles):.2f}°, std = {np.std(polar_angles):.2f}°")
    print(f"Histogram uniformity deviation = {np.std(hist)/np.mean(hist):.3e}")


# ----------------------------------------------------------------------
#                          CLI ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate isotropic unit vectors for Monte Carlo analysis.")
    parser.add_argument("--N", type=int, default=1000, help="Number of vectors to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--validate", action="store_true", help="Run isotropy diagnostics after generation")
    args = parser.parse_args()

    vectors = isotropic_vectors(args.N, args.seed)
    print(f"Generated {len(vectors):,} isotropic vectors.")

    if args.validate:
        _validate_isotropy(vectors)
