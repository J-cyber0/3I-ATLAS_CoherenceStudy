#!/usr/bin/env python3
"""
Orbital element transformations and angular computations.

Implements the geometry described in §III.A–B of
"Preferential Plane Alignment of Interstellar Object 3I/ATLAS" (2025).

Functions
---------
rotation_matrix(i, Ω, ω)
    Return the 3×3 direction-cosine matrix from orbital to reference frame.
angle_to_plane(vectors, plane_normal)
    Compute the absolute angle (deg) between vectors and a plane normal.
"""

import argparse
import numpy as np


# ----------------------------------------------------------------------
#                         ROTATION MATRIX
# ----------------------------------------------------------------------
def rotation_matrix(i: float, Omega: float, omega: float) -> np.ndarray:
    """
    Construct the classical orbital rotation matrix.

    Parameters
    ----------
    i : float
        Inclination (radians)
    Omega : float
        Longitude of ascending node (radians)
    omega : float
        Argument of perihelion (radians)

    Returns
    -------
    R : (3,3) ndarray
        Composite rotation matrix Rz(Ω)·Rx(i)·Rz(ω)
    """
    Rz1 = np.array([
        [np.cos(Omega), -np.sin(Omega), 0.0],
        [np.sin(Omega),  np.cos(Omega), 0.0],
        [0.0, 0.0, 1.0],
    ])
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(i), -np.sin(i)],
        [0.0, np.sin(i),  np.cos(i)],
    ])
    Rz2 = np.array([
        [np.cos(omega), -np.sin(omega), 0.0],
        [np.sin(omega),  np.cos(omega), 0.0],
        [0.0, 0.0, 1.0],
    ])

    R = Rz1 @ Rx @ Rz2

    # Optional orthogonality check
    if not np.allclose(R @ R.T, np.identity(3), atol=1e-10):
        raise ValueError("Rotation matrix not orthogonal within tolerance.")
    return R


# ----------------------------------------------------------------------
#                      ANGLE TO REFERENCE PLANE
# ----------------------------------------------------------------------
def angle_to_plane(vectors, plane_normal):
    """
    Return the absolute angle (in DEGREES) between each vector and the given plane.
    Angles are defined within [0°, 90°].
    """
    # normalize
    v = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    n = plane_normal / np.linalg.norm(plane_normal)

    # angle between vector and normal
    cos_theta = np.abs(np.dot(v, n))
    theta = np.degrees(np.pi / 2 - np.arccos(cos_theta))  # angle from plane, not axis

    # ensure valid range
    theta = np.clip(theta, 0, 90)
    return theta

# ----------------------------------------------------------------------
#                           CLI TEST
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotation and plane-angle test utility.")
    parser.add_argument("--i", type=float, default=10.0, help="Inclination in degrees")
    parser.add_argument("--Omega", type=float, default=80.0, help="Longitude of ascending node in degrees")
    parser.add_argument("--omega", type=float, default=30.0, help="Argument of perihelion in degrees")
    args = parser.parse_args()

    R = rotation_matrix(np.radians(args.i), np.radians(args.Omega), np.radians(args.omega))
    print("Rotation matrix:\n", R)
    print("Orthogonality check (R·Rᵀ):\n", np.round(R @ R.T, 10))
