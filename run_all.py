#!/usr/bin/env python3
"""
Run all core analyses for the 3I/ATLAS coherence study.
Reproduces the Monte Carlo simulation and alignment probability plot.
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from numpy.random import default_rng

def rotation_matrix(i, Omega, omega):
    Rz1 = np.array([[np.cos(Omega), -np.sin(Omega), 0],
                    [np.sin(Omega), np.cos(Omega), 0],
                    [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(i), -np.sin(i)],
                   [0, np.sin(i), np.cos(i)]])
    Rz2 = np.array([[np.cos(omega), -np.sin(omega), 0],
                    [np.sin(omega), np.cos(omega), 0],
                    [0, 0, 1]])
    return Rz1 @ Rx @ Rz2

def angle_to_plane(vectors, plane_normal):
    dot_products = np.abs(np.dot(vectors, plane_normal)) / np.linalg.norm(plane_normal)
    angles = np.degrees(90 - np.degrees(np.arccos(dot_products)))
    return angles

def main():
    print("=== 3I/ATLAS Coherence Study: Reproducible Monte Carlo Simulation ===")

    # Load data
    orbital_df = pd.read_csv("data/orbital_elements/3I_ATLAS_MPC.csv")
    with open("data/reference_planes/JUPITER_LAPLACE.json") as f:
        ref_planes = json.load(f)

    print("Loaded orbital elements and reference-plane parameters.")
    print(orbital_df.head())
    print(ref_planes)

    # Transformations (example only)
    elements = orbital_df.iloc[0]
    rot = rotation_matrix(np.radians(elements.i),
                          np.radians(elements.Omega),
                          np.radians(elements.omega))
    print("Rotation matrix:
", rot)

    # Monte Carlo isotropic sampling
    rng = default_rng(seed=42)
    N = 10**6
    theta = np.arccos(1 - 2 * rng.random(N))
    phi = 2 * np.pi * rng.random(N)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vectors = np.vstack((x, y, z)).T

    print(f"Generated {N:,} isotropic arrival vectors.")

    # Compute alignment probability
    plane_normal = np.array([0, 0, 1])  # ecliptic approximation
    angles = angle_to_plane(vectors, plane_normal)
    theta0 = 2.5
    p_within = np.mean(np.abs(angles) <= theta0)
    print(f"Computed probability: P(|θ| ≤ {theta0}°) = {p_within:.4f}")

    # Plot results
    plt.hist(np.abs(angles), bins=200, density=True, color='royalblue', alpha=0.7)
    plt.axvline(theta0, color='crimson', linestyle='--', label=f'Observed (2.5°)')
    plt.xlabel('|θ| (deg)')
    plt.ylabel('Probability Density')
    plt.title('Isotropic Plane-Angle Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/alignment_cdf.png", dpi=300)
    print("Saved figure to figures/alignment_cdf.png")

    print("=== Simulation complete ===")

if __name__ == "__main__":
    main()
