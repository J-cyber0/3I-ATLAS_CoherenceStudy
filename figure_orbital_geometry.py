#!/usr/bin/env python3
"""
figure_orbital_geometry.py

Generates Figure 1: 3I/ATLAS orbital geometry relative to
the ecliptic and Jupiter’s Laplace plane.

Now data-driven — reads plane normals from:
    data/reference_planes/{PLANE}.json

Outputs:
  • figures/figure1_orbital_geometry.png
  • figures/figure1_orbital_geometry.pdf
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ----------------------------------------------------------------------
# Utility: safe unit vector
# ----------------------------------------------------------------------
def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


# ----------------------------------------------------------------------
# Load plane definition
# ----------------------------------------------------------------------
def load_plane_normal(plane_name, data_dir="data/reference_planes"):
    fpath = os.path.join(data_dir, f"{plane_name.upper()}.json")
    with open(fpath, "r") as f:
        data = json.load(f)

    if "normal_vector" in data:
        return unit(data["normal_vector"])

    # Support legacy format: i_J / Omega_J
    i = np.radians(data.get("i_J", 0.0))
    Omega = np.radians(data.get("Omega_J", 0.0))
    normal = [
        np.sin(i) * np.sin(Omega),
        -np.sin(i) * np.cos(Omega),
        np.cos(i)
    ]
    return unit(normal)

# ----------------------------------------------------------------------
# Main plotting routine
# ----------------------------------------------------------------------
def plot_orbital_geometry(
    plane_name="JUPITER_LAPLACE",
    out_dir="./figures",
    data_dir="./data/reference_planes",
):
    """
    Render 3I/ATLAS inbound trajectory relative to the ecliptic
    and the selected reference plane.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 3I/ATLAS inbound unit vector (example; replace if known)
    v_in = unit(np.array([0.185, -0.934, 0.306]))
    n_ecl = np.array([0, 0, 1])                     # ecliptic
    n_ref = load_plane_normal(plane_name, data_dir)

    # Plane grids
    grid_size = 1.2
    x = y = np.linspace(-grid_size, grid_size, 2)
    X, Y = np.meshgrid(x, y)
    Z_ecl = np.zeros_like(X)
    Z_ref = -(n_ref[0]*X + n_ref[1]*Y)/n_ref[2]

    # Inbound trajectory
    t = np.linspace(-1.2, 0.2, 100)
    traj = np.outer(t, -v_in)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    plt.style.use("seaborn-v0_8-muted")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25, azim=120)

    ax.plot_surface(X, Y, Z_ecl, alpha=0.25, color="lightgray", linewidth=0)
    ax.plot_surface(X, Y, Z_ref, alpha=0.35, color="lightblue", linewidth=0)

    ax.plot(traj[:,0], traj[:,1], traj[:,2],
            color="crimson", lw=2.0, label="3I/ATLAS Inbound Path")

    scale = 0.8
    ax.quiver(0, 0, 0, *n_ecl, color="dimgray", length=scale,
              arrow_length_ratio=0.1, label="Ecliptic Normal")
    ax.quiver(0, 0, 0, *n_ref, color="royalblue", length=scale,
              arrow_length_ratio=0.1, label=f"{plane_name.title()} Normal")

    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-0.5,0.8)
    ax.set_xlabel("X (AU, normalized)")
    ax.set_ylabel("Y (AU, normalized)")
    ax.set_zlabel("Z (AU, normalized)")
    ax.set_box_aspect([1,1,0.8])
    ax.set_title(f"3I/ATLAS Orbital Geometry — {plane_name}", fontsize=11, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.subplots_adjust(bottom=0.18)


    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = os.path.join(out_dir, f"{plane_name}_orbital_geometry.png")
    out_pdf = os.path.join(out_dir, f"{plane_name}_orbital_geometry.pdf")
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close(fig)
    print(f"✅ Saved {plane_name} geometry figure → {out_png}")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D orbital geometry figure for 3I/ATLAS.")
    parser.add_argument("--plane", type=str, default="JUPITER_LAPLACE",
                        help="Reference plane name (e.g., ECLIPTIC, JUPITER_LAPLACE).")
    parser.add_argument("--out", type=str, default="./figures",
                        help="Output directory for figures.")
    parser.add_argument("--data", type=str, default="./data/reference_planes",
                        help="Directory containing plane JSON files.")
    args = parser.parse_args()

    plot_orbital_geometry(args.plane, args.out, args.data)
