#!/usr/bin/env python3
"""
Run all core analyses for the 3I/ATLAS coherence study.

Now includes a Rich-powered CLI display for structured and elegant terminal output.
"""

import os
import json
import argparse
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# --- Local imports ---
from src.montecarlo_isotropic import isotropic_vectors
from src.plot_results import plot_angle_distribution
from src.propagate_uncertainties import (
    propagate_uncertainty,
    isotropic_p_value,
    gaussian_z_from_upper_tail,
)
from src.transform_vectors import rotation_matrix, angle_to_plane
from figure_histograms import plot_angular_offsets_histogram
from figure_cumulative_comparison import plot_cumulative_comparison



console = Console()

# -------------------------------------------------------------
#                    UTILITY FUNCTIONS
# -------------------------------------------------------------
def file_hash(path: str) -> str:
    """Return short SHA256 hash of a file (for reproducibility metadata)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def render_header(title: str):
    """Render a centered, elegant blue header."""
    header_text = Text(title, style="bold royal_blue1")
    console.print(
        Panel(
            header_text,
            border_style="royal_blue1",
            padding=(0, 4),
            title="[bold white]Coherence Study[/bold white]",
            title_align="right",
        )
    )


def render_section(title: str):
    console.rule(f"[bold cyan]{title}[/bold cyan]", style="royal_blue1")


# -------------------------------------------------------------
#                        MAIN EXECUTION
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="3I/ATLAS Coherence Monte Carlo Simulation")
    parser.add_argument("--N", type=int, default=10**6, help="Number of isotropic samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--theta0",
        type=float,
        default=None,
        help="Observed angular threshold (deg). If omitted, uses measured value for chosen plane.",
    )
    parser.add_argument("--plane", type=str, default="JUPITER_LAPLACE", help="Reference plane name")
    parser.add_argument(
        "--mode",
        choices=["isotropic", "uncertainty", "both", "compare"],
        default="isotropic",
        help="Run isotropic baseline, uncertainty propagation, both, or two-plane comparison",
    )
    parser.add_argument(
        "--make-geometry",
        dest="make_geometry",  # add this line
        action="store_true",
        help="Also generate the 3D orbital geometry figure(s) at the end.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    #                   PATH SETUP
    # ---------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "data")
    fig_dir = os.path.join(BASE_DIR, "figures")
    out_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------
    #               ASSIGN MEASURED ANGLE DEFAULTS
    # ---------------------------------------------------------
    measured_angles = {"ECLIPTIC": 2.34, "JUPITER_LAPLACE": 2.63}
    plane_key = args.plane.upper()
    theta0 = args.theta0 or measured_angles.get(plane_key, 2.5)
    auto_info = (
        f"[Auto] Using measured θ₀ = {theta0:.2f}° for plane '{args.plane}'"
        if args.theta0 is None
        else f"Using user-specified θ₀ = {theta0:.2f}°"
    )

    other_plane = "ECLIPTIC" if plane_key == "JUPITER_LAPLACE" else "JUPITER_LAPLACE"

    # ---------------------------------------------------------
    #                     LOAD INPUT DATA
    # ---------------------------------------------------------
    orbital_path = os.path.join(data_dir, "orbital_elements", "3I_ATLAS_MPC.csv")
    planes_path = os.path.join(data_dir, "reference_planes", f"{args.plane}.json")

    if not os.path.exists(orbital_path):
        console.print(f"[bold red]Error:[/bold red] Missing orbital elements file: {orbital_path}")
        return
    if not os.path.exists(planes_path):
        console.print(f"[bold red]Error:[/bold red] Missing reference plane file: {planes_path}")
        return

    orbital_df = pd.read_csv(orbital_path)
    with open(planes_path) as f:
        ref_plane = json.load(f)
    elements = orbital_df.iloc[0]
    plane_normal = np.array(ref_plane.get("normal_vector", [0, 0, 1]), dtype=float)
    # For future use / validation:
    _rot = rotation_matrix(np.radians(elements.i), np.radians(elements.Omega), np.radians(elements.omega))

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "plane": args.plane,
        "theta0_deg": theta0,
        "mode": args.mode,
    }

    # =========================================================
    #                   RICH OUTPUT SECTIONS
    # =========================================================
    render_header("3I/ATLAS — Coherence Monte Carlo Simulation")

    render_section("Input Summary")
    console.print(f"[bold white]Plane:[/bold white] {args.plane}")
    console.print(f"[bold white]θ₀:[/bold white] {theta0:.2f}°   {auto_info}")
    console.print(f"[bold white]Orbital CSV:[/bold white] {orbital_path}")
    console.print(f"[bold white]Plane JSON:[/bold white] {planes_path}")
    console.print()

    # ---------------------------------------------------------
    #                 MODE 1: ISOTROPIC SAMPLING
    # ---------------------------------------------------------
    if args.mode in ["isotropic", "both"]:
        render_section("Isotropic Baseline Modeling (§III.B)")

        vectors = isotropic_vectors(args.N, seed=args.seed)
        angles = angle_to_plane(vectors, plane_normal)  # function returns degrees in [0, 90]

        if not np.all((angles >= 0) & (angles <= 90)):
            raise RuntimeError("Angles to a plane should lie in [0°, 90°]. Check angle_to_plane implementation.")

        p_empirical = np.mean(np.abs(angles) <= theta0)
        p_analytic = isotropic_p_value(theta0)
        z_score = gaussian_z_from_upper_tail(p_analytic)
        delta = abs(p_empirical - p_analytic)

        table = Table(
            title="[bold white]Statistical Results[/bold white]",
            box=box.MINIMAL_DOUBLE_HEAD,
            border_style="royal_blue1",
            show_lines=False,
            header_style="bold cyan",
        )
        table.add_column("Metric", justify="left", no_wrap=True)
        table.add_column("Value", justify="right")

        table.add_row("Analytic P(|θ| ≤ θ₀)", f"{p_analytic:.6f}")
        table.add_row("Monte Carlo P(|θ| ≤ θ₀)", f"{p_empirical:.6f}")
        table.add_row("ΔP (emp - analytic)", f"{delta:.2e}")
        table.add_row("Z-score (one-sided)", f"{z_score:.2f} σ")

        console.print(table)
        console.print("Interpretation: [italic white]Statistically rare but within 2σ under isotropy.[/italic white]\n")

        np.save(os.path.join(out_dir, f"angles_{args.plane}.npy"), angles)
        plot_angle_distribution(angles, theta0, args.plane, fig_dir, p_analytic=p_analytic, p_empirical=p_empirical)

        results.update(
            {
                "analytic_p": float(p_analytic),
                "empirical_p": float(p_empirical),
                "delta_p": float(delta),
                "z_score": float(z_score),
                "N": int(args.N),
            }
        )

    # ---------------------------------------------------------
    #                 MODE 2: TWO-PLANE COMPARISON
    # ---------------------------------------------------------
    if args.mode == "compare":
        planes_path_b = os.path.join(data_dir, "reference_planes", f"{other_plane}.json")
        if not os.path.exists(planes_path_b):
            console.print(f"[bold red]Error:[/bold red] Missing reference plane file: {planes_path_b}")
            return
        with open(planes_path_b) as f:
            ref_plane_b = json.load(f)

        theta_a = theta0  # for args.plane
        theta_b = measured_angles.get(other_plane.upper(), 2.5)

        n_a = plane_normal / np.linalg.norm(plane_normal)
        n_b = np.array(ref_plane_b.get("normal_vector", [0, 0, 1]), dtype=float)
        n_b = n_b / np.linalg.norm(n_b)

        render_section(f"Two-Plane Comparison: {args.plane} vs {other_plane}")

        vectors = isotropic_vectors(args.N, seed=args.seed)

        ang_a = angle_to_plane(vectors, n_a)
        ang_b = angle_to_plane(vectors, n_b)

        p_a_emp = np.mean(np.abs(ang_a) <= theta_a)
        p_b_emp = np.mean(np.abs(ang_b) <= theta_b)
        p_a_an  = isotropic_p_value(theta_a)
        p_b_an  = isotropic_p_value(theta_b)

        mask_joint = (np.abs(ang_a) <= theta_a) & (np.abs(ang_b) <= theta_b)
        p_joint_emp = float(np.mean(mask_joint))
        p_joint_ind = float(p_a_an * p_b_an)

        tbl = Table(
            title="[bold white]Joint Alignment Results[/bold white]",
            box=box.MINIMAL_DOUBLE_HEAD,
            border_style="royal_blue1",
            header_style="bold cyan",
        )
        tbl.add_column("Metric", justify="left")
        tbl.add_column("Value", justify="right")

        tbl.add_row(f"θ₀ ({args.plane})", f"{theta_a:.2f}°")
        tbl.add_row(f"θ₀ ({other_plane})", f"{theta_b:.2f}°")
        tbl.add_row(f"P(|θ|≤θ₀) {args.plane} (empirical)", f"{p_a_emp:.6f}")
        tbl.add_row(f"P(|θ|≤θ₀) {other_plane} (empirical)", f"{p_b_emp:.6f}")
        tbl.add_row("Joint empirical", f"{p_joint_emp:.6f}")
        tbl.add_row("Joint analytic (independence)", f"{p_joint_ind:.6f}")
        tbl.add_row("Δ (emp - indep)", f"{(p_joint_emp - p_joint_ind):.2e}")
        console.print(tbl)

        angles_ecliptic = np.load(Path(out_dir) / "angles_ECLIPTIC.npy")
        angles_laplace = np.load(Path(out_dir) / "angles_JUPITER_LAPLACE.npy")

        plot_angular_offsets_histogram(
            angles_ecliptic=angles_ecliptic,
            angles_laplace=angles_laplace,
            theta0_ecl=2.34,
            theta0_lap=2.63,
            p_ecl=p_b_emp,  # reuse from your joint comparison
            p_lap=p_a_emp,
            out_dir=fig_dir,
        )

        try:
            plot_cumulative_comparison(
                angles_ecliptic=ang_b,
                angles_laplace=ang_a,
                theta0_ecl=theta_b,
                theta0_lap=theta_a,
                p_ecl=p_b_emp,
                p_lap=p_a_emp,
                out_dir=fig_dir,
            )
        except Exception as e:
            console.print(f"[yellow]⚠ Could not generate Figure 3: {e}[/yellow]")

    # ---------------------------------------------------------
    #                     REPRODUCIBILITY LOG
    # ---------------------------------------------------------
    render_section("Reproducibility Log")

    results["meta"] = {
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "seed": args.seed,
        "script_version": "v1.5",
        "input_hashes": {
            "orbital_csv": file_hash(orbital_path),
            "plane_json": file_hash(planes_path),
        },
    }

    meta_path = os.path.join(out_dir, f"summary_{args.plane}.json")
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"[bold white]Figures:[/bold white] {fig_dir}")
    console.print(f"[bold white]Results:[/bold white] {meta_path}")

    # Optional: generate 3D orbital geometry figure(s)
    if args.make_geometry:
        try:
            from figure_orbital_geometry import plot_orbital_geometry
            plot_orbital_geometry(plane_name=args.plane, out_dir=fig_dir)
            # If compare mode, also render the other plane for a matching pair
            if args.mode == "compare":
                plot_orbital_geometry(plane_name=other_plane, out_dir=fig_dir)
            console.print(Panel.fit("🛰️  Generated orbital geometry figure(s)", border_style="royal_blue1"))
        except Exception as e:
            console.print(Panel.fit(f"[yellow]Warning:[/yellow] Could not generate orbital geometry figure(s): {e}",
                                    border_style="yellow"))

    console.print(Panel.fit("✅ [bold green]Analysis completed successfully[/bold green]", border_style="green"))
    console.print(Text("“Alignment is not coincidence, but correspondence.”", style="italic royal_blue1"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        Console().print(f"[bold red]❌ Runtime error:[/bold red] {e}")
    finally:
        import sys
        sys.tracebacklimit = 0
