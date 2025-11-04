# Preferential Plane Alignment of Interstellar Object 3I/ATLAS

This repository supports the manuscript *“Statistical Evidence for Coherence-Weighted Trajectories”* (2025).
It provides all scripts, data, and computational modules required to reproduce the figures, tables, and numerical results presented in the study.

---

## Overview
This study investigates the remarkable coplanarity of interstellar object **3I/ATLAS**, whose inbound trajectory lies within a few degrees of both the ecliptic and Jupiter’s Laplace plane.  
Monte Carlo simulations and analytic modeling quantify the statistical rarity of this configuration relative to isotropic arrival expectations.

---

## Repository Structure
- **`data/`** — Orbital elements and reference-plane definitions from JPL SBDB and MPC sources.  
- **`src/`** — Core Python modules for geometry, sampling, visualization, and statistical analysis.  
- **`figures/`** — Generated plots for publication, including orbital geometry, isotropic histograms, and cumulative distributions.  
- **`results/`** — Output JSON and NumPy arrays summarizing Monte Carlo and analytic results.  
- **`run_all.py`** — Command-line entry point orchestrating all analyses and figure generation.

---

## Computational Environment
Verified with the following dependencies:

```
Python 3.12
NumPy 2.1
Astropy 6.1
Matplotlib 3.9
SciPy 1.14
Pandas 2.2
Jupyter 1.0
```

Install via:
```
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

---

## Running the Analysis
To reproduce all figures and results:

```
# 1. Generate baseline results for each reference plane
python run_all.py --plane ECLIPTIC --N 1000000 --seed 42
python run_all.py --plane JUPITER_LAPLACE --N 1000000 --seed 42

# 2. Compare both planes and generate publication figures
python run_all.py --mode compare --plane JUPITER_LAPLACE --N 1000000 --seed 42
```

All random sampling employs NumPy’s PCG64DXSM bit generator with fixed seeds to ensure deterministic reproducibility.  
Output figures and JSON summaries are saved under `figures/` and `results/`, respectively.

---

## Reproducibility and Data Integrity
Orbital data correspond to epoch **J2000 (MPC MPEC 2025-N12)** and include propagated uncertainties.  
Monte Carlo convergence was verified to within **<0.1%** across independent realizations.  
All scripts, figures, and configuration files are publicly available at:  
🔗 [https://github.com/J-Cyber0/3I-ATLAS_CoherenceStudy](https://github.com/J-Cyber0/3I-ATLAS_CoherenceStudy)

---

## Figure Generation Workflow
Each run of `run_all.py` automatically generates the corresponding publication figures:

| Figure | Output File | Description |
|:--|:--|:--|
| **1 — Orbital Geometry** | `Figure1_OrbitalGeometry_JUPITER_LAPLACE.png` | 3D visualization of 3I/ATLAS inbound trajectory relative to ecliptic and Laplace planes. |
| **2 — Angular Offsets Histogram** | `Figure2_AngularOffsetsHistogram.png` | Isotropic angular distribution with observed θ₀ overlays and inset CDF. |
| **3 — Cumulative Probability Comparison** | `Figure3_CumulativeProbabilityComparison.png` | Comparative CDFs showing isotropic vs. observed probabilities for both planes. |

All figures are exported automatically to the `figures/` directory in both `.png` and `.pdf` formats.

---

## Citation
If you use this repository, please cite:

**J. Cyber**, *Preferential Plane Alignment of Interstellar Object 3I/ATLAS*, 2025.  
DOI: [placeholder]

© 2025 J. Cyber. Released under the **MIT License**.
