# Preferential Plane Alignment of Interstellar Object 3I/ATLAS

This repository supports the manuscript *“Statistical Evidence for Coherence-Weighted Trajectories”* (2025).  
It provides all scripts, data, and computational notebooks required to reproduce the figures, tables, and numerical results presented in the study.

---

## Overview
This study investigates the remarkable coplanarity of interstellar object **3I/ATLAS**, whose inbound trajectory lies within a few degrees of both the ecliptic and Jupiter’s Laplace plane.  
Monte Carlo simulations and analytic modeling quantify the statistical rarity of this configuration relative to isotropic arrival expectations.

---

## Repository Structure
- **`data/`** — Orbital elements and reference-plane definitions from JPL SBDB and MPC sources.  
- **`src/`** — Core Python modules for geometry, sampling, and statistical analysis.  
- **`figures/`** — Generated probability plots and alignment distributions.  
- **`notebooks/`** — Jupyter notebook (`CoherenceAlignment_v1.2.ipynb`) reproducing the full workflow interactively.  
- **`run_all.py`** — Command-line entry point for batch execution.  

---

## Computational Environment
Verified with the following dependencies:
```bash
Python 3.12
NumPy 2.1
Astropy 6.1
Matplotlib 3.9
SciPy 1.14
Pandas 2.2
Jupyter 1.0
```
Install via:
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

---

## Running the Analysis
To reproduce all figures and results:
```bash
python run_all.py
```
All random sampling employs NumPy’s `PCG64DXSM` bit generator with fixed seeds to ensure deterministic reproducibility.  
Output figures and numerical summaries are saved in the `figures/` and `results/` directories, respectively.

---

## Reproducibility and Data Integrity
Orbital data correspond to epoch **J2000** (MPC MPEC 2025-N12) and include propagated uncertainties.  
Monte Carlo results converge within **<0.1%** across independent realizations, as verified in benchmark tests.  
All scripts and configuration files are publicly available at:  
🔗 [https://github.com/J-Cyber0/3I-ATLAS_CoherenceStudy](https://github.com/J-Cyber0/3I-ATLAS_CoherenceStudy)

---

## Citation
If you use this repository, please cite:
> J. Cyber, *Preferential Plane Alignment of Interstellar Object 3I/ATLAS*, 2025.  
> DOI: [placeholder]

---

© 2025 J. Cyber. Released under the MIT License.
