# Preferential Plane Alignment of Interstellar Object 3I/ATLAS

This repository accompanies the manuscript *"Statistical Evidence for Coherence-Weighted Trajectories"* (2025).  
It contains scripts and data used to reproduce all figures and numerical results.

## Structure
- `data/` — Raw orbital elements and reference-plane parameters (from JPL SBDB & MPC).
- `src/` — Python scripts for coordinate transformations, Monte Carlo sampling, and statistical evaluation.
- `notebooks/` — Jupyter notebook reproducing all analyses (`CoherenceAlignment_v1.2.ipynb`).
- `figures/` — Generated plots for inclusion in the manuscript.

## Environment
Tested with:
```
Python 3.12
NumPy 2.1
Astropy 6.1
Matplotlib 3.9
Jupyter 1.0
```
Install via:
```
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

## Reproducibility
All random sampling uses NumPy's `PCG64DXSM` RNG with fixed seeds.  
Running `CoherenceAlignment_v1.2.ipynb` reproduces all probabilities and figures within <0.1% numerical variance.

## Citation
If you use this code, please cite:
> [Your Name], *Preferential Plane Alignment of Interstellar Object 3I/ATLAS*, 2025.  
> DOI: [placeholder]

---

© 2025 [Your Name]. Released under the MIT License.
