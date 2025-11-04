# Preferential Plane Alignment of Interstellar Object 3I/ATLAS

This repository accompanies the manuscript *“Statistical Evidence for Coherence-Weighted Trajectories”* (2025).  
It provides all source code, data, and configuration files required to reproduce the study’s figures and numerical results.

## Structure
- `data/` — Raw orbital elements and reference-plane parameters (JPL SBDB & MPC).  
- `src/` — Core analysis scripts for coordinate transformations, Monte Carlo sampling, and probability estimation.  
- `figures/` — Output plots generated during simulation runs.  
- `run_all.py` — Main executable script that reproduces all analyses and figures.

## Environment
Tested with:
```
Python 3.12  
NumPy 2.1  
Astropy 6.1  
Matplotlib 3.9  
Pandas 2.2
```
Install dependencies via:
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

## Running the Analysis
To reproduce all results, run:
```bash
python run_all.py
```
All random sampling uses NumPy’s `PCG64DXSM` RNG with fixed seeds to ensure deterministic reproducibility.  
Output figures will be saved automatically in the `figures/` directory.

## Citation
If you use this code, please cite:
> A. Observer, *Preferential Plane Alignment of Interstellar Object 3I/ATLAS*, 2025.  
> DOI: [placeholder]

---

© 2025 A. Observer. Released under the MIT License.
