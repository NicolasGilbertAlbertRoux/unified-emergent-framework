This repository implements a unified emergent framework connecting transport, quantum-like, gauge-like, and geometric regimes from a single underlying structure.

# Unified Emergent Framework

Reference implementation for the manuscript:

**A Computational Framework for Emergent Multi-Sector Physical Structures from Discrete Energy Fields**

---

## Overview

This repository provides a reproducible implementation of a discrete emergent framework connecting:

- Schrödinger-like dynamics  
- Dirac-like spectral structure  
- Maxwell-like transverse modes  
- Geometry–source coupling  

The framework operates entirely from discrete scalar fields and graph-derived structures.

---

## Repository structure

core/            # graph construction, operators, utilities
physics/         # sector-specific analyses
preprocessing/   # data preparation scripts
configs/         # configuration files
data/
raw_maps/      # included minimal dataset
main_pipeline.py

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Included data

This repository includes a minimal set of raw input maps in:

data/raw_maps/

These files are sufficient to reproduce all results presented in the manuscript.

They correspond to a curated subset of the full experimental dataset.

---

## Pipeline structure

The full workflow is:
	1.	Load raw scalar-field maps
	2.	Extract filament graphs
	3.	Build collective density fields
	4.	Run physics sectors:
	- Schrödinger
	- Dirac
	- Maxwell
	- Einstein
	- Dispersion analysis

---

## Run everything

```bash
python main_pipeline.py
```

---

## Optional preprocessing only

```bash
python preprocessing/build_filament_graph_inputs.py
python preprocessing/build_collective_density_inputs.py
```

---

## Outputs

results/

Main outputs include:
	- schrodinger_dynamics_raw.csv
	- skeleton_hamiltonian_modes.csv
	- pre_dirac_summary.csv
	- pre_maxwell_transverse_summary.csv
	- einstein_source_metric_raw.csv
	- einstein_source_metric_summary.csv
	- dispersion_summary.csv

---

## Reproducibility

The pipeline is deterministic given the input data.

This repository provides a fully reproducible reference implementation of the analyses presented in the manuscript.

---

## Notes

	- Intermediate files are generated automatically and are not versioned
	- The included dataset is minimal but sufficient for full reproduction
	- Larger datasets can be used transparently by replacing data/raw_maps/

---

## License

MIT License

---

## Citation

If you use this code or build upon this framework, please cite the associated manuscript.

---

## Interactive explorer

A lightweight local interface is also provided:

```bash
streamlit run app.py
```

This interface launches actual simulation scripts from the repository and previews the real figures they generate.

Current domains include:
	- proto-atomic stable render
	- proto-periodic classification
	- dipole / binding regime
	- magnetic alignment
	- orbital regime
	- two-scale geometry
	- cosmology

The current version is intentionally conservative: it prioritizes honest execution of the real research code over decorative interactivity.

---

## Quick start

Generate the main reproducibility targets used in the paper:

```bash
python main.py --target all
```

Or run grouped domains individually:

```bash
python main.py --target laws
python main.py --target geometry
python main.py --target proto-atoms
python main.py --target magnetism
python main.py --target orbital
python main.py --target cosmology
```

---

## Final states

Selected .npy files are provided in results/final_states/ as reference end states for some key regimes. These files are included to facilitate rapid figure regeneration, while the corresponding scripts can also recompute them from scratch.

---

## Calibration

Although the model is formulated in abstract units, it naturally supports physically meaningful calibration.

In particular, the framework provides sufficient structure to enable external quantitative testing against known physical scales. Calibration is not imposed a priori, but can be derived from emergent structures and propagation regimes.

---

## Notes

The framework explores the emergence of wave-like, structured, flux-like, geometric, proto-atomic, magnetic, orbital and cosmological regimes from a unified discrete energetic substrate.

This repository is organized to support both direct reproducibility and further extension.

---

## License

MIT License

---

## Citation

If you use this code or build upon this framework, please cite the associated manuscript.
