This repository implements a unified emergent framework connecting transport, quantum-like, gauge-like, and geometric regimes from a single underlying structure.

# Unified Emergent Framework

Reference implementation for the paper:

**An Emergent Wave-Mantle Framework for Unified Effective Dynamics: From Transport Skeletons to Pre-Dirac, Pre-Maxwell, and Pre-Einstein Sectors**

## Overview

This repository provides a reproducible implementation of the main numerical pipeline used in the paper. The code is organized into:

	- `core/`: shared data loading, graph construction, operators, and utility functions
	- `physics/`: domain-specific analysis modules
	- `main_pipeline.py`: entry point running the selected analyses
	- `configs/`: simple configuration files

## Installation

```bash
pip install -r requirements.txt
```

## Expected input data

The pipeline expects precomputed input artifacts generated upstream from the experimental analysis:

- `logs/plots_dirac_filament_graph_t41a/filament_nodes_betaXX.XX_seedY.csv`
- `logs/plots_dirac_filament_graph_t41a/filament_edges_betaXX.XX_seedY.csv`
- `logs/plots_flux_collective_t47/collective_density_betaXX.XX_seedY.npy`

These files are not produced by `main_pipeline.py` itself.

## Upstream preprocessing

If the input directories are absent, generate them first with:

```bash
python build_filament_graph_inputs.py
python build_collective_density_inputs.py
```

This will create the required logs/... input structure used by the main unified pipeline.

## Outputs

Outputs are written to results/ as CSV summaries that can be used to regenerate tables and figures for the paper.

## Run everything

Once the required input files are present:

```bash
python main_pipeline.py
```

## Outputs

Outputs are written to results/ as CSV summaries that can be used to regenerate tables and figures for the paper.

The pipeline currently produces:
	- results/schrodinger/schrodinger_dynamics_raw.csv
	- results/schrodinger/skeleton_hamiltonian_modes.csv
	- results/dirac/pre_dirac_summary.csv
	- results/maxwell/pre_maxwell_transverse_summary.csv
	- results/einstein/einstein_source_metric_raw.csv
	- results/einstein/einstein_source_metric_summary.csv
	- results/dispersion/dispersion_summary.csv (if Hamiltonian modes are available)

## Reproducibility

The pipeline is deterministic given the input files.

Reproducibility therefore consists of two layers:
	1.	generating or providing the expected upstream input artifacts;
	2.	running main_pipeline.py on those fixed inputs.

No hidden manual step is required once the input directory structure is present.

## Current status

This repository is designed to mirror the analyses developed in the following stages:
	- Laplacian spectrum / effective Hamiltonian / Schrödinger sector
	- dispersion analysis
	- pre-Dirac sector
	- pre-Maxwell sector
	- pre-Einstein mantle geometry, metric, source, and closure

## License

This project is released under the MIT License. See the `LICENSE` file for details.

---

## Citation

If you use this code or build on this framework, please cite the associated manuscript.

## Notes

This repository provides a reproducible computational implementation of the analyses presented in the paper. It is intended for verification, extension, and critical comparison.# unified-emergent-framework
	- dispersion analysis
	- pre-Dirac sector
	- pre-Maxwell sector
	- pre-Einstein mantle geometry, metric, source, and closure

## License

This project is released under the MIT License. See the `LICENSE` file for details.

---

## Citation

If you use this code or build on this framework, please cite the associated manuscript.

## Notes

This repository provides a reproducible computational implementation of the analyses presented in the paper. It is intended for verification, extension, and critical comparison.# unified-emergent-framework
