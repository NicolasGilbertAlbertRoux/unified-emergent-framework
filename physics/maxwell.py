from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from core.io import load_nodes_edges
from core.graph import build_incidence
from core.operators import build_edge_laplacian
from core.utils import effective_support


def run_pre_maxwell_transverse(
    detail_dir: str | Path,
    results_dir: str | Path,
    betas: list[float],
    seeds: list[int],
    zero_tol: float = 1e-10,
    max_nonzero_modes: int = 24,
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing: list[tuple[float, int]] = []

    for beta in betas:
        for seed in seeds:
            nodes, edges = load_nodes_edges(detail_dir, beta, seed)
            if nodes is None or edges is None:
                missing.append((beta, seed))
                continue

            incidence, edge_list, _ = build_incidence(
                nodes,
                edges,
                orient_with_coordinates=False,
            )
            edge_laplacian = build_edge_laplacian(incidence)

            evals, evecs = np.linalg.eigh(edge_laplacian)
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]

            zero_mask = np.abs(evals) < zero_tol
            n_zero = int(np.sum(zero_mask))

            nonzero_evals = evals[~zero_mask]
            nonzero_evecs = evecs[:, ~zero_mask]

            n_take = min(max_nonzero_modes, len(nonzero_evals))
            nz_vals = nonzero_evals[:n_take]
            nz_vecs = nonzero_evecs[:, :n_take]

            if n_take >= 5:
                mode_idx = np.arange(1, n_take + 1, dtype=float)
                k_eff = np.sqrt(mode_idx)

                lr_lam = linregress(k_eff ** 2, nz_vals)
                lr_om = linregress(k_eff, np.sqrt(np.clip(nz_vals, 0.0, None)))

                lambda_slope = float(lr_lam.slope)
                lambda_fit_r = float(lr_lam.rvalue)
                omega_slope = float(lr_om.slope)
                omega_fit_r = float(lr_om.rvalue)
            else:
                lambda_slope = float("nan")
                lambda_fit_r = float("nan")
                omega_slope = float("nan")
                omega_fit_r = float("nan")

            div_norms: list[float] = []
            supports: list[float] = []

            for j in range(n_take):
                vec = nz_vecs[:, j]
                div_norms.append(float(np.linalg.norm(incidence @ vec)))
                supports.append(effective_support(vec))

            rows.append(
                {
                    "beta": beta,
                    "seed": seed,
                    "n_edges": len(edge_list),
                    "n_zero_modes": n_zero,
                    "n_nonzero_used": n_take,
                    "lambda_slope": lambda_slope,
                    "lambda_fit_r": lambda_fit_r,
                    "omega_slope": omega_slope,
                    "omega_fit_r": omega_fit_r,
                    "mean_divergence_nonzero": float(np.mean(div_norms)) if div_norms else float("nan"),
                    "mean_support_nonzero": float(np.mean(supports)) if supports else float("nan"),
                }
            )

    columns = [
        "beta",
        "seed",
        "n_edges",
        "n_zero_modes",
        "n_nonzero_used",
        "lambda_slope",
        "lambda_fit_r",
        "omega_slope",
        "omega_fit_r",
        "mean_divergence_nonzero",
        "mean_support_nonzero",
    ]

    if not rows:
        out = pd.DataFrame(columns=columns)
        out.to_csv(results_dir / "pre_maxwell_transverse_summary.csv", index=False)
        missing_str = ", ".join([f"(beta={b}, seed={s})" for b, s in missing])
        raise FileNotFoundError(
            "No pre-Maxwell input files were found. "
            f"Checked combinations: {missing_str}. "
            "Verify configs/default.yaml and the graph CSV files."
        )

    out = pd.DataFrame(rows, columns=columns).sort_values(["beta", "seed"])
    out.to_csv(results_dir / "pre_maxwell_transverse_summary.csv", index=False)
    return out