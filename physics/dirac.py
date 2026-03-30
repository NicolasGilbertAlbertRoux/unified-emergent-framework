from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import linregress

from core.io import load_nodes_edges
from core.graph import build_incidence
from core.operators import build_dirac_like
from core.utils import effective_support


def run_pre_dirac(
    detail_dir: str | Path,
    results_dir: str | Path,
    betas: list[float],
    seeds: list[int],
    max_mode: int = 20,
    zero_tol: float = 1e-8,
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

            incidence, edge_list, node_ids = build_incidence(nodes, edges)
            operator = build_dirac_like(incidence)

            evals, evecs = np.linalg.eigh(operator)
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]

            n_zero = int(np.sum(np.abs(evals) < zero_tol))

            evals_sorted = np.sort(evals)
            pos = evals_sorted[evals_sorted > zero_tol]
            neg = evals_sorted[evals_sorted < -zero_tol]
            neg_abs = np.sort(np.abs(neg))

            p = min(len(pos), len(neg_abs))
            symmetry_error = (
                float(np.mean(np.abs(pos[:p] - neg_abs[:p]))) if p > 0 else float("nan")
            )

            near = np.sort(np.abs(evals[np.abs(evals) > zero_tol]))[:max_mode]
            if len(near) >= 5:
                mode_idx = np.arange(1, len(near) + 1, dtype=float)
                k_eff = np.sqrt(mode_idx)
                lr = linregress(k_eff, near)
                slope_v = float(lr.slope)
                intercept = float(lr.intercept)
                fit_r = float(lr.rvalue)
            else:
                slope_v = intercept = fit_r = float("nan")

            zero_order = np.argsort(np.abs(evals))
            support_mode0 = effective_support(evecs[:, zero_order[0]]) if len(evals) else float("nan")

            rows.append(
                {
                    "beta": beta,
                    "seed": seed,
                    "n_nodes": len(node_ids),
                    "n_edges": len(edge_list),
                    "dirac_dim": operator.shape[0],
                    "n_zero_modes": n_zero,
                    "symmetry_error": symmetry_error,
                    "linear_dispersion_slope": slope_v,
                    "linear_dispersion_intercept": intercept,
                    "linear_dispersion_fit_r": fit_r,
                    "support_mode0": support_mode0,
                }
            )

    columns = [
        "beta",
        "seed",
        "n_nodes",
        "n_edges",
        "dirac_dim",
        "n_zero_modes",
        "symmetry_error",
        "linear_dispersion_slope",
        "linear_dispersion_intercept",
        "linear_dispersion_fit_r",
        "support_mode0",
    ]

    if not rows:
        out = pd.DataFrame(columns=columns)
        out.to_csv(results_dir / "pre_dirac_summary.csv", index=False)
        missing_str = ", ".join([f"(beta={b}, seed={s})" for b, s in missing])
        raise FileNotFoundError(
            "No pre-Dirac input files were found. "
            f"Checked combinations: {missing_str}. "
            "Verify configs/default.yaml and the graph CSV files."
        )

    out = pd.DataFrame(rows, columns=columns).sort_values(["beta", "seed"])
    out.to_csv(results_dir / "pre_dirac_summary.csv", index=False)
    return out            evals, evecs = np.linalg.eigh(operator)
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]

            n_zero = int(np.sum(np.abs(evals) < zero_tol))

            evals_sorted = np.sort(evals)
            pos = evals_sorted[evals_sorted > zero_tol]
            neg = evals_sorted[evals_sorted < -zero_tol]
            neg_abs = np.sort(np.abs(neg))
            p = min(len(pos), len(neg_abs))
            symmetry_error = float(np.mean(np.abs(pos[:p] - neg_abs[:p]))) if p > 0 else float("nan")

            near = np.sort(np.abs(evals[np.abs(evals) > zero_tol]))[:max_mode]
            if len(near) >= 5:
                mode_idx = np.arange(1, len(near) + 1, dtype=float)
                k_eff = np.sqrt(mode_idx)
                lr = linregress(k_eff, near)
                slope_v = float(lr.slope)
                intercept = float(lr.intercept)
                fit_r = float(lr.rvalue)
            else:
                slope_v = intercept = fit_r = float("nan")

            rows.append(
                {
                    "beta": beta,
                    "seed": seed,
                    "n_nodes": len(node_ids),
                    "n_edges": len(edge_list),
                    "dirac_dim": operator.shape[0],
                    "n_zero_modes": n_zero,
                    "symmetry_error": symmetry_error,
                    "linear_dispersion_slope": slope_v,
                    "linear_dispersion_intercept": intercept,
                    "linear_dispersion_fit_r": fit_r,
                    "support_mode0": effective_support(evecs[:, np.argsort(np.abs(evals))[0]]),
                }
            )

    out = pd.DataFrame(rows).sort_values(["beta", "seed"])
    out.to_csv(results_dir / "pre_dirac_summary.csv", index=False)
    return out
