from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from core.io import load_density
from core.utils import (
    central_gradient,
    central_laplacian,
    hessian_tensor,
    smooth_and_normalize,
)


def run_einstein_source_metric(
    density_dir: str | Path,
    results_dir: str | Path,
    betas: list[float],
    seeds: list[int],
    sigma: float,
    alpha_metric: float,
    beta_metric: float,
    gamma_source: float,
    eta_source: float,
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing: list[tuple[float, int]] = []

    for beta in betas:
        for seed in seeds:
            rho = load_density(density_dir, beta, seed)
            if rho is None:
                missing.append((beta, seed))
                continue

            rho = smooth_and_normalize(rho.astype(float), sigma=sigma)
            ndim = rho.ndim

            grads = [central_gradient(rho, axis=ax) for ax in range(ndim)]
            hess = hessian_tensor(rho)

            for idx in np.ndindex(rho.shape):
                gvec = np.array([grads[i][idx] for i in range(ndim)], dtype=float)
                grad2 = float(np.sum(gvec * gvec))
                lap_val = float(
                    sum(central_laplacian(rho, axis=ax)[idx] for ax in range(ndim))
                )
                hloc = np.array(
                    [[hess[i, j][idx] for j in range(ndim)] for i in range(ndim)],
                    dtype=float,
                )

                outer = np.outer(gvec, gvec)
                g_eff = np.eye(ndim) + alpha_metric * outer + beta_metric * hloc
                eig_g = np.linalg.eigvalsh(g_eff)

                g_trace = float(np.sum(eig_g))
                g_det = float(np.prod(eig_g))
                g_anis = float(
                    np.max(np.abs(eig_g)) / (np.min(np.abs(eig_g)) + 1e-12)
                )

                t_eff = (
                    outer
                    + gamma_source * np.eye(ndim) * grad2
                    + eta_source * np.eye(ndim) * lap_val
                )
                eig_t = np.linalg.eigvalsh(t_eff)

                t_trace = float(np.sum(eig_t))
                t_anis = float(
                    np.max(np.abs(eig_t)) / (np.min(np.abs(eig_t)) + 1e-12)
                )

                rows.append(
                    {
                        "beta": beta,
                        "seed": seed,
                        "grad2": grad2,
                        "lap": lap_val,
                        "curvature": -lap_val,
                        "T_trace": t_trace,
                        "T_anis": t_anis,
                        "g_trace": g_trace,
                        "g_det": g_det,
                        "g_anis": g_anis,
                    }
                )

    raw_columns = [
        "beta",
        "seed",
        "grad2",
        "lap",
        "curvature",
        "T_trace",
        "T_anis",
        "g_trace",
        "g_det",
        "g_anis",
    ]

    if not rows:
        raw = pd.DataFrame(columns=raw_columns)
        raw.to_csv(results_dir / "einstein_source_metric_raw.csv", index=False)
        missing_str = ", ".join([f"(beta={b}, seed={s})" for b, s in missing])
        raise FileNotFoundError(
            "No Einstein density inputs were found. "
            f"Checked combinations: {missing_str}. "
            "Verify configs/default.yaml and the collective density NPY files."
        )

    df = pd.DataFrame(rows, columns=raw_columns)
    df.to_csv(results_dir / "einstein_source_metric_raw.csv", index=False)

    summary_rows: list[dict] = []

    for beta in betas:
        sub = df[df["beta"] == beta]
        tests = [
            ("T_trace", "g_trace"),
            ("T_trace", "g_det"),
            ("T_anis", "g_anis"),
            ("T_trace", "curvature"),
        ]

        for xvar, yvar in tests:
            x = sub[xvar].to_numpy(dtype=float)
            y = sub[yvar].to_numpy(dtype=float)

            if len(x) > 10 and np.std(x) > 0 and np.std(y) > 0:
                pear = float(pearsonr(x, y)[0])
                spear = float(spearmanr(x, y)[0])
            else:
                pear = float("nan")
                spear = float("nan")

            summary_rows.append(
                {
                    "beta": beta,
                    "x": xvar,
                    "y": yvar,
                    "pearson": pear,
                    "spearman": spear,
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(["beta", "x", "y"])
    summary.to_csv(results_dir / "einstein_source_metric_summary.csv", index=False)
    return summary