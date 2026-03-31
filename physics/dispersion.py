from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress


def run_dispersion_fit(
    modes_csv: str | Path,
    results_dir: str | Path,
    best_params: dict[str, dict[str, float]],
    max_mode: int = 11,
) -> pd.DataFrame:
    modes_csv = Path(modes_csv)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(modes_csv)
    fit_rows: list[dict] = []

    for beta_str, params in best_params.items():
        beta = float(beta_str)
        sub = df[
            (np.isclose(df["beta"], beta))
            & (np.isclose(df["alpha"], params["alpha"]))
            & (np.isclose(df["beta_pot"], params["beta_pot"]))
            & (df["mode"] <= max_mode)
        ].copy()

        if sub.empty:
            continue

        avg = (
            sub.groupby("mode", as_index=False)
            .agg(
                eigenvalue_mean=("eigenvalue", "mean"),
                effective_support_mean=("effective_support", "mean"),
            )
            .sort_values("mode")
        )

        avg["k_eff"] = np.sqrt(avg["mode"].astype(float))
        avg["k_eff2"] = avg["k_eff"] ** 2

        fit = avg[avg["mode"] >= 1].copy()
        if len(fit) >= 3:
            lr = linregress(fit["k_eff2"], fit["eigenvalue_mean"])
            a = float(lr.slope)
            e0 = float(lr.intercept)
            r = float(lr.rvalue)
            k1 = float(fit["k_eff"].iloc[0])
            vg1 = float(2.0 * a * k1)
        else:
            a = e0 = r = vg1 = float("nan")

        fit_rows.append(
            {
                "beta": beta,
                "alpha": params["alpha"],
                "beta_pot": params["beta_pot"],
                "dispersion_slope_a": a,
                "dispersion_intercept_E0": e0,
                "dispersion_fit_r": r,
                "group_velocity_mode1": vg1,
            }
        )

    out = pd.DataFrame(fit_rows).sort_values("beta")
    out.to_csv(results_dir / "dispersion_summary.csv", index=False)
    return out