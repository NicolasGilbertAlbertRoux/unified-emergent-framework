from __future__ import annotations

from pathlib import Path

import yaml

from physics.schrodinger import run_schrodinger_dynamics
from physics.dispersion import run_dispersion_fit
from physics.dirac import run_pre_dirac
from physics.maxwell import run_pre_maxwell_transverse
from physics.einstein import run_einstein_source_metric


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    cfg = load_config()

    betas = cfg["betas"]
    seeds = cfg["seeds"]
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    detail_dir = cfg["paths"]["skeleton_detail_dir"]
    density_dir = cfg["paths"]["mantle_density_dir"]

    print("[RUN] Schrödinger sector")
    run_schrodinger_dynamics(
        detail_dir=detail_dir,
        results_dir=results_dir / "schrodinger",
        betas=betas,
        seeds=seeds,
        alpha=cfg["schrodinger"]["alpha"],
        beta_pot=cfg["schrodinger"]["beta_pot"],
        dt=cfg["schrodinger"]["dt"],
        n_steps=cfg["schrodinger"]["n_steps"],
    )

    print("[RUN] Pre-Dirac sector")
    run_pre_dirac(
        detail_dir=detail_dir,
        results_dir=results_dir / "dirac",
        betas=betas,
        seeds=seeds,
    )

    print("[RUN] Pre-Maxwell sector")
    run_pre_maxwell_transverse(
        detail_dir=detail_dir,
        results_dir=results_dir / "maxwell",
        betas=betas,
        seeds=seeds,
    )

    print("[RUN] Pre-Einstein sector")
    run_einstein_source_metric(
        density_dir=density_dir,
        results_dir=results_dir / "einstein",
        betas=betas,
        seeds=seeds,
        sigma=cfg["einstein"]["sigma"],
        alpha_metric=cfg["einstein"]["alpha_metric"],
        beta_metric=cfg["einstein"]["beta_metric"],
        gamma_source=cfg["einstein"]["gamma_source"],
        eta_source=cfg["einstein"]["eta_source"],
    )

    dispersion_modes_csv = (
        results_dir / "hamiltonian" / "skeleton_hamiltonian_modes.csv"
    )
    if dispersion_modes_csv.exists():
        print("[RUN] Dispersion sector")
        run_dispersion_fit(
            modes_csv=dispersion_modes_csv,
            results_dir=results_dir / "dispersion",
            best_params=cfg["dispersion"]["best"],
        )
    else:
        print("[SKIP] Dispersion sector (missing Hamiltonian modes CSV)")

    print("[DONE] Unified pipeline complete.")


if __name__ == "__main__":
    main()