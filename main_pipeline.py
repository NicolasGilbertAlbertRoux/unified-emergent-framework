from __future__ import annotations

from pathlib import Path
import yaml

from preprocessing.build_filament_graph_inputs import ensure_filament_graph_inputs
from preprocessing.build_collective_density_inputs import ensure_collective_density_inputs

from physics.schrodinger import run_schrodinger_dynamics
from physics.dispersion import run_dispersion_fit
from physics.dirac import run_pre_dirac
from physics.maxwell import run_pre_maxwell_transverse
from physics.einstein import run_einstein_source_metric


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def require_raw_maps(raw_map_dir: Path, betas: list[float], seeds: list[int]) -> None:
    if not raw_map_dir.exists():
        raise FileNotFoundError(
            f"Missing raw_map_dir: {raw_map_dir}\n"
            "Please place the upstream raw NPY maps in this directory before running the pipeline."
        )

    expected_patterns = [
        f"*beta{beta:.2f}_seed{seed}_hot_fx0_fz0_V12x12x12x12*.npy"
        for beta in betas
        for seed in seeds
    ]

    missing = []
    for pattern in expected_patterns:
        matches = list(raw_map_dir.glob(pattern))
        if not matches:
            missing.append(pattern)

    if missing:
        raise FileNotFoundError(
            "Some raw NPY maps are missing in data/raw_maps.\n"
            "Missing patterns:\n- "
            + "\n- ".join(missing)
            + "\n\nPlease copy or generate the required upstream raw maps before running the pipeline."
        )


def main() -> None:
    cfg = load_config()

    betas = cfg["betas"]
    seeds = cfg["seeds"]

    raw_map_dir = Path(cfg["paths"]["raw_map_dir"])
    filament_graph_dir = Path(cfg["paths"]["filament_graph_dir"])
    collective_density_dir = Path(cfg["paths"]["collective_density_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    require_raw_maps(raw_map_dir, betas, seeds)

    ensure_filament_graph_inputs(
        raw_map_dir=raw_map_dir,
        out_dir=filament_graph_dir,
        betas=betas,
        seeds=seeds,
        smooth_sigma=cfg["preprocessing"]["filament"]["smooth_sigma"],
        crit_percentile=cfg["preprocessing"]["filament"]["crit_percentile"],
        force_rebuild=cfg["preprocessing"]["filament"]["force_rebuild"],
    )

    ensure_collective_density_inputs(
        graph_dir=filament_graph_dir,
        out_dir=collective_density_dir,
        betas=betas,
        seeds=seeds,
        shape=tuple(cfg["preprocessing"]["collective_density"]["shape"]),
        top_k_hubs=cfg["preprocessing"]["collective_density"]["top_k_hubs"],
        max_pairs=cfg["preprocessing"]["collective_density"]["max_pairs"],
        save_images=cfg["preprocessing"]["collective_density"]["save_images"],
        force_rebuild=cfg["preprocessing"]["collective_density"]["force_rebuild"],
    )

    print("[RUN] Schrödinger sector")
    run_schrodinger_dynamics(
        detail_dir=filament_graph_dir,
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
        detail_dir=filament_graph_dir,
        results_dir=results_dir / "dirac",
        betas=betas,
        seeds=seeds,
    )

    print("[RUN] Pre-Maxwell sector")
    run_pre_maxwell_transverse(
        detail_dir=filament_graph_dir,
        results_dir=results_dir / "maxwell",
        betas=betas,
        seeds=seeds,
    )

    print("[RUN] Pre-Einstein sector")
    run_einstein_source_metric(
        density_dir=collective_density_dir,
        results_dir=results_dir / "einstein",
        betas=betas,
        seeds=seeds,
        sigma=cfg["einstein"]["sigma"],
        alpha_metric=cfg["einstein"]["alpha_metric"],
        beta_metric=cfg["einstein"]["beta_metric"],
        gamma_source=cfg["einstein"]["gamma_source"],
        eta_source=cfg["einstein"]["eta_source"],
    )

    dispersion_modes_csv = results_dir / "schrodinger" / "skeleton_hamiltonian_modes.csv"
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