from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.io import load_nodes_edges
from core.graph import build_adjacency
from core.operators import graph_laplacians, build_effective_hamiltonian
from core.utils import wave_effective_support, shannon_entropy, effective_support


def run_schrodinger_dynamics(
    detail_dir: str | Path,
    results_dir: str | Path,
    betas: list[float],
    seeds: list[int],
    alpha: float,
    beta_pot: float,
    dt: float,
    n_steps: int,
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    mode_rows: list[dict] = []
    missing: list[tuple[float, int]] = []

    for beta in betas:
        for seed in seeds:
            nodes, edges = load_nodes_edges(detail_dir, beta, seed)
            if nodes is None or edges is None:
                missing.append((beta, seed))
                continue

            adjacency, _ = build_adjacency(nodes, edges, use_edge_weight=False)
            if adjacency.shape[0] == 0:
                continue

            _, lap_norm, degree = graph_laplacians(adjacency)
            h_eff = build_effective_hamiltonian(lap_norm, degree, alpha, beta_pot)

            evals, evecs = np.linalg.eigh(h_eff)

            for mode, eig in enumerate(evals):
                mode_rows.append(
                    {
                        "beta": beta,
                        "seed": seed,
                        "alpha": alpha,
                        "beta_pot": beta_pot,
                        "mode": int(mode),
                        "eigenvalue": float(eig),
                        "effective_support": effective_support(evecs[:, mode]),
                    }
                )

            i0 = int(np.argmax(degree))
            psi0 = np.zeros(adjacency.shape[0], dtype=np.complex128)
            psi0[i0] = 1.0

            coeffs = evecs.conj().T @ psi0

            for step in range(n_steps + 1):
                t = step * dt
                phase = np.exp(-1j * evals * t)
                psi_t = evecs @ (phase * coeffs)

                rows.append(
                    {
                        "beta": beta,
                        "seed": seed,
                        "t": float(t),
                        "max_prob": float(np.max(np.abs(psi_t) ** 2)),
                        "effective_support": wave_effective_support(psi_t),
                        "entropy": shannon_entropy(psi_t),
                        "return_prob": float(np.abs(np.vdot(psi0, psi_t)) ** 2),
                    }
                )

    raw_columns = [
        "beta",
        "seed",
        "t",
        "max_prob",
        "effective_support",
        "entropy",
        "return_prob",
    ]

    if not rows:
        df = pd.DataFrame(columns=raw_columns)
        df.to_csv(results_dir / "schrodinger_dynamics_raw.csv", index=False)
        missing_str = ", ".join([f"(beta={b}, seed={s})" for b, s in missing])
        raise FileNotFoundError(
            "No Schrödinger graph inputs were found. "
            f"Checked combinations: {missing_str}. "
            "Verify configs/default.yaml and the graph CSV files."
        )

    df = pd.DataFrame(rows, columns=raw_columns).sort_values(["beta", "seed", "t"])
    df.to_csv(results_dir / "schrodinger_dynamics_raw.csv", index=False)

    if mode_rows:
        modes_df = pd.DataFrame(mode_rows).sort_values(["beta", "seed", "mode"])
        modes_df.to_csv(results_dir / "skeleton_hamiltonian_modes.csv", index=False)

    return df