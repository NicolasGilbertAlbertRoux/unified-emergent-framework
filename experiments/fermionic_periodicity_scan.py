#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.io import load_nodes_edges
from core.graph import build_incidence
from core.operators import build_dirac_like
from core.utils import effective_support, shannon_entropy


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_initial_state_from_low_modes(
    evals: np.ndarray,
    evecs: np.ndarray,
    zero_tol: float = 1e-8,
    n_pairs: int = 3,
) -> np.ndarray:
    """
    Build an initial state from the lowest nonzero paired modes ±E.
    """
    pos_idx = np.where(evals > zero_tol)[0]
    neg_idx = np.where(evals < -zero_tol)[0]

    pos_vals = evals[pos_idx]
    neg_vals = np.abs(evals[neg_idx])

    if len(pos_vals) == 0 or len(neg_vals) == 0:
        # fallback: smallest absolute nonzero mode
        nz = np.where(np.abs(evals) > zero_tol)[0]
        if len(nz) == 0:
            psi0 = np.zeros(len(evals), dtype=np.complex128)
            psi0[0] = 1.0
            return psi0
        idx0 = nz[np.argmin(np.abs(evals[nz]))]
        psi0 = evecs[:, idx0].astype(np.complex128)
        psi0 /= np.linalg.norm(psi0)
        return psi0

    chosen = []
    used_neg = set()

    for ip in pos_idx[np.argsort(pos_vals)]:
        target = abs(evals[ip])
        best_j = None
        best_err = None
        for jn in neg_idx:
            if jn in used_neg:
                continue
            err = abs(abs(evals[jn]) - target)
            if best_err is None or err < best_err:
                best_err = err
                best_j = jn
        if best_j is not None:
            chosen.append((best_j, ip))
            used_neg.add(best_j)
        if len(chosen) >= n_pairs:
            break

    if not chosen:
        nz = np.where(np.abs(evals) > zero_tol)[0]
        idx0 = nz[np.argmin(np.abs(evals[nz]))]
        psi0 = evecs[:, idx0].astype(np.complex128)
        psi0 /= np.linalg.norm(psi0)
        return psi0

    psi0 = np.zeros(evecs.shape[0], dtype=np.complex128)
    for jn, ip in chosen:
        psi0 += evecs[:, jn] + evecs[:, ip]

    norm = np.linalg.norm(psi0)
    if norm <= 0:
        psi0 = evecs[:, chosen[0][1]].astype(np.complex128)
        norm = np.linalg.norm(psi0)

    psi0 /= norm
    return psi0


def time_evolve_dirac(
    evals: np.ndarray,
    evecs: np.ndarray,
    psi0: np.ndarray,
    t_max: float,
    n_steps: int,
) -> pd.DataFrame:
    coeffs = evecs.conj().T @ psi0
    times = np.linspace(0.0, t_max, n_steps + 1)

    rows = []
    for t in times:
        phase = np.exp(-1j * evals * t)
        psi_t = evecs @ (phase * coeffs)

        overlap = np.vdot(psi0, psi_t)
        prob = np.abs(psi_t) ** 2

        rows.append(
            {
                "t": float(t),
                "autocorr": float(np.abs(overlap) ** 2),
                "effective_support": float(effective_support(psi_t.real) if np.allclose(psi_t.imag, 0) else 1.0 / np.sum((prob / prob.sum()) ** 2) if prob.sum() > 0 else np.nan),
                "entropy": float(shannon_entropy(psi_t)),
                "max_prob": float(np.max(prob)),
            }
        )

    return pd.DataFrame(rows)


def dominant_frequency(times: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """
    Returns dominant frequency and relative peak strength.
    """
    if len(times) < 8:
        return float("nan"), float("nan")

    dt = float(times[1] - times[0])
    x = values - np.mean(values)

    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=dt)

    if len(freqs) <= 1:
        return float("nan"), float("nan")

    amps = np.abs(fft)
    amps[0] = 0.0

    idx = int(np.argmax(amps))
    peak_freq = float(freqs[idx])
    peak_amp = float(amps[idx])
    total_amp = float(np.sum(amps)) + 1e-12
    rel_strength = peak_amp / total_amp

    return peak_freq, rel_strength


def estimate_recurrence_quality(values: np.ndarray) -> float:
    """
    Simple periodicity quality indicator from normalized autocorrelation amplitude variation.
    """
    if len(values) < 8:
        return float("nan")
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax <= 0:
        return float("nan")
    return (vmax - vmin) / vmax


def run_periodicity_scan(
    detail_dir: str | Path,
    results_dir: str | Path,
    betas: list[float],
    seeds: list[int],
    zero_tol: float = 1e-8,
    n_pairs: int = 3,
    t_max: float = 200.0,
    n_steps: int = 2000,
) -> pd.DataFrame:
    detail_dir = Path(detail_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = results_dir / "raw_timeseries"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []

    for beta in betas:
        for seed in seeds:
            nodes, edges = load_nodes_edges(detail_dir, beta, seed)
            if nodes is None or edges is None:
                missing.append((beta, seed))
                continue

            incidence, edge_list, node_ids = build_incidence(nodes, edges)
            D = build_dirac_like(incidence)

            evals, evecs = np.linalg.eigh(D)
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]

            psi0 = build_initial_state_from_low_modes(
                evals=evals,
                evecs=evecs,
                zero_tol=zero_tol,
                n_pairs=n_pairs,
            )

            ts = time_evolve_dirac(
                evals=evals,
                evecs=evecs,
                psi0=psi0,
                t_max=t_max,
                n_steps=n_steps,
            )
            ts["beta"] = beta
            ts["seed"] = seed

            raw_path = raw_dir / f"fermionic_periodicity_beta{beta:.2f}_seed{seed}.csv"
            ts.to_csv(raw_path, index=False)

            peak_freq, rel_strength = dominant_frequency(
                ts["t"].to_numpy(dtype=float),
                ts["autocorr"].to_numpy(dtype=float),
            )
            recurrence_quality = estimate_recurrence_quality(
                ts["autocorr"].to_numpy(dtype=float)
            )

            # low-mode spacing diagnostic
            nz = np.sort(np.abs(evals[np.abs(evals) > zero_tol]))
            first_gap = float(nz[1] - nz[0]) if len(nz) >= 2 else float("nan")
            first_mode = float(nz[0]) if len(nz) >= 1 else float("nan")

            rows.append(
                {
                    "beta": beta,
                    "seed": seed,
                    "dirac_dim": int(D.shape[0]),
                    "n_nodes": int(len(node_ids)),
                    "n_edges": int(len(edge_list)),
                    "lowest_nonzero_mode": first_mode,
                    "first_gap": first_gap,
                    "dominant_frequency": peak_freq,
                    "relative_peak_strength": rel_strength,
                    "recurrence_quality": recurrence_quality,
                    "autocorr_mean": float(ts["autocorr"].mean()),
                    "autocorr_std": float(ts["autocorr"].std()),
                    "support_mean": float(ts["effective_support"].mean()),
                    "entropy_mean": float(ts["entropy"].mean()),
                }
            )

            # figure
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

            axes[0].plot(ts["t"], ts["autocorr"])
            axes[0].set_title(f"Autocorrelation | beta={beta:.2f}, seed={seed}")
            axes[0].set_xlabel("t")
            axes[0].set_ylabel("C(t)")

            x = ts["autocorr"].to_numpy(dtype=float) - float(ts["autocorr"].mean())
            dt = float(ts["t"].iloc[1] - ts["t"].iloc[0]) if len(ts) > 1 else 1.0
            freqs = np.fft.rfftfreq(len(x), d=dt)
            amps = np.abs(np.fft.rfft(x))
            if len(amps) > 0:
                amps[0] = 0.0

            axes[1].plot(freqs, amps)
            axes[1].set_title("Frequency spectrum of C(t)")
            axes[1].set_xlabel("frequency")
            axes[1].set_ylabel("amplitude")

            fig.tight_layout()
            fig.savefig(fig_dir / f"fermionic_periodicity_beta{beta:.2f}_seed{seed}.png", dpi=180)
            plt.close(fig)

            print(f"[OK] fermionic periodicity beta={beta:.2f} seed={seed}")

    columns = [
        "beta",
        "seed",
        "dirac_dim",
        "n_nodes",
        "n_edges",
        "lowest_nonzero_mode",
        "first_gap",
        "dominant_frequency",
        "relative_peak_strength",
        "recurrence_quality",
        "autocorr_mean",
        "autocorr_std",
        "support_mean",
        "entropy_mean",
    ]

    if not rows:
        missing_str = ", ".join(f"(beta={b}, seed={s})" for b, s in missing)
        raise FileNotFoundError(
            "No inputs found for the fermionic periodicity scan. "
            f"Checked combinations: {missing_str}"
        )

    out = pd.DataFrame(rows, columns=columns).sort_values(["beta", "seed"])
    out.to_csv(results_dir / "fermionic_periodicity_summary.csv", index=False)

    return out


if __name__ == "__main__":
    cfg = load_config()
    run_periodicity_scan(
        detail_dir=cfg["paths"]["filament_graph_dir"],
        results_dir="results/fermionic_periodicity",
        betas=cfg["betas"],
        seeds=cfg["seeds"],
        zero_tol=1e-8,
        n_pairs=3,
        t_max=200.0,
        n_steps=2000,
    )