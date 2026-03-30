#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from core.io import load_nodes_edges
from core.graph import build_incidence
from core.operators import build_dirac_like


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def pair_modes(evals: np.ndarray, zero_tol: float = 1e-8, max_pairs: int = 12):
    pos_idx = np.where(evals > zero_tol)[0]
    neg_idx = np.where(evals < -zero_tol)[0]

    pos_sorted = pos_idx[np.argsort(evals[pos_idx])]
    neg_sorted = neg_idx[np.argsort(np.abs(evals[neg_idx]))]

    used_neg = set()
    pairs = []

    for ip in pos_sorted:
        target = abs(evals[ip])
        best_j = None
        best_err = None

        for jn in neg_sorted:
            if jn in used_neg:
                continue
            err = abs(abs(evals[jn]) - target)
            if best_err is None or err < best_err:
                best_err = err
                best_j = jn

        if best_j is not None:
            used_neg.add(best_j)
            pairs.append((best_j, ip, float(abs(evals[ip])), float(best_err)))

        if len(pairs) >= max_pairs:
            break

    return pairs


def evolve_pair(
    evals: np.ndarray,
    evecs: np.ndarray,
    jn: int,
    ip: int,
    t_max: float = 200.0,
    n_steps: int = 4000,
):
    psi0 = (evecs[:, jn] + evecs[:, ip]).astype(np.complex128)
    psi0 /= np.linalg.norm(psi0)

    coeffs = evecs.conj().T @ psi0
    times = np.linspace(0.0, t_max, n_steps + 1)

    autocorr = []
    for t in times:
        phase = np.exp(-1j * evals * t)
        psi_t = evecs @ (phase * coeffs)
        overlap = np.vdot(psi0, psi_t)
        autocorr.append(float(np.abs(overlap) ** 2))

    return times, np.asarray(autocorr, dtype=float)


def dominant_frequency(times: np.ndarray, values: np.ndarray):
    if len(times) < 8:
        return float("nan"), float("nan")

    dt = float(times[1] - times[0])
    x = values - np.mean(values)

    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=dt)
    amps = np.abs(fft)

    if len(amps) > 0:
        amps[0] = 0.0

    idx = int(np.argmax(amps))
    peak_freq = float(freqs[idx])
    rel_strength = float(amps[idx] / (np.sum(amps) + 1e-12))

    return peak_freq, rel_strength


def recurrence_quality(values: np.ndarray) -> float:
    vmax = float(np.max(values))
    vmin = float(np.min(values))
    if vmax <= 0:
        return float("nan")
    return (vmax - vmin) / vmax


def run_single_pair_scan(
    detail_dir: str | Path,
    results_dir: str | Path,
    betas: list[float],
    seeds: list[int],
    zero_tol: float = 1e-8,
    max_pairs: int = 10,
    t_max: float = 200.0,
    n_steps: int = 4000,
) -> pd.DataFrame:
    detail_dir = Path(detail_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = results_dir / "raw_timeseries"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for beta in betas:
        for seed in seeds:
            nodes, edges = load_nodes_edges(detail_dir, beta, seed)
            if nodes is None or edges is None:
                print(f"[WARN] missing input for beta={beta:.2f}, seed={seed}")
                continue

            incidence, edge_list, node_ids = build_incidence(nodes, edges)
            D = build_dirac_like(incidence)

            evals, evecs = np.linalg.eigh(D)
            idx = np.argsort(evals)
            evals = evals[idx]
            evecs = evecs[:, idx]

            pairs = pair_modes(evals, zero_tol=zero_tol, max_pairs=max_pairs)
            if not pairs:
                print(f"[WARN] no valid pairs for beta={beta:.2f}, seed={seed}")
                continue

            fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 3 * len(pairs)))
            if len(pairs) == 1:
                axes = np.array([axes])

            for k, (jn, ip, energy_abs, pair_err) in enumerate(pairs, start=1):
                times, autocorr = evolve_pair(
                    evals=evals,
                    evecs=evecs,
                    jn=jn,
                    ip=ip,
                    t_max=t_max,
                    n_steps=n_steps,
                )

                peak_freq, rel_strength = dominant_frequency(times, autocorr)
                rq = recurrence_quality(autocorr)

                expected_pair_freq = float(abs(evals[ip] - evals[jn]) / (2.0 * np.pi))

                rows.append(
                    {
                        "beta": beta,
                        "seed": seed,
                        "pair_index": k,
                        "neg_eval": float(evals[jn]),
                        "pos_eval": float(evals[ip]),
                        "abs_energy": energy_abs,
                        "pairing_error": pair_err,
                        "dominant_frequency": peak_freq,
                        "expected_pair_frequency": expected_pair_freq,
                        "frequency_ratio": float(peak_freq / expected_pair_freq) if expected_pair_freq > 0 else float("nan"),
                        "relative_peak_strength": rel_strength,
                        "recurrence_quality": rq,
                        "dirac_dim": int(D.shape[0]),
                        "n_nodes": int(len(node_ids)),
                        "n_edges": int(len(edge_list)),
                    }
                )

                ts = pd.DataFrame(
                    {
                        "t": times,
                        "autocorr": autocorr,
                        "beta": beta,
                        "seed": seed,
                        "pair_index": k,
                    }
                )
                ts.to_csv(
                    raw_dir / f"single_pair_beta{beta:.2f}_seed{seed}_pair{k}.csv",
                    index=False,
                )

                # time-domain plot
                axes[k - 1, 0].plot(times, autocorr)
                axes[k - 1, 0].set_title(f"beta={beta:.2f}, seed={seed}, pair={k}")
                axes[k - 1, 0].set_xlabel("t")
                axes[k - 1, 0].set_ylabel("C(t)")

                # frequency-domain plot
                dt = float(times[1] - times[0])
                x = autocorr - np.mean(autocorr)
                freqs = np.fft.rfftfreq(len(x), d=dt)
                amps = np.abs(np.fft.rfft(x))
                if len(amps) > 0:
                    amps[0] = 0.0

                axes[k - 1, 1].plot(freqs, amps)
                axes[k - 1, 1].axvline(expected_pair_freq, linestyle="--")
                axes[k - 1, 1].set_xlabel("frequency")
                axes[k - 1, 1].set_ylabel("amplitude")
                axes[k - 1, 1].set_title(
                    f"peak={peak_freq:.4f}, expected={expected_pair_freq:.4f}"
                )

            fig.tight_layout()
            fig.savefig(fig_dir / f"single_pair_scan_beta{beta:.2f}_seed{seed}.png", dpi=180)
            plt.close(fig)

            print(f"[OK] single-pair scan beta={beta:.2f} seed={seed}")

    if not rows:
        raise RuntimeError("No results produced in single-pair Dirac scan.")

    out = pd.DataFrame(rows).sort_values(["beta", "seed", "pair_index"])
    out.to_csv(results_dir / "single_pair_dirac_summary.csv", index=False)

    # aggregate summary
    agg = (
        out.groupby(["beta", "pair_index"], as_index=False)
        .agg(
            dominant_frequency_mean=("dominant_frequency", "mean"),
            dominant_frequency_std=("dominant_frequency", "std"),
            expected_pair_frequency_mean=("expected_pair_frequency", "mean"),
            frequency_ratio_mean=("frequency_ratio", "mean"),
            frequency_ratio_std=("frequency_ratio", "std"),
            relative_peak_strength_mean=("relative_peak_strength", "mean"),
            recurrence_quality_mean=("recurrence_quality", "mean"),
        )
        .sort_values(["beta", "pair_index"])
    )
    agg.to_csv(results_dir / "single_pair_dirac_aggregate.csv", index=False)

    return out


if __name__ == "__main__":
    cfg = load_config()
    run_single_pair_scan(
        detail_dir=cfg["paths"]["filament_graph_dir"],
        results_dir="results/single_pair_dirac_scan",
        betas=cfg["betas"],
        seeds=cfg["seeds"],
        zero_tol=1e-8,
        max_pairs=8,
        t_max=200.0,
        n_steps=4000,
    )
