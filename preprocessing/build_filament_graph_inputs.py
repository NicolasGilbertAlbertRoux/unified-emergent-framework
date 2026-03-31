#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter


AXIS_NAMES = ["x", "y", "z", "t"]


def periodic_diff(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.roll(arr, -1, axis=axis) - arr


def gradient_norm(arr: np.ndarray) -> np.ndarray:
    grads = [periodic_diff(arr, ax) for ax in range(arr.ndim)]
    return np.sqrt(sum(g * g for g in grads))


def build_critical_field(arr: np.ndarray, smooth_sigma: float) -> np.ndarray:
    grad = gradient_norm(arr)
    g_mean = float(np.mean(grad))
    g_std = float(np.std(grad))

    if g_std <= 0:
        z_grad = np.zeros_like(grad)
    else:
        z_grad = (grad - g_mean) / g_std

    crit = np.maximum(z_grad, 0.0) ** 2
    if smooth_sigma > 0:
        crit = gaussian_filter(crit, sigma=smooth_sigma)
    return crit


def iter_neighbors_pos(coord: tuple[int, ...], shape: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
    for ax in range(len(shape)):
        nxt = list(coord)
        nxt[ax] = (nxt[ax] + 1) % shape[ax]
        yield tuple(nxt)


def find_map_file(raw_map_dir: Path, beta: float, seed: int) -> Path | None:
    pattern = f"*beta{beta:.2f}_seed{seed}_hot_fx0_fz0_V12x12x12x12*.npy"
    files = sorted(raw_map_dir.glob(pattern))
    return files[0] if files else None


def expected_graph_paths(out_dir: Path, beta: float, seed: int) -> tuple[Path, Path]:
    tag = f"beta{beta:.2f}_seed{seed}"
    return (
        out_dir / f"filament_nodes_{tag}.csv",
        out_dir / f"filament_edges_{tag}.csv",
    )


def filament_graph_inputs_exist(out_dir: str | Path, betas: list[float], seeds: list[int]) -> bool:
    out_dir = Path(out_dir)
    for beta in betas:
        for seed in seeds:
            nodes_path, edges_path = expected_graph_paths(out_dir, beta, seed)
            if not nodes_path.exists() or not edges_path.exists():
                return False
    return True


def run_filament_graph_preprocessing(
    raw_map_dir: str | Path,
    out_dir: str | Path,
    betas: list[float],
    seeds: list[int],
    smooth_sigma: float = 1.0,
    crit_percentile: float = 99.0,
) -> pd.DataFrame:
    raw_map_dir = Path(raw_map_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_map_dir.exists():
        raise FileNotFoundError(f"Missing raw_map_dir: {raw_map_dir}")

    summary_rows: list[dict] = []
    missing: list[tuple[float, int]] = []

    for beta in betas:
        for seed in seeds:
            map_path = find_map_file(raw_map_dir, beta, seed)
            if map_path is None:
                missing.append((beta, seed))
                print(f"[WARN] missing raw map for beta={beta:.2f} seed={seed}")
                continue

            print(f"[LOAD] filament map beta={beta:.2f} seed={seed}: {map_path.name}")
            arr = np.load(map_path)

            if arr.ndim != 4:
                raise ValueError(
                    f"Expected a 4D map for beta={beta:.2f}, seed={seed}, got shape={arr.shape}"
                )

            crit = build_critical_field(arr, smooth_sigma=smooth_sigma)
            thr = float(np.percentile(crit, crit_percentile))
            mask = crit >= thr

            coords = np.argwhere(mask)
            coord_set = {tuple(map(int, c)) for c in coords}

            node_rows: list[dict] = []
            coord_to_id: dict[tuple[int, ...], int] = {}

            for node_id, c in enumerate(coords):
                coord = tuple(map(int, c))
                coord_to_id[coord] = node_id

                row = {
                    "node_id": node_id,
                    "beta": beta,
                    "seed": seed,
                    "crit_value": float(crit[coord]),
                }
                for i, axis_name in enumerate(AXIS_NAMES[: len(coord)]):
                    row[axis_name] = coord[i]
                node_rows.append(row)

            nodes_df = pd.DataFrame(node_rows)

            edge_rows: list[dict] = []
            for c in coords:
                u = tuple(map(int, c))
                uid = coord_to_id[u]

                for v in iter_neighbors_pos(u, mask.shape):
                    if v in coord_set:
                        vid = coord_to_id[v]
                        row = {
                            "beta": beta,
                            "seed": seed,
                            "u_id": uid,
                            "v_id": vid,
                        }
                        for i, axis_name in enumerate(AXIS_NAMES[: len(u)]):
                            row[f"u_{axis_name}"] = u[i]
                            row[f"v_{axis_name}"] = v[i]
                        edge_rows.append(row)

            edges_df = pd.DataFrame(edge_rows)

            deg = np.zeros(len(nodes_df), dtype=int)
            if not edges_df.empty:
                for _, r in edges_df.iterrows():
                    deg[int(r["u_id"])] += 1
                    deg[int(r["v_id"])] += 1

            if len(nodes_df) > 0:
                nodes_df["degree"] = deg
            else:
                nodes_df["degree"] = pd.Series(dtype=int)

            nodes_path, edges_path = expected_graph_paths(out_dir, beta, seed)
            nodes_df.to_csv(nodes_path, index=False)
            edges_df.to_csv(edges_path, index=False)

            summary_rows.append(
                {
                    "beta": beta,
                    "seed": seed,
                    "source_map": map_path.name,
                    "threshold": thr,
                    "n_nodes": int(len(nodes_df)),
                    "n_edges": int(len(edges_df)),
                    "mean_degree": float(np.mean(deg)) if len(deg) > 0 else 0.0,
                    "leaf_count": int(np.sum(deg == 1)),
                    "branch_count": int(np.sum(deg >= 3)),
                }
            )

            print(f"[OK] wrote {nodes_path.name}")
            print(f"[OK] wrote {edges_path.name}")

    columns = [
        "beta",
        "seed",
        "source_map",
        "threshold",
        "n_nodes",
        "n_edges",
        "mean_degree",
        "leaf_count",
        "branch_count",
    ]
    summary_df = pd.DataFrame(summary_rows, columns=columns).sort_values(["beta", "seed"])
    summary_path = out_dir / "filament_graph_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if not summary_rows:
        missing_str = ", ".join(f"(beta={b}, seed={s})" for b, s in missing)
        raise FileNotFoundError(
            "No filament graph inputs could be generated because no raw NPY maps were found. "
            f"Checked combinations: {missing_str}. Verify raw_map_dir={raw_map_dir}"
        )

    print(f"[OK] wrote {summary_path}")
    print("[DONE] filament graph preprocessing complete.")
    return summary_df


def ensure_filament_graph_inputs(
    raw_map_dir: str | Path,
    out_dir: str | Path,
    betas: list[float],
    seeds: list[int],
    smooth_sigma: float = 1.0,
    crit_percentile: float = 99.0,
    force_rebuild: bool = False,
) -> None:
    if not force_rebuild and filament_graph_inputs_exist(out_dir, betas, seeds):
        print("[SKIP] filament graph preprocessing already complete.")
        return

    print("[RUN] filament graph preprocessing")
    run_filament_graph_preprocessing(
        raw_map_dir=raw_map_dir,
        out_dir=out_dir,
        betas=betas,
        seeds=seeds,
        smooth_sigma=smooth_sigma,
        crit_percentile=crit_percentile,
    )


if __name__ == "__main__":
    with Path("configs/default.yaml").open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    ensure_filament_graph_inputs(
        raw_map_dir=cfg["paths"]["raw_map_dir"],
        out_dir=cfg["paths"]["filament_graph_dir"],
        betas=cfg["betas"],
        seeds=cfg["seeds"],
        smooth_sigma=cfg["preprocessing"]["filament"]["smooth_sigma"],
        crit_percentile=cfg["preprocessing"]["filament"]["crit_percentile"],
        force_rebuild=cfg["preprocessing"]["filament"]["force_rebuild"],
    )