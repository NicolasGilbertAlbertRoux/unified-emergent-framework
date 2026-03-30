#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml


def load_graph(graph_dir: Path, beta: float, seed: int):
    tag = f"beta{beta:.2f}_seed{seed}"
    nodes_path = graph_dir / f"filament_nodes_{tag}.csv"
    edges_path = graph_dir / f"filament_edges_{tag}.csv"

    if not nodes_path.exists() or not edges_path.exists():
        return None, None, None

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    graph = nx.Graph()

    for _, r in nodes.iterrows():
        nid = int(r["node_id"])
        graph.add_node(
            nid,
            x=int(r["x"]),
            y=int(r["y"]),
            z=int(r["z"]),
            t=int(r["t"]),
            degree=int(r.get("degree", 0)),
            crit_value=float(r.get("crit_value", 0.0)),
        )

    for _, r in edges.iterrows():
        graph.add_edge(int(r["u_id"]), int(r["v_id"]))

    return graph, nodes, edges


def largest_component_subgraph(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph.copy()
    component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(component).copy()


def choose_hubs(graph: nx.Graph, top_k: int) -> list[int]:
    hubs = sorted(
        graph.nodes(),
        key=lambda n: (graph.degree(n), graph.nodes[n].get("crit_value", 0.0)),
        reverse=True,
    )
    return hubs[: min(top_k, len(hubs))]


def build_collective_density(
    graph: nx.Graph,
    hubs: list[int],
    max_pairs: int | None = None,
) -> tuple[dict[int, float], int]:
    node_counts = {n: 0.0 for n in graph.nodes()}

    pairs = list(itertools.combinations(hubs, 2))
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    used_pairs = 0
    for u, v in pairs:
        try:
            path = nx.shortest_path(graph, source=u, target=v)
        except nx.NetworkXNoPath:
            continue

        used_pairs += 1
        for n in path:
            node_counts[n] += 1.0

    return node_counts, used_pairs


def node_counts_to_grid(
    graph: nx.Graph,
    node_counts: dict[int, float],
    shape: tuple[int, int, int, int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=float)
    for n, w in node_counts.items():
        if w <= 0:
            continue
        x = graph.nodes[n]["x"]
        y = graph.nodes[n]["y"]
        z = graph.nodes[n]["z"]
        t = graph.nodes[n]["t"]
        grid[x, y, z, t] += w
    return grid


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    m = float(np.max(grid)) if grid.size else 0.0
    if m <= 0:
        return grid.copy()
    return grid / m


def save_projection_images(out_dir: Path, beta: float, seed: int, grid_n: np.ndarray) -> None:
    proj_xy = grid_n.sum(axis=(2, 3))
    proj_xz = grid_n.sum(axis=(1, 3))
    proj_yz = grid_n.sum(axis=(0, 3))
    slice_xy_t0 = grid_n[:, :, :, 0].sum(axis=2)

    items = [
        ("proj_xy", proj_xy),
        ("proj_xz", proj_xz),
        ("proj_yz", proj_yz),
        ("slice_xy_t0", slice_xy_t0),
    ]

    for name, arr in items:
        plt.figure(figsize=(5, 4))
        plt.imshow(arr, origin="lower")
        plt.colorbar(label="normalized collective occupancy")
        plt.title(f"{name} | beta={beta:.2f}, seed={seed}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_beta{beta:.2f}_seed{seed}.png", dpi=160)
        plt.close()


def expected_density_path(out_dir: Path, beta: float, seed: int) -> Path:
    return out_dir / f"collective_density_beta{beta:.2f}_seed{seed}.npy"


def collective_density_inputs_exist(out_dir: str | Path, betas: list[float], seeds: list[int]) -> bool:
    out_dir = Path(out_dir)
    for beta in betas:
        for seed in seeds:
            if not expected_density_path(out_dir, beta, seed).exists():
                return False
    return True


def run_collective_density_preprocessing(
    graph_dir: str | Path,
    out_dir: str | Path,
    betas: list[float],
    seeds: list[int],
    shape: tuple[int, int, int, int] = (12, 12, 12, 12),
    top_k_hubs: int = 20,
    max_pairs: int = 120,
    save_images: bool = True,
) -> pd.DataFrame:
    graph_dir = Path(graph_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not graph_dir.exists():
        raise FileNotFoundError(f"Missing graph_dir: {graph_dir}")

    rows: list[dict] = []
    missing: list[tuple[float, int]] = []

    for beta in betas:
        for seed in seeds:
            graph, _, _ = load_graph(graph_dir, beta, seed)
            if graph is None:
                missing.append((beta, seed))
                print(f"[WARN] missing graph for beta={beta:.2f} seed={seed}")
                continue

            print(f"[LOAD] collective density beta={beta:.2f} seed={seed}")

            g_big = largest_component_subgraph(graph)
            hubs = choose_hubs(g_big, top_k_hubs)

            node_counts, used_pairs = build_collective_density(
                g_big,
                hubs,
                max_pairs=max_pairs,
            )

            grid = node_counts_to_grid(g_big, node_counts, shape=shape)
            grid_n = normalize_grid(grid)

            npy_path = expected_density_path(out_dir, beta, seed)
            np.save(npy_path, grid_n)

            if save_images:
                save_projection_images(out_dir, beta, seed, grid_n)

            nonzero = grid_n[grid_n > 0]

            rows.append(
                {
                    "beta": beta,
                    "seed": seed,
                    "largest_component_nodes": int(g_big.number_of_nodes()),
                    "largest_component_edges": int(g_big.number_of_edges()),
                    "n_hubs": int(len(hubs)),
                    "used_pairs": int(used_pairs),
                    "n_active_voxels": int(np.sum(grid_n > 0)),
                    "max_density": float(grid_n.max()) if np.any(grid_n > 0) else 0.0,
                    "mean_nonzero_density": float(nonzero.mean()) if len(nonzero) > 0 else 0.0,
                    "sum_density": float(grid_n.sum()),
                }
            )

            print(f"[OK] wrote {npy_path.name}")

    columns = [
        "beta",
        "seed",
        "largest_component_nodes",
        "largest_component_edges",
        "n_hubs",
        "used_pairs",
        "n_active_voxels",
        "max_density",
        "mean_nonzero_density",
        "sum_density",
    ]
    summary_df = pd.DataFrame(rows, columns=columns).sort_values(["beta", "seed"])
    summary_path = out_dir / "collective_density_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if not rows:
        missing_str = ", ".join(f"(beta={b}, seed={s})" for b, s in missing)
        raise FileNotFoundError(
            "No collective density inputs could be generated because no filament graph CSVs were found. "
            f"Checked combinations: {missing_str}. Verify graph_dir={graph_dir}"
        )

    print(f"[OK] wrote {summary_path}")
    print("[DONE] collective density preprocessing complete.")
    return summary_df


def ensure_collective_density_inputs(
    graph_dir: str | Path,
    out_dir: str | Path,
    betas: list[float],
    seeds: list[int],
    shape: tuple[int, int, int, int] = (12, 12, 12, 12),
    top_k_hubs: int = 20,
    max_pairs: int = 120,
    save_images: bool = True,
    force_rebuild: bool = False,
) -> None:
    if not force_rebuild and collective_density_inputs_exist(out_dir, betas, seeds):
        print("[SKIP] collective density preprocessing already complete.")
        return

    print("[RUN] collective density preprocessing")
    run_collective_density_preprocessing(
        graph_dir=graph_dir,
        out_dir=out_dir,
        betas=betas,
        seeds=seeds,
        shape=shape,
        top_k_hubs=top_k_hubs,
        max_pairs=max_pairs,
        save_images=save_images,
    )


if __name__ == "__main__":
    with Path("configs/default.yaml").open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    ensure_collective_density_inputs(
        graph_dir=cfg["paths"]["filament_graph_dir"],
        out_dir=cfg["paths"]["collective_density_dir"],
        betas=cfg["betas"],
        seeds=cfg["seeds"],
        shape=tuple(cfg["preprocessing"]["collective_density"]["shape"]),
        top_k_hubs=cfg["preprocessing"]["collective_density"]["top_k_hubs"],
        max_pairs=cfg["preprocessing"]["collective_density"]["max_pairs"],
        save_images=cfg["preprocessing"]["collective_density"]["save_images"],
        force_rebuild=cfg["preprocessing"]["collective_density"]["force_rebuild"],
    )
