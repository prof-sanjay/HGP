"""
Graph assembly, storage and the experimental assignments of graphs to horizon branches.

Graph convention everywhere: G[i, j] is the weight of the directed edge source j -> target i; zero diagonal.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def standardised_effect(psi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """a[i, j] = |Psi_h[i, j]| sigma_j / sigma_i, zero diagonal."""
    a = np.abs(psi) * (sigma[None, :] / sigma[:, None])
    np.fill_diagonal(a, 0.0)
    return a


def var_top_k(psi: np.ndarray, k: int) -> np.ndarray:
    """VAR-only reference graph: the k sources with the largest |Psi_h| per target, weighted by |Psi_h|."""
    a = np.abs(psi).astype(np.float64)
    np.fill_diagonal(a, -np.inf)
    k = min(k, len(a) - 1)
    top = np.argsort(-a, axis=1, kind="stable")[:, :k]
    graph = np.zeros_like(a)
    rows = np.repeat(np.arange(len(a)), k)
    graph[rows, top.ravel()] = a[rows, top.ravel()]
    return np.maximum(graph, 0.0)


def assemble(selected: np.ndarray, effect: np.ndarray, rho: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Weights of the selected edges: the standardised VAR effect where the VAR supports the edge, otherwise
    kappa * rho with one global kappa that puts the median added weight at the median VAR weight; then one global
    factor so the mean weight equals that of the VAR-only reference graph of the same horizon.
    """
    selected = selected.copy()
    np.fill_diagonal(selected, False)
    in_var = selected & (effect > 0)
    added = selected & ~in_var & (rho > 0)
    w = np.zeros(effect.shape, dtype=np.float64)
    w[in_var] = effect[in_var]
    kappa = np.median(effect[in_var]) / np.median(rho[added]) if in_var.any() and added.any() else 1.0
    w[added] = kappa * rho[added]
    factor = float(reference[reference > 0].mean() / w[w > 0].mean()) if (w > 0).any() else 1.0
    w *= factor
    degree = (w > 0).sum(axis=1)
    info = {"edges": int((w > 0).sum()), "var_supported": int(in_var.sum()), "added": int(added.sum()), "kappa": float(kappa),
            "scale_factor": factor, "sources_per_target": float(degree.mean()), "targets_without_sources": int((degree == 0).sum())}
    return w.astype(np.float32), info


def save_graph(path: Path, graph: np.ndarray, sensors: np.ndarray, horizon: int, train_end: int, **meta) -> None:
    np.savez_compressed(path, adjacency=graph.astype(np.float32), sensors=np.asarray(sensors, np.int64), horizon=horizon,
                        train_end=train_end, orientation="target_source", **meta)


def load_graph(path: Path, horizon: int | None = None) -> np.ndarray:
    with np.load(path) as z:
        if horizon is not None and int(z["horizon"]) != horizon:
            raise ValueError(f"{path}: graph is for horizon {int(z['horizon'])}, not {horizon}")
        g = np.asarray(z["adjacency"], dtype=np.float32)
    np.fill_diagonal(g, 0.0)
    return g


def row_normalise(graph: np.ndarray) -> np.ndarray:
    degree = graph.sum(axis=1, keepdims=True)
    return np.where(degree > 0, graph / np.maximum(degree, 1e-8), 0.0).astype(np.float32)


def to_sparse(graph: np.ndarray, device) -> torch.Tensor:
    rows, cols = np.nonzero(graph)
    idx = torch.as_tensor(np.stack([rows, cols]), dtype=torch.int64)
    return torch.sparse_coo_tensor(idx, torch.as_tensor(graph[rows, cols]), graph.shape, device=device).coalesce()


def road_reach(road: np.ndarray, hops: int) -> np.ndarray:
    A = (road != 0).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    reach, power = A.copy(), A.copy()
    for _ in range(hops - 1):
        power = ((power @ A) > 0).astype(np.float32)
        reach = np.maximum(reach, power)
    np.fill_diagonal(reach, 0.0)
    return reach


@dataclass
class Assignment:
    """The graph used in each horizon branch (None = no graph)."""
    name: str
    graphs: dict[int, np.ndarray | None]
    source: dict[int, str]

    @classmethod
    def build(cls, name: str, horizons: tuple[int, ...], graph_dir: Path, road: np.ndarray | None = None, seed: int = 0):
        """
        family      G_h in branch h
        g12_all     G_12 in every branch        g3_all   G_3 in every branch
        shuffled    a derangement of the family (seeded)
        road        road adjacency              road_power  road reach within ceil(h/3) hops
        none        no graph anywhere
        """
        graph_dir = Path(graph_dir)
        load = lambda h: load_graph(graph_dir / f"G_h{h}.npz", h)
        if name == "family":
            graphs, source = {h: load(h) for h in horizons}, {h: f"G_{h}" for h in horizons}
        elif name in ("g12_all", "g3_all"):
            fixed = 12 if name == "g12_all" else 3
            g = load(fixed)
            graphs, source = {h: g for h in horizons}, {h: f"G_{fixed}" for h in horizons}
        elif name == "shuffled":
            rng, n = np.random.default_rng(seed), len(horizons)
            perm = np.arange(n)
            while np.any(perm == np.arange(n)):
                perm = rng.permutation(n)
            loaded = {h: load(h) for h in horizons}
            graphs = {h: loaded[horizons[perm[k]]] for k, h in enumerate(horizons)}
            source = {h: f"G_{horizons[perm[k]]}" for k, h in enumerate(horizons)}
        elif name == "road":
            g = (road != 0).astype(np.float32)
            np.fill_diagonal(g, 0.0)
            graphs, source = {h: g for h in horizons}, {h: "road" for h in horizons}
        elif name == "road_power":
            hops = {h: int(np.ceil(h / 3)) for h in horizons}
            graphs, source = {h: road_reach(road, hops[h]) for h in horizons}, {h: f"road^{hops[h]}" for h in horizons}
        elif name == "none":
            graphs, source = {h: None for h in horizons}, {h: "none" for h in horizons}
        else:
            raise ValueError(f"unknown graph configuration {name!r}")
        return cls(name, graphs, source)

    def to_sparse(self, device) -> dict[int, torch.Tensor | None]:
        return {h: None if g is None else to_sparse(row_normalise(g), device) for h, g in self.graphs.items()}

    def summary(self) -> dict:
        return {h: {"source": self.source[h], "edges": 0 if g is None else int((g > 0).sum()),
                    "sources_per_target": 0.0 if g is None else float((g > 0).sum(1).mean())} for h, g in self.graphs.items()}
