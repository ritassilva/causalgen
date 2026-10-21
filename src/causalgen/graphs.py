from __future__ import annotations

from typing import List
import networkx as nx
import numpy as np

from .config import GeneratorConfig
from .mechanisms import DAGMethod

def _erdos_renyi(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    adj = (rng.random((n, n)) < p).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def _barabasi_albert(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    m = max(1, int(m))
    G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(1, 2**31)))
    return nx.to_numpy_array(G, dtype=int)


def _watts_strogatz(n: int, k: int, p: float, rng: np.random.Generator) -> np.ndarray:
    G = nx.watts_strogatz_graph(n, k, p, seed=int(rng.integers(1, 2**31)))
    return nx.to_numpy_array(G, dtype=int)


def _chain(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


def _binary_tree(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=int)
    for i in range(1, n):
        parent = (i - 1) // 2
        adj[parent, i] = 1
        adj[i, parent] = 1
    return adj

def _modular_biology_dag(
    n: int,
    n_modules: int,
    p_intra: float,
    p_inter: float,
    n_hubs_per_module: int,
    p_hub: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Biology-inspired modular DAG.

    - Modules mimic biological pathways.
    - Dense intra-module edges mimic local pathway regulation.
    - Sparse inter-module edges mimic pathway crosstalk.
    - Hub nodes mimic transcription factors / master regulators.

    Returns a DAG directly.
    """

    if n_modules > n:
        raise ValueError("n_modules cannot be larger than n_nodes.")

    modules = [list(m) for m in np.array_split(np.arange(n), n_modules)]

    dag = np.zeros((n, n), dtype=int)

    # Global topological order: module-ordered, shuffled inside each module
    topo_order = []
    for module_nodes in modules:
        shuffled = module_nodes.copy()
        rng.shuffle(shuffled)
        topo_order.extend(shuffled)

    rank = np.empty(n, dtype=int)
    rank[topo_order] = np.arange(n)

    # 1. Dense intra-module regulation
    for module_nodes in modules:
        for i in module_nodes:
            for j in module_nodes:
                if rank[i] < rank[j] and rng.random() < p_intra:
                    dag[i, j] = 1

    # 2. Hub / master regulator edges
    for module_nodes in modules:
        sorted_nodes = sorted(module_nodes, key=lambda x: rank[x])
        hubs = sorted_nodes[:min(n_hubs_per_module, len(sorted_nodes))]

        for hub in hubs:
            for target in module_nodes:
                if rank[hub] < rank[target] and rng.random() < p_hub:
                    dag[hub, target] = 1

    # 3. Sparse inter-module crosstalk
    for m1 in range(n_modules):
        for m2 in range(m1 + 1, n_modules):
            for i in modules[m1]:
                for j in modules[m2]:
                    if rank[i] < rank[j] and rng.random() < p_inter:
                        dag[i, j] = 1

    return dag


def build_undirected(cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    n, t = cfg.n_nodes, cfg.graph_type
    if t == "erdos_renyi":
        return _erdos_renyi(n, cfg.edge_probability, rng)
    if t == "barabasi_albert":
        return _barabasi_albert(n, cfg.edges_per_new_node, rng)
    if t == "watts_strogatz":
        return _watts_strogatz(n, cfg.n_neighbors, cfg.rewiring_probability, rng)
    if t == "chain":
        return _chain(n)
    if t == "tree":
        return _binary_tree(n)
    raise ValueError(f"Unknown graph_type: {t!r}")


# ---------------------------------------------------------------------------
# DAG enforcement
# ---------------------------------------------------------------------------

def enforce_dag(adj: np.ndarray, method: DAGMethod, rng: np.random.Generator) -> np.ndarray:
    """Orient every undirected edge to make a DAG, using a node ordering."""
    n = adj.shape[0]
    order = np.arange(n) if method == "topological" else rng.permutation(n)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)

    dag = np.zeros_like(adj)
    for i in range(n):
        for j in range(n):
            if adj[i, j] and rank[i] < rank[j]:
                dag[i, j] = 1
    return dag


# ---------------------------------------------------------------------------
# Confounder assignment
# ---------------------------------------------------------------------------

def assign_confounders(
    dag: np.ndarray,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> List[int]:
    """
    Returns list of confounder node indices.
    A valid confounder must have >= 2 children in the DAG.
    """
    n = dag.shape[0]
    children_count = dag.sum(axis=1)          # out-degree = number of children
    eligible = [i for i in range(n) if children_count[i] >= 2]

    # Reserve manually selected nodes before selecting either group randomly.
    manual = set((cfg.latent_confounder_nodes or []) + (cfg.observed_confounder_nodes or []))
    invalid = manual - set(eligible)
    if invalid:
        raise ValueError(f"Confounder nodes {sorted(invalid)} need at least two children in the generated DAG.")
    available = [i for i in eligible if i not in manual]
    groups = []
    for kind in ("latent", "observed"):
        nodes = getattr(cfg, f"{kind}_confounder_nodes")
        count = getattr(cfg, f"n_{kind}_confounders")
        if nodes is None:
            if count > len(available):
                raise ValueError(f"Requested {count} {kind} confounders, but only {len(available)} eligible nodes remain. Increase graph density or reduce confounders.")
            nodes = rng.choice(available, size=count, replace=False).tolist()
            available = [i for i in available if i not in nodes]
        groups.extend(sorted(nodes))
    return groups


