from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd

from .config import GeneratorConfig
from .mechanisms import Mechanism, _sample_noise, _compute_node_value, _resolve_node_mechanisms

def generate_samples(
    dag: np.ndarray,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    n_samples: Optional[int] = None,
    intervention_node: Optional[int] = None,
    intervention_value: float = 0.0,
    weight_matrix: Optional[np.ndarray] = None,
    root_means: Optional[Dict[int, float]] = None,
    root_scales: Optional[Dict[int, float]] = None,
    target_node: Optional[int] = None,
    target_noise_scale: Optional[float] = None,
    node_forms: Optional[List[Mechanism]] = None
) -> Tuple[pd.DataFrame, np.ndarray, List[Mechanism]]:
    """
    Generate samples following the causal ordering of the DAG.

    Parameters
    ----------
    dag               : adjacency matrix (n x n), dag[i,j]=1 means i -> j
    cfg               : generator config
    rng               : numpy random generator
    n_samples         : override cfg.n_samples if provided
    intervention_node : apply do(X_i = intervention_value); cuts parent edges to i
    intervention_value: fixed value for the intervened node
    weight_matrix     : (n x n) edge weights; generated randomly if None

    Returns
    -------
    df            : DataFrame with columns X0 … X{n-1}
    weight_matrix : the weight matrix used (for reproducibility)
    """
    n = dag.shape[0]
    ns = n_samples if n_samples is not None else cfg.n_samples

    # Draw fixed edge weights unless a shared weight matrix is supplied.
    if weight_matrix is None:
        weight_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if dag[i, j]:
                    n_parents = dag[:, j].sum()
                    if cfg.equal_contributions:
                        weight_matrix[i, j] = 1.0 / n_parents
                    else:
                        w = rng.uniform(cfg.weight_min, cfg.weight_max)
                        if cfg.allow_negative_weights:
                            w *= rng.choice([-1, 1])
                        weight_matrix[i, j] = w

    # Topological order
    G = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    topo = list(nx.topological_sort(G))

    data = np.zeros((ns, n))
    if node_forms is None:
        node_forms = _resolve_node_mechanisms(cfg, n, rng)

    for s in range(ns):
        x = np.zeros(n)
        for node in topo:
            if intervention_node is not None and node == intervention_node:
                # do(X_node = value): replace this node's structural equation.
                # Its incoming causal effects are therefore ignored, while its
                # outgoing effects on descendants must remain active.
                x[node] = intervention_value
                continue

            parents = np.where(dag[:, node] == 1)[0]

            if len(parents) == 0:
                # root node — use per-environment distribution if provided
                n_mean  = root_means[node]  if root_means  and node in root_means  else cfg.root_noise_mean
                n_scale = root_scales[node] if root_scales and node in root_scales else cfg.root_noise_scale
                noise_args = (
                    (1, cfg.root_noise_dist, n_mean, n_scale, rng)
                    if cfg.root_noise_sampler is None
                    else (1, cfg.root_noise_dist, n_mean, n_scale, rng,
                          cfg.root_noise_sampler)
                )
                x[node] = _sample_noise(*noise_args)[0]
            else:
                w = weight_matrix[parents, node]
                val = _compute_node_value(x[parents], w, node_forms[node])
                # use target-specific noise scale if this is the target node
                n_scale = target_noise_scale if (target_node is not None and node == target_node and target_noise_scale is not None) else cfg.node_noise_scale
                noise_args = (
                    (1, cfg.node_noise_dist, cfg.node_noise_mean,
                     n_scale, rng)
                    if cfg.node_noise_sampler is None
                    else (1, cfg.node_noise_dist, cfg.node_noise_mean,
                          n_scale, rng, cfg.node_noise_sampler)
                )
                noise = _sample_noise(*noise_args)[0]
                x[node] = val + noise
        data[s] = x

    cols = [f"X{i}" for i in range(n)]
    df = pd.DataFrame(data, columns=cols)
    return df, weight_matrix, node_forms


