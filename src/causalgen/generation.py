from __future__ import annotations

import random
import networkx as nx
import numpy as np
import pandas as pd

from .config import GeneratorConfig, EnvironmentConfig
from .dataset import CausalDataset
from .graphs import build_undirected, enforce_dag, assign_confounders, _modular_biology_dag
from .mechanisms import _resolve_node_mechanisms, _mechanism_name, _noise_sampler_metadata
from .sampling import generate_samples

def generate_environments(cfg: GeneratorConfig, env_cfg: EnvironmentConfig) -> CausalDataset:
    """
    Generate a multi-environment dataset where:
    - Root node distributions shift across environments (upstream shift)
    - Non-target mechanisms can shift across environments (soft interventions)
    - The mechanism of every node EXCEPT the target stays fixed in terms of
      which nodes cause it, but the target node's mechanism is fully fixed
      (same coefficients) across all environments
    - Target node is the last node in topological order
    """
    seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**31)

    all_obs    = []
    all_interv = []

    # Generate base DAG and confounders once — shared across all environments
    rng = np.random.default_rng(seed)
    # 1-2. Build DAG
    if cfg.graph_type == "modular_biology":
        dag = _modular_biology_dag(
            n=cfg.n_nodes,
            n_modules=cfg.n_modules,
            p_intra=cfg.p_intra,
            p_inter=cfg.p_inter,
            n_hubs_per_module=cfg.n_hubs_per_module,
            p_hub=cfg.p_hub,
            rng=rng,
        )
    else:
        adj = build_undirected(cfg, rng)
        dag = enforce_dag(adj, cfg.dag_method, rng)
    confounders = assign_confounders(dag, cfg, rng)

    # Target node = last in topological order
    G = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    topo = list(nx.topological_sort(G))
    target_node = topo[-1]

    # Weight matrix is fixed once and shared across all environments
    # so the target mechanism is truly invariant
    # Independent streams derived from the exact user seed, without offsets.
    streams = np.random.SeedSequence(seed).spawn(2 + env_cfg.n_envs)
    rng_weights = np.random.default_rng(streams[0])
    n = dag.shape[0]
    weight_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if dag[i, j]:
                n_parents = dag[:, j].sum()
                if cfg.equal_contributions:
                    weight_matrix[i, j] = 1.0 / n_parents
                else:
                    w = rng_weights.uniform(cfg.weight_min, cfg.weight_max)
                    if cfg.allow_negative_weights:
                        w *= rng_weights.choice([-1, 1])
                    weight_matrix[i, j] = w

    # Generate node forms once so they are consistent across environments.
    rng_forms = np.random.default_rng(streams[1])
    node_forms = _resolve_node_mechanisms(cfg, n, rng_forms)

    for env in range(env_cfg.n_envs):
        # Different sources of randomness for independent processes
        data_stream, upstream_stream, soft_stream = streams[2 + env].spawn(3)
        rng_env = np.random.default_rng(data_stream)
        rng_upstream = np.random.default_rng(upstream_stream)

        root_nodes  = [i for i in range(n) if dag[:, i].sum() == 0]
        root_means  = {}
        root_scales = {}
        for r in root_nodes:
            if env_cfg.upstream_shift:
                root_means[r]  = rng_upstream.uniform(-2.0, 2.0)
                root_scales[r] = rng_upstream.uniform(0.5, 2.0)
            else:
                root_means[r]  = cfg.root_noise_mean
                root_scales[r] = cfg.root_noise_scale

        # --- soft interventions: non-target mechanisms get shifted weights ---
        env_weight_matrix = weight_matrix.copy()
        if env_cfg.soft_interventions:
            rng_soft = np.random.default_rng(soft_stream)
            for j in range(n):
                if j == target_node:
                    continue   # target mechanism is never touched
                parents = np.where(dag[:, j] == 1)[0]
                for i in parents:
                    shift = rng_soft.uniform(-env_cfg.soft_intervention_scale, env_cfg.soft_intervention_scale)
                    env_weight_matrix[i, j] = weight_matrix[i, j] + shift

        # --- heteroskedastic noise on target ---
        target_noise_scale = (
            cfg.node_noise_scale * (1.0 + 0.25 * env)
            if env_cfg.heteroskedastic_noise
            else cfg.node_noise_scale
        )

        # --- generate samples for this environment ---
        env_cfg_copy = GeneratorConfig(
            **{**cfg.__dict__,
               "root_noise_mean":  cfg.root_noise_mean,
                "root_noise_scale": cfg.root_noise_scale,
                "node_noise_mean":  cfg.node_noise_mean,
                "node_noise_scale": cfg.node_noise_scale,
            }
        )
        obs_df, _, _ = generate_samples(
            dag, env_cfg_copy, rng_env,
            n_samples=env_cfg.n_samples_per_env,
            weight_matrix=env_weight_matrix,
            root_means=root_means,
            root_scales=root_scales,
            target_node=target_node,
            target_noise_scale=target_noise_scale,
            node_forms= node_forms
        )
        obs_df["env"] = env
        all_obs.append(obs_df)

        if cfg.intervention_node is not None:
            interv_df, _, _ = generate_samples(
                dag, env_cfg_copy, rng_env,
                n_samples=env_cfg.n_samples_per_env,
                weight_matrix=env_weight_matrix,
                root_means=root_means,
                root_scales=root_scales,
                target_node=target_node,
                target_noise_scale=target_noise_scale,
                intervention_node=cfg.intervention_node,
                intervention_value=cfg.intervention_value,
                node_forms= node_forms
            )
            interv_df["env"] = env
            all_interv.append(interv_df)

    obs_full    = pd.concat(all_obs, ignore_index=True)
    interv_full = pd.concat(all_interv, ignore_index=True) if all_interv else None

    # --- metadata ---
    edge_list = [
        {"source": f"X{i}", "target": f"X{j}"}
        for i in range(n) for j in range(n) if dag[i, j]
    ]
    metadata = {
        "seed": seed,
        "graph_type": cfg.graph_type,
        "n_nodes": n,
        "n_edges": int(dag.sum()),
        "edge_probability": cfg.edge_probability,
        "edges_per_new_node": cfg.edges_per_new_node,
        "rewiring_probability": cfg.rewiring_probability,
        "n_neighbors": cfg.n_neighbors if cfg.graph_type == "watts_strogatz" else None,
        "dag_method": cfg.dag_method,
        "mechanism": _mechanism_name(cfg.mechanism),
        "mechanism_pool": (
            [_mechanism_name(form) for form in cfg.mechanism_pool]
            if cfg.mechanism_pool is not None else None
        ),
        "root_noise_dist":  cfg.root_noise_dist,
        "root_noise_mean":  cfg.root_noise_mean,
        "root_noise_scale": cfg.root_noise_scale,
        "node_noise_dist":  cfg.node_noise_dist,
        "node_noise_mean":  cfg.node_noise_mean,
        "node_noise_scale": cfg.node_noise_scale,
        "root_noise_sampler": _noise_sampler_metadata(cfg.root_noise_sampler),
        "node_noise_sampler": _noise_sampler_metadata(cfg.node_noise_sampler),
        "node_forms": {
            f"X{i}": _mechanism_name(form) for i, form in enumerate(node_forms)
        },
        "n_envs": env_cfg.n_envs,
        "n_samples_per_env": env_cfg.n_samples_per_env,
        "total_samples": env_cfg.n_envs * env_cfg.n_samples_per_env,
        "target_node": f"X{target_node}",
        "upstream_shift": env_cfg.upstream_shift,
        "soft_interventions": env_cfg.soft_interventions,
        "heteroskedastic_noise": env_cfg.heteroskedastic_noise,
        "confounders":          [f"X{i}" for i in confounders],
        "latent_confounders":   [f"X{i}" for i in confounders[:cfg.n_latent_confounders]],
        "observed_confounders": [f"X{i}" for i in confounders[cfg.n_latent_confounders:]],
        "n_latent_confounders":  cfg.n_latent_confounders,
        "n_observed_confounders":cfg.n_observed_confounders,
        "edge_list": edge_list,
    }

    return CausalDataset(
        dag=dag,
        confounders=confounders,
        full_observational=obs_full,
        full_interventional=interv_full,
        weight_matrix=weight_matrix,
        config=cfg,
        metadata=metadata,
    )

def generate(cfg: GeneratorConfig) -> CausalDataset:
    """
    Run the full causal dataset generation pipeline.

    Parameters
    ----------
    cfg : GeneratorConfig

    Returns
    -------
    CausalDataset
    """
    seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**31)
    rng = np.random.default_rng(seed)

    # 1-2. Build DAG
    if cfg.graph_type == "modular_biology":
        dag = _modular_biology_dag(
            n=cfg.n_nodes,
            n_modules=cfg.n_modules,
            p_intra=cfg.p_intra,
            p_inter=cfg.p_inter,
            n_hubs_per_module=cfg.n_hubs_per_module,
            p_hub=cfg.p_hub,
            rng=rng,
        )
    else:
        adj = build_undirected(cfg, rng)
        dag = enforce_dag(adj, cfg.dag_method, rng)

    # 3. Assign confounders
    confounders = assign_confounders(dag, cfg, rng)

    # 4. Generate observational data
    obs_df, weight_matrix, node_forms = generate_samples(dag, cfg, rng)

    # 5. Generate interventional data (if requested)
    interv_df = None
    if cfg.intervention_node is not None:
        interv_df, _ , _ = generate_samples(
            dag, cfg, rng,
            intervention_node=cfg.intervention_node,
            intervention_value=cfg.intervention_value,
            weight_matrix=weight_matrix,
            node_forms= node_forms
        )

    # 6. Build metadata
    n = dag.shape[0]
    edge_list = [
        {"source": f"X{i}", "target": f"X{j}"}
        for i in range(n) for j in range(n) if dag[i, j]
    ]
    children = {
        f"X{i}": [f"X{j}" for j in range(n) if dag[i, j]]
        for i in range(n)
    }
    metadata = {
        "seed": seed,
        "graph_type": cfg.graph_type,
        "n_nodes": n,
        "n_edges": int(dag.sum()),
        "edge_probability": cfg.edge_probability,
        "edges_per_new_node": cfg.edges_per_new_node,
        "rewiring_probability": cfg.rewiring_probability,
        "n_neighbors": cfg.n_neighbors if cfg.graph_type == "watts_strogatz" else None,
        "dag_method": cfg.dag_method,
        "mechanism": _mechanism_name(cfg.mechanism),
        "mechanism_pool": (
            [_mechanism_name(form) for form in cfg.mechanism_pool]
            if cfg.mechanism_pool is not None else None
        ),
        "node_forms": (
            [_mechanism_name(form) for form in node_forms]
            if cfg.mechanism == "mixed" else None
        ),
        "root_noise_dist":  cfg.root_noise_dist,
        "root_noise_mean":  cfg.root_noise_mean,
        "root_noise_scale": cfg.root_noise_scale,
        "node_noise_dist":  cfg.node_noise_dist,
        "node_noise_mean":  cfg.node_noise_mean,
        "node_noise_scale": cfg.node_noise_scale,
        "root_noise_sampler": _noise_sampler_metadata(cfg.root_noise_sampler),
        "node_noise_sampler": _noise_sampler_metadata(cfg.node_noise_sampler),
        "node_mechanisms": (
            {f"X{i}": _mechanism_name(form) for i, form in cfg.node_mechanisms.items()}
            if cfg.node_mechanisms else None
        ),
        "n_samples": cfg.n_samples,
        "confounders": [f"X{i}" for i in confounders],
        "n_latent_confounders":   cfg.n_latent_confounders,
        "n_observed_confounders": cfg.n_observed_confounders,
        "latent_confounders":  [f"X{i}" for i in confounders[:cfg.n_latent_confounders]],
        "observed_confounders": [f"X{i}" for i in confounders[cfg.n_latent_confounders:]],
        "children": children,
        "edge_list": edge_list,
        "intervention": (
            {"node": f"X{cfg.intervention_node}", "value": cfg.intervention_value}
            if cfg.intervention_node is not None else None
        ),
    }

    return CausalDataset(
        dag=dag,
        confounders=confounders,
        full_observational=obs_df,
        full_interventional=interv_df,
        weight_matrix=weight_matrix,
        config=cfg,
        metadata=metadata,
    )
