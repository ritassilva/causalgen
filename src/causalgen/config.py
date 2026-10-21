from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence
import numpy as np

from .mechanisms import FuncForm, Mechanism, NoiseDist, NoiseSampler, GraphType, DAGMethod, _validate_mechanism

@dataclass
class GeneratorConfig:
    """Graph, mechanism, noise and observation settings for one causal model."""

    # Graph
    graph_type:   GraphType  = "erdos_renyi"
    n_nodes:      int        = 8
    edge_probability: float = 0.35
    edges_per_new_node: int = 2
    rewiring_probability: float = 0.35
    n_neighbors:         int        = 4      # initial number of nearest neighbours for WS
    dag_method:   DAGMethod  = "topological" # node-index order or seeded random order

    # Confounders
    observed_confounder_nodes: Optional[List[int]] = None
    latent_confounder_nodes: Optional[List[int]] = None
    n_latent_confounders: int = 0
    n_observed_confounders: int = 0

    # Modular biology graph parameters
    n_modules: int = 5
    p_intra: float = 0.25
    p_inter: float = 0.03
    n_hubs_per_module: int = 1
    p_hub: float = 0.5

    # Data generation
    mechanism: FuncForm  = "linear"
    node_mechanisms: Optional[Dict[int, Mechanism]] = None  # per-node overrides
    mechanism_pool: Optional[Sequence[Mechanism]] = None  # candidates used by "mixed"
    root_noise_dist:  NoiseDist = "gaussian"
    root_noise_mean:  float     = 0.0
    root_noise_scale: float     = 1.0
    root_noise_sampler: Optional[NoiseSampler] = None

    node_noise_dist:  NoiseDist = "gaussian"
    node_noise_mean:  float     = 0.0
    node_noise_scale: float     = 1.0
    node_noise_sampler: Optional[NoiseSampler] = None
    n_samples:       int       = 500

    # Edge weights
    weight_min:              float = 0.5
    weight_max:              float = 2.0
    allow_negative_weights:  bool  = True
    equal_contributions:     bool  = False  # if True, each parent contributes 1/n_parents

    # Intervention (do-calculus)
    intervention_node:  Optional[int]   = None   # None = no interventional dataset
    intervention_value: float           = 1.0

    # Reproducibility
    seed: Optional[int] = 42

    def __post_init__(self):
        def integer(name, minimum, maximum=None):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum or (maximum is not None and value > maximum):
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum or 'infinity'}.")
        integer("n_nodes", 3)
        integer("n_samples", 1)
        if self.seed is not None:
            integer("seed", 0)
        if self.graph_type not in {"erdos_renyi", "barabasi_albert", "watts_strogatz", "chain", "tree", "modular_biology"}:
            raise ValueError(f"Unknown graph_type: {self.graph_type!r}.")
        if self.dag_method not in {"topological", "random"}:
            raise ValueError("dag_method must be 'topological' or 'random'.")
        for name in ("edge_probability", "rewiring_probability", "p_intra", "p_inter", "p_hub"):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.graph_type == "barabasi_albert":
            integer("edges_per_new_node", 1, self.n_nodes - 1)
        if self.graph_type == "modular_biology":
            integer("n_modules", 1, self.n_nodes)
            integer("n_hubs_per_module", 0)
        for prefix in ("root", "node"):
            scale = getattr(self, f"{prefix}_noise_scale")
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"{prefix}_noise_scale must be finite and positive.")
            if not np.isfinite(getattr(self, f"{prefix}_noise_mean")):
                raise ValueError(f"{prefix}_noise_mean must be finite.")
            if getattr(self, f"{prefix}_noise_dist") not in {"gaussian", "laplace", "uniform"}:
                raise ValueError(f"Unknown {prefix}_noise_dist.")
        if not np.isfinite(self.weight_min) or not np.isfinite(self.weight_max) or not 0 <= self.weight_min <= self.weight_max:
            raise ValueError("Weights must satisfy 0 <= weight_min <= weight_max and be finite.")
        if self.intervention_node is not None:
            integer("intervention_node", 0, self.n_nodes - 1)
        if not np.isfinite(self.intervention_value):
            raise ValueError("intervention_value must be finite.")
        selected = []
        for kind in ("latent", "observed"):
            name = f"n_{kind}_confounders"
            integer(name, 0, self.n_nodes)
            nodes = getattr(self, f"{kind}_confounder_nodes")
            if nodes is not None:
                if len(nodes) != len(set(nodes)) or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < self.n_nodes for i in nodes):
                    raise ValueError(f"{kind}_confounder_nodes must contain unique valid node indices.")
                if getattr(self, name) not in (0, len(nodes)):
                    raise ValueError(f"{name} conflicts with {kind}_confounder_nodes.")
                setattr(self, name, len(nodes))
                selected.extend(nodes)
        if len(selected) != len(set(selected)):
            raise ValueError("Observed and latent confounder nodes must not overlap.")
        if self.n_latent_confounders + self.n_observed_confounders > self.n_nodes:
            raise ValueError("Total confounders cannot exceed n_nodes.")
        if self.node_mechanisms is not None and any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < self.n_nodes for i in self.node_mechanisms):
            raise ValueError("node_mechanisms keys must be valid node indices.")
        if self.mechanism_pool is not None:
            if self.mechanism != "mixed":
                raise ValueError(
                    "mechanism_pool can only be used with mechanism='mixed'."
                )
            if len(self.mechanism_pool) == 0:
                raise ValueError("mechanism_pool cannot be empty.")
            for mechanism in self.mechanism_pool:
                _validate_mechanism(mechanism, allow_mixed=False)
        _validate_mechanism(self.mechanism, allow_mixed=True)
        if self.node_mechanisms is not None:
            for mechanism in self.node_mechanisms.values():
                _validate_mechanism(mechanism, allow_mixed=False)
        for field_name, sampler in (
            ("root_noise_sampler", self.root_noise_sampler),
            ("node_noise_sampler", self.node_noise_sampler),
        ):
            if sampler is not None and not callable(sampler):
                raise TypeError(f"{field_name} must be callable or None.")
        if self.graph_type == "watts_strogatz":
            integer("n_neighbors", 2, self.n_nodes - 1)
            if self.n_neighbors < 2 or self.n_neighbors >= self.n_nodes:
                raise ValueError("n_neighbors must satisfy 2 <= n_neighbors < n_nodes.")
            if self.n_neighbors % 2 != 0:
                raise ValueError(
                    "n_neighbors must be even so each node has the same number "
                    "of neighbours on both sides."
                )

@dataclass
class EnvironmentConfig:
    """Sample counts and distribution shifts for a shared multi-environment model."""

    n_envs:                  int   = 4
    n_samples_per_env:       int   = 500
    heteroskedastic_noise:   bool  = False  # noise scale on Y grows with env index
    upstream_shift:          bool  = True   # shift root node distributions per env
    soft_interventions:      bool  = True   # shift non-target mechanisms per env
    soft_intervention_scale: float = 0.3

    def __post_init__(self):
        for name in ("n_envs", "n_samples_per_env"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(self.soft_intervention_scale) or self.soft_intervention_scale < 0:
            raise ValueError("soft_intervention_scale must be finite and non-negative.")



