from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Mapping, Optional, Union
import numpy as np

if TYPE_CHECKING:
    from .config import GeneratorConfig

GraphType = Literal[ "erdos_renyi", "barabasi_albert", "watts_strogatz", "chain", "tree", "modular_biology"]
BuiltInFuncForm = Literal["linear", "tanh", "sin", "sigmoid", "quadratic", "relu", "abs"]
Mechanism = Union[BuiltInFuncForm, Callable[[float], float]]
FuncForm = Union[Mechanism, Literal["mixed"]]
NoiseDist   = Literal["gaussian", "laplace", "uniform"]
NoiseSampler = Callable[[np.random.Generator, int], np.ndarray]
DAGMethod   = Literal["topological", "random"]

# Built-in mechanism order used for mixed draws.
FUNCFORM = ["linear", "tanh", "mixed", "sin", "sigmoid", "quadratic", "relu", "abs"]
BUILTIN_MECHANISMS = tuple(form for form in FUNCFORM if form != "mixed")


@dataclass(frozen=True)
class ParentMechanism:
    """Wrap f(parent_values, weights); parents are ordered by node index."""
    function: Callable

    def __post_init__(self):
        if not callable(self.function):
            raise ValueError("ParentMechanism requires a callable.")

    def __call__(self, parents, weights):
        return self.function(parents, weights)


@dataclass(frozen=True)
class NumpyNoise:
    """Reusable noise sampler backed by ``numpy.random.Generator``.

    Parameters are forwarded unchanged to the named NumPy distribution; the
    sample size is supplied by the causal generator.  For example,
    ``NumpyNoise("gamma", {"shape": 2.0, "scale": 0.5})``.
    """

    distribution: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.distribution, str) or not self.distribution:
            raise ValueError("distribution must be a non-empty string")
        method = getattr(np.random.default_rng(), self.distribution, None)
        if method is None or not callable(method):
            raise ValueError(
                f"Unknown NumPy Generator distribution: {self.distribution!r}"
            )
        object.__setattr__(self, "parameters", dict(self.parameters))

    def __call__(self, rng: np.random.Generator, size: int) -> np.ndarray:
        method = getattr(rng, self.distribution)
        return np.asarray(method(size=size, **self.parameters))


def numpy_noise(distribution: str, **parameters: Any) -> NumpyNoise:
    """Create a seeded sampler for any NumPy ``Generator`` distribution."""

    return NumpyNoise(distribution, parameters)

# ---------------------------------------------------------------------------
# Mechanism validation and metadata
# ---------------------------------------------------------------------------
def _validate_mechanism(mechanism: FuncForm, allow_mixed: bool) -> None:
    if callable(mechanism):
        return
    allowed = FUNCFORM if allow_mixed else BUILTIN_MECHANISMS
    if not isinstance(mechanism, str) or mechanism not in allowed:
        raise ValueError(
            f"Unknown mechanism {mechanism!r}. Available built-ins: {list(allowed)}; "
            "a callable is also accepted."
        )


def _callable_name(function: Callable) -> str:
    if isinstance(function, np.ufunc):
        return f"numpy.{function.__name__}"
    if isinstance(function, ParentMechanism):
        return "parents:" + _callable_name(function.function)
    module = getattr(function, "__module__", None)
    name = getattr(function, "__qualname__", None) or getattr(function, "__name__", None)
    if name is None:
        name = function.__class__.__qualname__
    if module and module not in {"__main__", "builtins"}:
        return f"{module}.{name}"
    return name


def _mechanism_name(mechanism: FuncForm) -> str:
    return mechanism if isinstance(mechanism, str) else _callable_name(mechanism)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _noise_sampler_metadata(sampler: Optional[NoiseSampler]) -> Optional[Dict[str, Any]]:
    if sampler is None:
        return None
    metadata: Dict[str, Any] = {"callable": _callable_name(sampler)}
    if isinstance(sampler, NumpyNoise):
        metadata.update(
            distribution=sampler.distribution,
            parameters=_json_safe(sampler.parameters),
        )
    return metadata


def _resolve_node_mechanisms(
    cfg: GeneratorConfig,
    n: int,
    rng: np.random.Generator,
) -> List[Mechanism]:
    if cfg.mechanism == "mixed":
        pool = list(cfg.mechanism_pool or BUILTIN_MECHANISMS)
        # Object dtype allows built-in names and Python callables in one pool.
        pool_array = np.asarray(pool, dtype=object)
        node_forms = [rng.choice(pool_array) for _ in range(n)]
    else:
        node_forms = [cfg.mechanism] * n

    if cfg.node_mechanisms is not None:
        for node_idx, mechanism in cfg.node_mechanisms.items():
            if 0 <= node_idx < n:
                node_forms[node_idx] = mechanism
            else:
                warnings.warn(
                    f"node_mechanisms: node {node_idx} does not exist (n={n}), skipping."
                )
    return node_forms


# ---------------------------------------------------------------------------
# Noise samplers
# ---------------------------------------------------------------------------

def _sample_noise(
    n: int,
    dist: NoiseDist,
    mean: float,
    scale: float,
    rng: np.random.Generator,
    sampler: Optional[NoiseSampler] = None,
) -> np.ndarray:
    if sampler is not None:
        try:
            samples = np.asarray(sampler(rng, n), dtype=float)
        except Exception as exc:
            raise ValueError(
                f"Custom noise sampler {_callable_name(sampler)!r} failed."
            ) from exc
        if samples.ndim == 0 and n == 1:
            samples = samples.reshape(1)
        if samples.shape != (n,):
            raise ValueError(
                f"Custom noise sampler {_callable_name(sampler)!r} returned shape "
                f"{samples.shape}; expected ({n},)."
            )
        return samples
    if dist == "gaussian":
        return rng.normal(mean, scale, n)
    if dist == "laplace":
        return rng.laplace(mean, scale, n)
    if dist == "uniform":
        return rng.uniform(mean - scale, mean + scale, n)
    raise ValueError(f"Unknown noise_dist: {dist!r}")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _compute_node_value(
    parents_values: np.ndarray,
    weights: np.ndarray,
    form: Mechanism,
) -> float:
    if len(parents_values) == 0:
        return 0.0
    weighted_parents = float(weights @ parents_values)
    if callable(form):
        try:
            result = np.asarray(form(parents_values.copy(), weights.copy()) if isinstance(form, ParentMechanism) else form(weighted_parents))
        except Exception as exc:
            raise ValueError(
                f"Custom mechanism {_callable_name(form)!r} failed for input "
                f"z={weighted_parents}."
            ) from exc
        if result.size != 1:
            raise ValueError(
                f"Custom mechanism {_callable_name(form)!r} must return one "
                f"numeric value, got shape {result.shape}."
            )
        try:
            return float(result.reshape(-1)[0])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Custom mechanism {_callable_name(form)!r} must return a numeric value."
            ) from exc
    if form == "linear":
        return weighted_parents
    if form == "tanh":
        return float(np.tanh(weighted_parents))
    if form == "sin":
        return float(np.sin(weighted_parents))
    if form == "sigmoid":
        z = weighted_parents
        return float(1 / (1 + np.exp(-z)))
    if form == "relu":
        return float(np.maximum(0, weighted_parents))
    if form == "quadratic":
        z = weighted_parents
        return np.tanh(0.5 * z**2)
    if form == "abs":
        return float(np.abs(weighted_parents))
    raise ValueError(f"Unknown mechanism: {form!r}")


