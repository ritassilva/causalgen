"""Compatibility imports; use the public causalgen API in new code."""
from .config import GeneratorConfig, EnvironmentConfig
from .dataset import CausalDataset
from .generation import generate, generate_environments
from .graphs import build_undirected, enforce_dag
from .sampling import generate_samples
from .mechanisms import NumpyNoise, ParentMechanism, numpy_noise
