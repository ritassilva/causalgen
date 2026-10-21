"""Synthetic causal dataset generation."""

from .causal_generator import (
    CausalDataset,
    EnvironmentConfig,
    GeneratorConfig,
    NumpyNoise,
    ParentMechanism,
    generate,
    generate_environments,
    numpy_noise,
)

__all__ = [
    "CausalDataset", "EnvironmentConfig", "GeneratorConfig", "NumpyNoise", "ParentMechanism",
    "generate", "generate_environments", "numpy_noise",
]
__version__ = "0.1.0"
