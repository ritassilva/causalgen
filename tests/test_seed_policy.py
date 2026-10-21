import numpy as np
import pandas as pd

from causalgen import EnvironmentConfig, GeneratorConfig, generate, generate_environments
from causalgen.causal_generator import build_undirected, enforce_dag


def test_zero_seed_is_used_without_offset():
    cfg = GeneratorConfig(seed=0)
    rng = np.random.default_rng(0)
    expected = enforce_dag(build_undirected(cfg, rng), cfg.dag_method, rng)
    dataset = generate(cfg)
    np.testing.assert_array_equal(dataset.dag, expected)
    assert dataset.metadata["seed"] == 0
    assert cfg.seed == 0
    pd.testing.assert_frame_equal(dataset.observational, generate(cfg).observational)


def test_environment_streams_are_repeatable_and_sample_count_is_respected():
    cfg = GeneratorConfig(seed=0, n_samples=11, intervention_node=2)
    env_cfg = EnvironmentConfig(n_envs=2, n_samples_per_env=7)
    first = generate_environments(cfg, env_cfg)
    second = generate_environments(cfg, env_cfg)
    assert first.metadata["seed"] == 0
    assert first.observational.groupby("env").size().tolist() == [7, 7]
    assert first.interventional.groupby("env").size().tolist() == [7, 7]
    pd.testing.assert_frame_equal(first.observational, second.observational)
    assert not first.observational.iloc[:7].drop(columns="env").reset_index(drop=True).equals(
        first.observational.iloc[7:].drop(columns="env").reset_index(drop=True)
    )
