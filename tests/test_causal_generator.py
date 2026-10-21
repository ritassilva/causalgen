from unittest.mock import patch

import numpy as np

from causalgen.causal_generator import (
    GeneratorConfig,
    build_undirected,
    generate,
    generate_samples,
    numpy_noise,
)


def _zero_noise(n, dist, mean, scale, rng):
    return np.zeros(n)


def test_do_intervention_preserves_effects_on_descendants():
    """do(X0=3) must still propagate through X0 -> X1 -> X2."""
    dag = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    weights = np.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0],
        ]
    )
    cfg = GeneratorConfig(n_nodes=3, n_samples=1, mechanism="linear")

    with patch(
        "causalgen.sampling._sample_noise",
        side_effect=_zero_noise,
    ):
        data, _, _ = generate_samples(
            dag,
            cfg,
            np.random.default_rng(1),
            intervention_node=0,
            intervention_value=3.0,
            weight_matrix=weights,
        )

    np.testing.assert_allclose(data.iloc[0].to_numpy(), [3.0, 6.0, 24.0])


def test_do_intervention_replaces_only_target_structural_equation():
    """do(X1=5) ignores X1's parent but preserves X1 -> X2."""
    dag = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    weights = np.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0],
        ]
    )
    cfg = GeneratorConfig(n_nodes=3, n_samples=1, mechanism="linear")

    with patch(
        "causalgen.sampling._sample_noise",
        side_effect=_zero_noise,
    ):
        data, _, _ = generate_samples(
            dag,
            cfg,
            np.random.default_rng(1),
            intervention_node=1,
            intervention_value=5.0,
            weight_matrix=weights,
        )

    np.testing.assert_allclose(data.iloc[0].to_numpy(), [0.0, 5.0, 20.0])


def test_watts_strogatz_neighbour_count_is_configurable():
    cfg_k2 = GeneratorConfig(
        graph_type="watts_strogatz", n_nodes=12, rewiring_probability=0.0, n_neighbors=2
    )
    cfg_k6 = GeneratorConfig(
        graph_type="watts_strogatz", n_nodes=12, rewiring_probability=0.0, n_neighbors=6
    )

    graph_k2 = build_undirected(cfg_k2, np.random.default_rng(1))
    graph_k6 = build_undirected(cfg_k6, np.random.default_rng(1))

    # Symmetric adjacency matrices count each undirected edge twice.
    assert graph_k2.sum() // 2 == 12
    assert graph_k6.sum() // 2 == 36


def test_watts_strogatz_neighbour_count_must_be_valid():
    for invalid_k in (1, 3, 12):
        try:
            GeneratorConfig(
                graph_type="watts_strogatz",
                n_nodes=12,
                rewiring_probability=0.2,
                n_neighbors=invalid_k,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"n_neighbors={invalid_k} should have been rejected")


def test_custom_numpy_mechanism_receives_weighted_parent_value():
    dag = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    weights = np.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
        ]
    )
    cfg = GeneratorConfig(
        n_nodes=3,
        n_samples=1,
        mechanism=np.square,
        root_noise_sampler=lambda rng, size: np.ones(size),
        node_noise_sampler=lambda rng, size: np.zeros(size),
    )

    data, _, node_forms = generate_samples(
        dag, cfg, np.random.default_rng(1), weight_matrix=weights
    )

    # X0=1; X1=(2*1)^2=4; X2=(3*4)^2=144.
    np.testing.assert_allclose(data.iloc[0].to_numpy(), [1.0, 4.0, 144.0])
    assert node_forms == [np.square, np.square, np.square]


def test_mixed_pool_accepts_builtin_and_custom_mechanisms_with_override():
    cfg = GeneratorConfig(
        n_nodes=5,
        n_samples=3,
        mechanism="mixed",
        mechanism_pool=["linear", np.cos],
        node_mechanisms={2: np.sin},
        seed=7,
    )

    dataset = generate(cfg)

    assert dataset.metadata["mechanism_pool"] == ["linear", "numpy.cos"]
    assert dataset.metadata["node_mechanisms"] == {"X2": "numpy.sin"}
    assert dataset.metadata["node_forms"][2] == "numpy.sin"
    assert set(dataset.metadata["node_forms"]) <= {
        "linear", "numpy.cos", "numpy.sin"
    }


def test_numpy_noise_supports_arbitrary_generator_distribution_and_seed():
    sampler = numpy_noise("gamma", shape=2.0, scale=0.5)
    cfg = GeneratorConfig(
        graph_type="chain",
        n_nodes=3,
        n_samples=6,
        root_noise_sampler=sampler,
        node_noise_sampler=sampler,
        seed=19,
    )

    first = generate(cfg)
    second = generate(cfg)

    np.testing.assert_allclose(first.observational, second.observational)
    assert first.metadata["node_noise_sampler"] == {
        "callable": "causalgen.mechanisms.NumpyNoise",
        "distribution": "gamma",
        "parameters": {"shape": 2.0, "scale": 0.5},
    }


def test_default_generation_is_unchanged_when_new_options_are_omitted():
    implicit_defaults = GeneratorConfig(
        graph_type="chain",
        n_nodes=4,
        n_samples=8,
        mechanism="mixed",
        seed=23,
    )
    explicit_defaults = GeneratorConfig(
        graph_type="chain",
        n_nodes=4,
        n_samples=8,
        mechanism="mixed",
        mechanism_pool=None,
        root_noise_sampler=None,
        node_noise_sampler=None,
        seed=23,
    )

    first = generate(implicit_defaults)
    second = generate(explicit_defaults)

    np.testing.assert_allclose(first.observational, second.observational)
    assert first.metadata["node_forms"] == second.metadata["node_forms"]
