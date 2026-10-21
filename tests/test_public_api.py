import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from causalgen import EnvironmentConfig, GeneratorConfig, ParentMechanism, generate
from causalgen.cli import main
from causalgen.sampling import generate_samples


@pytest.mark.parametrize('options, message', [
    ({'n_nodes': 2}, 'n_nodes'),
    ({'n_samples': 0}, 'n_samples'),
    ({'seed': -1}, 'seed'),
    ({'graph_type': 'missing'}, 'graph_type'),
    ({'dag_method': 'missing'}, 'dag_method'),
    ({'edge_probability': 1.1}, 'edge_probability'),
    ({'graph_type': 'barabasi_albert', 'edges_per_new_node': 1.5}, 'edges_per_new_node'),
    ({'node_noise_scale': float('nan')}, 'node_noise_scale'),
    ({'node_mechanisms': {8: 'sin'}}, 'node_mechanisms'),
    ({'observed_confounder_nodes': [0], 'latent_confounder_nodes': [0]}, 'overlap'),
    ({'observed_confounder_nodes': [0], 'n_observed_confounders': 2}, 'conflicts'),
    ({'intervention_node': 8}, 'intervention_node'),
])
def test_invalid_settings_explain_the_problem(options, message):
    with pytest.raises(ValueError, match=message):
        GeneratorConfig(**options)


@pytest.mark.parametrize('graph_type, options', [
    ('erdos_renyi', {'edge_probability': 0.5}),
    ('barabasi_albert', {'edges_per_new_node': 2}),
    ('watts_strogatz', {'n_neighbors': 4, 'rewiring_probability': 0.2}),
    ('chain', {}), ('tree', {}), ('modular_biology', {'n_modules': 2}),
])
def test_topologies_work_through_public_api(graph_type, options):
    dataset = generate(GeneratorConfig(graph_type=graph_type, n_samples=3, **options))
    assert dataset.data.shape == (3, 8)


def test_manual_confounders_hide_only_latent_variables_and_export(tmp_path):
    dataset = generate(GeneratorConfig(
        n_nodes=5, edge_probability=1, latent_confounder_nodes=[0],
        observed_confounder_nodes=[1], intervention_node=2, n_samples=4,
    ))
    assert dataset.config.n_latent_confounders == 1
    assert dataset.confounders == [0, 1]
    assert 'X0' not in dataset.data
    assert 'X1' in dataset.data
    assert 'X0' not in dataset.interventional
    assert dataset.full_observational.shape == (4, 5)
    assert dataset.adjacency.shape == (4, 4)
    paths = dataset.save(tmp_path, include_plot=False)
    assert 'dag_plot' not in paths
    pd.testing.assert_frame_equal(dataset.data, pd.read_csv(paths['observational']))
    assert json.loads(paths['metadata'].read_text())['latent_confounders'] == ['X0']


def test_impossible_confounders_fail_instead_of_silently_disappearing():
    with pytest.raises(ValueError, match='eligible'):
        generate(GeneratorConfig(graph_type='chain', n_latent_confounders=1))
    with pytest.raises(ValueError, match='two children'):
        generate(GeneratorConfig(graph_type='chain', latent_confounder_nodes=[0]))


def test_parent_mechanism_can_distinguish_individual_parents():
    dag = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    weights = np.array([[0., 0., 2.], [0., 0., 3.], [0., 0., 0.]])
    mechanism = ParentMechanism(lambda parents, weights: parents[0] * weights[1] + parents[1] ** 2)
    config = GeneratorConfig(
        n_nodes=3, n_samples=1, mechanism=mechanism,
        root_noise_sampler=lambda rng, size: np.ones(size),
        node_noise_sampler=lambda rng, size: np.zeros(size),
    )
    data, _, _ = generate_samples(dag, config, np.random.default_rng(1), weight_matrix=weights)
    np.testing.assert_array_equal(data.iloc[0], [1, 1, 4])
    assert generate(config).metadata['mechanism'].startswith('parents:')


def test_cli_can_generate_without_rendering_a_plot(tmp_path):
    with patch('causalgen.dataset.CausalDataset.plot_dag') as plot:
        main(['--n-samples', '3', '--seed', '0', '--no-plot', '--output-dir', str(tmp_path)])
    plot.assert_not_called()
    assert len(pd.read_csv(tmp_path / 'causal_observational.csv')) == 3
    assert json.loads((tmp_path / 'causal_metadata.json').read_text())['seed'] == 0


def test_cli_multiple_environments(tmp_path):
    main(['--n-envs', '2', '--n-samples-per-env', '3', '--no-plot', '--output-dir', str(tmp_path)])
    data = pd.read_csv(tmp_path / 'causal_observational.csv')
    assert data.groupby('env').size().tolist() == [3, 3]


def test_environment_count_must_be_positive():
    with pytest.raises(ValueError, match='n_envs'):
        EnvironmentConfig(n_envs=0)


def test_export_accepts_numpy_scalar_configuration(tmp_path):
    dataset = generate(GeneratorConfig(
        n_nodes=np.int64(5), n_samples=np.int64(3), seed=np.int64(0),
        node_noise_scale=np.float32(0.5),
    ))
    paths = dataset.save(tmp_path, include_plot=False)
    metadata = json.loads(paths['metadata'].read_text())
    assert metadata['seed'] == 0
    assert metadata['n_samples'] == 3
    assert metadata['node_noise_scale'] == 0.5
