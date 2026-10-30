# causalgen

Generate synthetic causal datasets with configurable graphs, mechanisms, noise,
confounders, interventions and multiple environments. Use it to benchmark causal
discovery methods or explore causal models with known ground truth. Results
include observation tables, graph structure and metadata that can be exported
for use in your own analysis.

Start with the [quick start](#start-with-a-dataset), then explore the
[examples](#examples) for complete, runnable workflows.

## Install

From a checkout of this repository:

```bash
python -m pip install .
```

Requires Python 3.10 or later. This version has not yet been published to PyPI.

## Start with a dataset

```python
from causalgen import GeneratorConfig, generate

config = GeneratorConfig(n_nodes=10, n_samples=2000, seed=42)
dataset = generate(config)
data = dataset.data                       # pandas DataFrame, observed variables
adjacency = dataset.adjacency             # rows = sources, columns = targets
print(dataset.summary())
dataset.save("results", prefix="experiment", include_plot=False)
```

`n_samples` is the number of rows generated directly; there is no subsequent
subsampling. Columns are named `X0`, `X1`, … and node indices start at zero.
`n_nodes` is the total number of nodes, including any latent variables.

## Choose a graph

Use the parameters relevant to the selected `graph_type`:

| `graph_type` | Parameters | Defaults |
|---|---|---|
| `erdos_renyi` | `edge_probability` | `0.35` |
| `barabasi_albert` | `edges_per_new_node` | `2` |
| `watts_strogatz` | `n_neighbors`, `rewiring_probability` | `4`, `0.35` |
| `chain` | None | — |
| `tree` | None (binary tree) | — |
| `modular_biology` | `n_modules`, `p_intra`, `p_inter`, `n_hubs_per_module`, `p_hub` | `5`, `0.25`, `0.03`, `1`, `0.5` |

```python
config = GeneratorConfig(
    graph_type="barabasi_albert", edges_per_new_node=2,
    n_nodes=20, n_samples=2000, seed=42,
)
dataset = generate(config)
```

For the first five topologies, `dag_method="topological"` orients edges using
node index order; `"random"` uses a seeded random order. `modular_biology`
constructs its own DAG and does not use `dag_method`. Orientation retains only
edges pointing forward in the selected order. ER samples an ordered adjacency
before orientation; `edge_probability` is not a final mean-degree parameter.
Watts–Strogatz requires an even `n_neighbors` smaller than `n_nodes`.

## Choose causal mechanisms

Set `mechanism` to `linear` (default), `tanh`, `sin`, `sigmoid`, `quadratic`,
`relu`, `abs`, or `mixed`. Given parent values and edge weights, the built-ins
apply their function to `z = weights @ parents`. The quadratic mechanism is
`np.tanh(0.5 * z**2)`. Noise is added after applying the mechanism.
Root nodes are generated from their noise distribution, without a mechanism.

Simple Python callables receive the scalar `z` and return one numeric value:

```python
import numpy as np

config = GeneratorConfig(mechanism=np.cos, n_samples=2000)
dataset = generate(config)
```

With `mixed`, a mechanism is chosen once per node, not per observation.
Use `mechanism_pool` to specify candidates and `node_mechanisms` for overrides:

```python
config = GeneratorConfig(
    mechanism="mixed",
    mechanism_pool=["linear", "tanh", np.cos],
    node_mechanisms={2: np.sin},
    seed=42,
)
dataset = generate(config)
```

For functions needing each parent separately, wrap a two-argument callable with
`ParentMechanism`. It receives the parent values and corresponding weights,
ordered by parent node index. Both arguments are one-dimensional NumPy arrays:

```python
from causalgen import ParentMechanism

def interaction(parents, weights):
    return float(weights @ parents + 0.1 * np.prod(parents))

config = GeneratorConfig(mechanism=ParentMechanism(interaction))
dataset = generate(config)
```

Custom mechanisms must return a scalar and should be deterministic for
reproducibility. JSON metadata records callable names, not executable code.

## Configure noise and weights

| Setting | Applies to | Default |
|---|---|---|
| `root_noise_dist` | Nodes without parents | `"gaussian"` |
| `node_noise_dist` | Nodes with parents | `"gaussian"` |
| `root_noise_mean`, `node_noise_mean` | Distribution location | `0.0` |
| `root_noise_scale`, `node_noise_scale` | Distribution scale | `1.0` |
| `weight_min`, `weight_max` | Edge-weight magnitude range | `0.5`, `2.0` |
| `allow_negative_weights` | Randomly assign positive/negative signs | `True` |
| `equal_contributions` | Use weight `1 / number_of_parents` instead of random weights | `False` |

Noise distributions: `gaussian` uses scale as standard deviation, `laplace`
uses its scale parameter, and `uniform` samples between `mean - scale` and
`mean + scale`. Scale must be positive. Weights are fixed across observations.

For other distributions, supply `root_noise_sampler` or `node_noise_sampler`.
Samplers receive `(rng, size)` and return an array of shape `(size,)`:

```python
from causalgen import numpy_noise

config = GeneratorConfig(
    root_noise_sampler=numpy_noise("gamma", shape=2.0, scale=0.5),
    node_noise_sampler=lambda rng, size: rng.standard_t(df=4, size=size),
)
dataset = generate(config)
```

A custom sampler owns all distribution parameters. Its corresponding built-in
distribution, mean and scale are not applied to its output. In multiple
environments it also overrides configured root distribution shifts or target
noise scales. Use the supplied RNG for reproducibility.

## Confounders and latent variables

A confounder must have at least two children in the generated DAG. Selection
marks existing nodes; it does not add nodes or edges. Random selection:

```python
config = GeneratorConfig(
    n_nodes=10, edge_probability=0.5,
    n_observed_confounders=1, n_latent_confounders=1,
)
dataset = generate(config)
```

Alternatively, specify separate node lists. Each list determines its group's
count; the matching `n_*_confounders` may be omitted or must match its length.

```python
config = GeneratorConfig(
    n_nodes=5, edge_probability=1.0,
    observed_confounder_nodes=[1], latent_confounder_nodes=[0],
)
dataset = generate(config)
```

Manual nodes must be eligible in the resulting DAG and the two lists cannot
overlap. You can specify one group manually and let the other be selected
randomly. If the DAG has too few eligible nodes, generation raises a descriptive
`ValueError`; increase graph density or reduce the requested count.

## Access and export results

| Attribute or method | Result |
|---|---|
| `dataset.data` / `dataset.observational` | Observational DataFrame with latent columns excluded |
| `dataset.interventional` | Interventional DataFrame with latent columns excluded, or `None` |
| `dataset.adjacency` | Adjacency DataFrame restricted to observed nodes |
| `dataset.full_observational` | All generated variables, including latent variables |
| `dataset.full_interventional` | Full interventional data, or `None` |
| `dataset.dag` | Full NumPy adjacency matrix; `[i, j] = 1` means `i → j` |
| `dataset.weight_matrix` | Full NumPy weight matrix, same orientation |
| `dataset.confounders` | Selected node indices, latent group first |
| `dataset.metadata` | Configuration and ground-truth information |
| `dataset.summary()` | Human-readable summary |
| `dataset.plot_dag(show=True)` | Draw the full DAG and return a Matplotlib figure |

`observational_df()`, `interventional_df()` and `adjacency_df()` are also
available. Observed adjacency is an induced subgraph, not a latent projection;
it does not represent confounding edges caused by hidden variables.

`dataset.save("results", prefix="experiment", include_plot=True)` writes:

- `experiment_observational.csv`
- `experiment_interventional.csv` when an intervention is configured
- `experiment_adjacency.csv` for observed nodes
- `experiment_edge_list.csv` for the full ground-truth DAG
- `experiment_metadata.json` including latent-node identities
- `experiment_dag.png` for the full DAG, unless `include_plot=False`

Metadata, edge lists and plots deliberately retain latent ground truth.

## Interventions

```python
config = GeneratorConfig(intervention_node=2, intervention_value=1.5)
dataset = generate(config)
interventional_data = dataset.interventional
```

This generates observational data and a second dataset under `do(X2 = 1.5)`.
The target's structural equation is replaced with a constant; outgoing effects
on descendants remain active. Both datasets share weights and mechanisms, but
use newly drawn noise. They are not paired counterfactual observations.
The returned DAG is the original ground-truth DAG, before intervention.

## Multiple environments

```python
from causalgen import EnvironmentConfig, generate_environments

config = GeneratorConfig(n_nodes=10, seed=42)
environments = EnvironmentConfig(n_envs=4, n_samples_per_env=500)
dataset = generate_environments(config, environments)
# dataset.data has 2000 rows and an additional 'env' column.
```

The DAG is shared across environments. The target is the last node in the
DAG's topological order. Environment options:

| Setting | Default | Behavior |
|---|---|---|
| `n_envs` | `4` | Number of environments |
| `n_samples_per_env` | `500` | Rows per environment; replaces `config.n_samples` |
| `upstream_shift` | `True` | Draw different means/scales for root distributions |
| `soft_interventions` | `True` | Perturb non-target edge weights |
| `soft_intervention_scale` | `0.3` | Maximum absolute weight perturbation |
| `heteroskedastic_noise` | `False` | Increase target noise scale with environment index |

The target mechanism remains fixed. The base weight matrix is available as
`dataset.weight_matrix`; per-environment weight perturbations are not returned.
`env` is a dataset column, not a graph node.

## Seed policy

`seed=42` is the default, `seed=0` is valid, and `seed=None` selects a random
seed recorded in metadata. No arithmetic offset or dataset counter is added.
`generate()` uses `default_rng(seed)`. Multiple environments use independent
streams derived with `SeedSequence(seed).spawn(...)`. For reproducibility,
keep configurations, callables and dependency versions fixed.

## Command line

```bash
causalgen --n-nodes 10 --n-samples 2000 --seed 42 --no-plot --output-dir results
causalgen --graph-type barabasi_albert --edges-per-new-node 2 --mechanism tanh
causalgen --n-envs 4 --n-samples-per-env 500 --no-plot
python -m causalgen --help
```

The CLI supports graph parameters, noise, confounders, weights, interventions
and environments. Custom Python callables and per-node mechanism overrides
use the Python API. `--no-plot` skips rendering entirely.

## Examples

The example scripts are available in the repository’s `examples/` directory.
Install the package as described above, then run them from the repository root.

Start with the basic example:

```bash
python examples/01_quickstart.py
```

| Example | Topics |
|---|---|
| [01_quickstart.py](examples/01_quickstart.py) | Configuration, observed data, adjacency and weights |
| [02_graph_topologies.py](examples/02_graph_topologies.py) | All six topologies and graph-specific parameters |
| [03_causal_mechanisms.py](examples/03_causal_mechanisms.py) | Built-ins, scalar functions, mixed pools and node overrides |
| [04_parent_interactions.py](examples/04_parent_interactions.py) | Functions receiving individual parents and weights |
| [05_noise_and_weights.py](examples/05_noise_and_weights.py) | Built-in/custom noise, positive weights and equal contributions |
| [06_confounders.py](examples/06_confounders.py) | Manual/random selection and access to latent ground truth |
| [07_interventions.py](examples/07_interventions.py) | Hard interventions and observational/interventional data |
| [08_multiple_environments.py](examples/08_multiple_environments.py) | Root shifts, soft interventions and heteroskedastic noise |
| [09_reproducibility_and_export.py](examples/09_reproducibility_and_export.py) | Seeds, CSV/JSON export and DAG images |

The numbered examples use small datasets. Only example 09 writes files, under
`example_output/` in the current working directory. It saves a DAG image without
opening a window.

### Generate and save two datasets

[generate_datasets.py](examples/generate_datasets.py) generates two datasets of 500 rows:
a linear Erdős–Rényi dataset and a nonlinear scale-free dataset with Laplace
noise on non-root nodes.

```bash
python examples/generate_datasets.py
```

Each dataset is saved to a separate folder under `example_output/`, with CSV
data, adjacency, edge lists and JSON metadata. Edit the configurations in the
script to change sample counts, mechanisms, noise or graph parameters.

## License

CausalGen is distributed under the [MIT license](LICENSE).

## Contributing and development

To work on the package, install the development dependencies from the repository root:

```bash
python -m pip install -e ".[dev]"
```

Code is organized into `config`, `graphs`, `mechanisms`, `sampling`,
`generation`, `dataset` and `cli`. Import public objects directly from `causalgen`.

```bash
python -m pytest
python -m build
```

Tests stay in the repository and are not included in the installed package.
