"""Select confounders and distinguish observed data from full ground truth."""
from causalgen import GeneratorConfig, generate


def main():
    # A complete DAG makes nodes 0 and 1 eligible: each has at least two
    # children. n_nodes includes both observed and latent variables.
    manual = generate(GeneratorConfig(
        n_nodes=5, edge_probability=1.0,
        observed_confounder_nodes=[1], latent_confounder_nodes=[0],
        n_samples=200, seed=42,
    ))
    print("Observed columns:", manual.data.columns.tolist())
    print("All columns:", manual.full_observational.columns.tolist())
    print("Latent nodes:", manual.metadata["latent_confounders"])
    print("Observed-node adjacency:", manual.adjacency)

    # Counts select existing eligible nodes without adding nodes or edges.
    # Selection fails if the generated graph has too few eligible nodes.
    random_selection = generate(GeneratorConfig(
        n_nodes=8, edge_probability=1.0,
        n_observed_confounders=1, n_latent_confounders=2,
        n_samples=200, seed=42,
    ))
    print("Randomly selected confounders:", random_selection.confounders)

    # Full DAGs, edge lists and metadata retain latent ground truth. The
    # observed adjacency is an induced subgraph, not a latent projection.


if __name__ == "__main__":
    main()
