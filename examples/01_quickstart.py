"""Generate a dataset and inspect its data and ground-truth graph."""
from causalgen import GeneratorConfig, generate


def main():
    config = GeneratorConfig(n_nodes=8, n_samples=500, seed=42)
    dataset = generate(config)

    # Rows are observations; columns are the observed variables X0, X1, ... .
    print(dataset.data.head())
    print(dataset.summary())

    # Entry [i, j] is 1 when the graph contains the edge Xi -> Xj.
    print(dataset.adjacency)
    print("Edge weights:", dataset.weight_matrix)


if __name__ == "__main__":
    main()
