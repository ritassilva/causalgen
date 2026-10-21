"""Generate and save two small datasets with different graph structures."""
from pathlib import Path

from causalgen import GeneratorConfig, generate


def main():
    output = Path("example_output")
    configurations = {
        "linear_erdos_renyi": GeneratorConfig(
            graph_type="erdos_renyi", n_nodes=8, edge_probability=0.3,
            mechanism="linear", n_samples=500, seed=42,
        ),
        "nonlinear_scale_free": GeneratorConfig(
            graph_type="barabasi_albert", n_nodes=8, edges_per_new_node=2,
            mechanism="tanh", node_noise_dist="laplace",
            n_samples=500, seed=42,
        ),
    }

    # Each configuration specifies the complete dataset. Add a configuration
    # to this mapping to generate another dataset in its own output folder.
    for name, config in configurations.items():
        dataset = generate(config)
        paths = dataset.save(output / name, prefix=name, include_plot=False)
        print(f"{name}: {dataset.data.shape[0]} rows, {dataset.data.shape[1]} variables")
        for kind, path in paths.items():
            print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
