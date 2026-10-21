"""Generate each supported graph topology with its own parameters."""
from causalgen import GeneratorConfig, generate


def main():
    topologies = {
        "erdos_renyi": {"edge_probability": 0.3},
        "barabasi_albert": {"edges_per_new_node": 2},
        "watts_strogatz": {"n_neighbors": 4, "rewiring_probability": 0.2},
        "chain": {},
        "tree": {},
        "modular_biology": {
            "n_modules": 3, "p_intra": 0.4, "p_inter": 0.05,
            "n_hubs_per_module": 1, "p_hub": 0.7,
        },
    }
    for graph_type, parameters in topologies.items():
        config = GeneratorConfig(
            graph_type=graph_type, n_nodes=12, n_samples=100,
            dag_method="random", seed=42, **parameters,
        )
        dataset = generate(config)
        print(f"{graph_type}: {dataset.data.shape}, {dataset.dag.sum()} edges")

    # Random orientation uses a seeded node order. Modular biology constructs
    # its own directed graph and does not use dag_method.


if __name__ == "__main__":
    main()
