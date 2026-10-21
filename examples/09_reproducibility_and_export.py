"""Repeat a seeded run and export data, ground truth and a DAG image."""
from pathlib import Path

import matplotlib.pyplot as plt

from causalgen import GeneratorConfig, generate


def main():
    config = GeneratorConfig(n_nodes=6, n_samples=200, mechanism="tanh", seed=0)
    first = generate(config)
    second = generate(config)

    # Zero is a valid seed. Repeated runs with identical settings and dependency
    # versions produce the same data; no seed offset is added.
    print("Identical seeded data:", first.data.equals(second.data))

    # seed=None chooses a random seed. Store the resolved value to repeat the
    # run later with the same configuration and software versions.
    unseeded = generate(GeneratorConfig(n_nodes=6, n_samples=200, seed=None))
    print("Resolved random seed:", unseeded.metadata["seed"])

    output = Path("example_output")
    paths = first.save(output, prefix="seeded", include_plot=False)
    for kind, path in paths.items():
        print(f"{kind}: {path}")

    # Plotting can be requested separately without opening an interactive window.
    figure = first.plot_dag(save_path=output / "seeded_dag.png", show=False)
    plt.close(figure)


if __name__ == "__main__":
    main()
