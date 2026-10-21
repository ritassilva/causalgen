"""Define a mechanism that uses individual parents and their interactions."""
import numpy as np

from causalgen import GeneratorConfig, ParentMechanism, generate


def interacting_parents(parents, weights):
    """Combine additive effects with an interaction between the first two parents."""
    response = weights @ parents
    if len(parents) >= 2:
        response += 0.1 * parents[0] * parents[1]
    return float(np.tanh(response))


def main():
    # Arrays contain only the current node's parents, in ascending node-index
    # order. The function must also handle nodes with a single parent.
    config = GeneratorConfig(
        n_nodes=8, edge_probability=0.6,
        mechanism=ParentMechanism(interacting_parents), n_samples=300, seed=42,
    )
    dataset = generate(config)
    print(dataset.data.head())

    # ParentMechanism can also appear in mechanism_pool or node_mechanisms.
    overridden = generate(GeneratorConfig(
        mechanism="linear", node_mechanisms={3: ParentMechanism(interacting_parents)},
        n_samples=100, seed=42,
    ))
    print(overridden.metadata["node_mechanisms"])


if __name__ == "__main__":
    main()
