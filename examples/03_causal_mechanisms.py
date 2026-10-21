"""Use built-in mechanisms, mixed pools and scalar Python functions."""
import numpy as np

from causalgen import GeneratorConfig, generate


def smooth_response(z):
    """Map the weighted parent sum to a bounded nonlinear response."""
    return np.tanh(z) + 0.2 * np.sin(z)


def main():
    for mechanism in ("linear", "tanh", "sin", "sigmoid", "quadratic", "relu", "abs"):
        dataset = generate(GeneratorConfig(mechanism=mechanism, n_samples=100))
        print(mechanism, dataset.data.mean().round(3).to_dict())

    # Scalar functions receive z = weights @ parent_values. Root nodes are
    # drawn from the root noise distribution, without applying a mechanism.
    custom = generate(GeneratorConfig(mechanism=smooth_response, n_samples=100))
    print("Custom response:", custom.data.head())

    # A mixed mechanism is selected once per node and reused for every row.
    # Explicit per-node mechanisms take precedence over the mixed pool.
    mixed = generate(GeneratorConfig(
        mechanism="mixed", mechanism_pool=["linear", "tanh", np.cos],
        node_mechanisms={2: smooth_response, 4: "sin"}, n_samples=100, seed=7,
    ))
    print("Selected mechanisms:", mixed.metadata["node_forms"])


if __name__ == "__main__":
    main()
