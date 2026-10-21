"""Configure root noise, node noise and edge-weight policies independently."""
from causalgen import GeneratorConfig, generate, numpy_noise


def student_noise(rng, size):
    """Draw heavy-tailed additive noise using the generator's random stream."""
    return rng.standard_t(df=4, size=size)


def main():
    config = GeneratorConfig(
        n_nodes=8, n_samples=300,
        root_noise_dist="laplace", root_noise_mean=0.0, root_noise_scale=1.0,
        node_noise_dist="uniform", node_noise_mean=0.0, node_noise_scale=0.5,
        weight_min=0.2, weight_max=0.8, allow_negative_weights=False, seed=42,
    )
    dataset = generate(config)
    print("Positive edge weights:", dataset.weight_matrix[dataset.dag == 1])

    # equal_contributions replaces random edge weights with 1 / n_parents.
    equal = generate(GeneratorConfig(equal_contributions=True, n_samples=100))
    print("Equal parent contributions:", equal.weight_matrix)

    # Custom samplers own their distribution parameters: the matching built-in
    # noise mean and scale are not applied to the returned samples.
    custom = generate(GeneratorConfig(
        root_noise_sampler=numpy_noise("gamma", shape=2.0, scale=0.5),
        node_noise_sampler=student_noise, n_samples=300, seed=42,
    ))
    print("Custom noise:", custom.data.head())


if __name__ == "__main__":
    main()
