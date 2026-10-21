"""Generate environments with distribution shifts and a shared causal DAG."""
from causalgen import EnvironmentConfig, GeneratorConfig, generate_environments


def main():
    config = GeneratorConfig(n_nodes=8, mechanism="tanh", seed=42)
    environments = EnvironmentConfig(
        n_envs=3, n_samples_per_env=200,
        upstream_shift=True, soft_interventions=True,
        soft_intervention_scale=0.2, heteroskedastic_noise=True,
    )
    dataset = generate_environments(config, environments)

    # n_samples_per_env sets the row count here; config.n_samples is unused.
    # The env column identifies the environment and is not a graph node.
    print("Rows per environment:", dataset.data.groupby("env").size())
    print("Means by environment:", dataset.data.groupby("env").mean())
    print("Invariant target:", dataset.metadata["target_node"])

    # The target is the last node in topological order. Its mechanism stays
    # fixed; heteroskedastic_noise changes its noise scale across environments.
    # Disable shifts and perturbations for environments differing only in noise.
    noise_only = generate_environments(config, EnvironmentConfig(
        n_envs=2, n_samples_per_env=100,
        upstream_shift=False, soft_interventions=False, heteroskedastic_noise=True,
    ))
    print("Noise-only environments:", noise_only.data.groupby("env").std())


if __name__ == "__main__":
    main()
