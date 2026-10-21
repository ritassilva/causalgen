"""Command-line interface. Custom Python functions use the Python API."""
import argparse

from . import EnvironmentConfig, GeneratorConfig, generate, generate_environments


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic causal datasets.")
    graph = parser.add_argument_group("Graph")
    graph.add_argument("--graph-type", default="erdos_renyi", choices=["erdos_renyi", "barabasi_albert", "watts_strogatz", "chain", "tree", "modular_biology"])
    graph.add_argument("--n-nodes", type=int, default=8)
    graph.add_argument("--edge-probability", type=float, default=0.35, help="Erdos-Renyi edge probability.")
    graph.add_argument("--edges-per-new-node", type=int, default=2, help="Barabasi-Albert attachment count.")
    graph.add_argument("--rewiring-probability", type=float, default=0.35, help="Watts-Strogatz rewiring probability.")
    graph.add_argument("--n-neighbors", type=int, default=4, help="Watts-Strogatz initial neighbor count (even).")
    graph.add_argument("--dag-method", choices=["topological", "random"], default="topological")
    graph.add_argument("--n-modules", type=int, default=5)
    graph.add_argument("--p-intra", type=float, default=0.25)
    graph.add_argument("--p-inter", type=float, default=0.03)
    graph.add_argument("--n-hubs-per-module", type=int, default=1)
    graph.add_argument("--p-hub", type=float, default=0.5)
    confounders = parser.add_argument_group("Confounders")
    confounders.add_argument("--n-observed-confounders", type=int, default=0)
    confounders.add_argument("--n-latent-confounders", type=int, default=0)
    confounders.add_argument("--observed-confounder-nodes", type=int, nargs="+")
    confounders.add_argument("--latent-confounder-nodes", type=int, nargs="+")
    data = parser.add_argument_group("Mechanisms, noise and samples")
    forms = ["linear", "tanh", "mixed", "sin", "sigmoid", "quadratic", "relu", "abs"]
    data.add_argument("--mechanism", default="linear", choices=forms)
    data.add_argument("--mechanism-pool", nargs="+", choices=[f for f in forms if f != "mixed"])
    for prefix in ("root", "node"):
        data.add_argument(f"--{prefix}-noise-dist", default="gaussian", choices=["gaussian", "laplace", "uniform"])
        data.add_argument(f"--{prefix}-noise-mean", type=float, default=0.0)
        data.add_argument(f"--{prefix}-noise-scale", type=float, default=1.0)
    data.add_argument("--n-samples", type=int, default=500)
    data.add_argument("--weight-min", type=float, default=0.5)
    data.add_argument("--weight-max", type=float, default=2.0)
    data.add_argument("--positive-weights-only", action="store_true")
    data.add_argument("--equal-contributions", action="store_true")
    data.add_argument("--intervention-node", type=int)
    data.add_argument("--intervention-value", type=float, default=1.0)
    seed = data.add_mutually_exclusive_group()
    seed.add_argument("--seed", type=int, default=42)
    seed.add_argument("--no-seed", action="store_true")
    env = parser.add_argument_group("Multiple environments (enabled by --n-envs)")
    env.add_argument("--n-envs", type=int)
    env.add_argument("--n-samples-per-env", type=int, default=500)
    env.add_argument("--no-upstream-shift", action="store_true")
    env.add_argument("--no-soft-interventions", action="store_true")
    env.add_argument("--soft-intervention-scale", type=float, default=0.3)
    env.add_argument("--heteroskedastic-noise", action="store_true")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--prefix", default="causal")
    parser.add_argument("--no-plot", action="store_true")
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    options = vars(args).copy()
    for name in ("output_dir", "prefix", "no_plot", "n_envs", "n_samples_per_env", "no_upstream_shift", "no_soft_interventions", "soft_intervention_scale", "heteroskedastic_noise"):
        options.pop(name)
    options["allow_negative_weights"] = not options.pop("positive_weights_only")
    if options.pop("no_seed"):
        options["seed"] = None
    try:
        config = GeneratorConfig(**options)
        if args.n_envs is None:
            dataset = generate(config)
        else:
            dataset = generate_environments(config, EnvironmentConfig(
                n_envs=args.n_envs, n_samples_per_env=args.n_samples_per_env,
                upstream_shift=not args.no_upstream_shift,
                soft_interventions=not args.no_soft_interventions,
                soft_intervention_scale=args.soft_intervention_scale,
                heteroskedastic_noise=args.heteroskedastic_noise,
            ))
        paths = dataset.save(args.output_dir, args.prefix, include_plot=not args.no_plot)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(dataset.summary())
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
