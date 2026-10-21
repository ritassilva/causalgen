"""Generate observational data and data under a hard intervention."""
from causalgen import GeneratorConfig, generate


def main():
    dataset = generate(GeneratorConfig(
        graph_type="chain", n_nodes=5, mechanism="linear",
        intervention_node=1, intervention_value=2.0,
        n_samples=500, seed=42,
    ))

    # do(X1=2) replaces X1's equation, ignoring its incoming edge. Its outgoing
    # effects on X2, X3 and X4 remain active.
    print("Observational means:", dataset.data.mean())
    print("Interventional means:", dataset.interventional.mean())
    print("Values of X1 under intervention:", dataset.interventional["X1"].unique())

    # Weights and mechanisms are shared, but noise is drawn again. Matching
    # row numbers do not identify paired counterfactual observations.
    # dataset.dag remains the original graph before the intervention.


if __name__ == "__main__":
    main()
