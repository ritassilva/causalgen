from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .config import GeneratorConfig
from .mechanisms import _mechanism_name, _callable_name, _json_safe

@dataclass
class CausalDataset:
    dag:              np.ndarray
    confounders:      List[int]
    full_observational: pd.DataFrame
    full_interventional: Optional[pd.DataFrame]
    weight_matrix:    np.ndarray
    config:           GeneratorConfig
    metadata:         Dict

    @property
    def data(self) -> pd.DataFrame:
        """Observed data, with latent variables excluded."""
        return self.observational_df()

    @property
    def observational(self) -> pd.DataFrame:
        return self.observational_df()

    @property
    def interventional(self) -> Optional[pd.DataFrame]:
        return self.interventional_df()

    @property
    def adjacency(self) -> pd.DataFrame:
        """Observed-node adjacency; this is not a latent projection."""
        return self.adjacency_df()

    def observed_columns(self) -> List[str]:
        """Columns present in the output (respects latent_confounders)."""
        if self.config.n_latent_confounders > 0:
            drop = {f"X{i}" for i in self.confounders[:self.config.n_latent_confounders]}
            return [c for c in self.full_observational.columns if c not in drop]
        return list(self.full_observational.columns)

    def observational_df(self) -> pd.DataFrame:
        return self.full_observational[self.observed_columns()]

    def interventional_df(self) -> Optional[pd.DataFrame]:
        if self.full_interventional is None:
            return None
        return self.full_interventional[self.observed_columns()]

    def adjacency_df(self) -> pd.DataFrame:
        n = self.dag.shape[0]
        cols = [f"X{i}" for i in range(n)]
        df = pd.DataFrame(self.dag, index=cols, columns=cols)
        obs_final = [col for col in self.observed_columns() if col != "env"]
        return df.loc[obs_final, obs_final]

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def save(self, output_dir: Union[str, Path] = ".", prefix: str = "causal", *, include_plot: bool = True) -> Dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}

        # Observational CSV
        p = out / f"{prefix}_observational.csv"
        self.observational_df().to_csv(p, index=False)
        paths["observational"] = p

        # Interventional CSV
        if self.interventional_df() is not None:
            p = out / f"{prefix}_interventional.csv"
            self.interventional_df().to_csv(p, index=False)
            paths["interventional"] = p

        # Adjacency matrix CSV
        p = out / f"{prefix}_adjacency.csv"
        self.adjacency_df().to_csv(p)
        paths["adjacency"] = p

        # Edge list CSV
        edges = [(f"X{i}", f"X{j}") for i in range(self.dag.shape[0])
                 for j in range(self.dag.shape[0]) if self.dag[i, j]]
        p = out / f"{prefix}_edge_list.csv"
        pd.DataFrame(edges, columns=["source", "target"]).to_csv(p, index=False)
        paths["edge_list"] = p

        # Metadata JSON
        p = out / f"{prefix}_metadata.json"
        with open(p, "w") as f:
            json.dump(_json_safe(self.metadata), f, indent=2)
        paths["metadata"] = p

        # DAG visualisation
        p = out / f"{prefix}_dag.png"
        if include_plot:
            figure = self.plot_dag(save_path=p, show=False)
            plt.close(figure)
            paths["dag_plot"] = p

        return paths

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_dag(
        self,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        G = nx.from_numpy_array(self.dag, create_using=nx.DiGraph)
        mapping = {i: f"X{i}" for i in range(self.dag.shape[0])}
        G = nx.relabel_nodes(G, mapping)

        conf_labels = {f"X{i}" for i in self.confounders}
        latent_labels = {f"X{i}" for i in self.confounders[:self.config.n_latent_confounders]}

        node_colors = []
        node_styles = []
        for node in G.nodes():
            if node in latent_labels:
                node_colors.append("#CECBF6")   # purple – latent confounder
                node_styles.append("dashed")
            elif node in conf_labels:
                node_colors.append("#F5C4B3")   # coral – observed confounder
                node_styles.append("solid")
            else:
                node_colors.append("#B5D4F4")   # blue – regular node
                node_styles.append("solid")

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#F8F8F6")
        fig.patch.set_facecolor("#F8F8F6")

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=self.config.seed or 0)

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color="#888780",
            arrows=True,
            arrowsize=14,
            width=1.2,
            connectionstyle="arc3,rad=0.05",
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=900,
            linewidths=1.5,
            edgecolors=["#7F77DD" if s == "dashed" else "#378ADD" if c == "#B5D4F4" else "#D85A30"
                        for s, c in zip(node_styles, node_colors)],
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="#1a1a1a")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#B5D4F4", edgecolor="#378ADD", label="Observed node")]
        if self.config.n_latent_confounders > 0:
            legend_elements.append(Patch(facecolor="#CECBF6", edgecolor="#7F77DD", label="Latent confounder"))
        if self.config.n_observed_confounders > 0:
            legend_elements.append(Patch(facecolor="#F5C4B3", edgecolor="#D85A30", label="Observed confounder"))
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.8)

        ax.set_title(
            f"Causal DAG  |  {self.dag.shape[0]} nodes  |  "
            f"{int(self.dag.sum())} edges  |  "
            f"{len(self.confounders)} confounder(s)",
            fontsize=10, pad=10
        )
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def summary(self) -> str:
        n = self.dag.shape[0]
        edges = int(self.dag.sum())
        conf_str = ", ".join(f"X{i}" for i in self.confounders) or "none"
        overrides = (
            ", ".join(
                f"X{i}={_mechanism_name(form)}"
                for i, form in self.config.node_mechanisms.items()
            )
            if self.config.node_mechanisms else "none"
        )
        mechanism = _mechanism_name(self.config.mechanism)
        root_noise = (
            _callable_name(self.config.root_noise_sampler)
            if self.config.root_noise_sampler is not None
            else self.config.root_noise_dist
        )
        node_noise = (
            _callable_name(self.config.node_noise_sampler)
            if self.config.node_noise_sampler is not None
            else self.config.node_noise_dist
        )
        lines = [
            "=" * 52,
            "  Causal Dataset Summary",
            "=" * 52,
            f"  Graph type      : {self.config.graph_type}",
            f"  Nodes           : {n}",
            f"  Edges           : {edges}",
            f"  Avg out-degree  : {edges/n:.2f}",
            f"  Confounders     : {conf_str}",
            f"  Latent conf.    : {self.config.n_latent_confounders}  ({', '.join(f'X{i}' for i in self.confounders[:self.config.n_latent_confounders]) or 'none'})",
            f"  Observed conf.  : {self.config.n_observed_confounders}  ({', '.join(f'X{i}' for i in self.confounders[self.config.n_latent_confounders:]) or 'none'})",
            f"  Mechanism       : {mechanism}",
            f"  Node overrides  : {overrides}",
            f"  Root noise      : {root_noise}  (mean={self.config.root_noise_mean}, scale={self.config.root_noise_scale})",
            f"  Node noise      : {node_noise}  (mean={self.config.node_noise_mean}, scale={self.config.node_noise_scale})",
            f"  Samples         : {len(self.full_observational)}",
            f"  Observed cols   : {len(self.observed_columns())}",
        ]
        if self.full_interventional is not None:
            lines.append(
                f"  Intervention    : do(X{self.config.intervention_node} = "
                f"{self.config.intervention_value})"
            )
        lines += [
            f"  Seed            : {self.config.seed}",
            "=" * 52,
        ]
        return "\n".join(lines)


