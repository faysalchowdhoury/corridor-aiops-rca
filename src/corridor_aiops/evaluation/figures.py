from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["crimson", "royalblue", "mediumpurple", "lightseagreen"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "font.size": 9,
        }
    )


def _save(fig: plt.Figure, destination: Path) -> None:
    fig.savefig(destination.with_suffix(".pdf"), dpi=350, bbox_inches="tight", facecolor="#FFFFFF")
    fig.savefig(destination.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def generate_figures(
    results_dir: Path = Path("results"), figure_dir: Path = Path("paper/figures")
) -> None:
    _style()
    summary = json.loads((results_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    labels = [
        "Published data\nVeReMi | USDOT\nPeMS | Caltrans",
        "Parse and validate\nPydantic event schema",
        "Replay adapters\nVissim | SUMO\nCARLA / ROS 2",
        "Anomaly scoring\n3 implemented baselines",
        "Human review\nNo automated control",
    ]
    positions = [0.02, 0.22, 0.42, 0.62, 0.82]
    box_width = 0.15
    for index, (x, label) in enumerate(zip(positions, labels, strict=True)):
        ax.text(
            x + box_width / 2,
            0.58,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.5",
                "fc": "#FFFFFF",
                "ec": COLORS[index % len(COLORS)],
            },
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(positions[index + 1] - 0.005, 0.58),
                xytext=(x + box_width + 0.005, 0.58),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
            )
    ax.text(
        0.5,
        0.19,
        "Data sources are not synchronized; the benchmark uses only official VeReMi splits.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, figure_dir / "fig1_architecture")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for index, (name, model) in enumerate(summary["models"].items()):
        color = COLORS[index % len(COLORS)]
        roc = model["roc"]
        axes[0].plot(roc["fpr"], roc["tpr"], label=f"{name} ({model['auroc']:.2f})", color=color)
        pr = model["precision_recall"]
        axes[1].plot(pr["recall"], pr["precision"], label=name, color=color)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[0].legend(fontsize=6, loc="lower right")
    axes[1].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    _save(fig, figure_dir / "fig2_detection_roc_pr")

    with (results_dir / "seed_metrics.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    models = list(summary["models"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for axis, metric, title in zip(
        axes,
        ("false_alarm_rate_benign", "f1_cyber_vs_benign"),
        ("False alarm rate on non-malicious VeReMi records", "Cyber/misbehavior F1"),
        strict=True,
    ):
        grouped = [
            [
                float(row[metric])
                for row in rows
                if row["model"] == model and row[metric] not in ("", "nan", "None")
            ]
            for model in models
        ]
        axis.boxplot(grouped, showfliers=False, patch_artist=True)
        for index, values in enumerate(grouped):
            jitter = np.linspace(-0.07, 0.07, len(values))
            axis.scatter(
                index + 1 + jitter,
                values,
                s=8,
                color=COLORS[index % len(COLORS)],
                alpha=0.75,
                zorder=3,
            )
        axis.set_xticks(range(1, len(models) + 1), models)
        axis.set_title(title)
        axis.set_ylabel("Rate")
        axis.tick_params(axis="x", labelrotation=25, labelsize=7)
        values = [value for group in grouped for value in group]
        padding = max((max(values) - min(values)) * 0.1, 0.01)
        axis.set_ylim(max(0, min(values) - padding), min(1.03, max(values) + padding))
    axes[0].text(
        0.5,
        -0.43,
        "Work-zone FAR is unmeasured: current Caltrans closures do not align with VeReMi.",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=7,
    )
    fig.tight_layout()
    _save(fig, figure_dir / "fig3_workzone_false_alarms")

    feature_differences = summary["standardized_feature_mean_difference"]
    features = list(feature_differences)
    attribution = np.asarray([list(feature_differences.values())], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    image = ax.imshow(attribution, cmap="Purples", aspect="auto")
    ax.set_xticks(range(len(features)), features, rotation=20, ha="right")
    ax.set_yticks([0], ["attacker vs. benign"])
    ax.set_title("Standardized feature separation (not SHAP or causal evidence)")
    fig.colorbar(image, ax=ax, label="absolute standardized mean difference")
    fig.tight_layout()
    _save(fig, figure_dir / "fig4_shap_causal_attribution")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    generate_figures(args.results, args.figures)


if __name__ == "__main__":
    main()
