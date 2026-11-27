"""Publication figures generated only from results/*.json.

Standards: #FFFFFF background, 350 DPI, Times New Roman, a validated colorblind-safe
Okabe-Ito order with redundant line styles and markers, no em or en dashes, and both
PDF and PNG outputs. Every plotted number is read from the benchmark outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# Okabe-Ito order validated with the dataviz palette checker on a white surface
# (adjacent CVD Delta E >= 11). Orange is below 3:1 contrast, so each series also has
# a legend entry, a line style, and a marker, and all values are in the LaTeX tables.
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)))
MARKERS = ("o", "s", "^", "D", "v", "P")
UNSUPERVISED_COLOR = PALETTE[0]
SUPERVISED_COLOR = PALETTE[2]
INK = "#1F1F1F"
MUTED = "#666666"
FAINT = "#9A9A9A"
GRID = "#E6E6E6"
PANEL = "#F7F7F7"
DPI = 350
FORBIDDEN_DASHES = ("—", "–")
FIGURE_NAMES = (
    "fig1_architecture",
    "fig2_detection_roc_pr",
    "fig3_localization_and_context",
    "fig4_ablation_and_robustness",
)
# Diverging scale centred on chance AUROC: vermillion below, neutral grey at 0.5,
# blue above (blue/orange is the colorblind-safe diverging pair).
CHANCE_CMAP = LinearSegmentedColormap.from_list(
    "chance_diverging", ["#D55E00", "#F0D9C8", "#F2F2F2", "#C6DBEF", "#0072B2"]
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "axes.edgecolor": FAINT,
            "axes.linewidth": 0.7,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 8,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _check_text(fig: plt.Figure) -> None:
    for text in fig.findobj(matplotlib.text.Text):
        content = text.get_text()
        if any(dash in content for dash in FORBIDDEN_DASHES):
            raise ValueError(f"Figure text contains an en or em dash: {content!r}")


def _save(fig: plt.Figure, destination: Path) -> None:
    _check_text(fig)
    for suffix in (".pdf", ".png"):
        fig.savefig(
            destination.with_suffix(suffix),
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="#FFFFFF",
            edgecolor="#FFFFFF",
        )
    plt.close(fig)


def _seeds(count: int) -> str:
    return f"{count} training seed" + ("" if count == 1 else "s")


def _display(summary: dict[str, Any], name: str) -> str:
    return summary["models"][name]["display_name"]


def _stat(block: dict[str, Any], key: str) -> tuple[float, float]:
    entry = block["mean_std_across_training_seeds"][key]
    return entry["mean"], entry["std"]


def _group(summary: dict[str, Any], regime: str) -> list[str]:
    return [
        name for name, block in summary["models"].items() if block["supervision_regime"] == regime
    ]


def _regime_color(summary: dict[str, Any], name: str) -> str:
    regime = summary["models"][name]["supervision_regime"]
    return UNSUPERVISED_COLOR if regime == "unsupervised_or_one_class" else SUPERVISED_COLOR


def _panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"({label}) {title}", loc="left", pad=6)


# ---------------------------------------------------------------------------
# Figure 1: layered system architecture
# ---------------------------------------------------------------------------


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    edge: str,
    face: str = "#FFFFFF",
    dashed: bool = False,
    linewidth: float = 0.9,
    radius: float = 0.012,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=linewidth,
            edgecolor=edge,
            facecolor=face,
            linestyle=(0, (3, 2)) if dashed else "-",
            clip_on=False,
        )
    )


def _card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: list[str],
    *,
    edge: str,
    face: str = "#FFFFFF",
    dashed: bool = False,
    title_color: str = INK,
    body_color: str = MUTED,
    size: float = 6.6,
) -> None:
    _box(ax, x, y, width, height, edge=edge, face=face, dashed=dashed)
    ax.text(
        x + 0.008,
        y + height - 0.014,
        title,
        ha="left",
        va="top",
        fontsize=size + 0.6,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + 0.008,
        y + height - 0.05,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=size,
        color=body_color,
        linespacing=1.35,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.8,
            color=INK,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
        )
    )


def _header_card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    step: str,
    title: str,
    lines: list[str],
    color: str,
    *,
    dashed: bool = False,
) -> None:
    """Card with a colored header strip, used for the system overview."""
    _box(ax, x, y, width, height, edge=color, dashed=dashed, linewidth=0.8, radius=0.008)
    strip = 0.052
    ax.add_patch(
        FancyBboxPatch(
            (x, y + height - strip),
            width,
            strip,
            boxstyle="round,pad=0,rounding_size=0.008",
            linewidth=0,
            facecolor=color,
            alpha=0.16 if not dashed else 0.08,
            clip_on=False,
        )
    )
    ax.text(
        x + 0.008,
        y + height - strip / 2,
        f"{step}  {title}" if step else title,
        ha="left",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=INK if not dashed else MUTED,
    )
    ax.text(
        x + 0.008,
        y + height - strip - 0.014,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=5.9,
        color=MUTED if not dashed else FAINT,
        linespacing=1.38,
    )


def _block(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str = "",
    *,
    face: str = "#FFFFFF",
    edge: str = INK,
    size: float = 6.4,
) -> None:
    """Neural-network layer block with a centered name and optional detail line."""
    _box(ax, x, y, width, height, edge=edge, face=face, linewidth=0.8, radius=0.006)
    ax.text(
        x + width / 2,
        y + height / 2 + (0.012 if detail else 0),
        title,
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold",
    )
    if detail:
        ax.text(
            x + width / 2,
            y + height / 2 - 0.022,
            detail,
            ha="center",
            va="center",
            fontsize=size - 1.0,
            color=MUTED,
        )


def _matrix_glyph(
    ax: plt.Axes,
    x: float,
    y: float,
    cell_w: float,
    cell_h: float,
    rows: int,
    columns: int,
    fill: Any,
) -> None:
    for row in range(rows):
        for column in range(columns):
            face, hatch = fill(row, column)
            ax.add_patch(
                Rectangle(
                    (x + column * cell_w, y + (rows - 1 - row) * cell_h),
                    cell_w * 0.9,
                    cell_h * 0.88,
                    facecolor=face,
                    edgecolor="#B5B5B5" if hatch else "none",
                    hatch=hatch,
                    linewidth=0.25,
                    clip_on=False,
                )
            )


def _architecture(figure_dir: Path, summary: dict[str, Any]) -> None:
    splits = summary["official_splits"]
    audit = summary["composite_sender_cluster_audit"]
    bootstrap = summary["uncertainty"]["sender_cluster_bootstrap"]
    seeds = len(summary["protocol"]["training_seeds"])
    epochs = summary["protocol"]["epochs"]
    unsupervised = [
        _display(summary, name) for name in _group(summary, "unsupervised_or_one_class")
    ]
    supervised = [_display(summary, name) for name in _group(summary, "supervised")]
    clusters = audit["unique_composite_clusters_by_split"]

    fig = plt.figure(figsize=(7.16, 5.7))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    # ---------------- (a) system overview ----------------
    ax.text(
        0.005,
        0.985,
        "(a) System overview and evidence boundary",
        fontsize=8.2,
        fontweight="bold",
        va="top",
    )
    top, height = 0.72, 0.235
    columns = [
        (
            "1",
            "Evidence and audit",
            [
                "VeReMi NextGen, InTAS hwy 2",
                "dataReplay and dosAttack",
                "official train / val / test",
                "MD5 and SHA-256 verified",
                "ground-truth cross-check:",
                "  attack type = member paths",
                "  attacker 0 = official state",
            ],
            PALETTE[0],
        ),
        (
            "2",
            "Event parsing",
            [
                "receiver copies collapsed:",
                "  one record per broadcast",
                "  earliest rcvTime (ns)",
                f"train  {splits['train']['records']:,}",
                f"val      {splits['validation']['records']:,}",
                f"test    {splits['test']['records']:,}",
            ],
            PALETTE[0],
        ),
        (
            "3",
            "Causal representation",
            [
                "per-sender window, L = 10",
                "11 translation-invariant",
                "  kinematic and timing features",
                "feature and step masks",
                "clusters: (corridor_id, source_id)",
                f"  test clusters {clusters['test']:,}",
                f"  ID collisions {audit['cross_archive_source_id_collisions_by_split']['test']:,}",
            ],
            PALETTE[0],
        ),
        (
            "4",
            "Detectors",
            [
                f"unsupervised / one-class ({len(unsupervised)})",
                "  robust rules, isolation forest,",
                "  one-class SVM,",
                "  LSTM autoencoder",
                f"supervised ({len(supervised)})",
                "  logistic regression, random forest,",
                "  hist. gradient boosting, MLP,",
                "  Tabular Transformer",
            ],
            SUPERVISED_COLOR,
        ),
        (
            "5",
            "Evaluation",
            [
                f"{_seeds(seeds)}: mean and std",
                f"{bootstrap['requested_replicates']:,} paired cluster bootstrap",
                "FAR budgets 1% and 5%",
                "per-attack slices",
                "LOAO, both directions",
                "sender-disjoint holdout",
                "5 ablations, reproduction",
            ],
            INK,
        ),
    ]
    width, gap, left = 0.183, 0.0145, 0.005
    for index, (step, title, lines, color) in enumerate(columns):
        x = left + index * (width + gap)
        _header_card(ax, x, top, width, height, step, title, lines, color)
        if index < len(columns) - 1:
            _arrow(
                ax,
                (x + width + 0.001, top + height * 0.55),
                (x + width + gap - 0.001, top + height * 0.55),
            )
    band_y, band_h = 0.645, 0.06
    _box(ax, left, band_y, 0.578, band_h, edge=FAINT, dashed=True, linewidth=0.7, radius=0.006)
    ax.text(
        left + 0.008,
        band_y + band_h / 2,
        "Excluded: Tampa CV BSM and SPaT, PeMS-BAY, Caltrans LCS, CISA KEV, MITRE ATT&CK for ICS\n"
        "acquired and audited, but not time-aligned with VeReMi, so never fit or scored",
        fontsize=5.8,
        color=FAINT,
        va="center",
        linespacing=1.3,
    )
    arch_x = left + 0.578 + 0.0145
    _box(
        ax,
        arch_x,
        band_y,
        1 - arch_x - left,
        band_h,
        edge=FAINT,
        dashed=True,
        linewidth=0.7,
        radius=0.006,
    )
    ax.text(
        arch_x + 0.008,
        band_y + band_h / 2,
        "Architecture-only, not evaluated: CorridorLogTransformer,\n"
        "StaticGCN, AsyncSTGNN.  Output: alerts for human review.",
        fontsize=5.8,
        color=FAINT,
        va="center",
        linespacing=1.3,
    )

    # ---------------- (b) Tabular Transformer ----------------
    ax.text(
        0.005,
        0.615,
        "(b) Supervised Tabular Transformer: causal masked encoder for one sender window",
        fontsize=8.2,
        fontweight="bold",
        va="top",
    )
    base, tall = 0.12, 0.43
    mid = base + tall / 2

    # Inputs: window X and mask M glyphs
    rng = np.random.default_rng(11)
    masked_cells = rng.random((10, 11)) < 0.13

    def window_fill(row: int, column: int):
        padded = column < 2
        if padded:
            return "#E6E6E6", None
        if masked_cells[column, row]:
            return "#FFFFFF", "//////"
        return "#9CC3E0", None

    def mask_fill(row: int, column: int):
        padded = column < 2
        if padded or masked_cells[column, row]:
            return "#FFFFFF", None
        return "#4D4D4D", None

    cell_w, cell_h = 0.0072, 0.0115
    _matrix_glyph(ax, 0.018, mid + 0.025, cell_w, cell_h, 11, 10, window_fill)
    ax.text(
        0.018 + 5 * cell_w,
        mid + 0.025 + 11 * cell_h + 0.012,
        r"$X \in \mathbb{R}^{L \times F}$",
        ha="center",
        fontsize=6.6,
    )
    _matrix_glyph(ax, 0.018, mid - 0.025 - 11 * cell_h, cell_w, cell_h, 11, 10, mask_fill)
    ax.text(
        0.018 + 5 * cell_w,
        mid - 0.025 - 11 * cell_h - 0.018,
        r"$M \in \{0,1\}^{L \times F}$",
        ha="center",
        fontsize=6.6,
    )
    ax.text(
        0.018 + 5 * cell_w,
        base - 0.03,
        "L = 10 steps (oldest left),\nF = 11; grey = padded,\nhatched = unobserved",
        ha="center",
        va="top",
        fontsize=5.4,
        color=MUTED,
        linespacing=1.25,
    )

    stages_x = 0.115
    _arrow(ax, (0.094, mid + 0.07), (stages_x, mid + 0.02))
    _arrow(ax, (0.094, mid - 0.07), (stages_x, mid - 0.02))
    _block(
        ax,
        stages_x,
        mid - 0.075,
        0.082,
        0.15,
        "Masked\ninput",
        r"$[X \odot M,\ M]$" + "\n" + r"$L \times 2F$",
        face="#EEF5FB",
        edge=PALETTE[0],
    )
    _arrow(ax, (stages_x + 0.082, mid), (0.213, mid))
    _block(
        ax,
        0.213,
        mid - 0.075,
        0.09,
        0.15,
        "Linear\n2F to d",
        "d = 32\n+ position emb.",
        face="#EEF5FB",
        edge=PALETTE[0],
    )
    _arrow(ax, (0.303, mid), (0.322, mid))

    # Encoder block container
    enc_x, enc_w = 0.322, 0.395
    _box(
        ax,
        enc_x,
        base - 0.005,
        enc_w,
        tall + 0.01,
        edge=SUPERVISED_COLOR,
        face="#F2FAF6",
        linewidth=1.0,
        radius=0.01,
    )
    ax.text(
        enc_x + 0.008,
        base + tall - 0.008,
        "Pre-norm Transformer encoder layer  (x1, dropout 0.1)",
        fontsize=6.4,
        fontweight="bold",
        color=SUPERVISED_COLOR,
        va="top",
    )
    lane = mid - 0.005
    _block(ax, enc_x + 0.012, lane - 0.04, 0.055, 0.08, "Layer\nNorm", size=5.8)
    _arrow(ax, (enc_x + 0.067, lane), (enc_x + 0.08, lane))
    _block(
        ax,
        enc_x + 0.08,
        lane - 0.055,
        0.108,
        0.11,
        "Multi-head\nself-attention",
        "4 heads, causal +\nkey-padding masks",
        face="#FFFFFF",
        edge=SUPERVISED_COLOR,
        size=6.0,
    )
    _arrow(ax, (enc_x + 0.188, lane), (enc_x + 0.2, lane))
    ax.text(enc_x + 0.207, lane, r"$\oplus$", fontsize=10, ha="center", va="center")
    _arrow(ax, (enc_x + 0.214, lane), (enc_x + 0.226, lane))
    _block(ax, enc_x + 0.226, lane - 0.04, 0.055, 0.08, "Layer\nNorm", size=5.8)
    _arrow(ax, (enc_x + 0.281, lane), (enc_x + 0.293, lane))
    _block(
        ax,
        enc_x + 0.293,
        lane - 0.055,
        0.062,
        0.11,
        "FFN",
        "32 to 128\nto 32",
        face="#FFFFFF",
        edge=SUPERVISED_COLOR,
        size=6.0,
    )
    _arrow(ax, (enc_x + 0.355, lane), (enc_x + 0.366, lane))
    ax.text(enc_x + 0.373, lane, r"$\oplus$", fontsize=10, ha="center", va="center")
    # residual connections drawn above the lane
    for start, end in ((enc_x + 0.006, enc_x + 0.207), (enc_x + 0.214, enc_x + 0.373)):
        ax.plot(
            [start, start, end, end],
            [lane, lane + 0.085, lane + 0.085, lane + 0.012],
            color=FAINT,
            linewidth=0.7,
        )
    ax.text(enc_x + 0.1, lane + 0.09, "residual", fontsize=5.4, color=FAINT, ha="center")
    ax.text(enc_x + 0.3, lane + 0.09, "residual", fontsize=5.4, color=FAINT, ha="center")

    # causal attention mask glyph under the attention block
    glyph = 0.0066
    gx, gy = enc_x + 0.095, base + 0.02
    _matrix_glyph(
        ax,
        gx,
        gy,
        glyph,
        glyph * 1.05,
        10,
        10,
        lambda row, column: (
            "#FFFFFF" if column > row or column < 2 else SUPERVISED_COLOR,
            "//////" if column < 2 and column <= row else None,
        ),
    )
    ax.text(
        gx + 0.075,
        gy + 0.05,
        "allowed attention:\nquery step i sees keys\nj ≤ i only; padded\nkeys (hatched) blocked",
        fontsize=5.3,
        color=MUTED,
        va="center",
        linespacing=1.25,
    )
    ax.plot(
        [enc_x + 0.134, enc_x + 0.134],
        [gy + 10 * glyph * 1.05 + 0.004, lane - 0.055],
        color=FAINT,
        linewidth=0.6,
        linestyle=":",
    )

    out_x = enc_x + enc_w
    _arrow(ax, (out_x, mid), (out_x + 0.016, mid))
    _block(
        ax,
        out_x + 0.016,
        mid - 0.075,
        0.078,
        0.15,
        "Last valid\nstep",
        r"$h_{t}$, step mask" + "\nselects it",
        face="#EEF5FB",
        edge=PALETTE[0],
    )
    _arrow(ax, (out_x + 0.094, mid), (out_x + 0.11, mid))
    _block(
        ax,
        out_x + 0.11,
        mid - 0.075,
        0.07,
        0.15,
        "Linear\nd to 1",
        r"$s = \sigma(w^\top h_t)$",
        face="#EEF5FB",
        edge=PALETTE[0],
    )
    _arrow(ax, (out_x + 0.18, mid), (out_x + 0.195, mid))
    _block(
        ax,
        out_x + 0.195,
        mid - 0.075,
        0.083,
        0.15,
        "Alert if\n" + r"$s \geq \tau$",
        r"$\tau$: validation F1" + "\nor FAR budget",
        face="#FFFFFF",
        edge=INK,
    )

    ax.text(
        0.115,
        base - 0.03,
        "Training: BCE with positive weight N(benign) / N(attacker), AdamW (lr 5e-4), "
        f"batch 128, at most {epochs} epoch{'' if epochs == 1 else 's'}, "
        "checkpoint with best validation AUROC (patience 2), "
        f"{_seeds(seeds)}. Robust scaling and thresholds never use test data.",
        fontsize=5.9,
        color=MUTED,
        va="top",
        wrap=True,
    )
    _save(fig, figure_dir / "fig1_architecture")


# ---------------------------------------------------------------------------
# Figure 2: ROC with log FPR and precision-recall, split by regime
# ---------------------------------------------------------------------------


def _curves(figure_dir: Path, summary: dict[str, Any]) -> None:
    seed_count = len(summary["protocol"]["training_seeds"])
    curve_seed = summary["models"][next(iter(summary["models"]))]["curves_training_seed"]
    budgets = summary["protocol"]["far_budgets"]
    groups = (
        ("unsupervised_or_one_class", "Unsupervised or one-class"),
        ("supervised", "Supervised"),
    )
    fig = plt.figure(figsize=(7.16, 7.0))
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.0, 0.2, 1.0, 0.26],
        hspace=0.42,
        wspace=0.22,
        left=0.08,
        right=0.985,
        top=0.965,
        bottom=0.05,
    )
    for row, (regime, title) in enumerate(groups):
        roc_ax = fig.add_subplot(grid[row * 2, 0])
        pr_ax = fig.add_subplot(grid[row * 2, 1])
        legend_ax = fig.add_subplot(grid[row * 2 + 1, :])
        legend_ax.axis("off")
        handles = []
        for index, name in enumerate(_group(summary, regime)):
            block = summary["models"][name]
            auroc, auroc_std = _stat(block, "auroc")
            ap, ap_std = _stat(block, "average_precision")
            style = {
                "color": PALETTE[index % len(PALETTE)],
                "linestyle": LINESTYLES[index % len(LINESTYLES)],
                "marker": MARKERS[index % len(MARKERS)],
                "markevery": 0.14,
                "markersize": 3.6,
                "markeredgecolor": "#FFFFFF",
                "markeredgewidth": 0.5,
            }
            fpr = np.maximum(np.asarray(block["roc"]["fpr"]), 1e-4)
            (line,) = roc_ax.plot(
                fpr,
                block["roc"]["tpr"],
                label=(
                    f"{_display(summary, name)}: AUROC {auroc:.3f} ± {auroc_std:.3f}, "
                    f"AP {ap:.3f} ± {ap_std:.3f}"
                ),
                **style,
            )
            handles.append(line)
            pr_ax.plot(
                block["precision_recall"]["recall"],
                block["precision_recall"]["precision"],
                **style,
            )
        for budget in budgets:
            roc_ax.axvspan(
                1e-4, budget, color=GRID, alpha=0.45 if budget == budgets[0] else 0.22, lw=0
            )
            roc_ax.axvline(budget, color=FAINT, linewidth=0.6, linestyle="--")
            roc_ax.text(budget, 0.03, f" {budget * 100:g}%", fontsize=6, color=MUTED)
        roc_ax.set_xscale("log")
        roc_ax.set(
            xlim=(1e-4, 1),
            ylim=(0, 1.01),
            xlabel="False-positive rate on non-malicious records (log scale)",
            ylabel="True-positive rate",
        )
        pr_ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision")
        _panel_label(roc_ax, "abcd"[row * 2], f"{title}: ROC")
        _panel_label(pr_ax, "abcd"[row * 2 + 1], f"{title}: precision-recall")
        legend_ax.legend(
            handles=handles,
            loc="center",
            ncols=2,
            title=f"{title} (mean ± std over {_seeds(seed_count)})",
            title_fontsize=6.8,
            handlelength=2.8,
            columnspacing=2.0,
        )
    fig.text(
        0.5,
        0.005,
        f"Curves show training seed {curve_seed} on the official VeReMi test split; shaded bands "
        "mark the 1% and 5% false-alarm budgets.",
        ha="center",
        fontsize=6.8,
        color=MUTED,
    )
    _save(fig, figure_dir / "fig2_detection_roc_pr")


# ---------------------------------------------------------------------------
# Figure 3: generalization heatmap and sender ranking
# ---------------------------------------------------------------------------


def _contexts(summary: dict[str, Any]) -> list[tuple[str, dict[str, Any] | None, str | None]]:
    short = {"dataReplay": "Replay", "dosAttack": "DoS"}
    contexts: list[tuple[str, dict[str, Any] | None, str | None]] = [
        ("Official\ntest", summary["models"], None)
    ]
    for archive, block in summary["per_attack_category"].items():
        contexts.append((f"{short.get(archive, archive)}\nslice", block["models"], None))
    generalization = summary.get("generalization", {})
    for direction, block in generalization.get("leave_one_attack_type_out", {}).items():
        source, target = direction.split("->")
        label = f"LOAO\n{short.get(source, source)} to {short.get(target, target)}"
        if block["status"] == "EVALUABLE":
            contexts.append((label, block["models"], None))
        else:
            contexts.append((label, None, block.get("reason")))
    holdout = generalization.get("sender_disjoint_holdout")
    if holdout:
        evaluable = holdout["status"] == "EVALUABLE"
        contexts.append(
            (
                "Sender-\ndisjoint",
                holdout["models"] if evaluable else None,
                None if evaluable else holdout.get("reason"),
            )
        )
    return contexts


def _localization_and_context(figure_dir: Path, summary: dict[str, Any]) -> None:
    names = list(summary["models"])
    contexts = _contexts(summary)
    fig = plt.figure(figsize=(7.16, 4.6))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[2.45, 1.0], height_ratios=[1.0, 0.035], wspace=0.06, hspace=0.12
    )
    heat = fig.add_subplot(grid[0, 0])
    color_axis = fig.add_subplot(grid[1, 0])
    rank = fig.add_subplot(grid[0, 1])

    values = np.full((len(names), len(contexts)), np.nan)
    errors = np.full_like(values, np.nan)
    for column, (_, models, _) in enumerate(contexts):
        if models is None:
            continue
        for row, name in enumerate(names):
            if name in models:
                values[row, column], errors[row, column] = _stat(models[name], "auroc")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    image = heat.imshow(np.ma.masked_invalid(values), cmap=CHANCE_CMAP, norm=norm, aspect="auto")
    heat.grid(False)
    for row in range(len(names)):
        for column, (_, models, _) in enumerate(contexts):
            if models is None:
                heat.text(column, row, "n/e", ha="center", va="center", fontsize=6, color=MUTED)
                continue
            value = values[row, column]
            dark = abs(value - 0.5) > 0.36
            heat.text(
                column,
                row,
                f"{value:.3f}\n±{errors[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=6.9,
                color="#FFFFFF" if dark else INK,
                linespacing=1.05,
            )
    boundary = len(_group(summary, "unsupervised_or_one_class")) - 0.5
    heat.axhline(boundary, color=INK, linewidth=1.1)
    heat.axvline(0.5, color="#FFFFFF", linewidth=2.0)
    heat.axvline(2.5, color="#FFFFFF", linewidth=2.0)
    heat.set_xticks(range(len(contexts)), [label for label, _, _ in contexts], fontsize=7)
    heat.xaxis.tick_top()
    heat.tick_params(axis="x", length=0, pad=2)
    heat.set_yticks(range(len(names)), [_display(summary, name) for name in names], fontsize=6.8)
    heat.tick_params(axis="y", length=0)
    for label, name in zip(heat.get_yticklabels(), names, strict=True):
        label.set_color(_regime_color(summary, name))
    for spine in heat.spines.values():
        spine.set_visible(False)
    heat.set_title(
        "(a) Test AUROC across evaluation contexts (mean ± std over seeds)",
        loc="left",
        pad=26,
    )
    colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    colorbar.ax.tick_params(labelsize=6)
    colorbar.outline.set_linewidth(0.4)
    colorbar.set_label("Test AUROC (0.5 = chance; below 0.5 = inverted ranking)", fontsize=6.5)

    positions = np.arange(len(names))
    for metric, offset, marker, label in (
        ("attacker_emitter_recall_at_1", -0.17, "o", "k = 1"),
        ("attacker_emitter_recall_at_3", 0.17, "s", "k = 3"),
    ):
        for row, name in enumerate(names):
            stats = summary["models"][name]["emitter_ranking_and_simulated_latency"][
                "mean_std_across_training_seeds"
            ]
            entry = stats.get(metric)
            if entry is None:
                continue
            rank.errorbar(
                100 * entry["mean"],
                row + offset,
                xerr=100 * entry["std"],
                fmt=marker,
                color=_regime_color(summary, name),
                markerfacecolor=_regime_color(summary, name) if marker == "s" else "#FFFFFF",
                markersize=3.8,
                elinewidth=0.8,
                capsize=1.5,
                label=label if row == 0 else None,
            )
    rank.set_yticks(positions, [])
    rank.set_ylim(len(names) - 0.5, -0.5)
    rank.axhline(boundary, color=INK, linewidth=0.8)
    rank.set_xlim(left=0)
    rank.set_xlabel("Attacker-emitter recall (%)")
    rank.set_title("(b) Retrospective sender ranking", loc="left", pad=26)
    rank.grid(axis="y", visible=False)
    legend = rank.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=2,
        handletextpad=0.3,
        title="top-k senders",
        title_fontsize=6.4,
    )
    for handle in legend.legend_handles:
        handle.set_color(MUTED)
    fig.text(
        0.01,
        -0.05,
        "Blue labels: unsupervised or one-class; green labels: supervised. LOAO trains on one "
        "attack archive and tests on the other; n/e marks NOT_EVALUABLE contexts.",
        fontsize=6.6,
        color=MUTED,
    )
    _save(fig, figure_dir / "fig3_localization_and_context")


# ---------------------------------------------------------------------------
# Figure 4: ablations and paired bootstrap differences
# ---------------------------------------------------------------------------

ABLATION_LABELS = {
    "window_length": "Window L = 1",
    "coordinate_representation": "Raw coordinates",
    "timing_and_jitter": "No timing or jitter",
    "local_density": "No local density",
    "validity_masks": "No validity masks",
}


def _paired_seed_delta(
    variant_rows: list[dict[str, Any]], full_rows: list[dict[str, Any]], key: str
) -> tuple[float, float]:
    full = {row["seed"]: row[key] for row in full_rows}
    deltas = np.asarray(
        [row[key] - full[row["seed"]] for row in variant_rows if row["seed"] in full]
    )
    return float(deltas.mean()), float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0


def _forest(
    ax: plt.Axes,
    rows: list[tuple[str, float, float, float, str, str]],
    xlabel: str,
) -> None:
    for index, (_, center, low, high, color, marker) in enumerate(rows):
        ax.plot([low, high], [index, index], color=color, linewidth=1.6, solid_capstyle="round")
        ax.plot(
            center,
            index,
            marker=marker,
            color=color,
            markersize=4.2,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.6,
        )
    ax.axvline(0, color=INK, linewidth=0.7)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def _ablation_and_robustness(
    figure_dir: Path, summary: dict[str, Any], ablation: dict[str, Any] | None
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0))
    model_styles = {}
    if ablation:
        families = list(ablation["ablations"])
        models = ablation["models"]
        for name in models:
            model_styles[name] = (
                (SUPERVISED_COLOR, "o") if name == "tabular_transformer" else (PALETTE[1], "s")
            )
        for axis, key, _title, label in (
            (
                axes[0, 0],
                "auroc",
                "Ablation: AUROC change",
                "ΔAUROC (variant minus full), paired by seed",
            ),
            (
                axes[0, 1],
                "test_roc_tpr_at_far_1.0pct",
                "Ablation: TPR at 1% FAR change",
                "ΔTPR at FAR 1% (variant minus full)",
            ),
        ):
            offsets = np.linspace(-0.16, 0.16, len(models))
            for model_index, name in enumerate(models):
                color, marker = model_styles[name]
                for family_index, family in enumerate(families):
                    result = ablation["ablations"][family]["models"][name]
                    mean, std = _paired_seed_delta(
                        result["per_seed"], summary["models"][name]["per_seed"], key
                    )
                    axis.errorbar(
                        mean,
                        family_index + offsets[model_index],
                        xerr=std,
                        fmt=marker,
                        color=color,
                        markersize=4,
                        markeredgecolor="#FFFFFF",
                        markeredgewidth=0.5,
                        elinewidth=1.0,
                        capsize=1.8,
                        label=_display(summary, name) if family_index == 0 else None,
                    )
            axis.axvline(0, color=INK, linewidth=0.7)
            axis.set_yticks(range(len(families)), [ABLATION_LABELS.get(f, f) for f in families])
            axis.set_ylim(len(families) - 0.5, -0.5)
            axis.set_xlabel(label, fontsize=7)
            axis.grid(axis="y", visible=False)
        _panel_label(axes[0, 0], "a", "Ablations: ΔAUROC")
        _panel_label(axes[0, 1], "b", "Ablations: ΔTPR at 1% FAR")
        axes[0, 1].set_yticklabels([])
        axes[0, 0].legend(loc="lower left", handletextpad=0.3)
    else:
        for axis in axes[0]:
            axis.text(0.5, 0.5, "Ablations not run", ha="center", transform=axis.transAxes)

    bootstrap = summary["uncertainty"]["sender_cluster_bootstrap"]
    deltas = bootstrap["paired_differences_reference_minus_model"]
    names = [name for name in summary["models"] if name in deltas]
    for axis, metric, label, panel in (
        (axes[1, 0], "delta_auroc", "ΔAUROC, Tabular Transformer minus model", "c"),
        (axes[1, 1], "delta_f1_cyber_vs_benign", "ΔF1, Tabular Transformer minus model", "d"),
    ):
        rows = []
        for name in names:
            block = deltas[name][metric]
            color = _regime_color(summary, name)
            marker = "o" if color == UNSUPERVISED_COLOR else "s"
            rows.append((_display(summary, name), block["mean"], *block["ci_95pct"], color, marker))
        _forest(axis, rows, label)
        for tick, name in zip(axis.get_yticklabels(), names, strict=True):
            tick.set_color(_regime_color(summary, name))
        _panel_label(axis, panel, f"Bootstrap: {label.split(',')[0]} 95% CI")
    axes[1, 1].set_yticklabels([])
    fig.text(
        0.01,
        -0.02,
        f"(a, b) mean ± std of per-seed paired differences over "
        f"{_seeds(len(summary['protocol']['training_seeds']))}. "
        "\n(c, d) percentile 95% intervals from "
        f"{bootstrap['valid_replicates']:,} valid of {bootstrap['requested_replicates']:,} paired "
        f"resamples of {bootstrap['composite_clusters']:,} composite sender clusters; "
        "positive values "
        "favor the Tabular Transformer.",
        fontsize=6.6,
        color=MUTED,
        wrap=True,
    )
    fig.tight_layout(h_pad=1.8, w_pad=1.2)
    _save(fig, figure_dir / "fig4_ablation_and_robustness")


def generate_figures(
    results_dir: Path = Path("results"), figure_dir: Path = Path("paper/figures")
) -> list[Path]:
    _style()
    results_dir = Path(results_dir)
    figure_dir = Path(figure_dir)
    summary = json.loads((results_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    ablation_path = results_dir / "ablation_summary.json"
    ablation = (
        json.loads(ablation_path.read_text(encoding="utf-8")) if ablation_path.exists() else None
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    _architecture(figure_dir, summary)
    _curves(figure_dir, summary)
    _localization_and_context(figure_dir, summary)
    _ablation_and_robustness(figure_dir, summary, ablation)
    return [figure_dir / f"{name}{suffix}" for name in FIGURE_NAMES for suffix in (".pdf", ".png")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate publication figures.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    for path in generate_figures(args.results, args.figures):
        print(path)


if __name__ == "__main__":
    main()
