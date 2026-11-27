"""LaTeX tables generated from the machine-readable benchmark outputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

GROUPS = (
    ("unsupervised_or_one_class", "Unsupervised / one-class (fit on non-malicious records)"),
    ("supervised", "Supervised (fit on training labels)"),
)


def scientific(value: float) -> str:
    """Compact scientific notation for small differences; an exact zero prints as 0."""
    return "0" if value == 0 else f"{value:.1e}"


def _escape(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def _ms(stats: dict[str, Any], key: str, scale: float = 1.0, digits: int = 3) -> str:
    entry = stats.get(key)
    if entry is None:
        return "n/a"
    return f"{scale * entry['mean']:.{digits}f} $\\pm$ {scale * entry['std']:.{digits}f}"


def _ci(interval: list[float] | None, scale: float = 1.0, digits: int = 3) -> str:
    if interval is None:
        return "n/a"
    return f"[{scale * interval[0]:.{digits}f}, {scale * interval[1]:.{digits}f}]"


def _grouped(models: dict[str, Any]) -> list[tuple[str, list[str]]]:
    return [
        (title, [name for name, block in models.items() if block["supervision_regime"] == group])
        for group, title in GROUPS
    ]


def _write(path: Path, header: str, columns: int, body: list[str]) -> None:
    lines = [
        f"\\begin{{tabular}}{{{header}}}",
        "\\toprule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _main_table(summary: dict[str, Any]) -> list[str]:
    models = summary["models"]
    body = [
        "Model & F1 & AUROC & AP & FAR (\\%) \\\\",
    ]
    for title, names in _grouped(models):
        body += ["\\midrule", f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for name in names:
            stats = models[name]["mean_std_across_training_seeds"]
            body.append(
                f"{_escape(models[name]['display_name'])} & "
                f"{_ms(stats, 'f1_cyber_vs_benign')} & {_ms(stats, 'auroc')} & "
                f"{_ms(stats, 'average_precision')} & "
                f"{_ms(stats, 'false_alarm_rate_non_malicious', 100, 2)} \\\\"
            )
    return body


def _operating_table(summary: dict[str, Any]) -> list[str]:
    models = summary["models"]
    body = [
        "& \\multicolumn{2}{c}{Test ROC TPR} & \\multicolumn{2}{c}{Val.-calibrated recall "
        "(realized FAR)} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Model & FAR 1\\% & FAR 5\\% & budget 1\\% & budget 5\\% \\\\",
    ]
    for title, names in _grouped(models):
        body += ["\\midrule", f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for name in names:
            stats = models[name]["mean_std_across_training_seeds"]
            cells = [
                _ms(stats, "test_roc_tpr_at_far_1.0pct"),
                _ms(stats, "test_roc_tpr_at_far_5.0pct"),
            ]
            for budget in ("1.0", "5.0"):
                recall = stats[f"validation_calibrated_recall_at_far_{budget}pct"]["mean"]
                far = stats[f"validation_calibrated_realized_far_at_far_{budget}pct"]["mean"]
                cells.append(f"{recall:.3f} ({100 * far:.2f}\\%)")
            body.append(f"{_escape(models[name]['display_name'])} & " + " & ".join(cells) + " \\\\")
    return body


def _bootstrap_table(summary: dict[str, Any]) -> list[str]:
    models = summary["models"]
    bootstrap = summary["uncertainty"]["sender_cluster_bootstrap"]
    intervals = bootstrap["intervals_95pct"]
    deltas = bootstrap["paired_differences_reference_minus_model"]
    body = [
        "Model & F1 95\\% CI & AUROC 95\\% CI & "
        "$\\Delta$F1 95\\% CI & $\\Delta$AUROC 95\\% CI \\\\",
    ]
    for title, names in _grouped(models):
        body += ["\\midrule", f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for name in names:
            delta = deltas.get(name)
            body.append(
                f"{_escape(models[name]['display_name'])} & "
                f"{_ci(intervals[name]['f1_cyber_vs_benign'])} & "
                f"{_ci(intervals[name]['auroc'])} & "
                f"{_ci(delta['delta_f1_cyber_vs_benign']['ci_95pct']) if delta else 'reference'} & "
                f"{_ci(delta['delta_auroc']['ci_95pct']) if delta else 'reference'} \\\\"
            )
    return body


def generalization_focus_models(summary: dict[str, Any]) -> list[str]:
    """Models shown in the manuscript generalization table, chosen without test data.

    The unsupervised model and the two supervised baselines with the highest mean
    validation AUROC, plus the Tabular Transformer; all models stay in the JSON/CSV.
    """
    models = summary["models"]

    def validation_auroc(name: str) -> float:
        return models[name]["mean_std_across_training_seeds"]["validation_auroc"]["mean"]

    unsupervised = [n for n, b in models.items() if b["supervision_regime"] != "supervised"]
    supervised = [
        n
        for n, b in models.items()
        if b["supervision_regime"] == "supervised" and n != "tabular_transformer"
    ]
    chosen = {max(unsupervised, key=validation_auroc)}
    chosen.update(sorted(supervised, key=validation_auroc, reverse=True)[:2])
    if "tabular_transformer" in models:
        chosen.add("tabular_transformer")
    return [name for name in models if name in chosen]


def _generalization_table(summary: dict[str, Any]) -> list[str]:
    focus = generalization_focus_models(summary)
    body = ["Evaluation & Model & F1 & AUROC & TPR@FAR 1\\% \\\\"]
    rows: list[tuple[str, dict[str, Any] | None, str | None]] = []
    for archive, block in summary["per_attack_category"].items():
        rows.append((f"Per-attack test: {archive}", block["models"], None))
    generalization = summary.get("generalization", {})
    for direction, block in generalization.get("leave_one_attack_type_out", {}).items():
        label = "LOAO " + direction.replace("->", " $\\rightarrow$ ")
        if block["status"] == "EVALUABLE":
            rows.append((label, block["models"], None))
        else:
            rows.append((label, None, block["reason"]))
    holdout = generalization.get("sender_disjoint_holdout")
    if holdout:
        if holdout["status"] == "EVALUABLE":
            rows.append(("Sender-disjoint holdout", holdout["models"], None))
        else:
            rows.append(("Sender-disjoint holdout", None, holdout["reason"]))
    for label, models, reason in rows:
        body.append("\\midrule")
        if models is None:
            body.append(
                f"{label} & \\multicolumn{{4}}{{l}}{{NOT\\_EVALUABLE: {_escape(reason)}}} \\\\"
            )
            continue
        shown = [(name, block) for name, block in models.items() if name in focus]
        for index, (name, block) in enumerate(shown):
            stats = block["mean_std_across_training_seeds"]
            body.append(
                f"{label if index == 0 else ''} & "
                f"{_escape(summary['models'][name]['display_name'])} & "
                f"{_ms(stats, 'f1_cyber_vs_benign')} & {_ms(stats, 'auroc')} & "
                f"{_ms(stats, 'test_roc_tpr_at_far_1.0pct')} \\\\"
            )
    return body


def _ablation_table(summary: dict[str, Any], ablation: dict[str, Any]) -> list[str]:
    labels = {
        "window_length": "Window $L=1$",
        "coordinate_representation": "Raw coordinates",
        "timing_and_jitter": "No timing or jitter",
        "local_density": "No local density",
        "validity_masks": "No validity masks",
    }
    body = [
        "Ablation & Model & F1 & AUROC & $\\Delta$AUROC & $\\Delta$TPR@FAR 1\\% \\\\",
    ]
    full_names = summary["models"]
    for family, block in ablation["ablations"].items():
        body.append("\\midrule")
        for index, (name, result) in enumerate(block["models"].items()):
            stats = result["variant_mean_std_across_training_seeds"]
            delta = result["variant_minus_full_mean"]
            label = labels.get(family, _escape(family)) if index == 0 else ""
            body.append(
                f"{label} & {_escape(full_names[name]['display_name'])} & "
                f"{_ms(stats, 'f1_cyber_vs_benign')} & {_ms(stats, 'auroc')} & "
                f"{delta['auroc']:+.3f} & {delta['test_roc_tpr_at_far_1.0pct']:+.3f} \\\\"
            )
    return body


def write_tables(
    summary: dict[str, Any], ablation: dict[str, Any] | None, output_dir: Path
) -> None:
    output_dir = Path(output_dir)
    _write(output_dir / "results_table.tex", "lcccc", 5, _main_table(summary))
    _write(output_dir / "results_operating_points.tex", "lcccc", 5, _operating_table(summary))
    _write(output_dir / "results_bootstrap.tex", "lcccc", 5, _bootstrap_table(summary))
    if summary.get("generalization"):
        _write(
            output_dir / "results_generalization.tex",
            "llccc",
            5,
            _generalization_table(summary),
        )
    if ablation is not None:
        _write(output_dir / "results_ablation.tex", "llcccc", 6, _ablation_table(summary, ablation))
    write_macros(summary, ablation, output_dir / "results_macros.tex")


MACRO_MODEL_NAMES = {
    "robust_threshold_rules": "RobustRules",
    "isolation_forest": "IsolationForest",
    "one_class_svm": "OneClassSVM",
    "lstm_autoencoder": "LSTMAE",
    "logistic_regression": "LogReg",
    "random_forest": "RandomForest",
    "hist_gradient_boosting": "HGB",
    "xgboost": "XGBoost",
    "supervised_mlp": "MLP",
    "tabular_transformer": "TabTransformer",
}
MACRO_METRICS = {
    "f1_cyber_vs_benign": ("FOne", 1.0, 3),
    "auroc": ("AUROC", 1.0, 3),
    "average_precision": ("AP", 1.0, 3),
    "false_alarm_rate_non_malicious": ("FAR", 100.0, 2),
    "test_roc_tpr_at_far_1.0pct": ("TPRatOne", 1.0, 3),
    "test_roc_tpr_at_far_5.0pct": ("TPRatFive", 1.0, 3),
}
CONTEXT_NAMES = {"dataReplay": "Replay", "dosAttack": "DoS"}


def _macro_value(stats: dict[str, Any], key: str, scale: float, digits: int) -> str:
    entry = stats[key]
    return f"{scale * entry['mean']:.{digits}f} $\\pm$ {scale * entry['std']:.{digits}f}"


LOAO_NEAR_CHANCE_BOUND = 0.6
CURVE_FPR_POINT = 1e-3
CURVE_TPR_ONSET = 0.1
CURVE_RECALL_POINT = 0.9


def write_text_macros(
    summary: dict[str, Any],
    ablation: dict[str, Any] | None,
    capability: dict[str, Any],
    path: Path,
) -> None:
    """Numbers quoted in running text that are not cells of a generated table.

    Kept separate from ``results_macros.tex`` so the benchmark outputs compared by the
    reproduction audit stay unchanged. Curve readings use the stored first-seed curves.
    """
    lines = ["% Generated by corridor_aiops.evaluation.tables; do not edit by hand."]

    def define(name: str, value: Any) -> None:
        if not name.isalpha():
            raise ValueError(f"Invalid LaTeX macro name: {name}")
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def count(value: int) -> str:
        return f"{value:,}".replace(",", "{,}")

    models = summary["models"]
    supervised = [n for n, b in models.items() if b["supervision_regime"] == "supervised"]

    def mean(name: str, key: str, block: dict[str, Any] | None = None) -> float:
        source = block if block is not None else models
        return source[name]["mean_std_across_training_seeds"][key]["mean"]

    aurocs = [mean(n, "auroc") for n in supervised]
    define("ResSupAUROCMin", f"{min(aurocs):.3f}")
    define("ResSupAUROCMax", f"{max(aurocs):.3f}")

    key = "validation_calibrated_{}_at_far_1.0pct"
    recalls = [mean(n, key.format("recall")) for n in supervised]
    fars = [mean(n, key.format("realized_far")) for n in supervised]
    define("ResSupCalRecallMin", f"{min(recalls):.3f}")
    define("ResSupCalRecallMax", f"{max(recalls):.3f}")
    define("ResSupCalFARMin", f"{100 * min(fars):.2f}")
    define("ResSupCalFARMax", f"{100 * max(fars):.2f}")
    if "tabular_transformer" in models:
        define("ResTTCalRecallOne", f"{mean('tabular_transformer', key.format('recall')):.3f}")
        define(
            "ResTTCalFAROne",
            f"{100 * mean('tabular_transformer', key.format('realized_far')):.2f}",
        )

    def roc(name: str) -> tuple[list[float], list[float]]:
        return models[name]["roc"]["fpr"], models[name]["roc"]["tpr"]

    if "one_class_svm" in models:
        fpr, tpr = roc("one_class_svm")
        define("ResCurveFPRPoint", f"{CURVE_FPR_POINT:g}")
        define(
            "ResCurveOCSVMTPR",
            f"{max(t for f, t in zip(fpr, tpr, strict=True) if f <= CURVE_FPR_POINT):.2f}",
        )
    if "lstm_autoencoder" in models:
        fpr, tpr = roc("lstm_autoencoder")
        define("ResCurveTPROnset", f"{CURVE_TPR_ONSET:g}")
        define(
            "ResCurveLSTMAEOnsetFPR",
            f"{100 * min(f for f, t in zip(fpr, tpr, strict=True) if t >= CURVE_TPR_ONSET):.1f}",
        )
    precisions = []
    for name in supervised:
        curve = models[name]["precision_recall"]
        precisions.append(
            min(
                p
                for p, r in zip(curve["precision"], curve["recall"], strict=True)
                if r <= CURVE_RECALL_POINT
            )
        )
    define("ResCurveRecallPoint", f"{CURVE_RECALL_POINT:g}")
    define("ResCurveSupMinPrecision", f"{min(precisions):.3f}")

    loao = summary.get("generalization", {}).get("leave_one_attack_type_out", {})
    pairs = [
        mean(name, "auroc", block["models"])
        for block in loao.values()
        if block.get("status") == "EVALUABLE"
        for name in supervised
    ]
    define("ResLOAONearChanceBound", f"{LOAO_NEAR_CHANCE_BOUND:g}")
    define("ResLOAONearChanceCount", sum(value < LOAO_NEAR_CHANCE_BOUND for value in pairs))
    define("ResLOAOSupPairs", len(pairs))
    for direction, block in loao.items():
        if block.get("status") != "EVALUABLE":
            continue
        source, target = direction.split("->")
        tag = f"{CONTEXT_NAMES.get(source, source)}To{CONTEXT_NAMES.get(target, target)}"
        best = max(mean(name, "auroc", block["models"]) for name in supervised)
        define(f"ResLOAO{tag}SupMaxAUROC", f"{best:.3f}")

    if ablation is not None:
        losses = [
            -result["variant_minus_full_mean"]["test_roc_tpr_at_far_1.0pct"]
            for family in ("window_length", "coordinate_representation")
            for result in ablation["ablations"][family]["models"].values()
        ]
        define("ResAblMinTPRLoss", f"{min(losses):.2f}")

    outcomes: dict[str, dict[str, int]] = {}
    for archive in capability["datasets"]["veremi"]["official_ground_truth_cross_check"][
        "archives"
    ]:
        totals: dict[str, int] = defaultdict(int)
        for split_counts in archive["record_outcomes_by_split"].values():
            for outcome, value in split_counts.items():
                totals[outcome] += value
        outcomes[archive["attack_category"]] = totals
    tags = {
        "attacker_1_payload_differs": "Differs",
        "attacker_1_no_ground_truth_broadcast": "NoGT",
        "attacker_1_payload_matches": "Matches",
    }
    for category, totals in outcomes.items():
        for outcome, tag in tags.items():
            if outcome in totals:
                define(
                    f"ResAudit{CONTEXT_NAMES.get(category, category)}{tag}", count(totals[outcome])
                )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_macros(summary: dict[str, Any], ablation: dict[str, Any] | None, path: Path) -> None:
    """Emit \\newcommand definitions so manuscript numbers are generated, not typed."""
    lines = ["% Generated by corridor_aiops.evaluation.tables; do not edit by hand."]

    def define(name: str, value: Any) -> None:
        if not name.isalpha():
            raise ValueError(f"Invalid LaTeX macro name: {name}")
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def counts(value: int) -> str:
        return f"{value:,}".replace(",", "{,}")

    for split, block in summary["official_splits"].items():
        tag = split.capitalize()
        define(f"Res{tag}Records", counts(block["records"]))
        define(f"Res{tag}Benign", counts(block["attacker_0_non_malicious"]))
        define(f"Res{tag}Attacker", counts(block["attacker_1_misbehavior"]))
    audit = summary["composite_sender_cluster_audit"]
    for split, value in audit["unique_composite_clusters_by_split"].items():
        define(f"Res{split.capitalize()}Clusters", counts(value))
    for split, value in audit["cross_archive_source_id_collisions_by_split"].items():
        define(f"Res{split.capitalize()}Collisions", counts(value))
    overlap = next(iter(audit["train_test_sender_overlap_by_archive"].values()))
    define("ResTrainTestOverlap", overlap["train_test_source_id_overlap"])
    define("ResSeeds", len(summary["protocol"]["training_seeds"]))
    define("ResEpochs", summary["protocol"]["epochs"])
    bootstrap = summary["uncertainty"]["sender_cluster_bootstrap"]
    define("ResBootRequested", counts(bootstrap["requested_replicates"]))
    define("ResBootValid", counts(bootstrap["valid_replicates"]))
    define("ResBootSkipped", counts(bootstrap["skipped_replicates"]))
    define("ResBootClusters", counts(bootstrap["composite_clusters"]))

    for name, block in summary["models"].items():
        tag = MACRO_MODEL_NAMES[name]
        stats = block["mean_std_across_training_seeds"]
        for key, (suffix, scale, digits) in MACRO_METRICS.items():
            define(f"Res{tag}{suffix}", _macro_value(stats, key, scale, digits))
        interval = bootstrap["intervals_95pct"][name]
        define(f"Res{tag}FOneCI", _ci(interval["f1_cyber_vs_benign"]))
        define(f"Res{tag}AUROCCI", _ci(interval["auroc"]))
        delta = bootstrap["paired_differences_reference_minus_model"].get(name)
        if delta:
            define(f"Res{tag}DeltaFOneCI", _ci(delta["delta_f1_cyber_vs_benign"]["ci_95pct"]))
            define(f"Res{tag}DeltaAUROCCI", _ci(delta["delta_auroc"]["ci_95pct"]))
        localization = block["emitter_ranking_and_simulated_latency"][
            "mean_std_across_training_seeds"
        ]
        if "attacker_emitter_recall_at_3" in localization:
            define(
                f"Res{tag}RecallThree",
                _macro_value(localization, "attacker_emitter_recall_at_3", 100.0, 2),
            )
    recalls = [
        block["emitter_ranking_and_simulated_latency"]["mean_std_across_training_seeds"].get(
            "attacker_emitter_recall_at_3", {"mean": 0.0}
        )["mean"]
        for block in summary["models"].values()
    ]
    define("ResMaxRecallThree", f"{100 * max(recalls):.2f}")

    for archive, block in summary["per_attack_category"].items():
        context = CONTEXT_NAMES.get(archive, archive)
        for name, result in block["models"].items():
            stats = result["mean_std_across_training_seeds"]
            define(
                f"Res{context}Only{MACRO_MODEL_NAMES[name]}AUROC",
                _macro_value(stats, "auroc", 1.0, 3),
            )
    generalization = summary.get("generalization", {})
    for direction, block in generalization.get("leave_one_attack_type_out", {}).items():
        source, target = direction.split("->")
        context = f"LOAO{CONTEXT_NAMES.get(source, source)}To{CONTEXT_NAMES.get(target, target)}"
        define(f"Res{context}Status", block["status"].replace("_", "\\_"))
        if block["status"] != "EVALUABLE":
            continue
        for name, result in block["models"].items():
            stats = result["mean_std_across_training_seeds"]
            tag = MACRO_MODEL_NAMES[name]
            define(f"Res{context}{tag}AUROC", _macro_value(stats, "auroc", 1.0, 3))
            define(f"Res{context}{tag}FOne", _macro_value(stats, "f1_cyber_vs_benign", 1.0, 3))
    holdout = generalization.get("sender_disjoint_holdout")
    if holdout:
        define("ResSenderStatus", holdout["status"].replace("_", "\\_"))
        if holdout["status"] == "EVALUABLE":
            for split, value in holdout["senders_by_partition"].items():
                define(f"ResSender{split.capitalize()}Senders", counts(value))
            define("ResSenderOverlap", holdout["overlap"]["composite_train_test"])
            for name, result in holdout["models"].items():
                stats = result["mean_std_across_training_seeds"]
                tag = MACRO_MODEL_NAMES[name]
                define(f"ResSender{tag}AUROC", _macro_value(stats, "auroc", 1.0, 3))
                define(f"ResSender{tag}FOne", _macro_value(stats, "f1_cyber_vs_benign", 1.0, 3))
    if ablation is not None:
        define(
            "ResTopBaseline", summary["models"][ablation["top_supervised_baseline"]]["display_name"]
        )
        families = {
            "window_length": "Window",
            "coordinate_representation": "Coords",
            "timing_and_jitter": "Timing",
            "local_density": "Density",
            "validity_masks": "Masks",
        }
        for family, block in ablation["ablations"].items():
            for name, result in block["models"].items():
                tag = "TT" if name == "tabular_transformer" else "Top"
                stats = result["variant_mean_std_across_training_seeds"]
                define(f"ResAbl{families[family]}{tag}AUROC", _macro_value(stats, "auroc", 1.0, 3))
                define(
                    f"ResAbl{families[family]}{tag}Delta",
                    f"{result['variant_minus_full_mean']['auroc']:+.3f}",
                )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
