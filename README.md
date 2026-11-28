# CorridorAIOps-RCA

**Repository author:** Faysal A. Chowdhoury

**Repository:** [faysalchowdhoury/corridor-aiops-rca](https://github.com/faysalchowdhoury/corridor-aiops-rca)

**Manuscript:** *Attack-Specific Misbehavior Detection in Connected-Vehicle Networks: A Sender-Clustered Evaluation of Cross-Attack Generalization*, by Faysal A. Chowdhoury and Lu Sun ([paper/main.tex](paper/main.tex), Elsevier `elsarticle` format)

> **Scope.** This repository is an **evidence-bounded evaluation benchmark for simulated, attacker-labeled V2X attack detection**. It is **not** a validated corridor-level AIOps system and **not** a causal root-cause analysis (RCA) system. The held-out scenario `InTAS_urban_2` is a different area of the same InTAS simulation setup (Hermann et al., VNC 2026) with the same two attack categories, so it tests cross-area transfer; it is not validation on another simulator or on real-world connected-vehicle traffic.

This repository evaluates simulated V2X misbehavior detection on the published VeReMi NextGen InTAS highway 2 traces, with InTAS urban 2 as a held-out test scenario. It uses no traffic-physics model and does not generate synthetic traffic or attacks. The evidence supports simulated message-level misbehavior detection for the two evaluated attack categories (`dataReplay` and `dosAttack`) and retrospective sender ranking only. It does **not** support field deployment readiness, corridor-level operations analytics, causal root-cause analysis (RCA), work-zone false-alarm rates, cross-modal fusion, or production latency claims. The repository name predates this scope.

Features are **past-only (temporally causal)**: each window uses only the current and earlier messages of a sender, which prevents future leakage. This is not causal inference or causal root-cause discovery.

**Labels.** The VeReMi NextGen paper (Hermann et al., VNC 2026) defines the per-message attacker flag as a significant deviation of at least one message attribute from its original value. The ground-truth audit compares position and speed only. Some `dosAttack` attacker records match their ground-truth broadcast in both, so their detection cannot be attributed to falsified position or speed; window context such as flood timing is a plausible basis. Results are reported separately for message-corroborated and these state-matching attacker records. In `results/loao_diagnostics.json` this group keeps the legacy key `sender_level_genuine_state` for compatibility with existing consumers; the key name does not mean that the dataset labels whole senders.

**Generalization.** Three evaluations are reported separately: a sender-disjoint holdout (unseen senders within pooled time), a chronological holdout (later time, with sender overlap retained, so it is not sender-independent), and transfer to `InTAS_urban_2` (same simulator ecosystem, same `dataReplay` and `dosAttack` categories). Each evaluated archive holds exactly one attack category, so leave-one-attack-type-out transfer varies attack semantics and archive-specific simulation-generation effects together. The reversed kinematic signatures are observations in these traces; they help explain the archive-to-archive transfer failure but do not establish a mechanism for other attack implementations.

**Timing.** `inference_ms_per_event` measures in-memory model scoring on pre-extracted feature tensors only. It excludes ingestion, feature extraction, queueing, and end-to-end production latency, and supports no production service-level claim.

**Operating points.** The 1% and 5% test ROC markers are retrospective curve diagnostics. Validation-calibrated recall and the realized test false-alarm rate are the operationally relevant quantities.

## Evidence status

| Item | Status | Where it is recorded |
|---|---|---|
| Official VeReMi archives and ground-truth files | Zenodo MD5 and SHA-256 verified | `data/raw/manifest.json`, `results/dataset_capability_matrix.json` |
| Official ground-truth consistency audit | PASSED (consistency audit, not independent attacker-label validation) | `datasets.veremi.official_ground_truth_cross_check` |
| Sender clusters | composite `(corridor_id, source_id)` | `composite_sender_cluster_audit` in both result files |
| Unsupervised / one-class detectors | rules, isolation forest, one-class SVM, LSTM autoencoder | `results/benchmark_summary.json` |
| Supervised detectors | logistic regression, random forest, histogram gradient boosting, MLP, Tabular Transformer | `results/benchmark_summary.json` |
| XGBoost | NOT_RUN: the optional `xgboost` package is not installed | `models_not_run` |
| Training-seed variability | 5 seeds, mean and sample std | `mean_std_across_training_seeds`, `results/seed_metrics.csv` |
| Sender-cluster bootstrap | 1,000 paired replicates on the first training seed, 95% percentile CIs, paired deltas, CIs for calibrated recall and realized FAR | `uncertainty.sender_cluster_bootstrap` |
| Per-seed bootstrap sensitivity | bootstrap repeated for each of the 5 training seeds on identical resamples | `results/seed_bootstrap_sensitivity.json` |
| Matched FAR operating points | 1% and 5% budgets, seed mean ± std and bootstrap CIs | `results/results_operating_points.tex` |
| Threshold procedures | validation max-F1, Youden's J, and FAR-budget quantile thresholds, seed-wise paired differences | `threshold_procedures` in `results/loao_diagnostics.json` |
| Per-attack slices | category-specific thresholds and the single pooled threshold | `per_attack_category`, `results/loao_diagnostics.json` |
| LOAO, sender-disjoint holdout, chronological holdout | evaluated or `NOT_EVALUABLE` with reason | `generalization` in `results/benchmark_summary.json` |
| Cross-archive benign overlap audit | message, receive, sender, and L = 10 window levels | `benign_overlap_audit` in `results/loao_diagnostics.json` |
| Sender-disjoint deduplicated LOAO | both directions, zero shared senders and messages | `results/loao_sender_disjoint.json`, `results/results_loao_dedup.tex` |
| LOAO reversal diagnosis | feature directions by category, benign shift, ground-truth outcome subgroups, confusion matrices | `results/loao_diagnostics.json` |
| Held-out scenario `InTAS_urban_2` | EVALUABLE: official test split only, no fitting or selection on it, MD5 verified | `results/heldout_scenario.json`, `independent_scenario_holdout` |
| Held-out `InTAS_highway_7`, `InTAS_urban_7` | NOT_EVALUATED (archive size versus the evaluation machine's memory and disk) | `not_evaluated` in `results/heldout_scenario.json` |
| Five feature and temporal ablations | Tabular Transformer and top supervised baseline | `results/ablation_summary.json` |
| Matched L = 10 one-class controls | isolation forest and one-class SVM on the flattened L = 10 window | `one_class_window_length` in `results/ablation_summary.json` |
| Label-free autoencoder checkpoint | main `lstm_autoencoder` is `benign_only_training_with_validation_auroc_selection`; control `lstm_autoencoder_unsupervised_ckpt` selects on benign validation reconstruction error only | `autoencoder_checkpoint_selection` in `results/ablation_summary.json` |
| Seen versus unseen official test senders | official test split divided by whether the sender occurs in the official training split; all evaluated models, also on the sender-disjoint holdout | `sender_overlap` in `results/loao_diagnostics.json`, `results/results_sender_overlap.tex` |
| Message-corroborated versus state-matching attacker records | all evaluated models, pooled official test and LOAO | `label_granularity` in `results/loao_diagnostics.json`, `results/results_loao_subgroups.tex` |
| Seed-level effect sizes | mean paired differences, fraction of bootstrap replicates favoring the reference, standardized d_z | `effect_size` in `results/seed_bootstrap_sensitivity.json` |
| Determinism and leakage tests | timestamp-tie permutation invariance, no future influence, earliest receiver copy, source-only normalization and thresholds | `tests/test_p0_regressions.py` |
| Second-run reproduction | measured comparison, not assumed | `results/reproduction_audit.json` |
| Clean-checkout reproduction and Linux checks | headline rerun from a fresh clone; ruff and pytest on WSL Ubuntu | `results/clean_checkout_audit.json` |
| CorridorLogTransformer, StaticGCN, AsyncSTGNN | architecture-only, **not evaluated** | `results/model_eligibility_matrix.json` |
| Causal RCA, work-zone FAR, cross-modal fusion, field MTTD | NOT_EVALUABLE | `results/dataset_capability_matrix.json` |

### Official ground-truth consistency audit

The separately published VeReMi ground-truth files contain the genuine sender state of each broadcast but **no `attacker` or attack-type field**, so labels cannot be compared one-to-one. `cross_check_veremi_ground_truth` therefore enforces the invariants the files can verify and raises `ValueError` on any violation:

1. the attack category in the archive file name equals the category in every nested member path;
2. every `attacker = 0` record exists in the same split's ground truth with matching sender position and speed;
3. where a ground-truth record carries an `attacker` field, it equals the embedded label.

Attacker records are corroborated by state divergence (`dataReplay`), corroborated by the absence of a ground-truth broadcast (`dosAttack` injected messages), or uncorroborated because their position and speed match the ground-truth broadcast (`dosAttack` state-matching records; the audit does not compare other attributes). The counts are in the generated results below and in `results/dataset_capability_matrix.json`.

### Composite sender clusters

Both archives replay the same InTAS vehicle population, so every test `source_id` string occurs in both archives. All bootstrap resampling and overlap audits use `(corridor_id, source_id)`, where `corridor_id = <archive>-<split>`. Cluster counts, cross-archive collisions, per-archive train/test sender overlap, and valid vs. skipped replicates are recorded.

## Results

<!-- BEGIN GENERATED RESULTS -->
<!-- Generated by corridor_aiops.evaluation.readme from results/*.json; do not edit by hand. -->

### Headline results (official test split, 5 training seeds)

| Regime | Model | F1 | AUROC | Val.-calibrated recall, 1% budget (realized test FAR) | Test ROC TPR at FAR 1% (retrospective) |
|---|---|---|---|---|---|
| Unsupervised / one-class | Robust rules | 0.675 ± 0.000 | 0.789 ± 0.000 | 0.013 (0.40%) | 0.013 |
| Unsupervised / one-class | Isolation forest | 0.678 ± 0.022 | 0.797 ± 0.011 | 0.013 (1.91%) | 0.007 |
| Unsupervised / one-class | One-class SVM | 0.854 ± 0.000 | 0.891 ± 0.000 | 0.783 (2.60%) | 0.762 |
| Unsupervised / one-class | LSTM autoencoder | 0.860 ± 0.000 | 0.929 ± 0.000 | 0.007 (1.58%) | 0.006 |
| Supervised | Logistic regression | 0.921 ± 0.000 | 0.985 ± 0.000 | 0.909 (3.10%) | 0.882 |
| Supervised | Random forest | 0.946 ± 0.001 | 0.993 ± 0.000 | 0.934 (1.90%) | 0.918 |
| Supervised | Hist. gradient boosting | 0.943 ± 0.000 | 0.993 ± 0.000 | 0.943 (2.89%) | 0.919 |
| Supervised | Supervised MLP | 0.923 ± 0.005 | 0.988 ± 0.000 | 0.927 (3.17%) | 0.909 |
| Supervised | Tabular Transformer | 0.929 ± 0.004 | 0.989 ± 0.001 | 0.934 (3.13%) | 0.904 |

Values are mean ± sample standard deviation (or seed means) over training seeds. The primary operational metric is validation-calibrated recall with its realized test false-alarm rate; the test ROC TPR at 1% FPR places the threshold with test scores and is a retrospective curve diagnostic, not an achievable operating point.

* **Official ground-truth consistency audit:** 18,496 `dataReplay` attacker records are corroborated by state divergence, 42,192 `dosAttack` records by the absence of a ground-truth broadcast, and 20,140 `dosAttack` records match their ground-truth broadcast in position and speed and remain uncorroborated (the audit compares only these fields). This is a consistency audit, not independent label validation.
* **Composite sender-cluster bootstrap** (first training seed; 1,000 of 1,000 valid replicates, 952 clusters): Random forest and Hist. gradient boosting beat the Tabular Transformer with paired ΔF1 and ΔAUROC 95% intervals below zero.
* **Per-seed bootstrap sensitivity:** repeating the bootstrap for each training seed, 13 of 16 paired verdicts are stable across seeds. Bootstrap replicates are resamples of one set of test senders, not independent experiments.
* **Leave-one-attack-type-out:** supervised AUROC is below 0.5 in 6 of 10 model and direction pairs (lowest: Hist. gradient boosting 0.280, `dataReplay->dosAttack`). The detectors are attack-specific, not attack-general.
* **Cross-archive benign overlap:** the two archives share 30,095 identical non-malicious broadcasts, 19,613 identical L = 10 windows, and 674 sender IDs.
* **Sender-disjoint deduplicated LOAO:** with 0 shared senders and 0 shared non-malicious messages between training and test, supervised AUROC is below 0.5 in 5 of 10 pairs.
* **Why LOAO fails:** Horizontal displacement, speed change, and heading change mark attackers in opposite directions in the two attacks; non-malicious traffic does not shift between archives (cross-archive AUROC 0.476 to 0.508).
* **Threshold procedures:** relative to validation max-F1, Youden's J raises the test false-alarm rate for every seed in 4 of 5 supervised detectors; FAR-budget quantile thresholds are compared in `results/results_threshold_procedures.tex`.
* **Seen versus unseen senders:** 360 official test senders also occur in the official training split and 592 do not, so official-split scores are not sender-independent; supervised AUROC on seen minus unseen senders ranges from -0.001 to +0.006 (`results/results_sender_overlap.tex`).
* **Message-corroborated versus state-matching attacker records:** 10,674 official test attacker records diverge from the ground truth in position or speed or have no ground-truth broadcast, and 3,599 match it in position and speed; supervised recall on the state-matching group is 0.960 to 0.967, which cannot reflect falsified position or speed.
* **Effect sizes:** Tabular Transformer minus random forest AUROC averages -0.003 over seeds (d_z -3.1); the fraction of bootstrap replicates favoring the Transformer averages 0.00. Percentile cluster-bootstrap intervals on imbalanced clustered data and 5-seed effect sizes are approximate.
* **Sender-disjoint holdout:** supervised AUROC 0.992 to 0.996.
* **Chronological holdout:** supervised AUROC 0.988 to 0.996.
* **Matched L = 10 one-class controls:** AUROC Isolation forest 0.797 (L = 1) versus 0.664 (L = 10); One-class SVM 0.891 (L = 1) versus 0.944 (L = 10).
* **Label-free autoencoder checkpoint:** the main LSTM autoencoder trains on non-malicious sequences but selects its checkpoint by validation AUROC (attacker labels); selecting by benign validation reconstruction error instead gives AUROC 0.927 versus 0.929 and F1 0.865 versus 0.860.
* **Held-out scenario `InTAS_urban_2`** (275,813 test messages, nothing fit or selected on it): random forest AUROC 0.996, Tabular Transformer 0.993; Transformer FAR at its transferred threshold 5.04% versus 3.82% on highway 2. Same attacks and simulator, so this is scenario transfer.
* **Sender ranking (negative result):** attacker-emitter recall at k = 3 is at most 1.58%, so message-level detection does not provide reliable source localization.
* **Second-run reproduction:** NOT_BIT_IDENTICAL; 40 of 59,121 numeric values differed (maximum 9.2e-06).
* **Clean checkout** (commit `3b018b8`): 360 per-seed headline values, status REPRODUCED_WITHIN_TOLERANCE; Linux `ruff` PASSED, `pytest` PASSED (106 passed in 20.77s).

<!-- END GENERATED RESULTS -->

## Quick start

Requires Python 3.10 through 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,figures,torch]"
corridor-ingest --tampa-limit 2000
corridor-ingest --heldout-only
corridor-audit
ruff check .
pytest
corridor-benchmark --train-seeds 5 --bootstrap-samples 1000 --bootstrap-preview-rows 20 --epochs 5
corridor-audit
corridor-sensitivity --rerun
corridor-heldout --tables-only
corridor-figures
corridor-reproduce
corridor-readme
```

* `corridor-ingest --heldout-only` downloads the `InTAS_urban_2` archives and official test ground truth into `data/raw/veremi_heldout/` with Zenodo MD5 checks; `corridor-heldout` (without `--tables-only`) refits the highway 2 models and scores that held-out test split.
* `corridor-sensitivity --rerun` refits the official and LOAO models to save per-seed scores, fits the sender-disjoint deduplicated LOAO, and writes the per-seed bootstrap, benign overlap audit, threshold-procedure comparison, and LOAO diagnostics.
* `corridor-reproduce` reruns the benchmark in a fresh process into `results_repro_check/`, compares every numeric value and checkpoint SHA-256 against `results/`, writes `results/reproduction_audit.json`, and deletes the temporary directory.
* `corridor-readme` regenerates the results block above from `results/*.json`.
* Optional: `python -m pip install -e ".[xgboost]"` adds the XGBoost baseline, which is not installed for the reported results.

The full benchmark (5 seeds, all generalization contexts and ablations) runs on CPU in roughly one and a half to two hours; the sensitivity rerun adds about 40 minutes.

## Outputs

| File | Contents |
|---|---|
| `results/benchmark_summary.json` | main comparison, per-attack slices, LOAO, sender-disjoint and chronological holdouts, bootstrap, checkpoint hashes |
| `results/ablation_summary.json` | five ablations, matched L = 10 one-class controls, jitter and density definitions |
| `results/seed_metrics.csv` | one row per context, model, and training seed |
| `results/bootstrap_preview.csv` | first bootstrap replicates (not independent runs) |
| `results/results_*.tex`, `results/results_macros.tex` | generated LaTeX tables and number macros used by the manuscript |
| `results/reproduction_audit.json`, `results/reproduction_macros.tex` | second-run comparison |
| `results/seed_bootstrap_sensitivity.json`, `results/results_seed_bootstrap.tex` | per-seed bootstrap sensitivity |
| `results/loao_diagnostics.json` | LOAO reversal diagnosis, benign overlap audit, threshold procedures, pooled-threshold slices |
| `results/loao_sender_disjoint.json`, `results/results_loao_dedup.tex` | sender-disjoint deduplicated LOAO |
| `results/results_threshold_procedures.tex` | threshold-procedure comparison |
| `results/heldout_scenario.json`, `results/results_heldout.tex`, `results/heldout_macros.tex` | held-out scenario evaluation |
| `results/scores/per_seed_scores_meta.json` | SHA-256 and verification of the saved per-seed scores (`per_seed_scores.npz`, regenerated locally and not committed) |
| `results/clean_checkout_audit.json`, `results/clean_checkout_macros.tex` | clean-clone reproduction and Linux checks |
| `results/sensitivity_macros.tex`, `results/text_macros.tex` | generated numbers quoted in the manuscript text |
| `paper/figures/fig1..fig5` | 350 DPI PDF and PNG, white background, Times New Roman, validated colorblind-safe palette |

## Public data and provenance

| Source | Acquired subset | Used for | Important limitation |
|---|---|---|---|
| [VeReMi NextGen, DOI 10.5281/zenodo.19665762](https://doi.org/10.5281/zenodo.19665762) | InTAS highway 2 `dataReplay` and `dosAttack` archives plus official ground-truth files | All fitting, selection, and evaluation | Simulated VANET scenario; both archives share one scenario and vehicle population |
| Same record | InTAS urban 2 `dataReplay` and `dosAttack` archives and the official test ground truth | Held-out test only (official test split; nothing fit or selected on it) | Simulated; same attacks and simulator as the training scenario |
| Same record | InTAS highway 7 and urban 7 | Not acquired | Archive size versus the evaluation machine |
| [USDOT Tampa CV Pilot BSM](https://data.transportation.gov/d/nm7w-nvbm) and [SPaT](https://data.transportation.gov/d/xn7c-yu2n) | Public sample records | Ingestion and format checks only | Not time-aligned with each other or VeReMi |
| [PeMS-BAY District 4](https://github.com/benchoi93/PeMS-BAY-2022) | Five-minute station data for 2022-01-01 | Ingestion checks only | No incident or anomaly labels |
| [Caltrans Planned Lane Closures](https://lcswebreports.dot.ca.gov/) | District 4, Alameda County, US-80 report | Ingestion checks only | Not aligned with observed traffic |
| [CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) and [MITRE ATT&CK for ICS](https://attack.mitre.org/matrices/ics/) | Official JSON feeds | Context only | Knowledge bases, not corridor telemetry |

`data/raw/manifest.json` and `data/raw/veremi_heldout/manifest.json` store source URLs, byte sizes, and SHA-256 digests. Raw data are not committed. Dataset licensing and attribution remain with the publishers.

## Architecture

* `src/corridor_aiops/data/`: checksummed acquisition, VeReMi parsing and the official ground-truth consistency audit, past-only (temporally causal) trajectory features, dataset audit, model eligibility, and label-free V2X graph construction (`build_v2x_graph_slices` reads telemetry only; `attach_attacker_labels` adds labels separately for supervised use).
* `src/corridor_aiops/streaming/`: bounded asyncio producer-consumer pipeline; producer and consumer are monitored together so a handler failure cancels the producer and propagates without deadlock.
* `src/corridor_aiops/sim_bridge/`: read-only Vissim, SUMO, and CARLA/ROS 2 adapters (no live simulator integration is claimed).
* `src/corridor_aiops/models/`: baselines, masked LSTM autoencoder, Tabular Transformer, and architecture-only CorridorLogTransformer (with padding and causal attention masks), StaticGCN, and AsyncSTGNN.
* `src/corridor_aiops/evaluation/`: benchmark, sensitivity and LOAO diagnostics, held-out scenario, LaTeX tables and macros, figures, reproduction and clean-checkout audits, and the README results generator.
* `paper/`: Elsevier `elsarticle` manuscript, highlights, bibliography, and DOCX build script.

## Paper-writing standard

The manuscript follows the repository's evidence limits and standard pre-submission practices: map each stated gap to a method and a measured result, document provenance and split boundaries, fit preprocessing only on training data, justify baselines and metrics, and report dataset-specific limitations. No writing template guarantees acceptance.

## Citation

See [CITATION.cff](CITATION.cff) and [paper/references.bib](paper/references.bib). Cite each upstream dataset independently when using its records.
