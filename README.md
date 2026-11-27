# CorridorAIOps-RCA

**Author:** Faysal A. Chowdhoury

**Repository:** [faysalchowdhoury/corridor-aiops-rca](https://github.com/faysalchowdhoury/corridor-aiops-rca)

**Manuscript:** *Evidence-Limited Misbehavior Detection for Connected-Vehicle Telemetry Using Causal Trajectory Features and Sender-Cluster Evaluation* ([paper/main.tex](paper/main.tex))

This repository evaluates simulated V2X misbehavior detection on the published VeReMi NextGen InTAS highway 2 traces. It uses no traffic-physics model and does not generate synthetic traffic or attacks. The evidence supports simulated message-level misbehavior detection and retrospective sender ranking only. It does **not** support field deployment claims, causal root-cause analysis (RCA), work-zone false-alarm rates, cross-modal fusion, or production latency claims.

## Evidence status

| Item | Status | Where it is recorded |
|---|---|---|
| Official VeReMi archives and ground-truth files | Zenodo MD5 and SHA-256 verified | `data/raw/manifest.json`, `results/dataset_capability_matrix.json` |
| Embedded labels vs. official ground truth | PASSED hard cross-check (see below) | `datasets.veremi.official_ground_truth_cross_check` |
| Sender clusters | composite `(corridor_id, source_id)` | `composite_sender_cluster_audit` in both result files |
| Unsupervised / one-class detectors | rules, isolation forest, one-class SVM, LSTM autoencoder | `results/benchmark_summary.json` |
| Supervised detectors | logistic regression, random forest, histogram gradient boosting, MLP, Tabular Transformer | `results/benchmark_summary.json` |
| XGBoost | NOT_RUN unless the optional `xgboost` package is installed | `models_not_run` |
| Training-seed variability | 5 seeds, mean and sample std | `mean_std_across_training_seeds`, `results/seed_metrics.csv` |
| Sender-cluster bootstrap | 1,000 paired replicates, 95% percentile CIs, paired deltas | `uncertainty.sender_cluster_bootstrap` |
| Matched FAR operating points | 1% and 5% budgets | `test_roc_tpr_at_far_*`, `validation_calibrated_*` |
| Per-attack, LOAO, sender-disjoint holdout | evaluated or `NOT_EVALUABLE` with reason | `per_attack_category`, `generalization` |
| Five feature and temporal ablations | Tabular Transformer and top supervised baseline | `results/ablation_summary.json` |
| Second-run reproduction | measured comparison, not assumed | `results/reproduction_audit.json` |
| CorridorLogTransformer, StaticGCN, AsyncSTGNN | architecture-only, **not evaluated** | `results/model_eligibility_matrix.json` |
| Causal RCA, work-zone FAR, cross-modal fusion, field MTTD | NOT_EVALUABLE | `results/dataset_capability_matrix.json` |

### Official ground-truth cross-check

The separately published VeReMi ground-truth files contain the genuine sender state of each broadcast but **no `attacker` or attack-type field**, so labels cannot be compared one-to-one. `cross_check_veremi_ground_truth` therefore enforces the invariants the files can verify and raises `ValueError` on any violation:

1. the attack category in the archive file name equals the category in every nested member path;
2. every `attacker = 0` record exists in the same split's ground truth with matching sender position and speed;
3. where a ground-truth record carries an `attacker` field, it equals the embedded label.

Attacker-labeled outcomes are logged descriptively: every `dataReplay` attacker record diverges from the ground-truth state, while `dosAttack` attacker records either have no ground-truth broadcast (injected messages) or match it. The digest is stored in `results/dataset_capability_matrix.json`.

### Composite sender clusters

Both archives replay the same InTAS vehicle population, so every test `source_id` string occurs in both archives. Earlier versions resampled bare `source_id` strings, which merged two distinct simulated transmitters into one bootstrap cluster. All bootstrap resampling and overlap audits now use `(corridor_id, source_id)`, where `corridor_id = <archive>-<split>`. Cluster counts, cross-archive collisions, per-archive train/test sender overlap, and valid vs. skipped replicates are recorded.

### Headline results (official test split, 5 training seeds)

| Regime | Model | F1 | AUROC | TPR at FAR 1% |
|---|---|---|---|---|
| Unsupervised / one-class | Robust rules | 0.675 ± 0.000 | 0.789 ± 0.000 | 0.013 |
| Unsupervised / one-class | Isolation forest | 0.678 ± 0.022 | 0.797 ± 0.011 | 0.007 |
| Unsupervised / one-class | One-class SVM | 0.854 ± 0.000 | 0.891 ± 0.000 | 0.762 |
| Unsupervised / one-class | LSTM autoencoder | 0.860 ± 0.000 | 0.929 ± 0.000 | 0.006 |
| Supervised | Logistic regression | 0.921 ± 0.000 | 0.985 ± 0.000 | 0.882 |
| Supervised | Random forest | 0.946 ± 0.001 | 0.993 ± 0.000 | 0.918 |
| Supervised | Hist. gradient boosting | 0.943 ± 0.000 | 0.993 ± 0.000 | 0.919 |
| Supervised | Supervised MLP | 0.923 ± 0.005 | 0.988 ± 0.000 | 0.909 |
| Supervised | Tabular Transformer | 0.929 ± 0.004 | 0.989 ± 0.001 | 0.904 |

Values are mean ± sample std over training seeds; TPR is read from the test ROC at 1% FPR. Logistic regression and histogram gradient boosting are seed-invariant as configured, so their std is exactly zero.

* **Composite sender-cluster bootstrap** (1,000 of 1,000 valid replicates, 952 clusters): random forest and gradient boosting beat the Tabular Transformer, with paired ΔF1 and ΔAUROC 95% intervals entirely below zero. The Transformer beats every unsupervised model, logistic regression, and the MLP.
* **Sender-disjoint holdout** (zero composite overlap, 169 test senders): supervised AUROC stays at 0.992 to 0.996.
* **Leave-one-attack-type-out:** supervised detectors fall to or near chance AUROC in 7 of 10 model and direction pairs (for example, gradient boosting 0.280 on `dataReplay` to `dosAttack`). They do not transfer across attack types.
* **Ablations:** a single-message window and raw coordinates cost 0.03 to 0.07 AUROC and more than 0.1 TPR at 1% FAR. Removing timing and jitter features hurts the Transformer only. Density and masks have small effects.
* **Sender ranking:** attacker-emitter recall at k = 3 is at most 1.58%, so message detection does not imply emitter identification.

See `results/reproduction_audit.json` for the second-run audit status. Its current status is `NOT_COMPLETED`: two attempts did not finish (a terminated session, then an out-of-memory error while other workloads shared the machine), so no determinism claim is made. `corridor-reproduce` completes the audit on a machine with enough free memory.

## Quick start

Requires Python 3.10 through 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,figures,torch]"
corridor-ingest --tampa-limit 2000
corridor-audit
ruff check .
pytest -q
corridor-benchmark --train-seeds 5 --bootstrap-samples 1000 --bootstrap-preview-rows 20 --epochs 5
corridor-audit
corridor-reproduce
corridor-figures
```

* `corridor-benchmark --seeds` was renamed: `--bootstrap-preview-rows` sets how many bootstrap replicates are written to `results/bootstrap_preview.csv`, and `--train-seeds` (default 5) sets the number of independent model-training seeds. `--skip-generalization` and `--skip-ablations` exist for quick runs.
* Run `corridor-audit` again after the benchmark so `results/model_eligibility_matrix.json` records which models were evaluated.
* `corridor-reproduce` reruns the benchmark in a fresh process into `results_repro_check/`, compares every numeric value and checkpoint SHA-256 against `results/`, writes `results/reproduction_audit.json`, and deletes the temporary directory. Determinism is reported only from that comparison.
* Optional: `python -m pip install -e ".[xgboost]"` adds the XGBoost baseline.

The full benchmark (5 seeds, all generalization contexts and ablations) runs on CPU in roughly one to two hours.

## Outputs

| File | Contents |
|---|---|
| `results/benchmark_summary.json` | main comparison, per-attack slices, LOAO, sender-disjoint holdout, bootstrap, checkpoint hashes |
| `results/ablation_summary.json` | five ablations, jitter and density definitions |
| `results/seed_metrics.csv` | one row per context, model, and training seed |
| `results/bootstrap_preview.csv` | first bootstrap replicates (not independent runs) |
| `results/results_*.tex`, `results/results_macros.tex` | generated LaTeX tables and number macros used by the manuscript |
| `results/reproduction_audit.json`, `results/reproduction_macros.tex` | second-run comparison |
| `paper/figures/fig1..fig4` | 350 DPI PDF and PNG, white background, Times New Roman, validated colorblind-safe palette |

## Public data and provenance

| Source | Acquired subset | Used for | Important limitation |
|---|---|---|---|
| [VeReMi NextGen, DOI 10.5281/zenodo.19665762](https://doi.org/10.5281/zenodo.19665762) | InTAS highway 2 `dataReplay` and `dosAttack` archives plus official ground-truth files | All fitting and evaluation | Simulated VANET scenario; both archives share one scenario |
| [USDOT Tampa CV Pilot BSM](https://data.transportation.gov/d/nm7w-nvbm) and [SPaT](https://data.transportation.gov/d/xn7c-yu2n) | Public sample records | Ingestion and format checks only | Not time-aligned with each other or VeReMi |
| [PeMS-BAY District 4](https://github.com/benchoi93/PeMS-BAY-2022) | Five-minute station data for 2022-01-01 | Ingestion checks only | No incident or anomaly labels |
| [Caltrans Planned Lane Closures](https://lcswebreports.dot.ca.gov/) | District 4, Alameda County, US-80 report | Ingestion checks only | Not aligned with observed traffic |
| [CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) and [MITRE ATT&CK for ICS](https://attack.mitre.org/matrices/ics/) | Official JSON feeds | Context only | Knowledge bases, not corridor telemetry |

`data/raw/manifest.json` stores source URLs, byte sizes, and SHA-256 digests. Raw data are not committed. Dataset licensing and attribution remain with the publishers.

## Architecture

* `src/corridor_aiops/data/`: checksummed acquisition, VeReMi parsing and ground-truth cross-check, causal trajectory features, dataset audit, model eligibility, and label-free V2X graph construction (`build_v2x_graph_slices` reads telemetry only; `attach_attacker_labels` adds labels separately for supervised use).
* `src/corridor_aiops/streaming/`: bounded asyncio producer-consumer pipeline; producer and consumer are monitored together so a handler failure cancels the producer and propagates without deadlock.
* `src/corridor_aiops/sim_bridge/`: read-only Vissim, SUMO, and CARLA/ROS 2 adapters (no live simulator integration is claimed).
* `src/corridor_aiops/models/`: baselines, masked LSTM autoencoder, Tabular Transformer, and architecture-only CorridorLogTransformer (with padding and causal masks), StaticGCN, and AsyncSTGNN.
* `src/corridor_aiops/evaluation/`: benchmark, LaTeX tables and macros, figures, and reproduction audit.
* `paper/`: IEEEtran manuscript and bibliography.

## Paper-writing standard

The manuscript follows the repository's evidence limits and the pre-submission practices summarized by [Q1-Paper-Anatomy](https://github.com/Touhid-Alam/Q1-Paper-Anatomy): map each stated gap to a method and a measured result, document provenance and split boundaries, fit preprocessing only on training data, justify baselines and metrics, and report dataset-specific limitations. No writing template guarantees acceptance.

## Citation

See [CITATION.cff](CITATION.cff) and [paper/references.bib](paper/references.bib). Cite each upstream dataset independently when using its records.
