# CorridorAIOps-RCA

**Author:** Faysal A. Chowdhoury

**Repository:** [faysalchowdhoury/corridor-aiops-rca](https://github.com/faysalchowdhoury/corridor-aiops-rca)

CorridorAIOps-RCA is a data-driven traffic-operations observability prototype. It validates and replays published detector, connected-vehicle, simulator, and threat-intelligence events. It uses no traffic-physics model and does not generate synthetic traffic or cyberattacks.

## Evidence status

The acquisition workflow validates the official VeReMi NextGen archive checksums and official train, validation, and test paths. The VeReMi parser collapses repeated receiver-file observations of the same `sender_id`/`messageID`/`sendTime` broadcast, retaining the earliest receive time at nanosecond precision; it does not merge across official splits. It also downloads public Tampa CV Pilot BSM and SPaT sample records, a real PeMS-BAY detector file, a current Caltrans planned-lane-closure report, and CISA KEV and MITRE ATT&CK for ICS JSON feeds. Raw source files remain local and are not committed.

The executable benchmark evaluates five models on de-duplicated VeReMi records using data-derived trajectory features: isolation forest, one-class SVM, robust threshold rules, an LSTM autoencoder, and a tabular Transformer. The LSTM and Transformer are trained only on eligible official training data; controller-log and graph neural architectures remain untrained. Model thresholds are selected on the official validation split; test records are not subsampled. The 1,000 paired bootstrap replicates resample test sender IDs as clusters, and the 20 rows in `seed_metrics.csv` are bootstrap draws, not independent corridor-days. Current results are in `results/benchmark_summary.json`. The benchmark also reports sender-ranking recall and a simulator-time message detection delay for observed test-split attacker-label transitions; neither is causal RCA or field MTTD. The project does not claim that Tampa BSM/SPaT records, PeMS records, and Caltrans closures are time-aligned; the public samples cover different periods and locations. Therefore FAR under Caltrans work zones and causal root-cause accuracy remain unmeasured. Genuine non-malicious VeReMi records are measured separately from malicious records.

`corridor-audit` verifies the downloaded source checksums and writes `results/dataset_capability_matrix.json` and `results/model_eligibility_matrix.json`. The latter separates evidence eligibility from implementation, training, and evaluation status. The current Gate 0 permits simulated VeReMi misbehavior detection, attacker-emitter localization, and a narrowly defined message-level detection latency if a detector is evaluated. The two acquired attack subsets permit a cross-category holdout, but both use the same InTAS highway 2 scenario; independent-scenario generalization is unavailable. The data do not permit causal RCA or field-incident claims. The acquired PeMS file has no event labels or documented station ordering, and the Tampa BSM/SPaT samples do not overlap in time. Track A2 labeled evaluation and cross-dataset Track B are therefore not evaluable with the current files.

The benchmark trains an LSTM autoencoder on non-malicious training sequences and a tabular Transformer on official training labels. StaticGCN and AsyncSTGNN remain architecture-only, while CorridorLogTransformer is ineligible because the acquired VeReMi records contain no synchronized controller text. `build_v2x_graph_slices` forms time-local graphs from actual labeled BSM senders and their observed positions; it does not create traffic records or infer causal edges. SHAP attribution, Granger analysis, field evaluation, and production latency claims require synchronized corridor logs and suitable ground truth. Machine-readable outputs make these unmeasured items explicit instead of substituting generated examples.

## Public data and provenance

| Source | Acquired subset | Primary utility | Important limitation |
|---|---|---|---|
| [VeReMi NextGen, DOI 10.5281/zenodo.19665762](https://doi.org/10.5281/zenodo.19665762) | Highway 2 `dataReplay` and `dosAttack` archives plus official ground-truth files | Published attacker labels and train/validation/test traces | Simulated VANET scenarios; not field cyber incidents |
| [USDOT Tampa CV Pilot BSM](https://data.transportation.gov/d/nm7w-nvbm) and [SPaT](https://data.transportation.gov/d/xn7c-yu2n) | 1,000 public sample records per resource in the recorded acquisition | Real connected-vehicle and signal telemetry | Different timestamps and sample windows; not synchronized with PeMS or VeReMi |
| [PeMS-BAY District 4](https://github.com/benchoi93/PeMS-BAY-2022) | Public five-minute station data for 2022-01-01 | Real detector flow, occupancy, and speed | Single date; no message-level incident labels |
| [Caltrans Planned Lane Closures](https://lcswebreports.dot.ca.gov/) | Current District 4, Alameda County, US-80 report, up to seven days | Authentic work-zone text and planned times | Current query is not aligned with the historical PeMS or VeReMi test data |
| [CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) and [MITRE ATT&CK for ICS](https://attack.mitre.org/matrices/ics/) | Official JSON feeds | Threat-intelligence context | Knowledge base, not labeled corridor telemetry |

`data/raw/manifest.json` stores source URLs, record counts where available, byte sizes, and SHA-256 digests. VeReMi files are additionally verified against Zenodo's published MD5 values. Dataset licensing and attribution remain with the source publishers.

## Quick start

Requires Python 3.10 through 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,figures,torch]"
corridor-ingest --tampa-limit 2000
corridor-audit
pytest
ruff check .
corridor-benchmark --seeds 20 --bootstrap-samples 1000 --epochs 5
corridor-figures
```

`corridor-ingest` downloads the documented open subsets. The VeReMi archive files total tens of megabytes and are verified before use. To change the Tampa sample cap, pass `--tampa-limit`. The Caltrans report covers the latest one to seven days and is saved as source HTML before parsing.

## Architecture

* `src/corridor_aiops/data/`: checksummed acquisition, VeReMi and Tampa parsers, PeMS and Caltrans LCS ingestion, threat-feed validation, evidence-gated model eligibility, and time-local V2X graph construction.
* `src/corridor_aiops/streaming/`: Pydantic event schema and bounded asyncio producer-consumer pipeline.
* `src/corridor_aiops/sim_bridge/`: read-only Vissim FZP/LSA, SUMO FCD/TLS, and CARLA/ROS 2 JSONL adapters.
* `src/corridor_aiops/models/`: anomaly baselines, trained sequence baselines, and architecture-only log Transformer, StaticGCN, and AsyncSTGNN modules.
* `src/corridor_aiops/evaluation/`: official-split trajectory-feature evaluation, sender-cluster bootstrap intervals, and paper figures.
* `paper/`: IEEEtran manuscript and bibliography.

The current event-source ranker is a deviation-based diagnostic, not an identified causal effect. Feature separation plots are descriptive and are not SHAP values. No raw public dataset is checked into this repository.

## Paper-writing standard

The manuscript follows the repository's research-evidence limits and the pre-submission practices summarized by [Q1-Paper-Anatomy](https://github.com/Touhid-Alam/Q1-Paper-Anatomy): map each stated gap to a method and a measured result, document provenance and split boundaries, fit preprocessing only on training data, justify baselines and metrics, interpret results instead of repeating tables, and report dataset-specific limitations. These practices improve clarity and reviewability; no writing template can guarantee journal acceptance.

## Citation

See [CITATION.cff](CITATION.cff) and [paper/references.bib](paper/references.bib). Cite each upstream dataset independently when using its records.
