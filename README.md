# CorridorAIOps-RCA

**Author:** Faysal A. Chowdhoury

**Repository:** [faysalchowdhoury/corridor-aiops-rca](https://github.com/faysalchowdhoury/corridor-aiops-rca)

CorridorAIOps-RCA is a data-driven traffic-operations observability prototype. It validates and replays published detector, connected-vehicle, simulator, and threat-intelligence events. It uses no traffic-physics model and does not generate synthetic traffic or cyberattacks.

## Evidence status

The acquisition workflow validates the official VeReMi NextGen archive checksums and official train, validation, and test paths. It also downloads public Tampa CV Pilot BSM and SPaT sample records, a real PeMS-BAY detector file, a current Caltrans planned-lane-closure report, and CISA KEV and MITRE ATT&CK for ICS JSON feeds. Raw source files remain local and are not committed.

The executable benchmark currently evaluates three implemented, transparent baselines on official VeReMi labels: isolation forest, one-class SVM, and robust threshold rules. The 20 reported partitions are stratified subsamples of the official test split, not independent datasets. Record-level bootstrap intervals quantify test-record sampling uncertainty only. The project does not claim that Tampa BSM/SPaT records, PeMS records, and Caltrans closures are time-aligned; the public samples cover different periods and locations. Therefore FAR under Caltrans work zones, MTTD, and causal root-cause accuracy are not calculated. Genuine non-malicious VeReMi records are measured separately from malicious records.

PyTorch Transformer and LSTM module definitions are research scaffolding, not trained systems. A multimodal model comparison, SHAP attribution, Granger analysis, and production latency claim require a trained and validated model, synchronized corridor logs, and causal localization ground truth. Machine-readable outputs make these unmeasured items explicit instead of substituting generated examples.

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
python -m pip install -e ".[dev,figures]"
corridor-ingest --tampa-limit 2000
pytest
ruff check .
corridor-benchmark --seeds 20
corridor-figures
```

`corridor-ingest` downloads the documented open subsets. The VeReMi archive files total tens of megabytes and are verified before use. To change the Tampa sample cap, pass `--tampa-limit`. The Caltrans report covers the latest one to seven days and is saved as source HTML before parsing.

## Architecture

* `src/corridor_aiops/data/`: checksummed acquisition, VeReMi and Tampa parsers, PeMS and Caltrans LCS ingestion, threat-feed validation.
* `src/corridor_aiops/streaming/`: Pydantic event schema and bounded asyncio producer-consumer pipeline.
* `src/corridor_aiops/sim_bridge/`: read-only Vissim FZP/LSA, SUMO FCD/TLS, and CARLA/ROS 2 JSONL adapters.
* `src/corridor_aiops/models/`: anomaly baselines, deviation-based node ranking, optional Transformer and LSTM autoencoder architectures.
* `src/corridor_aiops/evaluation/`: fixed-seed evaluation of official test records, bootstrap intervals, and paper figures.
* `paper/`: IEEEtran manuscript and bibliography.

The current event-source ranker is a deviation-based diagnostic, not an identified causal effect. Feature separation plots are descriptive and are not SHAP values. No raw public dataset is checked into this repository.

## Paper-writing standard

The manuscript follows the repository's research-evidence limits and the pre-submission practices summarized by [Q1-Paper-Anatomy](https://github.com/Touhid-Alam/Q1-Paper-Anatomy): map each stated gap to a method and a measured result, document provenance and split boundaries, fit preprocessing only on training data, justify baselines and metrics, interpret results instead of repeating tables, and report dataset-specific limitations. These practices improve clarity and reviewability; no writing template can guarantee journal acceptance.

## Citation

See [CITATION.cff](CITATION.cff) and [paper/references.bib](paper/references.bib). Cite each upstream dataset independently when using its records.
