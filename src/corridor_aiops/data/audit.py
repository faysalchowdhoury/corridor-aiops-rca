from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from corridor_aiops.data.parse import (
    _iter_veremi_archive,
    caltrans_lcs_events,
    read_pems,
    verify_veremi_directory,
)

METRIC_NAMES = (
    "detection",
    "emitter_or_node_localization",
    "causal_rca",
    "mttd",
    "work_zone_far",
    "cross_modal_fusion",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _capability(
    status: str,
    reason: str,
    ground_truth_source: str | None,
    observation_unit: str,
) -> dict[str, str | None]:
    if status not in {"EVALUABLE", "NOT_EVALUABLE"}:
        raise ValueError(f"Unknown capability status: {status}")
    return {
        "status": status,
        "reason": reason,
        "ground_truth_source": ground_truth_source,
        "observation_unit": observation_unit,
    }


def _scan_veremi(path: Path) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    fields: set[str] = set()
    senders: set[str] = set()
    timestamps: list[int] = []
    split_timestamps: dict[str, list[int]] = defaultdict(list)
    split_senders: dict[str, set[str]] = defaultdict(set)
    positions = [[], [], []]
    time_slices: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    broadcast_receivers: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    broadcast_labels: dict[tuple[str, str, str, str], int] = {}
    broadcast_times: dict[tuple[str, str, str, str], int] = {}
    for split, record_file_id, _, record in _iter_veremi_archive(path.read_bytes()):
        if record.get("attacker") not in (0, 1, "0", "1"):
            raise ValueError(f"Invalid attacker label in {path.name}")
        sender_id = str(record.get("sender_id", ""))
        if not sender_id:
            raise ValueError(f"Missing sender_id in {path.name}")
        timestamp_ns = int(record["rcvTime"])
        split_counts[split] += 1
        label_counts[str(int(record["attacker"]))] += 1
        split_label_counts[split][str(int(record["attacker"]))] += 1
        fields.update(record)
        senders.add(sender_id)
        split_senders[split].add(sender_id)
        timestamps.append(timestamp_ns)
        split_timestamps[split].append(timestamp_ns)
        slice_key = (split, timestamp_ns // 1_000_000_000)
        time_slices[slice_key][sender_id] = (
            time_slices[slice_key].get(sender_id, False) or int(record["attacker"]) == 1
        )
        if record.get("messageID") is not None and record.get("sendTime") is not None:
            broadcast_key = (
                split,
                sender_id,
                str(record["messageID"]),
                str(record["sendTime"]),
            )
            label = int(record["attacker"])
            previous_label = broadcast_labels.get(broadcast_key)
            if previous_label is not None and previous_label != label:
                raise ValueError(f"Conflicting labels for broadcast in {path.name}")
            broadcast_labels[broadcast_key] = label
            previous_time = broadcast_times.get(broadcast_key)
            if previous_time is None or timestamp_ns < previous_time:
                broadcast_times[broadcast_key] = timestamp_ns
            broadcast_receivers[broadcast_key].add(record_file_id)

        position = (record.get("sender") or {}).get("pos", [])
        if isinstance(position, str):
            position = position.split(",")
        for axis in range(min(3, len(position))):
            positions[axis].append(float(position[axis]))

    slice_values = list(time_slices.values())
    multi_vehicle_slices = [rows for rows in slice_values if len(rows) >= 2]
    timelines: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for key, label in broadcast_labels.items():
        split, sender_id, _, _ = key
        timelines[(split, sender_id)].append((broadcast_times[key], label))
    onset_counts: Counter[str] = Counter()
    test_positive_senders: set[str] = set()
    train_validation_positive_senders: set[str] = set()
    unique_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key, label in broadcast_labels.items():
        unique_label_counts[key[0]][str(label)] += 1
    for (split, sender_id), timeline in timelines.items():
        timeline.sort()
        previous_label = 0
        for _, label in timeline:
            if label == 1 and previous_label == 0:
                onset_counts[split] += 1
            previous_label = label
        if split == "test" and any(label == 1 for _, label in timeline):
            test_positive_senders.add(sender_id)
        if split in {"train", "validation"} and any(label == 1 for _, label in timeline):
            train_validation_positive_senders.add(sender_id)
    return {
        "_split_sender_ids": {split: set(values) for split, values in split_senders.items()},
        "record_count": sum(split_counts.values()),
        "official_split_counts": dict(sorted(split_counts.items())),
        "attacker_label_counts": dict(sorted(label_counts.items())),
        "attacker_label_counts_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_label_counts.items())
        },
        "official_split_time_ns_min_max": {
            split: [min(values), max(values)] for split, values in sorted(split_timestamps.items())
        },
        "unique_sender_ids_by_split": {
            split: len(values) for split, values in sorted(split_senders.items())
        },
        "composite_sender_clusters_by_split": {
            split: len(values) for split, values in sorted(split_senders.items())
        },
        "sender_ids_shared_between_train_and_test": len(
            split_senders.get("train", set()) & split_senders.get("test", set())
        ),
        "archive_attack_category": path.stem.rsplit("_", 1)[-1],
        "observed_record_fields": sorted(fields),
        "unique_sender_ids": len(senders),
        "unique_broadcasts_by_split": dict(
            sorted(Counter(key[0] for key in broadcast_receivers).items())
        ),
        "attacker_label_counts_by_unique_broadcast": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(unique_label_counts.items())
        },
        "attacker_label_zero_to_one_transitions_by_split": dict(sorted(onset_counts.items())),
        "test_positive_sender_ids": len(test_positive_senders),
        "test_positive_sender_ids_also_positive_in_train_or_validation": len(
            test_positive_senders & train_validation_positive_senders
        ),
        "broadcasts_observed_in_multiple_record_files": sum(
            len(receiver_files) > 1 for receiver_files in broadcast_receivers.values()
        ),
        "raw_record_copies_after_unique_broadcast_deduplication": (
            sum(split_counts.values()) - len(broadcast_receivers)
        ),
        "receive_time_ns_min_max": [min(timestamps), max(timestamps)],
        "receive_time_unit": "nanoseconds, simulator time (not UTC wall-clock time)",
        "sender_position_m_ranges": [
            [min(axis_values), max(axis_values)] if axis_values else None
            for axis_values in positions
        ],
        "one_second_slices": len(time_slices),
        "multi_vehicle_one_second_slices": len(multi_vehicle_slices),
        "multi_vehicle_slices_with_attacker_labels": sum(
            any(rows.values()) for rows in multi_vehicle_slices
        ),
        "max_attacker_emitters_in_one_second_slice": max(
            (sum(is_attacker for is_attacker in rows.values()) for rows in slice_values),
            default=0,
        ),
    }


def composite_sender_cluster_audit(
    sender_sets: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    """Summarize composite (corridor_id, source_id) clusters across archives.

    ``sender_sets`` maps archive name to split to the set of bare ``source_id`` strings.
    A collision is a bare ``source_id`` that appears in more than one archive within the
    same split; composite keys keep those senders as separate bootstrap clusters.
    """
    splits = sorted({split for by_split in sender_sets.values() for split in by_split})
    clusters_by_split = {}
    collisions_by_split = {}
    for split in splits:
        owners: Counter[str] = Counter()
        total = 0
        for by_split in sender_sets.values():
            senders = by_split.get(split, set())
            total += len(senders)
            owners.update(senders)
        clusters_by_split[split] = total
        collisions_by_split[split] = sum(count > 1 for count in owners.values())
    overlap = {
        name: {
            "train_test_same_archive_source_id_overlap": len(
                by_split.get("train", set()) & by_split.get("test", set())
            ),
            "train_validation_same_archive_source_id_overlap": len(
                by_split.get("train", set()) & by_split.get("validation", set())
            ),
        }
        for name, by_split in sorted(sender_sets.items())
    }
    all_ids = [by_split for by_split in sender_sets.values()]
    scenario_ids = set().union(*(set().union(*s.values()) for s in all_ids)) if all_ids else set()
    return {
        "cluster_identity": "(corridor_id, source_id), corridor_id = <archive>-<split>",
        "unique_composite_clusters_by_split": clusters_by_split,
        "cross_archive_source_id_collisions_by_split": collisions_by_split,
        "train_test_sender_overlap_by_archive": overlap,
        "unique_bare_source_ids_across_archives_and_splits": len(scenario_ids),
        "interpretation": (
            "Both archives replay the same InTAS highway 2 vehicle population, so equal "
            "source_id strings across archives are kept as distinct composite clusters."
        ),
    }


def _read_json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a JSON array of records in {path}")
    return value


NOT_ACQUIRED_SCENARIOS = ("InTAS_highway_7", "InTAS_urban_7")


def heldout_scenario_status(raw_dir: Path) -> dict[str, Any]:
    """Status of the independently simulated held-out scenario for both result files.

    EVALUABLE when the checksum-verified held-out manifest written by
    ``corridor-ingest --heldout-only`` lists archives that are present on disk.
    """
    heldout_dir = Path(raw_dir) / "veremi_heldout"
    manifest_path = heldout_dir / "manifest.json"
    not_acquired = {
        scenario: (
            "NOT_EVALUATED: published in the same Zenodo record but not acquired because of "
            "archive size relative to the evaluation machine"
        )
        for scenario in NOT_ACQUIRED_SCENARIOS
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archives = [
            item["filename"]
            for item in manifest
            if item["filename"].endswith(".zip") and (heldout_dir / item["filename"]).exists()
        ]
        verified = all(item.get("zenodo_checksum_verified") for item in manifest)
        if archives and verified:
            scenario = archives[0].rsplit("_", 1)[0]
            return {
                "status": "EVALUABLE",
                "scenario": scenario,
                "archives": archives,
                "protocol": (
                    "official test split only; models, normalization, checkpoints, and "
                    "thresholds from InTAS_highway_2 applied unchanged"
                ),
                "results_file": "results/heldout_scenario.json",
                "not_evaluated": not_acquired,
            }
    return {
        "status": "NOT_EVALUABLE",
        "reason": (
            "No checksum-verified held-out scenario is present; run "
            "`corridor-ingest --heldout-only` to acquire InTAS_urban_2"
        ),
        "not_evaluated": not_acquired,
    }


def _tampa_summary(data_dir: Path) -> dict[str, Any]:
    bsm = _read_json_array(data_dir / "tampa" / "tampa_bsm.json")
    spat = _read_json_array(data_dir / "tampa" / "tampa_spat.json")
    bsm_times = [
        _utc_datetime(str(row["metadata_recordgeneratedat"]))
        for row in bsm
        if row.get("metadata_recordgeneratedat")
    ]
    spat_times = [
        _utc_datetime(str(row.get("metadata_generatedat") or row["spat_timestamp"]))
        for row in spat
        if row.get("metadata_generatedat") or row.get("spat_timestamp")
    ]
    if not bsm_times or not spat_times:
        raise ValueError("Tampa BSM and SPaT records must include timestamps")

    bsm_sources = {str(row.get("metadata_rsuid", "")) for row in bsm}
    spat_sources = {str(row.get("metadata_rsuid", "")) for row in spat}
    overlap = max(min(bsm_times), min(spat_times)) <= min(max(bsm_times), max(spat_times))
    return {
        "bsm_record_count": len(bsm),
        "bsm_time_min_max_utc_assumed": [
            min(bsm_times).isoformat(),
            max(bsm_times).isoformat(),
        ],
        "bsm_source_ids": sorted(bsm_sources),
        "bsm_records_with_lat_lon": sum(
            row.get("coredata_lat") is not None and row.get("coredata_long") is not None
            for row in bsm
        ),
        "spat_record_count": len(spat),
        "spat_time_min_max_utc_assumed": [
            min(spat_times).isoformat(),
            max(spat_times).isoformat(),
        ],
        "spat_source_ids": sorted(spat_sources),
        "spat_records_with_intersections": sum(
            any(key.startswith("spat_intersections") for key in row) for row in spat
        ),
        "bsm_spat_time_ranges_overlap": overlap,
        "source_timestamp_note": (
            "The acquired timestamp strings omit timezone offsets; UTC is the existing "
            "parser convention. The BSM and SPaT samples are separated by three days."
        ),
    }


def _capabilities(dataset: str) -> dict[str, dict[str, str | None]]:
    no_gt = "No independent ground-truth labels for this metric in the acquired records."
    not_applicable = "No aligned operational telemetry and ground truth for this metric."
    if dataset == "veremi":
        return {
            "detection": _capability(
                "EVALUABLE",
                "Published attacker==0/1 labels are present on official train, validation, "
                "and test BSM records. Conclusions are limited to simulated misbehavior "
                "detection, not field cyber incidents.",
                "VeReMi NextGen record field attacker",
                "labeled BSM record / causal per-vehicle window",
            ),
            "emitter_or_node_localization": _capability(
                "EVALUABLE",
                "Published attacker labels identify sender_id, and audited one-second "
                "slices contain multiple vehicles. This supports attacker-emitter "
                "localization only, not causal corridor-node localization.",
                "VeReMi NextGen record fields attacker and sender_id",
                "multi-vehicle one-second simulator-time slice",
            ),
            "causal_rca": _capability(
                "NOT_EVALUABLE",
                "The archive labels malicious messages and senders, but provides no "
                "independent corridor-disruption outcome or causal root-node annotations.",
                None,
                "causal incident episode with independently labeled origin",
            ),
            "mttd": _capability(
                "EVALUABLE",
                "A narrowly scoped simulated attack-message detection latency can use "
                "attacker-label 0-to-1 transitions within the official test timeline as "
                "episode onsets. It is not field incident MTTD and requires explicit "
                "handling of attacks already active before test.",
                "VeReMi NextGen attacker labels and rcvTime",
                "per-emitter labeled BSM episode; seconds of simulator time",
            ),
            "work_zone_far": _capability(
                "NOT_EVALUABLE",
                "VeReMi has no work-zone exposure labels or synchronized operational "
                "traffic observations.",
                None,
                "work-zone exposure window",
            ),
            "cross_modal_fusion": _capability(
                "NOT_EVALUABLE",
                "The acquired VeReMi subset contains BSM records only, without aligned "
                "controller, detector, SPaT, or operator-log streams.",
                None,
                "synchronized multimodal event window",
            ),
        }

    if dataset == "pems_bay":
        reasons = {
            "detection": (
                "Detector measurements are real, but the acquired day has no verified "
                "incident, anomaly, or attack labels."
            ),
            "emitter_or_node_localization": (
                "Station IDs identify measurement sources, but no ground-truth anomalous "
                "station or origin labels are present."
            ),
            "causal_rca": no_gt,
            "mttd": "No labeled event onset or detector alarm outcome is available.",
            "work_zone_far": (
                "The detector file covers 2022-01-01 and does not overlap the 2026 "
                "Caltrans closure report."
            ),
            "cross_modal_fusion": (
                "The acquired PeMS subset contains detector rows only and no synchronized "
                "BSM, SPaT, or controller-log records."
            ),
        }
        return {
            metric: _capability(
                "NOT_EVALUABLE",
                reasons[metric],
                None,
                "station five-minute interval",
            )
            for metric in METRIC_NAMES
        }

    if dataset == "tampa_cv":
        reasons = {
            "detection": "No independently labeled attack or anomaly records are present.",
            "emitter_or_node_localization": (
                "RSU IDs and BSM positions exist, but no anomalous-emitter ground truth "
                "is provided."
            ),
            "causal_rca": no_gt,
            "mttd": "No attack onset or detection outcome labels are present.",
            "work_zone_far": "No work-zone labels or aligned closure data are present.",
            "cross_modal_fusion": (
                "The 1,000 BSM samples are from 2021-03-19, while the 1,000 SPaT samples "
                "are from 2021-03-16; their acquired time ranges do not overlap."
            ),
        }
        return {
            metric: _capability(
                "NOT_EVALUABLE",
                reasons[metric],
                None,
                "individual BSM or SPaT record",
            )
            for metric in METRIC_NAMES
        }

    if dataset == "caltrans_lcs":
        reasons = {
            "detection": "Planned closure records are not sensor anomaly labels.",
            "emitter_or_node_localization": not_applicable,
            "causal_rca": not_applicable,
            "mttd": "No measured telemetry or detector alarm outcomes accompany the plans.",
            "work_zone_far": (
                "Planned closure windows exist, but no aligned traffic observations are "
                "available for measuring false alarms."
            ),
            "cross_modal_fusion": not_applicable,
        }
        return {
            metric: _capability(
                "NOT_EVALUABLE",
                reasons[metric],
                None,
                "planned lane-closure record",
            )
            for metric in METRIC_NAMES
        }

    if dataset in {"cisa_kev", "mitre_ics"}:
        return {
            metric: _capability(
                "NOT_EVALUABLE",
                "Threat-intelligence knowledge records are not corridor telemetry or "
                "labeled operational events.",
                None,
                "threat-intelligence record",
            )
            for metric in METRIC_NAMES
        }
    raise ValueError(f"Unknown dataset capability key: {dataset}")


def run_dataset_audit(
    data_dir: Path = Path("data/raw"),
    output_path: Path = Path("results/dataset_capability_matrix.json"),
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("data/raw/manifest.json must contain a list")

    files_by_hash = {
        _sha256(path): path
        for path in data_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    verified_sources = {}
    for item in manifest:
        digest = item.get("sha256")
        if not digest or digest not in files_by_hash:
            raise ValueError(f"Manifest checksum could not be verified: {item}")
        verified_sources[item.get("filename", item.get("dataset", "unknown"))] = {
            "sha256": digest,
            "path": files_by_hash[digest].relative_to(data_dir).as_posix(),
            "bytes": files_by_hash[digest].stat().st_size,
        }

    veremi_dir = data_dir / "veremi"
    veremi_archives = sorted(veremi_dir.glob("*.zip"))
    if not veremi_archives:
        raise FileNotFoundError(f"No VeReMi archives found in {veremi_dir}")
    veremi = {path.name: _scan_veremi(path) for path in veremi_archives}
    if len(veremi) != 2:
        raise ValueError("This audit expects the two documented InTAS highway 2 archives")
    sender_sets = {name: scan.pop("_split_sender_ids") for name, scan in veremi.items()}
    composite_audit = composite_sender_cluster_audit(sender_sets)
    ground_truth_cross_check = verify_veremi_directory(veremi_dir)

    tampa = _tampa_summary(data_dir)
    pems = read_pems(data_dir / "pems_bay_2022_01_01.txt.gz")
    closures = list(
        caltrans_lcs_events(data_dir / "caltrans" / "caltrans_lcs_d4_alameda_us80.html")
    )
    kev = json.loads((data_dir / "threat_intel" / "cisa_kev.json").read_text(encoding="utf-8"))
    ics = json.loads((data_dir / "threat_intel" / "ics_attack.json").read_text(encoding="utf-8"))
    ics_objects = ics.get("objects", [])
    pems_columns = [
        "timestamp",
        "station",
        "district",
        "freeway",
        "direction",
        "lane_type",
        "station_length",
        "samples",
        "observed_percentage",
        "total_flow",
        "avg_occupancy",
        "avg_speed",
    ]

    metrics = {
        "veremi": _capabilities("veremi"),
        "pems_bay": _capabilities("pems_bay"),
        "tampa_cv": _capabilities("tampa_cv"),
        "caltrans_lcs": _capabilities("caltrans_lcs"),
        "cisa_kev": _capabilities("cisa_kev"),
        "mitre_ics": _capabilities("mitre_ics"),
    }
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "audit_script": "corridor_aiops.data.audit.run_dataset_audit",
        "audit_script_sha256": _sha256(Path(__file__)),
        "repository_worktree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "source_manifest_sha256": _sha256(manifest_path),
        "verified_source_files": verified_sources,
        "datasets": {
            "veremi": {
                "dataset_name": "VeReMi NextGen",
                "dataset_version": "v1.0.0 / Zenodo record 19665762",
                "source_url": "https://doi.org/10.5281/zenodo.19665762",
                "attack_categories_in_acquired_subset": sorted(
                    item["archive_attack_category"] for item in veremi.values()
                ),
                "archives": veremi,
                "official_split_note": (
                    "Train, validation, and test are the official paths inside each "
                    "published archive. Their raw simulator-time ranges and sender-ID "
                    "overlap are recorded per archive; no overlapping random test "
                    "subsamples are treated as independent corridor-days."
                ),
                "receive_observation_note": (
                    "The raw files repeat a sender/messageID/sendTime broadcast across "
                    "receiver files. Trajectory parsing keeps the earliest receive "
                    "observation per broadcast and preserves its integer nanosecond time."
                ),
                "documentation_url": (
                    "https://github.com/VeReMi-dataset/VeReMi-NextGen/blob/main/Documentation/README.md"
                ),
                "composite_sender_cluster_audit": composite_audit,
                "official_ground_truth_cross_check": {
                    "status": "PASSED",
                    "note": (
                        "The official per-split ground-truth JSON files contain no attacker "
                        "or attack-type field. Hard checks: archive attack category equals "
                        "every nested member path; every attacker=0 record matches its "
                        "same-split official ground-truth position and speed; embedded "
                        "attacker equals any ground-truth attacker field. A violation "
                        "raises ValueError."
                    ),
                    "archives": ground_truth_cross_check,
                    "combined_cross_check_sha256": hashlib.sha256(
                        "".join(
                            item["cross_check_sha256"] for item in ground_truth_cross_check
                        ).encode("ascii")
                    ).hexdigest(),
                },
                "metrics": metrics["veremi"],
            },
            "pems_bay": {
                "dataset_name": "PeMS-BAY District 4",
                "dataset_version": "single source file for 2022-01-01",
                "record_count": len(pems),
                "station_count": int(pems["station"].nunique()),
                "timestamp_min_max_utc": [
                    pems["timestamp"].min().isoformat(),
                    pems["timestamp"].max().isoformat(),
                ],
                "observed_source_columns": pems_columns,
                "spatial_metadata": {
                    "station_ids": True,
                    "freeway_and_direction": True,
                    "station_coordinates": False,
                    "postmile_or_ordering_metadata": False,
                    "station_length_is_location_order": False,
                },
                "observed_percentage_min_max": [
                    float(pems["observed_percentage"].min()),
                    float(pems["observed_percentage"].max()),
                ],
                "ground_truth": "No incident, anomaly, detector-failure, or attack labels.",
                "metrics": metrics["pems_bay"],
            },
            "tampa_cv": {
                "dataset_name": "USDOT Tampa CV Pilot BSM and SPaT sample records",
                "dataset_version": "1,000 records per acquired public resource",
                **tampa,
                "metrics": metrics["tampa_cv"],
            },
            "caltrans_lcs": {
                "dataset_name": "Caltrans District 4 Alameda US-80 Planned Lane Closures",
                "record_count": len(closures),
                "planned_start_times_utc": [event.timestamp.isoformat() for event in closures],
                "observed_traffic_during_closures": False,
                "ground_truth": "Planned closure text only; no matched detector/BSM windows.",
                "metrics": metrics["caltrans_lcs"],
            },
            "cisa_kev": {
                "dataset_name": "CISA Known Exploited Vulnerabilities catalog",
                "catalog_version": kev.get("catalogVersion"),
                "record_count": len(kev.get("vulnerabilities", [])),
                "date_added_min_max": [
                    min(item["dateAdded"] for item in kev["vulnerabilities"]),
                    max(item["dateAdded"] for item in kev["vulnerabilities"]),
                ],
                "observation_unit": "vulnerability knowledge record, not corridor telemetry",
                "metrics": metrics["cisa_kev"],
            },
            "mitre_ics": {
                "dataset_name": "MITRE ATT&CK for ICS STIX",
                "record_count": len(ics_objects),
                "object_type_counts": dict(
                    sorted(Counter(obj.get("type", "unknown") for obj in ics_objects).items())
                ),
                "observation_unit": "threat-intelligence STIX object, not corridor telemetry",
                "metrics": metrics["mitre_ics"],
            },
        },
        "track_gate": {
            "track_a1_veremi_detection_and_emitter_localization": {
                "status": "EVALUABLE",
                "allowed_claim": (
                    "Simulated VeReMi misbehavior detection and attacker-emitter "
                    "localization using published message labels."
                ),
                "prohibited_claim": (
                    "Causal corridor root-cause identification or field-incident performance."
                ),
                "holdout_eligibility": {
                    "official_time_splits": {
                        "status": "EVALUABLE",
                        "sender_id_overlap_by_archive": {
                            name: archive["sender_ids_shared_between_train_and_test"]
                            for name, archive in veremi.items()
                        },
                    },
                    "attack_type_holdout": {
                        "status": "EVALUABLE",
                        "available_categories": sorted(
                            item["archive_attack_category"] for item in veremi.values()
                        ),
                        "note": (
                            "Only the acquired dataReplay and dosAttack subsets are "
                            "available for cross-category holdout."
                        ),
                    },
                    "sender_disjoint_holdout": {
                        "status": "EVALUABLE",
                        "cluster_identity": "(corridor_id, source_id); partition by source_id",
                        "note": (
                            "Senders are partitioned by bare source_id across both archives "
                            "and all official splits, which implies zero composite "
                            "(corridor_id, source_id) overlap. Official temporal split "
                            "boundaries are not preserved by this partition."
                        ),
                    },
                    "independent_scenario_holdout": heldout_scenario_status(data_dir),
                },
            },
            "track_a2_pems_bay_labeled_anomaly_evaluation": {
                "status": "NOT_EVALUABLE",
                "reason": (
                    "No anomaly ground truth and no documented station coordinates or "
                    "freeway ordering are present in the acquired source file."
                ),
            },
            "track_b_cross_dataset_contextual_evaluation": {
                "status": "NOT_EVALUABLE",
                "reason": (
                    "PeMS is from 2022-01-01, Caltrans closures are from 2026-11, and "
                    "the Tampa BSM/SPaT sample ranges do not overlap. VeReMi is a "
                    "separate simulator scenario."
                ),
            },
        },
        "limitations": [
            "VeReMi is simulated V2X misbehavior data, not field cyber-incident telemetry.",
            "Attacker-emitter localization is not causal root-cause identification.",
            "The current PeMS source file has observed_percentage equal to zero throughout; "
            "this value is not interpreted as a verified failure label.",
            "Tampa timestamp strings omit timezone offsets; existing parser behavior assumes UTC.",
            (
                "The current evidence does not support work-zone FAR, causal RCA, "
                "or cross-dataset fusion."
            ),
        ],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit public-data labels and track eligibility.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dataset_capability_matrix.json"),
    )
    parser.add_argument(
        "--model-eligibility-output",
        type=Path,
        default=Path("results/model_eligibility_matrix.json"),
    )
    args = parser.parse_args()
    result = run_dataset_audit(args.data_dir, args.output)
    from corridor_aiops.data.tracks import write_model_eligibility_matrix

    model_matrix = write_model_eligibility_matrix(args.output, args.model_eligibility_output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_eligibility_output": str(args.model_eligibility_output),
                "model_count": len(model_matrix["models"]),
                "track_gate": result["track_gate"],
                "verified_source_count": len(result["verified_source_files"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
