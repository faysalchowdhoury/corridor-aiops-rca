from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from corridor_aiops.schema import CorridorEvent, EventKind

PEMS_COLUMNS = [
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


def read_pems(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    frame = pd.read_csv(opener(path, "rt", encoding="utf-8"), header=None, usecols=range(12))
    frame.columns = PEMS_COLUMNS
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="%m/%d/%Y %H:%M:%S", utc=True)
    for column in ("station", "district", "freeway", "total_flow", "avg_occupancy", "avg_speed"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(
        subset=["timestamp", "station", "total_flow", "avg_occupancy", "avg_speed"]
    )
    frame["station"] = frame["station"].astype("int64").astype(str)
    return frame.sort_values(["timestamp", "station"]).reset_index(drop=True)


def pems_events(path: Path, corridor_id: str = "pems-bay-d04") -> Iterator[CorridorEvent]:
    frame = read_pems(path)
    for row in frame.itertuples(index=False):
        timestamp: datetime = row.timestamp.to_pydatetime()
        yield CorridorEvent(
            event_id=f"{row.station}-{timestamp.isoformat()}",
            timestamp=timestamp,
            corridor_id=corridor_id,
            source_id=f"detector-{row.station}",
            kind=EventKind.DETECTOR,
            metrics={
                "flow": float(row.total_flow),
                "occupancy": float(row.avg_occupancy),
                "speed": float(row.avg_speed),
                "observed_percentage": float(row.observed_percentage),
            },
        )


def _records_from_json(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        parsed = parsed.get("records", parsed.get("data", [parsed]))
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("JSON telemetry must contain an array of record objects")
    return parsed


def _dataset_split(path: str, inherited: str | None) -> str | None:
    parts = path.replace("\\", "/").lower().split("/")
    for split in ("train", "validation", "test"):
        if any(part == split or part.startswith(f"{split}_") for part in parts):
            return split
    return inherited


def _finite_metric(value: Any) -> tuple[float, float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0, 0.0
    if not math.isfinite(number):
        return 0.0, 0.0
    return number, 1.0


def _iter_veremi_archive(
    payload: bytes,
    inherited_split: str | None = None,
    splits: frozenset[str] | None = None,
) -> Iterator[tuple[str, str, int, dict[str, Any]]]:
    """Yield (split, file id, index, record); ``splits`` skips other official splits unread."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ValueError("VeReMi dataset file is not a valid ZIP archive") from error
    with archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            split = _dataset_split(member, inherited_split)
            if splits is not None and split is not None and split not in splits:
                continue
            content = archive.read(member)
            if member.lower().endswith(".zip"):
                yield from _iter_veremi_archive(content, split, splits)
            elif member.lower().endswith(".json"):
                if split is None:
                    raise ValueError(
                        f"VeReMi record has no official split in archive path: {member}"
                    )
                record_file_id = member.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
                for record_index, record in enumerate(_records_from_json(content)):
                    yield split, record_file_id, record_index, record


_GROUND_TRUTH_SPLIT_TOKENS = {"train": "train", "validation": "validation", "test": "test"}


def veremi_archive_identity(path: Path) -> tuple[str, str]:
    """Return (scenario, attack_category) from an official archive name.

    ``InTAS_highway_2_dataReplay.zip`` yields ``("InTAS_highway_2", "dataReplay")``.
    """
    scenario, _, category = Path(path).stem.rpartition("_")
    if not scenario or not category:
        raise ValueError(f"VeReMi archive name does not encode an attack category: {path}")
    return scenario, category


def veremi_ground_truth_paths(directory: Path, scenario: str) -> dict[str, Path]:
    """Locate the separately published per-split ground-truth JSON files."""
    found: dict[str, Path] = {}
    for candidate in sorted(Path(directory).glob(f"{scenario}_*.json")):
        name = candidate.stem.lower()
        if "ground" not in name:
            continue
        for split, token in _GROUND_TRUTH_SPLIT_TOKENS.items():
            if f"_{token}_" in f"_{name}_":
                if split in found:
                    raise ValueError(f"Multiple {split} ground-truth files for {scenario}")
                found[split] = candidate
    return found


def _ground_truth_state(record: dict[str, Any]) -> tuple[float, float, float] | None:
    sender = record.get("sender") if isinstance(record.get("sender"), dict) else record
    try:
        position = sender["pos"]
        if isinstance(position, str):
            position = position.split(",")
        values = (float(position[0]), float(position[1]), float(sender["spd"]))
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def _broadcast_key(split: str, record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        split,
        str(record.get("sender_id", "")),
        str(record.get("messageID", "")),
        str(record.get("sendTime", "")),
    )


def _nested_member_names(payload: bytes, prefix: str = "") -> Iterator[str]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.namelist():
            yield prefix + member
            if member.lower().endswith(".zip"):
                yield from _nested_member_names(archive.read(member), prefix + member + "/")


def cross_check_veremi_ground_truth(
    archive_path: Path,
    ground_truth_paths: dict[str, Path],
    *,
    tolerance: float = 1e-6,
    splits: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Cross-check embedded VeReMi labels against the official ground-truth files.

    The published InTAS ground-truth files list the genuine (unmanipulated) state of
    each broadcast but carry no ``attacker`` or attack-type field. The verifiable
    invariants are therefore:

    1. attack type: the category in the archive file name equals the category in
       every nested member path;
    2. every ``attacker == 0`` record exists in the same split's ground truth with
       matching sender position and speed (within ``tolerance``);
    3. if a ground-truth record does carry an ``attacker`` field, it equals the
       embedded label.

    Any violation raises ``ValueError``. Attacker-labeled records are summarized
    descriptively because manipulated or injected messages are expected to diverge.
    ``splits`` restricts the record checks to the named official splits, for example
    a held-out scenario whose test split alone is used.
    """
    all_splits = frozenset(("train", "validation", "test"))
    selected = frozenset(splits) if splits is not None else all_splits
    archive_path = Path(archive_path)
    scenario, category = veremi_archive_identity(archive_path)
    missing = set(selected) - set(ground_truth_paths)
    if missing:
        raise ValueError(f"Missing official ground-truth files for splits: {sorted(missing)}")
    payload = archive_path.read_bytes()

    member_categories: set[str] = set()
    for name in _nested_member_names(payload):
        for component in name.split("/"):
            stem = component[:-4] if component.lower().endswith(".zip") else component
            if stem.startswith(f"{scenario}_"):
                member_categories.add(stem[len(scenario) + 1 :])
    if member_categories != {category}:
        raise ValueError(
            f"Attack-type mismatch in {archive_path.name}: file name says {category!r}, "
            f"member paths say {sorted(member_categories)}"
        )

    ground_truth: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    file_digests = {}
    for split, path in sorted(ground_truth_paths.items()):
        if split not in selected:
            continue
        raw = Path(path).read_bytes()
        file_digests[split] = hashlib.sha256(raw).hexdigest()
        for record in _records_from_json(raw):
            ground_truth[_broadcast_key(split, record)] = record
        del raw

    counts: dict[str, dict[str, int]] = {}
    violations: list[str] = []
    for split, _, _, record in _iter_veremi_archive(payload, splits=selected):
        if record.get("attacker") not in (0, 1, "0", "1"):
            raise ValueError("VeReMi record is missing the official attacker label")
        label = int(record["attacker"])
        bucket = counts.setdefault(split, {})
        reference = ground_truth.get(_broadcast_key(split, record))
        if reference is not None and "attacker" in reference:
            if int(reference["attacker"]) != label:
                violations.append(f"{split}: ground-truth attacker label differs")
        if reference is None:
            outcome = "no_ground_truth_broadcast"
        else:
            embedded = _ground_truth_state(record)
            official = _ground_truth_state(reference)
            same = (
                embedded is not None
                and official is not None
                and all(
                    abs(left - right) <= tolerance
                    for left, right in zip(embedded, official, strict=True)
                )
            )
            outcome = "payload_matches" if same else "payload_differs"
        key = f"attacker_{label}_{outcome}"
        bucket[key] = bucket.get(key, 0) + 1
        if label == 0 and outcome != "payload_matches":
            violations.append(
                f"{split}: attacker=0 record {outcome} (sender {record.get('sender_id')})"
            )
    if violations:
        raise ValueError(
            f"Official ground-truth cross-check failed for {archive_path.name} "
            f"({len(violations)} violations): " + "; ".join(violations[:20])
        )
    summary = {
        "archive": archive_path.name,
        "scenario": scenario,
        "attack_category": category,
        "attack_category_member_paths": sorted(member_categories),
        "ground_truth_files": {
            split: Path(path).name
            for split, path in sorted(ground_truth_paths.items())
            if split in selected
        },
        "ground_truth_sha256": file_digests,
        "ground_truth_has_attacker_field": any(
            "attacker" in record for record in ground_truth.values()
        ),
        "record_outcomes_by_split": {
            split: dict(sorted(bucket.items())) for split, bucket in sorted(counts.items())
        },
        "hard_checks": [
            "attack category in file name equals nested member paths",
            "every attacker=0 record matches its same-split official ground-truth state",
            "embedded attacker equals ground-truth attacker where that field exists",
        ],
        "status": "PASSED",
    }
    if selected != all_splits:
        summary["splits_checked"] = sorted(selected)
    summary["cross_check_sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return summary


def verify_veremi_directory(directory: Path) -> list[dict[str, Any]]:
    """Run :func:`cross_check_veremi_ground_truth` for every archive in a directory."""
    results = []
    for archive in sorted(Path(directory).glob("*.zip")):
        scenario, _ = veremi_archive_identity(archive)
        results.append(
            cross_check_veremi_ground_truth(archive, veremi_ground_truth_paths(directory, scenario))
        )
    if not results:
        raise FileNotFoundError(f"No VeReMi archives found in {directory}")
    return results


def veremi_events(
    path: Path, splits: Iterable[str] | None = None
) -> Iterator[CorridorEvent]:
    """Read one earliest receive observation for each published BSM broadcast.

    ``splits`` restricts parsing to the named official splits (default: all).
    """
    selected = frozenset(splits) if splits is not None else None
    events: dict[tuple[str, str, str], CorridorEvent] = {}
    for split, record_file_id, record_index, record in _iter_veremi_archive(
        path.read_bytes(), splits=selected
    ):
        if "attacker" not in record or record["attacker"] not in (0, 1, "0", "1"):
            raise ValueError("VeReMi record is missing the official attacker label")
        try:
            receive_time_ns = int(record["rcvTime"])
            timestamp = datetime.fromtimestamp(
                receive_time_ns / 1e9,
                tz=timezone.utc,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("VeReMi record has an invalid receive timestamp") from error

        sender = record.get("sender") or {}
        position = sender.get("pos", [])
        if isinstance(position, str):
            try:
                position = json.loads(position)
            except json.JSONDecodeError:
                position = position.split(",")
        if not isinstance(position, list):
            position = []
        speed, speed_valid = _finite_metric(sender.get("spd"))
        acceleration, acceleration_valid = _finite_metric(sender.get("acl"))
        heading, heading_valid = _finite_metric(sender.get("hed"))
        coordinates = [
            _finite_metric(position[axis]) if len(position) > axis else (0.0, 0.0)
            for axis in range(3)
        ]
        metrics = {
            "speed": speed,
            "speed_valid": speed_valid,
            "acceleration": acceleration,
            "acceleration_valid": acceleration_valid,
            "heading": heading,
            "heading_valid": heading_valid,
            "position_x": coordinates[0][0],
            "position_y": coordinates[1][0],
            "position_z": coordinates[2][0],
            "position_xy_valid": coordinates[0][1] * coordinates[1][1],
            "position_z_valid": coordinates[2][1],
        }
        sender_id = str(record.get("sender_id", "unknown"))
        message_id = str(record.get("messageID", record.get("message_id", "")))
        send_time = record.get("sendTime")
        broadcast_id = (
            f"{sender_id}:{message_id}:{send_time}"
            if send_time is not None and message_id
            else None
        )
        event = CorridorEvent(
            event_id=(
                f"veremi-{path.stem}-{split}-{sender_id}-"
                f"{message_id}-{receive_time_ns}-{record_file_id}-{record_index}"
            ),
            timestamp=timestamp,
            corridor_id=f"{path.stem}-{split}",
            source_id=sender_id,
            kind=EventKind.BSM,
            metrics=metrics,
            label="cyber" if int(record["attacker"]) == 1 else "benign",
            broadcast_id=broadcast_id,
            receive_time_ns=receive_time_ns,
        )
        key = (event.corridor_id, event.source_id, broadcast_id or event.event_id)
        previous = events.get(key)
        if previous is not None and previous.label != event.label:
            raise ValueError(f"Conflicting labels for VeReMi broadcast {broadcast_id}")
        if previous is None or (
            event.receive_time_ns is not None
            and previous.receive_time_ns is not None
            and event.receive_time_ns < previous.receive_time_ns
        ):
            events[key] = event

    yield from sorted(
        events.values(),
        key=lambda event: (
            event.corridor_id,
            event.receive_time_ns if event.receive_time_ns is not None else 0,
            event.source_id,
            event.event_id,
        ),
    )


def _load_records(path: Path) -> list[dict[str, Any]]:
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON records file: {path}") from error
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected a JSON array of records in {path}")
    return records


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def tampa_bsm_events(path: Path) -> Iterator[CorridorEvent]:
    for row in _load_records(path):
        timestamp = _timestamp(str(row["metadata_recordgeneratedat"]))
        metrics = {}
        for target, source in (
            ("speed_raw", "coredata_speed"),
            ("heading_raw", "coredata_heading"),
            ("latitude_raw", "coredata_lat"),
            ("longitude_raw", "coredata_long"),
            ("message_count", "coredata_msgcnt"),
        ):
            try:
                metrics[target] = float(row[source])
            except (KeyError, TypeError, ValueError):
                continue
        source_id = str(row.get("metadata_rsuid", "unknown-rsu"))
        yield CorridorEvent(
            event_id=f"tampa-bsm-{source_id}-{row.get('coredata_msgcnt', timestamp.timestamp())}",
            timestamp=timestamp,
            corridor_id="tampa-cv-pilot",
            source_id=source_id,
            kind=EventKind.BSM,
            metrics=metrics,
        )


def tampa_spat_events(path: Path) -> Iterator[CorridorEvent]:
    for row in _load_records(path):
        timestamp_value = row.get("metadata_generatedat") or row.get("spat_timestamp")
        if not timestamp_value:
            continue
        timestamp = _timestamp(str(timestamp_value))
        source_id = str(row.get("metadata_rsuid", "unknown-rsu"))
        message = json.dumps(
            {
                key: value
                for key, value in row.items()
                if "movementstate" in key.lower() or "intersectionstate" in key.lower()
            },
            sort_keys=True,
        )
        yield CorridorEvent(
            event_id=f"tampa-spat-{source_id}-{timestamp.timestamp()}",
            timestamp=timestamp,
            corridor_id="tampa-cv-pilot",
            source_id=source_id,
            kind=EventKind.SPAT,
            message=message,
        )


class _ClosureTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.row is not None and self.cell is not None:
            lines = [" ".join(line.split()) for line in "".join(self.cell).splitlines()]
            self.row.append("\n".join(line for line in lines if line))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)


def caltrans_lcs_events(path: Path) -> Iterator[CorridorEvent]:
    parser = _ClosureTableParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    if not parser.rows:
        raise ValueError(f"No Caltrans closure table rows found in {path}")
    timezone_local = ZoneInfo("America/Los_Angeles")
    for row_index, row in enumerate(parser.rows):
        if len(row) != 8 or row[0].lower().startswith("county"):
            continue
        start_value = row[5].splitlines()[0].strip()
        try:
            start_time = datetime.strptime(start_value, "%m/%d/%y %I:%M %p").replace(
                tzinfo=timezone_local
            )
        except ValueError as error:
            raise ValueError(
                f"Caltrans closure row {row_index + 1} has an invalid planned start time"
            ) from error
        yield CorridorEvent(
            event_id=f"caltrans-lcs-{row[7].replace(' ', '-')}-{start_time.isoformat()}",
            timestamp=start_time,
            corridor_id="caltrans-d04-alameda-us80",
            source_id=f"caltrans-lcs-d04-us80-{row[7]}",
            kind=EventKind.INCIDENT_NOTE,
            message=" | ".join(row),
            metrics={},
            label="work_zone",
        )
