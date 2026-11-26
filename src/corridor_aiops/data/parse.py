from __future__ import annotations

import gzip
import io
import json
import zipfile
from collections.abc import Iterator
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


def _iter_veremi_archive(
    payload: bytes, inherited_split: str | None = None
) -> Iterator[tuple[str, dict[str, Any]]]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ValueError("VeReMi dataset file is not a valid ZIP archive") from error
    with archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            split = _dataset_split(member, inherited_split)
            content = archive.read(member)
            if member.lower().endswith(".zip"):
                yield from _iter_veremi_archive(content, split)
            elif member.lower().endswith(".json"):
                if split is None:
                    raise ValueError(
                        f"VeReMi record has no official split in archive path: {member}"
                    )
                for record in _records_from_json(content):
                    yield split, record


def veremi_events(path: Path) -> Iterator[CorridorEvent]:
    """Read published VeReMi traces without altering records or labels."""
    for split, record in _iter_veremi_archive(path.read_bytes()):
        if "attacker" not in record or record["attacker"] not in (0, 1, "0", "1"):
            raise ValueError("VeReMi record is missing the official attacker label")
        try:
            receive_time = float(record["rcvTime"])
            unit = "ns" if abs(receive_time) >= 1e12 else "s"
            timestamp = datetime.fromtimestamp(
                receive_time / (1e9 if unit == "ns" else 1.0),
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
        features = {
            "speed": sender.get("spd", 0.0),
            "acceleration": sender.get("acl", 0.0),
            "heading": sender.get("hed", 0.0),
            "position_x": position[0] if len(position) > 0 else 0.0,
            "position_y": position[1] if len(position) > 1 else 0.0,
            "position_z": position[2] if len(position) > 2 else 0.0,
        }
        try:
            metrics = {name: float(value) for name, value in features.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("VeReMi record contains non-numeric sender telemetry") from error
        sender_id = str(record.get("sender_id", "unknown"))
        message_id = str(record.get("messageID", record.get("message_id", "")))
        yield CorridorEvent(
            event_id=(
                f"veremi-{path.stem}-{split}-{sender_id}-"
                f"{message_id}-{timestamp.timestamp()}"
            ),
            timestamp=timestamp,
            corridor_id=f"{path.stem}-{split}",
            source_id=sender_id,
            kind=EventKind.BSM,
            metrics=metrics,
            label="cyber" if int(record["attacker"]) == 1 else "benign",
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
