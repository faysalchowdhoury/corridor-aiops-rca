from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from corridor_aiops.schema import CorridorEvent, EventKind


def _sim_timestamp(value: str | float | int) -> datetime:
    if isinstance(value, str) and ("T" in value or "-" in value):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


def _number(record: dict[str, Any], *names: str) -> float | None:
    normalized = {
        str(key).strip().lower().replace("_", "").replace(" ", ""): value
        for key, value in record.items()
    }
    for name in names:
        value = normalized.get(name.lower().replace("_", "").replace(" ", ""))
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


class BaseSimulatorAdapter(ABC):
    """Read-only parser interface for simulator events."""

    @abstractmethod
    def events(self, path: Path) -> Iterator[CorridorEvent]:
        raise NotImplementedError


def _vissim_rows(path: Path) -> Iterator[dict[str, str]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith(("*", "#"))
    ]
    if not lines:
        raise ValueError(f"Vissim export contains no tabular records: {path}")
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:5]), delimiters=";,\t")
    except csv.Error as error:
        raise ValueError(f"Unable to identify Vissim export delimiter in {path}") from error
    yield from csv.DictReader(lines, dialect=dialect)


class VissimEventStreamAdapter(BaseSimulatorAdapter):
    """Parse read-only Vissim FZP trajectory and LSA signal-state exports."""

    def events(self, path: Path) -> Iterator[CorridorEvent]:
        suffix = path.suffix.lower()
        if suffix not in {".fzp", ".lsa"}:
            raise ValueError("Vissim adapter accepts .fzp trajectory or .lsa signal exports")
        for index, row in enumerate(_vissim_rows(path)):
            relative_time = _number(row, "SIMSEC", "simulation_time", "time")
            if relative_time is None:
                raise ValueError(f"Vissim row {index + 2} has no simulation timestamp")
            normalized = {key.strip().lower().replace("_", ""): value for key, value in row.items()}
            signal_group = normalized.get("signalgroup")
            is_signal = suffix == ".lsa" or signal_group is not None
            source_id = (
                str(signal_group or normalized.get("controller", "signal-controller"))
                if is_signal
                else str(normalized.get("vehicle") or normalized.get("no") or f"vehicle-{index}")
            )
            metrics: dict[str, float] = {}
            for name, aliases in {
                "speed": ("SPEED", "speed"),
                "lane": ("LANE", "lane"),
                "link": ("LINK", "link"),
                "phase": ("STATE", "state", "signal_state"),
            }.items():
                value = _number(row, *aliases)
                if value is not None:
                    metrics[name] = value
            message = str(normalized.get("state", ""))
            timestamp = _sim_timestamp(relative_time)
            yield CorridorEvent(
                event_id=f"vissim-{path.stem}-{index}-{timestamp.timestamp()}",
                timestamp=timestamp,
                corridor_id="vissim-replay",
                source_id=source_id,
                kind=EventKind.SPAT if is_signal else EventKind.DETECTOR,
                metrics=metrics,
                message=message,
            )


class SumoTraCIAdapter(BaseSimulatorAdapter):
    """Parse SUMO FCD XML snapshots and traffic-light state records."""

    def events(self, path: Path) -> Iterator[CorridorEvent]:
        if path.suffix.lower() not in {".xml", ".fcd"}:
            raise ValueError("SUMO adapter expects fcd-output XML")
        try:
            simulation_time = "0"
            for event, element in ET.iterparse(path, events=("start", "end")):
                if event == "start" and element.tag == "timestep":
                    simulation_time = element.attrib.get("time", "0")
                    continue
                if event != "end":
                    continue
                if element.tag == "vehicle":
                    timestamp = _sim_timestamp(simulation_time)
                    vehicle_id = element.attrib.get("id", "unknown")
                    metrics = {}
                    for target, source in (
                        ("x", "x"),
                        ("y", "y"),
                        ("speed", "speed"),
                        ("heading", "angle"),
                        ("lane_position", "pos"),
                    ):
                        if source in element.attrib:
                            metrics[target] = float(element.attrib[source])
                    yield CorridorEvent(
                        event_id=f"sumo-{vehicle_id}-{timestamp.timestamp()}",
                        timestamp=timestamp,
                        corridor_id="sumo-replay",
                        source_id=vehicle_id,
                        kind=EventKind.BSM,
                        metrics=metrics,
                    )
                elif element.tag in {"tls", "trafficLight"}:
                    timestamp = _sim_timestamp(simulation_time)
                    source_id = element.attrib.get("id", "traffic-light")
                    state = element.attrib.get("state", "")
                    yield CorridorEvent(
                        event_id=f"sumo-tls-{source_id}-{timestamp.timestamp()}",
                        timestamp=timestamp,
                        corridor_id="sumo-replay",
                        source_id=source_id,
                        kind=EventKind.SPAT,
                        message=state,
                    )
                element.clear()
        except ET.ParseError as error:
            raise ValueError(f"Malformed SUMO FCD XML: {error}") from error


class CarlaTelemetryAdapter(BaseSimulatorAdapter):
    """Parse JSONL exports of CARLA or ROS 2 vehicle and traffic-light topics."""

    def events(self, path: Path) -> Iterator[CorridorEvent]:
        if path.suffix.lower() not in {".jsonl", ".ndjson"}:
            raise ValueError("CARLA adapter expects JSON Lines exported from CARLA or ROS 2")
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    timestamp = _sim_timestamp(record["timestamp"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid CARLA/ROS 2 record on line {line_number}") from error
                vehicle = record.get("vehicle_status")
                if isinstance(vehicle, dict):
                    source_id = str(vehicle.get("vehicle_id", record.get("vehicle_id", "vehicle")))
                    metrics = {
                        key: number
                        for key, aliases in {
                            "speed": ("speed", "velocity"),
                            "yaw": ("yaw", "heading"),
                            "x": ("x",),
                            "y": ("y",),
                        }.items()
                        if (number := _number(vehicle, *aliases)) is not None
                    }
                    yield CorridorEvent(
                        event_id=f"carla-vehicle-{source_id}-{timestamp.timestamp()}",
                        timestamp=timestamp,
                        corridor_id=str(record.get("corridor_id", "carla-replay")),
                        source_id=source_id,
                        kind=EventKind.BSM,
                        metrics=metrics,
                    )
                light = record.get("traffic_light")
                if isinstance(light, dict):
                    source_id = str(light.get("id", "traffic-light"))
                    state = str(light.get("state", "unknown"))
                    yield CorridorEvent(
                        event_id=f"carla-light-{source_id}-{timestamp.timestamp()}",
                        timestamp=timestamp,
                        corridor_id=str(record.get("corridor_id", "carla-replay")),
                        source_id=source_id,
                        kind=EventKind.SPAT,
                        message=state,
                    )
                if not isinstance(vehicle, dict) and not isinstance(light, dict):
                    raise ValueError(
                        f"CARLA/ROS 2 record on line {line_number} has no supported telemetry topic"
                    )
