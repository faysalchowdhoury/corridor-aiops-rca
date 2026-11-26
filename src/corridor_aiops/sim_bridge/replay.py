from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from corridor_aiops.data.parse import pems_events
from corridor_aiops.schema import CorridorEvent


def replay_pems(path: Path) -> Iterator[CorridorEvent]:
    yield from pems_events(path)


def replay_vissim_jsonl(path: Path) -> Iterator[CorridorEvent]:
    """Replay newline-delimited Vissim COM event export records."""
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield CorridorEvent.model_validate_json(line)
            except ValueError as error:
                raise ValueError(
                    f"invalid Vissim JSONL export at line {line_number}: {error}"
                ) from error


def write_jsonl(events: Iterator[CorridorEvent], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for event in events:
            stream.write(json.dumps(event.model_dump(mode="json")) + "\n")
            count += 1
    return count
