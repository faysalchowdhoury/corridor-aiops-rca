from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventKind(str, Enum):
    DETECTOR = "detector"
    BSM = "bsm"
    SPAT = "spat"
    CONTROLLER_LOG = "controller_log"
    INCIDENT_NOTE = "incident_note"


class CorridorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    timestamp: datetime
    corridor_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    kind: EventKind
    metrics: dict[str, float] = Field(default_factory=dict)
    message: str = ""
    label: str | None = None
    root_cause_id: str | None = None

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("metrics")
    @classmethod
    def metrics_must_be_finite(cls, value: dict[str, Any]) -> dict[str, float]:
        for key, number in value.items():
            if not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                raise ValueError(f"metric {key!r} must be finite numeric data")
        return {key: float(number) for key, number in value.items()}
