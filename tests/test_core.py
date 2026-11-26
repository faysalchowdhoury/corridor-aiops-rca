from __future__ import annotations

import asyncio
import gzip
import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pytest

from corridor_aiops.data.parse import (
    caltrans_lcs_events,
    read_pems,
    tampa_bsm_events,
    veremi_events,
)
from corridor_aiops.models.causal import rank_root_causes
from corridor_aiops.schema import CorridorEvent, EventKind
from corridor_aiops.sim_bridge.adapters import (
    CarlaTelemetryAdapter,
    SumoTraCIAdapter,
    VissimEventStreamAdapter,
)
from corridor_aiops.streaming.pipeline import run_pipeline


def _event(index: int, *, metrics: dict[str, float] | None = None) -> CorridorEvent:
    return CorridorEvent(
        event_id=f"e{index}",
        timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc),
        corridor_id="test",
        source_id=f"detector-{index}",
        kind=EventKind.DETECTOR,
        metrics=metrics or {"occupancy": 1.0, "speed": 50.0},
    )


def test_event_rejects_timezone_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone"):
        CorridorEvent(
            event_id="e",
            timestamp=datetime(2022, 1, 1),
            corridor_id="test",
            source_id="detector",
            kind=EventKind.DETECTOR,
        )


def test_event_rejects_non_finite_metric() -> None:
    with pytest.raises(ValueError, match="finite"):
        _event(0, metrics={"speed": float("nan")})


def test_pems_parser_reads_real_schema(tmp_path) -> None:
    path = tmp_path / "pems.gz"
    row = "01/01/2022 00:00:00,402000,4,80,N,ML,0.5,10,100,20,2.1,62.5\n"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(row)
    frame = read_pems(path)
    assert len(frame) == 1
    assert frame.iloc[0]["station"] == "402000"
    assert frame.iloc[0]["avg_speed"] == 62.5
    assert str(frame.iloc[0]["timestamp"].tz) == "UTC"


def test_veremi_parser_preserves_official_split_and_labels(tmp_path) -> None:
    record = {
        "rcvTime": 1_000_000_000,
        "sender_id": "veh-1",
        "messageID": 3,
        "attacker": 1,
        "sender": {"spd": 10, "acl": 0.1, "hed": 90, "pos": "1.0,2.0,0.0"},
    }
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("messages/veh-1.json", json.dumps([record]))
    path = tmp_path / "published.zip"
    with zipfile.ZipFile(path, "w") as outer:
        outer.writestr("InTAS/Train/InTAS-train.zip", inner.getvalue())
    event = next(veremi_events(path))
    assert event.corridor_id.endswith("-train")
    assert event.label == "cyber"
    assert event.metrics["position_y"] == 2.0


def test_tampa_bsm_parser_keeps_raw_measurements(tmp_path) -> None:
    path = tmp_path / "bsm.json"
    path.write_text(
        json.dumps(
            [
                {
                    "metadata_recordgeneratedat": "2021-03-19T05:14:51.706",
                    "metadata_rsuid": "thea18",
                    "coredata_speed": "796",
                    "coredata_heading": "21512",
                }
            ]
        ),
        encoding="utf-8",
    )
    event = next(tampa_bsm_events(path))
    assert event.kind == EventKind.BSM
    assert event.metrics["speed_raw"] == 796.0


def test_caltrans_lcs_parser_extracts_work_zone_text(tmp_path) -> None:
    path = tmp_path / "lcs.html"
    path.write_text(
        "<table><tr>"
        "<th>County / Route / Direction</th><th>PM</th><th>Location</th>"
        "<th>Facility</th><th>Lanes</th><th>Planned Start / End</th>"
        "<th>Work</th><th>Closure ID</th></tr>"
        "<tr><td>Alameda<br/>80<br/>WB</td><td>1.2<br/>1.7</td>"
        "<td>Oakland<br/>Meter lights</td><td>Mainline<br/>Alternating</td>"
        "<td>Right lane</td><td>11/20/26 07:30 AM<br/>11/27/26 08:01 PM</td>"
        "<td>Maintenance</td><td>Closure 2</td></tr></table>",
        encoding="utf-8",
    )
    event = next(caltrans_lcs_events(path))
    assert event.label == "work_zone"
    assert "Oakland" in event.message


def test_causal_ranker_orders_large_deviation_first() -> None:
    ranked = rank_root_causes(
        ["occupancy", "speed"],
        np.asarray([[1.0, 50.0], [1.0, 50.0], [20.0, 50.0]]),
        ["upstream", "upstream", "downstream"],
    )
    assert ranked[0][0] == "downstream"


def test_vissim_fzp_adapter_parses_trajectory_read_only(tmp_path) -> None:
    path = tmp_path / "trajectory.fzp"
    path.write_text(
        "SIMSEC;NO;VEHICLE;SPEED;LANE;LINK\n1.5;1;veh-1;13.2;2;7\n",
        encoding="utf-8",
    )
    event = next(VissimEventStreamAdapter().events(path))
    assert event.source_id == "veh-1"
    assert event.metrics["speed"] == 13.2
    assert event.kind == EventKind.DETECTOR


def test_vissim_lsa_adapter_emits_signal_state(tmp_path) -> None:
    path = tmp_path / "signal.lsa"
    path.write_text(
        "SIMSEC;SIGNALGROUP;STATE\n3.0;sg-2;green\n",
        encoding="utf-8",
    )
    event = next(VissimEventStreamAdapter().events(path))
    assert event.source_id == "sg-2"
    assert event.kind == EventKind.SPAT
    assert event.message == "green"


def test_sumo_adapter_reads_vehicle_and_tls_records(tmp_path) -> None:
    path = tmp_path / "fcd.xml"
    path.write_text(
        '<fcd-export><timestep time="2">'
        '<vehicle id="veh-1" x="1" y="2" speed="3" angle="4"/>'
        '<tls id="junction-1" state="Gr"/>'
        "</timestep></fcd-export>",
        encoding="utf-8",
    )
    events = list(SumoTraCIAdapter().events(path))
    assert len(events) == 2
    assert events[0].source_id == "veh-1"
    assert events[0].metrics["speed"] == 3.0
    assert events[1].kind == EventKind.SPAT


def test_carla_adapter_reads_vehicle_and_signal_topics(tmp_path) -> None:
    path = tmp_path / "carla.jsonl"
    path.write_text(
        '{"timestamp":"2022-01-01T00:00:00Z",'
        '"vehicle_status":{"vehicle_id":"car-1","speed":10},'
        '"traffic_light":{"id":"tls-1","state":"green"}}\n',
        encoding="utf-8",
    )
    events = list(CarlaTelemetryAdapter().events(path))
    assert len(events) == 2
    assert events[0].source_id == "car-1"
    assert events[1].kind == EventKind.SPAT


def test_async_pipeline_preserves_order() -> None:
    seen: list[str] = []

    async def source():
        for index in range(3):
            yield _event(index)

    async def handler(event: CorridorEvent) -> None:
        seen.append(event.event_id)

    asyncio.run(run_pipeline(source(), handler, max_queue_size=1))
    assert seen == ["e0", "e1", "e2"]
