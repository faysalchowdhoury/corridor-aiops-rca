from __future__ import annotations

import gzip

from corridor_aiops.data.download import _has_pems_records


def test_download_validator_requires_station_records(tmp_path) -> None:
    path = tmp_path / "sample.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("not a telemetry record\n")
    assert not _has_pems_records(path)
