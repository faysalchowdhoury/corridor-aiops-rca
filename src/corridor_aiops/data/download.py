from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

PEMS_MIRRORS = (
    "https://raw.githubusercontent.com/benchoi93/PeMS-BAY-2022/master/"
    "d04_text_station_5min_2022_01_01.txt.gz",
)
VEREMI_API = "https://zenodo.org/api/records/19665762"
VEREMI_FILES = (
    "InTAS_highway_2_dataReplay.zip",
    "InTAS_highway_2_dosAttack.zip",
    "InTAS_highway_2_Train_groundTruth.json",
    "InTAS_highway_2_Validation_groundTruth.json",
    "InTAS_highway_2_Test_ground_truth.json",
)
TAMPA_RESOURCES = {
    "tampa_bsm.json": (
        "https://data.transportation.gov/resource/nm7w-nvbm.json",
        "metadata_recordgeneratedat",
    ),
    "tampa_spat.json": (
        "https://data.transportation.gov/resource/xn7c-yu2n.json",
        "metadata_generatedat",
    ),
}
THREAT_FEEDS = {
    "cisa_kev.json": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
    "ics_attack.json": (
        "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
        "ics-attack/ics-attack.json"
    ),
}
CALTRANS_LCS_URL = "https://lcswebreports.dot.ca.gov/searchdistricts"


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "CorridorAIOps-RCA/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target)


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "CorridorAIOps-RCA/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _has_pems_records(path: Path) -> bool:
    try:
        import gzip

        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip("\r\n").split(",")
                if len(fields) >= 12 and "/" in fields[0] and fields[1].strip().isdigit():
                    return True
    except (OSError, UnicodeDecodeError):
        return False
    return False


def download_pems(destination: Path, mirrors: tuple[str, ...] = PEMS_MIRRORS) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for url in mirrors:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                suffix=destination.suffix,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            _download(url, temporary)
            if not _has_pems_records(temporary):
                raise ValueError("download did not contain valid PeMS station records")
            os.replace(temporary, destination)
            digest = _sha256(destination)
            return {
                "dataset": "PeMS-BAY District 4 station 5-minute data",
                "source_url": url,
                "sha256": digest,
                "bytes": destination.stat().st_size,
                "records_verified": True,
            }
        except (OSError, urllib.error.URLError, ValueError) as error:
            failures.append(f"{url}: {error}")
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError("All PeMS mirrors failed: " + "; ".join(failures))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, destination)


def download_veremi(data_dir: Path) -> list[dict[str, Any]]:
    metadata = json.loads(_read_url(VEREMI_API))
    available = {item["key"]: item for item in metadata["files"]}
    results = []
    for filename in VEREMI_FILES:
        if filename not in available:
            raise ValueError(f"VeReMi file is not listed in official Zenodo record: {filename}")
        details = available[filename]
        destination = data_dir / "veremi" / filename
        expected_md5 = details["checksum"].split(":", 1)[-1]
        if destination.exists():
            if destination.stat().st_size != details["size"]:
                raise ValueError(f"Cached VeReMi file has unexpected size: {destination}")
            if hashlib.md5(destination.read_bytes()).hexdigest() != expected_md5:
                raise ValueError(
                    f"Cached VeReMi file does not match Zenodo checksum: {destination}"
                )
        else:
            endpoints = (
                f"https://zenodo.org/records/19665762/files/{filename}?download=1",
                f"https://zenodo.org/api/records/19665762/files/{filename}/content",
            )
            failures = []
            for url in endpoints:
                try:
                    payload = _read_url(url)
                    if len(payload) != details["size"]:
                        raise ValueError(
                            f"expected {details['size']} bytes, received {len(payload)}"
                        )
                    actual_md5 = hashlib.md5(payload).hexdigest()
                    if actual_md5 != expected_md5:
                        raise ValueError("download checksum does not match Zenodo metadata")
                    _atomic_bytes(destination, payload)
                    break
                except (OSError, urllib.error.URLError, ValueError) as error:
                    failures.append(f"{url}: {error}")
            else:
                raise RuntimeError(f"Unable to acquire {filename}: " + "; ".join(failures))
        results.append(
            {
                "dataset": "VeReMi NextGen",
                "filename": filename,
                "source_url": f"{VEREMI_API}/files/{filename}/content",
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "zenodo_checksum_verified": True,
            }
        )
    return results


def download_tampa(data_dir: Path, limit: int = 2000) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("Tampa download limit must be positive")
    results = []
    for filename, (base_url, order_column) in TAMPA_RESOURCES.items():
        query = urlencode({"$limit": limit, "$order": order_column})
        url = f"{base_url}?{query}"
        payload = _read_url(url)
        records = json.loads(payload)
        if not isinstance(records, list) or not records or not isinstance(records[0], dict):
            raise ValueError(f"USDOT Tampa resource returned no valid records: {filename}")
        destination = data_dir / "tampa" / filename
        _atomic_bytes(destination, payload)
        results.append(
            {
                "dataset": "USDOT Tampa CV Pilot",
                "filename": filename,
                "source_url": url,
                "records_verified": len(records),
                "bytes": len(payload),
                "sha256": _sha256(destination),
            }
        )
    return results


def download_threat_feeds(data_dir: Path) -> list[dict[str, Any]]:
    results = []
    for filename, url in THREAT_FEEDS.items():
        payload = _read_url(url)
        parsed = json.loads(payload)
        if filename == "cisa_kev.json":
            records = parsed.get("vulnerabilities", [])
        else:
            records = parsed.get("objects", [])
        if not isinstance(records, list) or not records:
            raise ValueError(f"Threat feed has no validated records: {filename}")
        destination = data_dir / "threat_intel" / filename
        _atomic_bytes(destination, payload)
        results.append(
            {
                "dataset": "CISA KEV" if filename == "cisa_kev.json" else "MITRE ATT&CK for ICS",
                "filename": filename,
                "source_url": url,
                "records_verified": len(records),
                "bytes": len(payload),
                "sha256": _sha256(destination),
            }
        )
    return results


def download_caltrans_lcs(data_dir: Path, days: int = 7) -> dict[str, Any]:
    if not 1 <= days <= 7:
        raise ValueError("Caltrans LCS date range must be between one and seven days")
    today = datetime.now(ZoneInfo("America/Los_Angeles")).date()
    start = today - timedelta(days=days - 1)
    fields = [
        ("district", "4"),
        ("countiesList", "Alameda"),
        ("routesList", "80"),
        ("startDate", start.strftime("%m/%d/%Y")),
        ("endDate", today.strftime("%m/%d/%Y")),
        ("tocTypesOfClosure", "All"),
        ("incStatusSet", "InProgress"),
        ("incStatusSet", "Completed"),
        ("incStatusSet", "Canceled"),
        ("incStatusSet", "NoStatus"),
        ("incFacilitySet", "Carpool"),
        ("incFacilitySet", "Mainline"),
        ("incFacilitySet", "Connector"),
        ("incFacilitySet", "OnRamp"),
        ("incFacilitySet", "OffRamp"),
        ("incFacilitySet", "Other"),
        ("whenStatused", "Anytime"),
        ("timePeriod", "All"),
    ]
    request = urllib.request.Request(
        CALTRANS_LCS_URL,
        data=urlencode(fields).encode("ascii"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://lcswebreports.dot.ca.gov/searchdistricts?district=4",
            "User-Agent": "CorridorAIOps-RCA/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(f"Unable to fetch public Caltrans LCS records: {error}") from error
    decoded = payload.decode("utf-8", errors="replace")
    if "Lane Closure Search Results" not in decoded or "districtreporttable" not in decoded:
        raise ValueError("Caltrans LCS response is not a valid lane-closure report")
    destination = data_dir / "caltrans" / "caltrans_lcs_d4_alameda_us80.html"
    _atomic_bytes(destination, payload)
    from corridor_aiops.data.parse import caltrans_lcs_events

    return {
        "dataset": "Caltrans Planned Lane Closures",
        "source_url": CALTRANS_LCS_URL,
        "query": {
            "district": 4,
            "county": "Alameda",
            "route": "80",
            "start_date": start.isoformat(),
            "end_date": today.isoformat(),
        },
        "bytes": len(payload),
        "sha256": _sha256(destination),
        "records_verified": sum(1 for _ in caltrans_lcs_events(destination)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire and checksum published corridor datasets."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--tampa-limit", type=int, default=2000)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    manifest = download_veremi(args.data_dir)
    from corridor_aiops.data.parse import verify_veremi_directory

    for check in verify_veremi_directory(args.data_dir / "veremi"):
        print(f"Ground-truth cross-check passed: {check['archive']} {check['cross_check_sha256']}")
    manifest.extend(download_tampa(args.data_dir, args.tampa_limit))
    manifest.extend(download_threat_feeds(args.data_dir))
    manifest.append(download_pems(args.data_dir / "pems_bay_2022_01_01.txt.gz"))
    manifest.append(download_caltrans_lcs(args.data_dir))
    (args.data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
