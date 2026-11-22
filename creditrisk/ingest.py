"""Download the raw data once and keep a fingerprint of every file.

Five files: the cardholder data and its description from the UCI Machine Learning Repository, and three small series about
Pakistan's banks from the World Bank (non-performing loans from 2013, the same ratio for 1998 to 2020, and private credit).
All of them are open data (CC BY 4.0) and are stored in the repository, so the project runs without a download. data/raw/MANIFEST.json
records the source, time, size and SHA-256 of each file, and `python -m creditrisk verify` checks them.
The policy rate in data/reference is typed in by hand from SBP announcements, with a source for every row.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import config

TIMEOUT = 240           # the UCI server is slow, and the World Bank's API can take a minute and a half
ATTEMPTS = 3

# file, source url, what it is, licence
SOURCES = (
    (config.UCI_DATA_FILE, config.UCI_DATA_URL, "30,000 credit card accounts in Taiwan, April to September 2005 (UCI dataset 350)", "CC BY 4.0"),
    (config.UCI_META_FILE, config.UCI_META_URL, "UCI's description of the dataset: variables, notes, citation", "CC BY 4.0"),
    (config.WB_NPL_FILE, config.WB_NPL_URL, "Pakistan: bank non-performing loans to gross loans, % (World Bank FB.AST.NPER.ZS)", "CC BY 4.0"),
    (config.WB_NPL_LONG_FILE, config.WB_NPL_LONG_URL, "Pakistan: bank non-performing loans to gross loans, %, 1998 to 2020 (World Bank GFDD.SI.02)", "CC BY 4.0"),
    (config.WB_CREDIT_FILE, config.WB_CREDIT_URL, "Pakistan: domestic credit to the private sector, % of GDP (World Bank FS.AST.PRVT.GD.ZS)", "CC BY 4.0"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    """Fetch a URL to a file, trying a few times; the file appears only once it has arrived completely."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body = response.read()
            partial = destination.with_name(destination.name + ".part")
            partial.write_bytes(body)
            os.replace(partial, destination)
            return
        except Exception as error:  # a network problem: wait and try again
            last = error
            print(f"[ingest] {destination.name}: attempt {attempt} failed ({type(error).__name__})", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"could not download {url}: {last}")


# ---------------------------------------------------------------------------
# Readers for the small macro files
# ---------------------------------------------------------------------------
def parse_worldbank(text: str) -> tuple[pd.DataFrame, dict]:
    """A World Bank API answer as a table of year and value (years with no value dropped), and what the indicator is."""
    meta, records = json.loads(text)
    if not records:
        raise ValueError("the World Bank answer holds no records")
    rows = [(int(r["date"]), float(r["value"])) for r in records if r["value"] is not None]
    series = pd.DataFrame(rows, columns=["year", "value"]).sort_values("year").reset_index(drop=True)
    info = {"indicator": records[0]["indicator"]["id"], "name": records[0]["indicator"]["value"],
            "country": records[0]["country"]["value"], "last_updated": meta.get("lastupdated")}
    return series, info


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def _entry(path: Path, source: str, note: str, licence: str, retrieved: datetime | None = None) -> dict:
    when = retrieved or datetime.fromtimestamp(path.stat().st_mtime, UTC)
    return {"bytes": path.stat().st_size, "sha256": sha256(path), "source": source, "what": note, "licence": licence,
            "retrieved": when.strftime("%Y-%m-%dT%H:%M:%SZ")}


def run(refresh: bool = False) -> None:
    manifest = json.loads(config.MANIFEST.read_text(encoding="utf-8")) if config.MANIFEST.exists() else {"files": {}}
    for name, url, note, licence in SOURCES:
        path = config.RAW / name
        fresh = refresh or not path.exists()
        if fresh:
            print(f"[ingest] downloading {name} ...", flush=True)
            download(url, path)
        # a file that was already there keeps the time it was first retrieved
        old = manifest["files"].get(name)
        when = None if fresh or old is None else datetime.strptime(old["retrieved"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        manifest["files"][name] = _entry(path, url, note, licence, retrieved=when)
        print(f"[ingest] {name}: {path.stat().st_size:,} bytes", flush=True)
    manifest["citation"] = config.UCI_CITATION
    config.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    config.MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def verify() -> list[str]:
    """Problems found in the raw files: a file in the manifest that is missing, or whose checksum differs."""
    manifest = json.loads(config.MANIFEST.read_text(encoding="utf-8"))["files"]
    problems = []
    for name, entry in manifest.items():
        path = config.RAW / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif sha256(path) != entry["sha256"]:
            problems.append(f"{name}: checksum differs from the manifest")
    listed = {name for name, *_ in SOURCES}
    problems += [f"{name}: in the manifest but not a known source" for name in manifest if name not in listed]
    return problems
