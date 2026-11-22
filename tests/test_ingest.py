"""Downloading, fingerprinting and checking the raw files, without touching the network."""

import json
import urllib.request

import pytest

from creditrisk import config, ingest


@pytest.fixture
def raw_folder(tmp_path, monkeypatch):
    """A temporary data/raw with every source file present and a manifest that matches."""
    monkeypatch.setattr(config, "RAW", tmp_path)
    monkeypatch.setattr(config, "MANIFEST", tmp_path / "MANIFEST.json")
    entries = {}
    for name, url, note, licence in ingest.SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"contents of {name}".encode())
        entries[name] = {"bytes": path.stat().st_size, "sha256": ingest.sha256(path), "source": url, "what": note, "licence": licence, "retrieved": "2026-11-22T00:00:00Z"}
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": entries}), encoding="utf-8")
    return tmp_path


def test_sha256_of_a_known_string(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert ingest.sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_the_five_sources_are_listed_once_each():
    names = [name for name, *_ in ingest.SOURCES]
    assert len(names) == len(set(names)) == 5
    assert all(url.startswith("https://") for _, url, _, _ in ingest.SOURCES)
    assert {licence for *_, licence in ingest.SOURCES} == {"CC BY 4.0"}


def test_parse_worldbank_drops_years_with_no_value_and_sorts():
    text = json.dumps([{"lastupdated": "2026-11-11"}, [
        {"indicator": {"id": "X.Y", "value": "A series"}, "country": {"value": "Pakistan"}, "date": "2024", "value": 5.5},
        {"indicator": {"id": "X.Y", "value": "A series"}, "country": {"value": "Pakistan"}, "date": "2022", "value": None},
        {"indicator": {"id": "X.Y", "value": "A series"}, "country": {"value": "Pakistan"}, "date": "2023", "value": 6.5}]])
    series, info = ingest.parse_worldbank(text)
    assert list(series["year"]) == [2023, 2024] and list(series["value"]) == [6.5, 5.5]
    assert info == {"indicator": "X.Y", "name": "A series", "country": "Pakistan", "last_updated": "2026-11-11"}


def test_parse_worldbank_with_no_records_is_an_error():
    with pytest.raises(ValueError):
        ingest.parse_worldbank(json.dumps([{"page": 1}, None]))


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_download_writes_the_file_only_once_it_has_arrived(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse(b"hello"))
    target = tmp_path / "deep" / "folder" / "file.csv"
    ingest.download("https://example.org/file.csv", target)
    assert target.read_bytes() == b"hello" and not list(target.parent.glob("*.part"))


def test_download_tries_again_and_then_gives_up(tmp_path, monkeypatch):
    calls = []

    def fail(request, timeout):
        calls.append(1)
        raise TimeoutError

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="could not download"):
        ingest.download("https://example.org/x", tmp_path / "x")
    assert len(calls) == ingest.ATTEMPTS and not (tmp_path / "x").exists()


def test_a_matching_folder_verifies(raw_folder):
    assert ingest.verify() == []


def test_a_changed_file_is_reported(raw_folder):
    (raw_folder / config.UCI_DATA_FILE).write_bytes(b"changed")
    assert ingest.verify() == [f"{config.UCI_DATA_FILE}: checksum differs from the manifest"]


def test_a_missing_file_is_reported(raw_folder):
    (raw_folder / config.UCI_META_FILE).unlink()
    assert ingest.verify() == [f"{config.UCI_META_FILE}: missing"]


def test_a_file_in_the_manifest_that_is_not_a_source_is_reported(raw_folder):
    manifest = json.loads((raw_folder / "MANIFEST.json").read_text())
    manifest["files"]["extra/file.csv"] = {"sha256": "0" * 64}
    (raw_folder / "MANIFEST.json").write_text(json.dumps(manifest))
    (raw_folder / "extra").mkdir()
    (raw_folder / "extra" / "file.csv").write_bytes(b"x")
    assert any("not a known source" in p for p in ingest.verify())


def test_run_downloads_only_what_is_missing_and_keeps_old_retrieval_times(raw_folder, monkeypatch):
    fetched = []
    monkeypatch.setattr(ingest, "download", lambda url, destination: (fetched.append(destination.name), destination.write_bytes(b"new copy"))[1])
    (raw_folder / config.WB_CREDIT_FILE).unlink()
    ingest.run()
    assert fetched == [(raw_folder / config.WB_CREDIT_FILE).name]
    manifest = json.loads((raw_folder / "MANIFEST.json").read_text())
    assert manifest["files"][config.UCI_DATA_FILE]["retrieved"] == "2026-11-22T00:00:00Z"          # untouched file keeps its time
    assert manifest["files"][config.WB_CREDIT_FILE]["retrieved"] != "2026-11-22T00:00:00Z"
    assert manifest["citation"] == config.UCI_CITATION
    assert ingest.verify() == []


def test_refresh_downloads_everything(raw_folder, monkeypatch):
    fetched = []
    monkeypatch.setattr(ingest, "download", lambda url, destination: (fetched.append(destination.name), destination.write_bytes(b"fresh"))[1])
    ingest.run(refresh=True)
    assert len(fetched) == 5
