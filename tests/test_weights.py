"""Fetching the checkpoint: what is trusted, what is reused, and what an
interrupted or altered download leaves behind. Against local files; nothing
here reaches the network."""

from __future__ import annotations

import hashlib
import urllib.request

import pytest

from ara_doc_layout import weights
from ara_doc_layout.weights import CACHE_ENV, FILENAME, WEIGHTS_ENV, fetch, weights_path


@pytest.fixture
def release(tmp_path):
    source = tmp_path / "release" / FILENAME
    source.parent.mkdir()
    source.write_bytes(b"not really a checkpoint" * 1000)
    return source


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_download_is_kept_once_it_matches_its_hash(tmp_path, release) -> None:
    target = tmp_path / "cache" / FILENAME

    fetch(release.as_uri(), target, _sha(release))

    assert target.read_bytes() == release.read_bytes()
    assert list(target.parent.iterdir()) == [target]


def test_a_download_that_does_not_match_leaves_nothing_behind(tmp_path, release) -> None:
    target = tmp_path / "cache" / FILENAME

    with pytest.raises(ValueError, match="not the checkpoint"):
        fetch(release.as_uri(), target, "0" * 64)

    assert list(target.parent.iterdir()) == []


def test_an_interrupted_download_leaves_nothing_behind(tmp_path, monkeypatch) -> None:
    def broken(url):
        raise ConnectionResetError("the release went away")

    monkeypatch.setattr(urllib.request, "urlopen", broken)
    target = tmp_path / "cache" / FILENAME

    with pytest.raises(ConnectionResetError):
        fetch("https://example.invalid/x", target, "0" * 64)

    assert list(target.parent.iterdir()) == []


def test_the_first_use_downloads_and_the_next_reads_the_cache(
    tmp_path, release, monkeypatch
) -> None:
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(weights, "URL", release.as_uri())
    monkeypatch.setattr(weights, "SHA256", _sha(release))

    first = weights_path()
    release.unlink()

    assert weights_path() == first == tmp_path / "cache" / FILENAME


def test_a_file_named_in_the_environment_is_used_as_it_is(tmp_path, release, monkeypatch) -> None:
    monkeypatch.setenv(WEIGHTS_ENV, str(release))
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))

    assert weights_path() == release
    assert not (tmp_path / "cache").exists()


def test_a_missing_file_named_in_the_environment_is_an_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "nowhere.pdparams"))

    with pytest.raises(FileNotFoundError, match=WEIGHTS_ENV):
        weights_path()


def test_without_leave_to_download_nothing_is_fetched(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))

    with pytest.raises(FileNotFoundError, match="not allowed"):
        weights_path(download=False)
