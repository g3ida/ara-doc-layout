"""The checkpoint, fetched from the project's release on first use.

132 MB is past what a git host keeps in a repository and what belongs in a
wheel, so the weights are a release asset. They are downloaded once into the
user's cache, checked against the hash the package was tested with, and read
from there afterwards; a download that is cut short or altered is never used.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

FILENAME = "pp_doclayout_plusl.pdparams"
URL = f"https://github.com/g3ida/ara-doc-layout/releases/download/v0.1.0/{FILENAME}"
SHA256 = "6877e5c91f9de104db21beea27ab8de39d20ed2210c32a3f83daa9eafdde448e"

#: A weights file to use instead of the cached download — for a machine with
#: no network, or a checkpoint of your own.
WEIGHTS_ENV = "ARA_DOC_LAYOUT_WEIGHTS"
#: Where the download is kept; `$XDG_CACHE_HOME/ara-doc-layout` otherwise.
CACHE_ENV = "ARA_DOC_LAYOUT_CACHE"

_CHUNK = 1 << 20


def cache_dir() -> Path:
    if os.environ.get(CACHE_ENV):
        return Path(os.environ[CACHE_ENV])
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "ara-doc-layout"


def fetch(url: str, target: Path, sha256: str) -> Path:
    """Download *url* to *target*, refusing it unless it hashes to *sha256*.

    Written beside the target and moved into place only once checked, so an
    interrupted download leaves nothing that a later run would trust.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    print(f"ara-doc-layout: downloading the weights to {target}", file=sys.stderr)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as part:
        try:
            with urllib.request.urlopen(url) as response:
                while chunk := response.read(_CHUNK):
                    digest.update(chunk)
                    part.write(chunk)
        except BaseException:
            part.close()
            Path(part.name).unlink(missing_ok=True)
            raise
    if digest.hexdigest() != sha256:
        Path(part.name).unlink(missing_ok=True)
        raise ValueError(
            f"The weights at {url} hash to {digest.hexdigest()}, not {sha256}: "
            "not the checkpoint this package was tested with."
        )
    os.replace(part.name, target)
    return target


def weights_path(*, download: bool = True) -> Path:
    """The checkpoint on disk: the file `ARA_DOC_LAYOUT_WEIGHTS` names, else the
    cached download, fetched now if it is not there yet."""
    if os.environ.get(WEIGHTS_ENV):
        path = Path(os.environ[WEIGHTS_ENV])
        if not path.is_file():
            raise FileNotFoundError(f"{WEIGHTS_ENV} names {path}, which is not a file.")
        return path
    target = cache_dir() / FILENAME
    if target.is_file():
        return target
    if not download:
        raise FileNotFoundError(f"No weights at {target}, and downloading was not allowed.")
    return fetch(URL, target, SHA256)
