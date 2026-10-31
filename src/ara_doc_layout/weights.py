"""The checkpoints, fetched from the project's releases on first use.

They are past what a git host keeps in a repository and what belongs in a
wheel, so each is a release asset. It is downloaded once into the user's cache,
checked against the hash the package was tested with, and read from there
afterwards; a download that is cut short or altered is never used.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RELEASES = "https://github.com/g3ida/ara-doc-layout/releases/download"


@dataclass(frozen=True)
class Checkpoint:
    filename: str
    url: str
    sha256: str
    #: An environment variable naming a file to use instead of the download —
    #: for a machine with no network, or a checkpoint of your own.
    env: str


PP_DOCLAYOUT = Checkpoint(
    "pp_doclayout_plusl.pdparams",
    f"{RELEASES}/v0.1.0/pp_doclayout_plusl.pdparams",
    "6877e5c91f9de104db21beea27ab8de39d20ed2210c32a3f83daa9eafdde448e",
    "ARA_DOC_LAYOUT_WEIGHTS",
)
DFINE = Checkpoint(
    "dfine_l.onnx",
    f"{RELEASES}/v0.2.0/dfine_l.onnx",
    "b85900d259887d282a521b14ab0e7504c6256695458b658e87abb27edc6c29b5",
    "ARA_DOC_LAYOUT_DFINE_WEIGHTS",
)

# PP-DocLayout_plus-L's, under the names 0.1 gave them.
FILENAME, URL, SHA256, WEIGHTS_ENV = (
    PP_DOCLAYOUT.filename,
    PP_DOCLAYOUT.url,
    PP_DOCLAYOUT.sha256,
    PP_DOCLAYOUT.env,
)

#: Where downloads are kept; `$XDG_CACHE_HOME/ara-doc-layout` otherwise.
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
    print(f"ara-doc-layout: downloading {target.name} to {target.parent}", file=sys.stderr)
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


def weights_path(checkpoint: Checkpoint = PP_DOCLAYOUT, *, download: bool = True) -> Path:
    """*checkpoint* on disk: the file its environment variable names, else the
    cached download, fetched now if it is not there yet."""
    if os.environ.get(checkpoint.env):
        path = Path(os.environ[checkpoint.env])
        if not path.is_file():
            raise FileNotFoundError(f"{checkpoint.env} names {path}, which is not a file.")
        return path
    target = cache_dir() / checkpoint.filename
    if target.is_file():
        return target
    if not download:
        raise FileNotFoundError(f"No weights at {target}, and downloading was not allowed.")
    return fetch(checkpoint.url, target, checkpoint.sha256)
