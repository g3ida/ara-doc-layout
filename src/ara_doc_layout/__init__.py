"""Layout detection for pages of Arabic books.

Importing the package imports neither Paddle nor onnxruntime: the geometry and
the comparison are plain Python, and each model loads its runtime only when it
is built.
"""

from __future__ import annotations

from .compare import Disagreement, box_disagreements, role_disagreements
from .models import MODELS, load_model
from .regions import (
    CAPTION,
    CLASSES,
    FOOTER,
    FOOTNOTE,
    FURNITURE,
    HASHIYA,
    HEADER,
    OTHER,
    PAGE_NUMBER,
    READ,
    TABLE,
    TEXT,
    Box,
    Region,
    holds,
    reading_order,
    side_by_side,
)

__version__ = "0.2.0"

__all__ = [
    "CAPTION",
    "CLASSES",
    "FOOTER",
    "FOOTNOTE",
    "FURNITURE",
    "HASHIYA",
    "HEADER",
    "MODELS",
    "OTHER",
    "PAGE_NUMBER",
    "READ",
    "TABLE",
    "TEXT",
    "Box",
    "DFine",
    "Detector",
    "Disagreement",
    "PPDocLayout",
    "Region",
    "box_disagreements",
    "holds",
    "load_model",
    "reading_order",
    "role_disagreements",
    "side_by_side",
]

_LAZY = {
    "PPDocLayout": ("detector", "PPDocLayout"),
    "Detector": ("detector", "Detector"),
    "DFine": ("dfine", "DFine"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module, attribute = _LAZY[name]
        return getattr(importlib.import_module(f".{module}", __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
