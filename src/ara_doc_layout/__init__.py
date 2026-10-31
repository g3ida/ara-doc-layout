"""Layout detection for pages of Arabic books.

Importing the package does not import Paddle: the geometry is plain Python, and
:class:`Detector` loads the model only when it is built.
"""

from __future__ import annotations

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

__version__ = "0.1.0"

__all__ = [
    "CAPTION",
    "CLASSES",
    "FOOTER",
    "FOOTNOTE",
    "FURNITURE",
    "HASHIYA",
    "HEADER",
    "OTHER",
    "PAGE_NUMBER",
    "READ",
    "TABLE",
    "TEXT",
    "Box",
    "Detector",
    "Region",
    "holds",
    "reading_order",
    "side_by_side",
]


def __getattr__(name: str):
    if name == "Detector":
        from .detector import Detector

        return Detector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
