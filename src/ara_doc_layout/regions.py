"""What the detector finds, and the order a page's regions are read in.

Nothing here needs Paddle: a program that only orders regions, or reads them
from another process, imports this and not the model.

The classes are the checkpoint's own, and coarser than a typesetter's on
purpose: `text` holds headings and verse as well as prose, because a detector
asked to tell them apart was judging what only the words can settle.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── the classes ─────────────────────────────────────────────────────────────

TEXT = "text"
FOOTNOTE = "footnote"
HASHIYA = "hashiya"
HEADER = "header"
FOOTER = "footer"
PAGE_NUMBER = "page_number"
TABLE = "table"
CAPTION = "caption"
OTHER = "other"

#: In the order the checkpoints' class ids were trained with. Baked into the
#: weights: reordering it renames every region.
CLASSES = (
    TEXT,
    FOOTNOTE,
    HASHIYA,
    HEADER,
    FOOTER,
    PAGE_NUMBER,
    TABLE,
    CAPTION,
    OTHER,
)

#: Never read aloud: running heads and folios.
FURNITURE = frozenset({HEADER, FOOTER, PAGE_NUMBER})
#: What a reader is pointed at. `other` is a figure, an ornament or a stamp —
#: but text printed over an illustration lands in it too, so whoever reads the
#: page whole should not throw away what it found there.
READ = frozenset({TEXT, FOOTNOTE, HASHIYA, CAPTION, TABLE})

#: Regions whose x-centres are closer than this share of the page width are one
#: column.
COLUMN_GAP = 0.06

#: Two regions share a band of the page when they overlap vertically by this
#: share of the shorter one…
SIDE_BY_SIDE_OVERLAP = 0.3
#: …and horizontally by less than this share of the narrower one.
SIDE_BY_SIDE_APART = 0.2


@dataclass(frozen=True, slots=True)
class Box:
    """Top-left origin, in whatever unit the page was measured in — image
    pixels from the detector, page points once a caller has scaled them."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def scaled(self, sx: float, sy: float | None = None) -> Box:
        sy = sx if sy is None else sy
        return Box(self.x0 * sx, self.y0 * sy, self.x1 * sx, self.y1 * sy)


@dataclass(frozen=True, slots=True)
class Region:
    kind: str
    box: Box
    score: float = 1.0

    def scaled(self, sx: float, sy: float | None = None) -> Region:
        return Region(self.kind, self.box.scaled(sx, sy), self.score)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "box": [round(v, 2) for v in (self.box.x0, self.box.y0, self.box.x1, self.box.y1)],
            "score": round(self.score, 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Region:
        return cls(data["kind"], Box(*data["box"]), data.get("score", 1.0))


# ── reading order ───────────────────────────────────────────────────────────


def _columns(regions: list[Region], page_width: float) -> list[list[Region]]:
    """Regions clustered into columns on their x-centre, rightmost first."""
    if not regions:
        return []
    gap = page_width * COLUMN_GAP
    centred = sorted(regions, key=lambda r: (r.box.x0 + r.box.x1) / 2)
    columns, current = [], [centred[0]]
    for previous, region in zip(centred, centred[1:], strict=False):
        if (region.box.x0 + region.box.x1 - previous.box.x0 - previous.box.x1) / 2 <= gap:
            current.append(region)
        else:
            columns.append(current)
            current = [region]
    columns.append(current)
    columns.sort(key=lambda c: -sum(r.box.x0 + r.box.x1 for r in c) / len(c))
    return [sorted(c, key=lambda r: r.box.y0) for c in columns]


def reading_order(regions: list[Region], page_width: float) -> list[Region]:
    """The regions to be read, in order: the text's columns right to left, then
    the gloss, then the footnotes. Furniture and figures are left out.

    Right on ordinary pages, a guess on stacked articles — a geometric
    pre-fill, not a reading. The gloss comes after the text whichever margin it
    is printed in: read purely right to left, a gloss in the right margin came
    before the text it comments on.
    """
    flow = [r for r in regions if r.kind in READ and r.kind not in (FOOTNOTE, HASHIYA)]
    glosses = [r for r in regions if r.kind == HASHIYA]
    notes = sorted((r for r in regions if r.kind == FOOTNOTE), key=lambda r: r.box.y0)
    return (
        [r for column in _columns(flow, page_width) for r in column]
        + [r for column in _columns(glosses, page_width) for r in column]
        + notes
    )


def side_by_side(regions: list[Region]) -> bool:
    """Whether two regions to be read share a band of the page.

    Columns, or a gloss beside the text: the pages where an OCR engine that
    finds lines across the whole width joins a line of one to the line of the
    other at the same height, and the two texts come out interleaved.
    """
    boxes = [r.box for r in regions if r.kind in READ]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            vertical = min(a.y1, b.y1) - max(a.y0, b.y0)
            horizontal = min(a.x1, b.x1) - max(a.x0, b.x0)
            if vertical > SIDE_BY_SIDE_OVERLAP * min(
                a.height, b.height
            ) and horizontal < SIDE_BY_SIDE_APART * min(a.width, b.width):
                return True
    return False


def holds(region: Box, box: Box, slack: float = 0.0) -> bool:
    """Whether *box*'s centre lies in *region*, widened by *slack*."""
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    return (
        region.x0 - slack <= cx <= region.x1 + slack
        and region.y0 - slack <= cy <= region.y1 + slack
    )
