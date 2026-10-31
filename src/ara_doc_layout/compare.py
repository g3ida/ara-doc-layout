"""Where two detections of one page disagree.

Two models trained on the same pages with different architectures make
different mistakes, so where they part is where a page is worth a look. Most
of their differences change nothing for whoever reads the page, though: a
paragraph boxed as one block or as two, a page number boxed a little wide.
:func:`role_disagreements` keeps only the ones that change what the ink is read
*as* — body, footnote or gloss — or whether it is read at all;
:func:`box_disagreements` keeps every box the two do not draw alike.

Pure geometry: either side may come from any model, or from a person.
"""

from __future__ import annotations

from dataclasses import dataclass

from .regions import CAPTION, FOOTNOTE, HASHIYA, TABLE, TEXT, Box, Region

BODY = "body"
#: What each class of region is read as. A table and a caption are read with
#: the text; running heads, folios and figures are not read at all.
ROLES = {TEXT: BODY, TABLE: BODY, CAPTION: BODY, FOOTNOTE: FOOTNOTE, HASHIYA: HASHIYA}

#: A region whose area the other side covers less than this with regions of
#: its own role is read differently by the two.
COVERED = 0.5
#: Two boxes are one region when they overlap by this much, as strict F1
#: matches a prediction to the truth.
SAME_BOX = 0.5


@dataclass(frozen=True, slots=True)
class Disagreement:
    #: The region as the side that drew it drew it.
    region: Region
    #: 0 when the first detection drew it, 1 when the second did.
    by: int


def _area(box: Box) -> float:
    return max(0.0, box.x1 - box.x0) * max(0.0, box.y1 - box.y0)


def _overlap(a: Box, b: Box) -> float:
    return _area(Box(max(a.x0, b.x0), max(a.y0, b.y0), min(a.x1, b.x1), min(a.y1, b.y1)))


def iou(a: Box, b: Box) -> float:
    shared = _overlap(a, b)
    union = _area(a) + _area(b) - shared
    return shared / union if union > 0 else 0.0


def _covered(box: Box, others: list[Box]) -> float:
    area = _area(box)
    if not area:
        return 1.0
    return min(1.0, sum(_overlap(box, other) for other in others) / area)


def _unread(mine: list[Region], theirs: list[Region], covered: float) -> list[Region]:
    """Regions of *mine* that *theirs* does not read in the same role."""
    out = []
    for region in mine:
        role = ROLES.get(region.kind)
        if role is None:
            continue
        same = [r.box for r in theirs if ROLES.get(r.kind) == role]
        if _covered(region.box, same) < covered:
            out.append(region)
    return out


def role_disagreements(
    first: list[Region], second: list[Region], *, covered: float = COVERED
) -> list[Disagreement]:
    """Regions one side reads in a role the other does not cover.

    A footnote one side calls text, a running head one side reads as body, a
    gloss one side misses: each comes back as the region of the side that read
    it, so a disagreement over a footnote called text is two — the first's
    footnote and the second's text.
    """
    return [Disagreement(r, 0) for r in _unread(first, second, covered)] + [
        Disagreement(r, 1) for r in _unread(second, first, covered)
    ]


def box_disagreements(
    first: list[Region], second: list[Region], *, same_box: float = SAME_BOX
) -> list[Disagreement]:
    """Every region the two do not draw alike: matched one to one, most
    overlapping first, a region left over or matched to another class."""
    pairs = sorted(
        ((iou(a.box, b.box), i, j) for i, a in enumerate(first) for j, b in enumerate(second)),
        key=lambda p: -p[0],
    )
    matched_first: dict[int, int] = {}
    matched_second: set[int] = set()
    for overlap, i, j in pairs:
        if overlap < same_box:
            break
        if i in matched_first or j in matched_second:
            continue
        matched_first[i] = j
        matched_second.add(j)
    out = []
    for i, region in enumerate(first):
        j = matched_first.get(i)
        if j is None:
            out.append(Disagreement(region, 0))
        elif second[j].kind != region.kind:
            out += [Disagreement(region, 0), Disagreement(second[j], 1)]
    out += [Disagreement(r, 1) for j, r in enumerate(second) if j not in matched_second]
    return out
