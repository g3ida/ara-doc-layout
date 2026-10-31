"""Where two detections of a page disagree: the differences that change what
the ink is read as, and the ones that do not. Plain geometry; no model runs."""

from __future__ import annotations

from ara_doc_layout import (
    FOOTNOTE,
    HASHIYA,
    HEADER,
    PAGE_NUMBER,
    TEXT,
    Box,
    Disagreement,
    Region,
    box_disagreements,
    role_disagreements,
)


def _r(kind: str, x0: float, y0: float, x1: float, y1: float) -> Region:
    return Region(kind, Box(x0, y0, x1, y1))


BODY = _r(TEXT, 40, 60, 360, 400)
NOTES = _r(FOOTNOTE, 40, 420, 360, 480)
FOLIO = _r(PAGE_NUMBER, 190, 500, 210, 510)


def test_two_detections_that_agree_have_nothing_to_say() -> None:
    page = [BODY, NOTES, FOLIO]

    assert role_disagreements(page, list(page)) == []
    assert box_disagreements(page, list(page)) == []


def test_footnotes_read_as_text_by_one_side_are_a_disagreement_on_both() -> None:
    as_text = _r(TEXT, 40, 420, 360, 480)

    found = role_disagreements([BODY, NOTES], [BODY, as_text])

    assert found == [Disagreement(NOTES, 0), Disagreement(as_text, 1)]


def test_a_paragraph_split_in_three_is_read_the_same() -> None:
    """Every box differs, and nothing about what is read does. (In halves, one
    half overlaps the whole by exactly a half and is matched to it.)"""
    thirds = [_r(TEXT, 40, y, 360, y + 340 / 3) for y in (60, 60 + 340 / 3, 60 + 680 / 3)]

    assert role_disagreements([BODY], thirds) == []
    assert len(box_disagreements([BODY], thirds)) == 4


def test_a_page_number_boxed_differently_is_not_read_either_way() -> None:
    wider = _r(PAGE_NUMBER, 180, 495, 240, 515)

    assert role_disagreements([BODY, FOLIO], [BODY, wider]) == []
    assert box_disagreements([BODY, FOLIO], [BODY, wider]) == [
        Disagreement(FOLIO, 0),
        Disagreement(wider, 1),
    ]


def test_a_running_head_one_side_reads_as_text_is_flagged_on_that_side() -> None:
    head = _r(HEADER, 150, 30, 250, 45)
    head_as_text = _r(TEXT, 150, 30, 250, 45)

    assert role_disagreements([BODY, head], [BODY, head_as_text]) == [Disagreement(head_as_text, 1)]


def test_a_gloss_one_side_misses_is_flagged_where_it_was_found() -> None:
    gloss = _r(HASHIYA, 370, 60, 395, 300)

    assert role_disagreements([BODY], [BODY, gloss]) == [Disagreement(gloss, 1)]


def test_a_box_matched_to_another_class_is_a_disagreement_on_each_side() -> None:
    relabelled = _r(FOOTNOTE, 40, 60, 360, 400)

    assert box_disagreements([BODY], [relabelled]) == [
        Disagreement(BODY, 0),
        Disagreement(relabelled, 1),
    ]
