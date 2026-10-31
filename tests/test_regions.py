"""The geometry: which order a page's regions are read in, and which pages hold
regions side by side. Plain Python — none of these load the model."""

from __future__ import annotations

from ara_doc_layout import (
    FOOTNOTE,
    HASHIYA,
    HEADER,
    OTHER,
    PAGE_NUMBER,
    TEXT,
    Box,
    Region,
    holds,
    reading_order,
    side_by_side,
)

WIDTH = 400.0


def _region(kind: str, x0: float, y0: float, x1: float, y1: float) -> Region:
    return Region(kind, Box(x0, y0, x1, y1))


# ── reading order ───────────────────────────────────────────────────────────


def test_columns_are_read_right_to_left_and_footnotes_after_them() -> None:
    left = _region(TEXT, 40, 60, 195, 400)
    right = _region(TEXT, 205, 60, 360, 400)
    notes = _region(FOOTNOTE, 40, 420, 360, 480)
    head = _region(HEADER, 150, 30, 250, 45)

    order = reading_order([notes, left, head, right], WIDTH)

    assert order == [right, left, notes]


def test_a_gloss_is_read_after_the_text_whichever_margin_it_is_in() -> None:
    """Read right to left, a gloss in the right margin came before its text."""
    gloss = _region(HASHIYA, 280, 60, 360, 300)
    body = _region(TEXT, 40, 60, 270, 400)
    notes = _region(FOOTNOTE, 40, 420, 360, 480)

    assert reading_order([gloss, notes, body], WIDTH) == [body, gloss, notes]


def test_a_title_above_a_column_is_read_before_it() -> None:
    title = _region(TEXT, 120, 60, 300, 80)
    body = _region(TEXT, 40, 90, 360, 400)

    assert reading_order([body, title], WIDTH) == [title, body]


def test_footnotes_are_read_top_down_whatever_column_they_sit_in() -> None:
    lower = _region(FOOTNOTE, 205, 450, 360, 480)
    upper = _region(FOOTNOTE, 40, 420, 195, 440)

    assert reading_order([lower, upper], WIDTH) == [upper, lower]


def test_furniture_and_figures_are_not_read() -> None:
    folio = _region(PAGE_NUMBER, 190, 500, 210, 510)
    figure = _region(OTHER, 40, 60, 360, 200)

    assert reading_order([folio, figure], WIDTH) == []


# ── side by side ────────────────────────────────────────────────────────────


def test_columns_and_a_gloss_share_a_band_and_stacked_blocks_do_not() -> None:
    body = _region(TEXT, 140, 60, 360, 400)
    gloss = _region(HASHIYA, 40, 80, 125, 300)
    notes = _region(FOOTNOTE, 40, 420, 360, 480)
    figure = _region(OTHER, 40, 60, 125, 400)

    assert side_by_side([body, gloss])
    assert not side_by_side([body, notes])
    # A figure beside the text is nothing to read; reading the page whole
    # does not run its lines into anything.
    assert not side_by_side([body, figure, notes])


def test_two_detections_of_one_block_are_not_columns() -> None:
    """Overlapping boxes over one paragraph are a doubled detection, not a gutter."""
    first = _region(TEXT, 40, 60, 360, 400)
    second = _region(TEXT, 50, 70, 350, 390)

    assert not side_by_side([first, second])


# ── boxes and regions ───────────────────────────────────────────────────────


def test_a_box_is_held_by_its_centre() -> None:
    region = Box(100, 100, 200, 200)

    assert holds(region, Box(150, 150, 240, 160))
    assert not holds(region, Box(190, 150, 260, 160))
    assert holds(region, Box(190, 150, 260, 160), slack=30)


def test_a_region_survives_its_json_form() -> None:
    region = Region(FOOTNOTE, Box(10.123, 20.5, 300.0, 400.987), 0.91234)

    back = Region.from_dict(region.to_dict())

    assert back == Region(FOOTNOTE, Box(10.12, 20.5, 300.0, 400.99), 0.912)


def test_a_region_scales_from_pixels_to_points() -> None:
    region = _region(TEXT, 150, 300, 450, 600)

    assert region.scaled(72 / 150).box == Box(72, 144, 216, 288)
