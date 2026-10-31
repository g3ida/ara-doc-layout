"""The model on three synthetic pages, against what it found when it was run
through ppdet's `Trainer` with its original training config — the path every
measurement in the README was taken on. Skipped where Paddle is not installed.

`data/expected.json` was written by that path. A difference here means the
config, the page preparation or the weights no longer match the measured model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Skipped only without Paddle. With it, PaddleDetection must import: a skip
# here hid an install whose ppdet could not find `pkg_resources`.
pytest.importorskip("paddle")

from PIL import Image  # noqa: E402

from ara_doc_layout import Detector  # noqa: E402
from ara_doc_layout.cli import main  # noqa: E402

pytestmark = pytest.mark.detector

DATA = Path(__file__).with_name("data")
EXPECTED = json.loads((DATA / "expected.json").read_text())
#: Pixels; GPU kernels are not bit-for-bit repeatable from run to run.
BOX_TOLERANCE = 0.5
SCORE_TOLERANCE = 0.005


@pytest.fixture(scope="module")
def detector() -> Detector:
    return Detector()


@pytest.mark.parametrize("page", sorted(EXPECTED))
def test_the_page_comes_out_as_the_trained_model_saw_it(detector, page) -> None:
    regions = detector.detect(DATA / f"{page}.png")

    expected = EXPECTED[page]
    assert [r.kind for r in regions] == [r["kind"] for r in expected]
    for region, want in zip(regions, expected, strict=True):
        got = (region.box.x0, region.box.y0, region.box.x1, region.box.y1)
        assert max(abs(a - b) for a, b in zip(got, want["box"], strict=True)) < BOX_TOLERANCE
        assert abs(region.score - want["score"]) < SCORE_TOLERANCE


def test_a_page_in_memory_reads_as_the_same_page_on_disk(detector) -> None:
    path = DATA / "two_columns.png"
    with Image.open(path) as image:
        in_memory = detector.detect(image)

    assert in_memory == detector.detect(path)


def test_a_higher_floor_keeps_only_the_surer_regions(detector) -> None:
    sure = detector.detect(DATA / "two_columns.png", score=0.9)

    assert sure and all(r.score >= 0.9 for r in sure)
    assert len(sure) < len(EXPECTED["two_columns"])


def test_the_command_prints_each_page_and_draws_it(tmp_path, capfd) -> None:
    """ppdet's log lines went to stdout among the pages, which a pipe into a
    JSON reader choked on."""
    main([str(DATA / "two_columns.png"), "--draw", str(tmp_path)])

    lines = capfd.readouterr().out.splitlines()
    assert len(lines) == 1
    page = json.loads(lines[0])
    assert [r["kind"] for r in page["regions"]] == [r["kind"] for r in EXPECTED["two_columns"]]
    # Right column, left column, then the notes; running head and folio unread.
    assert [page["regions"][i]["kind"] for i in page["reading_order"]] == [
        "text",
        "text",
        "footnote",
    ]
    assert (tmp_path / "two_columns.png").is_file()


def test_the_command_reads_with_both_models_and_says_where_they_part(capfd) -> None:
    pytest.importorskip("onnxruntime")
    other = json.loads((DATA / "expected_dfine.json").read_text())["two_columns"]

    main([str(DATA / "two_columns.png"), "--compare"])

    page = json.loads(capfd.readouterr().out)
    assert page["model"] == "pp-doclayout"
    assert page["compared_with"]["model"] == "dfine"
    assert [r["kind"] for r in page["compared_with"]["regions"]] == [r["kind"] for r in other]
    # Both read the two columns as body and the notes as notes.
    assert page["disagreements"] == []
