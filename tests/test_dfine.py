"""D-FINE-L on three synthetic pages, against what the PyTorch model it was
exported from found on them on the CPU. Skipped where onnxruntime is not
installed.

`data/expected_dfine.json` was written by the PyTorch model. A difference here
means the export, the page preparation or the class ids no longer match it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")

from PIL import Image  # noqa: E402

from ara_doc_layout import DFine  # noqa: E402

pytestmark = pytest.mark.dfine

DATA = Path(__file__).with_name("data")
EXPECTED = json.loads((DATA / "expected_dfine.json").read_text())
#: onnxruntime and PyTorch on the CPU agreed to a ten-thousandth of a pixel;
#: this leaves room for a GPU, whose kernels round differently.
BOX_TOLERANCE = 0.5
SCORE_TOLERANCE = 0.005


@pytest.fixture(scope="module")
def model() -> DFine:
    return DFine(device="cpu")


@pytest.mark.parametrize("page", sorted(EXPECTED))
def test_the_page_comes_out_as_the_pytorch_model_saw_it(model, page) -> None:
    regions = model.detect(DATA / f"{page}.png")

    expected = EXPECTED[page]
    assert [r.kind for r in regions] == [r["kind"] for r in expected]
    for region, want in zip(regions, expected, strict=True):
        got = (region.box.x0, region.box.y0, region.box.x1, region.box.y1)
        assert max(abs(a - b) for a, b in zip(got, want["box"], strict=True)) < BOX_TOLERANCE
        assert abs(region.score - want["score"]) < SCORE_TOLERANCE


def test_a_page_in_memory_reads_as_the_same_page_on_disk(model) -> None:
    path = DATA / "gloss.png"
    with Image.open(path) as image:
        in_memory = model.detect(image)

    assert in_memory == model.detect(path)


def test_regions_come_most_confident_first(model) -> None:
    scores = [r.score for r in model.detect(DATA / "one_column.png")]

    assert scores == sorted(scores, reverse=True)
