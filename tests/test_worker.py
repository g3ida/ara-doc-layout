"""The worker's protocol, against a stand-in for the model: what a host in
another interpreter reads back, and that one bad page does not end it."""

from __future__ import annotations

import io
import json

from ara_doc_layout import TEXT, Box, Region
from ara_doc_layout.worker import MARK, serve


class _Detector:
    def detect(self, image):
        if image.endswith("broken.png"):
            raise OSError("cannot identify image file")
        return [Region(TEXT, Box(1, 2, 3, 4), 0.9)]


def _answers(requests: str, mark: str = MARK) -> list[dict]:
    out = io.StringIO()
    serve(_Detector(), io.StringIO(requests), out, mark)
    lines = out.getvalue().splitlines()
    assert all(line.startswith(mark) for line in lines)
    return [json.loads(line[len(mark) :]) for line in lines]


def test_the_worker_says_it_is_ready_before_anything_else() -> None:
    assert _answers("") == [{"ready": True}]


def test_every_image_gets_an_answer_in_order() -> None:
    answers = _answers(json.dumps({"images": ["a.png", "b.png"]}) + "\n")

    assert answers[1:] == [
        {"regions": [{"kind": "text", "box": [1, 2, 3, 4], "score": 0.9}]},
        {"regions": [{"kind": "text", "box": [1, 2, 3, 4], "score": 0.9}]},
    ]


def test_a_page_that_fails_is_an_error_and_the_next_is_still_read() -> None:
    answers = _answers(
        json.dumps({"images": ["broken.png"]}) + "\n\n" + json.dumps({"images": ["c.png"]}) + "\n"
    )

    assert answers[1] == {"error": "OSError: cannot identify image file"}
    assert "regions" in answers[2]


def test_the_host_chooses_the_mark_it_listens_for() -> None:
    assert _answers("", mark="@@apdf ") == [{"ready": True}]
