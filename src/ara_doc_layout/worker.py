"""The detector in a process of its own, for a program under another Python.

Paddle pins a CUDA build that an OCR or TTS environment rarely shares, so the
program that wants regions starts this under the interpreter that has Paddle
and speaks JSON lines to it:

    python -I -m ara_doc_layout.worker [--device cpu] [--score 0.5] [--mark "@@x "]

* the first answer is ``{"ready": true}``, once the model is loaded;
* each request ``{"images": [path, …]}`` gets one answer per image, in order:
  ``{"regions": [{"kind", "box", "score"}, …]}`` with the box in that image's
  pixels, or ``{"error": "…"}`` — one bad page does not end the worker;
* end of input ends it.

Answers are prefixed with the mark, because Paddle prints to stdout too and its
chatter must not be read as a reply.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TextIO

from .detector import CONFIG, SCORE

MARK = "@@ara-doc-layout "


def serve(detector, requests: TextIO, answers: TextIO, mark: str = MARK) -> None:
    """Answer every request on *requests* until it ends."""

    def send(payload: dict) -> None:
        answers.write(mark + json.dumps(payload) + "\n")
        answers.flush()

    send({"ready": True})
    for line in requests:
        if not line.strip():
            continue
        for image in json.loads(line)["images"]:
            try:
                send({"regions": [r.to_dict() for r in detector.detect(image)]})
            except Exception as exc:  # noqa: BLE001 - one bad page must not end the run
                send({"error": f"{type(exc).__name__}: {exc}"})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ara_doc_layout.worker")
    parser.add_argument("--weights", help="a .pdparams file; the released checkpoint if not given")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--device", help="gpu, gpu:1 or cpu; the GPU when Paddle was built for one")
    parser.add_argument("--score", type=float, default=SCORE)
    parser.add_argument("--mark", default=MARK, help="prefix of every answer line")
    args = parser.parse_args(argv)

    from .detector import Detector

    detector = Detector(
        weights=args.weights, config=args.config, device=args.device, score=args.score
    )
    serve(detector, sys.stdin, sys.stdout, args.mark)


if __name__ == "__main__":
    main()
