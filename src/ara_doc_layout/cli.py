"""`ara-doc-layout PAGE…`: the regions of each page, one JSON line per page."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .detector import SCORE
from .regions import reading_order


def _own_stdout():
    """Standard output for the pages alone, and everything else sent to stderr.

    ppdet prints warnings when it is imported and its log lines to stdout, and
    Paddle's custom operators print there from C++: piped into `jq`, the
    output was not JSON. So the JSON keeps the real stdout and file descriptor
    1 is pointed at stderr for whoever else writes to it — and `sys.stdout`
    too, for when it is not file descriptor 1, as under pytest.
    """
    sys.stdout.flush()
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ara-doc-layout",
        description="Find the text, footnotes, glosses, running heads, page "
        "numbers, tables, captions and figures on pages of Arabic books.",
    )
    parser.add_argument("pages", nargs="+", type=Path, help="page images")
    parser.add_argument("--device", help="gpu, gpu:1 or cpu; the GPU when Paddle was built for one")
    parser.add_argument("--score", type=float, default=SCORE)
    parser.add_argument("--weights", help="a .pdparams file; the released checkpoint if not given")
    parser.add_argument(
        "--draw", type=Path, metavar="DIR", help="also write each page with its regions drawn"
    )
    args = parser.parse_args(argv)

    out = _own_stdout()
    from PIL import Image

    from .detector import Detector

    detector = Detector(weights=args.weights, device=args.device, score=args.score)
    if args.draw:
        from .draw import draw

        args.draw.mkdir(parents=True, exist_ok=True)
    for path in args.pages:
        with Image.open(path) as image:
            page = image.convert("RGB")
        regions = detector.detect(page)
        order = reading_order(regions, page.width)
        answer = {
            "page": str(path),
            "size": [page.width, page.height],
            "regions": [r.to_dict() for r in regions],
            "reading_order": [regions.index(r) for r in order],
        }
        out.write(json.dumps(answer) + "\n")
        if args.draw:
            draw(page, regions).save(args.draw / f"{path.stem}.png")


if __name__ == "__main__":
    sys.exit(main())
