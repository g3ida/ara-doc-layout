"""`ara-doc-layout PAGE…`: the regions of each page, one JSON line per page.

With `--compare`, both models read every page, and each line also says where
they disagree about what the ink is read as (`compare.role_disagreements`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .compare import role_disagreements
from .detector import SCORE
from .models import MODELS, load_model
from .regions import reading_order


def _own_stdout():
    """Standard output for the pages alone, and everything else sent to stderr.

    ppdet prints warnings when it is imported and its log lines to stdout, and
    Paddle's custom operators print there from C++: piped into `jq`, the
    output was not JSON. onnxruntime writes to it from C++ as well. So the JSON keeps the real stdout and file descriptor
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
    parser.add_argument("--model", choices=list(MODELS), default="pp-doclayout")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="read with the other model too, and say where they disagree",
    )
    parser.add_argument("--device", help="gpu, gpu:1 or cpu; the GPU when the runtime has one")
    parser.add_argument("--score", type=float, default=SCORE)
    parser.add_argument(
        "--weights", help="a checkpoint of your own for --model; the released one if not given"
    )
    parser.add_argument(
        "--draw", type=Path, metavar="DIR", help="also write each page with its regions drawn"
    )
    args = parser.parse_args(argv)

    out = _own_stdout()
    from PIL import Image

    model = load_model(args.model, weights=args.weights, device=args.device, score=args.score)
    other_name = next(name for name in MODELS if name != args.model)
    other = load_model(other_name, device=args.device, score=args.score) if args.compare else None
    if args.draw:
        from .draw import draw

        args.draw.mkdir(parents=True, exist_ok=True)
    for path in args.pages:
        with Image.open(path) as image:
            page = image.convert("RGB")
        regions = model.detect(page)
        order = reading_order(regions, page.width)
        answer = {
            "page": str(path),
            "size": [page.width, page.height],
            "model": args.model,
            "regions": [r.to_dict() for r in regions],
            "reading_order": [regions.index(r) for r in order],
        }
        if other is not None:
            theirs = other.detect(page)
            names = (args.model, other_name)
            answer["compared_with"] = {
                "model": other_name,
                "regions": [r.to_dict() for r in theirs],
            }
            answer["disagreements"] = [
                {"model": names[d.by], **d.region.to_dict()}
                for d in role_disagreements(regions, theirs)
            ]
        out.write(json.dumps(answer) + "\n")
        if args.draw:
            draw(page, regions).save(args.draw / f"{path.stem}.png")


if __name__ == "__main__":
    sys.exit(main())
