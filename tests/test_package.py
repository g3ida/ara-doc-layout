"""What an install must carry, and what importing it must not cost."""

from __future__ import annotations

import re

from PIL import Image

from ara_doc_layout import CLASSES, TEXT, Box, Region
from ara_doc_layout.detector import CONFIG
from ara_doc_layout.draw import draw


def test_the_config_has_as_many_classes_as_the_names_given_to_them() -> None:
    declared = re.search(r"^num_classes:\s*(\d+)", CONFIG.read_text(), re.M)

    assert declared and int(declared.group(1)) == len(CLASSES)


def test_importing_the_package_leaves_both_runtimes_unloaded() -> None:
    import subprocess
    import sys

    probe = (
        "import sys, ara_doc_layout; print('paddle' in sys.modules, 'onnxruntime' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "False False"


def test_a_model_is_asked_for_by_one_of_its_names() -> None:
    import pytest

    from ara_doc_layout import MODELS, load_model

    with pytest.raises(ValueError, match="pp-doclayout, dfine"):
        load_model("yolo")
    assert set(MODELS) == {"pp-doclayout", "dfine"}


def test_a_drawn_page_keeps_its_size() -> None:
    page = Image.new("L", (300, 400), "white")

    drawn = draw(page, [Region(TEXT, Box(20, 30, 280, 380), 0.97)])

    assert drawn.size == page.size and drawn.mode == "RGB"
    assert drawn.getpixel((20, 200)) != (255, 255, 255)
