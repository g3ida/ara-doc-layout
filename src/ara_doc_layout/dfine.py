"""D-FINE-L, fine-tuned on the same pages, exported to ONNX and run on onnxruntime.

A different architecture from PP-DocLayout_plus-L trained on the same data, so
the two fail on different pages — which is what makes their disagreements
worth reading (`compare`). It needs neither Paddle nor PyTorch: the checkpoint
was exported the way D-FINE's own deployment script does, and on the CPU the
export finds the same boxes as the PyTorch model to a ten-thousandth of a
pixel. Traced at one page per call.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .detector import SCORE
from .regions import CLASSES, Box, Region
from .weights import DFINE, weights_path

#: Every page is squeezed to this square, as in training.
SIZE = 640


class DFine:
    """Loads the model once; :meth:`detect` answers for one page at a time."""

    def __init__(
        self,
        *,
        weights: str | Path | None = None,
        device: str | None = None,
        score: float = SCORE,
    ) -> None:
        import onnxruntime as ort

        weights = Path(weights) if weights is not None else weights_path(DFINE)
        if not weights.is_file():
            raise FileNotFoundError(f"No weights at {weights}.")
        available = ort.get_available_providers()
        # The GPU when onnxruntime was built for one, as PPDocLayout does.
        self.device = device or ("gpu" if "CUDAExecutionProvider" in available else "cpu")
        if self.device.startswith("gpu"):
            index = int(self.device.partition(":")[2] or 0)
            providers = [("CUDAExecutionProvider", {"device_id": index}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.score = score
        self._session = ort.InferenceSession(str(weights), providers=providers)

    def detect(self, image, score: float | None = None) -> list[Region]:
        """The regions on one page, in its own pixels, most confident first.

        *image* is a path, a PIL image or an RGB array. Every page is squeezed
        to 640×640 on the way in.
        """
        import numpy as np
        from PIL import Image

        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                page = opened.convert("RGB")
        elif isinstance(image, Image.Image):
            page = image.convert("RGB")
        else:
            page = Image.fromarray(np.asarray(image)).convert("RGB")
        # Bilinear, as torchvision's `Resize` does to a PIL image in training.
        pixels = np.asarray(page.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32)
        labels, boxes, scores = self._session.run(
            None,
            {
                "images": pixels.transpose(2, 0, 1)[None] / 255.0,
                "orig_target_sizes": np.array([[page.width, page.height]], dtype=np.int64),
            },
        )
        floor = self.score if score is None else score
        regions = [
            # Its class ids start at 1: the training data's are COCO's, and id 0
            # is a slot no region uses.
            Region(CLASSES[int(label) - 1], Box(*(float(v) for v in box)), float(confidence))
            for label, box, confidence in zip(labels[0], boxes[0], scores[0], strict=True)
            if confidence >= floor and 1 <= int(label) <= len(CLASSES)
        ]
        return sorted(regions, key=lambda r: -r.score)

    def detect_many(self, images: Iterable, score: float | None = None) -> list[list[Region]]:
        return [self.detect(image, score) for image in images]
