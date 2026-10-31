"""The model: a fine-tuned PP-DocLayout_plus-L, run through PaddleDetection.

Paddle 3.3 cannot export this model to a static graph — `export_model.py` fails
inside `hybrid_encoder` (`to_static`: "must be Value, but got Variable") — so it
runs the way it was trained, as a `ppdet` dygraph model built from its config.
What `ppdet`'s `Trainer` added around that was a test dataset read from an
annotation file, an `output/` directory made in the working directory, and
images only by path; this builds the same model, loads the same weights and
prepares a page with the same operators, and takes the page in memory.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .regions import CLASSES, Box, Region
from .weights import weights_path

CONFIG = Path(__file__).with_name("model") / "config.yml"

#: The operating point the checkpoint was selected at on the gold set; every
#: figure in the README is at it.
SCORE = 0.5


class Detector:
    """Loads the model once; :meth:`detect` answers for one page at a time.

    `ppdet` keeps the config it loads in a module-level registry, so a process
    holds one detector — a second built from another config would rebuild the
    first's modules from it.
    """

    def __init__(
        self,
        *,
        weights: str | Path | None = None,
        config: str | Path = CONFIG,
        device: str | None = None,
        score: float = SCORE,
    ) -> None:
        import paddle
        import ppdet  # noqa: F401 - registers every module the config names
        from ppdet.core.workspace import create, load_config
        from ppdet.data import transform
        from ppdet.utils.checkpoint import load_pretrain_weight

        weights = Path(weights) if weights is not None else weights_path()
        if not weights.is_file():
            raise FileNotFoundError(f"No weights at {weights}.")
        # ppdet finds the file by adding `.pdparams` to the name it is given
        # without its suffix, so a file named otherwise is never found.
        if weights.suffix != ".pdparams":
            raise ValueError(f"{weights}: ppdet loads only files ending in .pdparams.")
        self.score = score
        # The GPU when Paddle was built for one: a CPU install asked for the
        # GPU fails outright rather than falling back.
        self.device = device or ("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")
        paddle.set_device(self.device)
        cfg = load_config(str(config))
        steps = cfg["TestReader"]["sample_transforms"]
        self._model = create(cfg.architecture)
        self._model.load_meanstd(steps)
        load_pretrain_weight(self._model, str(weights))
        self._model.eval()
        self._steps = [
            getattr(transform, name)(**(args or {}))
            for step in steps
            for name, args in step.items()
            if name != "Decode"
        ]
        self._paddle = paddle

    def _sample(self, image) -> dict:
        """What ppdet's `Decode` makes of a file, from the page in hand: RGB,
        its own size, not yet scaled."""
        import numpy as np
        from PIL import Image

        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                pixels = np.asarray(opened.convert("RGB"))
        elif isinstance(image, Image.Image):
            pixels = np.asarray(image.convert("RGB"))
        else:
            pixels = np.asarray(image)
        sample = {
            "image": pixels,
            "h": pixels.shape[0],
            "w": pixels.shape[1],
            "im_shape": np.array(pixels.shape[:2], dtype=np.float32),
            "scale_factor": np.array([1.0, 1.0], dtype=np.float32),
        }
        for step in self._steps:
            sample = step(sample)
        return sample

    def detect(self, image, score: float | None = None) -> list[Region]:
        """The regions on one page, in its own pixels, most confident first.

        *image* is a path, a PIL image or an RGB array. Every page is squeezed
        to 800×800 on the way in, so resolution past ~150 dpi buys nothing.
        """
        floor = self.score if score is None else score
        sample = self._sample(image)
        paddle = self._paddle
        inputs = {
            key: paddle.to_tensor(sample[key][None])
            for key in ("image", "im_shape", "scale_factor")
        }
        with paddle.no_grad():
            outputs = self._model(inputs)
        regions = []
        for row in outputs["bbox"].numpy()[: int(outputs["bbox_num"].numpy()[0])]:
            label, confidence = int(row[0]), float(row[1])
            if confidence < floor or not 0 <= label < len(CLASSES):
                continue
            regions.append(Region(CLASSES[label], Box(*(float(v) for v in row[2:6])), confidence))
        return regions

    def detect_many(self, images: Iterable, score: float | None = None) -> list[list[Region]]:
        return [self.detect(image, score) for image in images]
