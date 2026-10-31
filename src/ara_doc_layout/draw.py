"""A page with its regions drawn on it, for looking at what the detector did."""

from __future__ import annotations

from .regions import (
    CAPTION,
    FOOTER,
    FOOTNOTE,
    HASHIYA,
    HEADER,
    OTHER,
    PAGE_NUMBER,
    TABLE,
    TEXT,
    Region,
)

#: One hue per class, chosen to stay apart from each other on a white page and
#: on a yellowed scan alike.
COLOURS = {
    TEXT: (31, 119, 180),
    FOOTNOTE: (214, 39, 40),
    HASHIYA: (148, 103, 189),
    HEADER: (44, 160, 44),
    FOOTER: (23, 190, 207),
    PAGE_NUMBER: (188, 189, 34),
    TABLE: (255, 127, 14),
    CAPTION: (227, 119, 194),
    OTHER: (127, 127, 127),
}


def draw(image, regions: list[Region], *, width: int | None = None):
    """A copy of *image* with each region outlined, tinted and labelled.

    *regions* are in the image's own pixels, as :meth:`Detector.detect` gives
    them; the line width follows the page's size unless *width* says.
    """
    from PIL import Image, ImageDraw

    page = image.convert("RGB")
    width = width or max(2, round(max(page.size) / 400))
    tint = Image.new("RGBA", page.size, (0, 0, 0, 0))
    ink = ImageDraw.Draw(tint)
    for region in regions:
        colour = COLOURS.get(region.kind, COLOURS[OTHER])
        box = (region.box.x0, region.box.y0, region.box.x1, region.box.y1)
        ink.rectangle(box, fill=(*colour, 40), outline=(*colour, 255), width=width)
    page = Image.alpha_composite(page.convert("RGBA"), tint).convert("RGB")
    labels = ImageDraw.Draw(page)
    size = max(12, round(max(page.size) / 90))
    font = _font(size)
    for region in regions:
        colour = COLOURS.get(region.kind, COLOURS[OTHER])
        text = f"{region.kind} {region.score:.2f}"
        x, y = region.box.x0, max(0, region.box.y0 - size - 4)
        left, top, right, bottom = labels.textbbox((x, y), text, font=font)
        labels.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=colour)
        labels.text((x, y), text, fill="white", font=font)
    return page


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1 has one size only
        return ImageFont.load_default()
