# ara-doc-layout

Arabic document page layout analyzer.

![Two real pages with PP-DocLayout_plus-L's regions drawn: a printed commentary with its gloss in the margin, and a modern typeset page](docs/example.png)

*Left:* Ibn Hamdun's gloss on al-Azhari's commentary on the *Ājurrūmiyya*,
page 10 — a public-domain print ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:%D8%AD%D8%A7%D8%B4%D9%8A%D8%A9_%D8%A7%D8%A8%D9%86_%D8%AD%D9%85%D8%AF%D9%88%D9%86_%D8%B9%D9%84%D9%89_%D8%B4%D8%B1%D8%AD_%D8%AE%D8%A7%D9%84%D8%AF_%D8%A7%D9%84%D8%A3%D8%B2%D9%87%D8%B1%D9%8A_%D8%B9%D9%84%D9%89_%D8%A7%D9%84%D8%A2%D8%AC%D8%B1%D9%88%D9%85%D9%8A%D8%A9.pdf),
Internet Archive `ajouroumya_ibnhamdoun`).
*Right:* Mayy Ziyada, *ʿĀʾisha Taymūr* (1926), page 15, Hindawi Foundation
edition, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:%D8%B9%D8%A7%D8%A6%D8%B4%D8%A9_%D8%AA%D9%8A%D9%85%D9%88%D8%B1_%D9%85%D9%8A_%D8%B2%D9%8A%D8%A7%D8%AF%D8%A9_1926.pdf)).
Both are shown with the detected regions drawn over them.

Two models, fine-tuned on the same 12,000 synthetic and 1,352 hand-labelled
real pages:

- **PP-DocLayout_plus-L** (RT-DETR-L on an HGNetV2 backbone), run through
  PaddleDetection.
- **D-FINE-L**, exported to ONNX and run on onnxruntime: no Paddle and no
  PyTorch.

They are different architectures, so they make different mistakes. Where they
disagree about a page is where it is worth a look (see *Comparing the two*).

## What it finds

Nine classes:

| class | Description |
|---|---|
| `text` | body text, including headings and text columns |
| `footnote` | the footnote apparatus, as **one** block |
| `hashiya` | a gloss set in the margin beside the main text |
| `table` | a table, (with or without borders) |
| `caption` | the caption of a figure or plate |
| `header` | running head |
| `footer` | running foot |
| `page_number` | does this need explanation ? |
| `other` | figures, ornaments, stamps... |

## Benchmarks

TODO

## Install

Both models, with an NVIDIA GPU:

```bash
pip install "ara-doc-layout[gpu] @ git+https://github.com/g3ida/ara-doc-layout.git" \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

Both models, on the CPU:

```bash
pip install "ara-doc-layout[cpu] @ git+https://github.com/g3ida/ara-doc-layout.git"
```

D-FINE-L alone, on onnxruntime: a 10-second install, no Paddle:

```bash
pip install "ara-doc-layout[dfine] @ git+https://github.com/g3ida/ara-doc-layout.git"
```

With no extra, only the region geometry and the comparison are installed
(`Region`, `reading_order`, `side_by_side`, `role_disagreements`). That is
enough for a program that talks to a model running in another environment.

- **The extra index:** Paddle's GPU wheels from 3.0 on are not on PyPI. They
  come from Paddle's own index, one per CUDA version. `cu129` is the build the
  numbers here were measured with; `cu118`, `cu126` and `cu130` carry the same
  3.3.1.
- **A virtualenv of its own:** Paddle pins numpy < 2 and brings its own CUDA
  libraries, which rarely sit well beside PyTorch.
- **The weights** are downloaded from the GitHub releases on first use into
  `~/.cache/ara-doc-layout/`: 132 MB for PP-DocLayout_plus-L, 124 MB for
  D-FINE-L. They are checked against their SHA-256 before use. To use a local
  copy, set `ARA_DOC_LAYOUT_WEIGHTS` to a `.pdparams` file for
  PP-DocLayout_plus-L, or `ARA_DOC_LAYOUT_DFINE_WEIGHTS` to an `.onnx` file
  for D-FINE-L.
- **PaddleDetection** is pinned to commit `65e6435`, the one the checkpoint
  was trained and measured with.

For development, from a clone:

```bash
pip install -e ".[gpu,test]" --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

## Use

### In Python

```python
from PIL import Image
from ara_doc_layout import PPDocLayout, reading_order, side_by_side

model = PPDocLayout()                      # or DFine(); on the GPU if the runtime has one
page = Image.open("page.png")
regions = model.detect(page)               # boxes in the image's own pixels

for region in reading_order(regions, page.width):
    print(region.kind, region.box, round(region.score, 2))

if side_by_side(regions):
    ...  # columns or a gloss beside the text: read region by region
```

- **What `detect` takes:** a path, a PIL image or an RGB array.
- **Resolution:** every page is squeezed to 800×800 (PP-DocLayout_plus-L) or
  640×640 (D-FINE-L) on the way in, so rendering a PDF page past ~150 dpi
  buys nothing.
- **Points, not pixels:** to put the boxes in PDF points, use
  `region.scaled(72 / dpi)`.
- **The floor:** `score=` sets the minimum confidence, per model or per call.
- **By name:** `load_model("pp-doclayout")` or `load_model("dfine")`.

### Comparing the two

```python
from ara_doc_layout import DFine, PPDocLayout, role_disagreements

first, second = PPDocLayout().detect(page), DFine().detect(page)
for d in role_disagreements(first, second):
    print(d.by, d.region.kind, d.region.box)   # by: 0 = first, 1 = second
```

`role_disagreements` keeps only the differences that change what the ink is
read as: body text, footnote or gloss, or not read at all. A footnote one model
calls text, a running head one reads as body, a gloss one misses. A paragraph
boxed as one block or as two, or a page number boxed a little wide, is not a
disagreement there. `box_disagreements` keeps every box the two draw
differently. Neither function cares which model produced which list: either
side can come from any detector, or from a person.

### From the command line

```bash
ara-doc-layout page-*.png --draw drawn/
ara-doc-layout page-*.png --model dfine
ara-doc-layout page-*.png --compare
```

Prints one JSON line per page: its size, its regions, and `reading_order`
(indices into `regions`). `--model` picks the model (`pp-doclayout` by
default). `--compare` reads each page with the other model as well, and adds
its regions and the two models' `disagreements`. With `--draw`, each page is
also written with its regions outlined. The runtimes' own chatter goes to
stderr, so the output can be piped.



## Speed

Measured on an RTX 5060 Laptop (8 GB) and a Core Ultra 9 275HX, with pages at
150 dpi:

| | load | per page (median) |
|---|---:|---:|
| PP-DocLayout_plus-L, GPU | 3.8 s | 48 ms |
| PP-DocLayout_plus-L, CPU (24 threads) | 4.1 s | 3.95 s |
| D-FINE-L, CPU (24 threads) | 1.2 s | 223 ms |

D-FINE-L has not been timed on the GPU.

## Licence

Apache-2.0, see [LICENSE](LICENSE). The base models (PP-DocLayout_plus-L and
D-FINE-L) and PaddleDetection are Apache-2.0 as well.