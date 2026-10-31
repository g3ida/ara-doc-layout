"""The models by name, for whoever picks one from a string: the command and the worker.

Two architectures trained on the same pages. Each needs its own runtime and
neither is imported until it is asked for, so an install with only one of them
can load that one.
"""

from __future__ import annotations

import importlib

#: name → (module, class)
MODELS = {
    "pp-doclayout": ("ara_doc_layout.detector", "PPDocLayout"),
    "dfine": ("ara_doc_layout.dfine", "DFine"),
}


def load_model(name: str, **options):
    """The model called *name*, built with *options* (`weights`, `device`, `score`)."""
    if name not in MODELS:
        raise ValueError(f"No model called {name!r}: {', '.join(MODELS)}.")
    module, cls = MODELS[name]
    return getattr(importlib.import_module(module), cls)(**options)
