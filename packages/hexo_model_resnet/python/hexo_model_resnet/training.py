"""Internal training helpers for the ResNet model family.

`trainer.py` is the only object called by `hexo_train`. This module is reserved
for pure helper functions used by that trainer once real optimizer code exists;
it intentionally does not define a second public training entry point.
"""

from __future__ import annotations

from typing import Mapping


def training_stub_metadata() -> Mapping[str, str]:
    """Describe the current training implementation status."""

    return {
        "status": "stub",
        "note": "ResNet optimizer helpers will live here when implemented.",
    }
