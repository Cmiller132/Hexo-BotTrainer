"""Container package for standalone Hexo model families.

This is a namespace shim, not a library: it grafts the lineage subpackages
(`packages/hexo_models/dense_cnn/python/hexo_models` and
`packages/hexo_models/hexgt/python/hexo_models`) onto `__path__` at import time
so that `import hexo_models.dense_cnn` / `import hexo_models.hexgt` resolve
even though those trees live in sibling directories. It deliberately
re-exports nothing; model implementations stay in their family subpackages.

The maturin-built native extension `hexo_models._rust` (crate root:
packages/hexo_models/rust/src/lib.rs) sits beside this file as a platform
`.so`/`.pyd`; it hosts the dense_cnn / hexgt / hexgnn Rust accelerators as
submodules. On interpreters where the artifact is absent (e.g. Windows-side
Python when only the WSL Linux build exists) `import hexo_models` still
succeeds -- the missing `_rust` is surfaced lazily by each lineage's
`rust_bridge.py` import guard.

Importers that rely on this stitching:
  - hexo_models.dense_cnn / hexo_models.hexgt themselves (tests, configs,
    hexo_train entry points `dense_cnn` and `hexgt`),
  - packages/dense_cnn_restnet/python/dense_cnn_restnet/rust_bridge.py and
    packages/hexgnn/python/hexgnn/rust_bridge.py (`from hexo_models import
    _rust`, the ACTIVE lineage's only native path),
  - packages/hexo_frontend/python/hexo_frontend/debug_infer.py (dashboard
    debug worker loading dense_cnn/hexgt checkpoints).
"""

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
# Two on-disk layouts are probed for each lineage:
#   - parents[1]/...: the source-tree layout, where this file lives at
#     packages/hexo_models/python/hexo_models/__init__.py and the lineage trees
#     are at packages/hexo_models/<lineage>/python/hexo_models;
#   - parent/...: the installed-wheel layout, where maturin (pyproject.toml
#     include rules) places dense_cnn/python/hexo_models and
#     hexgt/python/hexo_models directly under the installed package dir.
# Non-existent candidates are skipped, so both layouts share this one tuple.
# CAUTION: the wheel branch depends on maturin preserving that unusual nested
# directory structure inside the wheel -- a packaging change would break
# installed-wheel imports while source-tree imports keep working.
_MODEL_PACKAGE_ROOTS = (
    _PACKAGE_DIR.parents[1] / "dense_cnn" / "python" / "hexo_models",
    _PACKAGE_DIR.parents[1] / "hexgt" / "python" / "hexo_models",
    _PACKAGE_DIR.parent / "dense_cnn" / "python" / "hexo_models",
    _PACKAGE_DIR.parent / "hexgt" / "python" / "hexo_models",
)

for _root in _MODEL_PACKAGE_ROOTS:
    if _root.is_dir():
        _root_str = str(_root)
        if _root_str not in __path__:
            __path__.append(_root_str)

__all__: list[str] = []
