"""Container package for standalone Hexo model families.

Model implementations intentionally live in family-specific subpackages such as
`hexo_models.dense_cnn` instead of being re-exported through compatibility
aliases at the package root.
"""

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_MODEL_PACKAGE_ROOTS = (
    _PACKAGE_DIR.parents[1] / "dense_cnn" / "python" / "hexo_models",
    _PACKAGE_DIR.parents[1] / "hexformer_ar" / "python" / "hexo_models",
    _PACKAGE_DIR.parent / "dense_cnn" / "python" / "hexo_models",
    _PACKAGE_DIR.parent / "hexformer_ar" / "python" / "hexo_models",
)

for _root in _MODEL_PACKAGE_ROOTS:
    if _root.is_dir():
        _root_str = str(_root)
        if _root_str not in __path__:
            __path__.append(_root_str)

__all__: list[str] = []
