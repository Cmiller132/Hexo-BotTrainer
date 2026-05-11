"""Python control plane for the Hexo RL prototype."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import HexoConfig, load_config

__all__ = ["HexoConfig", "load_config"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .config import HexoConfig, load_config

        return {"HexoConfig": HexoConfig, "load_config": load_config}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
