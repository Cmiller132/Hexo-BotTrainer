"""Shared helpers for the hexfield test suite (not collected by pytest).

Adds packages/hexfield/python to sys.path — hexfield is deliberately never
installed into a shared venv (spec §5.1 build blast-radius discipline), so
tests import it via this shim.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_PACKAGES = Path(__file__).resolve().parent.parent / "packages"
sys.path.insert(0, str(_PACKAGES / "hexfield" / "python"))
# Test-only oracle package (not installed in the build venv): restnet's
# losses/samples/compact_io are imported by the M2 parity tests.
sys.path.insert(0, str(_PACKAGES / "dense_cnn_restnet" / "python"))

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield.geometry import unpack_action_id


def random_playout(seed: int, plies: int) -> "api.HexoState":
    """Play `plies` uniform-random legal placements (stops early on terminal)."""

    state = api.new_game()
    rng = random.Random(seed)
    for _ in range(plies):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        q, r = unpack_action_id(rng.choice(ids))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            break
    return state


def sample_decision_states(
    seeds: range, plies_choices: tuple[int, ...]
) -> list["api.HexoState"]:
    """Non-terminal states from seeded random playouts (decision rows only)."""

    states = []
    for seed in seeds:
        for plies in plies_choices:
            state = random_playout(seed * 1000 + plies, plies)
            if api.terminal(state) is None:
                states.append(state)
    return states
