"""Decision and game budget contracts.

The runner owns budget policy: how much time or work a participant gets, what
happens on timeout, and how budget use is reported. Engine and model packages
receive budget context but do not define runner policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecisionBudget:
    """Per-decision budget passed to a player."""

    max_time_ms: int | None = None
    max_nodes: int | None = None
    max_simulations: int | None = None


@dataclass(frozen=True, slots=True)
class GameBudget:
    """Whole-game budget tracked by the runner."""

    max_turns: int | None = None
    max_time_ms: int | None = None
    decision: DecisionBudget = DecisionBudget()


@dataclass(frozen=True, slots=True)
class BudgetReport:
    """Budget usage attached to runner events and results."""

    elapsed_ms: float
    nodes: int | None = None
    simulations: int | None = None
    timed_out: bool = False
