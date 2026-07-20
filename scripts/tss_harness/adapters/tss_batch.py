"""Adapter #1: the current TSS deep solver via the persistent batch API.

Runs in the harness-dev venv (wheel with Lane A's manifest/stats API).
Config vocabulary (all echoed through the Rust-side effective-config
resolver, never re-derived here):

    node_cap: int          horizon: int (0 = unbounded, else >= 16)
    ladder: bool           zone: bool
    wide: bool             goal: "win" | "loss" | "both" | "two_pass"
    shared_fragments: bool (drives the TSS_SHARED_FRAGMENTS env gate)

"two_pass" = win pass + full-budget loss pass on the unknowns (adapter-side
protocol; the engine sees two plain goal calls, so verification is
untouched). Cost per re-solved position = sum of both passes.

The env-gated flags are SET by this adapter around every engine call —
the incident rule (SOLVER_NOTES §5) is that arms must not depend on ambient
environment; the adapter owns the env, the manifest gate then verifies the
engine actually saw it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from ..contract import SolveRecord, Position

_V1_SOAK = Path(__file__).resolve().parents[2] / "_v1_soak"


def _corpus_lib():
    if str(_V1_SOAK) not in sys.path:
        sys.path.insert(0, str(_V1_SOAK))
    import arch_env  # noqa: F401  must precede hexfield_eq import
    import corpus_lib
    return corpus_lib


class TssBatchAdapter:
    name = "tss_batch"
    architecture = "tss-dfpn-v1"   # cost counters comparable within this key

    DEFAULTS: dict[str, Any] = {
        "node_cap": 500,
        "horizon": 0,
        "ladder": False,
        "zone": False,
        "wide": True,
        # "both" = production parity (mode 3). goal=win FILTERS loss facts at
        # the root (solve_goal_filters_root_facts) — a win-goal sweep reports
        # loss=0 by construction, which hid real loss coverage until
        # 2026-07-20. Under the wide profile Both gives the win attempt the
        # full budget (verdict-identical for wins) and surfaces the cheap
        # forced losses the primal search proves along the way.
        "goal": "both",
        # Engine-side unused-budget dual pass (R-TWOPASS-IMPL, owner-approved
        # 2026-07-20): an undecided wide Both WIN attempt's leftover nodes go
        # to the opponent-WIN attempt. Default off = pre-change engine.
        "dual_pass": False,
        "shared_fragments": False,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = dict(self.DEFAULTS)
        cfg.update(config or {})
        unknown = set(cfg) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"unknown config keys for {self.name}: {unknown}")
        if cfg["goal"] not in ("win", "loss", "both", "two_pass"):
            raise ValueError(f"goal must be win|loss|both|two_pass: {cfg['goal']!r}")
        self.config = cfg

    # -- env ownership ---------------------------------------------------- #
    def _apply_env(self) -> None:
        if self.config["shared_fragments"]:
            os.environ["TSS_SHARED_FRAGMENTS"] = "1"
        else:
            os.environ.pop("TSS_SHARED_FRAGMENTS", None)

    # -- contract --------------------------------------------------------- #
    def manifest(self) -> dict[str, Any]:
        self._apply_env()
        from hexfield_eq import _rust
        m = _rust.hexfield_eq_solver_manifest(
            int(self.config["node_cap"]),
            int(self.config["horizon"]),
            bool(self.config["ladder"]),
            bool(self.config["zone"]),
            bool(self.config["wide"]),
            bool(self.config["dual_pass"]),
        )
        m["goal"] = self.config["goal"]
        m["adapter"] = self.name
        m["architecture"] = self.architecture
        return m

    def solve_sequence(self, positions: list[Position]) -> list[SolveRecord]:
        self._apply_env()
        corpus_lib = _corpus_lib()
        from hexfield_eq import _rust

        states = [corpus_lib.build_state(list(p.moves)) for p in positions]

        def batch(goal: str, idx: list[int]) -> list[dict]:
            return _rust.hexfield_eq_deep_solve_batch(
                [states[i] for i in idx],
                int(self.config["node_cap"]),
                goal,
                int(self.config["horizon"]),
                bool(self.config["ladder"]),
                bool(self.config["zone"]),
                bool(self.config["wide"]),
                bool(self.config["dual_pass"]),
            )

        everyone = list(range(len(states)))
        if self.config["goal"] == "two_pass":
            # Win pass then loss pass on the unknowns, each with the FULL
            # budget — the wide profile's Both gives the dedicated loss
            # attempt zero budget (SOLVER_NOTES §6 P4), so a both arm only
            # surfaces primal-incidental losses; two_pass buys the rest for
            # a second budget spent ONLY where the win pass failed. Costs
            # are summed per position (nodes stay the honest currency).
            raw = batch("win", everyone)
            retry = [i for i, r in zip(everyone, raw) if r["status"] != "win"]
            for i, r in zip(retry, batch("loss", retry)):
                first = raw[i]
                merged = dict(r)
                merged["deep_nodes"] = (
                    int(first["deep_nodes"]) + int(r["deep_nodes"]))
                merged["wall_nanos"] = (
                    int(first["wall_nanos"]) + int(r["wall_nanos"]))
                merged["deep_verify_failed"] = (
                    int(first["deep_verify_failed"])
                    + int(r["deep_verify_failed"]))
                merged["win_pass_status"] = first["status"]
                raw[i] = merged
        else:
            raw = batch(str(self.config["goal"]), everyone)
        out: list[SolveRecord] = []
        for p, r in zip(positions, raw):
            status = r["status"]
            vf = int(r["deep_verify_failed"])
            counters = {
                k: v for k, v in r.items()
                if k not in ("status", "wall_nanos", "deep_verify_failed")
            }
            out.append(SolveRecord(
                pos_id=p.pos_id,
                status=status,
                # The engine's ONLY path to a decided verdict runs through the
                # independent verifier (tree.rs tss_solve_verified); a decided
                # status with a clean fatal counter IS verified.
                verified=status in ("win", "loss") and vf == 0,
                verify_failed=vf,
                wall_nanos=int(r["wall_nanos"]),
                cost=int(r["deep_nodes"]),
                counters=counters,
            ))
        return out


def declared_features(config: dict[str, Any]) -> tuple[str, ...]:
    """Feature names this config claims — each must have a canary
    (gates.gate_features_have_canaries). Structurally-dead machinery (zone,
    ladder, census gate: no known-firing fixture exists) is intentionally
    NOT claimable; if a config enables it anyway, the missing canary fails
    the run — which is exactly the honest outcome."""
    feats: list[str] = []
    if config.get("shared_fragments"):
        feats.append("warmth")
    if int(config.get("horizon", 0)) == 0:
        feats.append("unbounded_horizon")
    if config.get("dual_pass"):
        # Engine-side dual pass: the loss canary must observe real verified
        # LOSS verdicts from the arm's own goal path (both + dual_pass).
        feats.append("loss_detection")
    if config.get("goal", "win") in ("loss", "both", "two_pass"):
        # goal=both under the wide profile currently gives the loss attempt
        # zero budget (SOLVER_NOTES §5) — such an arm will FAIL this canary,
        # which is the honest outcome until the budget split is fixed.
        feats.append("loss_detection")
    if config.get("wide", True):
        feats.append("wide")
    if config.get("zone"):
        feats.append("zone")            # no canary exists -> unclaimable
    if config.get("ladder"):
        feats.append("ladder")          # no canary exists -> unclaimable
    return tuple(dict.fromkeys(feats))  # dedup, order-preserving
