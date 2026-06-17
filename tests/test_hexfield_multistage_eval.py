"""Unit tests for the hexfield multi-stage eval ORCHESTRATOR (pure CPU, no GPU).

These exercise ``packages/hexfield/python/hexfield/multistage_eval.py`` — the
layer that wires the game-running arena to the statistics core and emits a
verdict LABEL. They cover the four things the corrected, adversary-reviewed
design must get right at the orchestration level (the statistics themselves are
covered separately in ``test_hexfield_eval_stats.py``):

  1. OPPONENT-LADDER SELECTION (``select_opponents``): given a fake checkpoints
     dir, the right roster is chosen — SealBot zero-point, PERMANENT anchors
     (BC prefit + ep5, never sliding), the SLIDING bracket (nearest log-grid
     rungs strictly BELOW the candidate, NOT the immediately-prior checkpoint),
     and the single prior-champion PRIMARY target — and the bracket stays
     bounded as epochs grow.

  2. ROLLING BT POOL PERSISTENCE (``_save_pool`` / ``_load_pool`` /
     ``_bt_edges_from_pool`` and the end-to-end pool write): a round-trip is
     stable, a corrupt/missing/foreign pool degrades to a fresh empty pool
     (never wedges an eval-only run), and edges COMPOUND across epochs so the
     difference SE shrinks.

  3. VERDICT LOGIC from SYNTHETIC edge results: PROMOTE / REGRESS /
     INCONCLUSIVE thresholds, and — load-bearing — that ONLY the primary
     hypothesis (candidate vs prior champion) drives the verdict; blowing out a
     non-champion (descriptive) edge does NOT change the label.

  4. PURE EVAL: gating/promotion default OFF and the verdict is wired to nothing
     that mutates a run — flipping the verdict touches no run state, and the
     ``_assert_no_run_mutation`` tripwire fires the instant a gating/promotion
     knob is flipped on.

The arena (which needs the GPU + the SealBot checkout) is MOCKED throughout via
the ``play_checkpoint_match`` / ``play_sealbot_match`` injection seams, so this
collects and runs on a CPU-only interpreter without touching the live training
run. ``multistage_eval`` imports torch only lazily (inside the game path), so
importing it here is torch-free; we add hexfield's source to ``sys.path``
directly because it is deliberately never installed in a shared venv.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "hexfield" / "python"))

from hexfield import multistage_eval as mse  # noqa: E402
from hexfield.config import (  # noqa: E402
    MultiStageEvalSection,
    parse_hexfield_config,
)


# --------------------------------------------------------------------------- #
# Fixtures / builders.
# --------------------------------------------------------------------------- #
def _make_run(tmp_path: Path, epochs: tuple[int, ...] = (5, 10, 20, 40), *, bc: bool = True) -> Path:
    """A fake run tree: ``<tmp>/runs/r/checkpoints/epoch_*.pt`` (+ BC sibling).

    Mirrors the real layout ``select_opponents`` resolves against: in-run
    checkpoints under ``<run>/checkpoints`` and the BC prefit in the sibling
    ``runs/hexfield_bc_1`` referenced by the default permanent-anchor path.
    Files are empty stubs — selection is pure PATH resolution and never loads a
    checkpoint, so the bytes do not matter.
    """

    run = tmp_path / "runs" / "r"
    ckpts = run / "checkpoints"
    ckpts.mkdir(parents=True)
    for epoch in epochs:
        (ckpts / f"epoch_{epoch:06d}.pt").write_text("stub", encoding="utf-8")
    if bc:
        bc_dir = tmp_path / "runs" / "hexfield_bc_1"
        bc_dir.mkdir(parents=True)
        (bc_dir / "checkpoint_epoch2.pt").write_text("stub", encoding="utf-8")
    return run


def _paired_match(label_a: str, label_b: str, pair_scores: list[int]) -> dict:
    """A fake ``play_checkpoint_match`` result with a real pentanomial block.

    ``pair_scores`` is one net-A score per COMPLETE 2-game pair, in {0, 1, 2}
    (net-A wins among the pair's two decided games). The shape exactly matches
    ``eval_arena._build_match_result`` + ``_pentanomial_block`` (``score`` with
    ``a_wins`` net-A-centric, ``pentanomial.pairs`` rows + the ``histogram_a_wins``
    fallback), so the orchestrator's ``_checkpoint_edge_counts`` /
    ``_pentanomial_to_paired_result`` consume it unchanged.
    """

    pairs = []
    a_wins = b_wins = 0
    for i, score in enumerate(pair_scores):
        pairs.append(
            {
                "pair_index": i,
                "seed": 1000 + i,
                "game_indices": [2 * i, 2 * i + 1],
                "n_games": 2,
                "n_decided": 2,
                "a_wins": score,
                "b_wins": 2 - score,
                "pentanomial_a_score": score,
            }
        )
        a_wins += score
        b_wins += 2 - score
    hist = {"0": 0, "1": 0, "2": 0}
    for score in pair_scores:
        hist[str(score)] += 1
    decided = a_wins + b_wins
    return {
        "meta": {"label_a": label_a, "label_b": label_b, "games_requested": 2 * len(pair_scores)},
        "score": {
            "completed": 2 * len(pair_scores),
            "truncated": 0,
            "aborted_budget": 0,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "decided": decided,
            "a_winrate_decided": (a_wins / decided) if decided else None,
        },
        "pentanomial": {
            "n_pairs": len(pair_scores),
            "n_full_pairs": len(pair_scores),
            "pairs": pairs,
            "histogram_a_wins": hist,
        },
    }


def _sealbot_match(label: str, n: int, winrate: float) -> dict:
    """A fake ``play_sealbot_match`` result (unpaired, binomial), net-A-centric."""

    wins = int(round(winrate * n))
    return {
        "meta": {"games_requested": n},
        "score": {"completed": n, "a_wins": wins, "b_wins": n - wins, "decided": n},
    }


class _FakeArena:
    """Records calls and returns synthetic matches at a configured strength.

    ``ckpt_winrate(label_b) -> per-pair-score pattern`` lets a test set the
    candidate's strength PER opponent (so the primary edge can differ from the
    descriptive ones); ``sealbot_winrate`` sets the binomial SealBot edge.
    """

    def __init__(self, *, ckpt_scorer, sealbot_winrate: float = 0.55) -> None:
        self._ckpt_scorer = ckpt_scorer
        self._sealbot_winrate = sealbot_winrate
        self.ckpt_calls: list[tuple[str, int]] = []
        self.sealbot_calls: list[int] = []
        # Capture the visits the orchestrator threads (full-sims wiring check).
        self.ckpt_visits: list[int | None] = []
        self.sealbot_visits: list[int | None] = []

    def play_checkpoint_match(self, a, b, n, **kw) -> dict:
        label_b = kw["label_b"]
        self.ckpt_calls.append((label_b, n))
        self.ckpt_visits.append(kw.get("visits"))
        n_pairs = max(1, n // 2)
        match = _paired_match(kw["label_a"], label_b, self._ckpt_scorer(label_b, n_pairs))
        match["meta"]["visits"] = kw.get("visits")  # echo so provenance is testable
        return match

    def play_sealbot_match(self, ckpt, n, **kw) -> dict:
        self.sealbot_calls.append(n)
        self.sealbot_visits.append(kw.get("visits"))
        match = _sealbot_match(kw.get("label", "hexfield"), n, self._sealbot_winrate)
        match["meta"]["visits"] = kw.get("visits")
        return match


def _scores_for_winrate(target: float, n_pairs: int) -> list[int]:
    """A per-pair {0,1,2} pattern whose mean/2 approximates ``target`` win rate.

    Uses a mix of decisive (2 / 0) and split (1) pairs so the pentanomial has
    non-degenerate variance (degenerate all-WW pairs make the pair-level SE
    zero, which is not representative). The exact rate is not load-bearing —
    tests assert verdict DIRECTION, not a specific Elo.
    """

    if target >= 0.7:
        return [2 if i % 2 == 0 else 1 for i in range(n_pairs)]      # ~0.75
    if target <= 0.3:
        return [0 if i % 2 == 0 else 1 for i in range(n_pairs)]      # ~0.25
    return [2 if i % 3 == 0 else (0 if i % 3 == 1 else 1) for i in range(n_pairs)]  # ~0.5


def _no_sprt_config(**overrides) -> object:
    """Production-default config with Stage B SPRT disabled (it needs a champion
    match; the deep eval is what the verdict tests exercise)."""

    mse_section = {"sprt": {"enabled": False}, **overrides}
    return parse_hexfield_config({"multi_stage_eval": mse_section})


def _run(run_dir: Path, candidate_epoch: int, arena: _FakeArena, *, config=None, **kw) -> dict:
    cfg = config if config is not None else _no_sprt_config()
    return mse.run_multistage_eval(
        run_dir,
        run_dir / "checkpoints" / f"epoch_{candidate_epoch:06d}.pt",
        cfg,
        candidate_epoch=candidate_epoch,
        write_diagnostics=False,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
        **kw,
    )


# =========================================================================== #
# 1. Opponent-ladder selection.
# =========================================================================== #
def test_roster_roles_anchors_bracket_champion(tmp_path: Path) -> None:
    """At a mid-ladder epoch the roster has every role, with the corrected
    semantics: PERMANENT anchors (BC + ep5), a SLIDING bracket of the nearest
    log-grid rungs strictly below, and the single highest-prior-epoch champion."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = MultiStageEvalSection()
    roster = mse.select_opponents(run, run / "checkpoints" / "epoch_000040.pt", cfg, candidate_epoch=40)

    assert roster.candidate_label == "cand_ep40"
    assert roster.candidate_epoch == 40
    assert roster.sealbot is not None and roster.sealbot.role == "sealbot"
    assert roster.sealbot.ckpt is None  # SealBot is an external engine, not a ckpt

    by_label = {o.label: o for o in roster.opponents}
    # Permanent anchors (never slide): BC prefit + ep5.
    assert by_label["bc_prefit"].role == "anchor"
    assert by_label["ep5"].role == "anchor"
    # Sliding bracket: nearest log-grid rungs strictly below 40 -> {10, 20}. 20
    # is also the champion (highest prior epoch) so it is de-duped to "champion".
    assert by_label["ep10"].role == "bracket"
    # Champion: highest existing epoch strictly below the candidate (ep20, since
    # no ep30 exists on disk — NOT "the immediately-prior checkpoint").
    assert roster.champion is not None
    assert roster.champion.label == "ep20"
    assert roster.champion.epoch == 20
    assert by_label["ep20"].role == "champion"


def test_roster_dedupes_anchor_that_is_also_champion(tmp_path: Path) -> None:
    """When the prior champion IS a permanent anchor (ep5 at candidate ep10), it
    appears ONCE in the roster, flagged champion (so it is not double-counted as
    both an anchor edge and a champion edge)."""

    run = _make_run(tmp_path, epochs=(5, 10))
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000010.pt", MultiStageEvalSection(), candidate_epoch=10
    )
    ep5_entries = [o for o in roster.opponents if o.label == "ep5"]
    assert len(ep5_entries) == 1
    assert ep5_entries[0].role == "champion"
    assert roster.champion is not None and roster.champion.label == "ep5"
    # No duplicate labels anywhere in the roster.
    labels = [o.label for o in roster.opponents]
    assert len(labels) == len(set(labels))


def test_roster_first_eligible_epoch_has_no_champion(tmp_path: Path) -> None:
    """The lowest epoch (no prior checkpoint below it) yields champion=None -> no
    primary hypothesis exists (verdict will be INCONCLUSIVE downstream)."""

    run = _make_run(tmp_path, epochs=(5, 10, 20))
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000005.pt", MultiStageEvalSection(), candidate_epoch=5
    )
    assert roster.champion is None
    # Bracket is empty too (no grid rung strictly below 5); only the BC anchor
    # (and ep5-as-anchor, which resolves to the candidate file itself) remain.
    assert all(o.role != "bracket" for o in roster.opponents)


def test_bracket_is_bounded_and_slides_as_epochs_grow(tmp_path: Path) -> None:
    """The SLIDING bracket window is the nearest ``bracket_size`` log-grid rungs
    strictly below the candidate, it tracks the candidate epoch upward (the
    corrected design's nearest-N-below, not a growing set), and its top rung is
    de-duped into the champion role.

    We assert on the bracket+champion UNION = the nearest <=bracket_size grid
    rungs below, because the highest such rung is always the prior champion and
    is re-tagged "champion" (so the literal "bracket"-role set is the window
    minus its top rung)."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40, 80, 160))
    cfg = MultiStageEvalSection()  # bracket_size=2, log_grid=(5,10,20,40,80,160)
    grid = sorted(cfg.opponents.log_grid)

    def roster_for(cand_epoch: int):
        return mse.select_opponents(
            run, run / "checkpoints" / f"epoch_{cand_epoch:06d}.pt", cfg, candidate_epoch=cand_epoch
        )

    for cand in (40, 80, 160):
        roster = roster_for(cand)
        bracket = {o.label for o in roster.opponents if o.role == "bracket"}
        # Bracket window = the nearest <=2 grid rungs strictly below the candidate.
        window = [f"ep{g}" for g in grid if g < cand][-cfg.opponents.bracket_size:]
        # The literal "bracket" role is the window minus the champion (top rung).
        assert roster.champion is not None and roster.champion.label == window[-1]
        assert bracket == set(window) - {roster.champion.label}
        # Bounded: the bracket window never exceeds bracket_size.
        assert len({*bracket, roster.champion.label}) <= cfg.opponents.bracket_size

    # The window slides UP as the candidate climbs (nearest-below, not cumulative).
    assert roster_for(40).champion.label == "ep20"
    assert roster_for(80).champion.label == "ep40"
    assert roster_for(160).champion.label == "ep80"


def test_bracket_size_zero_yields_no_bracket(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = MultiStageEvalSection(
        opponents=dataclasses.replace(MultiStageEvalSection().opponents, bracket_size=0)
    )
    roster = mse.select_opponents(run, run / "checkpoints" / "epoch_000040.pt", cfg, candidate_epoch=40)
    assert all(o.role != "bracket" for o in roster.opponents)
    # Champion still resolves (it is independent of the bracket).
    assert roster.champion is not None and roster.champion.label == "ep20"


def test_missing_anchor_checkpoints_are_skipped(tmp_path: Path) -> None:
    """A permanent anchor whose file is absent (e.g. an early epoch predating
    ep5, or a missing BC prefit) is silently skipped, not an error.

    The default ``bc_prefit`` anchor path now resolves against the REPO tree too
    (fix #3: the BC prefit lives in the repo, not the run-data tree), so a test
    of the SKIP behavior must use anchor paths that resolve to NOTHING under any
    root — both a bare in-run filename absent from the checkpoints dir AND a
    repo-relative path under a directory that exists in neither the tmp run tree
    nor the repo tree. This keeps the test hermetic regardless of repo contents.
    """

    run = _make_run(tmp_path, epochs=(10, 20), bc=False)  # no BC file, no ep5 file
    cfg = MultiStageEvalSection(
        opponents=dataclasses.replace(
            MultiStageEvalSection().opponents,
            permanent_anchors=(
                # Repo-relative under a uniquely-named dir that exists nowhere.
                ("bc_prefit", "runs/__hexfield_no_such_bc__/checkpoint_epoch2.pt"),
                # Bare filename absent from the tmp checkpoints dir.
                ("ep5", "epoch_000005.pt"),
            ),
        )
    )
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000020.pt", cfg, candidate_epoch=20
    )
    labels = {o.label for o in roster.opponents}
    assert "bc_prefit" not in labels  # file resolves nowhere -> skipped
    assert "ep5" not in labels        # ep5 file absent -> skipped
    # The ep10 champion still resolves from what is on disk.
    assert roster.champion is not None and roster.champion.label == "ep10"


def test_sealbot_disabled_drops_zero_point(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = MultiStageEvalSection(
        opponents=dataclasses.replace(MultiStageEvalSection().opponents, sealbot_enabled=False)
    )
    roster = mse.select_opponents(run, run / "checkpoints" / "epoch_000040.pt", cfg, candidate_epoch=40)
    assert roster.sealbot is None
    assert mse.SEALBOT_LABEL not in roster.all_labels()


def test_allocate_budget_even_split_and_sealbot_share() -> None:
    """Budget allocation: SealBot gets a fixed share, the rest is split EVENLY
    across checkpoint opponents and rounded to an EVEN per-pairing count (paired
    games come two-per-pair)."""

    alloc = mse.allocate_budget(128, n_checkpoint_opponents=4, has_sealbot=True)
    assert alloc[mse.SEALBOT_LABEL] == 32  # 25% default share
    assert alloc["per_checkpoint"] % 2 == 0  # even pairings
    assert alloc["per_checkpoint"] == 24     # (128-32)//4 = 24, already even

    # No SealBot -> the whole budget goes to checkpoints.
    alloc2 = mse.allocate_budget(120, n_checkpoint_opponents=3, has_sealbot=False)
    assert alloc2[mse.SEALBOT_LABEL] == 0
    assert alloc2["per_checkpoint"] == 40

    # No checkpoint opponents (first epoch) -> everything to SealBot if present.
    alloc3 = mse.allocate_budget(100, n_checkpoint_opponents=0, has_sealbot=True)
    assert alloc3[mse.SEALBOT_LABEL] == 100
    assert alloc3["per_checkpoint"] == 0

    # Zero budget -> zeros, no crash.
    assert mse.allocate_budget(0, n_checkpoint_opponents=4, has_sealbot=True) == {
        mse.SEALBOT_LABEL: 0,
        "per_checkpoint": 0,
    }


# =========================================================================== #
# 2. Rolling BT pool persistence.
# =========================================================================== #
def test_pool_roundtrip_is_stable(tmp_path: Path) -> None:
    """Write a pool, reload it, and confirm the edges + anchor survive verbatim
    and project to the same BT edges (the persisted pool is the rolling rating's
    memory; a lossy round-trip would silently drift ratings)."""

    pool = tmp_path / "diagnostics" / "eval_pool.json"
    doc = {
        "format": mse.POOL_FORMAT,
        "version": mse.POOL_VERSION,
        "anchor": mse.SEALBOT_LABEL,
        "edges": [
            {"epoch": 10, "a": "cand_ep10", "b": "ep5", "wins_a": 41.0, "wins_b": 23.0,
             "weight": 1.0, "kind": "checkpoint", "raw": {}},
            {"epoch": 10, "a": "cand_ep10", "b": "sealbot", "wins_a": 30.0, "wins_b": 10.0,
             "weight": 0.5, "kind": "sealbot", "raw": {}},
        ],
    }
    mse._save_pool(pool, doc)
    back = mse._load_pool(pool)
    assert back["anchor"] == mse.SEALBOT_LABEL
    assert back["edges"] == doc["edges"]

    edges = {(e.a, e.b, e.weight): (e.wins_a, e.wins_b) for e in mse._bt_edges_from_pool(back)}
    assert edges[("cand_ep10", "ep5", 1.0)] == (41.0, 23.0)
    # SealBot edge keeps its down-weight (0.5), separate from full-weight edges.
    assert edges[("cand_ep10", "sealbot", 0.5)] == (30.0, 10.0)


def test_pool_load_degrades_gracefully(tmp_path: Path) -> None:
    """Missing / corrupt / foreign-format pools yield a FRESH empty pool rather
    than raising — a broken pool must never wedge an eval-only run (losing old
    edges only loosens CIs, it never touches training)."""

    missing = tmp_path / "nope.json"
    fresh = mse._load_pool(missing)
    assert fresh["format"] == mse.POOL_FORMAT and fresh["edges"] == []

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{ not valid json", encoding="utf-8")
    assert mse._load_pool(corrupt)["edges"] == []

    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps({"format": "something.else", "edges": [1, 2, 3]}), encoding="utf-8")
    fresh3 = mse._load_pool(foreign)
    assert fresh3["format"] == mse.POOL_FORMAT and fresh3["edges"] == []


def test_bt_edges_aggregate_across_epochs_and_keep_weights_separate() -> None:
    """The append-only edge log projects to one BT edge per (unordered pair,
    weight): repeated epochs of a pairing POOL their counts (this is how the
    rolling pool compounds), reversed directions canonicalise together, and the
    down-weighted SealBot edge never merges with a full-weight edge."""

    doc = {
        "edges": [
            {"a": "cand", "b": "ep5", "wins_a": 40, "wins_b": 20, "weight": 1.0},
            {"a": "cand", "b": "ep5", "wins_a": 30, "wins_b": 10, "weight": 1.0},   # same pair -> pooled
            {"a": "ep5", "b": "cand", "wins_a": 5, "wins_b": 5, "weight": 1.0},      # reversed -> canonicalised
            {"a": "cand", "b": "sealbot", "wins_a": 30, "wins_b": 10, "weight": 0.5},  # down-weighted, separate
        ]
    }
    edges = {(e.a, e.b, e.weight): (e.wins_a, e.wins_b) for e in mse._bt_edges_from_pool(doc)}
    # cand-ep5 pooled: 40+30 + (reversed b-wins) 5 = 75 cand wins; 20+10 + 5 = 35 ep5 wins.
    assert edges[("cand", "ep5", 1.0)] == (75.0, 35.0)
    assert edges[("cand", "sealbot", 0.5)] == (30.0, 10.0)
    assert len(edges) == 2  # exactly two distinct (pair, weight) edges


def test_pool_persists_and_compounds_across_runs(tmp_path: Path) -> None:
    """End-to-end: two eval runs of the same epoch append to the persisted pool
    (append-only audit trail) and the PRIMARY difference SE SHRINKS as the games
    compound — the only way the tight multi-epoch resolution is ever reached
    (fix #4). A single run does NOT achieve it."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)

    rep1 = mse.run_multistage_eval(
        run, run / "checkpoints" / "epoch_000040.pt", _no_sprt_config(),
        candidate_epoch=40, write_diagnostics=True,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
    )
    pool_path = run / "diagnostics" / "eval_pool.json"
    edges_after_1 = len(json.loads(pool_path.read_text(encoding="utf-8"))["edges"])
    assert edges_after_1 > 0

    rep2 = mse.run_multistage_eval(
        run, run / "checkpoints" / "epoch_000040.pt", _no_sprt_config(),
        candidate_epoch=40, write_diagnostics=True,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
    )
    doc2 = json.loads(pool_path.read_text(encoding="utf-8"))
    # Append-only: the second run doubles the edge rows.
    assert len(doc2["edges"]) == 2 * edges_after_1
    assert doc2["format"] == mse.POOL_FORMAT and doc2["anchor"] == mse.SEALBOT_LABEL

    # Compounding: more pooled games -> a strictly tighter primary difference SE.
    se1 = rep1["verdict"]["primary"]["se_elo"]
    se2 = rep2["verdict"]["primary"]["se_elo"]
    assert se2 < se1
    # Ratings stay sane on reload (anchor pinned, fit converged both times).
    assert rep1["ratings"]["fit"]["converged"] and rep2["ratings"]["fit"]["converged"]
    assert rep2["ratings"]["fit"]["anchor"] == mse.SEALBOT_LABEL


# =========================================================================== #
# 3. Verdict logic from synthetic edge results.
# =========================================================================== #
def test_verdict_promote_when_candidate_dominates_champion(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 40, arena)
    assert rep["verdict"]["label"] == "PROMOTE"
    lo, hi = rep["verdict"]["primary"]["elo_diff_ci95"]
    assert lo > 0.0  # whole difference CI above the promote threshold (0)
    assert rep["verdict"]["primary"]["champion"] == "ep20"
    assert rep["verdict"]["primary"]["candidate"] == "cand_ep40"


def test_verdict_regress_when_candidate_loses_to_champion(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.25, n), sealbot_winrate=0.3)
    rep = _run(run, 40, arena)
    assert rep["verdict"]["label"] == "REGRESS"
    lo, hi = rep["verdict"]["primary"]["elo_diff_ci95"]
    assert hi < 0.0  # whole difference CI below the regress threshold (0)


def test_verdict_inconclusive_when_even(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.5, n), sealbot_winrate=0.5)
    rep = _run(run, 40, arena)
    assert rep["verdict"]["label"] == "INCONCLUSIVE"
    lo, hi = rep["verdict"]["primary"]["elo_diff_ci95"]
    assert lo < 0.0 < hi  # CI straddles 0


def test_only_primary_hypothesis_drives_the_verdict(tmp_path: Path) -> None:
    """LOAD-BEARING (fix #3): the verdict rests on ONE pre-registered primary
    edge (candidate vs prior champion). Crushing every NON-champion (descriptive)
    edge while staying even with the champion must NOT move the label off
    INCONCLUSIVE, and the descriptive edges must carry NO significance verdict.
    """

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))

    def scorer(label_b: str, n_pairs: int) -> list[int]:
        # Even vs the champion (ep20); crush every other checkpoint opponent.
        target = 0.5 if label_b == "ep20" else 0.75
        return _scores_for_winrate(target, n_pairs)

    arena = _FakeArena(ckpt_scorer=scorer, sealbot_winrate=0.95)  # also crush SealBot
    rep = _run(run, 40, arena)

    # Champion edge is even -> INCONCLUSIVE, despite the blowouts elsewhere.
    assert rep["verdict"]["label"] == "INCONCLUSIVE"

    primary = [e for e in rep["edges"] if e.get("primary")]
    descriptive = [e for e in rep["edges"] if not e.get("primary")]
    # Exactly one primary edge, and it is the champion.
    assert [e["opponent"] for e in primary] == ["ep20"]
    # Every other edge (SealBot + anchors + bracket) is DESCRIPTIVE...
    assert {e["opponent"] for e in descriptive} == {"sealbot", "bc_prefit", "ep5", "ep10"}
    # ...and carries no significance verdict/label of its own.
    for e in descriptive:
        assert "label" not in e and "verdict" not in e
        assert e["primary"] is False


def test_no_champion_yields_inconclusive_with_no_primary(tmp_path: Path) -> None:
    """First eligible epoch (no prior champion) -> verdict INCONCLUSIVE and the
    primary block is absent: there is no hypothesis to test."""

    run = _make_run(tmp_path, epochs=(5, 10, 20))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 5, arena)
    assert rep["verdict"]["label"] == "INCONCLUSIVE"
    assert rep["verdict"].get("primary") is None
    # No edge is flagged primary.
    assert all(not e.get("primary") for e in rep["edges"])


def test_custom_thresholds_shift_the_verdict(tmp_path: Path) -> None:
    """Verdict thresholds are configurable: a demanding promote threshold turns a
    real-but-small edge INCONCLUSIVE, and a lenient regress threshold turns a
    mild deficit REGRESS. PURE EVAL — the thresholds only relabel; they gate
    nothing."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)

    # Baseline (thresholds 0/0) -> PROMOTE.
    assert _run(run, 40, arena)["verdict"]["label"] == "PROMOTE"

    # A promote threshold above the whole CI -> INCONCLUSIVE (same games).
    rep = _run(run, 40, arena, config=_no_sprt_config(promote_elo_threshold=10_000.0))
    assert rep["verdict"]["label"] == "INCONCLUSIVE"
    assert rep["verdict"]["primary"]["promote_threshold_elo"] == 10_000.0


def test_descriptive_edges_have_pairlevel_cis_not_pergame(tmp_path: Path) -> None:
    """Paired (checkpoint) edges report a pair-level CI block (fix #2): the
    descriptive edge carries ``elo_ci95_pairlevel`` and ``winrate_ci95`` and is
    flagged paired, so the report never advertises an anti-conservative per-game
    interval as the edge's CI."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 40, arena)
    ckpt_edges = [e for e in rep["edges"] if e["kind"] == "checkpoint"]
    assert ckpt_edges  # anchors + bracket + champion
    for e in ckpt_edges:
        assert e["paired"] is True
        assert "elo_ci95_pairlevel" in e
        lo, hi = e["winrate_ci95"]
        assert 0.0 <= lo <= hi <= 1.0
    # The SealBot edge is unpaired and reports its Wilson win-rate CI + down-weight.
    sb_edge = next(e for e in rep["edges"] if e["kind"] == "sealbot")
    assert sb_edge["paired"] is False
    assert sb_edge["down_weight"] == pytest.approx(0.5)  # over-dispersion (fix #5)
    assert rep["sealbot_winrate_ci95"] is not None


# =========================================================================== #
# 4. Pure-eval invariant: gating/promotion OFF, verdict mutates no run state.
# =========================================================================== #
def test_gating_and_promotion_default_off() -> None:
    cfg = MultiStageEvalSection()
    assert cfg.eval_gating_enabled is False
    assert cfg.eval_promotion_enabled is False
    # The tripwire passes for the default (PURE EVAL) config.
    mse._assert_no_run_mutation(cfg)


def test_assert_no_run_mutation_fires_when_a_knob_is_flipped() -> None:
    """The tripwire (a future-proofing guard) MUST fire the instant a
    gating/promotion knob is turned on, so a later edit that starts consuming
    them is caught in tests before it can touch a live run."""

    base = MultiStageEvalSection()
    with pytest.raises(AssertionError, match="eval_gating_enabled must be False"):
        mse._assert_no_run_mutation(dataclasses.replace(base, eval_gating_enabled=True))
    with pytest.raises(AssertionError, match="eval_promotion_enabled must be False"):
        mse._assert_no_run_mutation(dataclasses.replace(base, eval_promotion_enabled=True))


def test_report_advertises_pure_eval_and_off_knobs(tmp_path: Path) -> None:
    """Every report states it is pure eval with gating/promotion off — the label
    consumer can see at a glance that the verdict gates nothing."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 40, arena)
    assert rep["meta"]["pure_eval"] is True
    assert rep["meta"]["gating_enabled"] is False
    assert rep["meta"]["promotion_enabled"] is False
    assert "gates nothing" in rep["verdict"]["note"]


def test_write_diagnostics_false_writes_nothing(tmp_path: Path) -> None:
    """With ``write_diagnostics=False`` the run is fully PURE: no pool, no
    diagnostics JSON, no directory created anywhere under the run tree."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 40, arena)  # _run passes write_diagnostics=False
    assert rep["verdict"]["label"] == "PROMOTE"  # it still computes a verdict
    assert not (run / "diagnostics").exists()     # ...but writes NOTHING
    assert "diagnostics_path" not in rep["meta"]


def test_changing_the_verdict_touches_no_run_state(tmp_path: Path) -> None:
    """Flipping the verdict from PROMOTE to REGRESS (by swapping the synthetic
    edge results) must leave the run tree byte-identical apart from the eval-only
    pool/diagnostics — there is no checkpoint write, no flag, no run-state edit
    keyed on the verdict. We assert the ONLY new files are the eval pool + the
    per-epoch diagnostics JSON, for BOTH verdicts."""

    def run_with(strength: float, winrate: float, subdir: str) -> tuple[str, set[Path]]:
        run = _make_run(tmp_path / subdir, epochs=(5, 10, 20, 40))
        before = {p.relative_to(run) for p in run.rglob("*")}
        arena = _FakeArena(
            ckpt_scorer=lambda lb, n, _s=strength: _scores_for_winrate(_s, n), sealbot_winrate=winrate
        )
        rep = mse.run_multistage_eval(
            run, run / "checkpoints" / "epoch_000040.pt", _no_sprt_config(),
            candidate_epoch=40, write_diagnostics=True,
            play_checkpoint_match=arena.play_checkpoint_match,
            play_sealbot_match=arena.play_sealbot_match,
        )
        after = {p.relative_to(run) for p in run.rglob("*")}
        return rep["verdict"]["label"], after - before

    promote_label, promote_new = run_with(0.75, 0.7, "promote_run")
    regress_label, regress_new = run_with(0.25, 0.3, "regress_run")

    assert promote_label == "PROMOTE"
    assert regress_label == "REGRESS"
    # The set of newly created paths is the SAME for both verdicts: only the
    # eval-only pool + diagnostics tree. The verdict label changes nothing else.
    # (Compare as Path objects so the assertion is OS-separator agnostic.)
    eval_only = {
        Path("diagnostics"),
        Path("diagnostics") / "eval_pool.json",
        Path("diagnostics") / "hexfield.multistage_eval.epoch_000040.json",
    }
    assert promote_new == eval_only
    assert regress_new == eval_only
    # Crucially, NOTHING the trainer reads was created: no checkpoint, no flag.
    for new_set in (promote_new, regress_new):
        assert not any(p.suffix == ".pt" for p in new_set)
        assert not any(p.suffix == ".flag" for p in new_set)


def test_stage_flow_runs_all_four_stages(tmp_path: Path) -> None:
    """Sanity: the orchestrator returns the staged A-D structure with the deep
    eval and pool completing (Stage B is skipped here since SPRT is disabled)."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 40, arena)
    stages = {s["stage"]: s["status"] for s in rep["stages"]}
    assert stages["A_bridge"] == "ok"
    assert stages["B_sprt"] == "skipped"
    assert stages["C_deep"] == "completed"
    assert stages["D_pool"] == "completed"
    # The deep eval played the champion + every other opponent, plus SealBot.
    opponents_played = {label for label, _ in arena.ckpt_calls}
    assert {"ep20", "ep10", "ep5", "bc_prefit"} <= opponents_played
    assert arena.sealbot_calls  # SealBot zero-point edge was played


def test_sprt_stage_b_triage_runs_and_is_descriptive(tmp_path: Path) -> None:
    """With SPRT enabled, Stage B runs as a TRIAGE (its own verdict is reported
    but it never short-circuits the deep eval): the champion match is played and
    a triage label is attached, while the FINAL verdict still comes from Stage D.
    """

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    # SPRT enabled with a small cap (gross-regression triage; default elo0=0,
    # elo1=-50). M-1 coherent label set: accept_h1->regress_suspected,
    # accept_h0->ok, continue->escalate.
    cfg = parse_hexfield_config({"multi_stage_eval": {"sprt": {"enabled": True, "max_games": 16}}})
    rep = _run(run, 40, arena, config=cfg)

    stage_b = next(s for s in rep["stages"] if s["stage"] == "B_sprt")
    assert stage_b["status"] == "completed"
    assert stage_b["vs"] == "ep20"  # screened against the prior champion
    assert stage_b["triage"] in {"regress_suspected", "ok", "escalate"}
    # The deep eval + pool still run and produce the authoritative verdict.
    assert rep["verdict"]["label"] in {"PROMOTE", "REGRESS", "INCONCLUSIVE"}
    assert any(s["stage"] == "C_deep" and s["status"] == "completed" for s in rep["stages"])


# =========================================================================== #
# 5. FULL-sims wiring: Stage B + Stage C play at the production budget by default.
# =========================================================================== #
def test_full_sims_threaded_into_stage_b_and_c_by_default(tmp_path: Path) -> None:
    """By default the orchestrator threads the FULL production search budget
    (selfplay.search_visits=512) into BOTH the SPRT screen (Stage B) and the deep
    eval (Stage C checkpoint + SealBot), NOT the historical reduced eval_visits
    (128). Parallelism in the concurrent arena makes full sims affordable."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    # SPRT ON so Stage B plays a champion match too.
    cfg = parse_hexfield_config({"multi_stage_eval": {"sprt": {"enabled": True, "max_games": 16}}})
    assert cfg.selfplay.search_visits == 512
    assert cfg.multi_stage_eval.eval_visits == 128  # the OLD reduced default

    rep = _run(run, 40, arena, config=cfg)

    # Every checkpoint match (Stage B SPRT + Stage C top-ups) ran at 512.
    assert arena.ckpt_visits, "no checkpoint matches were played"
    assert all(v == 512 for v in arena.ckpt_visits), arena.ckpt_visits
    # The SealBot zero-point edge also ran at full sims.
    assert arena.sealbot_visits and all(v == 512 for v in arena.sealbot_visits)
    # The budget is recorded in each edge's provenance so the persisted pool —
    # which pools edges by (a, b) label only — stays auditable across a 128->512
    # budget change.
    for e in rep["edges"]:
        assert e["provenance"].get("eval_visits") == 512, (e["opponent"], e["provenance"])


def test_full_search_visits_knob_overrides_default(tmp_path: Path) -> None:
    """``full_search_visits`` pins the eval budget (configurable); when set it is
    threaded everywhere instead of selfplay.search_visits."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    cfg = parse_hexfield_config(
        {"multi_stage_eval": {"full_search_visits": 256, "sprt": {"enabled": True, "max_games": 16}}}
    )
    _run(run, 40, arena, config=cfg)
    assert all(v == 256 for v in arena.ckpt_visits), arena.ckpt_visits
    assert all(v == 256 for v in arena.sealbot_visits)
    # The reduced eval_visits knob is untouched and unused for the budget.
    assert cfg.multi_stage_eval.eval_visits == 128


# =========================================================================== #
# 6. Orchestrator robustness (hardening fixes #1-#5).
# =========================================================================== #
def test_sealbot_unavailable_is_dropped_and_eval_continues(tmp_path: Path) -> None:
    """FAIL-OPEN per opponent (fix #1): when ``play_sealbot_match`` raises (the
    extension isn't built / the adapter import fails / a worker dies), the SealBot
    edge is DROPPED with a logged reason and the eval CONTINUES — the checkpoint
    pairings still produce edges, the pool still anchors and fits, and a verdict
    is still produced. The exception must never propagate."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))

    def boom_sealbot(ckpt, n, **kw):
        # Mimic the real failure: SealBot's compiled extension is missing.
        raise RuntimeError("Compiled minimax_cpp extension not found in /mnt/e/SealBot/current")

    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n))
    rep = mse.run_multistage_eval(
        run, run / "checkpoints" / "epoch_000040.pt", _no_sprt_config(),
        candidate_epoch=40, write_diagnostics=False,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=boom_sealbot,  # the opponent that hard-crashed today
    )

    # The run did NOT abort and produced ratings + a verdict.
    assert rep["verdict"]["label"] in {"PROMOTE", "REGRESS", "INCONCLUSIVE"}
    assert rep["ratings"]["fit"]["converged"] is True
    # Stage C completed (checkpoint edges remain) and recorded WHY SealBot dropped.
    stage_c = next(s for s in rep["stages"] if s["stage"] == "C_deep")
    assert stage_c["status"] == "completed"
    assert "minimax_cpp" in stage_c["sealbot_unavailable"]
    # No SealBot edge in the pooled edges; the SealBot win-rate read is absent.
    assert mse.SEALBOT_LABEL not in {e["opponent"] for e in rep["edges"]}
    assert rep["sealbot_winrate_ci95"] is None
    # The pool anchored on a CHECKPOINT (not SealBot) and the fit still converged.
    assert rep["ratings"]["fit"]["anchor"] != mse.SEALBOT_LABEL
    assert rep["ratings"]["fit"]["anchor_is_sealbot"] is False


def test_anchor_pins_bc_prefit_when_sealbot_absent(tmp_path: Path) -> None:
    """ALWAYS-PIN-A-USABLE-ANCHOR (fix #2): with SealBot disabled, the BT fit
    anchors on the bc_prefit permanent anchor (the canonical lineage base) rather
    than degrading to "no anchor in pool"."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _no_sprt_config(
        opponents={"sealbot_enabled": False},
    )
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.6, n))
    rep = _run(run, 40, arena, config=cfg)

    assert rep["ratings"]["fit"]["converged"] is True
    assert rep["ratings"]["fit"]["anchor"] == "bc_prefit"
    assert rep["ratings"]["fit"]["anchor_is_sealbot"] is False
    assert rep["verdict"]["label"] in {"PROMOTE", "REGRESS", "INCONCLUSIVE"}


def test_anchor_falls_back_to_lowest_checkpoint(tmp_path: Path) -> None:
    """ALWAYS-PIN-A-USABLE-ANCHOR (fix #2), deepest fallback: SealBot off AND no
    bc_prefit/permanent anchors -> the LOWEST available checkpoint opponent (by
    epoch) anchors the pool. The pool never free-floats when any checkpoint edge
    exists."""

    run = _make_run(tmp_path, epochs=(10, 20, 40), bc=False)  # no BC, no ep5 file
    # SealBot off AND permanent anchors pointing at paths that resolve NOWHERE
    # (under no run-data/repo root), so neither bc_prefit nor ep5 is available —
    # forcing the lowest-checkpoint fallback. (The default bc_prefit path resolves
    # from the real repo tree per fix #3, hence the explicit unreachable paths.)
    cfg = _no_sprt_config(
        opponents={
            "sealbot_enabled": False,
            "permanent_anchors": (
                ("bc_prefit", "runs/__hexfield_no_such_bc__/checkpoint_epoch2.pt"),
                ("ep5", "epoch_000005.pt"),
            ),
        }
    )
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.6, n))
    rep = _run(run, 40, arena, config=cfg)

    assert rep["ratings"]["fit"]["converged"] is True
    # bc_prefit/ep5 are absent; the lowest checkpoint opponent that played is the
    # bracket rung ep10 (nearest log-grid rungs below 40 are {10,20}).
    assert rep["ratings"]["fit"]["anchor"] == "ep10"


def test_choose_anchor_order_sealbot_then_bc_then_lowest_checkpoint() -> None:
    """Unit-level preference order of _choose_anchor (fix #2): SealBot > bc_prefit
    > lowest-epoch checkpoint > any, and only ever a label that appears in an
    edge (so the BT anchor-in-edge guard can never trip)."""

    cand = "cand_ep40"
    roster = mse.Roster(
        candidate_label=cand, candidate_epoch=40,
        sealbot=mse.Opponent(label=mse.SEALBOT_LABEL, role="sealbot", ckpt=None, epoch=None),
        champion=mse.Opponent(label="ep20", role="champion", ckpt=Path("x"), epoch=20),
        opponents=(
            mse.Opponent(label="bc_prefit", role="anchor", ckpt=Path("b"), epoch=2),
            mse.Opponent(label="ep5", role="anchor", ckpt=Path("c"), epoch=5),
            mse.Opponent(label="ep10", role="bracket", ckpt=Path("d"), epoch=10),
            mse.Opponent(label="ep20", role="champion", ckpt=Path("e"), epoch=20),
        ),
    )
    BT = mse.eval_stats.BTEdge

    # 1. SealBot present in edges -> SealBot.
    edges = [BT(a=cand, b=mse.SEALBOT_LABEL, wins_a=10, wins_b=10, weight=0.5),
             BT(a=cand, b="bc_prefit", wins_a=10, wins_b=10, weight=1.0)]
    assert mse._choose_anchor(edges, roster) == mse.SEALBOT_LABEL

    # 2. No SealBot edge -> bc_prefit (canonical base, before other anchors).
    edges = [BT(a=cand, b="bc_prefit", wins_a=10, wins_b=10, weight=1.0),
             BT(a=cand, b="ep5", wins_a=10, wins_b=10, weight=1.0)]
    assert mse._choose_anchor(edges, roster) == "bc_prefit"

    # 3. No SealBot, no bc_prefit -> the lowest-epoch checkpoint with an edge.
    edges = [BT(a=cand, b="ep20", wins_a=10, wins_b=10, weight=1.0),
             BT(a=cand, b="ep10", wins_a=10, wins_b=10, weight=1.0)]
    assert mse._choose_anchor(edges, roster) == "ep10"

    # No edges at all -> None (nothing to pin to).
    assert mse._choose_anchor([], roster) is None


def test_bc_prefit_resolves_from_run_data_tree(tmp_path: Path) -> None:
    """bc_prefit PATH resolution (fix #3): when the BC prefit exists under the
    run tree's own ancestor (the run-data tree layout the tests build), it is
    resolved and pinned as an anchor — the run-data-tree root is preferred when
    it holds the file (back-compat with the repo-tree fallback)."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40), bc=True)  # builds the BC file
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000040.pt", MultiStageEvalSection(), candidate_epoch=40
    )
    by_label = {o.label: o for o in roster.opponents}
    assert "bc_prefit" in by_label
    # Resolved to the run-data-tree BC file the fixture created (preferred over
    # the repo tree because it actually exists here).
    assert by_label["bc_prefit"].ckpt == tmp_path / "runs" / "hexfield_bc_1" / "checkpoint_epoch2.pt"
    assert by_label["bc_prefit"].ckpt.is_file()


def test_allocate_budget_floors_each_opponent_to_one_pair() -> None:
    """allocate_budget MINIMUM (fix #4): at a small positive budget every selected
    opponent still gets >=1 CRN pair (2 games) instead of 0, so the champion edge
    (the primary hypothesis) is always played; a zero budget stays all-zeros."""

    # The reported failure: budget 4 used to give per_checkpoint=0 (champion
    # unplayed -> no anchor -> INCONCLUSIVE). Now floored to one pair.
    alloc = mse.allocate_budget(4, n_checkpoint_opponents=3, has_sealbot=True)
    assert alloc["per_checkpoint"] == 2
    assert alloc["per_checkpoint"] % 2 == 0
    assert alloc[mse.SEALBOT_LABEL] >= 2

    # Even at budget 4 with many opponents (the prompt's {sealbot:0,per:0} case).
    alloc2 = mse.allocate_budget(4, n_checkpoint_opponents=10, has_sealbot=True)
    assert alloc2["per_checkpoint"] == 2
    assert alloc2[mse.SEALBOT_LABEL] >= 2

    # Zero budget -> still all-zeros (nothing played at all).
    assert mse.allocate_budget(0, n_checkpoint_opponents=4, has_sealbot=True) == {
        mse.SEALBOT_LABEL: 0,
        "per_checkpoint": 0,
    }

    # Production budgets are unchanged (the floor only lifts small budgets).
    assert mse.allocate_budget(128, n_checkpoint_opponents=4, has_sealbot=True)["per_checkpoint"] == 24


def test_small_budget_run_still_pins_anchor_and_verdicts(tmp_path: Path) -> None:
    """End-to-end of fixes #2+#4 together: a tiny games_budget no longer degrades
    to no-anchor/INCONCLUSIVE-with-no-primary — the champion is played, the pool
    anchors, and a primary hypothesis is produced."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _no_sprt_config(games_budget=4)  # the failing small budget
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 40, arena, config=cfg)

    assert rep["ratings"]["fit"]["converged"] is True
    # The champion edge was actually played -> the primary block exists.
    assert rep["verdict"]["primary"] is not None
    assert rep["verdict"]["primary"]["champion"] == "ep20"
    # Every checkpoint opponent got at least one pair (>=2 decided games).
    champ_calls = [n for lb, n in arena.ckpt_calls if lb == "ep20"]
    assert champ_calls and all(n >= 2 for n in champ_calls)


def test_verdict_reference_lag_decorrelates_target_on_contiguous_ladder(tmp_path: Path) -> None:
    """L-2 STABLE verdict target (fix #5): on a CONTIGUOUS epoch ladder the verdict
    target is a STABLE, de-correlated reference (>= lag epochs below the
    candidate), NOT the immediately-prior checkpoint. The immediately-prior epoch
    still appears as a descriptive bracket-ish opponent, so its info is pooled —
    only the verdict label rests on the stable reference."""

    run = _make_run(tmp_path, epochs=(10, 11, 12, 13, 14, 15))
    # lag=5: candidate ep15 -> eligible <=10 -> reference ep10, NOT ep14.
    cfg = MultiStageEvalSection(verdict_reference_lag=5)
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000015.pt", cfg, candidate_epoch=15
    )
    assert roster.champion is not None
    assert roster.champion.label == "ep10"   # de-correlated reference
    assert roster.champion.epoch == 15 - 5

    # lag=0 restores the legacy immediately-prior behavior (ep14).
    cfg0 = MultiStageEvalSection(verdict_reference_lag=0)
    roster0 = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000015.pt", cfg0, candidate_epoch=15
    )
    assert roster0.champion is not None and roster0.champion.label == "ep14"


def test_verdict_reference_lag_falls_back_when_no_old_enough_prior(tmp_path: Path) -> None:
    """L-2 (fix #5) fallback: when no checkpoint is >= lag epochs below the
    candidate (the first few epochs of a run), the verdict target falls back to
    the nearest prior so a hypothesis still exists rather than vanishing."""

    run = _make_run(tmp_path, epochs=(1, 2, 3))
    cfg = MultiStageEvalSection(verdict_reference_lag=5)  # nothing is 5 below ep3
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000003.pt", cfg, candidate_epoch=3
    )
    # No epoch <= 3-5=-2 exists, so fall back to the nearest prior (ep2).
    assert roster.champion is not None and roster.champion.label == "ep2"


def test_verdict_reference_lag_in_config_summary(tmp_path: Path) -> None:
    """The reference lag is surfaced in the report's config summary for audit."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.6, n))
    rep = _run(run, 40, arena)
    assert rep["meta"]["config"]["verdict_reference_lag"] == 5
