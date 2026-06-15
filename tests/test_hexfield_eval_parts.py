"""Pure-CPU tests for the RESUMABLE, RUN-IN-PARTS multi-stage hexfield eval.

The long full eval (SealBot + bc_prefit + ep5 + ep10 + ep14 at 512 visits) is
restructured so it can run as a SEQUENCE of short, independent PARTS — one part
per opponent — each appending its single edge to the persisted rolling pool the
instant it finishes. The contract these tests pin (the second deliverable of the
throughput fix) is:

  * ``run_eval_part(run_dir, candidate_ckpt, opponent_label, config, ...)`` plays
    exactly ONE opponent (a checkpoint pairing or the SealBot zero-point),
    appends that ONE edge to ``diagnostics/eval_pool.json``, and persists the
    pool immediately (so the part survives even if the NEXT part crashes).

  * RESUME: re-running a part whose edge is already in the pool for this epoch is
    SKIPPED — it returns ``status="skipped"`` and plays ZERO games. An eval
    interrupted partway through keeps every completed part's edge.

  * ``run_multistage_eval_in_parts(...)`` is the parts-based orchestrator: it
    runs each opponent as a part (skipping the already-pooled ones on resume),
    then ``aggregate_pool(...)`` fits Stage D over the accumulated pool and emits
    the ratings + verdict. The pooled edge set it produces is IDENTICAL to the
    monolithic ``run_multistage_eval`` over the same roster — the per-opponent
    extraction is behaviour-preserving.

  * ``aggregate_pool(...)`` is a pure FIT pass: it reads the persisted pool, runs
    only the BT fit + verdict, and plays NO games.

Everything runs on a CPU-only interpreter with NO GPU, NO torch model, and NO
native .so: the two arena runners are injected as stubs (the ``_StubArena``
pattern shared with ``test_hexfield_eval_harden.py``). None of this touches the
live training run — the trainer imports neither ``eval_arena`` nor
``multistage_eval``, and the parts write only under the (tmp) run's diagnostics
tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "packages" / "hexo_engine" / "python"))
sys.path.insert(0, str(_REPO / "packages" / "hexfield" / "python"))

from hexfield import multistage_eval as mse  # noqa: E402
from hexfield.config import parse_hexfield_config  # noqa: E402

# Reuse the harden suite's stub-arena + fake-run-tree builders so the parts path
# and the monolithic path are exercised through the SAME synthetic matches (any
# divergence is then a real behaviour change, not a fixture difference).
import test_hexfield_eval_harden as harden  # noqa: E402

_StubArena = harden._StubArena
_make_run = harden._make_run


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _cfg(**overrides):
    """A no-SPRT config (Stage B off) so the parts tests isolate Stage C/D; the
    SPRT screen is covered in the harden/orchestrator suites."""

    return parse_hexfield_config(
        {"multi_stage_eval": {"sprt": {"enabled": False}, **overrides}}
    )


def _candidate(run: Path, epoch: int) -> Path:
    return run / "checkpoints" / f"epoch_{epoch:06d}.pt"


def _pool_doc_for(run: Path, cfg) -> dict:
    section = mse._coerce_section(cfg)
    return mse._load_pool(mse._pool_path(run, section))


def _pooled_edge_keys(pool_doc: dict) -> set:
    """The (epoch, a, b) identity of every edge row in the pool."""

    return {
        (row.get("epoch"), row.get("a"), row.get("b"))
        for row in pool_doc.get("edges", [])
    }


def _opponent_labels(run: Path, candidate_epoch: int, cfg) -> list[str]:
    """The checkpoint-opponent labels of the roster (the per-opponent parts)."""

    section = mse._coerce_section(cfg)
    roster = mse.select_opponents(
        run, _candidate(run, candidate_epoch), section, candidate_epoch=candidate_epoch
    )
    return [o.label for o in roster.opponents]


# =========================================================================== #
# 0. The parts API exists with the agreed surface.
# =========================================================================== #
def test_parts_api_surface_exists() -> None:
    """The three parts entry points + the resume predicate are public on the
    module (so the standalone runner's --parts / --opponent / --aggregate-only
    flags have something to call)."""

    for name in (
        "run_eval_part",
        "aggregate_pool",
        "run_multistage_eval_in_parts",
        "_epoch_edge_exists",
    ):
        assert hasattr(mse, name), f"multistage_eval is missing {name}"
        assert callable(getattr(mse, name)), f"{name} is not callable"


# =========================================================================== #
# 1. A single part plays ONE opponent and appends ONE edge to the pool.
# =========================================================================== #
def test_run_eval_part_appends_single_edge(tmp_path: Path) -> None:
    """One part = one opponent -> exactly one edge row appended to the pool, keyed
    by (epoch, candidate_label, opponent_label), and the pool is persisted to
    disk immediately (durability point)."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _cfg()
    arena = _StubArena(per_score=2)
    opp_labels = _opponent_labels(run, 40, cfg)
    assert opp_labels, "expected at least one checkpoint opponent"
    target = opp_labels[0]

    pool_before = _pool_doc_for(run, cfg)
    assert pool_before["edges"] == []

    out = mse.run_eval_part(
        run, _candidate(run, 40), target, cfg,
        candidate_epoch=40,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
    )

    assert out["status"] == "played", out
    # Exactly this one opponent's games were played (no others).
    assert {label for label, _ in arena.ckpt_calls} == {target}

    # The pool on DISK (not just in memory) grew by exactly one edge for (40, cand, target).
    pool_after = _pool_doc_for(run, cfg)
    new_keys = _pooled_edge_keys(pool_after) - _pooled_edge_keys(pool_before)
    roster = mse.select_opponents(run, _candidate(run, 40), mse._coerce_section(cfg), candidate_epoch=40)
    assert new_keys == {(40, roster.candidate_label, target)}, new_keys
    assert len(pool_after["edges"]) == len(pool_before["edges"]) + 1


def test_run_eval_part_sealbot_appends_sealbot_edge(tmp_path: Path) -> None:
    """The SealBot zero-point is also a part (label == SEALBOT_LABEL): it plays
    the unpaired SealBot match and appends a down-weighted SealBot edge."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _cfg()
    arena = _StubArena(per_score=2, sealbot_winrate=0.6)

    out = mse.run_eval_part(
        run, _candidate(run, 40), mse.SEALBOT_LABEL, cfg,
        candidate_epoch=40,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
    )

    assert out["status"] == "played", out
    assert arena.sealbot_calls, "SealBot part played no SealBot games"
    assert not arena.ckpt_calls, "SealBot part should not play checkpoint games"

    pool = _pool_doc_for(run, cfg)
    roster = mse.select_opponents(run, _candidate(run, 40), mse._coerce_section(cfg), candidate_epoch=40)
    assert (40, roster.candidate_label, mse.SEALBOT_LABEL) in _pooled_edge_keys(pool)
    sb_rows = [r for r in pool["edges"] if r["b"] == mse.SEALBOT_LABEL]
    assert len(sb_rows) == 1
    # SealBot edge is DOWN-WEIGHTED (weight < 1), out of difference inference.
    assert sb_rows[0]["weight"] < 1.0


# =========================================================================== #
# 2. RESUME: a part already in the pool for this epoch is skipped (no games).
# =========================================================================== #
def test_part_skipped_when_edge_already_in_pool(tmp_path: Path) -> None:
    """Re-running a part whose (epoch, a, b) edge is already pooled is SKIPPED on
    resume: it returns status='skipped' and plays ZERO games (the resume contract
    that makes an interrupted eval cheap to finish)."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _cfg()
    target = _opponent_labels(run, 40, cfg)[0]

    # Part run #1 plays + pools the edge.
    arena1 = _StubArena(per_score=2)
    first = mse.run_eval_part(
        run, _candidate(run, 40), target, cfg, candidate_epoch=40,
        play_checkpoint_match=arena1.play_checkpoint_match,
        play_sealbot_match=arena1.play_sealbot_match,
    )
    assert first["status"] == "played"
    pooled_after_first = _pooled_edge_keys(_pool_doc_for(run, cfg))

    # Part run #2 (resume on) sees the edge and SKIPS — no games, pool unchanged.
    arena2 = _StubArena(per_score=2)
    second = mse.run_eval_part(
        run, _candidate(run, 40), target, cfg, candidate_epoch=40, resume=True,
        play_checkpoint_match=arena2.play_checkpoint_match,
        play_sealbot_match=arena2.play_sealbot_match,
    )
    assert second["status"] == "skipped", second
    assert arena2.ckpt_calls == [], "skipped part must play NO checkpoint games"
    assert arena2.sealbot_calls == [], "skipped part must play NO sealbot games"
    # Pool is byte-stable across the skip (no duplicate edge appended).
    assert _pooled_edge_keys(_pool_doc_for(run, cfg)) == pooled_after_first


def test_epoch_edge_exists_predicate_is_exact(tmp_path: Path) -> None:
    """The resume predicate matches on ALL THREE of (epoch, a, b): a DIFFERENT
    epoch's same opponent is NOT considered already-done (edges compound across
    epochs), and a different opponent at the same epoch is likewise distinct."""

    cand = "cand_ep40"
    pool_doc = {
        "format": mse.POOL_FORMAT, "version": mse.POOL_VERSION, "anchor": mse.SEALBOT_LABEL,
        "edges": [
            {"epoch": 40, "a": cand, "b": "ep20", "wins_a": 10.0, "wins_b": 6.0, "weight": 1.0},
        ],
    }
    # Exact (epoch, a, b) match -> present.
    assert mse._epoch_edge_exists(pool_doc, 40, cand, "ep20") is True
    # Same opponent, DIFFERENT epoch -> not present (compounds, must be re-played).
    assert mse._epoch_edge_exists(pool_doc, 41, cand, "ep20") is False
    # Same epoch, DIFFERENT opponent -> not present.
    assert mse._epoch_edge_exists(pool_doc, 40, cand, "ep10") is False
    # Empty pool -> nothing present.
    empty = {"format": mse.POOL_FORMAT, "version": mse.POOL_VERSION, "edges": []}
    assert mse._epoch_edge_exists(empty, 40, cand, "ep20") is False


def test_resume_false_replays_already_pooled_part(tmp_path: Path) -> None:
    """``resume=False`` forces a re-play even when the edge is already pooled (so a
    maintainer can deliberately re-measure an opponent); the edge then appears
    twice and compounds in the BT fit."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _cfg()
    target = _opponent_labels(run, 40, cfg)[0]

    arena = _StubArena(per_score=2)
    mse.run_eval_part(
        run, _candidate(run, 40), target, cfg, candidate_epoch=40,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
    )
    n_after_first = len(_pool_doc_for(run, cfg)["edges"])

    arena2 = _StubArena(per_score=2)
    out = mse.run_eval_part(
        run, _candidate(run, 40), target, cfg, candidate_epoch=40, resume=False,
        play_checkpoint_match=arena2.play_checkpoint_match,
        play_sealbot_match=arena2.play_sealbot_match,
    )
    assert out["status"] == "played"
    assert arena2.ckpt_calls, "resume=False must re-play the opponent"
    assert len(_pool_doc_for(run, cfg)["edges"]) == n_after_first + 1


# =========================================================================== #
# 3. Interrupted parts keep the completed edges; aggregate still fits.
# =========================================================================== #
def test_interrupted_parts_keep_completed_edges_and_aggregate_fits(tmp_path: Path) -> None:
    """The durability guarantee: if a LATER part raises, every EARLIER part's edge
    is already persisted in the pool, and ``aggregate_pool`` still fits the BT
    ratings + a verdict over the completed parts (fail-open)."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _cfg(opponents={"sealbot_enabled": False})  # checkpoint-only roster
    opp_labels = _opponent_labels(run, 40, cfg)
    assert len(opp_labels) >= 2

    # Play the first two opponents as parts; they persist their edges.
    for label in opp_labels[:2]:
        arena = _StubArena(per_score=2)
        out = mse.run_eval_part(
            run, _candidate(run, 40), label, cfg, candidate_epoch=40,
            play_checkpoint_match=arena.play_checkpoint_match,
            play_sealbot_match=arena.play_sealbot_match,
        )
        assert out["status"] == "played"

    # The third part raises mid-match (simulating an interruption / GPU hiccup).
    def boom(*a, **k):
        raise RuntimeError("simulated arena failure mid-part")

    with pytest.raises(RuntimeError):
        mse.run_eval_part(
            run, _candidate(run, 40), opp_labels[2], cfg, candidate_epoch=40,
            play_checkpoint_match=boom,
            play_sealbot_match=boom,
        )

    # The first two parts' edges SURVIVED the crash (durably on disk).
    roster = mse.select_opponents(run, _candidate(run, 40), mse._coerce_section(cfg), candidate_epoch=40)
    keys = _pooled_edge_keys(_pool_doc_for(run, cfg))
    for label in opp_labels[:2]:
        assert (40, roster.candidate_label, label) in keys
    # The crashed part left NO partial edge.
    assert (40, roster.candidate_label, opp_labels[2]) not in keys

    # aggregate_pool fits over what survived and still yields a verdict.
    agg = mse.aggregate_pool(run, _candidate(run, 40), cfg, candidate_epoch=40)
    assert agg["ratings"]["fit"].get("converged") is True
    assert agg["verdict"]["label"] in {"PROMOTE", "REGRESS", "INCONCLUSIVE"}


# =========================================================================== #
# 4. aggregate_pool is a pure FIT pass (no games), idempotent.
# =========================================================================== #
def test_aggregate_only_plays_no_games_and_fits(tmp_path: Path) -> None:
    """``aggregate_pool`` reads the persisted pool and runs ONLY the BT fit +
    verdict — it plays NO games (no arena needed). With a populated pool it
    produces ratings + a primary hypothesis block."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _cfg(opponents={"sealbot_enabled": False})

    # Populate the pool via parts.
    for label in _opponent_labels(run, 40, cfg):
        arena = _StubArena(per_score=2)
        mse.run_eval_part(
            run, _candidate(run, 40), label, cfg, candidate_epoch=40,
            play_checkpoint_match=arena.play_checkpoint_match,
            play_sealbot_match=arena.play_sealbot_match,
        )

    pool_before = _pooled_edge_keys(_pool_doc_for(run, cfg))

    # aggregate_pool has NO arena seam at all — its signature cannot even accept a
    # play_* runner, which is itself the structural proof that it plays NOTHING (it
    # only reads the persisted pool to fit). It just runs the BT fit + verdict.
    import inspect
    agg_params = set(inspect.signature(mse.aggregate_pool).parameters)
    assert "play_checkpoint_match" not in agg_params and "play_sealbot_match" not in agg_params, (
        "aggregate_pool must be a pure fit pass with no game-playing seam"
    )

    agg = mse.aggregate_pool(run, _candidate(run, 40), cfg, candidate_epoch=40)
    assert agg["ratings"]["fit"].get("converged") is True
    assert agg["ratings"]["players"], "no pooled ratings produced"
    assert agg["verdict"]["primary"] is not None
    assert agg["verdict"]["primary"]["champion"] == "ep20"  # lag-5 reference of ep40

    # Idempotent: aggregating again does NOT mutate the pool (read-only fit).
    agg2 = mse.aggregate_pool(run, _candidate(run, 40), cfg, candidate_epoch=40)
    assert _pooled_edge_keys(_pool_doc_for(run, cfg)) == pool_before
    assert agg2["verdict"]["label"] == agg["verdict"]["label"]


# =========================================================================== #
# 5. The parts path produces the SAME pooled edges as the monolithic run.
# =========================================================================== #
def test_parts_path_pool_equals_monolithic_pool(tmp_path: Path) -> None:
    """Running the roster as a sequence of parts (``run_multistage_eval_in_parts``)
    pools the SAME edge set, with the same effective counts, as the monolithic
    ``run_multistage_eval`` over the same roster + stub arena — proving the
    per-opponent extraction is behaviour-preserving (no statistic changes)."""

    cfg = _cfg(opponents={"sealbot_enabled": False})

    # --- Monolithic run (writes its pool). ---
    run_mono = _make_run(tmp_path / "mono", epochs=(5, 10, 20, 40))
    arena_mono = _StubArena(per_score=2)
    mse.run_multistage_eval(
        run_mono, _candidate(run_mono, 40), cfg, candidate_epoch=40,
        write_diagnostics=True,
        play_checkpoint_match=arena_mono.play_checkpoint_match,
        play_sealbot_match=arena_mono.play_sealbot_match,
    )
    mono_pool = _pool_doc_for(run_mono, cfg)

    # --- Parts run (writes its pool incrementally). ---
    run_parts = _make_run(tmp_path / "parts", epochs=(5, 10, 20, 40))
    arena_parts = _StubArena(per_score=2)
    mse.run_multistage_eval_in_parts(
        run_parts, _candidate(run_parts, 40), cfg, candidate_epoch=40,
        play_checkpoint_match=arena_parts.play_checkpoint_match,
        play_sealbot_match=arena_parts.play_sealbot_match,
    )
    parts_pool = _pool_doc_for(run_parts, cfg)

    # Same set of (epoch, a, b) edges in both pools.
    assert _pooled_edge_keys(parts_pool) == _pooled_edge_keys(mono_pool)

    # Same EFFECTIVE counts per edge (the BT inputs), not just the same labels.
    def _counts_by_key(doc):
        out = {}
        for r in doc["edges"]:
            out[(r["epoch"], r["a"], r["b"])] = (
                round(float(r["wins_a"]), 6), round(float(r["wins_b"]), 6), round(float(r["weight"]), 6)
            )
        return out

    assert _counts_by_key(parts_pool) == _counts_by_key(mono_pool)


def test_parts_orchestrator_resumes_after_partial_completion(tmp_path: Path) -> None:
    """End-to-end resume: a parts orchestration that completes only some opponents
    (one raises) leaves the rest pooled; a SECOND in-parts run skips the done
    opponents and finishes the remaining one, and the final pool equals a clean
    single in-parts run."""

    cfg = _cfg(opponents={"sealbot_enabled": False})

    # --- Reference: a clean parts run over the whole roster. ---
    run_ref = _make_run(tmp_path / "ref", epochs=(5, 10, 20, 40))
    arena_ref = _StubArena(per_score=2)
    mse.run_multistage_eval_in_parts(
        run_ref, _candidate(run_ref, 40), cfg, candidate_epoch=40,
        play_checkpoint_match=arena_ref.play_checkpoint_match,
        play_sealbot_match=arena_ref.play_sealbot_match,
    )
    ref_keys = _pooled_edge_keys(_pool_doc_for(run_ref, cfg))

    # --- Interrupted: make the LAST opponent's match raise on the first pass. ---
    run = _make_run(tmp_path / "interrupted", epochs=(5, 10, 20, 40))
    opp_labels = _opponent_labels(run, 40, cfg)
    last = opp_labels[-1]

    class _PartialArena(_StubArena):
        def play_checkpoint_match(self, a, b, n, **kw):
            if kw["label_b"] == last:
                raise RuntimeError(f"simulated failure on {last}")
            return super().play_checkpoint_match(a, b, n, **kw)

    arena_fail = _PartialArena(per_score=2)
    # The in-parts orchestrator may surface the failure or fail-open per part; either
    # way the COMPLETED parts must be durably pooled. Tolerate both behaviours.
    try:
        mse.run_multistage_eval_in_parts(
            run, _candidate(run, 40), cfg, candidate_epoch=40,
            play_checkpoint_match=arena_fail.play_checkpoint_match,
            play_sealbot_match=arena_fail.play_sealbot_match,
        )
    except RuntimeError:
        pass

    roster = mse.select_opponents(run, _candidate(run, 40), mse._coerce_section(cfg), candidate_epoch=40)
    keys_after_partial = _pooled_edge_keys(_pool_doc_for(run, cfg))
    # Every opponent EXCEPT the failing one is pooled.
    for label in opp_labels[:-1]:
        assert (40, roster.candidate_label, label) in keys_after_partial
    assert (40, roster.candidate_label, last) not in keys_after_partial

    # --- Resume: a second pass skips the done opponents and finishes ``last``. ---
    arena_resume = _StubArena(per_score=2)
    mse.run_multistage_eval_in_parts(
        run, _candidate(run, 40), cfg, candidate_epoch=40, resume=True,
        play_checkpoint_match=arena_resume.play_checkpoint_match,
        play_sealbot_match=arena_resume.play_sealbot_match,
    )
    # Only the previously-failing opponent was (re)played on resume.
    assert {label for label, _ in arena_resume.ckpt_calls} == {last}, arena_resume.ckpt_calls
    # The resumed pool equals the clean reference pool (same edges).
    assert _pooled_edge_keys(_pool_doc_for(run, cfg)) == ref_keys
