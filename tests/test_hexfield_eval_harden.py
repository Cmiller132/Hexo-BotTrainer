"""Pure-CPU regression tests for the hexfield STANDALONE eval HARDENING fixes.

These pin the specific failures that broke the multi-stage eval on real runs (it
is implemented + concurrent + fast, but FAILED end-to-end), plus the three
adversarial-review follow-ups. They are deliberately a SEPARATE net from the
existing ``test_hexfield_multistage_eval.py`` / ``..._eval_arena_concurrent.py``
suites so a regression in any one fix is caught here independently.

Everything runs on a CPU-only interpreter with NO GPU, NO torch model, and NO
native .so: the orchestrator's two arena runners are injected as stubs
(``play_checkpoint_match`` / ``play_sealbot_match``), and the L-1 forced-opening
test drives the REAL ``eval_arena.play_checkpoint_match`` against a fake engine +
a fake multi-root session through the ``make_session`` / ``build_evaluators``
seams. None of this touches the live training run (the training pipeline does not
import ``eval_arena`` / ``multistage_eval``, and these tests write nothing under a
run tree — ``write_diagnostics=False`` throughout).

Coverage (one section per fix):
  1. FAIL-OPEN: a SealBot opponent that raises is DROPPED and the eval still
     completes with ratings from the remaining (checkpoint) opponents.
  2. ALWAYS-PIN-AN-ANCHOR: with SealBot off the anchor falls back to bc_prefit
     (resolved under the repo tree) or, failing that, the lowest checkpoint; the
     BT fit CONVERGES and ratings are produced (never INCONCLUSIVE-no-anchor).
  3. bc_prefit PATH resolves given a REPO-TREE layout (the run-data tree lacks
     the file; the repo tree has it).
  4. allocate_budget gives >=1 CRN pair per selected opponent at small budgets.
  5. L-1 forced-opening: a pair's two seat-swapped games share the RECORDED
     opening LINE even when the two nets would pick different opening moves.
  6. L-2: the verdict target is the STABLE reference (lag), not the
     immediately-prior epoch.
  7. M-1: the Stage B SPRT triage labels agree with the gross-regression config.
  8. PURE-EVAL invariant still holds (gating/promotion off; tripwire fires).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "packages" / "hexo_engine" / "python"))
sys.path.insert(0, str(_REPO / "packages" / "hexfield" / "python"))

from hexfield import eval_arena  # noqa: E402
from hexfield import multistage_eval as mse  # noqa: E402
from hexfield.config import MultiStageEvalSection, parse_hexfield_config  # noqa: E402
from hexfield.geometry import pack_action_id, unpack_action_id  # noqa: E402
from hexo_engine.types import Player  # noqa: E402


# --------------------------------------------------------------------------- #
# Orchestrator-level stubs (the two arena runners are injection seams).
# --------------------------------------------------------------------------- #
def _make_run(tmp_path: Path, epochs=(5, 10, 20, 40), *, bc: bool = True) -> Path:
    """A fake run tree ``<tmp>/runs/r/checkpoints/epoch_*.pt`` (+ BC sibling).

    Mirrors the real layout ``select_opponents`` resolves: in-run checkpoints
    under ``<run>/checkpoints`` and the BC prefit in a sibling ``hexfield_bc_1``
    referenced by the default permanent-anchor path. Files are empty stubs —
    selection is pure path resolution and never loads a checkpoint.
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


def _paired_match(label_a: str, label_b: str, n_pairs: int, a_score: int) -> dict:
    """Fake ``play_checkpoint_match`` result with a real pentanomial block.

    ``a_score`` is the per-pair net-A score in {0, 1, 2}; we mix it with a split
    pair so the pair-level SE is non-degenerate (all-identical pairs give SE 0,
    which is not representative). Shape matches ``eval_arena._build_match_result``
    + ``_pentanomial_block`` so the orchestrator consumes it unchanged.
    """

    pairs = []
    a_wins = b_wins = 0
    for i in range(max(n_pairs, 1)):
        score = a_score if i % 2 == 0 else 1  # alternate decisive / split
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
    for p in pairs:
        hist[str(p["pentanomial_a_score"])] += 1
    decided = a_wins + b_wins
    return {
        "meta": {"label_a": label_a, "label_b": label_b, "games_requested": 2 * len(pairs)},
        "score": {
            "completed": decided, "truncated": 0, "aborted_budget": 0,
            "a_wins": a_wins, "b_wins": b_wins, "decided": decided,
            "a_winrate_decided": (a_wins / decided) if decided else None,
        },
        "pentanomial": {
            "n_pairs": len(pairs), "n_full_pairs": len(pairs),
            "pairs": pairs, "histogram_a_wins": hist,
        },
    }


def _sealbot_match(label: str, n: int, winrate: float) -> dict:
    wins = int(round(winrate * n))
    return {
        "meta": {"games_requested": n},
        "score": {"completed": n, "a_wins": wins, "b_wins": n - wins, "decided": n},
    }


class _StubArena:
    """Records calls and returns synthetic matches at a configured strength.

    ``per_score`` is the per-pair net-A score (0/1/2) the candidate scores vs
    every checkpoint opponent; ``sealbot_winrate`` sets the binomial SealBot edge.
    ``sealbot_raises`` (a callable or None) lets a test make the SealBot runner
    blow up to exercise the fail-open path.
    """

    def __init__(self, *, per_score: int = 2, sealbot_winrate: float = 0.6,
                 sealbot_raises=None) -> None:
        self._per_score = per_score
        self._sealbot_winrate = sealbot_winrate
        self._sealbot_raises = sealbot_raises
        self.ckpt_calls: list[tuple[str, int]] = []
        self.sealbot_calls: list[int] = []

    def play_checkpoint_match(self, a, b, n, **kw) -> dict:
        self.ckpt_calls.append((kw["label_b"], n))
        return _paired_match(kw["label_a"], kw["label_b"], max(1, n // 2), self._per_score)

    def play_sealbot_match(self, ckpt, n, **kw) -> dict:
        if self._sealbot_raises is not None:
            self._sealbot_raises()
        self.sealbot_calls.append(n)
        return _sealbot_match(kw.get("label", "hexfield"), n, self._sealbot_winrate)


def _run(run_dir: Path, candidate_epoch: int, arena: _StubArena, *, config=None, **kw) -> dict:
    cfg = config if config is not None else parse_hexfield_config(
        {"multi_stage_eval": {"sprt": {"enabled": False}}}
    )
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


def _no_sprt(**overrides):
    return parse_hexfield_config({"multi_stage_eval": {"sprt": {"enabled": False}, **overrides}})


# =========================================================================== #
# (1) FAIL-OPEN: a SealBot opponent that raises is dropped; eval still completes.
# =========================================================================== #
def test_sealbot_unavailable_drops_edge_and_completes_with_ratings(tmp_path: Path) -> None:
    """The exact production failure: SealBot's compiled extension is missing, so
    ``play_sealbot_match`` raises. The orchestrator must DROP that one edge (per
    opponent), record WHY, and still complete with ratings from the checkpoint
    opponents — the exception must NEVER propagate out of the run."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))

    def boom():
        raise RuntimeError(
            "Compiled minimax_cpp extension not found in /mnt/e/SealBot/current"
        )

    arena = _StubArena(per_score=2, sealbot_raises=boom)
    # Drive the run directly (not via _run) so a propagating exception fails the
    # test loudly rather than being swallowed anywhere.
    rep = mse.run_multistage_eval(
        run, run / "checkpoints" / "epoch_000040.pt", _no_sprt(),
        candidate_epoch=40, write_diagnostics=False,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
    )

    # Ratings + a verdict were produced despite SealBot being unavailable.
    assert rep["ratings"]["fit"]["converged"] is True
    assert rep["verdict"]["label"] in {"PROMOTE", "REGRESS", "INCONCLUSIVE"}
    # Stage C completed (checkpoint edges remain) and recorded the drop reason.
    stage_c = next(s for s in rep["stages"] if s["stage"] == "C_deep")
    assert stage_c["status"] == "completed"
    assert "minimax_cpp" in stage_c["sealbot_unavailable"]
    # No SealBot edge pooled, and the SealBot win-rate read is absent.
    assert mse.SEALBOT_LABEL not in {e["opponent"] for e in rep["edges"]}
    assert rep["sealbot_winrate_ci95"] is None
    # The pool anchored on a CHECKPOINT, not on the missing SealBot.
    assert rep["ratings"]["fit"]["anchor"] != mse.SEALBOT_LABEL
    assert rep["ratings"]["fit"]["anchor_is_sealbot"] is False


def test_sealbot_import_error_is_also_failed_open(tmp_path: Path) -> None:
    """The adapter imports ``hexo_engine`` at module top, so a CPU-only box can
    raise ImportError/ModuleNotFoundError BEFORE the availability check is ever
    reached. The fail-open is broad enough to catch that too (not just
    SealBotUnavailableError)."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))

    def boom_import():
        raise ModuleNotFoundError("No module named 'hexo_engine'")

    arena = _StubArena(per_score=2, sealbot_raises=boom_import)
    rep = _run(run, 40, arena)
    assert rep["ratings"]["fit"]["converged"] is True
    stage_c = next(s for s in rep["stages"] if s["stage"] == "C_deep")
    assert stage_c["status"] == "completed"
    assert "ModuleNotFoundError" in stage_c["sealbot_unavailable"]


# =========================================================================== #
# (2) ALWAYS-PIN-AN-ANCHOR: bc_prefit (repo tree) or lowest checkpoint; fit
#     converges; ratings produced (never INCONCLUSIVE-no-anchor).
# =========================================================================== #
def test_anchor_pins_bc_prefit_when_sealbot_disabled(tmp_path: Path) -> None:
    """With SealBot disabled, the BT fit anchors on bc_prefit (the canonical
    lineage base) rather than degrading to 'no anchor in pool'; ratings exist."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _no_sprt(opponents={"sealbot_enabled": False})
    rep = _run(run, 40, _StubArena(per_score=2), config=cfg)

    assert rep["ratings"]["fit"]["converged"] is True
    assert rep["ratings"]["fit"]["anchor"] == "bc_prefit"
    assert rep["ratings"]["fit"]["anchor_is_sealbot"] is False
    # Ratings were actually produced (the failure mode was an empty table).
    assert rep["ratings"]["players"], "no pooled ratings produced"
    assert rep["verdict"]["label"] in {"PROMOTE", "REGRESS", "INCONCLUSIVE"}


def test_anchor_falls_back_to_lowest_checkpoint_when_no_anchors(tmp_path: Path) -> None:
    """Deepest fallback: SealBot off AND no resolvable bc_prefit/ep5 -> the LOWEST
    available checkpoint opponent (by epoch) anchors the pool. The pool must never
    free-float when ANY checkpoint edge exists."""

    run = _make_run(tmp_path, epochs=(10, 20, 40), bc=False)  # no BC, no ep5 file
    # Point the permanent anchors at paths that resolve NOWHERE so neither
    # bc_prefit nor ep5 is available, forcing the lowest-checkpoint fallback.
    cfg = _no_sprt(
        opponents={
            "sealbot_enabled": False,
            "permanent_anchors": (
                ("bc_prefit", "runs/__no_such_bc_dir__/checkpoint_epoch2.pt"),
                ("ep5", "epoch_000005.pt"),
            ),
        }
    )
    rep = _run(run, 40, _StubArena(per_score=2), config=cfg)

    assert rep["ratings"]["fit"]["converged"] is True
    # Nearest log-grid rungs below 40 are {10, 20}; the lowest played is ep10.
    assert rep["ratings"]["fit"]["anchor"] == "ep10"
    assert rep["ratings"]["players"]


def test_choose_anchor_preference_order() -> None:
    """Unit-level: SealBot > bc_prefit > lowest-epoch checkpoint > any, and only
    ever a label that appears in an edge (so the BT anchor-in-edge guard cannot
    trip). None only when there is literally no non-candidate edge."""

    cand = "cand_ep40"
    roster = mse.Roster(
        candidate_label=cand, candidate_epoch=40,
        sealbot=mse.Opponent(mse.SEALBOT_LABEL, "sealbot", None, None),
        champion=mse.Opponent("ep20", "champion", Path("e"), 20),
        opponents=(
            mse.Opponent("bc_prefit", "anchor", Path("b"), 2),
            mse.Opponent("ep5", "anchor", Path("c"), 5),
            mse.Opponent("ep10", "bracket", Path("d"), 10),
            mse.Opponent("ep20", "champion", Path("e"), 20),
        ),
    )
    BT = mse.eval_stats.BTEdge

    # SealBot present -> SealBot.
    assert mse._choose_anchor(
        [BT(cand, mse.SEALBOT_LABEL, 10, 10, 0.5), BT(cand, "bc_prefit", 10, 10, 1.0)], roster
    ) == mse.SEALBOT_LABEL
    # No SealBot edge -> bc_prefit before other anchors.
    assert mse._choose_anchor(
        [BT(cand, "bc_prefit", 10, 10, 1.0), BT(cand, "ep5", 10, 10, 1.0)], roster
    ) == "bc_prefit"
    # No SealBot/bc_prefit -> lowest-epoch checkpoint with an edge.
    assert mse._choose_anchor(
        [BT(cand, "ep20", 10, 10, 1.0), BT(cand, "ep10", 10, 10, 1.0)], roster
    ) == "ep10"
    # No edges at all -> None.
    assert mse._choose_anchor([], roster) is None
    # Only a candidate self-edge (no non-candidate label) -> None.
    assert mse._choose_anchor([BT(cand, cand, 1, 1, 1.0)], roster) is None


# =========================================================================== #
# (3) bc_prefit PATH resolves under a REPO-TREE layout.
# =========================================================================== #
def test_bc_prefit_resolves_from_repo_tree(tmp_path: Path, monkeypatch) -> None:
    """The production root cause: the RUN-DATA tree (where epoch_*.pt live) has a
    ``hexfield_bc_1/`` dir but NO top-level ``checkpoint_epoch2.pt`` — that file
    lives only in the REPO tree that ships this package. The resolver must try the
    repo-tree root (derived from the module's own location) and return the file
    that ACTUALLY exists there, not the non-existent run-data-tree path."""

    # Build a SEPARATE run-data tree with a hexfield_bc_1 DIR but no BC FILE.
    data_tree = tmp_path / "Hexo-BotTrainer"
    run = data_tree / "runs" / "hexfield_main_1"
    (run / "checkpoints").mkdir(parents=True)
    (data_tree / "runs" / "hexfield_bc_1").mkdir(parents=True)  # dir only, no file
    for epoch in (5, 10, 20, 40):
        (run / "checkpoints" / f"epoch_{epoch:06d}.pt").write_text("stub", encoding="utf-8")

    # Build a fake REPO tree that DOES hold the BC file, and make the resolver
    # treat it as the repo root by monkeypatching the module file location.
    repo_tree = tmp_path / "Hexo-BotTrainer-hexgt"
    bc_file = repo_tree / "runs" / "hexfield_bc_1" / "checkpoint_epoch2.pt"
    bc_file.parent.mkdir(parents=True)
    bc_file.write_text("stub", encoding="utf-8")
    # _resolve_anchor_path derives the repo root as Path(__file__).parents[4]; make
    # that resolve to our fake repo tree (parents[4] == repo_tree).
    fake_module_file = repo_tree / "packages" / "hexfield" / "python" / "hexfield" / "multistage_eval.py"
    fake_module_file.parent.mkdir(parents=True)
    fake_module_file.write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(mse, "__file__", str(fake_module_file))

    # The default permanent-anchor path is "runs/hexfield_bc_1/checkpoint_epoch2.pt".
    resolved = mse._resolve_anchor_path(
        run, run / "checkpoints", "runs/hexfield_bc_1/checkpoint_epoch2.pt"
    )
    # It resolved to the REPO-tree file (which exists), not the run-data path.
    assert resolved == bc_file
    assert resolved.is_file()
    # And select_opponents therefore pins bc_prefit as an anchor opponent.
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000040.pt", MultiStageEvalSection(), candidate_epoch=40
    )
    by_label = {o.label: o for o in roster.opponents}
    assert "bc_prefit" in by_label
    assert by_label["bc_prefit"].ckpt == bc_file


def test_bc_prefit_prefers_run_data_tree_when_present(tmp_path: Path) -> None:
    """Back-compat: when the BC file DOES exist under the run-data tree, that root
    is preferred (returned first), so the repo-tree fallback only fires when the
    run-data tree lacks the file."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40), bc=True)  # builds the BC file
    resolved = mse._resolve_anchor_path(
        run, run / "checkpoints", "runs/hexfield_bc_1/checkpoint_epoch2.pt"
    )
    assert resolved == tmp_path / "runs" / "hexfield_bc_1" / "checkpoint_epoch2.pt"
    assert resolved.is_file()


# =========================================================================== #
# (4) allocate_budget: >=1 CRN pair per selected opponent at small budgets.
# =========================================================================== #
def test_allocate_budget_floors_to_one_pair_at_small_budgets() -> None:
    """The reported failure (budget 4 -> per_checkpoint 0, champion unplayed -> no
    anchor -> INCONCLUSIVE) is fixed: a POSITIVE budget floors every selected
    opponent to one CRN pair (2 games), SealBot included; a ZERO budget still
    returns all-zeros (nothing played)."""

    # budget 4, 3 checkpoint opponents, SealBot on (the prompt's case).
    alloc = mse.allocate_budget(4, n_checkpoint_opponents=3, has_sealbot=True)
    assert alloc["per_checkpoint"] == 2
    assert alloc["per_checkpoint"] % 2 == 0  # stays an even pair count
    assert alloc[mse.SEALBOT_LABEL] >= 2

    # budget 4 with MANY opponents (the {sealbot:0, per:0} variant).
    alloc2 = mse.allocate_budget(4, n_checkpoint_opponents=10, has_sealbot=True)
    assert alloc2["per_checkpoint"] == 2
    assert alloc2[mse.SEALBOT_LABEL] >= 2

    # No SealBot: per_checkpoint still floored.
    assert mse.allocate_budget(4, n_checkpoint_opponents=3, has_sealbot=False)["per_checkpoint"] == 2

    # Zero budget -> all-zeros (the early-return is preserved).
    assert mse.allocate_budget(0, n_checkpoint_opponents=4, has_sealbot=True) == {
        mse.SEALBOT_LABEL: 0, "per_checkpoint": 0,
    }

    # Production budgets are UNCHANGED by the floor.
    assert mse.allocate_budget(128, n_checkpoint_opponents=4, has_sealbot=True)["per_checkpoint"] == 24
    assert mse.allocate_budget(120, n_checkpoint_opponents=3, has_sealbot=False)["per_checkpoint"] == 40


def test_small_budget_run_pins_anchor_and_produces_primary(tmp_path: Path) -> None:
    """End-to-end of the budget floor + anchor pin together: a tiny games_budget no
    longer degrades to no-anchor — the champion is played, the pool anchors, and a
    primary hypothesis block is produced."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    rep = _run(run, 40, _StubArena(per_score=2, sealbot_winrate=0.6), config=_no_sprt(games_budget=4))

    assert rep["ratings"]["fit"]["converged"] is True
    assert rep["verdict"]["primary"] is not None
    assert rep["verdict"]["primary"]["champion"] == "ep20"  # lag-5 reference of ep40


# =========================================================================== #
# (5) L-1 forced-opening: a pair's two games share the RECORDED opening LINE
#     even when the two nets would pick DIFFERENT opening moves.
# =========================================================================== #
class _AsymmEvaluator:
    """Carries a net tag so the asymmetric session can pick a tag-dependent move."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


class _AsymmSession:
    """A fake multi-root session whose OPENING move depends on the EVALUATOR (the
    moving net), not just the seed/position.

    This is the discriminating contract the symmetric stub cannot provide: at
    temperature > 0 net 'A' and net 'B' pick DIFFERENT opening actions from the
    SAME seed/position. So if the forced-opening REPLAY were absent, a pair's
    follower (whose ply-0 mover is net B after the seat swap) would diverge from
    its leader (ply-0 mover net A). The test asserts they do NOT diverge -> the
    follower replayed the leader's recorded line. The greedy tail (temp 0) is a
    pure function of position (seed/tag-independent) like real argmax search.
    """

    def __init__(self) -> None:
        self.discarded: list[int] = []

    @staticmethod
    def _move_for(tag: str, seed: int, temperature: float, ply: int) -> int:
        if temperature > 0.0:
            # Tag-dependent: 'A' and 'B' diverge at the same (seed, ply).
            base = 0 if tag == "A" else 3
            mix = (base + seed + ply) % 5
        else:
            mix = ply % 5  # greedy: position-only, tag/seed-independent
        q = (mix % 5) - 2
        r = ((mix // 5) % 5) - 2
        return pack_action_id(q, r)

    def search(self, game_keys, states, *, seed, evaluator, move_temperatures, **kw):
        return [
            {"action_id": self._move_for(evaluator.tag, seed, temp, st.ply)}
            for st, temp in zip(states, move_temperatures)
        ]

    def discard(self, index: int) -> None:
        self.discarded.append(index)


def _drive_asymmetric_pair(monkeypatch, *, n_games, opening_plies, game_len=8, batch_openings=False):
    """Run the REAL play_checkpoint_match against the fake engine + asymmetric
    session. Returns the result dict."""

    # Reuse the concurrent-test fake engine (Connect6 schedule, seedless).
    import test_hexfield_eval_arena_concurrent as conc

    decided_seat = {"v": 0}

    def decide(state):
        return decided_seat["v"]  # deterministic winner so games complete

    monkeypatch.setattr(eval_arena, "api", conc._FakeApi(game_len=game_len, decide_winner=decide))

    def factory():
        return _AsymmSession()

    def build_evals():
        return _AsymmEvaluator("A"), _AsymmEvaluator("B")

    return eval_arena.play_checkpoint_match(
        "a.pt", "b.pt", n_games,
        config=parse_hexfield_config({}),
        label_a="cand", label_b="opp",
        paired_openings=True,
        opening_plies=opening_plies,
        batch_openings=batch_openings,
        make_session=factory,
        build_evaluators=build_evals,
    )


def test_forced_opening_replay_shares_line_under_asymmetric_nets(monkeypatch) -> None:
    """L-1: with nets that DISAGREE on the opening move, a pair's two seat-swapped
    games STILL produce the identical opening prefix — proving the follower
    REPLAYS the leader's recorded actions rather than searching its own line
    (which a mere shared-RNG CRN would NOT guarantee, since the seat swap puts a
    different net to move at ply 0)."""

    opening_plies = 4
    result = _drive_asymmetric_pair(monkeypatch, n_games=8, opening_plies=opening_plies)

    games = {g["index"]: g for g in result["games"]}
    pairs = result["pentanomial"]["pairs"]
    assert pairs, "no pairs produced"
    for p in pairs:
        i0, i1 = p["game_indices"]
        op_leader = games[i0]["opening"][:opening_plies]
        op_follower = games[i1]["opening"][:opening_plies]
        assert op_leader, "leader recorded no opening"
        assert op_follower == op_leader, (
            f"pair {p['pair_index']}: follower opening {op_follower} != "
            f"leader {op_leader} (forced-opening replay failed)"
        )


def test_forced_opening_replay_actually_diverges_without_replay(monkeypatch) -> None:
    """Control: confirm the asymmetric stub WOULD diverge the two seats absent the
    replay — i.e. net A and net B pick different ply-0 opening moves from the same
    seed — so the previous test's agreement is meaningful (it can only come from
    the replay, not from a seat-symmetric stub)."""

    s = _AsymmSession()
    seed = 12345
    a_move = s._move_for("A", seed, 1.0, 0)
    b_move = s._move_for("B", seed, 1.0, 0)
    assert a_move != b_move, "asymmetric stub is not actually asymmetric"


def test_forced_opening_falls_back_to_search_when_leader_ended_early(monkeypatch) -> None:
    """Robustness of the replay: if the leader's game ends DURING its opening (so
    it recorded fewer than ``opening_plies`` actions), the follower falls back to a
    normal single-root search for the remaining opening plies rather than hanging
    or crashing. We force a very short game so the leader cannot fill the full
    opening, and assert the run still completes for every game."""

    # game_len shorter than opening_plies -> the leader ends mid-opening.
    result = _drive_asymmetric_pair(monkeypatch, n_games=4, opening_plies=8, game_len=3)
    # All games reached a terminal (none hung / aborted); the follower still moved.
    statuses = {g["status"] for g in result["games"]}
    assert statuses == {"completed"}, statuses
    assert result["pentanomial"]["n_pairs"] == 2


# =========================================================================== #
# (6) L-2: the verdict target is the STABLE reference (lag), not the prior epoch.
# =========================================================================== #
def test_verdict_target_is_stable_reference_on_contiguous_ladder(tmp_path: Path) -> None:
    """On a CONTIGUOUS ladder the verdict target (champion) is a STABLE,
    de-correlated reference >= ``verdict_reference_lag`` epochs below the candidate
    — NOT the immediately-prior, maximally-correlated checkpoint."""

    run = _make_run(tmp_path, epochs=(10, 11, 12, 13, 14, 15))
    cfg = MultiStageEvalSection(verdict_reference_lag=5)
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000015.pt", cfg, candidate_epoch=15
    )
    assert roster.champion is not None
    assert roster.champion.label == "ep10"  # 15 - 5, not ep14
    assert roster.champion.epoch == 10

    # lag=0 restores the legacy immediately-prior target (ep14).
    cfg0 = MultiStageEvalSection(verdict_reference_lag=0)
    roster0 = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000015.pt", cfg0, candidate_epoch=15
    )
    assert roster0.champion is not None and roster0.champion.label == "ep14"


def test_verdict_target_falls_back_when_no_old_enough_prior(tmp_path: Path) -> None:
    """L-2 fallback: when nothing is >= lag epochs below the candidate (first few
    epochs), the target falls back to the nearest prior so a hypothesis still
    exists rather than vanishing."""

    run = _make_run(tmp_path, epochs=(1, 2, 3))
    cfg = MultiStageEvalSection(verdict_reference_lag=5)  # nothing 5 below ep3
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000003.pt", cfg, candidate_epoch=3
    )
    assert roster.champion is not None and roster.champion.label == "ep2"


def test_immediately_prior_still_pooled_as_descriptive_edge(tmp_path: Path) -> None:
    """The immediately-prior checkpoint is NOT discarded — it still appears as a
    descriptive (bracket) opponent so its information is pooled into the BT fit;
    only the reported VERDICT target rests on the stable reference."""

    # A contiguous ladder so the bracket rungs include epochs ABOVE the lag-5
    # reference, e.g. the immediately-prior epoch.
    run = _make_run(tmp_path, epochs=(10, 11, 12, 13, 14, 15))
    cfg = _no_sprt(verdict_reference_lag=5)
    # bracket rungs come from the log_grid (5,10,20,...) strictly below 15 -> only
    # ep10 from the grid exists, but the champion stable reference is also ep10.
    # Use the explicit log_grid so a near-candidate rung is in play.
    cfg = _no_sprt(
        verdict_reference_lag=5,
        opponents={"log_grid": (10, 12, 14), "bracket_size": 2, "sealbot_enabled": False},
    )
    arena = _StubArena(per_score=2)
    rep = _run(run, 15, arena, config=cfg)

    # Verdict target is the STABLE reference (highest epoch <= 15-5=10) -> ep10.
    assert rep["verdict"]["primary"] is not None
    assert rep["verdict"]["primary"]["champion"] == "ep10"
    # The near-candidate bracket rungs (ep12, ep14) are STILL played + pooled as
    # descriptive edges even though they are not the verdict target.
    played = {label for label, _ in arena.ckpt_calls}
    assert {"ep12", "ep14"} <= played
    pooled = {e["opponent"] for e in rep["edges"]}
    assert {"ep12", "ep14"} <= pooled


# =========================================================================== #
# (7) M-1: Stage B SPRT triage labels agree with the gross-regression config.
# =========================================================================== #
def test_sprt_config_defaults_are_gross_regression_framed() -> None:
    """The config defaults frame H0=fine (elo0=0) vs H1=grossly-regressed
    (elo1 negative). This is the half of M-1 that lives in config."""

    sprt = MultiStageEvalSection().sprt
    assert sprt.elo0 == 0.0
    assert sprt.elo1 < 0.0  # gross-regression alternative (default -50)


def test_sprt_triage_maps_accept_h1_to_regress_and_h0_to_ok(tmp_path: Path, monkeypatch) -> None:
    """The label MAPPING agrees with the hypotheses: accept_h1 (record favours the
    gross-regression alternative) -> 'regress_suspected'; accept_h0 (record fine)
    -> 'ok'; continue -> 'escalate'. We drive each by stubbing eval_stats.sprt's
    mechanical verdict so the test is independent of the SPRT arithmetic."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _StubArena(per_score=1)  # champion match content is irrelevant here
    cfg = parse_hexfield_config({"multi_stage_eval": {"sprt": {"enabled": True, "max_games": 8}}})

    from hexfield import eval_stats

    cases = {
        "accept_h1": "regress_suspected",
        "accept_h0": "ok",
        "continue": "escalate",
    }
    for sprt_verdict, expected_label in cases.items():
        def fake_sprt(wins, losses, **kw):
            return eval_stats.SPRTResult(llr=0.0, lower=-1.0, upper=1.0, verdict=sprt_verdict)

        monkeypatch.setattr(mse.eval_stats, "sprt", fake_sprt)
        rep = _run(run, 40, arena, config=cfg)
        stage_b = next(s for s in rep["stages"] if s["stage"] == "B_sprt")
        assert stage_b["sprt_verdict"] == sprt_verdict
        assert stage_b["triage"] == expected_label, (sprt_verdict, stage_b["triage"])


def test_sprt_never_short_circuits_deep_eval(tmp_path: Path, monkeypatch) -> None:
    """PURE-EVAL framing of M-1: even when the screen flags a suspected regression,
    Stage C/D still run and produce the authoritative verdict (the screen is
    advisory, never a gate)."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = parse_hexfield_config({"multi_stage_eval": {"sprt": {"enabled": True, "max_games": 8}}})
    from hexfield import eval_stats

    def fake_sprt(wins, losses, **kw):
        return eval_stats.SPRTResult(llr=99.0, lower=-1.0, upper=1.0, verdict="accept_h1")

    monkeypatch.setattr(mse.eval_stats, "sprt", fake_sprt)
    rep = _run(run, 40, _StubArena(per_score=2), config=cfg)
    stage_b = next(s for s in rep["stages"] if s["stage"] == "B_sprt")
    assert stage_b["triage"] == "regress_suspected"
    # Deep eval + pool still ran and produced the authoritative verdict.
    assert any(s["stage"] == "C_deep" and s["status"] == "completed" for s in rep["stages"])
    assert any(s["stage"] == "D_pool" and s["status"] == "completed" for s in rep["stages"])
    assert rep["verdict"]["label"] in {"PROMOTE", "REGRESS", "INCONCLUSIVE"}


# =========================================================================== #
# (8) PURE-EVAL invariant still holds.
# =========================================================================== #
def test_pure_eval_knobs_default_off_and_reported(tmp_path: Path) -> None:
    """Gating/promotion default OFF and the report advertises pure-eval; the
    verdict is a reported label wired to nothing that mutates a run."""

    cfg = MultiStageEvalSection()
    assert cfg.eval_gating_enabled is False
    assert cfg.eval_promotion_enabled is False

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    rep = _run(run, 40, _StubArena(per_score=2))
    assert rep["meta"]["pure_eval"] is True
    assert rep["meta"]["gating_enabled"] is False
    assert rep["meta"]["promotion_enabled"] is False


def test_pure_eval_tripwire_fires_when_a_knob_is_flipped() -> None:
    """The _assert_no_run_mutation tripwire fires the instant a gating/promotion
    knob is flipped on, BEFORE any game runs — so a future edit that starts
    consuming them is caught in tests before it can touch a live run."""

    import dataclasses

    gating = dataclasses.replace(MultiStageEvalSection(), eval_gating_enabled=True)
    with pytest.raises(AssertionError):
        mse._assert_no_run_mutation(gating)

    promo = dataclasses.replace(MultiStageEvalSection(), eval_promotion_enabled=True)
    with pytest.raises(AssertionError):
        mse._assert_no_run_mutation(promo)


def test_run_writes_nothing_under_run_tree_when_diagnostics_off(tmp_path: Path) -> None:
    """With write_diagnostics=False the run touches NO file under the run tree —
    no pool, no diagnostics, no checkpoint, no flag (the standalone eval must never
    mutate the live run's directory)."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    before = {p for p in run.rglob("*")}
    _run(run, 40, _StubArena(per_score=2))
    after = {p for p in run.rglob("*")}
    assert before == after, after - before
