"""Game-running layer for the hexfield multi-stage strength evaluation.

PURELY EVAL. Nothing in this module gates, promotes, halts, or mutates a
training run; it only plays games and returns structured result dicts. The
statistical verdict layer (SPRT screen, pentanomial/Wilson CIs, the rolling
Bradley-Terry pool) lives in sibling modules and consumes these results.

Two runners, both returning the same result-dict shape as the standalone arena
(scripts/_wf_h2h2_arena.py): ``meta`` / ``score`` (with ``by_seat``) /
``game_lengths`` / ``opening_dedup`` / per-game rows, augmented with the
pair-level ``pentanomial`` block the corrected design needs.

play_checkpoint_match(model_a_ckpt, model_b_ckpt, ...)
    Hexfield-vs-hexfield, CONCURRENT and PAIRED. Both players are hexfield nets,
    so each ROUND batches whichever side is to move through THAT net's
    ``HexfieldMctsSession.search`` multi-root call (cross-game leaf batching) —
    two batched forwards per round (one per net), the natural generalization of
    ``play_sealbot_match`` with the serial SealBot drain replaced by a second
    batched hexfield search. Games are still coupled into MATCHED PAIRS via
    common random numbers (CRN): the same sampled opening is played from BOTH
    seats, so a pair is a paired comparison of the two nets on one shared line.
    Concurrency is orthogonal to the pairing (the CRN contract is a per-game
    property of seat + seed, untouched by batching) — see the CRN note below.

play_sealbot_match(model_ckpt, ...)
    Hexfield-vs-SealBot. A near-mechanical port of the dense concurrent SealBot
    loop (dense_cnn_restnet/evaluation.py:327-391): every game where hexfield is
    to move is searched together in ONE ``HexfieldMctsSession.search`` multi-root
    call (cross-game leaf batching), and SealBot's moves are drained serially per
    game through the hexo_runner SealBot adapter. No CRN pairing here — SealBot's
    minimax depth varies under load, so its games are not a matched comparison
    (the corrected design uses SealBot only as the pinned zero-point downstream).

Concurrency vs pairing — the two are orthogonal:
  * ``play_sealbot_match`` is CONCURRENT (many games in flight, batched forwards)
    but UNPAIRED.
  * ``play_checkpoint_match`` is BOTH CONCURRENT and PAIRED: many seat-swapped
    CRN games are in flight at once, batched per round through each net's
    session.

CRN / shared-opening note (load-bearing): ``hexo_engine.api.new_game(seed=...)``
does NOT randomize the opening — the engine is deterministic and the first move
is the forced centre stone (api.py docstring). ALL opening diversity comes from
the MCTS temperature sampling at the root (the first ``opening_plies`` plies
sample the move from the visit distribution using the per-search ``seed``).
Therefore a "shared opening" between the two seats of a pair is produced by
giving both seat-orderings the SAME per-ply search RNG ``pair_seed * 5003 + ply``
(matching the serial ``_play_pair``). With symmetric search configs the two
games then explore the same opening line from opposite seats — a genuine
common-random-number pairing — and any score difference inside the pair is
attributable to the seat swap / net strength, not to opening luck.

CRN UNDER BATCHING (why the concurrent runner can batch EVERY ply — opening and
greedy alike — while still giving each pair a shared opening LINE): the lockstep
``search`` builds a tree by DETERMINISTIC PUCT (no RNG in leaf selection) and
its only randomness is (a) optional root Dirichlet noise — never used in eval —
and (b) the final move selection. At ``temperature == 0`` (the greedy tail)
move selection is a pure deterministic argmax / LCB-of-Q (search.rs
select_action_from_policy / select_action_with_lcb), so a batched multi-root
greedy search yields the bit-identical move per game regardless of the shared
batch seed — greedy plies batch freely. At ``temperature > 0`` (the
``opening_plies`` sampled prefix) the per-root selection RNG is
``seed.wrapping_add(root_index)`` (search.rs:748-749), i.e. each root in a
batched call samples from a DISTINCT stream keyed by its batch position.

The shared opening LINE within a pair is NOT obtained by giving the two seats one
RNG stream (the seat swap means a different net moves at ply 0, so a shared seed
alone would not share the line). It is obtained by FORCED-OPENING REPLAY (L-1):
each pair has a LEADER (game 0) and a seat-swapped FOLLOWER (game 1); only the
LEADER searches its opening, and the FOLLOWER REPLAYS the leader's recorded
opening actions ply-for-ply (no search). Because the pairing depends ONLY on
``follower.opening == leader.opening`` — never on the leader's line being any
particular RNG draw — and LEADERS ARE INDEPENDENT GAMES (distinct ``pair_seed``,
no cross-leader CRN), the leaders' opening searches are batched cross-game into
one multi-root ``search`` per round per net exactly like the greedy tail. Each
leader root in that batch is seeded ``open_seed.wrapping_add(root_index)``, which
gives every leader its own decorrelated sampling stream (the per-root offset that
USED to be a problem for shared-seed pairs is exactly what we now WANT, since it
keeps independent leaders from collapsing onto one opening). Followers replay
their leader's recorded line; the rare leader-ended-mid-opening case falls back
to a single-root follower search. So the opening — which used to run one
serial single-root search PER LEADER PER PLY and starved the GPU — now runs as a
handful of fat multi-root forwards, while the long greedy tail stays batched as
before, all reusing the existing ``search`` ABI with NO Rust change. The
load-bearing invariant is the PAIRING (follower replays leader), NOT
byte-equivalence to the old single-root opening line (a batched leader samples a
different specific line, which is fine). (A pair whose two games agree on the
winner-by-color is an "even" pentanomial pair; a split pair is the informative
one.)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from .config import (
    ML_AUTO_DISABLED_FLAG,
    build_divergence_overrides,
    parse_hexfield_config,
)
from .geometry import pack_action_id, unpack_action_id

# Surfaces 0-record eval .hxr writes loudly so a future regression that empties
# ``_Game.actions`` is machine-visible (see E1) instead of being silently
# swallowed by the best-effort writer.
_EVAL_LOG = logging.getLogger("hexfield.eval")

# torch / HexfieldEvaluator / HexfieldNet are imported LAZILY inside the
# checkpoint-loading paths so this module is IMPORTABLE on a CPU-only host
# WITHOUT torch (the concurrent loop can then be unit-tested through the
# ``build_evaluators`` / ``make_session`` seams with a numpy stub evaluator —
# no torch, no GPU). Only ``_load_hexfield_net`` and the non-stub evaluator
# branches touch them.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from .model import HexfieldNet

# The native MCTS session lives in the maturin-built extension. Import lazily so
# this module is IMPORTABLE on hosts without the .so (e.g. CPU-only test runners
# that inject a fake session via ``make_session``/``build_opponent``); a real
# session construction without the extension still raises a clear error.
try:
    from . import _rust
except ImportError:  # pragma: no cover - exercised only on hosts without the .so
    _rust = None

# Per-side MCTS search-seed offsets, mirroring evaluation.py's reference ladder
# and the standalone arena: the two seats draw from decorrelated RNG streams so
# (where CRN is NOT requested) their opening samples never mirror each other.
# In CRN/paired mode these are deliberately bypassed (both seat orderings share
# one seed) — see play_checkpoint_match.
_SIDE_SEED_OFFSET = {"a": 0, "b": 500_009_999}

# Default search/opening knobs for the deep eval, matching the production eval
# protocol (evaluation.evaluate_epoch + the standalone arena): greedy after a
# temperature-sampled opening, no Dirichlet noise (eval games are not training
# games). ``opening_plies`` is in PLIES (single stones), like _play_pair.
DEFAULT_OPENING_PLIES = 8
DEFAULT_OPENING_TEMPERATURE = 1.0

# Hexfield has NO draws (the engine always resolves a winner before max_plies in
# practice; a max_plies truncation is the only non-decisive outcome and is
# reported separately as a "truncated" game, never as a draw).


# --------------------------------------------------------------------------- #
# Shared helpers (result-dict construction; numpy-free, importable on CPU)
# --------------------------------------------------------------------------- #


def _percentile(xs: list[int], q: float) -> int | None:
    if not xs:
        return None
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _length_stats(xs: list[int]) -> dict[str, Any] | None:
    """Same shape as scripts/_wf_h2h2_arena.py length_stats."""
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": round(sum(xs) / len(xs), 1),
        "median": _percentile(xs, 0.5),
        "p90": _percentile(xs, 0.9),
        "min": min(xs),
        "max": max(xs),
    }


def _opening_dedup(openings: list[tuple[int, ...]]) -> dict[str, Any]:
    """Distinct-opening count + duplicate groups, arena shape. ``openings`` is
    one opening-prefix tuple per game (in game order)."""
    groups: dict[tuple[int, ...], list[int]] = {}
    for game_index, key in enumerate(openings):
        groups.setdefault(key, []).append(game_index)
    dup_groups = {str(v[0]): v for v in groups.values() if len(v) > 1}
    return {
        "n_games": len(openings),
        "distinct_openings": len(groups),
        "duplicate_groups": dup_groups,
    }


def _new_rust_session(max_states: int) -> Any:
    """Construct a native ``HexfieldMctsSession`` (the default session factory).

    Raises a clear error when the maturin-built extension is unavailable rather
    than the bare ``AttributeError`` a ``None`` module would give. Tests that run
    without the .so inject a fake session via ``make_session`` and never hit this.
    """
    if _rust is None:
        raise RuntimeError(
            "hexfield._rust (the MCTS extension) is unavailable; build the .so or "
            "inject a session factory (make_session=) for a CPU-only run"
        )
    return _rust.HexfieldMctsSession(max_states=max_states)


def _load_hexfield_net(checkpoint: str | Path) -> HexfieldNet:
    """Strict-load a hexfield checkpoint into a fresh HexfieldNet.

    Mirrors evaluation.evaluate_epoch's loader (``payload["model"]``, strict).
    Strict by design: a value-/moves-left-head mismatch must surface here, not
    silently keep a random head.
    """
    import torch  # lazy: keep the module importable on CPU hosts without torch

    from .model import HexfieldNet

    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"hexfield checkpoint is not a readable file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"hexfield checkpoint payload has no 'model' state: {path}")
    sd = payload["model"]
    # Build the opponent net at ITS OWN width, not the process-global CHANNELS, so a
    # narrower (or wider) anchor — e.g. a c=96 main_4/main_2 checkpoint evaluated by a
    # c=128 run — loads instead of shape-mismatching. The width is the trunk channel
    # dim, read off the learned `tokens` parameter (NUM_TOKENS, c); fall back to the
    # stem bias (c). None => default-width construction (==CHANNELS), so single-width
    # runs are byte-identical to before.
    ckpt_channels = _infer_checkpoint_channels(sd)
    model = HexfieldNet() if ckpt_channels is None else HexfieldNet(channels=ckpt_channels)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        # (a) Older v3 checkpoint with the SHARED relative-position ``bias_table``,
        # saved BEFORE the per-block-bias deploy (main_3 ep31). The current model
        # expects one ``bias_tables.{i}`` per attention block; expand the shared
        # table into per-block copies (bit-identical — exactly the per-block
        # migration) so ep5..ep30-era opponents stay loadable in the eval. Without
        # this, these v3-but-shared-bias checkpoints fall through to the legacy v2
        # branch below and FAIL (they carry cell_q/conv6-7/LayerScale keys v2 lacks),
        # which silently drops the entire multi-checkpoint match to SealBot-only.
        remapped = None
        if "bias_table" in sd and any(k.startswith("bias_tables.") for k in model.state_dict()):
            remapped = {k: v for k, v in sd.items() if k != "bias_table"}
            for i in range(len(model.bias_tables)):
                remapped[f"bias_tables.{i}"] = sd["bias_table"].clone()
        if remapped is not None:
            try:
                model.load_state_dict(remapped, strict=True)
                model.eval()
                return model
            except RuntimeError:
                pass
        # (b) Legacy (pre-v3) checkpoint: a different architecture (6 conv blocks,
        # shared aux reduction, no cell_q / ml_reduction / LayerScale). Load it
        # into the FROZEN eval-only v2 snapshot so radius-4-native anchors trained
        # before the v3 arch change (e.g. main_2 epoch_000045.pt) stay playable.
        # Still strict — a genuine corruption must surface, not keep random heads.
        from .legacy_model_v2 import HexfieldNet as HexfieldNetV2

        model = HexfieldNetV2() if ckpt_channels is None else HexfieldNetV2(channels=ckpt_channels)
        model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def _infer_checkpoint_channels(sd: dict) -> int | None:
    """Trunk channel width of a hexfield checkpoint, or None if undeterminable.

    The width is the second dim of the learned ``tokens`` parameter (NUM_TOKENS, c);
    the stem conv bias (c,) is the fallback. Returning None means "use the default
    (process-global CHANNELS)", so a same-width run is constructed exactly as before.
    """
    for key in ("tokens", "stem.bias"):
        t = sd.get(key)
        if t is not None and hasattr(t, "shape") and len(t.shape) >= 1:
            return int(t.shape[-1])
    return None


def _resolve_eval_overrides(
    sp: Any,
    *,
    diagnostics_dir: str | Path | None,
    divergence_overrides: dict | None,
) -> dict:
    """The §5.4 divergence overrides the arena searches with.

    Default: mirror self-play exactly, including the heal-gate auto-disable flag
    (``ml_auto_disabled.flag`` in the run's diagnostics dir) so the arena
    measures the same engine the run actually plays. An explicit
    ``divergence_overrides`` (e.g. a parity-mode A/B) wins outright.
    """
    if divergence_overrides is not None:
        return divergence_overrides
    disabled = False
    if diagnostics_dir is not None:
        disabled = (Path(diagnostics_dir) / ML_AUTO_DISABLED_FLAG).exists()
    return build_divergence_overrides(sp, disabled=disabled)


def _write_eval_hxr(
    games,
    diagnostics_dir,
    label_a,
    label_b,
    *,
    kind="checkpoint",
    stats: dict | None = None,
) -> str | None:
    """Write the eval games as a ``.hxr`` record so the dashboard can REPLAY them
    (the History screen's "evaluation" source scans ``<run>/evaluation/*.hxr``).

    Best-effort + FAIL-SOFT: any error is swallowed so recording can never break
    the eval. One file per match at ``<run>/evaluation/epoch_NNNNNN/<a>_vs_<b>.hxr``
    (``<run>`` is the parent of ``diagnostics_dir``). Players are seat-labelled
    (player0/player1); each game id encodes the matchup + which seat the candidate
    held (seats swap per CRN pair), so the viewer shows the real board + winner.
    Returns the written path (str) or None.

    E1 hardening: a 0-record write (every game had falsy ``.actions``) is no
    longer silent — it emits a LOUD WARNING and the write exception (if any) is
    logged instead of being blanket-swallowed. If ``stats`` is passed, it is
    populated with ``games_written`` / ``games_skipped`` so the caller can thread
    the count into match meta (machine-visible 0-record detection).
    """

    if stats is not None:
        stats["games_written"] = 0
        stats["games_skipped"] = 0
    if diagnostics_dir is None:
        return None
    try:
        import re
        from pathlib import Path as _P

        from hexo_runner.records import AbortRecord, HexoRecordFile, HexoRecordPlayer

        run_dir = _P(diagnostics_dir).parent
        m = re.search(r"(\d+)", str(label_a))
        ep = int(m.group(1)) if m else 0
        rec_dir = run_dir / "evaluation" / f"epoch_{ep:06d}"
        rec_dir.mkdir(parents=True, exist_ok=True)
        safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))
        path = rec_dir / f"{safe(label_a)}_vs_{safe(label_b)}.hxr"
        players = (
            HexoRecordPlayer("seat0", "player0", f"{label_a}/{label_b} · seat 0"),
            HexoRecordPlayer("seat1", "player1", f"{label_a}/{label_b} · seat 1"),
        )
        n = 0
        skipped = 0
        with HexoRecordFile.create(path, api.engine_metadata(), players) as rf:
            for g in games:
                if not getattr(g, "actions", None):
                    skipped += 1
                    continue
                cand_seat = "candP0" if g.a_is_p0 else "candP1"
                # serial play_checkpoint_match's _Game exposes ``.index``; the
                # concurrent play_multi_checkpoint_match's _Game exposes
                # ``.local_index`` (no ``.index``). The shared writer must accept
                # BOTH — referencing ``g.index`` directly raised AttributeError on
                # every concurrent eval game, aborting the write after the header
                # and leaving a 0-record .hxr (the empty-replay bug).
                g_index = getattr(g, "index", None)
                if g_index is None:
                    g_index = getattr(g, "local_index", 0)
                writer = rf.begin_game(
                    f"ep{ep}-{label_a}-vs-{label_b}-g{g_index}-{cand_seat}", seed=g.seed
                )
                for aid in g.actions:
                    q, r = unpack_action_id(int(aid))
                    writer.record_action(PlacementAction(AxialCoord(q=q, r=r)))
                if g.winner is None:
                    writer.finish_aborted(
                        AbortRecord(
                            stage="evaluation",
                            exception_type="MaxPliesReached",
                            message="hexfield eval game reached max plies",
                        )
                    )
                else:
                    seat_w = 0 if ((g.winner == "A") == g.a_is_p0) else 1
                    writer.finish_completed(f"player{seat_w}", g.plies)
                n += 1
        if stats is not None:
            stats["games_written"] = n
            stats["games_skipped"] = skipped
        total = len(games) if hasattr(games, "__len__") else (n + skipped)
        if n == 0 and total > 0:
            # A 0-record file is produced when EVERY game had falsy .actions
            # (the regression that emptied live eval .hxr). Make it LOUD so it is
            # never silently swallowed again.
            _EVAL_LOG.warning(
                "eval .hxr wrote 0 of %d games (all .actions empty) -> %s",
                total,
                path,
            )
        return str(path) if n else None
    except Exception as exc:  # recording is best-effort; never break the eval
        _EVAL_LOG.warning("eval .hxr write failed: %r", exc)
        return None


# --------------------------------------------------------------------------- #
# (1) Hexfield checkpoint vs hexfield checkpoint — PAIRED (CRN) games
# --------------------------------------------------------------------------- #


def play_checkpoint_match(
    model_a_ckpt: str | Path,
    model_b_ckpt: str | Path,
    n_games: int,
    *,
    config: Any = None,
    label_a: str = "A",
    label_b: str = "B",
    paired_openings: bool = True,
    visits: int | None = None,
    virtual_batch_size: int | None = None,
    opening_plies: int = DEFAULT_OPENING_PLIES,
    opening_temperature: float = DEFAULT_OPENING_TEMPERATURE,
    divergence_overrides_a: dict | None = None,
    divergence_overrides_b: dict | None = None,
    diagnostics_dir: str | Path | None = None,
    max_states: int = 65_536,
    game_seed_base: int = 0,
    max_wall_seconds: float = 0.0,
    active_root_limit: int | None = None,
    batch_openings: bool = False,
    build_evaluators: Callable[..., tuple[Any, Any]] | None = None,
    make_session: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Play model A vs model B CONCURRENTLY and return a structured pentanomial
    result (drop-in: identical public signature and result-dict shape as before).

    Both players are hexfield nets. The runner keeps TWO persistent sessions /
    evaluators (one per net, keyed by game index) and plays all games in lockstep
    ROUNDS. Each round advances every active game by one ply: the games where net
    A is to move are batched through net A's session in one (or, above
    ``active_root_limit``, a few chunked) multi-root ``search`` call, and the
    games where net B is to move likewise through net B's session. The net's move
    is read from the search result and applied to that game's engine state. This
    is the same cross-game leaf batching self-play uses, at FULL search visits —
    parallelism, not fewer sims, is the speedup.

    PAIRING (``paired_openings=True``, the corrected design's default): games are
    grouped into ``n_pairs = ceil(n_games / 2)`` matched pairs. Both games of a
    pair use the SAME CRN ``pair_seed`` (so the opening-temperature sampling is
    shared — see the module CRN note) but SWAP seats: game 0 plays A-as-player0,
    game 1 plays B-as-player0. Concurrency does not change this: every in-flight
    game carries its own ``pair_index`` / ``pair_seed`` / ``a_is_p0`` / running
    ply count, so the per-game CRN contract survives batching. The shared opening
    LINE within a pair comes from FORCED-OPENING REPLAY: only each pair's LEADER
    searches its opening and the seat-swapped FOLLOWER replays the leader's
    recorded actions (no search). Because leaders are independent games, ALL
    leaders' opening-ply searches are BATCHED cross-game into one multi-root call
    per round per net (each leader root seeded ``open_seed + root_index`` so the
    independent leaders decorrelate), and the greedy tail batches freely (it is
    RNG-free at temperature 0) — see the module "CRN under batching" note.
    Each pair yields one pentanomial outcome: how many of its two games net A won
    (0, 1, or 2), plus the seat pattern, so the downstream pair-level SE (N_pairs
    units) and the pentanomial→BT mapping have what they need.

    UNPAIRED (``paired_openings=False``): every game gets an independent seed and
    seats simply alternate by game index (the legacy independent-Bernoulli
    layout). Returned for completeness / debugging; the statistical layer should
    prefer the paired result.

    FULL SIMS: ``visits=None`` defaults to ``cfg.selfplay.search_visits`` (the
    full production budget, 512), NOT the reduced ``cfg.evaluation.eval_visits``;
    an explicit ``visits`` overrides. Parallelism makes full sims affordable.

    ``active_root_limit`` caps a single multi-root batch (defaults to
    ``cfg.selfplay.active_root_limit``); larger to-move groups are chunked.
    ``batch_openings`` (default False): the leaders' opening plies are ALWAYS
    batched cross-game now (the serial single-root opening was the throughput
    bottleneck); the only thing this flag changes is the FOLLOWER opening. Left
    False (the default for any paired-pentanomial measurement) followers REPLAY
    their leader's recorded opening so the pair shares the real opening line.
    Set True it drops the leader/follower split entirely and batches every game's
    opening with one decorrelated per-root seed (no replay) — an ad-hoc throughput
    knob that forgoes the within-pair shared opening, so leave it False unless you
    do not need the paired pentanomial.
    ``build_evaluators`` / ``make_session`` are CPU-test injection seams: given
    them, the runner skips checkpoint loading / GPU and uses the supplied
    (eval_a, eval_b) and session factory so the loop is unit-testable with a stub
    evaluator and no torch/CUDA. ``build_evaluators`` is called with no args and
    returns ``(eval_a, eval_b)``; ``make_session`` is called with no args and
    returns a fresh session.

    Returns a dict with ``meta`` / ``score`` / ``pentanomial`` / ``game_lengths``
    / ``opening_dedup`` / ``games`` (see module docstring). Win counts in
    ``score`` are net-A-centric ("a_wins"). ``pentanomial.pairs`` is the list the
    pair-level statistics consume; for unpaired runs ``pentanomial`` is ``None``.

    Does NOT execute on the GPU here under the live constraint; callers run it.
    """

    cfg = config if config is not None else parse_hexfield_config({})
    sp = cfg.selfplay
    # FULL sims by default: fall back to the production search budget (512), not
    # the reduced eval_visits (128). Parallelism makes the full budget affordable.
    eval_visits = int(visits) if visits is not None else int(sp.search_visits)
    # EVAL-ONLY vbs override (LOCKED 16 by the multistage eval). Threaded here so
    # the eval can use a different MCTS leaf-parallelism than self-play WITHOUT
    # touching SelfplayConfig.virtual_batch_size (=4). None -> self-play value.
    vbs = int(virtual_batch_size) if virtual_batch_size is not None else int(sp.virtual_batch_size)
    root_limit = int(active_root_limit) if active_root_limit is not None else int(sp.active_root_limit)
    new_session = make_session if make_session is not None else (
        lambda: _new_rust_session(max_states)
    )

    started = time.perf_counter()
    if build_evaluators is not None:
        # CPU-test seam: skip checkpoint loading + GPU; use the supplied pair.
        eval_a, eval_b = build_evaluators()
    else:
        from .inference import HexfieldEvaluator  # lazy: torch only on the GPU path

        model_a = _load_hexfield_net(model_a_ckpt)
        model_b = _load_hexfield_net(model_b_ckpt)
        eval_a = HexfieldEvaluator(model_a, device=cfg.device)
        eval_b = HexfieldEvaluator(model_b, device=cfg.device)

    # Symmetric divergence overrides by default (so the win rate is unbiased); an
    # explicit per-net override drives a search-change A/B. The override follows
    # the SEARCHING net: ``ov_a`` whenever net A is to move, ``ov_b`` whenever net
    # B is to move — independent of which engine seat each net holds in a given
    # game. This is exactly equivalent to the serial ``_play_pair``, which passed
    # ``ov_a`` as the seat-player0 override and swapped it with ``ov_b`` between a
    # pair's two seat orderings — i.e. net A always searched with ``ov_a``. Each
    # round's two batched searches are single-net, so each carries one net's ov.
    ov_a = _resolve_eval_overrides(
        sp, diagnostics_dir=diagnostics_dir, divergence_overrides=divergence_overrides_a
    )
    ov_b = (
        ov_a
        if divergence_overrides_b is None
        else _resolve_eval_overrides(
            sp, diagnostics_dir=diagnostics_dir, divergence_overrides=divergence_overrides_b
        )
    )

    # ----- Build the in-flight game set (seats + CRN seeds), same layout as the
    # serial version, so pairing/CRN is preserved; only the execution is batched.
    class _Game:
        __slots__ = (
            "index", "pair_index", "a_is_p0", "seed", "state",
            "plies", "done", "status", "winner", "opening", "actions",
            "is_leader", "leader",
        )

        def __init__(self, index: int, pair_index: int, a_is_p0: bool, seed: int) -> None:
            self.index = index
            self.pair_index = pair_index
            self.a_is_p0 = a_is_p0
            self.seed = seed  # the CRN seed (== _play_pair's per-game seed)
            self.state = api.new_game()
            self.plies = 0
            self.done = False
            self.status = "truncated"
            self.winner: str | None = None  # "A" | "B" | None (net-A-centric)
            self.opening: list[int] = []
            # FULL ordered move sequence (action_ids, engine move order) so the
            # game is replayable as a .hxr record on the dashboard. Distinct from
            # ``opening`` (only the first ``opening_plies`` actions).
            self.actions: list[int] = []
            # FORCED-OPENING CRN (L-1): each pair has a LEADER (game 0, the first
            # seat ordering created) and a FOLLOWER (game 1, the seat-swapped
            # sibling). The leader searches its opening normally; the follower
            # REPLAYS the leader's recorded opening actions ply-for-ply (no
            # search) so the pair shares the real opening LINE, not merely the RNG
            # stream. ``leader`` points the follower at its leader so it can read
            # ``leader.opening[ply]``; the leader's ``leader`` is itself (unused).
            self.is_leader = True
            self.leader: _Game = self

        # Engine seat (player0/player1) net A occupies in this game.
        @property
        def a_role(self) -> Any:
            return api.Player.PLAYER_0 if self.a_is_p0 else api.Player.PLAYER_1

        @property
        def a_role_label(self) -> str:
            return "player0" if self.a_is_p0 else "player1"

        def a_to_move(self) -> bool:
            return api.current_player(self.state) == self.a_role

    games: list[_Game] = []
    pair_members: dict[int, list[_Game]] = {}
    if paired_openings:
        n_pairs = (n_games + 1) // 2
        for pair_index in range(n_pairs):
            pair_seed = game_seed_base + pair_index  # shared CRN seed (both seats)
            idx0 = pair_index * 2
            g0 = _Game(idx0, pair_index, a_is_p0=True, seed=pair_seed)
            games.append(g0)
            pair_members.setdefault(pair_index, []).append(g0)
            if idx0 + 1 < n_games:  # odd n_games -> last pair is a singleton
                g1 = _Game(idx0 + 1, pair_index, a_is_p0=False, seed=pair_seed)
                # FORCED-OPENING CRN: g1 follows g0 — it replays g0's opening line.
                g1.is_leader = False
                g1.leader = g0
                games.append(g1)
                pair_members[pair_index].append(g1)
    else:
        for game_index in range(n_games):
            seed = game_seed_base + game_index + (
                _SIDE_SEED_OFFSET["b"] if game_index % 2 else _SIDE_SEED_OFFSET["a"]
            )
            games.append(_Game(game_index, -1, a_is_p0=(game_index % 2 == 0), seed=seed))

    # ----- Two persistent sessions, one per NET, keyed by game index. The serial
    # version built a fresh session per game per seat; with a persistent session
    # the Rust per-game-key tree store keeps trees from crossing games (and we
    # discard each game's tree at end), so cross-game tree reuse never happens —
    # the concurrent equivalent of the serial fresh-session-per-game guarantee.
    s_net_a = new_session()
    s_net_b = new_session()
    budget_hit = False
    rounds = 0
    forward_batches = 0
    mcts_search_elapsed = 0.0

    def _finalize(g: _Game) -> None:
        terminal = api.terminal(g.state)
        if terminal is not None:
            g.status = "completed"
            if terminal.winner is None:
                g.winner = None  # hexo has no draws; defensive
            else:
                won_label = str(terminal.winner)  # "player0" / "player1"
                g.winner = "A" if won_label == g.a_role_label else "B"
        elif budget_hit:
            g.status = "aborted_budget"
            g.winner = None
        else:
            g.status = "truncated"
            g.winner = None
        g.done = True
        s_net_a.discard(g.index)
        s_net_b.discard(g.index)

    def _settle(g: _Game) -> None:
        if api.terminal(g.state) is not None or g.plies >= sp.max_game_plies:
            _finalize(g)

    # Common search knobs shared by every batched/single-root call (mirrors the
    # eval protocol: greedy after a sampled opening, no Dirichlet noise, the §5.4
    # divergences as self-play runs them; per-net override + session below).
    common = dict(
        visits=eval_visits,
        c_puct=sp.c_puct,
        temperature=0.0,
        virtual_batch_size=vbs,
        active_root_limit=root_limit,
        widening_policy_mass=sp.widening_policy_mass,
        widening_max_children=sp.widening_max_children,
        widening_min_children=sp.widening_min_children,
        fpu_reduction=sp.fpu_reduction,
        tss_enabled=sp.tss_enabled,
        search_parity_mode=sp.search_parity_mode,
    )

    def _net_for(g: _Game) -> str:
        """Which NET is to move in ``g`` ('A' or 'B')."""
        return "A" if g.a_to_move() else "B"

    def _temp(g: _Game) -> float:
        """Opening temperature off the GLOBAL ply count (== _play_pair, which
        keys temperature on ``ply < opening_plies``), then greedy."""
        return opening_temperature if (g.plies < opening_plies and opening_temperature > 0.0) else 0.0

    def _apply_search(g: _Game, search: dict[str, Any]) -> None:
        q, r = unpack_action_id(int(search["action_id"]))
        api.apply_action(g.state, PlacementAction(AxialCoord(q=q, r=r)))
        g.plies += 1
        g.actions.append(int(search["action_id"]))
        if len(g.opening) < opening_plies:
            g.opening.append(int(search["action_id"]))
        _settle(g)

    def _replay_action(g: _Game, action_id: int) -> None:
        """Apply a PRE-DECIDED opening action to a follower game without any
        search (forced-opening CRN). Identical bookkeeping to ``_apply_search``
        so the follower's plies/opening/settle path matches a searched ply — only
        the move SOURCE differs (the leader's recorded action, not a fresh
        search)."""
        q, r = unpack_action_id(int(action_id))
        api.apply_action(g.state, PlacementAction(AxialCoord(q=q, r=r)))
        g.plies += 1
        g.actions.append(int(action_id))
        if len(g.opening) < opening_plies:
            g.opening.append(int(action_id))
        _settle(g)

    def _follower_opening_action(g: _Game) -> int | None:
        """The leader's recorded action for the follower ``g``'s current opening
        ply, or ``None`` if the leader has no action for that ply yet (it should
        always have one — the round order guarantees the leader is strictly ahead
        of the follower at every follower move — but if the leader's game ended
        DURING its own opening it may have fewer than ``opening_plies`` recorded
        actions, in which case the follower falls back to a normal single-root
        CRN search for the remaining opening plies)."""
        line = g.leader.opening
        return line[g.plies] if g.plies < len(line) else None

    def _run_batch(net: str, batch: list[_Game], seed: int) -> int:
        """One multi-root ``search`` for ``batch`` (all to-move for ``net``),
        chunked at ``root_limit``. Returns the number of plies applied."""
        nonlocal mcts_search_elapsed, forward_batches
        session = s_net_a if net == "A" else s_net_b
        evaluator = eval_a if net == "A" else eval_b
        # ov is applied to the mover by SEAT in _play_pair; here every game in the
        # batch has the SAME net to move, but that net sits at different seats
        # across games. ov_a/ov_b are symmetric by default, so the seat the net
        # occupies determines its override: net A uses ov_a (its self-play
        # override), net B uses ov_b. (For the default symmetric case ov_a is
        # ov_b, so this is moot; it matters only for an explicit A/B lever-off.)
        overrides = ov_a if net == "A" else ov_b
        applied = 0
        for start in range(0, len(batch), root_limit):
            chunk = batch[start : start + root_limit]
            move_temperatures = [_temp(g) for g in chunk]
            t0 = time.perf_counter()
            searches = session.search(
                [g.index for g in chunk],
                tuple(g.state for g in chunk),
                seed=seed,
                evaluator=evaluator,
                move_temperatures=move_temperatures,
                divergence_overrides=overrides,
                **common,
            )
            mcts_search_elapsed += time.perf_counter() - t0
            forward_batches += 1
            if len(searches) != len(chunk):
                raise RuntimeError(
                    f"hexfield checkpoint eval search returned {len(searches)} "
                    f"results for {len(chunk)} games"
                )
            for g, search in zip(chunk, searches):
                _apply_search(g, search)
                applied += 1
        return applied

    def _run_opening_batch(net: str, batch: list[_Game], seed: int) -> int:
        """One multi-root ``search`` for the OPENING-ply LEADERS to-move for
        ``net`` (chunked at ``root_limit``), replacing the old per-leader serial
        single-root loop. Every root carries ``opening_temperature`` so each
        leader SAMPLES its opening move, and the native per-root selection seed
        ``seed.wrapping_add(root_index)`` (search.rs:748-749) gives each
        independent leader its own decorrelated stream — there is NO cross-leader
        CRN to preserve (CRN is strictly within a pair, leader<->follower, and the
        follower does not search but replays). Returns the number of plies applied.

        Identical in structure to ``_run_batch`` except the temperatures are pinned
        to ``opening_temperature`` (documenting intent — these games are all at
        ``plies < opening_plies`` so ``_temp`` would return the same value) and the
        base seed is the per-(net, round) opening stream rather than the greedy
        ``batch_seed``. Each leader's sampled action is recorded into ``g.opening``
        by ``_apply_search`` exactly as before, so followers still find a line to
        replay."""
        nonlocal mcts_search_elapsed, forward_batches
        session = s_net_a if net == "A" else s_net_b
        evaluator = eval_a if net == "A" else eval_b
        overrides = ov_a if net == "A" else ov_b
        applied = 0
        for start in range(0, len(batch), root_limit):
            chunk = batch[start : start + root_limit]
            t0 = time.perf_counter()
            searches = session.search(
                [g.index for g in chunk],
                tuple(g.state for g in chunk),
                seed=seed,
                evaluator=evaluator,
                move_temperatures=[opening_temperature] * len(chunk),
                divergence_overrides=overrides,
                **common,
            )
            mcts_search_elapsed += time.perf_counter() - t0
            forward_batches += 1
            if len(searches) != len(chunk):
                raise RuntimeError(
                    f"hexfield checkpoint eval opening search returned {len(searches)} "
                    f"results for {len(chunk)} games"
                )
            for g, search in zip(chunk, searches):
                _apply_search(g, search)
                applied += 1
        return applied

    def _run_single(g: _Game, net: str) -> None:
        """Single-root ``search`` for one game with the serial RNG
        (``seed = g.seed * 5003 + g.ply``, per-root index 0). Now used ONLY as the
        FOLLOWER fallback when its leader ended its own game before recording an
        action for this opening ply (nothing to replay), so the follower still
        moves. The leaders' opening plies are batched cross-game via
        ``_run_opening_batch`` and the greedy tail via ``_run_batch``. The follower
        shares its leader's seed, so this fallback's RNG is ``pair_seed*5003+ply``
        — consistent with the rest of the pair's stream."""
        nonlocal mcts_search_elapsed, forward_batches
        session = s_net_a if net == "A" else s_net_b
        evaluator = eval_a if net == "A" else eval_b
        overrides = ov_a if net == "A" else ov_b
        t0 = time.perf_counter()
        searches = session.search(
            [g.index],
            (g.state,),
            seed=g.seed * 5003 + g.plies,
            evaluator=evaluator,
            move_temperatures=[_temp(g)],
            divergence_overrides=overrides,
            **common,
        )
        mcts_search_elapsed += time.perf_counter() - t0
        forward_batches += 1
        _apply_search(g, searches[0])

    # ----- Round loop: each round advances every active game by at least one ply
    # (a game whose seat-to-move flips after net A's batch is also played in net
    # B's recomputed to-move set the same round, so it can advance up to 2 plies).
    # Per round and per net we batch BOTH the OPENING LEADERS (one multi-root
    # forward) and the GREEDY to-move games (another multi-root forward); followers
    # replay their leader's recorded opening (no search). ``batch_openings``
    # collapses the leader/follower distinction (everything in ``greedy``). A round
    # that makes no progress is a bug -> raise rather than hang.
    #
    # FORCED-OPENING CRN (L-1): within the opening, a pair's LEADER searches; its
    # seat-swapped FOLLOWER does NOT search — it REPLAYS the leader's recorded
    # action for that ply, so both games traverse the IDENTICAL opening LINE (not
    # merely the same RNG stream — the seat swap means a different net would move at
    # ply 0, so a shared seed alone does NOT share the line). The leaders are
    # INDEPENDENT games (no cross-leader CRN), so all leaders to-move for a net are
    # searched together in ONE multi-root ``_run_opening_batch`` call (each leader
    # root decorrelated by the native per-root ``seed+index`` offset) instead of the
    # old serial single-root-per-leader loop that starved the GPU. The round order
    # (net A pass then net B pass) guarantees the leader is strictly ahead of the
    # follower at every follower move, so the action to replay is always already
    # recorded; the rare leader-ended-mid-opening case falls back to a follower
    # single-root search.
    while True:
        active = [g for g in games if not g.done]
        if not active:
            break
        if max_wall_seconds and (time.perf_counter() - started) > max_wall_seconds:
            budget_hit = True
            for g in active:
                _finalize(g)
            break
        rounds += 1
        plies_this_round = 0
        for net in ("A", "B"):
            to_move = [g for g in active if not g.done and _net_for(g) == net]
            if not to_move:
                continue
            # Opening plies: in PAIRED mode (and unless batch_openings) each game
            # is handled per the forced-opening CRN — leaders search (batched
            # cross-game), followers replay the leader's recorded line; everything
            # past the opening batches too.
            if paired_openings and not batch_openings:
                openers = [g for g in to_move if g.plies < opening_plies and g.is_leader]
                followers = [g for g in to_move if g.plies < opening_plies and not g.is_leader]
                greedy = [g for g in to_move if g.plies >= opening_plies]
            else:
                openers = []
                followers = []
                greedy = to_move
            if openers:
                # All these leaders are independent games (no cross-leader CRN), so
                # batch their opening-ply searches into ONE multi-root call. The
                # per-(net, round) base seed plus the native per-root ``seed+index``
                # offset gives each leader its own decorrelated sampling stream. The
                # opening base offsets (13M/19M) are distinct from the greedy
                # offsets (0/7M below) for BOTH nets, so an opening batch and a
                # greedy batch in the same round never share a base seed (a round
                # in the opening->greedy transition can run both).
                open_seed = (
                    game_seed_base + (13_000_003 if net == "A" else 19_000_003) + rounds * 1_000_003
                )
                plies_this_round += _run_opening_batch(net, openers, open_seed)
            for g in followers:
                # Replay the leader's recorded opening action; if the leader ended
                # its game before reaching this ply (no recorded action), fall back
                # to a normal single-root CRN search so the follower still moves.
                replay = _follower_opening_action(g)
                if replay is not None:
                    _replay_action(g, replay)
                else:
                    _run_single(g, net)
                plies_this_round += 1
            if greedy:
                # Per-round, per-net batch seed (greedy plies are temperature 0, so
                # this RNG only tie-breaks; the value is decorrelated per net/round
                # like the SealBot loop's ``search_seed + rounds * 1_000_003``).
                batch_seed = (
                    game_seed_base + (0 if net == "A" else 7_000_003) + rounds * 1_000_003
                )
                plies_this_round += _run_batch(net, greedy, batch_seed)
        if plies_this_round == 0:
            raise RuntimeError(
                "hexfield checkpoint eval made no progress in a round; aborting to avoid a hang"
            )

    # ----- Re-key the in-flight games to the result rows + per-pair rows (the
    # exact shapes the serial version emitted, so _build_match_result and every
    # downstream consumer are unchanged).
    game_rows = [
        {
            "index": g.index,
            "seed": g.seed,
            "a_seat": "P0" if g.a_is_p0 else "P1",
            "status": g.status,
            "winner": g.winner,
            "plies": g.plies,
            "opening": list(g.opening),
        }
        for g in games
    ]
    pairs: list[dict[str, Any]] = []
    if paired_openings:
        for pair_index in sorted(pair_members):
            members = pair_members[pair_index]
            decided = [g for g in members if g.status == "completed"]
            a_wins_in_pair = sum(1 for g in decided if g.winner == "A")
            pairs.append(
                {
                    "pair_index": pair_index,
                    "seed": game_seed_base + pair_index,
                    "game_indices": [g.index for g in members],
                    "n_games": len(members),
                    "n_decided": len(decided),
                    "a_wins": a_wins_in_pair,
                    "b_wins": len(decided) - a_wins_in_pair,
                    # Pentanomial class for a 2-game pair: 2/1/0 net-A wins among
                    # the 2 decided games. Singleton/partial pairs report their
                    # decided count so the consumer can weight them correctly.
                    "pentanomial_a_score": a_wins_in_pair,
                }
            )

    # Persist the games as a replayable .hxr (dashboard "evaluation" source).
    # Best-effort: _write_eval_hxr is fully fail-soft.
    _hxr_stats: dict[str, int] = {}
    hxr_path = _write_eval_hxr(games, diagnostics_dir, label_a, label_b, stats=_hxr_stats)

    result = _build_match_result(
        games=game_rows,
        pairs=pairs if paired_openings else None,
        label_a=label_a,
        label_b=label_b,
        meta_extra={
            "kind": "hexfield_vs_hexfield",
            "hxr_record": hxr_path,
            "hxr_games_written": _hxr_stats.get("games_written", 0),
            "ckpt_a": {"label": label_a, "path": str(model_a_ckpt)},
            "ckpt_b": {"label": label_b, "path": str(model_b_ckpt)},
            "games_requested": n_games,
            "visits": eval_visits,
            "virtual_batch_size": vbs,
            "device": cfg.device,
            "paired_openings": paired_openings,
            "opening_plies": opening_plies,
            "opening_temperature": opening_temperature,
            "game_seed_base": game_seed_base,
            "divergence_overrides_a": ov_a,
            "divergence_overrides_b": ov_b,
            "budget_hit": budget_hit,
            # Concurrency telemetry (additive — downstream consumers ignore these).
            "concurrent": True,
            "batch_openings": bool(batch_openings),
            "rounds": rounds,
            "forward_batches": forward_batches,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "mcts_search_elapsed_seconds": round(mcts_search_elapsed, 2),
        },
    )
    return result


# --------------------------------------------------------------------------- #
# (1b) CONCURRENT MULTI-OPPONENT checkpoint match — ONE candidate forward across
#      EVERY opponent's candidate-to-move games per round (the shared-candidate
#      batch is the speed win), each opponent searched in its OWN session.
# --------------------------------------------------------------------------- #


def play_multi_checkpoint_match(
    candidate_ckpt: str | Path,
    opponents: list[tuple[str, str | Path]],
    n_games_per_opponent: int,
    *,
    config: Any = None,
    candidate_label: str = "cand",
    visits: int | None = None,
    virtual_batch_size: int | None = None,
    opening_plies: int = DEFAULT_OPENING_PLIES,
    opening_temperature: float = DEFAULT_OPENING_TEMPERATURE,
    divergence_overrides_candidate: dict | None = None,
    divergence_overrides_opponent: dict | None = None,
    diagnostics_dir: str | Path | None = None,
    max_states: int = 65_536,
    game_seed_base: int = 0,
    max_wall_seconds: float = 0.0,
    active_root_limit: int | None = None,
    build_candidate_evaluator: Callable[..., Any] | None = None,
    build_opponent_evaluator: Callable[..., Any] | None = None,
    make_session: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Play the candidate (always net A) vs MANY checkpoint opponents in ONE
    batched concurrent pass and return ``{opponent_label: match_result_dict}``.

    Each opponent's ``match_result_dict`` is BYTE-FOR-BYTE the shape
    :func:`play_checkpoint_match` returns (``meta`` / ``score`` /
    ``pentanomial`` / ``game_lengths`` / ``opening_dedup`` / ``games``), so the
    existing downstream (``multistage_eval._checkpoint_edge_counts`` ->
    ``eval_stats.effective_counts`` -> ``BTEdge``) consumes it UNCHANGED.

    THE SPEED WIN — SHARED CANDIDATE FORWARD. The candidate is net A in EVERY
    game across EVERY opponent. The candidate keeps ONE persistent session and
    ONE evaluator; each round, the GREEDY candidate-to-move games across ALL
    opponents are gathered and searched in ONE multi-root candidate-session call
    (the candidate net runs a single fat forward instead of one-per-opponent).
    Each opponent keeps its OWN session+evaluator and searches only its own
    games. Wall-clock is then MAX over opponents, not SUM.

    EXACT EQUIVALENCE TO N SERIAL ``play_checkpoint_match`` CALLS (the safety
    net the equivalence test pins). Each opponent group is constructed IDENTICALLY
    to a standalone ``play_checkpoint_match(candidate, opponent_b, n_games,
    paired_openings=True, game_seed_base=game_seed_base, ...)`` — same CRN pairs,
    same per-pair ``pair_seed = game_seed_base + pair_index``, same
    ``a_is_p0`` seat pattern, same leader/follower forced-opening replay. The
    ONLY thing that changes is WHEN the candidate's searches fire:

      * GREEDY plies (temperature 0): the native per-root selection seed is
        ``seed.wrapping_add(root_index)`` but at temperature 0 move selection is a
        pure deterministic argmax / LCB-of-Q (search.rs select_search_action), so
        the chosen move is SEED- AND BATCH-POSITION-INDEPENDENT. Therefore
        merging every opponent's greedy candidate games into one multi-root call
        yields the bit-identical per-game move a serial run would — greedy plies
        batch FREELY across opponents. This is the bulk of plies (the long tail).

      * OPENING-LEADER plies (temperature > 0): the per-root seed
        ``open_seed.wrapping_add(root_index)`` DOES matter, so to stay
        bit-identical to the serial run each opponent's candidate opening leaders
        are searched in their OWN per-opponent multi-root call with that
        opponent's own ``open_seed`` (= ``game_seed_base + 13_000_003 +
        rounds*1_000_003``, the SAME stream the serial run uses) and per-GROUP
        root_index. (Followers replay their leader's recorded line — no search.)
        The opening is a handful of plies; the per-opponent split here costs
        almost nothing while keeping the equivalence exact.

    GAME-KEY NAMESPACING. ``HexfieldMctsSession.search`` keys trees by game_key in
    a HashMap. The candidate session holds trees for games from ALL opponents at
    once, so each game's candidate-side key is a GLOBAL ``opp_index * KEY_STRIDE +
    local_index`` (KEY_STRIDE >> any plausible per-opponent game count), and is
    ``discard``-ed at game end so candidate trees never collide or leak across
    opponent groups. Each opponent session uses the local per-group index (its own
    games only), discarded at game end exactly like ``play_checkpoint_match``.

    Also writes one ``.hxr`` eval-game record per opponent under
    ``<run>/evaluation/epoch_N/`` via the existing ``_write_eval_hxr`` helper.

    ``build_candidate_evaluator`` / ``build_opponent_evaluator`` / ``make_session``
    are CPU-test injection seams (no torch/CUDA). ``build_candidate_evaluator()``
    -> candidate evaluator; ``build_opponent_evaluator(label, ckpt_path)`` ->
    that opponent's evaluator; ``make_session()`` -> a fresh session.

    Does NOT execute on the GPU here under the live constraint; callers run it.
    """

    cfg = config if config is not None else parse_hexfield_config({})
    sp = cfg.selfplay
    eval_visits = int(visits) if visits is not None else int(sp.search_visits)
    vbs = int(virtual_batch_size) if virtual_batch_size is not None else int(sp.virtual_batch_size)
    root_limit = int(active_root_limit) if active_root_limit is not None else int(sp.active_root_limit)
    new_session = make_session if make_session is not None else (
        lambda: _new_rust_session(max_states)
    )

    started = time.perf_counter()

    # ----- Evaluators: ONE candidate (net A everywhere) + ONE per opponent. -----
    if build_candidate_evaluator is not None:
        cand_eval = build_candidate_evaluator()
    else:
        from .inference import HexfieldEvaluator  # lazy: torch only on the GPU path

        cand_eval = HexfieldEvaluator(_load_hexfield_net(candidate_ckpt), device=cfg.device)

    # ov follows the SEARCHING net, exactly as in play_checkpoint_match: the
    # candidate (net A) always searches with ``ov_cand`` (its self-play override);
    # an opponent searches with ``ov_opp``. Symmetric by default (unbiased winrate).
    ov_cand = _resolve_eval_overrides(
        sp, diagnostics_dir=diagnostics_dir, divergence_overrides=divergence_overrides_candidate
    )
    ov_opp = (
        ov_cand
        if divergence_overrides_opponent is None
        else _resolve_eval_overrides(
            sp, diagnostics_dir=diagnostics_dir, divergence_overrides=divergence_overrides_opponent
        )
    )

    common = dict(
        visits=eval_visits,
        c_puct=sp.c_puct,
        temperature=0.0,
        virtual_batch_size=vbs,
        active_root_limit=root_limit,
        widening_policy_mass=sp.widening_policy_mass,
        widening_max_children=sp.widening_max_children,
        widening_min_children=sp.widening_min_children,
        fpu_reduction=sp.fpu_reduction,
        tss_enabled=sp.tss_enabled,
        search_parity_mode=sp.search_parity_mode,
    )

    # Per-game state. Mirrors play_checkpoint_match._Game but tracks the OPPONENT
    # group + a GLOBAL candidate-session key so candidate trees never collide.
    class _Game:
        __slots__ = (
            "opp_index", "local_index", "cand_key", "pair_index", "a_is_p0",
            "seed", "state", "plies", "done", "status", "winner", "opening",
            "actions", "is_leader", "leader",
        )

        def __init__(self, opp_index: int, local_index: int, cand_key: int,
                     pair_index: int, a_is_p0: bool, seed: int) -> None:
            self.opp_index = opp_index
            self.local_index = local_index  # key in THIS opponent's session
            self.cand_key = cand_key        # GLOBAL key in the candidate session
            self.pair_index = pair_index
            self.a_is_p0 = a_is_p0
            self.seed = seed
            self.state = api.new_game()
            self.plies = 0
            self.done = False
            self.status = "truncated"
            self.winner: str | None = None  # "A" | "B" | None (candidate-centric)
            self.opening: list[int] = []
            self.actions: list[int] = []
            self.is_leader = True
            self.leader: _Game = self

        @property
        def a_role(self) -> Any:
            return api.Player.PLAYER_0 if self.a_is_p0 else api.Player.PLAYER_1

        @property
        def a_role_label(self) -> str:
            return "player0" if self.a_is_p0 else "player1"

        def a_to_move(self) -> bool:
            return api.current_player(self.state) == self.a_role

    # KEY_STRIDE namespaces candidate-session game keys per opponent so two
    # opponents' games never share a candidate tree (n_games_per_opponent is tiny
    # vs this stride).
    KEY_STRIDE = 1_000_000

    # One opponent group per (label, ckpt). Each group is the EXACT game layout a
    # standalone play_checkpoint_match would build for that opponent.
    class _Group:
        __slots__ = ("opp_index", "label", "ckpt", "session", "evaluator",
                     "games", "pair_members")

        def __init__(self, opp_index, label, ckpt, session, evaluator):
            self.opp_index = opp_index
            self.label = label
            self.ckpt = ckpt
            self.session = session
            self.evaluator = evaluator
            self.games: list[_Game] = []
            self.pair_members: dict[int, list[_Game]] = {}

    groups: list[_Group] = []
    cand_session = new_session()
    for opp_index, (label, ckpt) in enumerate(opponents):
        if build_opponent_evaluator is not None:
            opp_eval = build_opponent_evaluator(label, ckpt)
        else:
            from .inference import HexfieldEvaluator  # lazy: torch only on GPU path

            opp_eval = HexfieldEvaluator(_load_hexfield_net(ckpt), device=cfg.device)
        grp = _Group(opp_index, label, ckpt, new_session(), opp_eval)
        # Build CRN pairs identically to play_checkpoint_match (paired_openings).
        n_pairs = (n_games_per_opponent + 1) // 2
        base = opp_index * KEY_STRIDE
        for pair_index in range(n_pairs):
            pair_seed = game_seed_base + pair_index  # shared CRN seed (both seats)
            idx0 = pair_index * 2
            g0 = _Game(opp_index, idx0, base + idx0, pair_index, a_is_p0=True, seed=pair_seed)
            grp.games.append(g0)
            grp.pair_members.setdefault(pair_index, []).append(g0)
            if idx0 + 1 < n_games_per_opponent:
                g1 = _Game(opp_index, idx0 + 1, base + idx0 + 1, pair_index,
                           a_is_p0=False, seed=pair_seed)
                g1.is_leader = False
                g1.leader = g0
                grp.games.append(g1)
                grp.pair_members[pair_index].append(g1)
        groups.append(grp)

    all_games: list[_Game] = [g for grp in groups for g in grp.games]

    budget_hit = False
    rounds = 0
    forward_batches = 0
    cand_forward_batches = 0
    mcts_search_elapsed = 0.0

    def _opp_session(g: _Game) -> Any:
        return groups[g.opp_index].session

    def _finalize(g: _Game) -> None:
        terminal = api.terminal(g.state)
        if terminal is not None:
            g.status = "completed"
            if terminal.winner is None:
                g.winner = None
            else:
                won_label = str(terminal.winner)  # "player0" / "player1"
                g.winner = "A" if won_label == g.a_role_label else "B"
        elif budget_hit:
            g.status = "aborted_budget"
            g.winner = None
        else:
            g.status = "truncated"
            g.winner = None
        g.done = True
        cand_session.discard(g.cand_key)
        _opp_session(g).discard(g.local_index)

    def _settle(g: _Game) -> None:
        if api.terminal(g.state) is not None or g.plies >= sp.max_game_plies:
            _finalize(g)

    def _temp(g: _Game) -> float:
        return opening_temperature if (g.plies < opening_plies and opening_temperature > 0.0) else 0.0

    def _apply_search(g: _Game, search: dict[str, Any]) -> None:
        q, r = unpack_action_id(int(search["action_id"]))
        api.apply_action(g.state, PlacementAction(AxialCoord(q=q, r=r)))
        g.plies += 1
        g.actions.append(int(search["action_id"]))
        if len(g.opening) < opening_plies:
            g.opening.append(int(search["action_id"]))
        _settle(g)

    def _replay_action(g: _Game, action_id: int) -> None:
        q, r = unpack_action_id(int(action_id))
        api.apply_action(g.state, PlacementAction(AxialCoord(q=q, r=r)))
        g.plies += 1
        g.actions.append(int(action_id))
        if len(g.opening) < opening_plies:
            g.opening.append(int(action_id))
        _settle(g)

    def _follower_opening_action(g: _Game) -> int | None:
        line = g.leader.opening
        return line[g.plies] if g.plies < len(line) else None

    def _candidate_key(g: _Game) -> int:
        return g.cand_key

    def _run_candidate_greedy(batch: list[_Game], seed: int) -> int:
        """ONE shared candidate-session multi-root search over the GREEDY
        candidate-to-move games across ALL opponents (chunked at root_limit). At
        temperature 0 the move is a deterministic seed-independent argmax, so this
        cross-opponent merge is bit-identical to per-opponent serial searches."""
        nonlocal mcts_search_elapsed, forward_batches, cand_forward_batches
        applied = 0
        for start in range(0, len(batch), root_limit):
            chunk = batch[start : start + root_limit]
            t0 = time.perf_counter()
            searches = cand_session.search(
                [_candidate_key(g) for g in chunk],
                tuple(g.state for g in chunk),
                seed=seed,
                evaluator=cand_eval,
                move_temperatures=[0.0] * len(chunk),
                divergence_overrides=ov_cand,
                **common,
            )
            mcts_search_elapsed += time.perf_counter() - t0
            forward_batches += 1
            cand_forward_batches += 1
            if len(searches) != len(chunk):
                raise RuntimeError(
                    f"hexfield multi-checkpoint candidate greedy search returned "
                    f"{len(searches)} results for {len(chunk)} games"
                )
            for g, search in zip(chunk, searches):
                _apply_search(g, search)
                applied += 1
        return applied

    def _run_candidate_opening(grp: "_Group", openers: list[_Game], seed: int) -> int:
        """Per-OPPONENT candidate opening-leader batch. Uses the candidate session
        + candidate evaluator but THIS opponent's own ``open_seed`` and per-group
        root_index, so the native ``seed+root_index`` per-root sampling stream is
        bit-identical to a serial play_checkpoint_match for this opponent."""
        nonlocal mcts_search_elapsed, forward_batches, cand_forward_batches
        applied = 0
        for start in range(0, len(openers), root_limit):
            chunk = openers[start : start + root_limit]
            t0 = time.perf_counter()
            searches = cand_session.search(
                [_candidate_key(g) for g in chunk],
                tuple(g.state for g in chunk),
                seed=seed,
                evaluator=cand_eval,
                move_temperatures=[opening_temperature] * len(chunk),
                divergence_overrides=ov_cand,
                **common,
            )
            mcts_search_elapsed += time.perf_counter() - t0
            forward_batches += 1
            cand_forward_batches += 1
            if len(searches) != len(chunk):
                raise RuntimeError(
                    f"hexfield multi-checkpoint candidate opening search returned "
                    f"{len(searches)} results for {len(chunk)} games"
                )
            for g, search in zip(chunk, searches):
                _apply_search(g, search)
                applied += 1
        return applied

    def _run_opponent_batch(grp: "_Group", batch: list[_Game], seed: int,
                            *, temperature: float | None) -> int:
        """One multi-root search for the opponent (net B) to-move games in THIS
        opponent's session (chunked at root_limit). ``temperature`` None -> per-game
        greedy/opening temperature via ``_temp``; a float pins it (opening leaders).
        Bit-identical to play_checkpoint_match's per-net batch for this opponent."""
        nonlocal mcts_search_elapsed, forward_batches
        applied = 0
        for start in range(0, len(batch), root_limit):
            chunk = batch[start : start + root_limit]
            temps = (
                [temperature] * len(chunk)
                if temperature is not None
                else [_temp(g) for g in chunk]
            )
            t0 = time.perf_counter()
            searches = grp.session.search(
                [g.local_index for g in chunk],
                tuple(g.state for g in chunk),
                seed=seed,
                evaluator=grp.evaluator,
                move_temperatures=temps,
                divergence_overrides=ov_opp,
                **common,
            )
            mcts_search_elapsed += time.perf_counter() - t0
            forward_batches += 1
            if len(searches) != len(chunk):
                raise RuntimeError(
                    f"hexfield multi-checkpoint opponent search returned "
                    f"{len(searches)} results for {len(chunk)} games"
                )
            for g, search in zip(chunk, searches):
                _apply_search(g, search)
                applied += 1
        return applied

    def _run_single(g: _Game, net: str) -> None:
        """Single-root follower fallback (leader ended mid-opening). Uses the
        serial RNG ``g.seed * 5003 + g.plies`` and the right session per net —
        exactly play_checkpoint_match's _run_single."""
        nonlocal mcts_search_elapsed, forward_batches, cand_forward_batches
        if net == "A":
            session, evaluator, key, ov = cand_session, cand_eval, g.cand_key, ov_cand
        else:
            grp = groups[g.opp_index]
            session, evaluator, key, ov = grp.session, grp.evaluator, g.local_index, ov_opp
        t0 = time.perf_counter()
        searches = session.search(
            [key],
            (g.state,),
            seed=g.seed * 5003 + g.plies,
            evaluator=evaluator,
            move_temperatures=[_temp(g)],
            divergence_overrides=ov,
            **common,
        )
        mcts_search_elapsed += time.perf_counter() - t0
        forward_batches += 1
        if net == "A":
            cand_forward_batches += 1
        _apply_search(g, searches[0])

    # ----- Round loop. Per round: (1) CANDIDATE pass — gather every opponent's
    # candidate-to-move games; search the OPENING leaders per-opponent (own
    # open_seed) and the GREEDY games in ONE shared cross-opponent call; followers
    # replay. (2) OPPONENT pass — each opponent searches its own to-move games in
    # its own session (openers per-opponent open_seed; greedy in one batch). This
    # ordering (candidate first, then each opponent) matches play_checkpoint_match's
    # net-A-then-net-B ordering per opponent group, so the leader is always strictly
    # ahead of its follower when the follower replays.
    while True:
        active = [g for g in all_games if not g.done]
        if not active:
            break
        if max_wall_seconds and (time.perf_counter() - started) > max_wall_seconds:
            budget_hit = True
            for g in active:
                _finalize(g)
            break
        rounds += 1
        plies_this_round = 0

        # ---- (1) CANDIDATE pass (net A), shared forward for the greedy tail. ----
        cand_to_move = [g for g in active if not g.done and g.a_to_move()]
        cand_openers_by_opp: dict[int, list[_Game]] = {}
        cand_followers: list[_Game] = []
        cand_greedy: list[_Game] = []
        for g in cand_to_move:
            if g.plies < opening_plies and g.is_leader:
                cand_openers_by_opp.setdefault(g.opp_index, []).append(g)
            elif g.plies < opening_plies and not g.is_leader:
                cand_followers.append(g)
            else:
                cand_greedy.append(g)
        # Opening leaders: per-opponent with that opponent's own open_seed (net A
        # offset 13_000_003, == play_checkpoint_match), so each leader's per-root
        # seed (open_seed+root_index) is bit-identical to the serial run.
        for opp_index, openers in cand_openers_by_opp.items():
            open_seed = game_seed_base + 13_000_003 + rounds * 1_000_003
            plies_this_round += _run_candidate_opening(groups[opp_index], openers, open_seed)
        # Followers replay their leader's recorded opening line (no search).
        for g in cand_followers:
            replay = _follower_opening_action(g)
            if replay is not None:
                _replay_action(g, replay)
            else:
                _run_single(g, "A")
            plies_this_round += 1
        # Greedy: ONE shared candidate forward across ALL opponents (temp 0 ->
        # seed-independent argmax, so the cross-opponent merge is exact).
        if cand_greedy:
            cand_seed = game_seed_base + rounds * 1_000_003
            plies_this_round += _run_candidate_greedy(cand_greedy, cand_seed)

        # ---- (2) OPPONENT pass (net B), each in its own session. ----
        active2 = [g for g in all_games if not g.done]
        for grp in groups:
            to_move = [g for g in active2 if g.opp_index == grp.opp_index and not g.done and not g.a_to_move()]
            if not to_move:
                continue
            openers = [g for g in to_move if g.plies < opening_plies and g.is_leader]
            followers = [g for g in to_move if g.plies < opening_plies and not g.is_leader]
            greedy = [g for g in to_move if g.plies >= opening_plies]
            if openers:
                # Net B opening offset 19_000_003 == play_checkpoint_match.
                open_seed = game_seed_base + 19_000_003 + rounds * 1_000_003
                plies_this_round += _run_opponent_batch(
                    grp, openers, open_seed, temperature=opening_temperature
                )
            for g in followers:
                replay = _follower_opening_action(g)
                if replay is not None:
                    _replay_action(g, replay)
                else:
                    _run_single(g, "B")
                plies_this_round += 1
            if greedy:
                # Net B greedy offset 7_000_003 == play_checkpoint_match.
                batch_seed = game_seed_base + 7_000_003 + rounds * 1_000_003
                plies_this_round += _run_opponent_batch(grp, greedy, batch_seed, temperature=None)

        if plies_this_round == 0:
            raise RuntimeError(
                "hexfield multi-checkpoint eval made no progress in a round; "
                "aborting to avoid a hang"
            )

    # ----- Build ONE result dict PER opponent, in play_checkpoint_match's shape. -
    elapsed = round(time.perf_counter() - started, 2)
    results: dict[str, Any] = {}
    for grp in groups:
        game_rows = [
            {
                "index": g.local_index,
                "seed": g.seed,
                "a_seat": "P0" if g.a_is_p0 else "P1",
                "status": g.status,
                "winner": g.winner,
                "plies": g.plies,
                "opening": list(g.opening),
            }
            for g in grp.games
        ]
        pairs: list[dict[str, Any]] = []
        for pair_index in sorted(grp.pair_members):
            members = grp.pair_members[pair_index]
            decided = [g for g in members if g.status == "completed"]
            a_wins_in_pair = sum(1 for g in decided if g.winner == "A")
            pairs.append(
                {
                    "pair_index": pair_index,
                    "seed": game_seed_base + pair_index,
                    "game_indices": [g.local_index for g in members],
                    "n_games": len(members),
                    "n_decided": len(decided),
                    "a_wins": a_wins_in_pair,
                    "b_wins": len(decided) - a_wins_in_pair,
                    "pentanomial_a_score": a_wins_in_pair,
                }
            )
        _hxr_stats: dict[str, int] = {}
        hxr_path = _write_eval_hxr(
            grp.games, diagnostics_dir, candidate_label, grp.label, stats=_hxr_stats
        )
        results[grp.label] = _build_match_result(
            games=game_rows,
            pairs=pairs,
            label_a=candidate_label,
            label_b=grp.label,
            meta_extra={
                "kind": "hexfield_vs_hexfield",
                "hxr_record": hxr_path,
                "hxr_games_written": _hxr_stats.get("games_written", 0),
                "ckpt_a": {"label": candidate_label, "path": str(candidate_ckpt)},
                "ckpt_b": {"label": grp.label, "path": str(grp.ckpt)},
                "games_requested": n_games_per_opponent,
                "visits": eval_visits,
                "virtual_batch_size": vbs,
                "device": cfg.device,
                "paired_openings": True,
                "opening_plies": opening_plies,
                "opening_temperature": opening_temperature,
                "game_seed_base": game_seed_base,
                "divergence_overrides_a": ov_cand,
                "divergence_overrides_b": ov_opp,
                "budget_hit": budget_hit,
                # Concurrency telemetry (additive — downstream consumers ignore).
                "concurrent": True,
                "multi_opponent": True,
                "n_opponents": len(groups),
                "rounds": rounds,
                "forward_batches": forward_batches,
                "candidate_forward_batches": cand_forward_batches,
                "elapsed_seconds": elapsed,
                "mcts_search_elapsed_seconds": round(mcts_search_elapsed, 2),
            },
        )
    return results


# --------------------------------------------------------------------------- #
# (2) Hexfield checkpoint vs SealBot — concurrent, UNPAIRED
# --------------------------------------------------------------------------- #


def play_sealbot_match(
    model_ckpt: str | Path,
    n_games: int,
    *,
    config: Any = None,
    label: str = "hexfield",
    sealbot_variant: str = "best",
    sealbot_time_limit: float = 0.05,
    sealbot_path: str | Path | None = None,
    visits: int | None = None,
    virtual_batch_size: int | None = None,
    opening_plies: int = DEFAULT_OPENING_PLIES,
    opening_temperature: float = DEFAULT_OPENING_TEMPERATURE,
    divergence_overrides: dict | None = None,
    diagnostics_dir: str | Path | None = None,
    max_states: int = 65_536,
    game_seed_base: int = 0,
    active_root_limit: int | None = None,
    max_wall_seconds: float = 0.0,
    build_opponent: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Play hexfield vs SealBot concurrently and return a structured result.

    Mechanical port of the dense concurrent SealBot loop
    (dense_cnn_restnet/evaluation.py:327-391), adapted to the hexfield search
    ABI:
      * One persistent ``HexfieldMctsSession``; every game where hexfield is to
        move is searched together in ONE ``session.search([keys], (states,),
        ..., evaluator=...)`` multi-root call so the net batches leaves across
        all in-flight games. Search knobs mirror the eval protocol (greedy after
        a sampled opening; no Dirichlet noise; the §5.4 divergences as self-play
        runs them).
      * SealBot's turn is drained serially per game through the hexo_runner
        ``SealBotPlayer`` adapter (each game keeps its own isolated worker, like
        the dense loop), because the two SealBot variants cannot coexist in one
        process and its think time is a fixed wall, not the bottleneck.

    NO CRN pairing: SealBot's minimax depth varies under GPU/CPU load, so two
    SealBot games are not a matched comparison. The corrected design uses SealBot
    only as the pinned zero-point downstream; this runner just produces the
    hexfield-vs-SealBot edge (Wilson CI on the binomial win rate).

    Seats alternate by game index (even -> hexfield is player0). ``build_opponent``
    is an injection seam (tests fake the bot); default builds a real
    ``SealBotPlayer`` per game.

    Does NOT execute here under the live-GPU constraint; callers run it.
    """

    # Imported lazily so importing this module never requires the SealBot
    # checkout (the checkpoint runner above does not need it).
    from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer

    cfg = config if config is not None else parse_hexfield_config({})
    sp = cfg.selfplay
    # FULL sims by default, matching play_checkpoint_match (sp.search_visits, the
    # production budget) — NOT the reduced cfg.evaluation.eval_visits. The
    # orchestrator already threads the full value explicitly, but the default must
    # agree with the checkpoint runner so an ad-hoc caller that omits ``visits``
    # measures SealBot at the SAME strength as the checkpoint edges (a latent
    # foot-gun otherwise: the two runners would silently default to different
    # budgets). An explicit ``visits`` (e.g. cfg.eval_visits) still overrides.
    eval_visits = int(visits) if visits is not None else int(sp.search_visits)
    # EVAL-ONLY vbs override (LOCKED 16), same contract as play_checkpoint_match;
    # None -> self-play value. Does NOT touch SelfplayConfig.virtual_batch_size.
    vbs = int(virtual_batch_size) if virtual_batch_size is not None else int(sp.virtual_batch_size)
    root_limit = int(active_root_limit) if active_root_limit is not None else sp.active_root_limit

    overrides = _resolve_eval_overrides(
        sp, diagnostics_dir=diagnostics_dir, divergence_overrides=divergence_overrides
    )

    sealbot_config = SealBotConfig(
        path=sealbot_path,
        variant=sealbot_variant,
        time_limit=sealbot_time_limit,
    )
    sealbot_config.validate()  # raises SealBotUnavailableError if the bot is missing

    def _make_opponent() -> Any:
        if build_opponent is not None:
            return build_opponent(sealbot_config)
        return SealBotPlayer(sealbot_config, player_id=f"sealbot-{sealbot_variant}")

    from .inference import HexfieldEvaluator  # lazy: torch only on the GPU path

    started = time.perf_counter()
    model = _load_hexfield_net(model_ckpt)
    evaluator = HexfieldEvaluator(model, device=cfg.device)
    session = _new_rust_session(max_states)

    # One in-flight game. Mirrors dense_cnn_restnet/evaluation._EvalGame.
    class _Game:
        __slots__ = (
            "index", "seed", "hex_is_p0", "state", "opponent",
            "plies", "hex_decisions", "done", "status", "winner", "opening",
            "actions",
        )

        def __init__(self, index: int) -> None:
            self.index = index
            self.seed = game_seed_base + index
            self.hex_is_p0 = index % 2 == 0
            self.state = api.new_game(seed=self.seed)
            self.opponent = _make_opponent()
            self.plies = 0
            self.hex_decisions = 0
            self.done = False
            self.status = "truncated"
            self.winner: str | None = None  # "hex" | "sealbot" | None
            self.opening: list[int] = []
            # Full move stream (BOTH players) for the replayable .hxr record.
            self.actions: list[int] = []

        @property
        def hex_role(self) -> Any:
            return api.Player.PLAYER_0 if self.hex_is_p0 else api.Player.PLAYER_1

        @property
        def hex_role_label(self) -> str:
            return "player0" if self.hex_is_p0 else "player1"

        def hex_to_move(self) -> bool:
            return api.current_player(self.state) == self.hex_role

    games = [_Game(i) for i in range(n_games)]
    search_seed = game_seed_base + 7_000_003
    mcts_search_elapsed = 0.0
    opponent_elapsed = 0.0
    rounds = 0
    forward_batches = 0
    budget_hit = False

    def _finalize(game: _Game) -> None:
        terminal = api.terminal(game.state)
        if terminal is not None:
            game.status = "completed"
            if terminal.winner is None:
                game.winner = None
            else:
                won_label = str(terminal.winner)  # "player0" / "player1"
                game.winner = "hex" if won_label == game.hex_role_label else "sealbot"
        elif budget_hit:
            game.status = "aborted_budget"
            game.winner = None
        else:
            game.status = "truncated"
            game.winner = None
        game.done = True
        session.discard(game.index)

    def _settle(game: _Game) -> bool:
        if api.terminal(game.state) is not None or game.plies >= sp.max_game_plies:
            _finalize(game)
            return True
        return False

    try:
        while True:
            active = [g for g in games if not g.done]
            if not active:
                break
            if max_wall_seconds and (time.perf_counter() - started) > max_wall_seconds:
                budget_hit = True
                for g in active:
                    _finalize(g)
                break
            rounds += 1
            plies_this_round = 0

            # --- Batched hexfield ply across every game where hex is to move. ---
            hex_games = [g for g in active if g.hex_to_move()]
            if hex_games:
                # Cap the multi-root batch at the session's strict active-root
                # limit; if more games than the limit are simultaneously
                # hex-to-move, search them in chunks (the dense loop never hits
                # this because games_per_epoch < limit, but n_games here can be
                # large). Each chunk is one multi-root forward.
                for chunk_start in range(0, len(hex_games), root_limit):
                    chunk = hex_games[chunk_start : chunk_start + root_limit]
                    move_temperatures = [
                        opening_temperature
                        if (g.hex_decisions < opening_plies and opening_temperature > 0.0)
                        else 0.0
                        for g in chunk
                    ]
                    t0 = time.perf_counter()
                    searches = session.search(
                        [g.index for g in chunk],
                        tuple(g.state for g in chunk),
                        visits=eval_visits,
                        c_puct=sp.c_puct,
                        temperature=0.0,
                        seed=search_seed + rounds * 1_000_003,
                        evaluator=evaluator,
                        virtual_batch_size=vbs,
                        active_root_limit=root_limit,
                        widening_policy_mass=sp.widening_policy_mass,
                        widening_max_children=sp.widening_max_children,
                        widening_min_children=sp.widening_min_children,
                        fpu_reduction=sp.fpu_reduction,
                        tss_enabled=sp.tss_enabled,
                        search_parity_mode=sp.search_parity_mode,
                        move_temperatures=move_temperatures,
                        divergence_overrides=overrides,
                    )
                    mcts_search_elapsed += time.perf_counter() - t0
                    forward_batches += 1
                    if len(searches) != len(chunk):
                        raise RuntimeError(
                            f"hexfield SealBot eval search returned {len(searches)} "
                            f"results for {len(chunk)} games"
                        )
                    for g, search in zip(chunk, searches):
                        q, r = unpack_action_id(int(search["action_id"]))
                        api.apply_action(g.state, PlacementAction(AxialCoord(q=q, r=r)))
                        g.plies += 1
                        g.hex_decisions += 1
                        g.actions.append(int(search["action_id"]))
                        if len(g.opening) < opening_plies:
                            g.opening.append(int(search["action_id"]))
                        plies_this_round += 1
                        _settle(g)

            # --- SealBot turns, serially per game, fully drained per turn. ---
            for g in active:
                if g.done:
                    continue
                while not g.done and not g.hex_to_move():
                    t0 = time.perf_counter()
                    decision = g.opponent.decide(g.state)
                    opponent_elapsed += time.perf_counter() - t0
                    api.apply_action(g.state, decision.action)
                    g.plies += 1
                    coord = decision.action.coord
                    g.actions.append(pack_action_id(coord.q, coord.r))
                    if len(g.opening) < opening_plies:
                        g.opening.append(pack_action_id(coord.q, coord.r))
                    plies_this_round += 1
                    _settle(g)

            if plies_this_round == 0:
                raise RuntimeError(
                    "hexfield SealBot eval made no progress in a round; aborting to avoid a hang"
                )
    finally:
        for g in games:
            try:
                g.opponent.close()
            except Exception:
                pass

    # Persist the SealBot games as a replayable .hxr (dashboard "evaluation"
    # source), mirroring the checkpoint runners. Best-effort / fail-soft. The
    # writer is net-A-centric (expects .a_is_p0 and .winner in {"A","B",None});
    # adapt the SealBot _Game (hexfield IS the candidate = net A, winner is
    # "hex"/"sealbot") onto that shape without disturbing the result mapping below.
    from types import SimpleNamespace

    _hxr_stats: dict[str, int] = {}
    _hxr_games = [
        SimpleNamespace(
            actions=g.actions,
            a_is_p0=g.hex_is_p0,
            seed=g.seed,
            index=g.index,
            plies=g.plies,
            winner=("A" if g.winner == "hex" else ("B" if g.winner == "sealbot" else None)),
        )
        for g in games
    ]
    hxr_path = _write_eval_hxr(
        _hxr_games, diagnostics_dir, label, f"SealBot {sealbot_variant}", stats=_hxr_stats
    )

    # Re-key game rows to the hexfield-vs-X result shape (winner relative to the
    # FIRST label = hexfield). _build_match_result is net-A-centric, so map
    # hexfield -> "A", sealbot -> "B".
    game_rows = [
        {
            "index": g.index,
            "seed": g.seed,
            "a_seat": "P0" if g.hex_is_p0 else "P1",
            "status": g.status,
            "winner": (
                "A" if g.winner == "hex" else ("B" if g.winner == "sealbot" else None)
            ),
            "plies": g.plies,
            "opening": list(g.opening),
        }
        for g in games
    ]
    result = _build_match_result(
        games=game_rows,
        pairs=None,  # SealBot games are unpaired
        label_a=label,
        label_b=f"SealBot {sealbot_variant}",
        meta_extra={
            "kind": "hexfield_vs_sealbot",
            "ckpt": {"label": label, "path": str(model_ckpt)},
            "sealbot": {"variant": sealbot_variant, "time_limit": sealbot_time_limit},
            "games_requested": n_games,
            "visits": eval_visits,
            "virtual_batch_size": vbs,
            "device": cfg.device,
            "opening_plies": opening_plies,
            "opening_temperature": opening_temperature,
            "game_seed_base": game_seed_base,
            "divergence_overrides": overrides,
            "budget_hit": budget_hit,
            "rounds": rounds,
            "forward_batches": forward_batches,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "mcts_search_elapsed_seconds": round(mcts_search_elapsed, 2),
            "opponent_elapsed_seconds": round(opponent_elapsed, 2),
            "hxr_record": hxr_path,
            "hxr_games_written": _hxr_stats.get("games_written", 0),
        },
    )
    return result


# --------------------------------------------------------------------------- #
# Result-dict builder (shared by both runners; arena shape + pentanomial)
# --------------------------------------------------------------------------- #


def _build_match_result(
    *,
    games: list[dict[str, Any]],
    pairs: list[dict[str, Any]] | None,
    label_a: str,
    label_b: str,
    meta_extra: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the standalone-arena result dict (meta/score/by-seat/lengths/
    opening-dedup/games) plus a ``pentanomial`` block for paired matches.

    ``games`` rows are net-A-centric: ``winner`` in {"A", "B", None}. No draws in
    hexo, so ``winner is None`` means the game was truncated/aborted (not
    decided), and such games are EXCLUDED from win rates and CIs but reported in
    the status counts.
    """

    completed = [g for g in games if g["status"] == "completed"]
    a_wins = sum(1 for g in completed if g["winner"] == "A")
    b_wins = sum(1 for g in completed if g["winner"] == "B")
    decided = a_wins + b_wins
    lo, hi = _wilson_ci(a_wins, decided)

    p0_games = [g for g in completed if g["a_seat"] == "P0"]
    p1_games = [g for g in completed if g["a_seat"] == "P1"]

    def _seat_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "a_wins": sum(1 for g in rows if g["winner"] == "A"),
            "b_wins": sum(1 for g in rows if g["winner"] == "B"),
        }

    lengths_all = [g["plies"] for g in completed]
    openings = [tuple(g.get("opening") or ()) for g in games]

    score: dict[str, Any] = {
        "completed": len(completed),
        "truncated": sum(1 for g in games if g["status"] == "truncated"),
        "aborted_budget": sum(1 for g in games if g["status"] == "aborted_budget"),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "decided": decided,
        "a_winrate_decided": round(a_wins / decided, 4) if decided else None,
        # 95% Wilson on the binomial win rate. PER-GAME (unit = game). For PAIRED
        # matches this UNDERSTATES the SE because paired games are correlated;
        # the pair-level SE in the ``pentanomial`` block is the correct one — see
        # the corrected-design note. This CI is descriptive only.
        "a_winrate_ci95": [round(lo, 4), round(hi, 4)] if decided else None,
        "by_seat": {"A_as_P0": _seat_block(p0_games), "A_as_P1": _seat_block(p1_games)},
    }

    result: dict[str, Any] = {
        "meta": {"label_a": label_a, "label_b": label_b, **meta_extra},
        "score": score,
        "game_lengths": {
            "overall": _length_stats(lengths_all),
            "A_as_P0": _length_stats([g["plies"] for g in p0_games]),
            "A_as_P1": _length_stats([g["plies"] for g in p1_games]),
            "a_won": _length_stats([g["plies"] for g in completed if g["winner"] == "A"]),
            "b_won": _length_stats([g["plies"] for g in completed if g["winner"] == "B"]),
        },
        "opening_dedup": _opening_dedup(openings),
        "games": games,
    }
    result["pentanomial"] = _pentanomial_block(pairs) if pairs is not None else None
    return result


def _pentanomial_block(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair-level pentanomial summary + the pair-unit win-rate SE.

    The pentanomial counts (over 2-game pairs, by net-A score in {0, 1, 2}) and
    the pair-level standard error are the load-bearing corrected-design outputs:
    the BT/Wilson inference downstream MUST use N_pairs units, not N_games, or
    the CIs are anti-conservative (paired games are correlated, not independent
    Bernoulli). We surface the raw per-pair scores so the verdict layer can
    apply its own over-dispersion / sandwich correction.

    For a complete 2-game pair the per-pair net-A score ``s in {0, 1, 2}`` is
    the number of games net A won; the natural pair-level statistic is
    ``s / 2 in {0, 0.5, 1}`` (an "even" pair = 1 each = 0.5). We compute the
    mean and the standard error of that pair statistic across pairs (treating
    each pair as one i.i.d. draw — the correct unit). Singleton/partial pairs
    (only at an odd ``n_games`` tail) are reported but, having ``n_games < 2``,
    are excluded from the pentanomial 0/1/2 histogram (they cannot be classed),
    and their lone game is folded into the pair statistic at ``s/ n_decided``.
    """

    full_pairs = [p for p in pairs if p["n_games"] == 2 and p["n_decided"] == 2]
    pent = {0: 0, 1: 0, 2: 0}  # net-A wins among the pair's 2 decided games
    for p in full_pairs:
        pent[p["pentanomial_a_score"]] += 1

    # Pair statistic in [0, 1]: per-pair net-A win fraction over DECIDED games.
    # Pairs with zero decided games (both truncated) carry no information and are
    # dropped from the mean/SE.
    stats = [
        p["a_wins"] / p["n_decided"]
        for p in pairs
        if p["n_decided"] > 0
    ]
    n = len(stats)
    mean = sum(stats) / n if n else None
    if n > 1 and mean is not None:
        var = sum((x - mean) ** 2 for x in stats) / (n - 1)  # sample variance
        se = (var / n) ** 0.5  # SE of the mean, N_pairs units
    else:
        var = None
        se = None

    return {
        "n_pairs": len(pairs),
        "n_full_pairs": len(full_pairs),
        "n_informative_pairs": n,
        # Histogram over full (2-decided-game) pairs, keyed by net-A wins.
        "histogram_a_wins": {"0": pent[0], "1": pent[1], "2": pent[2]},
        "pair_winrate_mean": round(mean, 4) if mean is not None else None,
        # Pair-LEVEL SE (N_pairs units). This is the SE the corrected design
        # feeds the inference — NOT the per-game Wilson half-width in ``score``.
        "pair_winrate_se": round(se, 4) if se is not None else None,
        "pair_winrate_sample_variance": round(var, 6) if var is not None else None,
        "pairs": pairs,
    }


def _wilson_ci(wins: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval (verbatim from scripts/_wf_h2h2_arena.py).

    PER-UNIT: pass independent counts. For paired matches the unit is the PAIR,
    not the game — callers wanting a paired CI use the pair-level SE in the
    pentanomial block, not this function on per-game counts.
    """
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))
