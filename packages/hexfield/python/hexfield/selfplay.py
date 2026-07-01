"""Continuous self-play epoch driver: the Python side of the on_move protocol.

Per game the driver tracks placement history incrementally (ordinal
phase/player and window_scan hot/standing-win cells), records FULL-search
decisions as pending samples, applies the chosen action through hexo_engine,
and at game end finalizes targets (hard z, opp policy with fast-masking, STV,
moves_left) and writes one hexfield_compact_v1 shard. Truncated games
(max_game_plies reached, no engine winner) are also written: their
outcome-independent heads (policy, opp_policy) train normally while the
value/stvalue/cell_q/moves_left heads are masked via the truncated flag
(outcome_valid=0 column -> value_mask=0 at expand).
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from typing import Any

import numpy as np

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction
from hexo_runner.records import AbortRecord, HexoRecordFile, HexoRecordPlayer

from . import _rust
from .config import ML_AUTO_DISABLED_FLAG, build_divergence_overrides, parse_hexfield_config
from .engine_facts import player_int
from .features import record_phase, record_player, window_scan
from .geometry import pack_action_id, unpack_action_id
from .inference import HexfieldEvaluator
from .samples import (
    STV_HORIZONS,
    HexfieldSampleData,
    _policy_surprise_kl,
    finalize_game_samples,
)
from .shards import write_compact_shard


class _GameTape:
    __slots__ = ("key", "state", "records", "pending", "ply")

    def __init__(self, key: int):
        self.key = key
        self.state = api.new_game()
        self.records: list[tuple[int, int, int, int]] = []
        self.pending: list[tuple[int, HexfieldSampleData, float]] = []
        self.ply = 0


class ContinuousDriver:
    def __init__(self, *, epoch: int, games_target: int, max_plies: int, out_dir,
                 horizons=STV_HORIZONS, record_file=None, diag_dir=None, active_limit=0):
        self.epoch = epoch
        self.games_target = games_target
        self.max_plies = max_plies
        self.out_dir = out_dir
        self.horizons = horizons
        # .hxr game-record file for the epoch; None disables recording.
        # Set by generate_selfplay_epoch.
        self.record_file = record_file
        # Directory for the live progress file
        # <diag_dir>/hexfield.selfplay.live.json, written every LIVE_INTERVAL_S
        # while running. None disables the live file.
        self.diag_dir = diag_dir
        self.active_limit = int(active_limit)
        self._t0 = time.time()
        self._last_live = 0.0
        self.games: dict[int, _GameTape] = {}
        self.games_started = 0
        self.games_finished = 0
        self.games_truncated = 0
        self.rows_written = 0
        self.decisions = 0
        self.full_decisions = 0
        self.game_lengths: list[int] = []
        self.policy_entropies: list[float] = []
        self.root_values: list[float] = []
        self.next_key = epoch * 1_000_000
        # Background shard writer: the per-game finalize + .hxr record + zlib npz
        # write runs off the on_move callback thread.
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_errors: list[BaseException] = []
        self._writer_failed = threading.Event()
        self._writer_thread: threading.Thread | None = None

    LIVE_INTERVAL_S = 3.0

    def _write_live(self, status: str) -> None:
        """Write hexfield.selfplay.live.json with epoch progress and
        positions/second. Throttled to LIVE_INTERVAL_S while status=="running";
        other statuses always write. No-op when diag_dir is None."""

        if self.diag_dir is None:
            return
        now = time.time()
        if status == "running" and (now - self._last_live) < self.LIVE_INTERVAL_S:
            return
        self._last_live = now
        elapsed = max(now - self._t0, 1e-9)
        pps = self.decisions / elapsed
        payload = {
            "status": status,
            "epoch": self.epoch,
            "timestamp": now,
            "requested_games": self.games_target,
            "games_started": self.games_started,
            "completed_games": self.games_finished - self.games_truncated,
            "truncated_games": self.games_truncated,
            "games_finished": self.games_finished,
            "active_games": len(self.games),
            "active_limit": self.active_limit,
            "searched_positions": self.decisions,
            "elapsed_seconds": elapsed,
            "search_positions_per_second": pps,
            "positions_per_second": pps,
            "full_decisions": self.full_decisions,
            "scheduler": "continuous",
        }
        # Write failures here are swallowed so they cannot interrupt self-play.
        try:
            path = self.diag_dir / "hexfield.selfplay.live.json"
            tmp = self.diag_dir / "hexfield.selfplay.live.json.tmp"
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(path)  # atomic replace
        except Exception:
            pass

    def start_games(self, count: int) -> list[_GameTape]:
        tapes = []
        for _ in range(count):
            tape = _GameTape(self.next_key)
            self.next_key += 1
            self.games_started += 1
            self.games[tape.key] = tape
            tapes.append(tape)
        return tapes

    def __call__(self, game_key: int, payload: dict[str, Any]):
        tape = self.games[game_key]
        action_id = int(payload["action_id"])
        full = bool(payload["pcr_full"])
        init = bool(payload["policy_init"])
        self.decisions += 1
        self._write_live("running")

        current = record_player(tape.ply)
        if full and not init:
            self.full_decisions += 1
            ids = np.frombuffer(bytes(payload["visit_policy_action_ids_bytes"]), dtype=np.uint32)
            weights = np.frombuffer(bytes(payload["visit_policy_weights_bytes"]), dtype=np.float32)
            # Per-cell Q: one Q per recorded action, same set and order as the
            # visit policy. Feeds the cell_q head.
            qs = np.frombuffer(bytes(payload["visit_policy_q_bytes"]), dtype=np.float32)
            # Policy-surprise = KL(visit || root prior).
            prior_ids = np.frombuffer(
                bytes(payload["root_prior_policy_action_ids_bytes"]), dtype=np.uint32
            )
            prior_weights = np.frombuffer(
                bytes(payload["root_prior_policy_weights_bytes"]), dtype=np.float32
            )
            surprise = _policy_surprise_kl(ids, weights, prior_ids, prior_weights)
            # Improved-policy target π' and raw root logits are present only when
            # gumbel_target is enabled; otherwise the keys are absent from the
            # payload and these stay empty (visit policy is used as the target).
            gumbel_pairs: tuple[tuple[int, float], ...] = ()
            prior_logit_pairs: tuple[tuple[int, float], ...] = ()
            if "gumbel_policy_action_ids_bytes" in payload:
                g_ids = np.frombuffer(
                    bytes(payload["gumbel_policy_action_ids_bytes"]), dtype=np.uint32
                )
                g_weights = np.frombuffer(
                    bytes(payload["gumbel_policy_weights_bytes"]), dtype=np.float32
                )
                gumbel_pairs = tuple(
                    zip((int(a) for a in g_ids), (float(w) for w in g_weights))
                )
                if "root_prior_logits_bytes" in payload:
                    g_logits = np.frombuffer(
                        bytes(payload["root_prior_logits_bytes"]), dtype=np.float32
                    )
                    prior_logit_pairs = tuple(
                        zip((int(a) for a in g_ids), (float(l) for l in g_logits))
                    )
            phase = record_phase(tape.ply)
            first_stone = (
                (tape.records[-1][0], tape.records[-1][1]) if phase == "SecondStone" else None
            )
            own_hot, opp_hot, own_win, opp_win = window_scan(
                tuple(tape.records), current, len(tape.records)
            )
            sample = HexfieldSampleData(
                game_id=str(game_key),
                turn_index=tape.ply,
                current_player=current,
                phase=phase,
                records=tuple(tape.records),
                first_stone=first_stone,
                own_hot=own_hot,
                opp_hot=opp_hot,
                own_win=own_win,
                opp_win=opp_win,
                policy=tuple(zip((int(a) for a in ids), (float(w) for w in weights))),
                q_policy=tuple(zip((int(a) for a in ids), (float(q) for q in qs))),
                gumbel_policy=gumbel_pairs,
                prior_logit=prior_logit_pairs,
                policy_surprise=float(surprise),
                metadata={"pcr_full": True},
            )
            tape.pending.append((current, sample, float(payload["root_value"])))
            probs = weights[weights > 0]
            if probs.size:
                self.policy_entropies.append(float(-(probs * np.log(probs)).sum()))
            self.root_values.append(float(payload["root_value"]))
        elif not full and not init:
            # Fast rows are not written, but the pending list keeps every
            # decision so opp-policy lookup and moves_left counts remain complete
            # (mask_opp_from_fast at finalize).
            sample = HexfieldSampleData(
                game_id=str(game_key), turn_index=tape.ply, current_player=current,
                phase=record_phase(tape.ply), records=tuple(tape.records),
                first_stone=None, own_hot=(), opp_hot=(), own_win=(), opp_win=(),
                policy=(), metadata={"pcr_full": False},
            )
            tape.pending.append((current, sample, float(payload["root_value"])))
        else:
            sample = HexfieldSampleData(
                game_id=str(game_key), turn_index=tape.ply, current_player=current,
                phase=record_phase(tape.ply), records=tuple(tape.records),
                first_stone=None, own_hot=(), opp_hot=(), own_win=(), opp_win=(),
                policy=(), metadata={"pcr_full": False, "policy_init": True},
            )
            tape.pending.append((current, sample, float(payload["root_value"])))

        q, r = unpack_action_id(action_id)
        result = api.apply_action(tape.state, PlacementAction(AxialCoord(q=q, r=r)))
        tape.records.append((q, r, current, tape.ply + 1))
        tape.ply += 1

        if result.terminal:
            terminal = api.terminal(tape.state)
            self._finish(tape, winner=player_int(terminal.winner), truncated=False)
        elif tape.ply >= self.max_plies:
            self._finish(tape, winner=None, truncated=True)
        else:
            return ("advance", tape.state)

        del self.games[game_key]
        if self.games_started < self.games_target:
            fresh = self.start_games(1)[0]
            return ("replace", fresh.key, fresh.state)
        return None

    def _write_record(self, tape: _GameTape, *, winner, truncated: bool) -> None:
        """Write one ``.hxr`` game record. Every finished game is recorded
        (completed and truncated). Records the placement sequence in move order,
        then closes the game with the engine winner label (``player0``/``player1``)
        for completed games or an abort record for truncated games. No-op when
        record_file is None."""

        if self.record_file is None:
            return
        writer = self.record_file.begin_game(
            f"epoch-{self.epoch:06d}-game-{tape.key}", seed=tape.key
        )
        for q, r, _player, _ply in tape.records:
            writer.record_action(PlacementAction(AxialCoord(q=int(q), r=int(r))))
        if truncated:
            writer.finish_aborted(
                AbortRecord(
                    stage="selfplay",
                    exception_type="MaxPliesReached",
                    message=f"hexfield self-play reached max_plies={self.max_plies}",
                )
            )
        else:
            writer.finish_completed(f"player{int(winner)}", tape.ply)

    def _finish(self, tape: _GameTape, *, winner, truncated: bool) -> None:
        self.games_finished += 1
        self.game_lengths.append(tape.ply)
        if truncated:
            self.games_truncated += 1
        else:
            # winner is None only on the truncated path.
            assert winner is not None, "non-truncated finish requires an engine winner"
        # Surface a prior writer-thread failure before queueing more work.
        if self._writer_failed.is_set():
            raise self._writer_errors[0]
        # Hand the finished tape to the background writer. The tape is not mutated
        # after the game ends; __call__ deletes it from self.games after this
        # returns.
        self._write_queue.put((tape, winner, truncated))

    def _writer_loop(self) -> None:
        """Background shard writer. Drains finished games from _write_queue and
        does the I/O -- .hxr record, finalize, and the zlib `write_compact_shard`
        -- off the search-callback thread. A write failure is captured in
        _writer_errors and _writer_failed is set. Exits on a None sentinel."""

        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return
                tape, winner, truncated = item
                # Record the game (completed and truncated) before finalizing.
                self._write_record(tape, winner=winner, truncated=truncated)
                # Truncated games (max_game_plies hit, no engine winner) still
                # have rows written: the outcome-independent heads (policy,
                # opp_policy) train on them, while the value / stvalue / cell_q /
                # moves_left heads are masked downstream (truncated metadata flag
                # -> outcome_valid=0 shard column -> value_mask=0 + zeroed
                # stvalue/cell_q masks at expand).
                finalized = finalize_game_samples(
                    tape.pending, winner, self.horizons,
                    truncated=truncated, mask_opp_from_fast=True,
                )
                rows = [
                    s for s in finalized
                    if s.metadata.get("pcr_full", False)                      # all full rows (completed + truncated)
                    or (not truncated and not s.metadata.get("policy_init", False))  # fast rows from completed games; excludes init
                ]
                if rows:
                    path = self.out_dir / f"game_{tape.key}.npz"
                    self.rows_written += write_compact_shard(
                        path, rows, short_term_value_horizons=self.horizons,
                        sidecar={
                            "epoch": self.epoch, "game_key": tape.key,
                            "winner": winner, "truncated": bool(truncated),
                        },
                    )
            except BaseException as exc:  # noqa: BLE001
                self._writer_errors.append(exc)
                self._writer_failed.set()
            finally:
                self._write_queue.task_done()

    def _start_writer(self) -> None:
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="hexfield-selfplay-writer", daemon=True
        )
        self._writer_thread.start()

    def _stop_writer(self) -> None:
        """Enqueue the None sentinel, join the writer thread, then re-raise any
        writer error. No-op when no writer thread is running."""

        if self._writer_thread is None:
            return
        self._write_queue.put(None)
        self._writer_thread.join()
        self._writer_thread = None
        if self._writer_errors:
            raise self._writer_errors[0]

    def stats(self) -> dict[str, Any]:
        lengths = np.asarray(self.game_lengths or [0], dtype=np.float64)
        return {
            "games_started": self.games_started,
            "games_finished": self.games_finished,
            "truncated_games": self.games_truncated,
            "rows_written": self.rows_written,
            "total_decisions": self.decisions,
            "full_decisions": self.full_decisions,
            "mean_game_length": float(lengths.mean()),
            "p90_game_length": float(np.percentile(lengths, 90)),
            "root_policy_entropy_mean": float(np.mean(self.policy_entropies)) if self.policy_entropies else None,
            "root_value_mean": float(np.mean(self.root_values)) if self.root_values else None,
        }


def generate_selfplay_epoch(*, ctx, components, epoch: int, games_per_epoch: int) -> dict[str, Any]:
    cfg = parse_hexfield_config(ctx.config.model.config)
    sp = cfg.selfplay
    model = components.model.model
    evaluator = HexfieldEvaluator(model, device=cfg.device)

    out_dir = ctx.samples_dir / f"epoch_{epoch:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    games_target = max(int(games_per_epoch), 1)
    # Resume support: completed games already wrote their shards. On a restart,
    # keep the existing shards and generate only the remainder, using keys past
    # any the interrupted run assigned. In-flight (unfinished) games are not
    # recovered; the remainder replaces them with fresh games.
    existing = sorted(out_dir.glob("game_*.npz"))
    already_done = len(existing)
    remaining = max(games_target - already_done, 0)
    resuming = already_done > 0

    if remaining == 0:
        # All of the epoch's self-play is already on disk; skip regeneration.
        driver = ContinuousDriver(
            epoch=epoch, games_target=0, max_plies=sp.max_game_plies, out_dir=out_dir,
            diag_dir=ctx.diagnostics_dir, active_limit=0,
        )
        result = {
            "status": "completed", "epoch": epoch, "elapsed_seconds": 0.0,
            "search_visits": sp.search_visits, "scheduler": {},
            "resumed_existing_games": already_done, **driver.stats(),
        }
        diag_path = ctx.diagnostics_dir / f"hexfield.selfplay.epoch_{epoch:06d}.json"
        diag_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result

    slots = min(sp.active_games, remaining)
    driver = ContinuousDriver(
        epoch=epoch, games_target=remaining, max_plies=sp.max_game_plies, out_dir=out_dir,
        diag_dir=ctx.diagnostics_dir, active_limit=slots,
    )
    if resuming:
        existing_keys = []
        for p in existing:
            try:
                existing_keys.append(int(p.stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                pass
        driver.next_key = (max(existing_keys) + 1) if existing_keys else epoch * 1_000_000
    tapes = driver.start_games(slots)

    # Per-epoch .hxr game records under <run>/selfplay.
    record_dir = ctx.output_dir / "selfplay"
    record_dir.mkdir(parents=True, exist_ok=True)
    # On resume, write to a separate .hxr path (HexoRecordFile.create overwrites
    # an existing file). Fresh epochs use the canonical path.
    record_path = record_dir / (
        f"epoch_{epoch:06d}_resume{already_done:03d}.hxr" if resuming
        else f"epoch_{epoch:06d}.hxr"
    )
    players = (
        HexoRecordPlayer("hexfield-a", "player0", "Hexfield A"),
        HexoRecordPlayer("hexfield-b", "player1", "Hexfield B"),
    )

    session = _rust.HexfieldMctsSession(max_states=sp.cache_max_states)
    started = time.time()
    driver._t0 = started  # anchor live pos/s to self-play start
    driver._write_live("running")  # initial progress before the first move
    noise_kwargs = {}
    if sp.root_dirichlet_noise_fraction > 0:
        noise_kwargs = dict(
            root_dirichlet_total_alpha=sp.root_dirichlet_total_alpha,
            root_dirichlet_noise_fraction=sp.root_dirichlet_noise_fraction,
        )
    # Context-managed so the .hxr is finalized even if run_continuous raises.
    with HexoRecordFile.create(record_path, api.engine_metadata(), players) as record_file:
        driver.record_file = record_file
        driver._start_writer()
        scheduler_stats = session.run_continuous(
            [tape.key for tape in tapes],
            tuple(tape.state for tape in tapes),
            evaluator=evaluator,
            on_move=driver,
            visits=sp.search_visits,
            c_puct=sp.c_puct,
            base_seed=(ctx.config.run.seed or 1) * 1_000_003 + epoch,
            virtual_batch_size=sp.virtual_batch_size,
            flush_target=sp.flush_target,
            active_root_limit=sp.active_root_limit,
            temperature_by_ply=cfg.temperature_by_ply(),
            root_policy_temperature=sp.root_policy_temperature,
            root_policy_temperature_early=sp.root_policy_temperature_early or None,
            root_policy_temperature_halflife=sp.root_policy_temperature_halflife or None,
            fpu_reduction=sp.fpu_reduction,
            virtual_loss=sp.virtual_loss,
            widening_policy_mass=sp.widening_policy_mass,
            widening_max_children=sp.widening_max_children,
            widening_min_children=sp.widening_min_children,
            forced_playout_k=sp.forced_playout_k,
            pcr_full_proportion=sp.pcr_full_proportion,
            pcr_fast_visits=sp.pcr_fast_visits,
            policy_init_fraction=sp.policy_init_fraction,
            policy_init_avg_plies=sp.policy_init_avg_plies,
            policy_init_max_plies=sp.policy_init_max_plies,
            policy_init_temperature=sp.policy_init_temperature,
            tss_enabled=sp.tss_enabled,
            # Root FPU reduction. root_fpu_zero_under_noise and search_parity_mode
            # gate how this interacts with root Dirichlet noise (handled in Rust).
            root_fpu_reduction=sp.root_fpu_reduction,
            root_fpu_zero_under_noise=sp.root_fpu_zero_under_noise,
            search_parity_mode=sp.search_parity_mode,
            divergence_overrides=build_divergence_overrides(
                sp, disabled=(ctx.diagnostics_dir / ML_AUTO_DISABLED_FLAG).exists()
            ),
            **noise_kwargs,
        )
        # Drain and join the writer while the .hxr file is still open; re-raises
        # any write error so all finished games are on disk before the epoch closes.
        driver._stop_writer()
    driver.record_file = None
    driver._write_live("completed")  # final progress marking the epoch done

    elapsed = time.time() - started
    result = {
        "status": "completed",
        "epoch": epoch,
        "elapsed_seconds": elapsed,
        "search_visits": sp.search_visits,
        "scheduler": {k: v for k, v in scheduler_stats.items() if not isinstance(v, dict)},
        **driver.stats(),
    }
    # Attach the cuda.Event GPU-busy report (None unless HEXFIELD_PERF_TRACE=1).
    # getattr defaults to None for evaluators without perf_trace_report.
    perf_report = getattr(evaluator, "perf_trace_report", lambda: None)()
    if perf_report is not None:
        result["perf_trace"] = perf_report
    diag_path = ctx.diagnostics_dir / f"hexfield.selfplay.epoch_{epoch:06d}.json"
    diag_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
