"""Continuous self-play epoch driver: run_continuous owns the epoch; this module
owns the Python side of the on_move protocol.

Per game the driver tracks the placement history incrementally (the same
tested derivations the BC writer uses: ordinal phase/player, window_scan hot/
standing-win cells), records FULL-search decisions as pending samples, applies
the chosen action through hexo_engine, and at game end finalizes targets
(hard z, opp policy with fast-masking, STV, moves_left) and writes one
hexfield_compact_v1 shard. Truncated games (max_game_plies reached, no engine
winner) ARE written too: their outcome-INDEPENDENT heads (policy, opp_policy)
train normally while the value/stvalue/cell_q/moves_left heads are masked via
the truncated flag (outcome_valid=0 column → value_mask=0 at expand).
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
        # Open .hxr game-record file for the epoch (dashboard-viewable replays);
        # None disables recording. Set by generate_selfplay_epoch.
        self.record_file = record_file
        # Live within-epoch progress for the :8080 dashboard (progress bar +
        # positions/second). Written to <diag_dir>/hexfield.selfplay.live.json
        # every LIVE_INTERVAL_S during the epoch; the dashboard reads it
        # lineage-aware. None disables the live file.
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
        # Background shard writer: the heavy per-game finalize + .hxr record + zlib
        # npz write runs off the Rust on_move callback thread so game completions
        # never stall the search loop.
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_errors: list[BaseException] = []
        self._writer_failed = threading.Event()
        self._writer_thread: threading.Thread | None = None

    LIVE_INTERVAL_S = 3.0

    def _write_live(self, status: str) -> None:
        """Emit hexfield.selfplay.live.json for the dashboard (progress bar +
        pos/s). Throttled to LIVE_INTERVAL_S while running; forced on
        start/completed. Emits the field contract the dashboard's live-status panel
        + sub-phase derivation read."""

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
        # Cosmetic dashboard file: a write failure must NEVER crash self-play.
        try:
            path = self.diag_dir / "hexfield.selfplay.live.json"
            tmp = self.diag_dir / "hexfield.selfplay.live.json.tmp"
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(path)  # atomic: dashboard never reads a half-written file
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
        self._write_live("running")  # throttled live progress for the dashboard

        current = record_player(tape.ply)
        if full and not init:
            self.full_decisions += 1
            ids = np.frombuffer(bytes(payload["visit_policy_action_ids_bytes"]), dtype=np.uint32)
            weights = np.frombuffer(bytes(payload["visit_policy_weights_bytes"]), dtype=np.float32)
            # Per-cell Q (one Q per recorded action, SAME set+order as the visit
            # policy — Rust contract) feeds the train-only cell_q head.
            qs = np.frombuffer(bytes(payload["visit_policy_q_bytes"]), dtype=np.float32)
            # Policy-surprise = KL(visit ‖ root prior); reweights the self CE.
            prior_ids = np.frombuffer(
                bytes(payload["root_prior_policy_action_ids_bytes"]), dtype=np.uint32
            )
            prior_weights = np.frombuffer(
                bytes(payload["root_prior_policy_weights_bytes"]), dtype=np.float32
            )
            surprise = _policy_surprise_kl(ids, weights, prior_ids, prior_weights)
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
                policy_surprise=float(surprise),
                metadata={"pcr_full": True},
            )
            tape.pending.append((current, sample, float(payload["root_value"])))
            probs = weights[weights > 0]
            if probs.size:
                self.policy_entropies.append(float(-(probs * np.log(probs)).sum()))
            self.root_values.append(float(payload["root_value"]))
        elif not full and not init:
            # Fast rows are never written, but the pending list keeps every
            # decision so opp-policy lookup + moves_left counts stay exact
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
        """Write one ``.hxr`` game record so the dashboard can replay the model's
        play. EVERY finished game is recorded (completed AND truncated) — only the
        TRAINING rows are completed-games-only (handled in ``_finish``). Records
        the full placement sequence in move order, then closes the game with the
        engine winner label (``player0``/``player1``) or an abort for truncation."""

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
            # Contract: a written (non-truncated) game has a concrete engine
            # winner. `_winner_value(None, ...)` silently codes a draw (z = 0),
            # so a missing winner would poison every row's value target instead
            # of raising — the truncated path is the ONLY legitimate `None`.
            assert winner is not None, "non-truncated finish requires an engine winner"
        # A prior writer-thread failure aborts the epoch here rather than letting
        # the run silently drop shards while the queue backs up.
        if self._writer_failed.is_set():
            raise self._writer_errors[0]
        # Hand the finished (about-to-be-detached) tape to the background writer.
        # The tape is immutable after the game ends, so the handoff is race-free;
        # __call__ deletes it from self.games immediately after this returns.
        self._write_queue.put((tape, winner, truncated))

    def _writer_loop(self) -> None:
        """Background shard writer. Drains finished games and does the heavy I/O —
        .hxr record, finalize,
        and the zlib `write_compact_shard` — off the search-callback thread.
        Bytes are byte-identical to the inline path; only the writing thread
        moves. A write failure is captured and surfaced, never swallowed."""

        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return
                tape, winner, truncated = item
                # Record the game for dashboard replay (completed AND truncated),
                # mirroring the old inline order.
                self._write_record(tape, winner=winner, truncated=truncated)
                # Truncated games (max_game_plies hit, no engine winner) are NO
                # LONGER dropped: their rows ARE written so the outcome-INDEPENDENT
                # heads (policy, opp_policy) train on them, while the value /
                # stvalue / cell_q / moves_left heads are masked downstream (the
                # truncated metadata flag → outcome_valid=0 shard column →
                # value_mask=0 + zeroed stvalue/cell_q masks at expand). Completed
                # games are finalized exactly as before (truncated=False), so their
                # training stays byte-identical.
                finalized = finalize_game_samples(
                    tape.pending, winner, self.horizons,
                    truncated=truncated, mask_opp_from_fast=True,
                )
                rows = [s for s in finalized if s.metadata.get("pcr_full", False)]
                if rows:
                    path = self.out_dir / f"game_{tape.key}.npz"
                    self.rows_written += write_compact_shard(
                        path, rows, short_term_value_horizons=self.horizons,
                        sidecar={
                            "epoch": self.epoch, "game_key": tape.key,
                            "winner": winner, "truncated": bool(truncated),
                        },
                    )
            except BaseException as exc:  # noqa: BLE001 — surface, don't swallow
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
        """Drain queued games and join the writer while the .hxr file is still
        open, then re-raise any writer error so a write failure fails the epoch
        instead of silently dropping shards."""

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
    # Resume support: completed games already wrote their shards, and training
    # reads a rolling mtime window over all epoch_*/game_*.npz, so finished games
    # are already usable. On a mid-epoch restart, KEEP them and generate only the
    # remainder (with keys past any the interrupted run assigned, so nothing is
    # overwritten) instead of recomputing finished play. In-flight (unfinished)
    # games can't be recovered — the remainder replaces them with fresh games.
    existing = sorted(out_dir.glob("game_*.npz"))
    already_done = len(existing)
    remaining = max(games_target - already_done, 0)
    resuming = already_done > 0

    if remaining == 0:
        # Epoch's self-play already on disk (crashed after self-play, before the
        # checkpoint). Skip regeneration entirely.
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

    # Per-epoch .hxr game records under <run>/selfplay (the layout the dashboard
    # scans), so every self-play game is replayable on the History screen.
    record_dir = ctx.output_dir / "selfplay"
    record_dir.mkdir(parents=True, exist_ok=True)
    # On resume use a separate .hxr so the interrupted run's replays aren't
    # truncated (HexoRecordFile.create clobbers); the topup games are recorded
    # alongside. Fresh epochs keep the canonical path the dashboard scans.
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
    driver._t0 = started  # anchor live pos/s to actual self-play start
    driver._write_live("running")  # initial 0% progress bar before the first move
    noise_kwargs = {}
    if sp.root_dirichlet_noise_fraction > 0:
        noise_kwargs = dict(
            root_dirichlet_total_alpha=sp.root_dirichlet_total_alpha,
            root_dirichlet_noise_fraction=sp.root_dirichlet_noise_fraction,
        )
    # Context-managed so the .hxr is finalized (valid file with all completed
    # games) even if run_continuous raises.
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
            root_fpu_zero_under_noise=sp.root_fpu_zero_under_noise,
            search_parity_mode=sp.search_parity_mode,
            divergence_overrides=build_divergence_overrides(
                sp, disabled=(ctx.diagnostics_dir / ML_AUTO_DISABLED_FLAG).exists()
            ),
            **noise_kwargs,
        )
        # Drain + join the writer while the .hxr file is still open (re-raises any
        # write error), so every finished game is on disk before the epoch closes.
        driver._stop_writer()
    driver.record_file = None
    driver._write_live("completed")  # final 100% so the dashboard marks the epoch done

    elapsed = time.time() - started
    result = {
        "status": "completed",
        "epoch": epoch,
        "elapsed_seconds": elapsed,
        "search_visits": sp.search_visits,
        "scheduler": {k: v for k, v in scheduler_stats.items() if not isinstance(v, dict)},
        **driver.stats(),
    }
    diag_path = ctx.diagnostics_dir / f"hexfield.selfplay.epoch_{epoch:06d}.json"
    diag_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
