"""Build a dense_cnn bootstrap checkpoint from classical heuristic games.

The production trainer is intentionally self-play first. This script creates a
reproducible curriculum checkpoint before self-play starts: it plays complete
games with a tactical classical policy, finalizes compact schema-2 samples
through the dense_cnn Rust finalizer, trains on exactly the requested sample
count with random D6 expansion, then writes a normal dense_cnn checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_train", "hexo_engine", "hexo_runner", "hexo_utils"):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hexo_engine as engine  # noqa: E402
from hexo_engine.types import AxialCoord, PlacementAction, Player, unpack_coord_id  # noqa: E402
from hexo_models.dense_cnn.architecture import Model1Network  # noqa: E402
from hexo_models.dense_cnn.config import parse_model1_config  # noqa: E402
from hexo_models.dense_cnn.samples import (  # noqa: E402
    CURRENT_TARGET_SCHEMA_VERSION,
    CompressedSample,
    SampleBuffer,
    finalize_game_samples,
    sample_from_state,
)
from hexo_models.dense_cnn.trainer import DenseCNNTrainer, DenseSampleWindow  # noqa: E402
from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer  # noqa: E402
from hexo_runner.loop import run_match_loop  # noqa: E402
from hexo_runner.player import DecisionResult, FinalSummary, GameContext, TransitionEvent, WorkerContext  # noqa: E402
from hexo_runner.records import HexoRecordFile, HexoRecordPlayer  # noqa: E402
from hexo_runner.session import GameSpec  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


AXIS_DELTAS = {
    "Q": (1, 0),
    "R": (0, 1),
    "QR": (1, -1),
}
AXIS_NORMALS = {
    "Q": (0, 1),
    "R": (1, 0),
    "QR": (0, 1),
}


class JsonDiagnostics:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: object) -> Path:
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
        return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    rng = random.Random(args.seed)

    config_path = Path(args.config).resolve()
    raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    model_config = dict(raw_config["model"]["config"])
    if args.device:
        model_config["device"] = args.device
    config = parse_model1_config(model_config)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = JsonDiagnostics(output_dir / "diagnostics")

    model = Model1Network(
        in_channels=config.architecture.input_channels,
        channels=config.architecture.channels,
        blocks=config.architecture.residual_blocks,
        dropout=config.architecture.dropout,
        lookahead_horizons=config.architecture.lookahead_horizons,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    base_payload: Mapping[str, Any] = {}
    base_checkpoint = Path(args.base_checkpoint).resolve() if args.base_checkpoint else None
    base_epoch = 0
    if base_checkpoint is not None:
        base_payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(base_payload["model_state"])
        base_epoch = int(base_payload.get("epoch") or 0)

    buffer = SampleBuffer(
        capacity=max(config.samples.capacity, int(args.sample_count)),
        recency_halflife=config.samples.recency_halflife,
        compression_level=config.samples.compression_level,
    )

    generation_started = time.perf_counter()
    samples_checkpoint = Path(args.samples_checkpoint).resolve() if args.samples_checkpoint else None
    if samples_checkpoint is not None:
        generation = load_samples_from_checkpoint(
            buffer=buffer,
            samples_checkpoint=samples_checkpoint,
            sample_count=int(args.sample_count),
            require_classical=not bool(args.allow_non_classical_samples),
            diagnostics=diagnostics,
        )
    elif args.source == "sealbot":
        generation = generate_sealbot_samples(
            buffer=buffer,
            sample_count=int(args.sample_count),
            seed=int(args.seed),
            max_actions=int(args.max_actions),
            sealbot_path=args.sealbot_path,
            sealbot_variant=str(args.sealbot_variant),
            sealbot_time_limit=float(args.sealbot_time_limit),
            compression_level=config.samples.compression_level,
            diagnostics=diagnostics,
            output_dir=output_dir,
        )
    else:
        generation = generate_classical_samples(
            buffer=buffer,
            sample_count=int(args.sample_count),
            seed=int(args.seed),
            max_actions=int(args.max_actions),
            horizons=config.architecture.lookahead_horizons,
            policy_top_k=int(args.policy_top_k),
            policy_temperature=float(args.policy_temperature),
            compression_level=config.samples.compression_level,
            rng=rng,
            diagnostics=diagnostics,
        )
    generation_elapsed = time.perf_counter() - generation_started

    trainer = DenseCNNTrainer(
        model=model,
        config=config,
        buffer=buffer,
        optimizer=optimizer,
    )
    trainer.training_batch_size = int(args.batch_size)
    train_result: Mapping[str, Any] = {
        "status": "skipped",
        "reason": "train_passes was 0",
    }
    if int(args.train_passes) > 0:
        window = DenseSampleWindow(
            records=buffer.entries[: int(args.sample_count)],
            seed=int(args.seed),
            epoch=0,
            index=SimpleNamespace(sample_count=buffer.sample_count, store=None),
            window_size=int(args.sample_count),
            metadata={
                "source": "classical_bootstrap",
                "sample_count": int(args.sample_count),
            },
        )
        train_result = trainer.train_passes(
            passes=int(args.train_passes),
            sample_window=window,
            sample_symmetries=SimpleNamespace(symmetries=()),
            ctx=SimpleNamespace(
                config=SimpleNamespace(run=SimpleNamespace(seed=int(args.seed))),
                diagnostics=diagnostics,
            ),
            components=SimpleNamespace(),
            epoch=0,
        )

    checkpoint_epoch = int(args.checkpoint_epoch if args.checkpoint_epoch is not None else base_epoch)
    output_checkpoint = Path(args.output_checkpoint).resolve()
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "hexo_models.dense_cnn",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sample_buffer": buffer.state_dict(),
        "epoch": checkpoint_epoch,
        "metadata": {
            "run": "dense_cnn_model1",
            "bootstrap": f"classical_{args.source}",
            "bootstrap_sample_count": int(args.sample_count),
            "bootstrap_train_passes": int(args.train_passes),
            "bootstrap_generation": generation,
            "base_checkpoint": str(base_checkpoint) if base_checkpoint is not None else None,
            "base_epoch": base_epoch,
            "sample_count": buffer.sample_count,
            "target_schema_version": CURRENT_TARGET_SCHEMA_VERSION,
        },
    }
    torch.save(payload, output_checkpoint)

    pointer_result = None
    if args.publish_pointer:
        pointer = Path(args.publish_pointer).resolve()
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(str(output_checkpoint), encoding="utf-8")
        pointer_result = {"status": "updated", "pointer_path": str(pointer)}

    summary = {
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - started,
        "generation_elapsed_seconds": generation_elapsed,
        "config": str(config_path),
        "base_checkpoint": str(base_checkpoint) if base_checkpoint is not None else None,
        "checkpoint": str(output_checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "pointer": pointer_result,
        "sample_count": buffer.sample_count,
        "compressed_bytes": buffer.compressed_bytes,
        "target_schema_version": CURRENT_TARGET_SCHEMA_VERSION,
        "generation": generation,
        "training": train_result,
    }
    diagnostics.write_json("classical_bootstrap.summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


def load_samples_from_checkpoint(
    *,
    buffer: SampleBuffer,
    samples_checkpoint: Path,
    sample_count: int,
    require_classical: bool,
    diagnostics: JsonDiagnostics,
) -> dict[str, Any]:
    payload = torch.load(samples_checkpoint, map_location="cpu", weights_only=False)
    sample_state = payload.get("sample_buffer")
    if not isinstance(sample_state, Mapping):
        raise ValueError(f"Checkpoint has no dense_cnn sample_buffer: {samples_checkpoint}")
    load_stats = buffer.load_state_dict(sample_state)
    if buffer.sample_count < int(sample_count):
        raise ValueError(
            f"Sample checkpoint only loaded {buffer.sample_count} samples; "
            f"{int(sample_count)} are required."
        )
    if buffer.sample_count > int(sample_count):
        selected = buffer.entries[: int(sample_count)]
        buffer._samples = []  # noqa: SLF001 - script-owned controlled trim.
        buffer._total_appended = 0  # noqa: SLF001
        buffer.extend(selected)

    source_counts = _sample_source_counts(buffer.entries)
    if require_classical:
        non_classical = {
            source: count
            for source, count in source_counts.items()
            if not _is_classical_sample_source(source)
        }
        if non_classical:
            raise ValueError(
                "Sample checkpoint contains non-classical sources; "
                f"pass --allow-non-classical-samples to override: {non_classical}"
            )

    result = {
        "source": "checkpoint_sample_buffer",
        "sample_checkpoint": str(samples_checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "samples": buffer.sample_count,
        "target_samples": int(sample_count),
        "loaded_samples": int(load_stats.get("loaded", buffer.sample_count)),
        "filtered_samples": int(load_stats.get("filtered", 0)),
        "source_counts": dict(source_counts),
        "checkpoint_metadata": dict(payload.get("metadata", {}) or {}),
    }
    diagnostics.write_json(
        "classical_bootstrap.progress.json",
        {
            "source": result["source"],
            "samples": buffer.sample_count,
            "target_samples": int(sample_count),
            "sample_checkpoint": str(samples_checkpoint),
            "source_counts": dict(source_counts),
        },
    )
    return result


def generate_classical_samples(
    *,
    buffer: SampleBuffer,
    sample_count: int,
    seed: int,
    max_actions: int,
    horizons: Sequence[int],
    policy_top_k: int,
    policy_temperature: float,
    compression_level: int,
    rng: random.Random,
    diagnostics: JsonDiagnostics,
) -> dict[str, Any]:
    games = 0
    generated = 0
    lengths: list[int] = []
    winners: Counter[str] = Counter()
    truncated_games = 0
    preview_games: list[dict[str, Any]] = []

    while buffer.sample_count < sample_count:
        game_id = f"classical-bootstrap-{games:06d}"
        state = engine.new_game(seed=seed + games)
        plan = build_tactical_plan(games, rng)
        pending = []
        actions: list[int] = []
        root_values: list[float] = []

        for turn_index in range(max_actions):
            terminal = engine.terminal(state)
            if terminal is not None:
                break
            policy, root_value = planned_classical_policy(
                state,
                plan=plan,
                rng=rng,
                top_k=policy_top_k,
                temperature=policy_temperature,
            )
            sample = sample_from_state(
                state,
                game_id=game_id,
                turn_index=turn_index,
                policy=policy,
                value=root_value,
                metadata={
                    "bootstrap": "classical_tactical",
                    "sample_source": "classical_tactical_bootstrap",
                    "classical_winner_plan": plan["winner"],
                    "search_visits": 0,
                    "configured_search_visits": 0,
                    "mcts_sims_exact": False,
                },
            )
            pending.append((sample.current_player, sample, root_value))
            action_id = sample_action(policy, rng)
            actions.append(action_id)
            root_values.append(root_value)
            engine.apply_action(state, PlacementAction(unpack_coord_id(action_id)))

        terminal = engine.terminal(state)
        truncated = terminal is None
        winner = str(terminal.winner) if terminal is not None and terminal.winner is not None else None
        finalized = finalize_game_samples(
            pending,
            winner=winner,
            horizons=horizons,
            truncated=truncated,
        )
        remaining = sample_count - buffer.sample_count
        buffer.extend(finalized[:remaining])
        generated += min(len(finalized), remaining)
        games += 1
        lengths.append(len(actions))
        if truncated:
            truncated_games += 1
        else:
            winners.update([winner or "draw"])
        if len(preview_games) < 8:
            preview_games.append(
                {
                    "game_id": game_id,
                    "winner": winner,
                    "planned_winner": plan["winner"],
                    "winner_line": [list(coord_key(coord)) for coord in plan["winner_line"]],
                    "truncated": truncated,
                    "length": len(actions),
                    "first_actions": actions[:32],
                    "root_value_min": min(root_values) if root_values else None,
                    "root_value_max": max(root_values) if root_values else None,
                }
            )
        if games % 25 == 0 or buffer.sample_count >= sample_count:
            diagnostics.write_json(
                "classical_bootstrap.progress.json",
                {
                    "games": games,
                    "samples": buffer.sample_count,
                    "target_samples": sample_count,
                    "latest_length": lengths[-1] if lengths else None,
                    "winners": dict(winners),
                    "truncated_games": truncated_games,
                },
            )

    lengths_sorted = sorted(lengths)
    return {
        "games": games,
        "samples": generated,
        "target_samples": sample_count,
        "winners": dict(winners),
        "truncated_games": truncated_games,
        "min_length": min(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
        "mean_length": (sum(lengths) / len(lengths)) if lengths else None,
        "median_length": lengths_sorted[len(lengths_sorted) // 2] if lengths_sorted else None,
        "preview_games": preview_games,
    }


class RecordingClassicalPlayer:
    def __init__(
        self,
        inner: SealBotPlayer,
        *,
        sample_sink: list[tuple[str, object, float]],
        action_counter: list[int],
        source: str,
        compression_level: int,
    ) -> None:
        self.inner = inner
        self.identity = inner.identity
        self.sample_sink = sample_sink
        self.action_counter = action_counter
        self.source = source
        self.compression_level = compression_level
        self.game_id = ""

    def setup_worker(self, context: WorkerContext) -> None:
        self.inner.setup_worker(context)

    def start_game(self, context: GameContext) -> None:
        self.game_id = context.game_id
        self.inner.start_game(context)

    def decide(self, state: engine.HexoState) -> DecisionResult:
        decision = self.inner.decide(state)
        action_id = int(engine.action_id(decision.action))
        turn_index = int(self.action_counter[0])
        sample = sample_from_state(
            state,
            game_id=self.game_id,
            turn_index=turn_index,
            policy={action_id: 1.0},
            value=0.0,
            metadata={
                "bootstrap": self.source,
                "sample_source": self.source,
                "classical_policy": self.inner.identity.player_id,
                "search_visits": 0,
                "configured_search_visits": 0,
                "mcts_sims_exact": False,
                "lookahead_source": "none_non_mcts_classical",
            },
        )
        self.sample_sink.append(
            (
                sample.current_player,
                CompressedSample.from_data(sample, compression_level=self.compression_level),
                0.0,
            )
        )
        self.action_counter[0] = turn_index + 1
        return decision

    def observe_transition(self, transition: TransitionEvent) -> None:
        self.inner.observe_transition(transition)

    def finish_game(self, final_summary: FinalSummary) -> None:
        self.inner.finish_game(final_summary)

    def close(self) -> None:
        self.inner.close()


def generate_sealbot_samples(
    *,
    buffer: SampleBuffer,
    sample_count: int,
    seed: int,
    max_actions: int,
    sealbot_path: str | None,
    sealbot_variant: str,
    sealbot_time_limit: float,
    compression_level: int,
    diagnostics: JsonDiagnostics,
    output_dir: Path,
) -> dict[str, Any]:
    config = SealBotConfig(
        path=sealbot_path,
        variant=sealbot_variant,
        time_limit=sealbot_time_limit,
        response_timeout=max(30.0, sealbot_time_limit * 20.0 + 5.0),
    )
    config.validate()

    record_path = output_dir / "classical_sealbot_bootstrap.hxr"
    players_payload = (
        HexoRecordPlayer(f"sealbot-{sealbot_variant}-p0", "player0", f"SealBot {sealbot_variant} P0"),
        HexoRecordPlayer(f"sealbot-{sealbot_variant}-p1", "player1", f"SealBot {sealbot_variant} P1"),
    )
    sample_source = f"classical_sealbot_{sealbot_variant}_bootstrap"
    games = 0
    generated = 0
    lengths: list[int] = []
    winners: Counter[str] = Counter()
    truncated_games = 0
    aborted_games = 0
    preview_games: list[dict[str, Any]] = []

    game_pending: list[tuple[str, object, float]] = []
    action_counter = [0]
    player0 = RecordingClassicalPlayer(
        SealBotPlayer(config, player_id=f"sealbot-{sealbot_variant}-p0"),
        sample_sink=game_pending,
        action_counter=action_counter,
        source=sample_source,
        compression_level=compression_level,
    )
    player1 = RecordingClassicalPlayer(
        SealBotPlayer(config, player_id=f"sealbot-{sealbot_variant}-p1"),
        sample_sink=game_pending,
        action_counter=action_counter,
        source=sample_source,
        compression_level=compression_level,
    )
    worker = WorkerContext(worker_id=0, engine_metadata=engine.engine_metadata())
    player0.setup_worker(worker)
    player1.setup_worker(worker)
    try:
        with HexoRecordFile.create(record_path, engine.engine_metadata(), players_payload) as record_file:
            while buffer.sample_count < sample_count:
                game_pending.clear()
                action_counter[0] = 0
                game_id = f"classical-sealbot-{games:06d}"
                result = run_match_loop(
                    GameSpec(
                        game_id=game_id,
                        seed=seed + games,
                        max_actions=max_actions,
                        mode="bootstrap",
                        is_evaluation=False,
                    ),
                    (player0, player1),
                    record_file,
                    worker_context=worker,
                    setup_players=False,
                    close_players=False,
                )
                pending = list(game_pending)
                truncated = str(result.status) != "completed" or result.winner is None
                finalized = finalize_game_samples(
                    pending,
                    winner=str(result.winner) if result.winner is not None else None,
                    horizons=(),
                    truncated=truncated,
                )
                finalized = [
                    replace(
                        sample,
                        metadata={
                            **dict(sample.metadata),
                            "opp_policy_source": (
                                "future_opponent_classical" if sample.opp_policy else "none"
                            ),
                        },
                    )
                    for sample in finalized
                ]
                remaining = sample_count - buffer.sample_count
                buffer.extend(finalized[:remaining])
                generated += min(len(finalized), remaining)
                games += 1
                lengths.append(int(result.turns))
                if str(result.status) != "completed":
                    aborted_games += 1
                elif truncated:
                    truncated_games += 1
                else:
                    winners.update([str(result.winner)])
                if len(preview_games) < 8:
                    preview_games.append(
                        {
                            "game_id": game_id,
                            "winner": result.winner,
                            "status": str(result.status),
                            "length": int(result.turns),
                            "samples": len(finalized),
                            "abort": asdict(result.abort) if result.abort is not None else None,
                        }
                    )
                if games % 10 == 0 or buffer.sample_count >= sample_count:
                    diagnostics.write_json(
                        "classical_bootstrap.progress.json",
                        {
                            "source": sample_source,
                            "games": games,
                            "samples": buffer.sample_count,
                            "target_samples": sample_count,
                            "latest_length": lengths[-1] if lengths else None,
                            "winners": dict(winners),
                            "truncated_games": truncated_games,
                            "aborted_games": aborted_games,
                            "record_path": str(record_path),
                        },
                    )
    finally:
        player0.close()
        player1.close()

    lengths_sorted = sorted(lengths)
    return {
        "source": sample_source,
        "sealbot_variant": sealbot_variant,
        "sealbot_time_limit": sealbot_time_limit,
        "record_path": str(record_path),
        "games": games,
        "samples": generated,
        "target_samples": sample_count,
        "winners": dict(winners),
        "truncated_games": truncated_games,
        "aborted_games": aborted_games,
        "min_length": min(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
        "mean_length": (sum(lengths) / len(lengths)) if lengths else None,
        "median_length": lengths_sorted[len(lengths_sorted) // 2] if lengths_sorted else None,
        "preview_games": preview_games,
        "lookahead_targets": "omitted_non_mcts_classical",
    }


def build_tactical_plan(game_index: int, rng: random.Random) -> dict[str, Any]:
    winner = Player.PLAYER_0 if game_index % 2 == 0 else Player.PLAYER_1
    axis = rng.choice(tuple(AXIS_DELTAS))
    dq, dr = AXIS_DELTAS[axis]
    offset = rng.randrange(6)
    if winner == Player.PLAYER_0:
        start = AxialCoord(q=-dq * offset, r=-dr * offset)
    else:
        normal_q, normal_r = AXIS_NORMALS[axis]
        side = -1 if rng.random() < 0.5 else 1
        distance = rng.choice((1, 2))
        start = AxialCoord(
            q=-dq * offset + normal_q * side * distance,
            r=-dr * offset + normal_r * side * distance,
        )
    return {
        "winner": str(winner),
        "axis": axis,
        "winner_line": tuple(AxialCoord(start.q + dq * index, start.r + dr * index) for index in range(6)),
    }


def planned_classical_policy(
    state: object,
    *,
    plan: Mapping[str, Any],
    rng: random.Random,
    top_k: int,
    temperature: float,
) -> tuple[dict[int, float], float]:
    mirror = engine.to_python_state(state)
    current = mirror.current_player
    planned_winner = Player(plan["winner"])
    winner_line = tuple(plan["winner_line"])
    winner_line_keys = {coord_key(coord) for coord in winner_line}
    legal_ids = tuple(int(action_id) for action_id in engine.legal_action_ids(state))
    legal_coords = {action_id: unpack_coord_id(action_id) for action_id in legal_ids}
    occupied = {coord_key(coord): player for coord, player in mirror.board.stones}
    own_line_count = sum(1 for coord in winner_line if occupied.get(coord_key(coord)) == planned_winner)

    if current == planned_winner:
        target_ids = [
            action_id
            for action_id, coord in legal_coords.items()
            if coord_key(coord) in winner_line_keys
        ]
        if target_ids:
            scored = sorted(
                (
                    line_completion_score(legal_coords[action_id], winner_line, occupied, planned_winner)
                    + rng.random() * 1.0e-4,
                    action_id,
                )
                for action_id in target_ids
            )
            scored.reverse()
            top = scored[: max(1, min(top_k, len(scored)))]
            weights = softmax_scores(top, temperature=max(1.0e-6, temperature))
            value = math.tanh((own_line_count + 1.0) / 3.0)
            return {action_id: weight for (_score, action_id), weight in zip(top, weights)}, value

    distractors = [
        action_id
        for action_id, coord in legal_coords.items()
        if coord_key(coord) not in winner_line_keys
    ]
    if distractors:
        scored = sorted(
            (
                distractor_score(legal_coords[action_id], winner_line, mirror)
                + rng.random() * 1.0e-4,
                action_id,
            )
            for action_id in distractors
        )
        scored.reverse()
        top = scored[: max(1, min(top_k, len(scored)))]
        weights = softmax_scores(top, temperature=max(1.0e-6, temperature))
        value = -math.tanh(own_line_count / 3.0) if current != planned_winner else 0.0
        return {action_id: weight for (_score, action_id), weight in zip(top, weights)}, value

    return classical_policy(state, rng=rng, top_k=top_k, temperature=temperature)


def line_completion_score(
    coord: AxialCoord,
    winner_line: Sequence[AxialCoord],
    occupied: Mapping[tuple[int, int], Player],
    winner: Player,
) -> float:
    keys = [coord_key(item) for item in winner_line]
    index = keys.index(coord_key(coord))
    adjacent = 0
    for neighbor_index in (index - 1, index + 1):
        if 0 <= neighbor_index < len(keys) and occupied.get(keys[neighbor_index]) == winner:
            adjacent += 1
    center_bonus = 3.0 - abs(index - 2.5)
    return 100.0 + adjacent * 25.0 + center_bonus


def distractor_score(coord: AxialCoord, winner_line: Sequence[AxialCoord], mirror: Any) -> float:
    distance = min(hex_distance(coord.q, coord.r, target.q, target.r) for target in winner_line)
    stone_distance = min(
        (
            hex_distance(coord.q, coord.r, stone.q, stone.r)
            for stone, _player in mirror.board.stones
        ),
        default=0,
    )
    return distance * 3.0 - abs(stone_distance - 1) * 0.25 + centrality_score(coord)


def classical_policy(
    state: object,
    *,
    rng: random.Random,
    top_k: int,
    temperature: float,
) -> tuple[dict[int, float], float]:
    mirror = engine.to_python_state(state)
    current = mirror.current_player
    opponent = Player.PLAYER_1 if current == Player.PLAYER_0 else Player.PLAYER_0
    legal_ids = tuple(int(action_id) for action_id in engine.legal_action_ids(state))
    legal_coords = {action_id: unpack_coord_id(action_id) for action_id in legal_ids}
    legal_by_coord = {coord_key(coord): action_id for action_id, coord in legal_coords.items()}
    scores = {
        action_id: centrality_score(coord)
        for action_id, coord in legal_coords.items()
    }
    own_index = player_index(current)
    opponent_index = player_index(opponent)

    for entry in mirror.board.windows.entries:
        masks = tuple(int(item) for item in entry.masks)
        own_mask = masks[own_index]
        opponent_mask = masks[opponent_index]
        occupied_mask = own_mask | opponent_mask
        own_count = own_mask.bit_count()
        opponent_count = opponent_mask.bit_count()
        cells = window_cells(entry.key.start, str(entry.key.axis))
        for index, coord in enumerate(cells):
            if occupied_mask & (1 << index):
                continue
            action_id = legal_by_coord.get(coord_key(coord))
            if action_id is None:
                continue
            if own_count and not opponent_count:
                scores[action_id] += own_window_score(own_count)
            if opponent_count and not own_count:
                scores[action_id] += opponent_window_score(opponent_count)

    add_local_shape_scores(scores, legal_coords, mirror, current, opponent)
    scored = sorted(
        (
            (score + rng.random() * 1.0e-4, action_id)
            for action_id, score in scores.items()
        ),
        reverse=True,
    )
    top = scored[: max(1, min(int(top_k), len(scored)))]
    weights = softmax_scores(top, temperature=max(1.0e-6, float(temperature)))
    policy = {action_id: weight for (_score, action_id), weight in zip(top, weights)}
    root_value = heuristic_root_value(scores, scored)
    return policy, root_value


def add_local_shape_scores(
    scores: dict[int, float],
    legal_coords: Mapping[int, AxialCoord],
    mirror: Any,
    current: Player,
    opponent: Player,
) -> None:
    own = [coord for coord, player in mirror.board.stones if player == current]
    opp = [coord for coord, player in mirror.board.stones if player == opponent]
    if not own and not opp:
        return
    occupied = {coord_key(coord): player for coord, player in mirror.board.stones}
    neighbor_deltas = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
    for action_id, coord in legal_coords.items():
        own_neighbors = 0
        opp_neighbors = 0
        for dq, dr in neighbor_deltas:
            player = occupied.get((coord.q + dq, coord.r + dr))
            if player == current:
                own_neighbors += 1
            elif player == opponent:
                opp_neighbors += 1
        scores[action_id] += own_neighbors * 1.25 + opp_neighbors * 0.75


def own_window_score(count: int) -> float:
    if count >= 5:
        return 10000.0
    if count == 4:
        return 400.0
    if count == 3:
        return 48.0
    if count == 2:
        return 9.0
    return 2.0


def opponent_window_score(count: int) -> float:
    if count >= 5:
        return 9000.0
    if count == 4:
        return 360.0
    if count == 3:
        return 36.0
    if count == 2:
        return 7.0
    return 1.5


def centrality_score(coord: AxialCoord) -> float:
    return -0.02 * hex_distance(coord.q, coord.r, 0, 0)


def heuristic_root_value(scores: Mapping[int, float], scored: Sequence[tuple[float, int]]) -> float:
    if not scored:
        return 0.0
    best = float(scored[0][0])
    fifth = float(scored[min(4, len(scored) - 1)][0])
    return math.tanh((best - fifth) / 600.0)


def sample_action(policy: Mapping[int, float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    last = None
    for action_id, weight in policy.items():
        cumulative += float(weight)
        last = int(action_id)
        if threshold <= cumulative:
            return int(action_id)
    if last is None:
        raise ValueError("cannot sample from empty policy")
    return last


def softmax_scores(scored: Sequence[tuple[float, int]], *, temperature: float) -> list[float]:
    values = [score / temperature for score, _action_id in scored]
    offset = max(values)
    exp_values = [math.exp(max(-60.0, value - offset)) for value in values]
    total = sum(exp_values)
    if total <= 0.0:
        return [1.0 / len(exp_values)] * len(exp_values)
    return [value / total for value in exp_values]


def window_cells(start: AxialCoord, axis: str) -> tuple[AxialCoord, ...]:
    dq, dr = AXIS_DELTAS[axis]
    return tuple(AxialCoord(start.q + dq * index, start.r + dr * index) for index in range(6))


def coord_key(coord: AxialCoord) -> tuple[int, int]:
    return (int(coord.q), int(coord.r))


def player_index(player: Player) -> int:
    return 0 if player == Player.PLAYER_0 else 1


def hex_distance(q0: int, r0: int, q1: int, r1: int) -> int:
    dq = q0 - q1
    dr = r0 - r1
    ds = -dq - dr
    return max(abs(dq), abs(dr), abs(ds))


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "dense_cnn_model1.toml"))
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "dense_cnn_model1" / "bootstrap" / "classical_050000"))
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--publish-pointer", default=None)
    parser.add_argument("--sample-count", type=int, default=50_000)
    parser.add_argument("--source", choices=("tactical", "sealbot"), default="sealbot")
    parser.add_argument(
        "--samples-checkpoint",
        default=None,
        help="Reuse the sample_buffer from an existing dense_cnn checkpoint instead of generating games.",
    )
    parser.add_argument(
        "--allow-non-classical-samples",
        action="store_true",
        help="Allow --samples-checkpoint to contain non-classical replay rows.",
    )
    parser.add_argument("--train-passes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-epoch", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=1024)
    parser.add_argument("--policy-top-k", type=int, default=16)
    parser.add_argument("--policy-temperature", type=float, default=0.75)
    parser.add_argument("--sealbot-path", default=None)
    parser.add_argument("--sealbot-variant", default="best")
    parser.add_argument("--sealbot-time-limit", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def _sample_source_counts(records: Sequence[CompressedSample]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sample in records:
        data = sample.decode() if isinstance(sample, CompressedSample) else sample
        metadata = dict(getattr(data, "metadata", {}) or {})
        source = (
            metadata.get("sample_source")
            or metadata.get("bootstrap")
            or metadata.get("source")
            or "unknown"
        )
        counts[str(source)] += 1
    return {source: int(counts[source]) for source in sorted(counts)}


def _is_classical_sample_source(source: str) -> bool:
    normalized = source.lower()
    return normalized.startswith("classical") or "bootstrap" in normalized


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
