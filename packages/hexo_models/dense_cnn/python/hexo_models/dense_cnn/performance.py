"""Automatic performance calibration for dense CNN training and inference."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

import torch

from .config import Model1Config
from .constants import BOARD_SIZE, INPUT_CHANNELS
from .losses import model1_loss


def calibrate_dense_cnn(
    *,
    model: torch.nn.Module,
    config: Model1Config,
    optimizer: torch.optim.Optimizer | None = None,
    ctx: Any | None = None,
    inference_batch_candidates: Sequence[int] | None = None,
    training_batch_candidates: Sequence[int] | None = None,
    selfplay_batch_candidates: Sequence[int] | None = None,
    mcts_virtual_batch_candidates: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Benchmark candidate batch settings and return measured recommendations."""

    perf = config.performance
    if not perf.calibrate:
        return {
            "status": "skipped",
            "reason": "performance calibration disabled",
            "target_selfplay_positions_per_second": perf.target_selfplay_positions_per_second,
            "meets_target": False,
        }

    requested = torch.device(config.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    model_state = _clone_state_dict(model.state_dict())
    optimizer_state = _clone_optimizer_state(optimizer.state_dict()) if optimizer is not None else None
    try:
        inference_results = _benchmark_inference(
            model,
            device=device,
            amp=config.training.amp and device.type == "cuda",
            candidates=inference_batch_candidates or perf.inference_batch_candidates,
            probe_batches=perf.probe_batches,
        )
        training_results = _benchmark_training(
            model,
            optimizer=optimizer,
            device=device,
            amp=config.training.amp and device.type == "cuda",
            candidates=training_batch_candidates or perf.training_batch_candidates,
            probe_batches=perf.probe_batches,
        )
        selfplay_results = _benchmark_selfplay(
            model,
            config=config,
            batch_candidates=selfplay_batch_candidates or perf.selfplay_batch_candidates,
            virtual_batch_candidates=mcts_virtual_batch_candidates or perf.mcts_virtual_batch_candidates,
            visits=config.selfplay.search_visits,
            probe_positions=perf.selfplay_probe_positions,
        )
    finally:
        model.load_state_dict(model_state)
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    selected_inference_benchmark = _best_batch(inference_results)
    selected_training = _best_batch(training_results)
    selected_selfplay = _select_selfplay_setting(
        selfplay_results,
        target=perf.target_selfplay_positions_per_second,
        configured_visits=config.selfplay.search_visits,
    )
    measured_selfplay = float(selected_selfplay.get("positions_per_second", 0.0))
    selected_exact = (
        int(selected_selfplay.get("visits", 0)) == int(config.selfplay.search_visits)
        and bool(selected_selfplay.get("all_searches_exact", False))
    )
    result = {
        "status": "completed",
        "device": str(device),
        "amp": bool(config.training.amp and device.type == "cuda"),
        "selected_inference_batch_size": int(selected_inference_benchmark.get("batch_size", 1)),
        "selected_selfplay_batch_size": int(selected_selfplay.get("selfplay_batch_size", config.selfplay.active_games)),
        "selected_mcts_virtual_batch_size": int(selected_selfplay.get("mcts_virtual_batch_size", 0)),
        "selected_training_batch_size": int(selected_training.get("batch_size", config.training.batch_size)),
        "selected_mcts_visits": int(selected_selfplay.get("visits", config.selfplay.search_visits)),
        "inference_positions_per_second": float(selected_inference_benchmark.get("positions_per_second", 0.0)),
        "training_samples_per_second": float(selected_training.get("positions_per_second", 0.0)),
        "inference": inference_results,
        "training": training_results,
        "selfplay": selfplay_results,
        "target_selfplay_positions_per_second": perf.target_selfplay_positions_per_second,
        "target_mcts_simulations_per_position": int(config.selfplay.search_visits),
        "measured_selfplay_positions_per_second": measured_selfplay,
        "searched_positions": int(selected_selfplay.get("searched_positions", selected_selfplay.get("positions", 0))),
        "recorded_positions": int(selected_selfplay.get("recorded_positions", selected_selfplay.get("positions", 0))),
        "mcts_simulations": int(selected_selfplay.get("mcts_simulations", 0)),
        "exact_visit_results": int(selected_selfplay.get("exact_visit_results", 0)),
        "all_searches_exact": selected_exact,
        "meets_target": measured_selfplay >= perf.target_selfplay_positions_per_second and selected_exact,
    }
    if ctx is not None:
        path = ctx.diagnostics.write_json("dense_cnn.performance_calibration.json", result)
        result["calibration_path"] = str(path)
    return result


def calibrate_performance(
    *,
    model: torch.nn.Module,
    config: Model1Config,
    optimizer: torch.optim.Optimizer | None = None,
    inference_batch_candidates: Sequence[int] | None = None,
    training_batch_candidates: Sequence[int] | None = None,
    selfplay_batch_candidates: Sequence[int] | None = None,
    mcts_virtual_batch_candidates: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Public calibration wrapper used by tests and external tooling."""

    return calibrate_dense_cnn(
        model=model,
        config=config,
        optimizer=optimizer,
        inference_batch_candidates=inference_batch_candidates,
        training_batch_candidates=training_batch_candidates,
        selfplay_batch_candidates=selfplay_batch_candidates,
        mcts_virtual_batch_candidates=mcts_virtual_batch_candidates,
    )


def build_benchmark_report(
    *,
    config: Model1Config,
    measurements: dict[str, float],
) -> dict[str, Any]:
    """Build a target-aware benchmark report without inferring success."""

    selfplay = float(measurements.get("selfplay_positions_per_second", measurements.get("measured_selfplay_positions_per_second", 0.0)))
    target = float(config.performance.target_selfplay_positions_per_second)
    return {
        "target_selfplay_positions_per_second": target,
        "selfplay_positions_per_second": selfplay,
        "inference_positions_per_second": float(measurements.get("inference_positions_per_second", 0.0)),
        "training_samples_per_second": float(measurements.get("training_samples_per_second", 0.0)),
        "meets_target": selfplay >= target,
    }


benchmark_report = build_benchmark_report
report_performance_benchmark = build_benchmark_report
calibrate_model1_performance = calibrate_performance
calibrate_dense_cnn_performance = calibrate_performance


def _benchmark_inference(
    model: torch.nn.Module,
    *,
    device: torch.device,
    amp: bool,
    candidates: Sequence[int],
    probe_batches: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for batch_size in candidates:
        batch = max(1, int(batch_size))
        try:
            inputs = torch.randn(batch, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device)
            _sync(device)
            started = perf_counter()
            with torch.no_grad():
                for _ in range(probe_batches):
                    with torch.autocast(device_type=device.type, enabled=amp):
                        _ = model(inputs)
            _sync(device)
            elapsed = perf_counter() - started
            results.append(_throughput_payload(batch, probe_batches, elapsed))
        except RuntimeError as exc:
            if _is_oom(exc):
                _clear_cache(device)
                results.append({"batch_size": batch, "status": "oom", "error": str(exc)})
                continue
            raise
    return results


def _benchmark_training(
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp: bool,
    candidates: Sequence[int],
    probe_batches: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    owns_optimizer = optimizer is None
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    was_training = model.training
    model.train()
    for batch_size in candidates:
        batch = max(1, int(batch_size))
        try:
            inputs = torch.randn(batch, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device)
            target = {
                "policy": torch.full((batch, BOARD_SIZE * BOARD_SIZE), 1.0 / (BOARD_SIZE * BOARD_SIZE), device=device),
                "opp_policy": torch.full((batch, BOARD_SIZE * BOARD_SIZE), 1.0 / (BOARD_SIZE * BOARD_SIZE), device=device),
                "value": torch.zeros(batch, device=device),
            }
            _sync(device)
            started = perf_counter()
            for _ in range(probe_batches):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    outputs = model(inputs)
                    loss, _components = model1_loss(outputs, target)
                loss.backward()
                optimizer.step()
            _sync(device)
            elapsed = perf_counter() - started
            results.append(_throughput_payload(batch, probe_batches, elapsed))
        except RuntimeError as exc:
            if _is_oom(exc):
                _clear_cache(device)
                results.append({"batch_size": batch, "status": "oom", "error": str(exc)})
                continue
            raise
    if not was_training:
        model.eval()
    return results


def _benchmark_selfplay(
    model: torch.nn.Module,
    *,
    config: Model1Config,
    batch_candidates: Sequence[int],
    virtual_batch_candidates: Sequence[int],
    visits: int,
    probe_positions: int,
) -> list[dict[str, Any]]:
    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id

    from .inference import DenseCNNInference
    from .mcts import run_batched_mcts

    results: list[dict[str, Any]] = []
    inference = DenseCNNInference(
        model,
        device=config.device,
        amp=config.training.amp,
        return_logits=False,
    )
    for selfplay_batch_size in batch_candidates:
        for virtual_batch_size in virtual_batch_candidates:
            results.append(
                _benchmark_selfplay_setting(
                    inference=inference,
                    config=config,
                    selfplay_batch_size=max(1, int(selfplay_batch_size)),
                    virtual_batch_size=max(1, int(virtual_batch_size)),
                    visits=max(1, int(visits)),
                    probe_positions=max(1, int(probe_positions)),
                )
            )
    return results


def _benchmark_selfplay_setting(
    *,
    inference: Any,
    config: Model1Config,
    selfplay_batch_size: int,
    virtual_batch_size: int,
    visits: int,
    probe_positions: int,
) -> dict[str, Any]:
    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id

    from .mcts import run_batched_mcts

    resolved_visits = max(1, int(visits))
    target_positions = max(1, int(probe_positions))
    active_limit = max(1, int(selfplay_batch_size))
    games = [
        {
            "state": engine.new_game(seed=31_337 + index),
            "actions": [],
        }
        for index in range(active_limit)
    ]
    positions = 0
    mcts_simulations = 0
    completed_games = 0
    exact_visit_results = 0
    started = perf_counter()
    while positions < target_positions:
        playable = [
            game
            for game in games
            if engine.terminal(game["state"]) is None
            and positions < target_positions
        ]
        if not playable:
            completed_games += len(games)
            games = [
                {
                    "state": engine.new_game(seed=91_000 + completed_games + index),
                    "actions": [],
                }
                for index in range(active_limit)
            ]
            continue
        searches = run_batched_mcts(
            [game["state"] for game in playable],
            inference,
            visits=resolved_visits,
            temperature=config.selfplay.temperature,
            seed=17_000 + resolved_visits + positions,
            virtual_batch_size=virtual_batch_size,
        )
        for game, search in zip(playable, searches):
            if search.visits == resolved_visits:
                exact_visit_results += 1
            mcts_simulations += int(search.visits)
            engine.apply_action(game["state"], engine.PlacementAction(unpack_coord_id(search.action_id)))
            game["actions"].append(int(search.action_id))
            positions += 1
    elapsed = perf_counter() - started
    return {
        "status": "completed",
        "visits": resolved_visits,
        "selfplay_batch_size": active_limit,
        "inference_batch_size": active_limit,
        "batch_size": active_limit,
        "mcts_virtual_batch_size": virtual_batch_size,
        "positions": int(positions),
        "searched_positions": int(positions),
        "recorded_positions": int(positions),
        "mcts_simulations": int(mcts_simulations),
        "exact_visit_results": int(exact_visit_results),
        "all_searches_exact": exact_visit_results == positions,
        "elapsed_seconds": elapsed,
        "positions_per_second": positions / max(elapsed, 1.0e-9),
    }


def _best_batch(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item.get("status") == "completed"]
    if not completed:
        return {"batch_size": 1, "positions_per_second": 0.0}
    return max(completed, key=lambda item: float(item.get("positions_per_second", 0.0)))


def _select_selfplay_setting(
    results: Sequence[dict[str, Any]],
    *,
    target: float,
    configured_visits: int,
) -> dict[str, Any]:
    completed = [item for item in results if item.get("status") == "completed"]
    if not completed:
        return {"visits": configured_visits, "positions_per_second": 0.0}
    viable = [
        item
        for item in completed
        if float(item.get("positions_per_second", 0.0)) >= float(target)
        and int(item.get("visits", 0)) == int(configured_visits)
        and bool(item.get("all_searches_exact", False))
    ]
    if viable:
        return max(
            viable,
            key=lambda item: (
                int(item.get("visits", 0)),
                float(item.get("positions_per_second", 0.0)),
            ),
        )
    exact = [
        item
        for item in completed
        if int(item.get("visits", 0)) == int(configured_visits)
        and bool(item.get("all_searches_exact", False))
    ]
    if exact:
        return max(exact, key=lambda item: float(item.get("positions_per_second", 0.0)))
    return max(completed, key=lambda item: float(item.get("positions_per_second", 0.0)))


def _throughput_payload(batch_size: int, probe_batches: int, elapsed: float) -> dict[str, Any]:
    positions = int(batch_size) * int(probe_batches)
    return {
        "status": "completed",
        "batch_size": int(batch_size),
        "probe_batches": int(probe_batches),
        "positions": positions,
        "elapsed_seconds": elapsed,
        "positions_per_second": positions / max(elapsed, 1.0e-9),
    }


def _is_oom(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: memory" in text


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone().cpu() for key, value in state.items()}


def _clone_optimizer_state(state: dict[str, Any]) -> dict[str, Any]:
    return _clone_nested(state)


def _clone_nested(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().cpu()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return value
