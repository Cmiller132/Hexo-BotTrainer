"""Automatic performance calibration for dense CNN training and inference."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping, Sequence

import torch

from .config import Model1Config
from .constants import BOARD_SIZE, INPUT_CHANNELS
from .losses import model1_loss

CALIBRATION_CACHE_VERSION = 5
MCTS_BACKEND_SIGNATURE = "dense_cnn_katago_tree_reuse_staged_edges_bounded_cache_v1"
MCTS_EVAL_CHUNK_STATES = 1024


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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model.to(device=device, memory_format=torch.channels_last)
    else:
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
        if device.type == "cuda":
            model.to(device=device, memory_format=torch.channels_last)
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        _clear_cache(device)
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
        "calibration_cache_version": CALIBRATION_CACHE_VERSION,
        "mcts_backend_signature": MCTS_BACKEND_SIGNATURE,
        "mcts_tree_reuse_session": True,
        "mcts_eval_chunk_states": MCTS_EVAL_CHUNK_STATES,
        "selected_inference_batch_size": int(selected_inference_benchmark.get("batch_size", 1)),
        "selected_selfplay_batch_size": int(selected_selfplay.get("selfplay_batch_size", config.selfplay.active_games)),
        "selected_mcts_virtual_batch_size": int(selected_selfplay.get("mcts_virtual_batch_size", 0)),
        "mcts_progressive_widening_initial_actions": config.selfplay.progressive_widening_initial_actions,
        "mcts_progressive_widening_child_initial_actions": config.selfplay.progressive_widening_child_initial_actions,
        "mcts_progressive_widening_candidate_actions": config.selfplay.progressive_widening_candidate_actions,
        "mcts_progressive_widening_growth_interval": config.selfplay.progressive_widening_growth_interval,
        "mcts_progressive_widening_growth_base": config.selfplay.progressive_widening_growth_base,
        "mcts_active_root_limit": config.selfplay.mcts_active_root_limit,
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
            if device.type == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)
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
            if device.type == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)
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
    from .mcts import new_mcts_session, run_batched_mcts

    results: list[dict[str, Any]] = []
    inference = DenseCNNInference(
        model,
        device=config.device,
        amp=config.training.amp,
        return_logits=False,
            max_batch_size=max(1, min(1024, max(int(item) for item in config.performance.inference_batch_candidates))),
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

    from .mcts import new_mcts_session, run_batched_mcts

    resolved_visits = max(1, int(visits))
    target_positions = max(1, int(probe_positions))
    active_limit = max(1, int(selfplay_batch_size))
    games = [
        {
            "search_key": index,
            "state": engine.new_game(seed=31_337 + index),
            "actions": [],
        }
        for index in range(active_limit)
    ]
    positions = 0
    mcts_simulations = 0
    completed_games = 0
    exact_visit_results = 0
    mcts_diagnostic_batches: list[Mapping[str, Any]] = []
    mcts_session = new_mcts_session(max_states=config.selfplay.mcts_evaluation_cache_max_states)
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
                    "search_key": completed_games + index,
                    "state": engine.new_game(seed=91_000 + completed_games + index),
                    "actions": [],
                }
                for index in range(active_limit)
            ]
            continue
        if hasattr(inference, "evaluate_model1_payload"):
            searches = mcts_session.run(
                [int(game["search_key"]) for game in playable],
                [game["state"] for game in playable],
                inference,
                visits=resolved_visits,
                temperature=config.selfplay.temperature,
                seed=17_000 + resolved_visits + positions,
                virtual_batch_size=virtual_batch_size,
                progressive_widening_initial_actions=config.selfplay.progressive_widening_initial_actions,
                progressive_widening_child_initial_actions=config.selfplay.progressive_widening_child_initial_actions,
                progressive_widening_candidate_actions=config.selfplay.progressive_widening_candidate_actions,
                progressive_widening_growth_interval=config.selfplay.progressive_widening_growth_interval,
                progressive_widening_growth_base=config.selfplay.progressive_widening_growth_base,
                active_root_limit=config.selfplay.mcts_active_root_limit,
            )
        else:
            searches = run_batched_mcts(
                [game["state"] for game in playable],
                inference,
                visits=resolved_visits,
                temperature=config.selfplay.temperature,
                seed=17_000 + resolved_visits + positions,
                virtual_batch_size=virtual_batch_size,
                progressive_widening_initial_actions=config.selfplay.progressive_widening_initial_actions,
                progressive_widening_child_initial_actions=config.selfplay.progressive_widening_child_initial_actions,
                progressive_widening_candidate_actions=config.selfplay.progressive_widening_candidate_actions,
                progressive_widening_growth_interval=config.selfplay.progressive_widening_growth_interval,
                progressive_widening_growth_base=config.selfplay.progressive_widening_growth_base,
                evaluation_cache=None,
                active_root_limit=config.selfplay.mcts_active_root_limit,
            )
        if searches:
            _extend_mcts_diagnostic_batches(mcts_diagnostic_batches, searches)
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
        "inference_batch_size": int(getattr(inference, "max_batch_size", active_limit)),
        "batch_size": active_limit,
        "mcts_virtual_batch_size": virtual_batch_size,
        "mcts_tree_reuse_session": True,
        "mcts_progressive_widening_initial_actions": config.selfplay.progressive_widening_initial_actions,
        "mcts_progressive_widening_child_initial_actions": config.selfplay.progressive_widening_child_initial_actions,
        "mcts_progressive_widening_candidate_actions": config.selfplay.progressive_widening_candidate_actions,
        "mcts_progressive_widening_growth_interval": config.selfplay.progressive_widening_growth_interval,
        "mcts_progressive_widening_growth_base": config.selfplay.progressive_widening_growth_base,
        "mcts_evaluation_cache_max_states": config.selfplay.mcts_evaluation_cache_max_states,
        "mcts_active_root_limit": config.selfplay.mcts_active_root_limit,
        "mcts_diagnostics": _summarize_mcts_diagnostic_batches(mcts_diagnostic_batches),
        "positions": int(positions),
        "searched_positions": int(positions),
        "recorded_positions": int(positions),
        "mcts_simulations": int(mcts_simulations),
        "exact_visit_results": int(exact_visit_results),
        "all_searches_exact": exact_visit_results == positions,
        "elapsed_seconds": elapsed,
        "positions_per_second": positions / max(elapsed, 1.0e-9),
    }


def _summarize_mcts_diagnostic_batches(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"batch_count": len(batches)}
    if not batches:
        return summary

    tree_sum_fields = (
        "node_count",
        "active_edge_count",
        "hidden_prior_count",
        "completed_visits",
        "widened_edges_total",
    )
    tree_max_fields = (
        "root_count",
        "max_nodes_per_root",
        "max_active_edges_per_root",
        "max_hidden_priors_per_root",
        "max_active_edges_per_node",
        "max_hidden_priors_per_node",
    )
    eval_sum_fields = (
        "requested_states",
        "cache_hits",
        "duplicate_hits",
        "unique_states",
        "evaluator_chunks",
        "encoded_states",
        "encoded_legal_actions",
        "input_bytes",
        "legal_index_bytes",
        "value_bytes",
        "prior_bytes",
        "cache_inserts",
        "cache_insert_skipped",
    )
    eval_max_fields = ("max_chunk_states", "max_chunk_legal_actions", "cache_size", "cache_size_peak")
    eval_float_sum_fields = ("encoding_seconds", "evaluator_seconds")

    for field in tree_sum_fields:
        summary[f"tree_{field}"] = sum(_int_nested(batch, "tree", field) for batch in batches)
    for field in tree_max_fields:
        summary[f"tree_{field}"] = max(_int_nested(batch, "tree", field) for batch in batches)
    for field in eval_sum_fields:
        summary[f"eval_{field}"] = sum(_int_nested(batch, "evaluation", field) for batch in batches)
    for field in eval_max_fields:
        summary[f"eval_{field}"] = max(_int_nested(batch, "evaluation", field) for batch in batches)
    for field in eval_float_sum_fields:
        summary[f"eval_{field}"] = sum(_float_nested(batch, "evaluation", field) for batch in batches)
    return summary


def _extend_mcts_diagnostic_batches(destination: list[Mapping[str, Any]], searches: Sequence[Any]) -> None:
    seen: set[int] = set()
    for search in searches:
        diagnostics = getattr(search, "diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            continue
        if "batch" not in diagnostics:
            continue
        batch_diagnostics = diagnostics["batch"]
        if not isinstance(batch_diagnostics, Mapping):
            continue
        marker = id(batch_diagnostics)
        if marker in seen:
            continue
        seen.add(marker)
        destination.append(batch_diagnostics)


def _int_nested(mapping: Mapping[str, Any], section: str, field: str) -> int:
    section_value = mapping.get(section, {})
    if not isinstance(section_value, Mapping):
        return 0
    try:
        return int(section_value.get(field, 0))
    except (TypeError, ValueError):
        return 0


def _float_nested(mapping: Mapping[str, Any], section: str, field: str) -> float:
    section_value = mapping.get(section, {})
    if not isinstance(section_value, Mapping):
        return 0.0
    try:
        return float(section_value.get(field, 0.0))
    except (TypeError, ValueError):
        return 0.0


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
