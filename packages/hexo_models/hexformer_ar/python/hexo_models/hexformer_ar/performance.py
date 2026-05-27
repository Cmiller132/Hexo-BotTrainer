"""Automatic performance calibration for Hexformer AR."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

import torch

from .config import HexformerConfig
from .input import SparseDecisionInput, collate_sparse_inputs
from .losses import hexformer_loss


def calibrate_hexformer(*, ctx: Any, components: Any) -> dict[str, Any]:
    trainer = components.model.trainer
    config = trainer.config
    perf = config.performance
    if not perf.calibrate:
        return {
            "status": "skipped",
            "model": "hexo_models.hexformer_ar",
            "reason": "performance calibration disabled",
            "training_batch_size": trainer.training_batch_size,
            "search_visits": trainer.search_visits,
        }

    model = components.model.model
    optimizer = components.model.optimizer
    device = trainer.device
    model.to(device)
    model_state = _clone_state_dict(model.state_dict())
    optimizer_state = _clone_optimizer_state(optimizer.state_dict()) if optimizer is not None else None
    try:
        inference = _benchmark_inference(
            model,
            config=config,
            device=device,
            candidates=perf.inference_batch_candidates,
            probe_batches=perf.probe_batches,
        )
        training = _benchmark_training(
            model,
            optimizer=optimizer,
            config=config,
            device=device,
            candidates=perf.training_batch_candidates,
            probe_batches=perf.probe_batches,
        )
    finally:
        model.load_state_dict(model_state)
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    selected_inference = max(inference, key=lambda item: float(item.get("positions_per_second", 0.0)), default={})
    selected_training = max(training, key=lambda item: float(item.get("positions_per_second", 0.0)), default={})
    result = {
        "status": "completed",
        "model": "hexo_models.hexformer_ar",
        "device": str(device),
        "selected_inference_batch_size": int(selected_inference.get("batch_size", 1)),
        "selected_training_batch_size": int(selected_training.get("batch_size", config.training.batch_size)),
        "inference": inference,
        "training": training,
    }
    path = ctx.diagnostics.write_json("hexformer_ar.performance_calibration.json", result)
    return {**result, "calibration_path": str(path)}


def _benchmark_inference(
    model: torch.nn.Module,
    *,
    config: HexformerConfig,
    device: torch.device,
    candidates: Sequence[int],
    probe_batches: int,
) -> list[dict[str, Any]]:
    model.eval()
    results = []
    for batch_size in candidates:
        batch = _synthetic_batch(config, int(batch_size), device)
        started = perf_counter()
        with torch.no_grad():
            for _ in range(max(1, int(probe_batches))):
                with torch.autocast(device_type=device.type, enabled=config.training.amp and device.type == "cuda"):
                    model(batch)
        elapsed = perf_counter() - started
        positions = int(batch_size) * max(1, int(probe_batches))
        results.append(
            {
                "batch_size": int(batch_size),
                "elapsed_seconds": elapsed,
                "positions_per_second": positions / max(elapsed, 1.0e-9),
            }
        )
    return results


def _benchmark_training(
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None,
    config: HexformerConfig,
    device: torch.device,
    candidates: Sequence[int],
    probe_batches: int,
) -> list[dict[str, Any]]:
    if optimizer is None:
        return []
    model.train()
    results = []
    for batch_size in candidates:
        batch = _synthetic_batch(config, int(batch_size), device)
        started = perf_counter()
        for _ in range(max(1, int(probe_batches))):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.training.amp and device.type == "cuda"):
                outputs = model(batch)
                loss, _components = hexformer_loss(outputs, batch)
            loss.backward()
            optimizer.step()
        elapsed = perf_counter() - started
        positions = int(batch_size) * max(1, int(probe_batches))
        results.append(
            {
                "batch_size": int(batch_size),
                "elapsed_seconds": elapsed,
                "positions_per_second": positions / max(elapsed, 1.0e-9),
            }
        )
    return results


def _synthetic_batch(config: HexformerConfig, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    arch = config.architecture
    samples = []
    for _ in range(max(1, int(batch_size))):
        samples.append(
            SparseDecisionInput(
                candidate_action_ids=tuple(range(arch.max_candidates)),
                candidate_features=torch.zeros((arch.max_candidates, arch.candidate_feature_dim), dtype=torch.float32),
                candidate_coords=torch.zeros((arch.max_candidates, 5), dtype=torch.float32),
                candidate_mask=torch.ones((arch.max_candidates,), dtype=torch.bool),
                stone_features=torch.zeros((min(arch.max_stones, 32), arch.stone_feature_dim), dtype=torch.float32),
                stone_coords=torch.zeros((min(arch.max_stones, 32), 5), dtype=torch.float32),
                stone_mask=torch.ones((min(arch.max_stones, 32),), dtype=torch.bool),
                window_features=torch.zeros((min(arch.max_windows, 32), arch.window_feature_dim), dtype=torch.float32),
                window_coords=torch.zeros((min(arch.max_windows, 32), 5), dtype=torch.float32),
                window_mask=torch.ones((min(arch.max_windows, 32),), dtype=torch.bool),
                local_input=torch.zeros((arch.local_input_channels, arch.local_crop_size, arch.local_crop_size), dtype=torch.float32),
                local_inputs=torch.zeros((arch.max_local_windows, arch.local_input_channels, arch.local_crop_size, arch.local_crop_size), dtype=torch.float32),
                local_window_coords=torch.zeros((arch.max_local_windows, 5), dtype=torch.float32),
                local_window_mask=torch.ones((arch.max_local_windows,), dtype=torch.bool),
                rel_edge_index=torch.zeros((0, 2), dtype=torch.long),
                rel_edge_features=torch.zeros((0, arch.rel_edge_feature_dim), dtype=torch.float32),
                rel_edge_mask=torch.zeros((0,), dtype=torch.bool),
                global_features=torch.zeros((arch.global_feature_dim,), dtype=torch.float32),
                policy_target=torch.full((arch.max_candidates,), 1.0 / max(1, arch.max_candidates), dtype=torch.float32),
                wdl_target=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
                distance_target=torch.tensor(0.0, dtype=torch.float32),
                threat_target=torch.zeros((arch.max_candidates,), dtype=torch.long),
                relevance_target=torch.zeros((arch.max_candidates,), dtype=torch.float32),
            )
        )
    return {key: value.to(device, non_blocking=True) for key, value in collate_sparse_inputs(samples).items()}


def _clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state.items()}


def _clone_optimizer_state(state: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().clone()
        elif isinstance(value, dict):
            cloned[key] = _clone_optimizer_state(value)
        elif isinstance(value, list):
            cloned[key] = [_clone_optimizer_state(item) if isinstance(item, dict) else item for item in value]
        else:
            cloned[key] = value
    return cloned
