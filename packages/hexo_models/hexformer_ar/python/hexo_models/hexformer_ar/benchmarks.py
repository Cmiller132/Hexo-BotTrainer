"""Benchmark and ablation contracts for Hexformer AR evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    name: str
    primary_metric: str
    secondary_metric: str
    validates: str
    metadata: Mapping[str, object] | None = None

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata or {})
        return payload


@dataclass(frozen=True, slots=True)
class AblationSpec:
    name: str
    compare: str
    expected_signal: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("self_play_elo", "elo_or_win_rate", "confidence_interval", "overall playing strength"),
    BenchmarkSpec("tactical_suite", "solve_rate", "top_k_recall", "forced threat recognition"),
    BenchmarkSpec("search_efficiency", "visits_to_win_or_latency", "candidate_count", "planning efficiency"),
    BenchmarkSpec("calibration", "brier_or_ece", "value_mse", "policy/value target fidelity"),
    BenchmarkSpec("large_span_generalization", "large_span_win_rate", "policy_entropy", "relative sparse extrapolation"),
    BenchmarkSpec("d6_symmetry", "orbit_consistency_error", "policy_gap", "equivariance of sparse tensors and IDs"),
)


DEFAULT_ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec("local_only_vs_hybrid", "HexaConv CNN vs HexaConv plus GraphGPS", "hybrid improves long-range tactics"),
    AblationSpec("pointer_vs_crop_policy", "candidate pointer vs flattened crop logits", "pointer wastes less probability mass"),
    AblationSpec("relative_vs_absolute_positions", "relative axial/cube vs absolute crop index", "relative generalizes farther"),
    AblationSpec("threat_auxiliaries", "with vs without threat/relevance heads", "solve rate and search efficiency improve"),
    AblationSpec("opponent_policy", "with vs without opponent-policy head", "midgame policy regularization improves"),
    AblationSpec("window_tokens", "with vs without tactical window tokens", "faster tactical learning"),
    AblationSpec("d6_augmentation", "D6 sparse augmentation on vs off", "symmetry consistency improves"),
    AblationSpec("candidate_frontier", "legal-only vs legal+tactical+neighbors", "branching factor falls without missing wins"),
)


def benchmark_plan() -> dict[str, object]:
    return {
        "benchmarks": [item.to_json() for item in DEFAULT_BENCHMARKS],
        "ablations": [item.to_json() for item in DEFAULT_ABLATIONS],
    }
