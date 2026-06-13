"""hexfield run configuration (the [model.config] sections of a run toml).

Defaults are the PRODUCTION values: §5.4 divergences ON, the §5.1 quarantined
knobs OFF (no FPU noise-zeroing, root policy temperature 1.0 / no ramp), other
search knobs mirroring production main_4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SelfplayConfig:
    search_visits: int = 512
    pcr_full_proportion: float = 0.33
    pcr_fast_visits: int = 128
    active_games: int = 128
    c_puct: float = 1.5
    virtual_batch_size: int = 4
    flush_target: int = 256
    active_root_limit: int = 256
    root_dirichlet_total_alpha: float = 10.83
    root_dirichlet_noise_fraction: float = 0.25
    # QUARANTINED (spec §5.1): defaults off.
    root_policy_temperature: float = 1.0
    root_policy_temperature_early: float = 0.0
    root_policy_temperature_halflife: float = 0.0
    root_fpu_zero_under_noise: bool = False
    fpu_reduction: float = 0.2
    virtual_loss: float = 1.0
    widening_policy_mass: float = 0.95
    widening_max_children: int = 96
    widening_min_children: int = 2
    forced_playout_k: float = 2.0
    policy_init_fraction: float = 0.25
    policy_init_avg_plies: float = 4.0
    policy_init_max_plies: int = 8
    policy_init_temperature: float = 1.4
    temperature: float = 1.0
    temperature_floor: float = 0.1
    temperature_halflife_plies: float = 30.0
    max_game_plies: int = 512
    tss_enabled: bool = True
    search_parity_mode: bool = False
    cache_max_states: int = 262_144


@dataclass(frozen=True)
class TrainingSection:
    batch_rows: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 0  # fresh-init runs warm-start from the BC prefit
    shuffle_keep_target_rows: int = 300_000


@dataclass(frozen=True)
class EvaluationSection:
    games_per_epoch: int = 16
    eval_visits: int = 128
    # Run the H2H arena every Nth epoch (the lockstep arena is single-game and
    # slow; over a long run, evaluating every epoch dominates wall-clock).
    eval_every: int = 1


@dataclass(frozen=True)
class HexfieldConfig:
    device: str = "cuda"
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    training: TrainingSection = field(default_factory=TrainingSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)

    def temperature_by_ply(self) -> list[float]:
        sp = self.selfplay
        out = []
        for ply in range(self.selfplay.max_game_plies):
            t = sp.temperature * (0.5 ** (ply / max(sp.temperature_halflife_plies, 1e-9)))
            out.append(max(sp.temperature_floor, t))
        return out


def _merge(cls, section: Mapping[str, Any]):
    known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**section)


def parse_hexfield_config(config: Mapping[str, Any]) -> HexfieldConfig:
    config = dict(config or {})
    # Reject unknown TOP-LEVEL keys, not just unknown sub-keys: a typo'd section
    # (e.g. [model.config.slfplay]) would otherwise be silently dropped, leaving
    # production knobs at their defaults with no error.
    known_top = set(HexfieldConfig.__dataclass_fields__)
    unknown_top = set(config) - known_top
    if unknown_top:
        raise ValueError(f"unknown HexfieldConfig keys: {sorted(unknown_top)}")
    return HexfieldConfig(
        device=str(config.get("device", "cuda")),
        selfplay=_merge(SelfplayConfig, dict(config.get("selfplay", {}))),
        training=_merge(TrainingSection, dict(config.get("training", {}))),
        evaluation=_merge(EvaluationSection, dict(config.get("evaluation", {}))),
    )
