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
    # §5.4.4 moves-left utility (MLH decisiveness). PRODUCTION = ON, two-sided,
    # with the final-move tie-break. The Rust Divergences mirror these; passing
    # them as divergence_overrides makes the lever controllable + auditable from
    # config instead of being baked into Divergences::production(). Set
    # moves_left_utility=False (or search_parity_mode=True) for the byte-identical
    # no-MLH baseline. ml_auto_disabled / the run-dir ml_auto_disabled.flag force
    # the lever off mid-run when the per-epoch head-health monitor trips.
    moves_left_utility: bool = True
    ml_weight: float = 0.03
    ml_scale: float = 32.0
    ml_q_gate: float = 0.6
    ml_two_sided: bool = True
    ml_final_pick: bool = True
    ml_final_pick_band: float = 0.05
    ml_auto_disabled: bool = False
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


# --- Multi-stage standalone evaluation (purely measurement) ------------------
#
# This is a SEPARATE, opt-in evaluator from the per-epoch ``EvaluationSection``
# lockstep arena above. It is run by a STANDALONE script (scripts/), NOT inside
# the training pipeline, so it never interrupts the live run. Its only product
# is a verdict LABEL (PROMOTE / REGRESS / INCONCLUSIVE) plus rolling ratings;
# it MUST NOT gate, promote, halt, or otherwise alter the run. The two
# ``*_gating_*`` / ``*_promotion_*`` knobs below default OFF and are wired to
# nothing that changes training — they exist only so a future, explicitly
# opted-in consumer could read them.
#
# Statistical design notes (the corrected, adversary-reviewed design):
#  - SealBot is the cross-lineage zero-point ONLY (pinned at 0 Elo); its depth
#    varies under GPU load, so it must NOT enter difference inference at full
#    weight (see ``sealbot_overdispersion``).
#  - Permanent anchors (BC prefit + ep5) never slide; the sliding bracket is the
#    nearest two fixed log-grid rungs BELOW the current epoch (NOT "vs prior").
#  - The 128 games/epoch are PAIRED (shared openings / common random numbers)
#    and scored pentanomially; pair-level SE + paired/effective counts feed the
#    Bradley-Terry likelihood. The BT fit must CONVERGE (max|grad| < tol) before
#    any covariance is computed.
#  - HONEST RESOLUTION: a single 128-game epoch resolves only ~100-120 Elo
#    (single-epoch SE(r_L - r_B) ~= 40-55 Elo). The ~15-20 Elo resolution is a
#    MULTI-EPOCH ROLLING ASYMPTOTE of the persisted pool, never a per-epoch
#    property. Stage B (SPRT) is a GROSS-regression triage, not a calibrated
#    5%/5% test.


@dataclass(frozen=True)
class MultiStageEvalOpponents:
    """Opponent roster for the deep (Stage C) eval and the rolling pool.

    Three roles, per the corrected design:
      * SealBot  -- cross-lineage zero-point / calibrator, pinned at 0 Elo.
      * permanent anchors -- never slide (BC prefit + ep5 by default).
      * sliding bracket -- the nearest ``bracket_size`` rungs of ``log_grid``
        strictly BELOW the current epoch (NOT the immediately-prior checkpoint).
    """

    # SealBot zero-point. ``sealbot_path`` falls back to $SEALBOT_PATH when None
    # (matches hexo_runner SealBotConfig.resolved_path). Disable when SealBot is
    # unavailable; the pool then floats relative to the permanent anchors.
    sealbot_enabled: bool = True
    sealbot_path: str | None = None
    sealbot_variant: str = "current"
    sealbot_time_limit: float = 0.05
    # PERMANENT anchors, as (label, checkpoint-path) pairs relative to the run
    # tree. These never slide. Paths use forward slashes (resolved by the runner
    # against the repo/run root). Defaults: the BC prefit and ep5.
    permanent_anchors: tuple[tuple[str, str], ...] = (
        ("bc_prefit", "runs/hexfield_bc_1/checkpoint_epoch2.pt"),
        ("ep5", "epoch_000005.pt"),
    )
    # Fixed log-grid of epochs the SLIDING bracket is drawn from. The bracket is
    # the nearest ``bracket_size`` rungs strictly below the current epoch.
    log_grid: tuple[int, ...] = (5, 10, 20, 40, 80, 160)
    bracket_size: int = 2


@dataclass(frozen=True)
class MultiStageEvalSprt:
    """Stage B SPRT screen parameters -- a coherent GROSS-REGRESSION TRIAGE.

    This is NOT a calibrated two-sided test of a small edge; it is a one-sided
    cheap filter that asks only "did the candidate grossly regress?". The two
    simple hypotheses are framed so the test and its label mapping agree:

      * H0 (``elo0 = 0``): the candidate is FINE -- Elo gap ~0 vs the screen
        opponent. This is the null we hope to keep.
      * H1 (``elo1 = -50``): the candidate GROSSLY REGRESSED -- a large negative
        Elo gap. ``winrate_from_elo`` makes ``p1 < 0.5 < p0``, so a record
        dominated by LOSSES drives the LLR up to ``upper`` and accepts H1.

    Label mapping (lives in multistage_eval._stage_b_sprt; stated here so the two
    files reconcile):

      * ``accept_h1`` -> ``"regress_suspected"`` (record favours the gross-
        regression hypothesis; flag for the deep eval to confirm).
      * ``accept_h0`` -> ``"ok"`` / escalate-to-deep (candidate looks fine; the
        calibrated verdict is still Stage C/D, never this screen).
      * ``continue`` -> ``"escalate"`` (undecided under the cap -> deep eval).

    PURE EVAL: the screen NEVER short-circuits Stage C and NEVER gates/promotes;
    Stage C/D (paired games + BT pool) is always the authoritative measurement.
    With a small ``max_games`` cap the honest expected-N near the indifference
    region (order ~285 decided games) means most non-gross candidates simply
    ``escalate`` rather than resolve here -- by design. ``elo0``/``elo1`` are the
    H0/H1 Elo bounds; ``alpha``/``beta`` the nominal error rates (advisory, given
    the cap); ``max_games`` caps the screen.
    """

    enabled: bool = True
    # H0: candidate is fine (Elo gap ~0). H1: candidate grossly regressed
    # (~-50 Elo). See class docstring for the accept_h0/accept_h1 -> label map.
    elo0: float = 0.0
    elo1: float = -50.0
    alpha: float = 0.05
    beta: float = 0.05
    max_games: int = 64


@dataclass(frozen=True)
class MultiStageEvalSection:
    """Standalone, opt-in multi-stage strength eval -- PURELY MEASUREMENT.

    Emits a verdict LABEL and updates a persisted, SealBot-pinned Bradley-Terry
    pool; never gates/promotes/halts the run. Disabled by default
    (``enabled=False``) and only ever invoked by a standalone script.
    """

    # Master switch. Off by default: the multi-stage eval is opt-in + standalone.
    enabled: bool = False
    # Stage C budget: paired games per epoch evaluated (shared openings). These
    # COMPOUND into the rolling pool across epochs.
    games_budget: int = 128
    # Run the standalone eval against every Nth produced checkpoint/epoch.
    every_n_epochs: int = 5
    # Search budget for eval games. Historically the reduced 128-visit eval
    # budget; kept for back-compat / a deliberately cheap screen. NOTE the
    # concurrent arena (eval_arena.play_checkpoint_match / play_sealbot_match)
    # batches games across the GPU, so FULL sims are now affordable — the
    # orchestrator runs eval at ``full_search_visits`` (below) by default, not
    # this value. Leave this for an explicit reduced-budget override.
    eval_visits: int = 128
    # Eval search budget. ``None`` -> use the production ``selfplay.search_visits``
    # (512); an int pins a specific budget. LOCKED to the production 512 for the
    # in-run comparable eval: default ``None`` so an omitted value can NEVER
    # silently under-shoot to a reduced budget — every eval, every epoch, every
    # opponent plays at 512. The deep eval (and the SPRT screen, when enabled) play
    # at this budget (threaded into Stage B + Stage C by the orchestrator's
    # _eval_visits). Pin an int only for a deliberate reduced-budget experiment.
    full_search_visits: int | None = None
    # EVAL-ONLY MCTS leaf-parallelism / virtual-loss batch. LOCKED 16 and UNIFORM
    # across all epochs + opponents so every measurement is comparable. Threaded
    # into the eval search calls ONLY (via the orchestrator's
    # _eval_virtual_batch_size); it does NOT touch SelfplayConfig.virtual_batch_size
    # (=4) — self-play strength is unchanged. Negligible, symmetric strength effect
    # at fixed visits; locked once purely for cross-epoch comparability.
    eval_virtual_batch_size: int = 16
    # Opening plies temperature-sampled to diversify PAIRED lines (shared opening
    # seed per pair => common random numbers across the two seat-swapped games).
    opening_plies: int = 8
    opening_temperature: float = 1.0
    # PRIMARY hypothesis: candidate L vs prior champion B, via the BT
    # difference-CI (includes the Cov_LB term). All OTHER opponent edges are
    # DESCRIPTIVE only (Wilson/Elo CIs, no significance verdict). When more than
    # one edge must carry a verdict, apply Bonferroni: per-edge alpha = 0.05/k.
    primary_alpha: float = 0.05
    bonferroni_correction: bool = True
    # SealBot's non-deterministic depth must not enter difference inference at
    # full weight: scale its edge's effective count by this over-dispersion
    # factor (< 1 down-weights). It stays the pinned zero-point regardless.
    sealbot_overdispersion: float = 0.5
    # Fraction of ``games_budget`` allocated to the SealBot zero-point pairing
    # (the rest is split evenly across the checkpoint opponents). Threaded into
    # ``allocate_budget`` by the orchestrator so the in-run split is config-driven.
    # Default 0.25 matches the historical allocate_budget default (the existing
    # 0.25-split tests stay green); the production config raises this to 0.5 for a
    # 1:1 SealBot-vs-checkpoint split (32 SealBot + 32 checkpoint at budget 64).
    sealbot_share: float = 0.25
    # Bradley-Terry convergence guard: ASSERT max|grad| < this before computing
    # covariance. The legacy fixed-step GD does NOT meet this (max|grad| ~0.30);
    # the corrected fit (Newton / scipy.optimize.minimize) must.
    bt_grad_tol: float = 1e-6
    bt_max_iters: int = 200
    # Persisted rolling pool so per-epoch edges COMPOUND. Relative to the run
    # diagnostics dir.
    pool_path: str = "diagnostics/eval_pool.json"
    # Verdict thresholds (Elo, on the BT difference r_L - r_B). The CI must clear
    # these to label PROMOTE / REGRESS; otherwise INCONCLUSIVE. NB: a single
    # epoch only resolves ~100-120 Elo -- tight thresholds resolve only as the
    # rolling pool accumulates.
    promote_elo_threshold: float = 0.0
    regress_elo_threshold: float = 0.0
    # L-2: the PRIMARY verdict compares the candidate to a STABLE reference, NOT
    # the immediately-prior (highly-correlated) checkpoint. The reference is the
    # highest checkpoint at least ``verdict_reference_lag`` epochs below the
    # candidate; 0 keeps the legacy immediately-prior behavior. A lag of 5 matches
    # ``every_n_epochs`` (the natural eval cadence) and the log-grid spacing, so a
    # contiguous ladder (candidate ep16) targets ~ep10/ep11 instead of ep15 -- a
    # genuinely de-correlated target. The immediately-prior checkpoint still
    # appears as a DESCRIPTIVE bracket edge, so its information is still pooled
    # into the BT fit; only the reported verdict target rests on the stable
    # reference. PURE EVAL -- this ONLY chooses the reported verdict target; it is
    # NOT a promotion/registry pointer and gates nothing.
    verdict_reference_lag: int = 5
    opponents: MultiStageEvalOpponents = field(default_factory=MultiStageEvalOpponents)
    sprt: MultiStageEvalSprt = field(default_factory=MultiStageEvalSprt)
    # --- Gating / promotion hooks: PURELY EVAL -> HARD OFF by default. --------
    # These exist so a future opted-in consumer COULD wire them, but in this
    # feature they are wired to NOTHING that alters the run. Leave False.
    eval_gating_enabled: bool = False
    eval_promotion_enabled: bool = False


@dataclass(frozen=True)
class HexfieldConfig:
    device: str = "cuda"
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    training: TrainingSection = field(default_factory=TrainingSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)
    multi_stage_eval: MultiStageEvalSection = field(default_factory=MultiStageEvalSection)

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


def _merge_multi_stage_eval(section: Mapping[str, Any]) -> "MultiStageEvalSection":
    """Merge the multi-stage eval section, recursing into the ``opponents`` and
    ``sprt`` sub-tables. A flat ``_merge`` cannot handle these because the
    dataclass fields hold dataclass instances, not dicts; missing sub-tables
    fall back to their dataclass defaults so an absent toml -> all defaults."""
    section = dict(section)
    nested = {
        "opponents": MultiStageEvalOpponents,
        "sprt": MultiStageEvalSprt,
    }
    merged: dict[str, Any] = {}
    for key, sub_cls in nested.items():
        if key in section:
            merged[key] = _merge(sub_cls, dict(section.pop(key)))
    # ``section`` now holds only the scalar fields; reuse the flat merge for the
    # unknown-key guard, then overlay the parsed sub-sections.
    return _merge(MultiStageEvalSection, {**section, **merged})


ML_AUTO_DISABLED_FLAG = "ml_auto_disabled.flag"


def build_divergence_overrides(sp: SelfplayConfig, *, disabled: bool = False) -> dict:
    """The §5.4.4 moves-left divergence knobs as a Rust ``divergence_overrides``
    dict, so the lever is driven by config (controllable + auditable) rather than
    baked into ``Divergences::production()``. When ``disabled`` — either the
    ``ml_auto_disabled`` config field or the run-dir heal-gate flag — the whole
    lever (descent bonus, two-sided branch, final pick) is forced off so a
    miscalibrated head stops steering search; the constants are still passed so a
    later re-enable uses the validated values. All values are concrete bool/float
    (never None) because ``resolve_divergences`` calls ``.extract()``."""
    off = bool(disabled or sp.ml_auto_disabled)
    return {
        "moves_left_utility": bool(sp.moves_left_utility) and not off,
        "ml_weight": float(sp.ml_weight),
        "ml_scale": float(sp.ml_scale),
        "ml_q_gate": float(sp.ml_q_gate),
        "ml_two_sided": bool(sp.ml_two_sided) and not off,
        "ml_final_pick": bool(sp.ml_final_pick) and not off,
        "ml_final_pick_band": float(sp.ml_final_pick_band),
    }


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
        multi_stage_eval=_merge_multi_stage_eval(dict(config.get("multi_stage_eval", {}))),
    )
