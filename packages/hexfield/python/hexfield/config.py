"""hexfield run configuration (the [model.config] sections of a run toml).

A run's toml overrides these defaults; read the run toml for the values in
effect. Notable defaults: the moves-left divergences are on, root policy
temperature is 1.0 with no ramp, and FPU noise-zeroing is off.
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
    # Dynamic c_puct + LCB knobs, read by resolve_divergences.
    # c_for(N) = c_puct + c_scale*ln((N + c_base) / c_base); visit_scaled_c_puct
    # gates the log term; lcb_z is the LCB z-score.
    c_scale: float = 0.45
    c_base: float = 500.0
    visit_scaled_c_puct: bool = True
    lcb_z: float = 1.6
    # Six search divergences, on by default for self-play, individually
    # controllable from config. Applied only when search_parity_mode is False;
    # the parity path (search_parity_mode=True) ignores them.
    nucleus_f64: bool = True
    new_child_fpu: bool = True
    lazy_widening: bool = True
    clean_root_prior_cache: bool = True
    dirichlet_shaped: bool = True
    pruned_dynamic_cpuct: bool = True
    # Root policy temperature ramp; defaults disable the ramp (temperature 1.0).
    root_policy_temperature: float = 1.0
    root_policy_temperature_early: float = 0.0
    root_policy_temperature_halflife: float = 0.0
    # root_fpu_reduction is the root FPU reduction (default 0.0).
    # root_fpu_zero_under_noise gates a noise-conditioned FPU branch used only by
    # the parity path; default False leaves that branch inactive.
    root_fpu_reduction: float = 0.0
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
    # Moves-left utility: default on, two-sided, with the final-move tie-break.
    # Passed to Rust as divergence_overrides. Set moves_left_utility=False (or
    # search_parity_mode=True) for the no-MLH baseline. ml_auto_disabled, or the
    # run-dir ml_auto_disabled.flag, forces the lever off mid-run.
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
    # --- Adaptive grad-clip --------------------------------------------------
    adaptive_clip: bool = True
    clip_c: float = 1.75
    clip_ema_decay: float = 0.99
    clip_warmup_steps: int = 50
    # Loss weights; defaults match the constants in losses.py.
    policy_weight: float = 1.0
    value_weight: float = 1.0
    opp_policy_weight: float = 0.25
    short_term_value_weight: float = 0.1
    moves_left_weight: float = 0.1
    q_head_weight: float = 0.1
    # Loss weight for the train-only soft_policy head (CE against the
    # (visit_policy + 1e-7)^(1/4) renormalized soft target). Mirrors
    # losses.SOFT_POLICY_WEIGHT; hardcoded here because config.py does not import
    # losses.
    soft_policy_weight: float = 8.0
    # Policy-surprise self-CE reweight.
    policy_surprise_uniform_fraction: float = 0.5
    policy_surprise_max_weight: float = 8.0
    warmup_steps: int = 0
    shuffle_keep_target_rows: int = 300_000
    # Replay-buffer window knobs. taper_window_exponent and
    # expand_window_per_row are dimensionless; min_rows/scale/targets are sized
    # to hexfield's row stream.
    shuffle_min_rows: int = 20_000
    shuffle_taper_window_exponent: float = 0.65
    shuffle_expand_window_per_row: float = 0.4
    shuffle_taper_window_scale: float = 20_000.0
    validation_fraction: float = 0.0
    train_samples_per_epoch: int = 100_000
    max_train_bucket_per_new_data: float = 8.0
    max_train_bucket_size: float = 500_000.0
    no_repeat_files: bool = False
    expand_backend: str = "serial"
    expand_workers: int = 0


@dataclass(frozen=True)
class EvaluationSection:
    games_per_epoch: int = 16
    eval_visits: int = 128
    # Run the H2H arena every Nth epoch.
    eval_every: int = 1


# --- Multi-stage standalone evaluation ---------------------------------------
#
# A separate, opt-in evaluator from the per-epoch ``EvaluationSection`` arena
# above. Run by a standalone script (scripts/), not inside the training
# pipeline. Its product is a verdict label (PROMOTE / REGRESS / INCONCLUSIVE)
# plus rolling ratings; it does not gate, promote, or halt the run. The
# ``*_gating_*`` / ``*_promotion_*`` knobs below default off and are not wired
# to anything that changes training.
#
# Statistical design:
#  - SealBot is the cross-lineage zero-point, pinned at 0 Elo. Its depth varies
#    under GPU load, so its edge is down-weighted in difference inference (see
#    ``sealbot_overdispersion``).
#  - Permanent anchors (BC prefit + ep5) never slide; the sliding bracket is the
#    nearest two fixed log-grid rungs below the current epoch.
#  - Games per epoch are paired (shared openings / common random numbers) and
#    scored pentanomially; pair-level SE and paired/effective counts feed the
#    Bradley-Terry likelihood. The BT fit must converge (max|grad| < tol) before
#    covariance is computed.
#  - A single 128-game epoch resolves roughly 100-120 Elo (single-epoch
#    SE(r_L - r_B) ~= 40-55 Elo); finer resolution comes from the persisted pool
#    accumulating across epochs. Stage B (SPRT) is a gross-regression triage,
#    not a calibrated two-sided test.


@dataclass(frozen=True)
class MultiStageEvalOpponents:
    """Opponent roster for the deep (Stage C) eval and the rolling pool.

    Three roles:
      * SealBot -- cross-lineage zero-point / calibrator, pinned at 0 Elo.
      * permanent anchors -- never slide (BC prefit + ep5 by default).
      * sliding bracket -- the nearest ``bracket_size`` rungs of ``log_grid``
        strictly below the current epoch.
    """

    # SealBot zero-point. ``sealbot_path`` falls back to $SEALBOT_PATH when None
    # (matches hexo_runner SealBotConfig.resolved_path). When disabled, the pool
    # floats relative to the permanent anchors.
    sealbot_enabled: bool = True
    sealbot_path: str | None = None
    sealbot_variant: str = "current"
    sealbot_time_limit: float = 0.05
    # Permanent anchors, as (label, checkpoint-path) pairs. These never slide.
    # Paths use forward slashes and are resolved by ``_resolve_anchor_path``
    # against the repo/run root; an absolute path is used as-is. The env var
    # ``HEXFIELD_ANCHOR_ROOTS`` (os.pathsep-separated dirs) prepends extra search
    # roots. An unresolved anchor is recorded in roster.dropped_anchors.
    permanent_anchors: tuple[tuple[str, str], ...] = (
        ("bc_prefit", "runs/hexfield_bc_1/checkpoint_epoch2.pt"),
        ("ep5", "epoch_000005.pt"),
    )
    # Fixed log-grid of epochs the SLIDING bracket is drawn from. The bracket is
    # the nearest ``bracket_size`` rungs strictly below the current epoch.
    log_grid: tuple[int, ...] = (5, 10, 20, 40, 80, 160)
    bracket_size: int = 2
    # Labels of opponents trained at the radius-8 legality era. The support
    # radius is a process-global read once per process, so every opponent is
    # featurized at the live HEXFIELD_SUPPORT_RADIUS; a radius-8-era net run at a
    # smaller radius is out-of-distribution. Edges to these opponents are
    # annotated ``featurized_ood`` and excluded from the pinned BT zero-point;
    # they still participate descriptively.
    radius8_opponents: tuple[str, ...] = ("bc_prefit",)


@dataclass(frozen=True)
class MultiStageEvalSprt:
    """Stage B SPRT screen parameters -- a one-sided gross-regression triage.

    A one-sided filter that tests whether the candidate grossly regressed. The
    two simple hypotheses:

      * H0 (``elo0 = 0``): Elo gap ~0 vs the screen opponent.
      * H1 (``elo1 = -50``): large negative Elo gap. ``winrate_from_elo`` makes
        ``p1 < 0.5 < p0``, so a record dominated by losses drives the LLR up to
        ``upper`` and accepts H1.

    Label mapping (implemented in multistage_eval._stage_b_sprt):

      * ``accept_h1`` -> ``"regress_suspected"``.
      * ``accept_h0`` -> ``"ok"`` / escalate-to-deep.
      * ``continue`` -> ``"escalate"`` (undecided under the cap).

    The screen does not short-circuit Stage C and does not gate or promote;
    Stage C/D (paired games + BT pool) is the authoritative measurement. Under a
    small ``max_games`` cap, expected-N near the indifference region is on the
    order of ~285 decided games, so most non-gross candidates ``escalate``
    rather than resolve here. ``elo0``/``elo1`` are the H0/H1 Elo bounds;
    ``alpha``/``beta`` the nominal error rates; ``max_games`` caps the screen.
    """

    enabled: bool = True
    # H0: Elo gap ~0. H1: gross regression (~-50 Elo). See the class docstring
    # for the accept_h0/accept_h1 -> label map.
    elo0: float = 0.0
    elo1: float = -50.0
    alpha: float = 0.05
    beta: float = 0.05
    max_games: int = 64


@dataclass(frozen=True)
class MultiStageEvalSection:
    """Standalone, opt-in multi-stage strength eval.

    Emits a verdict label and updates a persisted, SealBot-pinned Bradley-Terry
    pool; does not gate, promote, or halt the run. Disabled by default
    (``enabled=False``) and invoked only by a standalone script.
    """

    # Master switch. Off by default.
    enabled: bool = False
    # Stage C budget: paired games per epoch (shared openings). Compound into the
    # rolling pool across epochs.
    games_budget: int = 128
    # Run the standalone eval against every Nth produced checkpoint/epoch.
    every_n_epochs: int = 5
    # Reduced-budget search visits for eval games. The orchestrator runs eval at
    # ``full_search_visits`` (below) by default; this value is used only as an
    # explicit reduced-budget override.
    eval_visits: int = 128
    # Eval search budget. ``None`` uses ``selfplay.search_visits`` (512); an int
    # pins a specific budget. Applied to the deep eval and the SPRT screen (when
    # enabled) via the orchestrator's _eval_visits.
    full_search_visits: int | None = None
    # Eval-only MCTS leaf-parallelism / virtual-loss batch, applied to the eval
    # search calls via the orchestrator's _eval_virtual_batch_size. Does not
    # affect SelfplayConfig.virtual_batch_size (=4).
    eval_virtual_batch_size: int = 16
    # Opening plies temperature-sampled to diversify paired lines (shared opening
    # seed per pair => common random numbers across the two seat-swapped games).
    opening_plies: int = 8
    opening_temperature: float = 1.0
    # Primary hypothesis: candidate L vs reference B, via the BT difference-CI
    # (includes the Cov_LB term). Other opponent edges are descriptive only
    # (Wilson/Elo CIs, no significance verdict). With bonferroni_correction and
    # more than one verdict-carrying edge, per-edge alpha = primary_alpha / k.
    primary_alpha: float = 0.05
    bonferroni_correction: bool = True
    # Over-dispersion factor scaling SealBot's edge effective count in difference
    # inference (< 1 down-weights). SealBot stays the pinned zero-point.
    sealbot_overdispersion: float = 0.5
    # Fraction of ``games_budget`` allocated to the SealBot pairing; the rest is
    # split evenly across the checkpoint opponents. Applied by the orchestrator's
    # ``allocate_budget``.
    sealbot_share: float = 0.25
    # Bradley-Terry convergence guard: max|grad| must be below this before
    # covariance is computed.
    bt_grad_tol: float = 1e-6
    bt_max_iters: int = 200
    # Persisted rolling pool, relative to the run diagnostics dir.
    pool_path: str = "diagnostics/eval_pool.json"
    # Verdict thresholds (Elo, on the BT difference r_L - r_B). The CI must clear
    # these to label PROMOTE / REGRESS; otherwise INCONCLUSIVE.
    promote_elo_threshold: float = 0.0
    regress_elo_threshold: float = 0.0
    # The primary verdict compares the candidate to the highest checkpoint at
    # least ``verdict_reference_lag`` epochs below it; 0 uses the
    # immediately-prior checkpoint. The immediately-prior checkpoint still
    # appears as a descriptive bracket edge and is pooled into the BT fit; this
    # only selects the reported verdict target and gates nothing.
    verdict_reference_lag: int = 5
    opponents: MultiStageEvalOpponents = field(default_factory=MultiStageEvalOpponents)
    sprt: MultiStageEvalSprt = field(default_factory=MultiStageEvalSprt)
    # Gating / promotion hooks: off by default and not wired to anything that
    # alters the run.
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
    """Build the Rust ``divergence_overrides`` dict from ``sp``.

    Three groups of keys:
      - moves-left knobs: the boolean levers are gated by ``off`` (set when
        ``disabled`` is passed or ``sp.ml_auto_disabled`` is set); the numeric
        constants are always passed.
      - dynamic c_puct / LCB knobs (c_scale, c_base, visit_scaled_c_puct,
        lcb_z), read by ``resolve_divergences``.
      - six search divergences (nucleus_f64, new_child_fpu, lazy_widening,
        clean_root_prior_cache, dirichlet_shaped, pruned_dynamic_cpuct).

    This dict is applied on top of the base selected by ``search_parity_mode``
    inside ``resolve_divergences``. All values are concrete bool/float (never
    None) because ``resolve_divergences`` calls ``.extract()``."""
    off = bool(disabled or sp.ml_auto_disabled)
    return {
        # Moves-left utility (gated by off).
        "moves_left_utility": bool(sp.moves_left_utility) and not off,
        "ml_weight": float(sp.ml_weight),
        "ml_scale": float(sp.ml_scale),
        "ml_q_gate": float(sp.ml_q_gate),
        "ml_two_sided": bool(sp.ml_two_sided) and not off,
        "ml_final_pick": bool(sp.ml_final_pick) and not off,
        "ml_final_pick_band": float(sp.ml_final_pick_band),
        # Dynamic c_puct + LCB.
        "c_scale": float(sp.c_scale),
        "c_base": float(sp.c_base),
        "visit_scaled_c_puct": bool(sp.visit_scaled_c_puct),
        "lcb_z": float(sp.lcb_z),
        # Search divergences.
        "nucleus_f64": bool(sp.nucleus_f64),
        "new_child_fpu": bool(sp.new_child_fpu),
        "lazy_widening": bool(sp.lazy_widening),
        "clean_root_prior_cache": bool(sp.clean_root_prior_cache),
        "dirichlet_shaped": bool(sp.dirichlet_shaped),
        "pruned_dynamic_cpuct": bool(sp.pruned_dynamic_cpuct),
    }


def parse_hexfield_config(config: Mapping[str, Any]) -> HexfieldConfig:
    config = dict(config or {})
    # Reject unknown top-level keys so a typo'd section name raises instead of
    # being dropped.
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
