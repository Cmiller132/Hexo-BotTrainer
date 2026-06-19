# HEXFIELD main_4 / KataGo-faithful build — change log

Branch: `claude/hexfield-main4`  (worktree `/mnt/e/Hexo-BotTrainer-main4`,
Windows `E:\Hexo-BotTrainer-main4`). PREPARE-ONLY: this branch does **not**
launch main_4 and does **not** touch the live main_3 run. All edits are
uncommitted working-tree changes (no commits yet); read the diff with
`git diff` (there are no commits between `main` and the branch tip).

This document enumerates every change on the branch, cited to `file:function`,
with the KataGo target → what we had → the gap/bug → the change → the
faithful-vs-adapted call for each mechanism.

---

## 1. Goal + the four binding scope decisions

**Goal.** Make hexfield's Rust MCTS core and Python training loop KataGo-faithful
for the new `hexfield_main_4` run, fixing the confirmed search divergences/bugs
and adding the KataGo auxiliary **soft policy** target (new head + loss + soft
target), while keeping the M5/M6 parity/differential harness byte-identical.

**Owner scope decisions (binding):**
1. **PREPARE ONLY** — do not launch main_4, do not stop main_3.
2. **Include the KataGo auxiliary soft policy target now** (new model head + loss
   + soft target).
3. **Default to KataGo-faithful behavior; keep the M5/M6 parity/differential
   scaffolding.** Remove only dead/legacy fallbacks, not the parity harness.
4. **Build + CPU tests in an isolated venv; never touch the shared venv or GPU.**

**Files changed (working tree):**

| File | What |
|---|---|
| `packages/hexfield/rust/src/tree.rs` | shaped Dirichlet, clean-root-prior cache + reuse-reset, new-child FPU, lazy/live widening, nucleus f64+sentinel, six divergence flags, Q-scale rationale comment, 15 unit tests |
| `packages/hexfield/rust/src/search.rs` | first-class `root_fpu_reduction`, wire `dirichlet_shaped` into root-noise builders, dynamic-c_puct in pruned target, resolve six new divergence overrides, PyO3 signatures |
| `packages/hexfield/python/hexfield/config.py` | `SelfplayConfig` c_scale/c_base/visit_scaled_c_puct/lcb_z/root_fpu_reduction + six divergence flags; `TrainingSection.soft_policy_weight`; `build_divergence_overrides` emits the new keys |
| `packages/hexfield/python/hexfield/model.py` | train-only `soft_policy_conv`/`soft_policy_head`; emit `soft_policy` in `forward` (not serve) |
| `packages/hexfield/python/hexfield/losses.py` | `SOFT_POLICY_WEIGHT=8.0`; `soft_policy` CE component in `hexfield_loss` |
| `packages/hexfield/python/hexfield/batching.py` | derive the soft target in `collate_training` (backend-agnostic) |
| `packages/hexfield/python/hexfield/trainer.py` | pass `soft_policy_weight`; fix grad-norm bucket predicate (`bias_table`→`bias_tables.*`) |
| `packages/hexfield/python/hexfield/selfplay.py` | pass `root_fpu_reduction` to the session |
| `packages/hexfield/python/hexfield/checkpoints.py` | tolerant `warm_start_into` for the fresh soft head on the initialize_from / BC path; strict resume untouched |
| `configs/hexfield_main_4.toml` | new run config (copy of main_3 + new knobs) |
| `tests/test_hexfield_soft_policy.py` | new pure-CPU torch tests for the soft target / head / loss / warm start |

---

## 2. Per-mechanism: KataGo target → what we had → gap → change → faithful call

### 2.1 Q-scale + c_puct radius rationale (documented, math unchanged)

- **KataGo target:** PUCT adds value (Q) and exploration (U) on the *same*
  utility radius; KataGo's `cpuctExploration` (1.0) is calibrated to its utility
  width (~1.4 once score terms are included).
- **What we had:** edge `Q = value_sum/visits` on the symmetric `[-1,1]` interval
  (width 2.0), no per-node normalization; we ship `c_puct ≈ 1.5`. The rationale
  was implicit / undocumented.
- **Gap:** the "arbitrary/unusual" thing was the *missing rationale*, not the
  math. Hexo has no territory/score term, so the faithful KataGo radius collapses
  to `winLossUtilityFactor = 1.0` → width 2.0. `c_puct = 1.5` is the
  radius-rescaled KataGo 1.0 (`1.0 * 2.0/1.4 ≈ 1.43 ≈ 1.5`); `c/radius` is
  near-identical (KataGo `1.0/1.4 = 0.714` vs hexfield `1.5/2.0 = 0.75`).
- **Change:** `tree.rs` top-of-file doc-block + two named constants
  `Q_UTILITY_WIDTH = 2.0` and `KATAGO_UTILITY_WIDTH = 1.4` (documentation only;
  not used in the math). No numeric change.
- **Faithful vs adapted:** **Faithful.** Keep Q on `[-1,1]`. Remapping to `[0,1]`
  would force re-deriving `fpu_reduction (0.2)`, the moves-left weight/gate,
  `lcb_z`, the forced-playout U comparison, and the recorded target — all
  calibrated against the `[-1,1]` radius. We deliberately did **not** remap.

### 2.2 Dynamic c_puct (wiring gap closed)

- **KataGo target:** `cPUCTExploration + cPUCTExplorationLog * ln((N + cPUCTExplorationBase)/cPUCTExplorationBase)`.
- **What we had:** `tree.rs::RustSearch::c_for` already computes
  `c_puct + c_scale*ln((N + c_base)/c_base)` with `c_scale=0.45`, `c_base=500`,
  gated by `visit_scaled_c_puct` — structurally KataGo's formula. The Rust
  `resolve_divergences` already *reads* `c_scale`/`c_base`/`visit_scaled_c_puct`/
  `lcb_z` from the overrides dict, but Python's `build_divergence_overrides`
  never *emitted* them (the established wiring gap), so they were implicit
  baked-in `production()` defaults.
- **Change:** `config.py::SelfplayConfig` now carries first-class
  `c_scale=0.45`, `c_base=500.0`, `visit_scaled_c_puct=True`, `lcb_z=1.6`;
  `config.py::build_divergence_overrides` emits all four. Values equal the prior
  baked-in defaults, so **behavior is unchanged** — only auditability/control is
  new. No Rust change needed (the reader already existed).
- **Faithful vs adapted:** **Faithful + now auditable.** (Owner-open: KataGo
  *self-play* uses `cpuctExplorationLog=0.0`; `0.45` is the analysis/strong-play
  value hexfield has shipped. We keep `0.45` to match current behavior — see
  open decision in §9.)

### 2.3 Shaped Dirichlet + reuse-compounding fix

- **KataGo target:** `computeDirichletAlphaDistribution` (searchhelpers.cpp): a
  *shaped* per-move alpha from the clean NN policy, mixed at fraction 0.25 after
  the root policy temperature, computed from the clean policy.
- **What we had:** a *flat* symmetric `Dir(total_alpha/count)`
  (`tree.rs::dirichlet_samples`). `total_alpha=10.83` and `fraction=0.25` already
  matched KataGo; only the *shape* was wrong. Separately, the noise/temperature
  were applied **in place** to `edge.prior`/`candidate.prior`, so on a
  reused/promoted root the mix re-powered/re-mixed *already-noised* priors →
  compounding across plies (bug [4]).
- **Change:**
  - `tree.rs::shaped_alpha` implements `computeDirichletAlphaDistribution`
    verbatim: `a[i]=log(min(0.01,p[i])+1e-20)`; subtract the log-mean and clamp
    to `max(0,·)`; if `sum≤0` uniform `1/n`, else `0.5*(a[i]/sum + 1/n)`.
  - `tree.rs::shaped_dirichlet_samples` draws per-move concentration
    `shaped_alpha(clean) * total_alpha` through the same `DirichletSampler` /
    seed stream and returns a sum-to-1 vector (same contract as the flat path).
  - **Clean cache (bug [4] fix):** `RustSearch.clean_root_priors:
    Option<HashMap<PackedCoord,f32>>` caches the **post-temp, pre-noise** policy.
    `owned_root_from_evaluation` populates it on first setup;
    `apply_root_policy_temperature` and `apply_root_dirichlet_noise`
    (`capture_clean_root_priors` / `reset_root_priors_from_clean_cache`) reset
    edge/candidate priors from the cache *before* re-applying temp/noise, so noise
    never compounds. The shaped-alpha input is fed from this clean cache.
    `advance_root` clears the cache (`self.clean_root_priors = None`) so a
    promoted root re-captures its own clean priors.
  - `RootDirichletNoise.shaped: bool` carries the flag from the call site;
    `search.rs::root_noise` / `root_noise_exact` set it from
    `divergences.dirichlet_shaped`.
  - **visible_total scaling note:** when the clean cache is active the priors sum
    to 1 after reset, so the existing `visible_total` mix scaling is ~1.0 and the
    mix matches KataGo's sum-to-1 contract (documented at the call site). We
    normalize-via-reset rather than special-case the scaling.
- **Faithful vs adapted:** **Faithful** (shape verbatim; defaults already
  matched; reuse-reset makes the mix sum-to-1 like KataGo).

### 2.4 Root FPU (spec correction — first-class `rootFpuReductionMax`)

- **KataGo target (corrected):** modern KataGo has **no** "zero FPU under noise"
  branch. It uses a separate `rootFpuReductionMax` that **self-play sets to 0.0**.
- **What we had:** `search.rs::root_fpu_for` zeroed FPU only at noised Full roots
  when the quarantined `root_fpu_zero_under_noise` knob was set
  (default false). The infra `RustNode.root_fpu_reduction` /
  `set_root_fpu_reduction` already existed and was used in selection.
- **Change:** added a first-class `root_fpu_reduction: Option<f32>` to both PyO3
  session constructors (`search`, `run_continuous`) and to
  `ContinuousMovePolicy`. `root_fpu_for` returns the configured value when
  `Some` (applies to every move class — the root descent always uses the
  root-specific reduction); otherwise it falls through to the legacy
  noise-conditioned branch (parity). The lockstep `search` path likewise prefers
  the explicit value, validated `>= 0` via `validate_nonnegative_f32`.
  `config.py::SelfplayConfig.root_fpu_reduction=0.0` (KataGo self-play default)
  is threaded through `selfplay.py`.
- **Faithful vs adapted:** **Faithful (spec correction).** Production self-play
  uses `root_fpu_reduction=0.0`. The legacy noise-conditioned branch is **kept,
  not deleted** (parity path / golden vectors); `root_fpu_zero_under_noise` stays
  `false` so it is inert in production.

### 2.5 No-widening: nucleus sentinel + live `can_widen` + candidate FPU

- **KataGo target:** no progressive widening — every legal child is a selectable
  candidate; FPU (`getNewExploreSelectionValue == fpu + U`) is the sole gate, and
  forced playouts are self-limiting.
- **What we had:** progressive widening via a `max_eligible_children` cap derived
  from a nucleus (cumulative-mass) count, plus three confirmed bugs (§3).
- **Change (all gated behind `lazy_widening` + `new_child_fpu`, coupled):**
  - **Live `can_widen`** (`tree.rs::select_or_materialize_edge`): replaced
    `edges.len() < max_eligible_children` with
    `self.nodes[node_id].peek_next_candidate().is_some()` — eligibility is "an
    unexpanded candidate exists." This subsumes the frozen-cap bug **and** removes
    the nucleus truncation as the eligibility gate (FPU becomes the sole gate).
  - **New-child FPU** (`tree.rs::new_child_score`): a materialized (visits==0)
    candidate scores `value_or_fpu(parent_value, fpu_reduction) + prior*scale`
    (KataGo `fpu + U`, with the `/(1+0)=/1` denominator), matching an existing
    unvisited edge exactly. The legacy U-only `prior*scale` is kept under the flag
    (parity). This is **required together with** lazy widening so the lifted cap
    never runs with U-only scoring (the sign-of-parent-value bias would flip
    selection otherwise).
  - **Nucleus f64 + sentinel** (`tree.rs::nucleus_count_values`): accumulate the
    cumulative mass in f64 and short-circuit to "take all" (`hi`) when
    `widening.mass >= 1.0`. The legacy f32 loop (parity) is kept verbatim. This
    keeps the nucleus count correct for the `can_widen`-off (parity) path; under
    `lazy_widening` the nucleus is no longer the eligibility gate.
  - **Why forced playouts stay bounded:** `n_forced = sqrt(k*P*N)` with `k=2.0`
    (already KataGo-faithful) self-limits — at `N=512`, `n_forced = 32*sqrt(P)`,
    so only `P > ~0.001` priors ever earn a forced visit. They are pruned from
    the recorded target. We keep forced playouts unchanged.
- **Faithful vs adapted:** **Faithful** (FPU-as-sole-gate, lazy materialization
  via `peek_next_candidate`, forced playouts unchanged). The `max_eligible_children`
  field and the nucleus cap are retained for the parity path (deferred deletion,
  §4).

### 2.6 Pruned-target c_puct (selection/target consistency)

- **KataGo target:** the exported playout/visit policy target should be derived
  consistently with the search that produced it.
- **What we had:** selection used `c_for(root.visits)` but the recorded-target
  forced-playout pruning used **static** `c_puct`
  (`search.rs::pruned_visit_policy`, `explore = c_puct * sqrt(N)`) — an
  export-only divergence (bug [5]).
- **Change:** `tree.rs::RustSearch::effective_pruning_c_puct(c_puct, root_visits)`
  returns `c_for(...)` when `pruned_dynamic_cpuct` is on, else static `c_puct`.
  `search.rs::build_search_result_payloads` computes
  `effective_c = search.effective_pruning_c_puct(c_puct, root.visits)` and passes
  it into `pruned_visit_policy`. Low blast radius (only the exported target
  weights, not live search).
- **Faithful vs adapted:** **Faithful** (target now matches selection's dynamic c).

### 2.7 KataGo auxiliary soft policy head (new)

- **KataGo target:** a second policy output trained to predict a *softened*
  version of the visit policy: `target_soft = (target_policy + 1e-7)^(1/T)`
  renormalized, with `T=4` (exponent 0.25), at a small auxiliary loss weight
  (`-soft-policy-weight-scale` default 8.0). KataGo emits it as an extra **output
  channel** of its single multi-channel policy head.
- **What we had:** no soft head. Hexfield uses a separate `conv + Linear(c,1)`
  per target (`policy`, `opp_policy`, `cell_q`).
- **Change:**
  - **Model** (`model.py::HexfieldNet.__init__` / `forward`): added
    `soft_policy_conv = HexNodeConv(c,c)` + `soft_policy_head = nn.Linear(c,1)`
    and emit `out['soft_policy'] = self._policy_logits(...)` in `forward`.
    **Train-only:** NOT added to `forward_policy_value` (serve), exactly like
    `cell_q` / `opp_policy`. Fresh/zero-init via `_init_weights`.
  - **Soft target** (`batching.py::collate_training`): derived from the already
    packed `policy` tensor — `p = policy / row_sum`; `soft = (p + 1e-7)^0.25`
    confined to each row's legal prefix `[0,n)` (off-prefix slots stay exactly 0
    so no mass lands off the legal prefix, which `segment_policy_ce` treats as a
    hard error). **Pure function of the visit policy, computed AFTER expand**, so
    serial/pool/rust expand backends stay element-identical and **no shard-schema
    or `replay_expand.rs` change is needed** (main_4 keeps `expand_backend=rust`).
  - **Loss** (`losses.py`): `SOFT_POLICY_WEIGHT = 8.0`; `hexfield_loss` adds a
    `soft_policy` component `segment_policy_ce(outputs['soft_policy'],
    legal_counts, batch['soft_policy'], denominator=rows)` weighted by
    `soft_policy_weight`, gated on both `'soft_policy' in outputs` and `in batch`
    (so a model without the head is a no-op). **Flat `rows` denominator** — the
    aux soft loss is *not* surprise-reweighted.
  - **Wiring** (`trainer.py`, `config.py`): `TrainingSection.soft_policy_weight=8.0`
    (kept in sync with `losses.SOFT_POLICY_WEIGHT`); `trainer.py` passes
    `soft_policy_weight=tcfg.soft_policy_weight` into `hexfield_loss`.
  - **Checkpoint** (`checkpoints.py`): tolerant warm-start so the fresh head loads
    cleanly (§3, item: warm start).
- **Faithful vs adapted:** **Faithful target/loss** (verbatim `(p+1e-7)^(1/4)`,
  weight 8.0). **Adapted head layout:** own-conv (mirroring `opp_policy`/`cell_q`)
  rather than KataGo's shared-conv channel slice — a documented Hexfield
  per-head-conv adaptation, flagged as not byte-faithful to KataGo's channel
  layout (owner-confirmed default; see §9). **Player-only** soft policy (no soft
  *opponent* head; owner open decision §9).

---

## 3. Bugs fixed (itemized)

1. **Nucleus f32 truncation / false "mass=1.0 disables cap"**
   (`tree.rs::nucleus_count_values`). The f32 cumulative loop could reach `mass`
   early and truncate the low-prior tail, and `mass==1.0` did **not** disable the
   cap. Fixed under `nucleus_f64`: f64 accumulation + explicit
   `mass >= 1.0 => return hi` sentinel. Tests:
   `nucleus_f64_sentinel_returns_total_at_mass_one`,
   `nucleus_f64_truncation_vs_legacy_f32`,
   `nucleus_f32_matches_legacy_for_normal_mass`,
   `nucleus_respects_min_and_max_clamp`.
2. **Frozen `max_eligible_children`** (`tree.rs`; the field was set at node
   creation and `recompute_accounting` never re-derived it → promoted/mid-run
   nodes kept the stale cap). Fixed under `lazy_widening`: eligibility is now a
   **live** `peek_next_candidate().is_some()` check, so there is no stale cap to
   carry.
3. **New-child scored U-only** (`tree.rs::select_or_materialize_edge`). The
   materialized candidate got `prior*scale` while existing edges got
   `value_or_fpu + U + ml_bonus`; the bias flips with `sign(parent_value)`. Fixed
   under `new_child_fpu` via `new_child_score` (= `fpu + U`). Tests:
   `new_child_score_fpu_vs_u_only`, `new_child_score_matches_existing_zero_visit_edge`.
4. **Dirichlet reuse-compounding** (`tree.rs::apply_root_dirichlet_noise` /
   `apply_root_policy_temperature`). Noise/temperature re-mixed into already-noised
   priors on reused/promoted roots → compounding across plies. Fixed under
   `clean_root_prior_cache`: cache the clean post-temp pre-noise priors and reset
   from them before re-mixing. Tests: `clean_cache_reset_stops_compounding`
   (production: no drift) and `legacy_no_cache_does_compound` (parity: still
   compounds — pins the parity path unchanged).
5. **Pruned-target static-vs-dynamic c_puct** (`search.rs::build_search_result_payloads`
   → `pruned_visit_policy`). Fixed under `pruned_dynamic_cpuct` via
   `effective_pruning_c_puct`.
6. **Grad-norm bucket mislabel (diagnostic-only)**
   (`trainer.py::_build_grad_norm_groups`). The predicate referenced a stale
   single `bias_table`; the arch renamed it to per-block `bias_tables.*`. Changed
   to `name == "tokens" or name.startswith("bias_tables")` so per-block
   bias tables bucket into `trunk_attn`. New `soft_policy_*` params auto-bucket
   into `heads` (no change needed).
7. **Strict-load failure on the fresh soft head** (`checkpoints.py`). The v3 BC
   prefit has no `soft_policy_*` keys, so a strict `initialize_from` load would
   raise. Fixed with `warm_start_into` (tolerant, weights-only, key+shape match,
   missing keys keep their `_init_weights` value) on the initialize_from / BC
   path **only**; the strict `resume` path is untouched. Test:
   `test_warm_start_zero_inits_missing_soft_head`.

---

## 4. Dead code removed vs parity harness kept

**Net removals on this branch: NONE.** The "remove only dead legacy fallbacks"
directive yields an empty net removal set right now. The only candidate deletions
are **deferred-until-flag-flip** and remain gated behind `parity()`:

- `RustNode.max_eligible_children` + its recompute — still referenced by the
  `lazy_widening`-off (parity) path; delete only after `lazy_widening` is the
  permanent production default and no test/parity path references it.
- The noise-conditioned `root_fpu_zero_under_noise` branch in
  `search.rs::root_fpu_for` — kept behind `parity()` so golden vectors stay
  byte-identical; only the production **default** moves to
  `root_fpu_reduction=0.0`. This is a flag flip, not a deletion.

**Explicitly KEPT (owner-mandated):** `Divergences::parity()`/`production()`
(`tree.rs`), `search_parity_mode` plumbing, `resolve_divergences` +
`build_divergence_overrides`, `debug_lcb_pick` / `debug_ml_bonus` pyfunctions,
the M5/M6 golden-vector tests, the serial/pool/rust expand ladder, the
FlexAttention / async-eval / inference dual-paths (default-off, out of CPU-test
scope), `legacy_model_v2.py` (eval anchor compat — the soft head was NOT added
there), `soft_z_lambda` (inert restnet-parity *value*-target hook, distinct from
the new soft *policy* head — not conflated).

---

## 5. `configs/hexfield_main_4.toml` deltas (vs `hexfield_main_3.toml`)

Copy of main_3 with identity + new knobs. Verified key/value deltas (non-comment):

- `[run] name = "hexfield_main_4"`, `output_dir = ".../runs/hexfield_main_4"`.
- `[checkpoint] initialize_from` = the same v3 BC prefit as main_3
  (`/mnt/e/Hexo-BotTrainer-hexgt/runs/hexfield_bc_v3/checkpoint_epoch0.pt`,
  a read-only reference into the live tree), `warmup_steps=0`, fresh warm start
  via the tolerant load (soft head zero-inits).
- `[model.config.selfplay]` NEW: `root_fpu_reduction = 0.0`,
  `root_fpu_zero_under_noise = false`, `visit_scaled_c_puct = true`,
  `c_scale = 0.45`, `c_base = 500.0`, `lcb_z = 1.6`, and the six divergence flags
  `nucleus_f64 = new_child_fpu = lazy_widening = clean_root_prior_cache =
  dirichlet_shaped = pruned_dynamic_cpuct = true`. `fpu_reduction = 0.2`
  (interior) unchanged.
- `[model.config.training]` NEW: `soft_policy_weight = 8.0`.
- **KEPT verbatim from main_3:** `search_visits=512`, `c_puct=1.5`,
  `root_dirichlet_total_alpha=10.83`, `root_dirichlet_noise_fraction=0.25`,
  `forced_playout_k=2.0`, `widening_policy_mass=0.95`,
  `widening_max_children=96`, `widening_min_children=2`,
  `temperature=1.0`/`temperature_floor=0.15`/`temperature_halflife_plies=60.0`,
  `root_policy_temperature=1.07` (flat, early=0/halflife=0), `max_game_plies=256`,
  `ml_two_sided=false`, `ml_final_pick_band=0.08`, the full replay-buffer block,
  `learning_rate=3e-4`, `train_samples_per_epoch=48000`,
  **`expand_backend="rust"`** (valid — the soft target is derived in collate, not
  expand).

> Note: `widening_max_children=96` is still passed; with `lazy_widening=true` the
> live `can_widen` check is the real gate, but the cap is kept for the parity path
> and for flipping `lazy_widening` off.

---

## 6. Parity strategy

Every new KataGo-faithful behavior ships behind a per-knob flag on the
`Divergences` struct (`tree.rs`): `nucleus_f64`, `new_child_fpu`,
`lazy_widening`, `clean_root_prior_cache`, `dirichlet_shaped`,
`pruned_dynamic_cpuct`, plus the first-class `root_fpu_reduction` value and the
now-first-class `c_scale`/`c_base`/`visit_scaled_c_puct`/`lcb_z`.

- `Divergences::parity()` sets **every** new flag to the **legacy/current** value
  (flat Dirichlet, frozen cap, U-only new child, static-c_puct pruning, in-place
  noise compounding, f32 nucleus, legacy noise-conditioned root FPU). The M5/M6
  golden vectors and the differential harness select `parity()` via
  `search_parity_mode=true`, so they stay **byte-identical**.
- `Divergences::production()` sets every new flag to the KataGo-faithful value.
  main_4's toml runs `production()` (`search_parity_mode=false`, the existing
  default).
- `resolve_divergences` (`search.rs`) reads all six new keys from the overrides
  dict, so M6 property gates / M10 lesions can flip each flag individually.
- `parity()` / `production()` / `resolve_divergences` / `search_parity_mode` /
  the debug pyfunctions are structurally untouched.
- The soft-policy head is **train-only** (`forward`, not `forward_policy_value`),
  so serve parity is unaffected; the soft target is derived in `collate_training`
  from `policy` (a pure function), so all expand backends and any expand-parity
  oracle stay element-identical.

Unit tests pinning the strategy: `parity_disables_all_main4_divergences`,
`production_enables_all_main4_divergences`,
`parity_dirichlet_is_flat_and_byte_identical`, `legacy_no_cache_does_compound`.

---

## 7. Verified vs deferred

**VERIFIED (this session):**

- **Rust `cargo test` — all 22 tests pass** (7 pre-existing `threats_shared` + 15
  new `tree::tests`). Run inside the worktree (compiles to the worktree's own
  `target/`, touches no venv):
  ```
  PATH=$HOME/.cargo/bin:$PATH \
  PYO3_PYTHON=/root/.venvs/hexfield-dev/bin/python \
  RUSTFLAGS="-L native=/usr/lib/x86_64-linux-gnu -C link-arg=-lpython3.12" \
  cargo test -p hexfield --lib --features python
  ```
  > The `tree`/`search` modules are `#[cfg(feature="python")]`, so the new tests
  > only compile under `--features python`. The `extension-module` build does not
  > link libpython into a test binary (symbols are resolved by the interpreter at
  > runtime), so the test binary is linked against the system
  > `libpython3.12.so` via `RUSTFLAGS`. This **reads** the hexfield-dev venv's
  > interpreter for config only; it does **not** `pip install` or modify any venv,
  > and does not run `maturin`. (Used the rustup-managed cargo 1.95 from
  > `~/.cargo/bin`; the default `/usr/bin/cargo` 1.75 cannot parse the v4
  > `Cargo.lock`.)
- **Python `py_compile`** of all changed modules + the new test file — all OK
  (`batching.py`, `checkpoints.py`, `config.py`, `losses.py`, `model.py`,
  `selfplay.py`, `trainer.py`, `tests/test_hexfield_soft_policy.py`).
- **PyO3 signatures / `validate_nonnegative_f32`** confirmed present; full crate
  (incl. `search.rs`, `payload.rs`, `replay_expand.rs`) compiles under
  `--features python`.

**DEFERRED:**

- **Torch CPU unit tests** (`tests/test_hexfield_soft_policy.py`). The
  `hexfield-dev` venv has **no torch** and the directive forbids touching the
  shared `hexgt-build` venv. Creating a fresh isolated venv and pip-installing a
  CPU torch is a large multi-hundred-MB download (not "quick"), so per the safety
  rule the torch tests are **documented as deferred** rather than risk it. The
  file is syntactically verified (py_compile) and asserts the exact KataGo
  transform / CE / loss-wiring / warm-start behavior; run it once a CPU-torch
  isolated venv is available:
  ```
  python -m pytest packages/hexfield/python ... tests/test_hexfield_soft_policy.py
  ```
- **The actual maturin build into a real venv** — forbidden by scope (no
  `_rebuild_hexo_models_hexgt.sh`, no maturin into the shared venv). main_4
  requires a `.so` rebuilt from this branch's Rust before launch (see §8).
- **Launch / any GPU / selfplay / eval** — out of scope (PREPARE ONLY).

---

## 8. How to launch main_4 when the owner frees the GPU

1. **Sync the branch** to wherever the run reads its `.py` from (the live run
   imports `.py` via PYTHONPATH and holds its `.so` in memory; sync at a clean
   epoch boundary). Do **not** disturb the running main_3.
2. **Rebuild the hexfield `.so` from this branch** into an isolated/dev venv
   (NOT the shared `hexgt-build`, NOT during a main_3 epoch). hexfield has its own
   cdylib (`hexfield._rust`) deliberately separate from `hexo_models._rust`, so
   building it cannot replace main_3's live `.so`. Use the hexfield rebuild script
   (`scripts/_rebuild_hexfield.sh`, hexfield-dev venv) — **not**
   `_rebuild_hexo_models_hexgt.sh`.
3. **Confirm the warm start.** `[checkpoint].initialize_from` points at the v3 BC
   prefit; the tolerant `warm_start_into` path will load matching trunk/head
   weights and fresh/zero-init `soft_policy_*`. Check the returned `warm_start`
   summary logs `shape_mismatch == []`, `unexpected == []`, and `missing ==`
   only `soft_policy_*` keys.
4. **Launch with `configs/hexfield_main_4.toml`** (`[run].name=hexfield_main_4`,
   `output_dir=.../runs/hexfield_main_4`). It points at no live run, so launching
   cannot disturb main_3. `search_parity_mode` stays false → `production()`.
5. **Sanity-gate (optional but recommended):** before/early in the run, re-run the
   M5/M6 golden-vector + differential harness to confirm `parity()` is still
   byte-identical, and flip individual divergence flags via `divergence_overrides`
   for the M6/M10 lesions.

---

## 9. Open owner decisions (baked-in defaults, flip in toml/code if changed)

1. **Soft-policy trunk sharing** — defaulted to **own-conv** (Hexfield per-head
   pattern). KataGo shares one multi-channel policy conv. Documented divergence.
2. **Soft opponent policy** — **not added** (player-only). KataGo also trains a
   soft opponent channel at 8×.
3. **No-widening scope** — `lazy_widening` + `new_child_fpu` both **ON** (FPU as
   sole gate). Alternative: keep a bounded cap + `new_child_fpu` only.
4. **Dirichlet normalization** — handled via the clean-cache reset (priors sum to
   1 after reset → mix matches KataGo). 
5. **BC prefit handling** — **tolerant-load** the existing v3 prefit (no GPU/prefit
   run needed). Alternative: regenerate a prefit with the soft head.
6. **`cpuctUtilityStdevScale`** (per-node variance c_puct factor) — **left out**
   (KataGo default 0.0 = no-op).
7. **`c_scale`** — **RESOLVED → 0.0** (see §10). KataGo *self-play* uses
   `cpuctExplorationLog=0.0`; the 0.45 was the analysis/strength value.
8. **`root_policy_temperature`** — **RESOLVED → 1.1 / early 1.25 / halflife 19**
   (see §10), matching KataGo self-play (was flat 1.07).
9. **`c_puct` value** — **STILL OPEN**: kept **1.5**. Strict KataGo self-play =
   **1.1** (Hexo is pure win/loss, so its `[-1,1]` Q equals KataGo's winLoss
   utility scale — no radius rescale applies, contra the earlier "1.5 ≈ rescaled"
   note, which assumed a Go score-utility radius that does not exist here). 1.1
   cuts exploration ~27% vs 1.5 and compounds with the other exploration-raising
   main_4 changes, so it is left at 1.5 pending an explicit owner call.

## 10. Follow-up faithfulness changes (post-review, config-only)

Two KataGo self-play divergences that the initial build left as owner decisions
were applied. Both are `configs/hexfield_main_4.toml`-only (no Rust/Python; the
interpolation and the c_scale wiring already existed). Sourced from KataGo
`cpp/configs/training/selfplay8b20.cfg` + `cpp/search/searchparams.{h,cpp}`.

### 10.1 `c_scale` 0.45 → 0.0  (dynamic-c_puct log term)
- KataGo self-play sets `cpuctExplorationLog = 0.0`. The `0.45` value is the
  gtp/analysis strength preset, not the training value. Library default is also
  `0.0` (`searchparams.cpp`).
- Effect: `c_for(N) = c_puct + c_scale·ln((N+c_base)/c_base)` loses its log term →
  `c` is constant `= c_puct` for the whole search (previously it ramped
  ~1.5→~1.82 over a 512-visit search). `visit_scaled_c_puct=true` is now a no-op.
- Open: `c_puct` itself kept at **1.5**; strict self-play faithful is **1.1**
  (§9.9). No rescale applies because Hexo has no score utility, so its `[-1,1]` Q
  is exactly KataGo's winLoss utility scale.

### 10.2 `root_policy_temperature` flat 1.07 → 1.1 / early 1.25 / halflife 19
- KataGo self-play: `rootPolicyTemperature=1.1`, `rootPolicyTemperatureEarly=1.25`,
  early→steady decay over `chosenMoveTemperatureHalflife=19` moves.
- Hexfield `root_temp_for` already implements the identical interpolation
  `temp(ply)=steady+(early−steady)·0.5^(ply/halflife)` (KataGo `interpolateEarly`),
  gated off when early/halflife ≤ 0. Setting early=1.25, steady=1.1, halflife=19
  enables it: the opening prior is flattened more (1.25), decaying to 1.1.
- Adaptation: halflife is in plies. KataGo's 19 is board-size-scaled; Hexo has no
  fixed board, so 19 is the literal transfer (tunable; Hexo's move-selection
  halflife is 60, so 19 is on the shorter/earlier side). Flagged as adaptable.

### 10.3 NOT changed — FPU `√(policyProbMassVisited)` scaling (faithfulness item #4)
KataGo's FPU reduction is `fpuReductionMax · √(policyProbMassVisited)` (weaker at
lightly-explored nodes, growing toward the max as visited mass accumulates);
Hexfield uses a flat `fpuReductionMax` (0.2). Left unchanged because its impact is
**low here**: (a) main_4 sets `rootFpuReductionMax=0`, so the ROOT — where FPU
most affects opening breadth and the recorded target — has no FPU penalty at all,
making the scaling irrelevant there; (b) it therefore only alters INTERIOR-node
breadth, which does not directly shape the trained policy target; (c) flat-0.2 vs
√-scaled-0.2 differ only in how quickly a node transitions from broad to focused,
a second-order effect. It is a real but low-priority faithfulness gap (interior
search breadth only) and a small `tree.rs` follow-up if desired.
