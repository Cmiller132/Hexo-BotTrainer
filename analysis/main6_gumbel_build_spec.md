# main_6: FULL Gumbel AlphaZero — Definitive BUILD SPEC

**Author:** lead engineer (hexfield)
**Date:** 2026-06-30
**Branch:** `claude/hexfield-gumbel` (worktree `E:/Hexo-BotTrainer-gumbel`)
**Status:** BUILD SPEC — locked. Supersedes `analysis/main6_gumbel_az_plan.md`.
**Mandate:** Implement **ALL THREE** Gumbel AlphaZero mechanisms (Danihelka et al., ICLR 2022), uncompromised, behind the repo's `Divergences` flag pattern, default-OFF, so `production()` stays KataGo-PUCT and golden-vector parity stays byte-identical. `main_6` opts in via config.

> **This spec DROPS the prior plan's framing.** The old plan was "mechanism #3 only / defer #1/#2 / hedge against main_4 value-Q saturation." That framing is **discarded**. We build #1 (Gumbel-Top-k root + Sequential Halving), #2 (deterministic non-root selection), and #3 (improved completedQ target) as one coherent flag-set. The main_4 saturation concern is **not** a design driver — standard Gumbel hyperparameters are implemented faithfully to the paper, not as defensive crutches. Q is on `[-1,1]` here (`Q_UTILITY_WIDTH=2.0`, tree.rs:49), a clean σ input.

---

## 0. Canonical definitions (locked — referenced by every step)

### 0.1 The σ transform
```
σ(q) = (c_visit + max_b N(b)) · c_scale · q
```
- `c_visit = 50.0` (default), `c_scale = 1.0` (default) — the canonical Danihelka constants.
- `max_b N(b)` = the maximum visit count over the children of the node σ is being applied at (root for the target/root-selection; the interior node for #2).
- `q ∈ [-1,1]` already (no Q rescale — unlike KataGo). Monotone increasing in q. `σ(0)=0`.
- Config knobs: `gumbel_c_visit`, `gumbel_c_scale` (both `SelfplayConfig`, reach Rust).

### 0.2 completedQ
For action `a` at a node with children visit counts `N(a)`, value means `Q(a)=value_sum/visits` (RustEdge::value, tree.rs:191-197):
```
completedQ(a) = Q(a)                       if N(a) > 0   (visited)
              = v_mix                       if N(a) == 0  (unvisited fallback)
v_mix = ( Σ_b π(b)·Q(b) ) / ( Σ_b π(b) )    over VISITED b only
```
where `π(b) = softmax(logits)(b)` is the network prior probability. `v_mix` is the **visit-weighted (prior-weighted over visited) node value** — the paper's value fallback for unvisited actions. This is computed once per node from visited children; it is NOT a raw child value-head read.

### 0.3 Improved policy / training target
```
π'(a) = softmax_a( logits(a) + σ(completedQ(a)) )
```
over the full legal/candidate support. This is BOTH the training target (#3) and the score basis for non-root selection (#2) and the SH halving rank (#1).

### 0.4 Non-root deterministic selection (#2)
```
select argmax_a [ logits(a) + σ(completedQ(a)) ]
```
No PUCT, no `c_puct`, no FPU, no widening at interior nodes. (Implemented as: pick the action maximizing `logits(a)+σ(completedQ(a)) - (fractional visit-count correction)` — the canonical Gumbel deterministic action, which for the simple form is argmax of `logits+σ(completedQ)` with the standard "most-visited-deficit" tie handling; we implement the simple argmax form, matching the paper's Eq. for non-root.)

### 0.5 Gumbel-Top-k root sampling (#1)
- Draw `g(a) ~ Gumbel(0,1)` i.i.d. per legal root action.
- Candidate set `A_topm` = the top-`m` actions by `logits(a) + g(a)`.
- This is provably sampling `m` actions **without replacement** from `softmax(logits)` (Gumbel-max trick).

### 0.6 Sequential Halving (#1, ROOT-ONLY — canonical Gumbel)
Budget `n` visits over `m` candidates, `R = ceil(log2(m))` rounds:
- Round `r` allocates `floor( n / (R · |A_r|) )` visits to **each** surviving candidate (equal split per round), where `A_r` is the survivor set at round `r` (`|A_0| = m`).
- After each candidate in `A_r` is visited its per-round quota, rank survivors by `g(a) + logits(a) + σ(completedQ(a))`, keep the top `ceil(|A_r|/2)`, advance.
- Terminate at 1 survivor → that is the played move. SH is **ROOT-ONLY**; interior selection is #2.

### 0.7 m default vs branching factor
- `m = 16` default (`gumbel_m`). Branching here is 337→~1000 and **grows** late-game (`LEGAL_RADIUS=8` halo), so SH never starves. Budget 1024 over `R=4` rounds ≈ 256 visits/round; final 2 candidates get ~256+ each.
- `m = min(gumbel_m, n_legal)` clamp at the root (opening forces `(0,0)` → m=1 trivially).
- m applies to **Full** moves only. Fast (192) and Init (1) skip SH (and skip Gumbel root sampling).

### 0.8 RNG seed stream for Gumbel draws
- Add `pub const SEED_STREAM_GUMBEL: u64 = 6;` (search.rs:43-48 block; next free id after `SEED_STREAM_POLICY_INIT_SAMPLE=5`). Mirror how `SEED_STREAM_ROOT_NOISE=0` is consumed in `apply_root_dirichlet_noise` / `owned_root_from_evaluation`. Per-root Gumbel draws seed off this stream + the root index (same `wrapping_add(index)` discipline as the move-select seed at search.rs:1925).
- Do NOT reuse `SEED_STREAM_ROOT_NOISE` (it would correlate Dirichlet and Gumbel draws and break parity-mode reproducibility expectations). A dedicated stream keeps Gumbel draws independent and reproducible.

### 0.9 How SH rounds map onto the async continuous scheduler (ROOT-ONLY, intra-budget)
- The scheduler (`select_continuous_pass` search.rs:1412, rayon `.par_iter_mut()` :1423; global flush `continuous_flush_decision` search.rs:235-247; per-slot completion `continuous_completion_ready` search.rs:258-259) is **per-root async with a global queue flush**. There is **NO cross-slot barrier** and we add none.
- SH is realized as **per-slot (per-root) state**: each slot tracks `root_candidates`, `current_round`, `round_budget`, and per-candidate accumulated visits. Halving fires **within a slot** only when ALL surviving candidates in that slot have reached the round's per-candidate cap (the **intra-slot barrier** — preserves SH equal-allocation). Slots advance rounds **independently** of each other, so the global flush keeps batching leaf evals across all roots and the async overlap / GPU amortization is untouched.
- Net: equal-allocation-per-round **within** each root (SH guarantee held); full async **across** roots (scheduler untouched). The leaf-selection for a slot in round `r` is constrained to that slot's surviving candidate subtree; interior selection within that subtree is #2 (`argmax[logits+σ(completedQ)]`).

---

## 1. Divergence-flag design (the gating contract)

New fields on `struct Divergences` (tree.rs:69-120), each defaulting to its **OFF/legacy** value in `parity()` (tree.rs:123) and OFF in `production()` too (Gumbel is opt-in **only** via the `main_6` config — `production()` MUST stay KataGo-PUCT so existing main_5 self-play and golden vectors are unaffected). Add a convenience `Divergences::gumbel()` profile that starts from `production()` and flips the Gumbel set ON, for tests.

New flag fields (all `bool` unless noted):
- `gumbel_target: bool` — #3 improved-policy target export.
- `gumbel_root: bool` — #1 Gumbel-Top-k root candidate sampling.
- `gumbel_sequential_halving: bool` — #1 SH budget allocation (requires `gumbel_root`).
- `gumbel_nonroot_select: bool` — #2 deterministic non-root selection.
- `gumbel_c_visit: f32` (default 50.0), `gumbel_c_scale: f32` (default 1.0) — σ constants.
- `gumbel_m: u32` (default 16) — candidate count.
- `gumbel_target_min_visits: u32` (default 1) — support floor for the target.

`parity()` sets `gumbel_target=false, gumbel_root=false, gumbel_sequential_halving=false, gumbel_nonroot_select=false` and the scalar defaults above. `production()` sets the same four bools `false` (inherits via `..Self::parity()`). `gumbel()` = `production()` with the four bools `true`.

**Contract:** with all four bools false, every Gumbel code path is bypassed and the output is byte-identical to today. The M5/M6 golden-vector parity tests construct `parity()` directly and never touch these fields.

---

## S1 — config knobs + Divergences gumbel flags (default-OFF)

**Goal:** add all config + flag plumbing, compile, pass baseline cargo tests, with every Gumbel path dormant.

### Files & changes
1. **`packages/hexfield/rust/src/tree.rs`** (~69-170): add the new fields to `struct Divergences` (after `pruned_dynamic_cpuct`, tree.rs:119); set OFF/default values in `parity()` (tree.rs:124-149) and `production()` (tree.rs:153-167, inherit `false` via `..Self::parity()`); add `pub fn gumbel() -> Self` profile.
2. **`packages/hexfield/rust/src/search.rs`** (~43-48): add `pub const SEED_STREAM_GUMBEL: u64 = 6;`.
3. **`packages/hexfield/rust/src/search.rs`** `run_continuous` pyo3 signature (search.rs:763) + the `divergence_overrides` extract path (`resolve_divergences`): add the new flags so config can drive them. Mirror how the six main_4 flags (`nucleus_f64` … `pruned_dynamic_cpuct`) are extracted from the overrides dict.
4. **`packages/hexfield/python/hexfield/config.py`**:
   - `SelfplayConfig` (config.py:16): add fields `gumbel_target_enabled: bool = False`, `gumbel_root_enabled: bool = False`, `gumbel_sequential_halving: bool = False`, `gumbel_nonroot_select: bool = False`, `gumbel_c_visit: float = 50.0`, `gumbel_c_scale: float = 1.0`, `gumbel_m: int = 16`, `gumbel_target_min_visits: int = 1`, `export_root_prior_logits: bool = False`.
   - `TrainingSection` (config.py:97): add `policy_target: str = "visit"` ({"visit","gumbel"}).
   - `build_divergence_overrides` (config.py:407-457): emit the four bool flags + `gumbel_c_visit`/`gumbel_c_scale`/`gumbel_m`/`gumbel_target_min_visits` into the overrides dict (concrete bool/float/int, never None — `resolve_divergences` calls `.extract()`).
   - **Strict unknown-key guard (config.py:377-381):** every selfplay knob MUST be a `SelfplayConfig` field and live under `[model.config.selfplay]`; `policy_target` MUST be a `TrainingSection` field under `[model.config.training]`. Mis-placing any key raises `ValueError` at load. (Confirmed: `losses.py` sources the training-side selector from `TrainingSection`, NOT `SelfplayConfig`.)
5. **`configs/hexfield_main_6.toml`** (new, copy of `hexfield_main_5.toml`): add the Gumbel knobs default-OFF under the correct tables (per §9 of the prior plan; `policy_target="visit"` under `[model.config.training]`). Warm-start `initialize_from` a healthy main_5 checkpoint.

### Build/test go/no-go (S1)
```
# BUILD (release, isolated dev venv)
wsl.exe -e bash -lc "source /root/.venvs/hexfield-dev/bin/activate; export PATH=/root/.cargo/bin:\$PATH; cd /mnt/e/Hexo-BotTrainer-gumbel && maturin develop --release -m packages/hexfield/Cargo.toml 2>&1 | tail -50"
# mirror the .so into the worktree source tree
wsl.exe -e bash -lc "cp \$(ls /root/.venvs/hexfield-dev/lib/python3.12/site-packages/hexfield/_rust*.so | head -1) /mnt/e/Hexo-BotTrainer-gumbel/packages/hexfield/python/hexfield/"
# CARGO UNIT TESTS (baseline must still pass; ~22 expected)
wsl.exe -e bash -lc "export PATH=/root/.cargo/bin:\$PATH; cd /mnt/e/Hexo-BotTrainer-gumbel && cargo test -p hexfield 2>&1 | tail -40"
```
**GO** iff: maturin build succeeds; all baseline cargo tests pass (Gumbel paths dormant); `python -c "from hexfield.config import parse_hexfield_config"` loads `hexfield_main_6.toml` without `ValueError`. No behavior change yet.

---

## S2 — Python→Rust raw-logit plumbing

**Goal:** carry pre-softmax network policy **logits** across the evaluator boundary onto the root/edges so σ/Gumbel/target math has true logits (they do NOT exist in Rust today — only softmaxed `priors`). Optional/absent-tolerant so parity replies and old evaluators still load. No behavior change with flags off.

### Files & changes (in dependency order)
1. **`packages/hexfield/python/hexfield/inference.py`** `_decode_group` (inference.py:457-478): `logits = out["policy"].float()` is already in hand at :466. Gather `logits[legal]` exactly as `priors[legal]` is gathered (:475-478) and return it; serialize as a new reply field `priors_logits_bytes` (fp32, same positional layout as `priors_bytes`). Update the module doc header to list the new field. **Only emit it when `export_root_prior_logits`/gumbel is on** (so parity replies stay minimal); gate via the evaluator's request flags (mirror `request_moves_left`).
2. **`packages/hexfield/rust/src/payload.rs`** `parse_chunk_reply` (payload.rs:162-181): parse `priors_logits_bytes` with the same `require_exact_bytes` length validation as `priors_bytes` (:181), into a `Vec<(PackedCoord, f32)>`. Make it **optional** — absent ⇒ `None` (mirror the `request_moves_left` optional block at payload.rs:182-193). Build it aligned to the same legal-id ordering as `priors`.
3. **`packages/hexfield/rust/src/cache.rs`** `RustEvaluation` (cache.rs:17-28): add `pub logits: Option<Vec<(PackedCoord, f32)>>` (after `moves_left`).
4. **`packages/hexfield/rust/src/tree.rs`** `owned_root_from_evaluation` (tree.rs:1248-1317): carry `evaluation.logits` onto the root node. Add `pub root_logits: Option<HashMap<PackedCoord, f32>>` to `RustNode` (populated next to `eval_value`, tree.rs:1305), built from `evaluation.logits` keyed by action_id so the target-build, Gumbel sampler, and #2 selection can look up `logits(a)` aligned to the candidate set. Logits are stored RAW (pre-temperature, pre-Dirichlet) — they are NOT subjected to `apply_root_policy_temperature_to` (:1270) or noise.
   - For interior nodes (#2), logits must also be carried onto expanded child nodes; store them analogously on each `RustNode` at expansion time from that node's own `RustEvaluation.logits`.

### Build/test go/no-go (S2)
Same build commands as S1. **GO** iff: builds; baseline cargo tests still pass (logits `None` everywhere in parity, all flags off ⇒ no path reads them); a new unit test asserts: (a) a reply WITH `priors_logits_bytes` parses into `Some(..)` aligned to `priors`; (b) a reply WITHOUT it parses into `None` and search proceeds (parity untouched). Golden-vector parity byte-identical.

---

## S3 — Gumbel-Top-k root sampling + Sequential Halving (gated)

**Goal:** at Full roots, when `gumbel_root` is on, replace Dirichlet+temperature root diversity with one Gumbel-Top-k draw of `m` candidates, and (when `gumbel_sequential_halving` on) allocate the per-move visit budget via root-only SH. Flags off ⇒ today's PUCT+Dirichlet path verbatim.

### Files & changes
1. **`packages/hexfield/rust/src/tree.rs`** `owned_root_from_evaluation` (tree.rs:1248-1317):
   - When `divergences.gumbel_root`: SKIP `apply_root_policy_temperature_to` (:1270), SKIP `apply_dirichlet_noise` (:1288) (Gumbel draw is the sole diversity source — §4.4 of the plan: leaving them on double-counts exploration). Also disable nucleus widening + forced-playout discovery on Gumbel roots (SH's m-candidate set IS the discovery mechanism).
   - Draw `g(a) ~ Gumbel(0,1)` per legal root action using `SEED_STREAM_GUMBEL` (§0.8) + root index. Compute `logits(a)+g(a)` from `root_logits` (S2). Select top-`m = min(gumbel_m, n_legal)` → the root candidate set. Store the candidate set + per-candidate `g(a)` on the root for SH ranking.
2. **`packages/hexfield/rust/src/search.rs`** continuous-scheduler slot state + selection:
   - Per-slot SH state (§0.9): `root_candidates: Vec<PackedCoord>`, `current_round: u32`, `round_budget: u32`, per-candidate accumulated-visit tracking. Initialize after RootInit backup (`backup_continuous_items` search.rs:1455), before leaf selection, when `gumbel_root` on.
   - Constrain root leaf-selection to the current survivor set; allocate `floor(n/(R·|A_r|))` per survivor per round (§0.6).
   - **Intra-slot barrier:** halving fires only when ALL survivors in the slot reached the round cap (§0.9). On halving, rank survivors by `g(a)+logits(a)+σ(completedQ(a))` and keep `ceil(|A_r|/2)`; advance round. NO cross-slot barrier — `continuous_flush_decision` (search.rs:235-247) and the global flush path are **unmodified** (design constraint).
   - When `gumbel_sequential_halving` off but `gumbel_root` on: candidates are the Gumbel-Top-m set but visits allocate via normal PUCT among them (intermediate mode for A/B).
3. **σ helper** (new fn in tree.rs or search.rs): `sigma(q, max_n, c_visit, c_scale) = (c_visit + max_n as f32) * c_scale * q` (§0.1). `completed_q(edge, node, root_logits)` per §0.2 (visit-weighted v_mix fallback). Shared by S3/S4/S5.

### Build/test go/no-go (S3)
Build commands as S1. **GO** iff: builds; baseline + S2 tests pass; with `gumbel_root=false` golden vectors byte-identical; a gated smoke run (`gumbel()` profile, tiny budget) produces a played move and an `m`-sized candidate set without panics. Unit tests for Gumbel-Top-k and SH allocation land in S6 (can be written here and run).

---

## S4 — deterministic non-root selection argmax[logits + σ(completedQ)] (gated)

**Goal:** when `gumbel_nonroot_select` on, interior (non-root) node selection uses `argmax_a [logits(a) + σ(completedQ(a))]` — no PUCT/c_puct/FPU/widening. Flag off ⇒ `select_or_materialize_edge` PUCT verbatim.

### Files & changes
1. **`packages/hexfield/rust/src/tree.rs`** `select_or_materialize_edge` (tree.rs:831-864):
   - Keep the TSS forced-edge first-visit loop (:833-837) unchanged (it precedes scoring; Gumbel does not change forced-edge semantics).
   - Branch on `self.divergences.gumbel_nonroot_select && node_id != 0`:
     - Compute `max_n = max_b N(b)` over the node's edges; `v_mix` per §0.2 from this node's visited edges and this node's `logits` (carried via S2 onto the child `RustNode`).
     - For each candidate edge, `score = logits(a) + sigma(completed_q(a), max_n, c_visit, c_scale)` with `completed_q = edge.value()` if visited else `v_mix`. Pick argmax (reuse `compare_edge_score` tie discipline for determinism).
   - Else fall through to the existing PUCT path (:839-864). **Root (node_id==0) is NEVER this path** — root uses #1 (Gumbel-Top-k+SH) or PUCT.
   - Widening: when `gumbel_nonroot_select` on, the candidate set is the node's expanded+materializable edges as usual; do NOT apply the lazy-widening FPU gate scoring (the score formula above replaces it). Materialization of a new child still uses the existing peek/expand machinery, but its selection priority is `logits(a)+σ(v_mix)`.

### Build/test go/no-go (S4)
Build as S1. **GO** iff: builds; baseline + S2 + S3 tests pass; `gumbel_nonroot_select=false` byte-identical golden vectors; gated smoke (`gumbel()`) runs interior selection without panic and visits concentrate on high `logits+σ(Q)` actions (sanity: a forced-win child gets selected once Q rises).

---

## S5 — completedQ improved-policy TARGET export + Python consumption (gated)

**Goal:** export `π'(a)=softmax(logits+σ(completedQ))` as the training policy target for Full rows when `gumbel_target` on, plumb it through the shard/window/expand pipeline, and let `losses.py` select it via `policy_target`. Flags off ⇒ today's visit-count target.

### Rust export
1. **`packages/hexfield/rust/src/search.rs`** `build_search_result_payloads` (search.rs:1886-1980), alongside the `visit_policy_*` / `root_prior_policy_*` exports (search.rs:1948-1960):
   - When `gumbel_target` on, compute the target over the root candidate support: `weight'(a) = softmax_a( logits(a) + σ(completedQ(a)) )`, where `logits(a)` = `root_logits` (S2), `completedQ(a)` = `edge.value()` for visited / `v_mix` (§0.2) for unvisited-but-in-support, `max_n` = root max visit. **Support floor:** actions with `N(a) < gumbel_target_min_visits` are EXCLUDED from the softmax support (then renormalize over survivors) — this denies the value head a write on un-searched cells. (NOT a `cell_q_mask` operation — that is the per-action Q-head presence mask, samples.py:267-275; this is choosing which actions enter the policy softmax.)
   - Export `gumbel_policy_action_ids_bytes` (u32) + `gumbel_policy_weights_bytes` (f32) via the existing `to_bytes`/`to_bytes_f32` closures (search.rs:1938-1947). Also export `gumbel_policy_count`.
   - Also export `root_prior_logits_bytes` (the raw logits column) for offline audit and so the target can alternatively be rebuilt in Python (Route A fallback).
2. **`packages/hexfield/rust/src/lib.rs`**: the payload dict already flows from `build_search_result_payloads`; no separate registration needed beyond the new `set_item` keys above. (Confirm no schema assertion in lib.rs rejects extra keys.)

### Python ingest → shard → window → expand
3. **`packages/hexfield/python/hexfield/selfplay.py`** (~162, next to `visit_policy_q_bytes`): read `gumbel_policy_action_ids_bytes`/`gumbel_policy_weights_bytes` (+ optional `root_prior_logits_bytes`) into new optional `HexfieldSampleData` fields `gumbel_policy: tuple[(action_id, float), ...]` and `prior_logits: tuple[(action_id, float), ...]`.
4. **`packages/hexfield/python/hexfield/samples.py`** `expand_sample` (~258-275 region, parallel to `cell_q`): project `gumbel_policy` onto this row's legal set as a dense `gumbel_policy (L,)` distribution (renormalized over kept support), validated finite & summing to 1; store `prior_logit (L,)` array parallel to `pol_act`. Field defs added near samples.py:42.
5. **`packages/hexfield/python/hexfield/shards.py`**: add per-action `prior_logit` array parallel to `pol_act` AND the dense `gumbel_pol` array parallel to the visit policy. Write (~shards.py:191) and read (~shards.py:268-270) with a legacy-absent guard like `if "q_pol_q" in arrays`. **Bump `SCHEMA_VERSION`** (shards.py:29, currently 1 → 2) and update the version-accept guard (shards.py:223-224) so the live reader accepts both (forward-compatible: old shard ⇒ field absent ⇒ fallback to visit target).
6. **`packages/hexfield/python/hexfield/window.py`** (MUST update or the column is written-but-unread, the exact q_pol_q reader bug): add `prior_logit` to the `CSR_GROUPS` pol-offset group (window.py:83); add `gumbel_pol` to the appropriate dense group; add both to the `PackedWindow` dataclass view (~window.py:142-144), the `_view_row` slice (~window.py:280-282), and the dtype map (~window.py:314-316). Mirror `q_pol_q` everywhere.
7. **`packages/hexfield/python/hexfield/expand_backends.py`** (MUST update): add `prior_logit` + `gumbel_pol` to the CSR-data column tuple (~expand_backends.py:81-83, next to `q_pol_q`) so `_window_columns_as_bytes` (expand_backends.py:281) packs them for the Rust expand kernel.
8. **`packages/hexfield/rust/src/replay_expand.rs`**: project the new columns. Rust struct field names `prior_logits` / `gumbel_policy` (mirror the `q_policy`↔`q_pol_q` two-name convention, replay_expand.rs:123,601). Project in BOTH the Rust kernel AND the serial (non-Rust) expand path so both backends agree. Export/projection-only — no scheduler touch.

### Loss consumption
9. **`packages/hexfield/python/hexfield/batching.py`** `collate_training` (~129-135): emit `gumbel_policy (B,L)` tensor alongside the existing dense `policy (B,L)`.
10. **`packages/hexfield/python/hexfield/losses.py`** `hexfield_loss`: read `policy_target` from `TrainingSection`; when `"gumbel"` and the row has a gumbel target (Full row, present), drive the main-policy CE from `gumbel_policy`; else `policy (visit)`. Old/absent rows ⇒ always `visit` (transition safety). Default `"visit"`.

### Selfplay wiring
11. **`packages/hexfield/python/hexfield/selfplay.py`**: pass `gumbel_target`/`export_root_prior_logits` through to the Rust `run_continuous` call (via `build_divergence_overrides`); ensure the new bytes are requested only when enabled.

### Build/test go/no-go (S5)
Build as S1, then a Python smoke:
```
wsl.exe -e bash -lc "source /root/.venvs/hexfield-dev/bin/activate; cd /mnt/e/Hexo-BotTrainer-gumbel && python -m pytest packages/hexfield -k 'gumbel or shard or window or expand' 2>&1 | tail -40"
```
**GO** iff: builds; with `gumbel_target=false` the export omits the new keys and shards/window/expand/loss behave exactly as today (byte-identical golden vectors); with `gumbel_target=true` the exported `gumbel_policy` sums to 1 over support and round-trips shard→window→expand→collate→loss; a MIXED old+new shard batch loads (old rows fall back to `visit`, new accepted by the bumped `SCHEMA_VERSION`).

---

## S6 — tests (cargo unit + python parity + smoke)

### Cargo unit tests (`packages/hexfield/rust/src` test modules)
1. **σ transform:** `sigma(q,max_n,50,1) == (50+max_n)*q`; monotone in q on `[-1,1]`; `sigma(0,..)=0`; boundaries `q=±1`.
2. **completedQ math:** hand-built tree — `completedQ(visited)=value_sum/visits`; `completedQ(unvisited)=v_mix` (prior-weighted over visited), NOT a child eval read; floored-out actions excluded from support.
3. **Gumbel-Top-k == sampling-without-replacement (concrete gate):** over ≥100 random logit vectors (varied length/entropy), draw top-m of `logits+g` for ≥10,000 Gumbel seeds each; compare empirical **first-pick** frequencies to analytic `softmax(logits)` via **chi-squared GoF, gate p>0.01**, with **≤5% of vectors allowed to fail** at α=0.01. Second-pick conditional frequencies match `softmax` renormalized over the remaining support (same gate).
4. **SH allocation:** per-round per-survivor budget `= floor(n/(R·|A_r|))`; survivors `= ceil(prev/2)`; final 1 survivor; total visits ≤ budget. **Intra-slot barrier:** construct a slot where one candidate fills early and assert NO premature halving (halving only after all survivors reach the round cap).
5. **Target softmax:** exported `gumbel_policy` sums to 1 over support and equals `softmax(logits+σ(completedQ))` against a reference.
6. **`Divergences::gumbel()` profile:** asserts the four bools ON and scalars at defaults; `parity()`/`production()` assert all four OFF.

### Python parity / transition tests
7. **Parity byte-identical:** golden-vector `Divergences::parity()` unchanged with all flags default-OFF; add an explicit case asserting `gumbel_target=false, gumbel_root=false, gumbel_nonroot_select=false, gumbel_sequential_halving=false` in parity mode.
8. **Mixed old/new shards:** a batch with old shards (no `prior_logit`/`gumbel_pol`, old `SCHEMA_VERSION`) AND new shards loads through `collate_training`/window/expand; absent-field fallback executes; old rows ⇒ `policy_target="visit"`; bumped `SCHEMA_VERSION` accepted by the reader (shards.py:223-224).
9. **Config load:** `hexfield_main_6.toml` parses; mis-placing `policy_target` under `[model.config.selfplay]` raises `ValueError` (guard test).

### Smoke
10. A short gated self-play smoke under `Divergences::gumbel()` (tiny budget, few games): runs Gumbel-Top-k root + SH + #2 non-root + #3 target export end-to-end without panics; produces a normalized `gumbel_policy` per Full row.

### Commands
```
# cargo unit (full)
wsl.exe -e bash -lc "export PATH=/root/.cargo/bin:\$PATH; cd /mnt/e/Hexo-BotTrainer-gumbel && cargo test -p hexfield 2>&1 | tail -60"
# python parity/transition/config
wsl.exe -e bash -lc "source /root/.venvs/hexfield-dev/bin/activate; cd /mnt/e/Hexo-BotTrainer-gumbel && python -m pytest packages/hexfield 2>&1 | tail -60"
```
**GO (final)** iff: all baseline (~22) + new cargo tests pass; all python parity/transition/config tests pass; golden vectors byte-identical with flags off; gated smoke completes. Then commit on `claude/hexfield-gumbel` (final phase only; no push).

---

## Build / test commands (established)
- **Build (release, dev venv) + mirror .so:**
  ```
  wsl.exe -e bash -lc "source /root/.venvs/hexfield-dev/bin/activate; export PATH=/root/.cargo/bin:\$PATH; cd /mnt/e/Hexo-BotTrainer-gumbel && maturin develop --release -m packages/hexfield/Cargo.toml 2>&1 | tail -50"
  wsl.exe -e bash -lc "cp \$(ls /root/.venvs/hexfield-dev/lib/python3.12/site-packages/hexfield/_rust*.so | head -1) /mnt/e/Hexo-BotTrainer-gumbel/packages/hexfield/python/hexfield/"
  ```
- **Cargo unit tests:** `wsl.exe -e bash -lc "export PATH=/root/.cargo/bin:\$PATH; cd /mnt/e/Hexo-BotTrainer-gumbel && cargo test -p hexfield 2>&1 | tail -40"` (if pyo3 linking fails, try `--no-default-features` or a `test` feature).
- **Python tests:** `wsl.exe -e bash -lc "source /root/.venvs/hexfield-dev/bin/activate; cd /mnt/e/Hexo-BotTrainer-gumbel && python -m pytest packages/hexfield 2>&1 | tail -60"`
- **Never** launch the supervisor/live run, touch `E:/Hexo-BotTrainer-hexgt` or the `hexgt-build` venv, or `git push`. Commit only in the final phase on `claude/hexfield-gumbel`.
