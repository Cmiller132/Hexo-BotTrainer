# hexfield — useful fixes & experiment backlog (from the 30-agent code-grounded review)

Status: **actionable extract**, 2026-06-13. Distilled from the 21-design competition
(`hexfield_v2_synthesis.md`) and its 30-agent code-grounded review. This doc lists only what the
review judged *useful*, with the evidence, the concrete code location, the change, the gate, and —
critically — **whether it can apply to the live `hexfield_main_1` run or only to the next fresh
run/lineage.**

> Line numbers are as of the review; reconfirm before editing (the package is under active dev).
> All runtime work is in the WSL `hexfield-dev` venv, never the live `hexgt-build` venv.

## TL;DR

| # | Fix / experiment | Applies to | Tier | Confidence it helps |
|---|---|---|---|---|
| 1 | Commit a real 512v throughput artifact (gate) | measurement | prereq | — |
| 2 | Measure length-band on the unbounded board (gate) | measurement | prereq | — |
| 3 | min-ECE BC→RL checkpoint selection | **next run** | implement | **high** |
| 4 | Port the 500-step LR warmup into the RL trainer | **next run** | implement | medium |
| 5 | `prefit.py` isfinite grad-norm logging fix | next BC run | implement | high (correctness) |
| 6 | Build a `value_weight` config path + down-weight value in BC | next run | optional | low/confounded |
| 7 | Keep moves-left utility ON (do **not** demote) | **live run** | decision | resolved |
| 8 | max-pool into value (A/B) | next run | experiment | low |
| 9 | axis-line convolution (A/B) | next run | experiment | low |
| 10 | threat-seeded tokens (A/B) | next run | experiment | low |
| 11 | FlexAttention serve perf (A/B) | next run / serve | experiment | low–med |
| — | graded planes, γ/focal, length-knees, graded-aux, tied-tap, D6-bias, packer-anchor, capacity | parked/dropped | — | — |

---

## 0. Critical context — do not break a healthy run

The live `hexfield_main_1` run (`configs/hexfield_main_1.toml`: 512v, PCR 128 @ 33%, divergences on,
`tss_enabled=true`, 200 epochs, warm-started `checkpoint_epoch2.pt` at line 74) is at ~epoch 14 and
**improving** (≈0.75 vs a fixed baseline) with the moves-left head **audited healthy** (conv-Spearman
0.68). Rising self-play loss is **benign** (longer games / non-stationarity), not degradation —
track strength with `scripts/_hexfield_fixed_baseline_arena.py`, not loss.

**Consequence:** none of the architectural changes and none of the curriculum fixes can be applied
to this run — the curriculum fixes target the BC phase + BC→RL handoff that are ~13 epochs behind,
and the architecture changes require a new model graph the live checkpoint can't continue into. The
correct action on the live run is **leave it running**; the fixes below are for the *next* fresh
run/lineage. The one genuinely live lever (moves-left) is already correctly set — see #7.

---

## 1. Prerequisite measurements (gate everything below)

Two load-bearing premises the roadmap leans on are **uncorroborated in-repo**. Resolve them before
spending any A/B slot, because every throughput/architecture verdict hangs on them.

### Measurement A — a real hexfield throughput artifact at production settings
- **Why:** the claim "5.82 pos/s @512v ≈ 1.2× dense, above the 0.8× floor" has **no committed
  hexfield artifact** behind it. `scripts/_hexfield_selfplay_throughput.py:85` hardcodes a *stale*
  reference (`restnet ~9.7 pos/s @256v` → 0.8× = 7.8 pos/s); the only perf logs in-repo
  (`_perf_512*.log`) are a 2026-06-02 hexgt-lineage profile. At 256v, 5.82 pos/s would be **below**
  floor; the "1.2×" depends on a matched-512v dense baseline that does not exist in-repo.
- **Do:** run `scripts/_hexfield_perf.py` (and/or the self-play throughput script) on a current
  hexfield checkpoint at 512v, and produce a 512v-vs-512v-dense comparison. Commit the artifact.
  Without it, the ≥0.8× floor status of the live model is unknown — and every "this graft costs
  ~3% / is throughput-neutral" claim (axis-line conv, graded planes, FlexAttention, capacity) is
  unverifiable.

### Measurement B — does the lengthening residual reproduce on the unbounded board?
- **Why:** the marathon disease was 47/47 verified as a radius-20 **crop** artifact
  (`docs/analysis/MAIN4_RECOMMENDATION.md`, crop-skip at `mcts_tree.rs:885-889`); hexfield deletes
  the crop (`support.py` BFS-9, `LEGAL_RADIUS=8`, no crop), so the primary driver is gone by
  construction. The `~+40 dec/game` *in-crop* residual (`MAIN4_RECOMMENDATION.md:34`) is the only
  motivation for the anti-lengthening grafts (γ/focal, length-knees) — and it is **unmeasured** on
  hexfield's unbounded board.
- **Do:** run `scripts/_hexfield_band_check.py` against the dense reference (game-length ratio target
  ~[0.5, 2.0]) on current production checkpoints. If length is in-band and improving, the
  anti-lengthening grafts have no problem to solve and stay parked.

---

## 2. Tier-1 fixes — near-certain, but **next fresh run only** (cannot retrofit the live run)

These address the one *measured* wound (BC value-calibration collapse) and the under-built BC→RL
transition. They are the highest-value items in the whole set — and none is an architectural graft.

### Fix 3 — min-ECE BC→RL checkpoint selection  ★ highest value
- **What:** select the BC checkpoint that RL warm-starts from by **minimum held-out `value_ece`**
  (and top-1), not by epoch index.
- **Evidence (verified from `runs/hexfield_bc_1/diagnostics.jsonl`):** held-out `value_ece` =
  **0.108 (ep0)** → 0.340 (ep1) → 0.299 (ep2), while train `value_CE` *falls* 0.666→0.549 and
  held-out `value_CE` *rises* 0.647→0.998 — textbook overfit on a value-poor signal (one bootstrap
  label shared by 60–200 correlated positions, no draws). The live run warm-starts **`epoch2`
  (ece 0.299)** — the second-worst-calibrated checkpoint — over `epoch0` (ece 0.108), for a top-1
  cost of only 0.018 (0.405 vs 0.387). *(Correction to earlier notes: the worst checkpoint is ep1
  at 0.340, not ep2; the argument is unchanged — min-ECE picks ep0.)*
- **Where:** `prefit.py:~377` saves every epoch with **no selection logic**;
  `configs/hexfield_main_1.toml:74` hardcodes the `initialize_from` path.
- **Change:** add min-`value_ece` (tie-break top-1) checkpoint selection to the prefit, and/or point
  the next run's `initialize_from` at `checkpoint_epoch0.pt`. Re-measure the BC→RL soak transition.
- **Gate:** none needed to ship the selection logic; the *benefit* shows in the next run's early-RL
  value calibration. **Land this before any value-head experiment** so the M3 `value_ece` gate stays
  attributable.
- **Caveat:** the M3 `value_ece ≤ 0.08` target may be **structurally unreachable** on this
  value-poor corpus (best-ever is 0.108 at ep0, pre-RL). Treat ≤0.08 as aspirational; the real test
  is RL recalibration on real outcomes.

### Fix 4 — port the 500-step LR warmup into the RL trainer
- **What:** add a linear LR warmup to the RL trainer's first ~500 steps, triggered on the BC→RL
  handoff. **Port the warmup ALONE — drop** the synthesis's bundled "head-LR re-warm" + "value-only
  warm pass" (unmotivated, non-bisectable scope creep).
- **Evidence:** `config.py:56` `warmup_steps=0` is **dead/unread**; the warmup logic lives only in
  `prefit.py` (`WARMUP_STEPS=500`, applied `prefit.py:~337-340`); `trainer.py` (`train_passes`,
  ~74-149) has **no LR/param-group scaffolding** and applies full `lr=1e-3` from step 1.
  `initialize_from` loads **weights only** (no optimizer state), so RL begins with BC-converged
  weights + a fresh-zeroed AdamW + full LR — a large unconditioned update at the most fragile moment.
- **Change:** port the linear warmup into `trainer.py`, **gated on optimizer-state-not-loaded** (NOT
  `global_step==0`, or genuine mid-run resumes get spuriously re-warmed — use the existing
  meta/resume branch as the gate). Add a first-500-RL-step `clip_fraction` tripwire.
- **Gate:** re-measure the soak transition; confirm no early-RL `clip_fraction≈1.0` storm.
- **Note:** no transition *damage* has been measured — this is preventive hardening, not a fix for a
  known break.

### Fix 5 — `prefit.py` isfinite grad-norm logging fix  (the "NaN storm" is a logging artifact)
- **What:** make the grad-norm log append conditional on `torch.isfinite`, mirroring the guard the
  RL trainer already has.
- **Evidence:** `prefit.py:~139` appends `float(grad_norm)` **unconditionally** on AMP-skip steps →
  `grad_norm_mean` logs NaN/Inf; meanwhile `grad_norm_p95` is a **healthy 4.0–4.2** and `amp_scale`
  *climbs* 8192→32768. The RL path `trainer.py:~126` is **already guarded**. So the "NaN/Inf storm"
  is a **metrology artifact**, not instability — and "harden GradScaler" from the synthesis is
  cosmetic. The real value-head problem is the label-correlation overfit (Fix 3), not NaNs.
- **Where:** `prefit.py:~139` (vs the guarded `trainer.py:~126`).
- **Change:** one-line `if torch.isfinite(grad_norm):` guard around the append. Make `nan_trips` a
  counted metric so a *real* future non-finite step is visible.
- **Gate:** none. Pure correctness; affects BC diagnostics only (RL path already correct).

### Fix 6 — (optional) build a `value_weight` config path + down-weight value in BC
- **What:** expose the value loss weight as a config/CLI knob and set it ~0.25–0.5 during BC.
- **Evidence / why it's only optional:** there is **no `w_value` knob** — `VALUE_WEIGHT=1.0` is
  hardcoded (`losses.py:27`, kwarg at `losses.py:197`), `prefit.py:129` calls `hexfield_loss` with
  no override, and `prefit.py` argparse (~283-292) exposes no flag. So this is an **unbuilt code
  path**, not a config flip. The rationale (policy is the BC product; MCTS + the engine verdict
  carry tactical value; RL relocates the scalar) is sound, but the value-head behavior is
  **confounded** with the corpus's value-poverty, so the benefit is uncertain.
- **Change:** thread `value_weight` through `TrainingSection` (`config.py`) → `prefit.py` →
  `hexfield_loss`. Default stays 1.0; set 0.25–0.5 only in the next BC config.
- **Gate:** next-run BC `value_ece` + top-1; revert if top-1 regresses. Lower priority than Fix 3 —
  min-ECE selection captures most of the calibration benefit without touching the loss.

---

## 3. Live-run guidance (`hexfield_main_1`, ~ep14)

### Fix 7 — keep the moves-left utility ON (do **not** demote to default-OFF)
- **Decision (reversed from the review's initial call):** the review's "demote to OFF" was premised
  on a *flood-damaged* head. The live run has now **disproven that premise** — the moves-left head
  was audited **healthy at ep14** (conv-Spearman 0.68, above the ≥0.6 bar; auto-disable armed). This
  is exactly the spec's counter-argument (`STAGE3_MOVES_LEFT_FEASIBILITY`): hexfield trains on clean
  targets from BC onward, so the blocker is **absent**. The open question is resolved **in favor of
  keeping it ON**.
- **Action:** no change. Continue monitoring the game-length drag from the two-sided form and the
  per-epoch L0 / nightly control-flip probe; rely on the auto-disable rather than a manual flip.

### Do not react to loss drift
Self-play loss rising = longer games (non-stationary), not degradation; the model is improving
(ep13 > ep5). Do not introduce anti-lengthening changes (γ/focal, length-knees) to "fix" it — see §5.

---

## 4. Candidate A/B experiments — next run, one at a time, isolated, gated

These *might* help but the play-strength delta is **unknown statically** and likely below the noise
floor of the current ~2000-row soak signal (arena within ~1.3σ of 50%). Run **at most one at a
time**, off the trusted backbone, **after** Measurements A+B, and never stacked on the value head.

### Experiment 8 — masked max-pool into the value/aux head
- **What:** value/aux input = `concat(T0, T1, mean_pool, max_pool)` = 384 → `Linear(384→96)` → …
- **Why (and the doubt):** sudden-death value hinges on the single worst window, which mean-pool over
  600–1500 cells attenuates. **But** the value head already reads `tokens[0],tokens[1]` after 3
  global-attention passes alongside `_pooled` (`model.py`), and the engine verdict **hard-overrides**
  exactly the one-window-decides regime — so the realizable signal is quiet-regime only; likely
  redundant.
- **Cost / landmine:** +~9k params; **mandatory fp32 `-inf` pad-fill before the max** (a missing fill
  leaks padded zeros into the max and silently corrupts value on small-N rows). Add a pad-inertness
  unit test to the M1/M2 exactness trio.
- **Gate:** isolated arena A/B (separated from Experiment 10's tokens); pad-inertness test must pass.

### Experiment 9 — depthwise axis-line convolution (×2)
- **What:** two depthwise 11-tap on-axis strip convs (one per Q/R/QR) + a C×C recombine; zero-init
  residual; rot60-tied filters. ~25k·N MACs (≈3% of the linear term *if the cost model holds*).
- **Why (and the doubt):** a 6-window spans 5 steps, so an O(1)-depth on-axis stencil reads it
  directly. **But** the long-range same-line relationship is *already* a first-class primitive via
  the on-axis relpos bias buckets + 3 attention blocks, and tactical completion is the engine's job.
- **Cost / landmine:** **needs its own composed on-axis index** `axis_idx(B,Npad,3,11)` built once
  per batch in Rust — `support.py`'s neighbor table is **radius-1/6-direction only**, so dilated
  gathers would reference off-support cells. The strip must be clipped to support (off-support taps →
  pad-zero, engine-correct), parity-pinned vs the engine window scan. The index-build cost is an
  **unmeasured O(N) serve tax**.
- **Gate:** M0 on-axis-index parity fixture + M1 oracle vs a stacked-isotropic reference + **its own
  M8 throughput re-measure (index build priced)** against the ≥0.8× floor + an arena A/B with a
  binary-plane lesion arm (to separate it from the graded-planes experiment).

### Experiment 10 — threat-seeded per-axis summary-token init
- **What:** initialize 3 of the 8 summary tokens as Q/R/QR aggregators pooled from existing
  hot/win-now cells; keep ≥1 learned free slot; empty-set → learned-constant fallback.
- **Why (and the doubt):** points global state at live threats from step 0; the free slots exist
  (tokens[4..7] are spare). **But** it's an **init-only** change — tokens drift off init within ~1k
  steps, making it nearly un-A/B-able, and three independent learned tokens are *not* permutation-tied
  so no D6 guarantee is gained.
- **Gate:** lowest priority of the three; an init-only change is hard to attribute even with a run.

### Experiment 11 — FlexAttention (serve-path perf)
- **What:** inline `score_mod` bias (reusing the closed-form hex index) to delete the materialized
  `(B,4,S,S)` bias-gather transient + enable jagged batching. **Numerically equivalent** to the
  current attention — so it does *not* change the model and is the only item here that could be
  trialed without altering the learning trajectory.
- **Why (and the doubt):** removes the ~10–15% bias-**materialization** wall-clock tax. **It does NOT
  change the O(S²) attention math** (≈33%/45% of serve MACs at N≈900/1500 — irreducible). The ≥10%
  headline is uncorroborated.
- **Gate:** oracle-equivalence vs the materialized impl + **≥10% measured win on the actual WSL/Triton
  stack, measured against the *already-live* `torch.compile(dynamic=True)`/fused-SDPA baseline (not
  eager)**. Keep the sdpa/materialized dual path as the always-correct default. Do not inject into the
  live production run; trial on a measurement clone.

---

## 5. Explicitly parked / dropped — do **not** implement (with reasons)

Recorded so these are not re-litigated:

- **Graded per-axis input planes (F=15→21)** — *park.* Owner already parked this (spec §12.7,
  "largest representational risk surface"; standing-win planes are "their safe slice"). It is the one
  graft that **breaks the byte-exact wire ABI** (`NUM_FEATURES=15` hardcoded both sides + parity
  fixture), count==4/5 is engine-overridden and already in binary planes 13/14, and it widens the
  stem at the fragile value-collapse boundary. Revisit only if a backbone-only baseline first clears
  the value-ece gate.
- **γ<1 value discount + sudden-death focal weighting** — *park.* Changes the value-target family
  (`samples.py:~148` is currently hard_z/root_value) to fight a **crop-era disease hexfield deletes**;
  MAIN4 attributes the residual to search/conversion, **not** value-myopia; γ<1 risks myopia about
  genuine long forced wins; and it would be a forbidden stacked shortening pressure with moves-left.
  Do not run until Measurement B shows a length problem actually exists on the unbounded board.
- **Promote C1 replay length-decay knees** — *park.* **Not config-only** as billed: hexfield has no
  policy-surprise row-weighting layer (`samples.finalize` uses pure hard_z, no reweight hook), so
  promotion requires porting machinery; and it targets the same cured crop-flood disease.
- **Train-only per-cell graded-potential aux head** — *drop.* Near-tautological (predict the per-axis
  fill you are fed as input), a contingency-on-a-contingency on the parked graded planes, hangs off
  the aux tower that got **zero** STV loss in BC, per-cell (not free), and unattributable at weight
  0.05.
- **Axis-permutation tied-tap HexNodeConv** — *park/drop.* Its gate has **not fired and won't**:
  `probe_d6_kl` is 0.093→0.064 and **falling** (augmentation is converging). It cuts directional conv
  capacity in a width-locked C=96 trunk for equivariance already achieved free.
- **D6-orbit-tied bias table** — *park/drop.* Same dead gate as tied-tap; hard-codes what augmentation
  already approximates while removing real anisotropy capacity.
- **Packer closed anchor list** — *park.* ~90% already shipped (`inference.py` `plan_groups`:
  `QUANT_NODES=64`, `WASTE_FRACTION=0.18`, `PAIR_CEILING=3.8e7`); the incremental closed-anchor-list is
  redundant with the live `torch.compile(dynamic=True)` and conflicts with the FlexAttention path.
- **Capacity contingency (depth / GatedResBlock / masked-BN)** — *park.* Correct **as a contingency**
  (fires only if M3/M9 underlearn after experiments 8/9); width correctly frozen at C=96 (conv MACs
  scale C²·N and dominate at the median). If it ever fires: add a 7th/8th plain conv block first; the
  GatedResBlock arm (+~50% conv MACs) is a median-N throughput regression and must sit **below** the
  plain-block arm; masked-BN only on a 2-of-3 quantitative trigger and never without re-deriving the
  exactness trio for the BN path.

---

## 6. Sequencing checklist

1. **[ship now, next-BC/next-run]** Fix 5 (isfinite one-liner) — trivial, unblocks honest grad
   metrology.
2. **[next run]** Fix 3 (min-ECE checkpoint selection) — isolated; land before any value-head work.
3. **[next run]** Fix 4 (LR-warmup port, alone) — re-measure the soak transition.
4. **[before any A/B]** Measurement A (512v throughput artifact) + Measurement B (length band).
5. **[live run]** Fix 7 — confirm moves-left stays ON; keep monitoring. No other live change.
6. **[next run, gated]** At most ONE of Experiments 8 / 9 / 10, isolated, value head re-stabilized
   between each; Experiment 11 only as a measured serve-perf clone.
7. **[parked]** Everything in §5 — leave parked/dropped; revisit only against their named triggers.

Cross-refs: `docs/specs/hexfield_v2_synthesis.md` (roadmap + review verdict),
`docs/specs/hexfield_model_spec.md` (as-built), `docs/analysis/MAIN4_RECOMMENDATION.md`,
`docs/analysis/STAGE3_MOVES_LEFT_FEASIBILITY.md`.
