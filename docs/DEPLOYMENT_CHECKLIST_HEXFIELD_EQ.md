# DEPLOYMENT CHECKLIST — hexfield_eq (D6-equivariant rewrite): prefit ladder → self-play soak

Status: ACTIVE checklist. Date: 2026-07-08. Companion to
`docs/PLAN_D6_EQUIVARIANT_REWRITE.md` (Phases 4–6) and
`docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md` (§3 arm ladder). Owner decisions
baked in: prefit corpus = **main_11 self-play data** (not the HF corpus);
width **C=192** (`GROUP_ORDER=12`, `C_ORBIT=16`, heads 3); depth 11; arms
per §3 below.

Everything here is scaffolded and validated up to (but not including) GPU
launches. Work through the sections in order; each has a gate.

---

## 0. Verified state (2026-07-08) and what it rests on

| Item | State |
|---|---|
| main_11 corpus | `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_11/samples/epoch_000001..000017` — 17 epochs × 512 game shards = **8,704 shards, ~309 MB**, schema v3, **100 % of rows carry the Gumbel completedQ target** (sampled; `gumbel_present=1`) |
| Shard converter | `scripts/convert_hexfield_shards_to_eq.py` — pure-numpy column transform (drop 8 hot/win arrays, version 3→4, all else **byte-exact**; deliberately not an object round-trip, which would zero `policy_surprise`). Idempotent; `--limit N` sampling; deterministic val split `game_key % 20 == 19` (~5 %, game-level) |
| Conversion validated | 2-shard sample converted + `scripts/validate_eq_shard_conversion.py` **PASS** under WSL (`hexgt-build` venv): byte-identical columns, eq-reader row equivalence v3 vs v4, 25-plane expand, `raylen (N,12) u8` present in ExpandedRow **and** threaded by `collate_training`, expanded policy/value/gumbel targets identical |
| `config.py` anchors (D4) | **Already correct** — `MultiStageEvalOpponents.permanent_anchors = ()` and `radius8_opponents = ()` with the D4 rationale in-line (`packages/hexfield_eq/python/hexfield_eq/config.py`) — no fix needed |
| Checkpoint meta round-trip | `model.arch_meta()` persists `group_order / c_orbit / channels / in_channels / attention_heads / trunk_layout / num_tokens / feature_width / equivariant / reg_lane / reg_tok_read / bias_reduction`; `checkpoints.py` saves it under `"meta"`; `infer_net_kwargs_from_state_dict(sd, meta)` reads meta first. **Gaps in §1 below.** |
| Prefit entry point | `hexfield_eq.prefit` CLI: `--data <dir with train/ + val/ of shard_*.npz> --out <run dir> --epochs --workers --seed [--limit-steps] [--resume]`; expand is identity-symmetry for `GROUP_ORDER>1`; param no-decay predicate already covers `bias_free_table`/`bias_theta`/`gate_bias` |
| Scaffolds created | `configs/hexfield_eq_main_1.toml`, `scripts/prefit_env/hexfield_eq_arm{1_vanilla,2_reglane,3_tokread,4_raylayout,4c_georay}.env`, `scripts/_hexfield_eq_supervise_main1.sh`, `scripts/systemd/hexfield-eq-supervisor-1.service` |

---

## 1. MUST-BE-GREEN-FIRST gaps (blockers, in priority order)

### B1 — prefit trains the policy head on the WRONG target for main_11 data (CRITICAL)

`hexfield_eq/losses.py::hexfield_loss` selects the Gumbel π′ target only when
called with `policy_target="gumbel"`; its default is `"visit"`.
`hexfield_eq/prefit.py` calls it **without** the argument in BOTH call sites
(`run_step` and `evaluate`) — so a prefit on main_11 data would do behavior
cloning of the **SH visit histogram**, which under Gumbel Sequential Halving
is a schedule artifact (equal per-round quotas). That is precisely the diffuse
target that caused the main_10 regression; 100 % of main_11 rows carry π′
(`gumbel_present`), so this affects every row.

Already correct without changes: `collate_training` packs
`gumbel_policy`/`gumbel_policy_valid` and builds the **soft-policy** target
from π′; the stored **opp_policy** target already prefers π′ (writer-side);
`prior_logit` is stored. Only the main-policy CE and the eval targets miss.

**Fix LANDED (2026-07-08, orchestrator):** `prefit.py` now takes
`--policy-target {visit,gumbel}` (default `visit`), threads it into
`hexfield_loss(...)` in BOTH `run_step` and `evaluate`, and `evaluate`'s top-1
target uses the same per-row π′ blend (`gumbel_policy_valid > 0`) as the loss.
Ladder launches must pass `--policy-target gumbel`.

Gate still to run before the ladder: a 1-shard `--limit-steps` smoke shows
`components["policy"]` responding to the switch (loss value changes vs
`visit`).

### B2 — prefit checkpoints carry no arch meta

`prefit.py::save_checkpoint` writes
`{model, optimizer, scaler, epoch, global_step}` — no `"meta"`. The trainer
path (`checkpoints.py`) writes `"meta": {lineage, epoch, **arch_meta}` and
every foreign loader (eval arena, dashboard) is meta-first. Arm checkpoints
without meta cannot be arch-rebuilt except via matching env — fragile across
5 arms with different `TRUNK`/`REG_*`.

**Fix LANDED (2026-07-08, orchestrator):** `prefit.save_checkpoint` now writes
`"meta": {"lineage": "hexfield_eq", "epoch": int(epoch), **model.arch_meta()}`
(on the eager `model`; the compiled wrapper is never passed to it).

### B3 — arms 4/4c blocked on Phase L landing — **RESOLVED (2026-07-08)**

Phase L0 + L1 are LANDED and gate-green: trunk grammar accepts `L`
(`CCLACCLACLA` validated), `HEXFIELD_EQ_RAY_BLOCKERS` exists (default 1; the
4c control sets 0 — a model-side mask variant, spec D-S16, so the same shards
serve both arms), `RayAttnBlock`/`head_perm6` landed with the full-net
L-layout equivariance gate (all 12 g, lane on and off), materialized ≡ flex
parity, empty-ray safety, and the K-slot-trap structure test
(`tests/test_hexfield_eq_ray_block.py`). `arch_meta` carries
`ray_blockers`/`ray_heads`; `prefit.load_checkpoint` asserts them on resume
(spec D-S32). Guards: an L-layout run RAISES if the expand kernel emits no
raylen or the first batch's raylen is all-zero over live cells (spec D-S31);
`PAIR_BUDGET` auto-drops to 8.0e6 for L layouts (`HEXFIELD_EQ_PAIR_BUDGET`
overrides — spec D-S30). Arms 4/4c are launchable once the R arms report.

### B4 — the Phase-4 gate reference needs an owner decision

`PLAN_D6_EQUIVARIANT_REWRITE.md` Phase 4 gates on "held-out top-1 within
tolerance of the current hexfield BC reference **on the same split**" — but
that reference was defined for the HF corpus, and the owner switched the
prefit corpus to main_11 self-play data. Options:
  (a) run a matched hexfield-lineage (15-plane, c=128) BC prefit on the SAME
      converted split as a cross-arch reference (extra GPU run), or
  (b) treat **arm 1 (vanilla eq)** as the ladder-internal reference and gate
      arms 2–4 against it (the R1/L2 gates), gating arm 1 itself only on
      absolute sanity (AMP no-NaN, `value_ece ≤ 0.08`, top-1 in a plausible
      band) plus the eval re-anchor result vs Strix/SealBot.
Recommendation: (b), with (a) only if arm 1's absolute numbers look
ambiguous. **RESOLVED (2026-07-08): owner signed off on (b)** — arm 1 is the
ladder-internal reference; arms 2–4 gate against it; arm 1 gates on absolute
sanity (AMP no-NaN, `value_ece ≤ 0.08`, plausible top-1) + the Phase-5
re-anchor result vs Strix/SealBot. Option (a) stays available if arm 1 looks
ambiguous.

### B5 — support radius must be pinned everywhere (silent-OOD trap) — **HARDENED (2026-07-08)**

The eq default is `HEXFIELD_EQ_SUPPORT_RADIUS = LEGAL_RADIUS = 8`
(`support.py`), but main_11 data was generated at radius 4 and the arm env
files / service unit set **4**. Every eq entry point (prefit, validation,
standalone eval, serve, tests against these checkpoints) MUST set
`HEXFIELD_EQ_SUPPORT_RADIUS=4` explicitly. The trap is now guarded (spec
D-S26): `arch_meta()` records `support_radius`; `checkpoints.load_into` and
`prefit.load_checkpoint` REFUSE a checkpoint whose recorded radius mismatches
the build; the Rust serve payload stamps `support_radius` and
`inference.submit_payload` asserts it; all three env readers (Python support,
expand backends, Rust) clamp identically to `[1, HALO_DIST]`. An unset env
still builds radius-8 — the guards turn the silent OOD into a loud error at
first load/serve, not a prevention. Keep the env pinned in every unit/log.

### B6 — smaller known items

- **Standalone eval runner is hexfield-bound** — **RESOLVED (2026-07-08,
  Phase L3 pass)**: `scripts/_hexfield_eq_run_multistage_eval.py` is the eq
  twin (eq PYTHONPATH incl. shared packages, eq config parsing, D4 anchor
  policy — Strix + SealBot pool, `--no-strix`/`--strix-ckpt` overrides, a B5
  radius warning, and asdict-based override merging so new toml keys are
  never silently dropped on the CLI round-trip). Verified: `--help` clean +
  `--dry-run` roster/allocation against `configs/hexfield_eq_main_1.toml`
  under the WSL hexgt-build venv. Same CLI surface as the hexfield twin
  (`--parts` / `--opponent` / `--aggregate-only` all ride the eq
  `multistage_eval` orchestrators).
- **Strix anchor ckpt lives in Downloads** — **RESOLVED (2026-07-08)**:
  COPIED (sha256-verified) to the stable
  `/mnt/e/Hexo-BotTrainer/anchors/strix/checkpoint_00237000.pt` and
  `configs/hexfield_eq_main_1.toml` now points there. The original
  `/mnt/c/Users/epicm/Downloads/checkpoint_00237000.pt` still exists as a
  backup. `hexfield_main_11.toml` was repointed to the stable path during the
  2026-07-09 repo declutter (no run was live); the other lineage tomls that
  referenced Downloads were removed in that same cleanup.
- **Serve-evaluator raylen threading (spec D-S18) — LANDED (2026-07-08,
  spec D-S34)**: L-layout nets serve through `inference.py` on every path
  (CSR pack, Rust pack, copy-stream, CUDA-graph statics, compiled dynamic);
  C/A nets are byte-identical to before (sha256 baseline). Gates in
  `tests/test_hexfield_eq_serve.py` (CPU green; CUDA gate green on the
  training host, 2026-07-08).
- **Equal-TIME eval leg (ray plan §3 L2/L3, spec D-S35) — LANDED
  (2026-07-08)**: per-arm visit calibration approximating a per-move
  wall-clock budget (the Rust core has no native time budget — documented
  approximation, see D-S35's limits). Knob:
  `multi_stage_eval.eval_time_budget_ms` (default 0.0 = off) or
  `play_checkpoint_match(..., time_ms_per_move=...)`. The arm-4 equal-time
  A/B leg is a standalone `play_checkpoint_match` (arm4 vs best-R-arm ckpts)
  with `time_ms_per_move` set to the measured C-layout ms/move; calibration
  records land in `meta.time_calibration_{a,b}`. Gates in
  `tests/test_hexfield_eq_time_budget.py`.
- **Env-prefix collision**: only ARCH knobs are `HEXFIELD_EQ_*`; perf/kernel
  knobs (`HEXFIELD_TRITON_*`, `HEXFIELD_SERVE_HALF`, `HEXFIELD_TRAIN_*`,
  `HEXFIELD_SERVE_FLEX`, ...) kept the bare prefix. Never rely on different
  values in a process importing both lineages (dashboard debug worker!).
- **Dashboard arch inferer**: the plan requires BOTH inferers meta-first; the
  eq package side is done — verify the dashboard's `debug_infer.py` can load
  eq checkpoints (or defer: not needed for the prefit ladder).
- **fp8 conv stays OFF** for v1 (plan §0 gotcha 1) — never set
  `HEXFIELD_CONV_FP8` on an eq service.

### B7 — 2026-07-08 pre-prefit fix bundle (LANDED; changes param counts + prefit regime)

The consolidated adversarial-review fixes are in
(`docs/SPEC_REGISTER_LANE_RAY_ATTENTION.md` §0.1, D-S20..D-S33). Ladder-visible
consequences:

- **State-dict shapes changed (D-S20/D-S21):** widened invariant head reads
  (`inv_read` 4× expansion, value reads all 6 tokens + the pre-ln token mean;
  per-cell heads get 2× expansions). Any eq checkpoint written BEFORE this
  bundle will NOT strict-load; there are none that matter (no prefit has
  run). Param counts at C=192: arm1 665,564 / arm2 (+lane) 764,948 /
  arm3 (+tok_read) 913,172 / arm4 `CCLACCLACLA`+lane+read 858,650 /
  arm4b `CCLACCLACCA`+lane+read 876,824.
- **Register-lane dynamics (D-S22/23/24):** out_proj near-zero init (grads
  live at step 0), gate_bias −2.5, learnable per-refresh `sum_scale`; watch
  `grad_norm_trunk_reg` + `token_stream_max` in diagnostics.
- **Prefit regime (D-S25):** cosine LR → LR/10, GRAD_CLIP 3.0, EMA twin
  (0.9995) with `ema_*` metrics — gate reads should prefer the EMA row when
  raw/EMA disagree late in the run; new diagnostics: per-group grad norms,
  `train_val_*_ce_gap`, `value_mae_ply_*`, `threat_*`; `--soft-policy-weight`
  CLI for the 4:1 soft-policy dominance probe (default 4.0 unchanged).
- **Resume safety (D-S32):** prefit resume asserts trunk_layout / reg_lane /
  reg_tok_read / ray_blockers / support_radius from meta.
- **Width contingency pre-authorized (D-S33):** if arm 1 underfits, C=288
  (`HEXFIELD_EQ_CHANNELS=288`, `HEXFIELD_EQ_C_ORBIT=24`, head_dim 96) is
  permitted — imports warn loudly that the bespoke Triton attention fast path
  will not engage (sdpa/flex serve only). Budget the serve ladder accordingly.

---

## 2. Full-corpus conversion (can run NOW; CPU-only, idempotent)

```bash
# WSL. ~8,704 shards; pure numpy. Sample already converted (idempotent skip).
/root/.venvs/hexgt-build/bin/python \
    /mnt/e/Hexo-BotTrainer-hexgt/scripts/convert_hexfield_shards_to_eq.py \
    --src /mnt/e/Hexo-BotTrainer/runs/hexfield_main_11/samples \
    --out /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit_main11
```

Expected: ~8,704 shards → `train/` ≈ 95 %, `val/` ≈ 5 %; ~0.9–1.0 M rows
(main_11 wrote ~50–56 k rows/epoch × 17); `gumbel_present` ≈ 100 %.

Then the FULL validation (column + reader checks on every pair, expand on the
first):

```bash
HEXFIELD_EQ_SUPPORT_RADIUS=4 HEXFIELD_EQ_CHANNELS=192 \
PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield_eq/python \
/root/.venvs/hexgt-build/bin/python \
    /mnt/e/Hexo-BotTrainer-hexgt/scripts/validate_eq_shard_conversion.py --all
```

**Gate:** `RESULT: PASS`, shard count matches the converter's summary, and the
printed `NUM_FEATURES=25 / SUPPORT_RADIUS=4` line is correct.

## 3. Prefit ladder (after B1 + B2 land; R arms first, per L7)

Data: `/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit_main11` (above).
Outputs: `/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit/<arm>/`.
All arms share `--data` and `--seed 1` (same split + shuffle streams); only
the env block differs. Run serially on the training GPU (do not overlap with
a live run). Epochs: start with 4 (prefit convention); extend only if the
diagnostics still improve epoch-over-epoch.

Per arm (arm 1 shown; arms 2/3/4/4c substitute their env file + out dir —
usage header inside each env file):

```bash
set -a; source /mnt/e/Hexo-BotTrainer-hexgt/scripts/prefit_env/hexfield_eq_arm1_vanilla.env; set +a
PYTHONPATH=/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield_eq/python \
/root/.venvs/hexgt-build/bin/python -m hexfield_eq.prefit \
    --data /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit_main11 \
    --out  /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit/arm1_vanilla \
    --epochs 4 --workers 6 --seed 1 \
    --policy-target gumbel        # <- exists only after B1 lands
```

Smoke first (once, arm 1): add `--limit-steps 30` and confirm one epoch loop,
finite losses, a checkpoint + `diagnostics.jsonl` + probe npz in the out dir.

Order and gates (from the plans — R1/L2 in
`PLAN_REGISTER_LANE_RAY_ATTENTION.md` §3; Phase 4 in the eq plan):

| # | Arm | Env file | Gate | Kill criterion |
|---|---|---|---|---|
| 1 | vanilla `CCCACCCACCA` | `arm1_vanilla.env` | AMP run no NaN; `value_ece ≤ 0.08`; top-1 vs the B4 reference decision; probe harness writing | **Project go/no-go** (eq plan Phase 4): a miss = stop, diagnose features vs equivariance vs width — no soak |
| 2 | + `REG_LANE=1` | `arm2_reglane.env` | ≥ arm 1 on held-out top-1 AND `value_ece`; look for the value/calibration win | arm 2 ≤ arm 1 on both → `REG_LANE=0`, stop lane arms |
| 3 | + `REG_TOK_READ=1` | `arm3_tokread.env` | ships only if > arm 2 | ≤ arm 2 → drop |
| 4 | `CCLACCLACLA` (+ lane verdict) | `arm4_raylayout.env` | ≥ best R arm (after B3 + Phase L0/L1 gates green) | L regresses → stay on `CCCACCCACCA` |
| 4c | + `RAY_BLOCKERS=0` control | `arm4c_georay.env` | attribution only (4 vs 4c separates blocker semantics) | — |

Read per-epoch numbers from `<out>/diagnostics.jsonl`
(`top1`, `value_ece`, `value_ce`, `policy_ce`, `probe_*`). The winner is the
last arm that beat its predecessor. Record the winner's env + best-epoch
checkpoint path.

## 4. Eval re-anchor (D4; eq plan Phase 5)

- `permanent_anchors = ()` — **verified in `config.py` defaults** and pinned
  as `permanent_anchors = []` in `configs/hexfield_eq_main_1.toml`. Never
  point the eq arena at 15-plane main5/main6 checkpoints (stem 25 vs 15
  always fails strict load).
- Pool = **Strix** (`strix_enabled=true`, sims 512 = candidate budget,
  full BT weight) + **SealBot** (`SEALBOT_PATH=/mnt/e/SealBot`, 0.5
  over-dispersion) + **self-anchors** (log-grid bracket accumulates once the
  run has epochs; Stage D tolerates the empty permanent list from epoch 0).
- Optional pre-soak sanity: an eval of the prefit winner vs Strix/SealBot via
  the eq standalone runner (B6 first item — landed):
  ```bash
  HEXFIELD_EQ_SUPPORT_RADIUS=4 \
  PYTHONPATH=... /root/.venvs/hexgt-build/bin/python \
      scripts/_hexfield_eq_run_multistage_eval.py <run_dir> <winner_ckpt> \
      --config configs/hexfield_eq_main_1.toml [--dry-run first]
  ```
- Arm-4 EQUAL-TIME leg (spec D-S35): after the fixed-visit L2 verdict, one
  A/B at `eval_time_budget_ms` (or `play_checkpoint_match(...,
  time_ms_per_move=)`) sized to the C-layout arm's measured ms/move at 512
  visits — this charges arm 4's flex-path latency instead of hiding it.
  Compare the pentanomial verdicts of the fixed-visit and equal-time legs.

**Gate:** a full multi-stage eval completes against Strix + SealBot and writes
`diagnostics/eval_pool.json`; no `dropped_anchors`; no `featurized_ood`
annotations (radius pinned per B5).

## 5. Serve + service (eq plan Phase 5 gates)

1. Set the winner's arch in `scripts/systemd/hexfield-eq-supervisor-1.service`
   (`HEXFIELD_EQ_TRUNK`, `HEXFIELD_EQ_REG_LANE`, `HEXFIELD_EQ_REG_TOK_READ`)
   — it must equal the `initialize_from` checkpoint's `arch_meta`.
2. Set `[checkpoint] initialize_from` in `configs/hexfield_eq_main_1.toml` to
   the winner checkpoint.
3. First launch runs the CONSERVATIVE serve profile (the unit ships with the
   Triton/flex/half/compile block commented out; the eq supervisor script
   defaults `HEXFIELD_SERVE_FLEX=0`).
4. Kernel enable ladder, one knob at a time, each behind the **serve parity
   gate (3e-3)** for the tied trunk + a throughput measurement:
   `HEXFIELD_SERVE_FLEX` → `HEXFIELD_TRITON_ATTN`/`HEXFIELD_FLEX_PAIR` →
   `HEXFIELD_TRITON_CONV`/`HEXFIELD_TRITON_CONV_LN` → `HEXFIELD_SERVE_HALF`
   (dense-weight cache must regenerate once, not per forward — plan §0
   gotcha 2) → `HEXFIELD_TRAIN_COMPILE`/`HEXFIELD_TRAIN_FLEX`. Never
   `HEXFIELD_CONV_FP8`. L-layout note (spec D-S17/D-S34): L blocks ride the
   plain flex carrier or the materialized path only — `HEXFIELD_TRITON_ATTN`
   / `HEXFIELD_FLEX_PAIR` accelerate the A blocks and never apply to L; the
   evaluator threads raylen on every path (graphs/half/compile included), and
   raylen stays u8 under `HEXFIELD_SERVE_HALF`. A gathered L kernel is
   deferred with a design sketch (spec D-S36, env `HEXFIELD_EQ_TRITON_RAY`
   reserved, default off).
5. Install + start:
   ```bash
   sudo cp /mnt/e/Hexo-BotTrainer-hexgt/scripts/systemd/hexfield-eq-supervisor-1.service /etc/systemd/system/
   sudo systemctl daemon-reload && sudo systemctl start hexfield-eq-supervisor-1
   ```

**Gate:** serve parity (3e-3) holds on every enabled knob; live serve
throughput within budget; `torch.compile(dynamic=True)` + CUDA-graph capture
clean with tied-weight generation (plan Phase 5).

### §5 addendum — fast profile ENABLED (2026-07-09)

The conservative eager launch measured **4.2 pos/s** (main_11 reference ~50).
The full profile was flipped on as a bundle after re-running the gates at the
arm-4 arch (`CCLACCLACLA`, lane on, C=192, radius 4):

- `tests/test_hexfield_eq_triton_ray.py` — 26/26 green (CPU + CUDA kernel
  parity at 3e-3, head_dim 32).
- `tests/test_hexfield_eq_serve.py` — 7/7 green **with the full fast-profile
  env set** (SERVE_FLEX + FLEX_PAIR + TRITON_CONV/ATTN/CONV_LN +
  EQ_TRITON_RAY + SERVE_HALF + RUST_PACK + COPY_STREAM), CUDA leg included.
- Ray-kernel bench (`scripts/bench_eq_ray_kernel.py`, blockers on, fp16):
  kernel beats the materialized path **5–33×** (B=24–48, Npad=256–512:
  8.5–33×) and flex 1.1–2.1×; the (B, Npad, 32) gather-index build is ~1 ms
  once per forward, shared by all 3 L blocks.
- Two KNOWN test artifacts at the arm-4 arch env (NOT regressions; both pass
  at the C=96/radius-8 default the suite was calibrated on):
  `test_hexfield_eq_ray_block.py::test_full_net_equivariance_L_layout[reg_lane]`
  trips its fixed `ATOL=1e-4` at C=192 (measured max|diff| 2.4–3.4e-4, pure
  fp32 accumulation growth — radius has zero effect; no symmetry break) and
  `test_hexfield_eq_raylen_parity.py` train-parity fixtures generate 0 window
  rows under `HEXFIELD_EQ_SUPPORT_RADIUS=4` (`assert n > 0`; fixture assumes
  the radius-8 default — the serve-vs-oracle raylen test passes at radius 4).

Flags now default ON in `scripts/_hexfield_eq_supervise_main1.sh` (each
`${VAR:-1}`, so `=0` reverts) and in the systemd template. CUDA graphs stay
OFF (the ray gather-index custom op syncs — incompatible; SUPERSEDED by the
round-2 addendum below: the build is sync-free since 2026-07-09 and graphs
are now ON).

**Live result:** 4.2 → **~15 pos/s** steady (3.5×), measured over settled
mid-epoch windows. `virtual_batch_size` 48→96 added ~+3% (submit loop was
bursty at 48); kept. The remaining gap to main_11's ~50 pos/s is ARCH COST,
not config: this net is ~2.75× the FLOPs of the 128c/9-block main_11 net
(2.25× width² × 11/9 blocks), so compute-parity with 50 pos/s is ~18 pos/s —
the soak runs at ~83% of that, with the residual in the per-forward ray
gather-index sync, the register-lane loop-carried tokens, and the group-affine
norm machinery. Relaunch mechanics:
`scripts/_hexfield_eq_relaunch_soak_main1.sh` (arm-4 arch env is NOT in the
supervise script — it must ride the launcher env).

### §5 addendum 2 — sync-free ray index + perm fold + CUDA graphs (2026-07-09)

Second serve round, deployed as one gated bundle. Live result:
**~15 → ~21 pos/s mid-epoch marginal** (samples 19.6–24.0 over a full-pool
epoch-4 window; cumulative-including-warmup 18.3) — ABOVE the naive ~18 pos/s
FLOPs-parity estimate, confirming the loop was submit-bound, not compute-bound.
Net from the eager launch: 4.2 → ~21 pos/s (5×).

1. **Sync-free gather-index build** (`_triton_ray.py`): the bbox-grid scatter
   (~8 device→host syncs/forward via `int(amin/amax)` + `live.any()`) is now a
   per-batch coordinate JOIN — pack (q, r) into an order-injective int64 key,
   sort live keys (pad → sentinel key), `searchsorted` the 32 neighbour keys.
   Zero syncs (pinned by a `set_sync_debug_mode("error")` CUDA test),
   bit-identical output (property test vs an embedded copy of the OLD
   algorithm), correct for ANY board shape — the hex-disk span bound is FALSE
   here (supports are unions of radius-9 disks around stone chains; elongated
   boards have O(N) span, which is WHY the join replaced a fixed-span grid).
2. **Coset-perm fold** (`model.py`/`register.py`): the head_perm gathers around
   every attention (3 A + 3 L + 8 lane refreshes) are folded into
   `EquivLinear`'s `_version`-keyed no-grad dense-weight cache
   (`set_serve_perms`: q/k/v `W[hp,:]` bit-identical; out_proj `W[:, hp]` —
   accumulation-reordered only). Runtime perms remain on the grad path; the
   fold and the skip share the literal `equivariant and not is_grad_enabled()`
   gate. state_dict keys unchanged (non-persistent buffers). Gate:
   `tests/test_hexfield_eq_perm_fold.py` (7/7 at C=96 AND C=192 — the full-net
   check scales its atol by each head's magnitude; GEMM-reorder error is
   relative to the accumulated row scale, ~2e-7 of it, NOT elementwise).
3. **CUDA graphs ON** (`HEXFIELD_CUDA_GRAPHS=1`, baked into the supervise
   script + systemd template): legalized by (1). Serve gate 7/7 with graphs +
   the full profile; `inference._GraphCache` captures per (B-bucket, Npad),
   raylen statics included.

Gates for the bundle (all at the arm-4 arch + full deploy env):
`test_hexfield_eq_triton_ray.py` 38/38 (both C=96 and C=192; includes v1-vs-v2
equality and the zero-sync test), `perm_fold` 7/7, `serve` 7/7 with AND without
graphs. Idle-GPU bench: index build 1.0 → ~0.4 ms, kernel unchanged.

**Negative result — all-heads-per-program kernel v2** (`HEXFIELD_RAY_V2=1`,
default OFF, kept in-tree): hoists the idx tile/liveness out of a
`tl.static_range` head loop. Idle bench at deploy shapes (B=24–48,
Npad=256–512): **~7% SLOWER** than v1 (kern/kern2 0.92–0.94; worse at larger
V2_BM — 0.7× at 32, ≤0.4× at 64). v1's six independent head programs hide the
gather latency better than one program's serialized head loop. Parity-correct
(serve 7/7 with it on); don't enable.

Config since this round: 256 games/epoch, `active_games`/`active_root_limit`
256 (both tomls; ~21 min self-play per epoch at ~21 pos/s).

## 6. Self-play soak (eq plan Phase 6 / ray plan Phase S)

- A few unattended epochs on `hexfield_eq_main_1`: watch entropy / game
  length / calibration bands, the winner-rate structure, and (if the lane is
  on) the lane's grad-norm group for sum-scale drift (ray plan risk 2).
- Multi-stage eval every 5 epochs accumulates the re-anchored pool.
- The equivariance A/B (D6): decide at this phase — `GROUP_ORDER=1` matched
  ablation (needs an augmentation port to be fair) vs absolute
  Strix/SealBot bar. Not needed to start the soak.

**Gate / kill:** eq plan Phase 6 — if the equivariant net does not at least
match the ablation (or the absolute bar under D6b) over the A/B window, fall
back per the plan.

---

## Appendix A — file inventory (this deployment prep)

| File | Role |
|---|---|
| `scripts/convert_hexfield_shards_to_eq.py` | main_11 → eq v4 shard converter (pure numpy, idempotent, `--limit`) |
| `scripts/validate_eq_shard_conversion.py` | conversion validator (byte / reader / expand / raylen / target-identity; `--all` for full corpus) |
| `configs/hexfield_eq_main_1.toml` | run toml (main_11 Gumbel regime; D4 anchors; warm-start placeholder) |
| `scripts/prefit_env/hexfield_eq_arm*.env` | per-arm ladder env (C=192 / ×12 / heads 3 / radius 4) |
| `scripts/_hexfield_eq_supervise_main1.sh` | eq supervisor (eq PYTHONPATH; conservative serve defaults) |
| `scripts/systemd/hexfield-eq-supervisor-1.service` | serve/train service template (full `HEXFIELD_EQ_*` block) |
| `/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_prefit_main11/` | converted corpus (train/ + val/) |

## Appendix B — validation evidence (2026-07-08, 2-shard sample)

```
effective NUM_FEATURES=25 HEXFIELD_EQ_SUPPORT_RADIUS=4
pair: epoch_000001/game_1000000.npz -> train/shard_game_1000000.npz
pair: epoch_000001/game_1000001.npz -> train/shard_game_1000001.npz
expand check (8 rows of shard_game_1000000.npz):
  collate_training keys: ['cell_q', 'cell_q_mask', 'coords', 'feats',
   'gumbel_policy', 'gumbel_policy_valid', 'legal_counts', 'mask',
   'moves_left', 'moves_left_mask', 'nbr', 'opp_coverage', 'opp_policy',
   'policy', 'policy_ce_weight', 'raylen', 'soft_policy', 'stvalue',
   'stvalue_mask', 'value', 'value_mask']
  [ok]   collate_training threads 'raylen' into the model batch
  gumbel_policy_valid rows in expand sample: 8/8
RESULT: PASS (2 shard pairs validated)
```

Converter sample summary: 2 shards → 250 rows, `gumbel_present` 250/250
(100 %), 0 orphans.
