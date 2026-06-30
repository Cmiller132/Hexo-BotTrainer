# hexfield_main_4 — Improvement Plan (loss-down / strength-flat)

Date: 2026-06-22
Author: autonomous analysis pass (root-cause + adversarial-verified)
Run: `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_4` (ckpts ep1..ep40, STOPPED)
Config: `configs/hexfield_main_4.toml`
Probe JSON: `analysis/improve_probes/out/<id>.json` (+ `.log`)

---

## 1. Root cause

hexfield_main_4 is a **healthy learner** whose training loss keeps hitting new lows while external strength has plateaued / slightly regressed. The cause is **dual, mutually-reinforcing under-training of the network**, NOT overfitting, past-peak rollback, learning rate, or search depth.

### PRIMARY — the value head cannot discriminate move quality in the decisive mid-game (30–70 ply)
The only supervision target that can teach per-move mid-game ranking — `cell_q` / per-action root-Q — is **never written during self-play**, so its mask is 100% zero and it contributes zero gradient. The value head therefore learns only the far-off terminal outcome: it sharpens already-decided positions (driving value loss to new lows) while learning nothing new about positions where games are actually won or lost.

Key numbers:
- **b3** (verdict POOR): mid-game ranking mean_rho **0.051**; by-ply rho is positive early then goes **negative in the decisive window** — 8–30 ply **+0.291** → 30–50 **−0.034** → 50–70 **−0.150** → 70–100 **−0.085**. top1_agreement **0.246**. Won/lost separation AUC: 0.538 (0–20) → **0.760 (40–60)** → 0.876 (60–80) → 0.991 (120+). The head is near-blind until the game is nearly over.
- **e2**: `cell_q` is dead — `batches_all_masked=6/6`, `total_covered_rows=0`, `grad_norm_ratio=0.0` at **both** ep20 and ep40.
- Mechanism (source-traced): `cell_q` loss is gated by `cell_q_mask` (`losses.py:303-310`), populated from `sample.q_policy` (`samples.py:264-273`). The mask is zero because **`q_policy` (per-action root-Q) is empty — self-play does not record root child-Q into the sample**. This is NOT the rust-expand `value_mask` bug from MEMORY (that is source-verified FIXED in main_4, `expand_backends.py:328-378`); it is a separate self-play-recording gap.
- **c1** corroborates: residual `value_gained_abs` is largest LATE (0.138) where the policy already agrees with search — the missing signal is value/eval, not move ranking.

### CONTRIBUTING — the optimizer is gradient-noise-dominated at batch_rows=32
Noise-dominated SGD lowers train loss by overfitting each tiny noisy micro-batch while barely moving the generalizing true-gradient direction that converts to strength. This is also why the policy prior is under-converged / too diffuse.

Key numbers:
- **f1**: mean|g|² **21.68 @ b32** vs **4.91 @ b128** — a 4.41× drop for a 4× batch, i.e. |G|²≈0, almost pure noise. (CAVEAT — see §2: f1's clean critical-batch estimator is INDETERMINATE; the raw g² ratio is the load-bearing evidence, not the "B≈527" fallback figure.)
- **f2** corroborates the signature: train-loss delta grows with lr (−0.358 → −0.755) while held-out delta does not — fitting micro-batches harder without generalizing.
- **c2**: policy prior `net_entropy` **2.116** vs MCTS target **1.517** (+0.60 nats too diffuse); `net_top1` 0.397 vs 0.542; verdict **underfit**. No memorization (train policy_loss 2.084 vs heldout-proxy 2.058, gap −0.026). `soft_policy_weight` is actively pulling the prior softer.

### STRUCTURAL FLOOR (not the cause) — trunk capacity binds for value
- **e1**: value linear-probe MSE **1.027** ≫ head **0.725** (advantage 0.302); sign 0.611 vs 0.729. The trunk does not linearly encode value; the head does real nonlinear work. Policy-vs-value internal grad cosine ≈0.014 — the shared ~1.28M-scalar trunk pays a capacity tax across two near-orthogonal objectives. This is the *last-resort* lever, not the cause.

**Two-sentence summary:** The value head is trained only on far-off terminal outcomes because `cell_q` — the one head built to teach per-move mid-game Q-ranking — gets zero supervision (self-play never records root child-Q into `sample.q_policy`), so loss falls by sharpening already-decided positions while mid-game move discrimination (b3 rho goes negative at 30–100 ply) never improves. This is amplified by a gradient-noise-dominated optimizer at batch_rows=32 (f1: mean|g|² 21.68@b32 vs 4.91@b128 ⇒ |G|²≈0), which lowers train loss by overfitting tiny micro-batches and leaves the policy prior too diffuse (c2: entropy 2.116 vs target 1.517).

---

## 2. What is ruled out (with evidence)

- **"ep40 is past-peak; roll back to ep30."** REFUTED. The rollback reading came entirely from **a1_ckpt_roundrobin**, which is degenerate: it ran only **7/28 pairings, all sharing the prefit anchor** (no direct ep30-vs-ep40 edge; `errors[0]="global time budget 3600s exceeded; stopped before pairing 10 vs 15"`), the anchor is saturated (ep30 and ep35 have IDENTICAL Elo 617.79), and a1's own verdict is PLATEAU, `significant=false`, ci95 [−291.83, 520.1]. Raw a1 actually favors ep40 (prefit wr 0.05 vs ep40 but 0.025 vs ep30 — prefit lost more to ep40). Direct measurements contradict rollback: **a3** h2h ep40 beat ep30 **6-3** (ep30 a_winrate 0.333), and the a3 sealbot ladder ranks **ep40 highest (elo 279.6, wr 0.833) > ep20 (185.0) > ep30 (140.9, weakest)**; `regressed_past_ep30: false`. Caveat: h2h is only 9 decided games (CI [0.12,0.65]) so exact ep30/ep40 ordering is unresolved — but rollback is decisively unsupported.
- **Search depth / visits.** RULED OUT. **a2**: candidate saturates wr 0.958@128 → 1.0@256 → 1.0@512 → 0.917@2048; top(2048)−ref = **−127.9 Elo**. **c1**: KL(visit‖prior) small and flat (~0.14 nats, top1_agree 0.729). Net-limited, not search-limited.
- **Learning rate.** RULED OUT. **f2**: delta_held_total essentially flat across [1e-4 … 5e-4] (−0.0357 to −0.0323), `all_similar=true, overshoot=false`. Do not decay lr.
- **Value miscalibration.** RULED OUT as the limiter. **b1**: ECE *improves* over training (0.0548 → 0.0438). **b3-C**: Brier 0.197 / ECE 0.036. **b2**: best offline tweak (label_smoothing ε=0.1) buys only −0.0054 ECE and *raises* NLL; none changes sign_acc (0.72). Calibration is near-optimal; offline rescaling cannot add discrimination.
- **rust-expand value_mask bug (from MEMORY).** RULED OUT for main_4. `configs/hexfield_main_4.toml` lines 82-88 document the claude/hexfield-cleanup merge FIXED it; `expand_backends.py:328-338,372` now emits `value_mask` from the rebuilt rust kernel (np.ones fallback only with an older .so). The MEMORY note says the bug is live in **main_3**, not main_4.
- **Data overfit / reuse staleness / coverage collapse.** NOT IMPLICATED (but genuinely UNMEASURED). The one direct test **d1** is BROKEN (`ValueError: invalid literal for int(): 'player0'` at `_common.py:740` — enum `.value` is a string). Proxies argue against it: **d2** `supports_fresher_data_lever: false`, corr(row_age, elo-delta) spearman +0.14 (wrong sign), reuse saturates ~5.97×; **d3** coverage healthy (opening unique_fraction 0.925, branching entropy 0.883). Fix d1 before trusting any reuse decision (fix #5).
- **Seat imbalance / komi / swap.** RULED OUT. **h1**: pooled P1 wr 0.525, CI [0.494,0.556], not significant. The ep40 seat-skew (P1 0.596) + length collapse (107→91) is a *symptom* of decisiveness-creep, not a structural seat problem.

### One correction carried from adversarial verification
Do NOT quote f1's `B_simple_median 526.7` ("16–29× current") as a hard measured number. f1's clean estimator is **INDETERMINATE** (`signal=INDETERMINATE`, `verdict="B_simple non-finite (G2_est<=0 or degenerate draws)"`, `G2_est = −0.676` negative, `B_simple = NaN`). The 526.7 is a noisy fallback summary. The de-noising lever is justified by the **raw g² ratio** (21.68@b32 vs 4.91@b128), not by the specific critical-batch figure. Present "bigger batch helps" as the conclusion; present any specific critical-batch number as indicative only.

---

## 3. Ranked fixes (exact edits, risk, expected effect, validation)

### #1 — Raise effective optimizer batch to de-noise the gradient  [config, risk=LOW, rebuild=NO]
- **Edit:** `configs/hexfield_main_4.toml` line **232**: `batch_rows = 32` → `batch_rows = 128`.
- **Why safe:** trainer does mathematically-exact gradient accumulation (`trainer.py:512/535/604`, step-global denominators in losses.py); per-forward micro-bucket size is unchanged ⇒ no OOM. Lower-variance updates are stabilizing, not destabilizing.
- **Expected effect:** lower-variance updates that translate train-loss progress into strength; sharper policy prior (entropy moving toward target 1.517 from 2.116). **Train loss will look WORSE per-step by design.**
- **Validate:** confirm the first resumed step logs PAIR_BUDGET-sized micro-buckets (per-forward size unchanged ⇒ no OOM). Validate ONLY on held-out loss + round-robin Elo vs ep40 baseline — **NEVER train loss.**
- **Rollback:** single-line revert to `batch_rows = 32`.

### #2 — Stop wasting the dead cell_q head's budget  [config, risk=LOW, rebuild=NO]
- **Edit:** `configs/hexfield_main_4.toml` line **260**: `q_head_weight = 0.1` → `q_head_weight = 0.0`.
- **Why safe:** verified config → `trainer.py:563` → `losses.py:214/310` (`total = total + q_head_weight * components["cell_q"]`). cell_q is the only consumer; the component is still computed (no KeyError at weight 0.0). Because cell_q_mask is already all-zero (e2), the head already contributes ~0 gradient — weight=0.0 is a **true no-op on dynamics**, pure hygiene.
- **Expected effect:** no change to current training dynamics; removes a dead head from the objective and clarifies that the real cell_q fix is code-fix #4.
- **Validate:** confirm e2 still reports cell_q `total_covered_rows=0` (unchanged) and overall loss curve unaffected vs a 1-epoch baseline.
- **Rollback:** set back to `q_head_weight = 0.1` (and do so once #4 lands so cell_q is actually supervised).

### #3 — Build a harder yardstick so further net gains are measurable  [eval-harness, risk=LOW, rebuild=NO]
- **Edit:** eval harness only (no training-path change). Add a higher-visit sealbot tier and a FULL round-robin among ep25..ep40 at ≥100 games/pairing (skip the saturated prefit anchor). Re-run a1/a3-style matches with `n_games>=100` and a sealbot at **≥1024 visits** as the new anchor; expand the ep30-vs-ep40 h2h from 9 to ~100 games. **The 3600s global time budget that broke a1 must be raised** or the round-robin truncates again exactly as before.
- **Expected effect:** a non-saturated strength signal that can register sub-100% gains, enabling valid go/no-go on every other lever. No effect on training.
- **Validate:** confirm the new anchor yields ep40 winrate strictly between ~0.4 and ~0.85 (not pinned at 1.0) so headroom exists in both directions.
- **Note:** this is a BUILD TASK with non-trivial wall-clock cost, not a one-liner.

### #4 — Fix cell_q supervision (record root child-Q into self-play samples)  [code + native REBUILD, risk=HIGH, rebuild=YES]
- **Edit:** populate `q_policy` (list of `(action_id, q)` with `q ∈ [-1,1]`) from the Rust MCTS root child-Q values, mirroring how `visit_policy` is already recorded. Touch points: the self-play / sample builder that constructs the sample consumed by `packages/hexfield/python/hexfield/samples.py:264-273` (which already projects `sample.q_policy` onto the legal set), **plus** the Rust MCTS export that currently emits visit counts but not child-Q. Then revert #2 (set `q_head_weight` back to 0.1) once `q_policy` is non-empty.
- **Expected effect:** mid-game value discrimination recovers; the value head learns to RANK candidate moves rather than only predict terminal outcome — directly attacks the b3 mid-game-rho deficit.
- **Validate:** add/extend a unit test asserting `q_policy` is non-empty and `q ∈ [-1,1]` for self-play samples; dump one self-play game and confirm `cell_q_mask>0` rows appear. Rebuild native .so, run cargo tests + hexfield head tests, then a 1–2 epoch smoke confirming cell_q `grad_norm_ratio>0` (e2) before full relaunch. **Strength gate:** b3 mid-game rho > 0.2 in the 30–70 ply buckets AND 40–60 AUC > 0.80 (IGNORE ECE).
- **Sequencing:** do NOT land #1 and #4 in the same uncontrolled step — #1 changes optimizer dynamics and #4 adds live cell_q gradient; the two interact. Apply #1/#2/#3 now; sequence #4 separately with its own smoke.
- **Rollback:** revert the sample-builder + Rust export change and rebuild; set `q_head_weight=0.0` again.

### #5 — Fix the d1 overfit probe so reuse/staleness is measurable  [code, risk=LOW, rebuild=NO]
- **Edit:** `analysis/improve_probes/_common.py:740` — replace `player = int(getattr(cp,'value',cp))` with a string-enum-safe parser, e.g.
  `v = getattr(cp,'value',cp); player = int(str(v)[6:]) if str(v).startswith('player') else int(v)`. Then re-run d1 for ep20/30/40.
- **Expected effect:** d1 produces a real train-vs-heldout gap and trend; enables an evidence-based reuse decision instead of guessing. No effect on the run.
- **Validate:** re-run d1; confirm `fresh_games_written>0` and `gap_raw` non-null for ep20/30/40.
- **Rollback:** git revert the one-line parser change.

### #6 — Capacity bump: widen trunk CHANNELS 96 → 128 (LAST RESORT)  [code + native REBUILD, risk=HIGH, rebuild=YES, NOT resumable]
- **Edit:** `packages/hexfield/python/hexfield/constants.py` — CHANNELS 96 → 128 (verify exact symbol name before editing). Requires a **fresh prefit + a full fresh run** (NOT resumable).
- **Expected effect:** higher value-representation ceiling IF targets (#4) and batch (#1) are already fixed; otherwise wasted compute overfitting the same impoverished targets harder.
- **Validate:** after #1+#4 are in and re-probed, re-run e1; if `head_mse_advantage` shrinks post-widening, capacity was binding. Validate on the new harder yardstick (#3).
- **Rollback:** revert constants.py; the prior run dir is untouched (fresh dir).

---

## 4. APPLY-NOW bundle (final, post-verification) + resume checkpoint

All three are SAFE for autonomous application (adversarial-verified). None touches the training math or requires a rebuild.

| # | Edit | Type | Rebuild | Validation |
|---|------|------|---------|------------|
| 1 | `configs/hexfield_main_4.toml:232` `batch_rows = 32` → `128` | config | NO | Held-out loss + round-robin Elo vs ep40. NEVER train loss (it will look worse by design). Confirm no OOM (per-forward size unchanged). |
| 2 | `configs/hexfield_main_4.toml:260` `q_head_weight = 0.1` → `0.0` | config | NO | e2 still `total_covered_rows=0`; loss curve unaffected vs 1-epoch baseline. True no-op on dynamics. |
| 3 | Eval harness: higher-visit (≥1024) sealbot anchor + full ep25..ep40 round-robin @ ≥100 games/pairing; raise the 3600s a1 time budget | eval-harness | NO | New anchor yields ep40 winrate strictly in ~0.4–0.85 (not pinned at 1.0). Build task, non-trivial wall-clock. |

**RESUME CHECKPOINT = ep40** (NOT ep30). Resume in a **FRESH run dir**, keeping ep1..ep40 immutable so every change is reversible. The "roll back to ep30" reading is an a1 artifact (degenerate single-anchor star graph, 7/28 pairings, saturated anchor, verdict PLATEAU/`significant=false`). Direct evidence favors ep40 (a3 h2h 6-3; sealbot ladder ep40 279.6 > ep30 140.9). ep40 also carries the matured trunk that the value-target fix (#4) needs to build on. Caveat: exact ep30/ep40 ordering is statistically unresolved (h2h n=9), but rollback is decisively unsupported.

**Sequencing note:** apply #1/#2/#3 now as one bundle; sequence the high-risk code+rebuild fix #4 **separately** with its own smoke and re-probe — #1 (optimizer dynamics) and #4 (new live cell_q gradient) interact and must not land together uncontrolled.

---

## 5. Deferred / bigger levers

- **#4 cell_q supervision (code + REBUILD, HIGH):** the real fix for the PRIMARY root cause. Gate on the b3 strength criteria above. Sequence after #1/#2/#3 land and the harder yardstick (#3) exists.
- **#5 d1 probe fix (code, LOW):** unblocks the genuinely-unmeasured reuse/staleness question. Cheap; do it so any future cut-reuse decision is evidence-based.
- **#6 CHANNELS 96 → 128 (code + REBUILD, HIGH, fresh run):** capacity is binding for value (e1) but is the LAST resort — only worthwhile once targets (#4) and batch (#1) are fixed, else it overfits the same impoverished targets harder.
- **Do NOT spend budget on:** more visits / search depth (a2/c1), lr decay (f2), calibration knobs (b1/b2/b3-C), value_mask "repair" (already FIXED in main_4), komi/swap (h1), or cut-reuse-as-default before #5 is done and d1 re-measured.

---

## 6. Rollback for each change

- **#1:** edit `configs/hexfield_main_4.toml:232` back to `batch_rows = 32`. Single line.
- **#2:** edit `configs/hexfield_main_4.toml:260` back to `q_head_weight = 0.1` (do this once #4 lands so cell_q is supervised).
- **#3:** eval-harness only — revert harness changes; no run state touched.
- **Resume:** fresh run dir keeps ep1..ep40 immutable — discard the new dir to fully revert.
- **#4:** revert the self-play sample-builder + Rust MCTS export change; rebuild native .so; set `q_head_weight=0.0` again. Covered by the immutable ep1..ep40.
- **#5:** git revert the one-line parser change at `_common.py:740`.
- **#6:** revert `constants.py` CHANNELS to 96; the prior run dir is untouched (fresh dir was required).

---

### Key source files (absolute paths)
- `E:\Hexo-BotTrainer-hexgt\configs\hexfield_main_4.toml` — `batch_rows:232`, `q_head_weight:260`, `moves_left_weight:259`, `learning_rate:240`, `soft_policy_weight` (§ after 260), `train_samples_per_epoch`
- `E:\Hexo-BotTrainer-hexgt\packages\hexfield\python\hexfield\samples.py:264-273` — q_policy → cell_q_mask population (empty in self-play = root cause)
- `E:\Hexo-BotTrainer-hexgt\packages\hexfield\python\hexfield\losses.py:303-310` (cell_q gating), `214/310` (total accumulation)
- `E:\Hexo-BotTrainer-hexgt\packages\hexfield\python\hexfield\expand_backends.py:328-378` — value_mask now correct (NOT the bug)
- `E:\Hexo-BotTrainer-hexgt\packages\hexfield\python\hexfield\trainer.py:512/535/563/604` — exact grad-accum + loss wiring
- `E:\Hexo-BotTrainer-hexgt\analysis\improve_probes\_common.py:740` — d1 `'player0'` int-parse bug (fix #5)
- `E:\Hexo-BotTrainer-hexgt\packages\hexfield\python\hexfield\constants.py` — CHANNELS (fix #6)
