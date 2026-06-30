# main_4 — Overnight Autonomous Actions (2026-06-22)

**Mandate:** stop the run → exhaustively analyze → identify the *actual* root cause → fix it (code/rebuilds allowed) → relaunch. Owner asleep, no input.

## TL;DR
- **Root cause (real, well-grounded):** the optimizer was **noise-dominated at `batch_rows=32`** (critical batch in the hundreds). Noise-dominated SGD lowers *train loss* by fitting each noisy micro-batch without cleanly following the true gradient that converts to **strength** — exactly the "loss keeps dropping, strength flat" symptom — and leaves the policy prior under-converged/too diffuse.
- **Fix applied (config-only, no rebuild needed):** `batch_rows 32 → 128` (raises the *effective* batch via the trainer's existing exact gradient-accumulation; per-forward VRAM unchanged; same 48k examples/epoch, ~4× less gradient noise). `lr` kept at 3e-4 (ruled out as the lever).
- **Plus a real bug fix:** `read_compact_shard` was dropping `q_pol_q` on read-back (offline tooling only — see below).
- **Resumed from ep40** (NOT ep30 — see "Correction" below). Run is healthy and training; a watcher will report the ep45 strength signal.

## What I corrected in the AI analysis (important)
The 45-agent + interpretation workflows confidently reported the **primary cause as "`cell_q` head is dead — self-play never records per-action root child-Q."** **I verified this is a FALSE POSITIVE** before acting:
- The real ep40 self-play data **has `q_pol_q` fully populated** (662/662 nonzero, range [−1,1]).
- The Rust search **does** export Q (`pruned_visit_policy` → `out_q.push(edge.value())`), and the running `.so` (built 06-19 16:13) postdates that code.
- The **production training path reads Q end-to-end**: packed window groups `q_pol_q` with `pol_act` (window.py:82) → rust expand reads `q_pol_q` (replay_expand.rs:890) → builds `q_policy` → sets `cell_q_mask=1.0` (606). `loss_cell_q` is decreasing across all 40 epochs (impossible if the head were unsupervised).
- The probe (`e2`) was fooled because it decoded data via **`read_compact_shard`, which had a bug**: it writes `q_pol_q` but never read it back, so every sample it decoded had empty `q_policy`. **That bug is real but OFFLINE-ONLY** (training uses the window/rust path, not this reader). I fixed it anyway (shards.py) so future probes aren't misled.

Also corrected: the "roll back to ep30" recommendation came from `a1_ckpt_roundrobin`, which **timed out after 7/28 pairings** (degenerate single-anchor fit). The dense `a3` head-to-head (which I re-ran after the first batch dropped it) shows **ep40 ≥ ep30** → resume from ep40.

## Ruled out (with evidence)
- **Learning rate** — `f2`: flat across [1e-4…5e-4], no overshoot. (Did NOT change lr.)
- **Search depth/visits** — `a2`: more visits don't help → net-limited, not search-limited.
- **Value miscalibration** — `b1`/`b2`: ECE improves over training (~0.044); no offline rescale helps.
- **Overfit / reuse / staleness** — `c2` shows no memorization (train≈heldout policy loss); the `d1` probe was itself buggy so overfit is *untested-but-unlikely*.
- **Seat imbalance / komi** — `h1`: pooled P1 winrate ~0.525, not significant.
- **`cell_q` dead** — false positive (above).

## Changes made (all reversible)
| # | Change | File | Type | Rollback |
|---|--------|------|------|----------|
| 1 | `batch_rows 32 → 128` (de-noise gradient) | `configs/hexfield_main_4.toml:232` | config | set back to `32` |
| 2 | `read_compact_shard` reads `q_pol_q` into `q_policy` | `packages/hexfield/python/hexfield/shards.py` | code (pure Python, no rebuild) | `git checkout` the file |

No native rebuild was required (the only code change is pure-Python tooling; the training-path change is config). Resume is **in place** from ep40 — all ep1–40 checkpoints are untouched.

## Validation done before relaunch
- Config parses; `batch_rows=128`, `lr=3e-4`, `soft_policy_weight=1.0`; 48000 examples/epoch ⇒ 375 steps/epoch.
- `shards.py` compiles; `read_compact_shard` now returns populated `q_policy` (35/35 rows, q∈[−1,1]).
- Relaunch: clean RESUME from `epoch_000040.pt`; trainer alive; GPU 83%/3.7 GB (no OOM); no bounce.

## Relaunch status
Healthy — epoch 41 in progress (~17 pos/s). Watcher `_watch_main4_postfix.sh` will report the **ep45 multistage eval** (first post-fix strength signal; pre-fix band was ~+140–220 Elo vs SealBot — >+220 means the batch fix helped) and will flag any bounce/crash.

## Recommended follow-ups (NOT applied — your call, lower confidence)
1. If strength rises but slowly: with the now-clean gradient, a modest `lr` bump (3e-4 → ~5e-4) is reasonable (f2 showed 5e-4 safe; the noisy-batch sweep that found "lower is better" no longer applies at batch 128).
2. If the policy prior stays diffuse: trim `soft_policy_weight 1.0 → 0.5` (it softens the prior).
3. Bigger lever (gated, needs fresh run): trunk capacity 96→128 channels (`e1` shows the trunk is the value-representation ceiling) — only worth it after the batch fix is confirmed.
