# dense_cnn_restnet_main_4 — Recommendation package

**Date:** 2026-06-12 (overnight session)
**Produced by:** 16-agent investigation workflow (6 evidence probes → 2 independent designers → judge → 5-verifier adversarial panel → revision), plus a GPU ckpt10-vs-ckpt5 head-to-head. Full machine-readable package: `scripts/_wf_r4_RESULT_recommendation.json` (+ `_wf_r4_RESULT_ev_*.json`, `_wf_r4_RESULT_verdicts.json`). Panel outcome: 4/5 targets sustained with minor notes; 1 core failure (gate realism) corrected in revision against actual ep1–10 diagnostics.
**Status:** main_3 halted 2026-06-12 00:25 local at epoch 10 (flag + log entry written; run dir preserved read-only). main_4 launch in progress under the standing owner goal directive ("model continually improving over 30 epochs").

---

## 1. Executive summary

main_3's marathon degradation has a **mechanical root cause**: Hexo is hex-grid Connect6 on an unbounded board, and the model's input/candidate crop is a **radius-20 disk around the stone centroid**. When a game's stone blob outgrows the crop, standing immediate wins whose completion cell falls just outside the rim become simultaneously **unplayable, unblockable, and invisible — for both players** (out-of-crop cells are absent from input planes, excluded from MCTS candidates, and explicitly skipped by the tactical-forcing mechanism, `mcts_tree.rs:885-889`). Games then freeze for hundreds of plies until glacial centroid drift (~0.02 cells/ply) re-admits a win cell — a coin flip.

Evidence: 47/47 engine-verified standing wins left unplayed; 99.9% (1307/1308) of ep9 missed-win plies had **every** win cell out-of-crop (median distance 21–22 vs crop 20); frozen-win games per 384: **2 → 0 → 1 → 23 → 78** across ep5–ep9; median length 509 with-frozen vs 183 without; every 1024-action truncation audited was a frozen game.

The RL-side damage is the **amplifier, not the root**: the coin-flip row flood (65% of ep9 rows ≥80 moves from the end at sign-accuracy 0.50) compressed the value head's amplitude (pooled std 0.486 → 0.273 — ckpt9 reads all marathons as decisive_frac 0.000 where ep5/ep7 read ~0.5), which degraded the *next* epoch's search — the classic one-epoch-behind runaway the handoff hypothesized, now with the trigger pinned.

**Refuted along the way** (this redirects the fix): H-B search-budget — 512 and even 1024 visits convert **0/9** of the discriminative squander positions that 128-visit search fails (identical hold rates 23/32; paired post-move delta +0.001; ckpt9 root Q +0.05 = flat value landscape, depth has no gradient to exploit). H-D temperature-sampling — squander-trigger moves are lens-argmax at exactly base rate (57.8% vs 59.0%, z=−0.20), and greedy-fast + temp-0.1 caps sampling-deviant play at ~2.4%. **Therefore: no PCR increase, no visits increase, no temperature/exploration changes.** The earlier draft main_4 config with `pcr_full_proportion 0.40` was rejected on this evidence.

**The fix:** restart from sealbot-validated ckpt5 with (C3) a surgical, fail-open, config-rollbackable **frozen-win override** that ends crop-frozen games the moment they arise, (C2) quarantine of truncation-poisoned rows (all-z=0 labels, confirmed in code), and (C1) a **dormant** length-decay row-weighting with pre-authorized activation triggers — plus healthy-era window seeding and EMA re-seeding. Everything with measured rationale in main_3's config is kept byte-identical.

**What restarting costs (measured honestly — this CORRECTS the workflow package's "sacrifices nothing" framing):** main_3's ep10 sealbot eval landed at 106/128 = 82.8% — *identical* to ep5 — but the direct GPU head-to-head (96 games, 512 visits both sides, seat-alternated, 96 distinct openings) shows **ckpt10 beats ckpt5 72–24 (75.0%, final, 96/96 completed, seat-balanced, 96 distinct openings, zero truncations)**. The sealbot arbiter is *saturated*, not merely blind: epochs 6–10 bought ~+180 Elo of real in-lineage strength. The restart still stands — that strength rides on a hedge-collapsed value head and a marathon-attractor playstyle that cannot produce healthy training data (ep11 selfplay was running 238+ dec/game when halted; ep8–10 shards are prohibited as seed material; the validated-healthy training dynamics belong to ckpt5), and the owner directed the ckpt5 start — but main_4 must **re-earn ~5 epochs of strength (~4–5 GPU-hours at healthy-era pace)**, and the goal metric is healthy compounding, not the sick snapshot's playing strength. Corroborating texture from the h2h: 16/96 games ground past 160 plies (the stall regime is reachable in full-512 mutual play, ~17% — it is the *mutual* sick-net dynamic, not the opponent, that triggers it), and ckpt5's wins took visibly longer than ckpt10's (mean 101 vs 80 plies). Full data: `scripts/_wf_r4_h2h_results.json`.

---

## 2. Causal verdict (final hypothesis weighting)

| Hypothesis | Verdict | Decisive evidence |
|---|---|---|
| **H-C** structural motif | **CONFIRMED** as the mechanical producer of the marathon tail, in a precise form: crop-rim frozen-win zugzwang | 47/47 engine-verified; 99.9% out-of-crop; frozen games 2/0/1/23/78; code skip at `mcts_tree.rs:885-889` |
| **H-A** value regression | **CONFIRMED as amplifier** (not in-game cause): amplitude compression from coin-flip flood damages the *next* epoch's search. ckpt8 saw the squandered edges at near-healthy fidelity (r=0.957, decisive_frac 0.432) while playing the marathons — the in-game failure is not value blindness | ckpt9 lens: 0 events, pooled std 0.273 vs 0.486; ckpt8 lens near-healthy on its own games |
| **H-B** fast-search bottleneck | **REFUTED** at position level (128/512/1024 visits identical; 0/9 conversions) | `_wf_r4_hb_full.json`; flat root Q |
| **H-D** temp-floor blunders | **REFUTED** (argmax at base rate; sampling ceiling ~2.4% « trigger frequency) | `_wf_r4_hd_table.txt` |

Ignition at ep7→ep8: ckpt7's longer/more-spread playstyle turned a rare tail accident (frozen games 2/0/1) into an attractor (23, then 78). **残 open:** ~+40 of ep8's +62 dec/game is *in-crop* lengthening with no launch-time fix — C1's triggers and the gates contain it if it re-emerges (plausible around epochs 11–13); that contained re-ignition is itself the informative experiment, run with armor instead of naked.

## 3. The intervention

### Code (all Python-only, flag-gated, config-rollbackable; tests + golden validation against the 47/47 engine-verified corpus)
- **C3 frozen-win override (enabled):** per-game incremental 6-cell-window tracker; when the mover has standing win(s) and ALL are out-of-crop → clone-verify → play the win instead of the search move; excluded from training rows; fail-open with failure counter. New `win_tracker.py` + `_on_move` hook + `frozen_win_override` config key + diagnostics counters.
- **C2 truncation quarantine (enabled):** `drop_truncated_rows = true` — truncated games (winner None → z=0 every row, loss unmasked) write no rows.
- **C1 length-decay row weighting (dormant):** decay `frequency_weight` for rows with `moves_left > knee` at the provably-unique choke point (`replay.py materialize_policy_surprise_rows`). Ships with knee 0.0; **pre-authorized config-only activation** (knee → 110.0) on: dec/game > 135 ×2 epochs, > 150 once, raw_samples > 15000, or amplitude probe pooled std < 0.35.
- **Stage 3 (NOT built; owner + engine-team):** Rust TSS out-of-crop unfreeze (if C3 too slow/unreliable); KataGo-style moves-left utility / adjudication (if in-crop conversion weakness proves structural).

### Config deltas vs main_3 (everything else byte-identical)
`initialize_from` → main_3 ckpt5 (weights-only; **run starts at epoch 6**, first eval at epoch 10) · `temperature_length_prior` 150 → **115** (healthy-era EMA band; never copy the contaminated `length_ema.json` @160.86) · `seed` 1 → 2 · the three new key groups above.

### Bootstrap recipe (executed this session)
1. Land C3+C2+C1 with tests (WSL venv, CPU).
2. `configs/dense_cnn_restnet_main_4.toml` (this repo).
3. Window seeding: main_3 `selfplay/epoch_00000[1-5]_game_*.npz` + `.json` sidecars (1920 shards = exactly ckpt5's 46,576-row corpus) → `main_4/selfplay/seed_main3/` with **preserved mtimes** (`cp -p`). No `.hxr`, no ep6–11 shards, no `length_ema.json`. Expected epoch-6 window ≈ 22.4k rows (vs ~6.4k unseeded); gate G7 checks ≥ 18,000.
4. Pre-launch gates: strict-load re-check under final code; seed shard count (1920+1920); 8-game CPU smoke with override on.
5. GPU validation batch (~128 games under ckpt9 with override on): predicts no 500+ ply games, overrides fire ~25+, truncations 0, mean length → in-crop residual (~150–180). Validates C3's mechanism + measures per-move overhead (G8) + quantifies the in-crop residual C1 must guard.
6. Launch detached via the restnet supervisor (`CONFIG`/`RUNDIR` overrides), single GPU, main_3 untouched.

## 4. Verification gates (per-epoch; calibrated so healthy ep1–7 pass and ep8 fires)

| Gate | Metric | Trip | Action |
|---|---|---|---|
| **G1** length (primary) | dec/game | YELLOW ≥135 (or >150 once → C1); RED ≥170 ×2 or >220 once | YELLOW: flip C1 + amplitude probe; RED: halt → stage 3 |
| **G2** override integrity | truncations + override counters | ≥2 truncations; any `override_failures` > 0 | debug tracker vs golden data; C2 contains poison meanwhile |
| **G3** value CE (confirmer, NOT detector) | training.loss_components.value | >0.78 once while G1 YELLOW or G5 tripped; ×2 standalone (ep6 advisory ≤0.85) | amplitude probe; CE is distribution-maskable — sick ep10 read 0.7695, UNDER the line |
| **G4** policy CE | …loss_components.policy | >3.0 once while G1 YELLOW; ×2 standalone | treat as G1 trigger |
| **G5** row flood (co-primary) | raw_samples | >15,000 (fires at ep8=16,210; never ep1–7) | flip C1; with C1 active, ratio must drop <0.90 |
| **G6** EMA recipe | expected_game_length @ep6 | ≠115.0±0.5 (150=prior unapplied; 160.86=contaminated copy) | halt, fix, relaunch (one epoch lost) |
| **G7** seeding | training.samples @ep6 | <18,000 (expected ~22,400; unseeded ~6,400) | halt, re-copy with `cp -p`, relaunch |
| **G8** override overhead | on_move_seconds/moves_decided | >2.0 ms/move ×2 or >4.0 once (healthy 0.37–0.55; sick-no-tracker 0.65–0.75) | disable C3 (config) + profile; NEVER on absolute time during G1 YELLOW |
| **G9** sealbot (external no-regression arbiter ONLY) | eval wins, mean_turns @10/15/20 | <96/128 or mean_turns >105; HARD ABORT <92/128 or >110 | <92: halt, check seeding; turns>105 with controlled selfplay: stage-3 trigger. **Measured blind spot:** sick ep10 scored 106/128 @81.6 — G9 cannot see the internal runaway; detection lives on G1/G5 |

**Predicted effects / falsifiers:** dec/game ∈ [95,135] epochs 6–10 (>135 at ep6/7 = warm start failed to reset; ≥165 anywhere = C1 fires + in-crop hypothesis live); truncations 0 (1 tolerable); overrides ≤3/epoch while healthy (>10 = early warning *before* lengths inflate); raw_samples ∈ [7.5k,14k]; value CE ≤0.78 from ep7 trending ≤0.70 by ep12–15; epoch wall 2.3–3.3 ks; ep10 eval ≥96/128 (parity at 106 = no-regression, *not* health proof); ep15 ≥106 with healthy internals.

**G10 (added post-h2h) — true strength arbiter:** sealbot is saturated (ckpt10 beat ckpt5 73.8% h2h despite identical 106/128 sealbot scores), so external progress is measured by **head-to-head vs main_3 ckpt10** with the existing arena harness (`scripts/_wf_r4_h2h_arena.py`, ~20 GPU-minutes per 96 games): at main_4 epoch ~15, expect ≥35–40%; by epoch ~20–25, ≥50% (re-earned the abandoned strength plus interest, from a healthy base). Persistent <30% at epoch 20+ with healthy internals = the restart is not compounding — escalate to owner.

## 5. Owner sign-off items (resolved under the standing goal directive)

1. **Halt main_3 now** — executed 00:25 local (ep10 eval landed; ep11 selfplay was burning GPU toward prohibited data).
2. **Restart re-confirmation given the healthy ep10 eval** — confirmed under the goal directive, with the trade stated plainly: h2h shows ckpt10 is genuinely ~180 Elo stronger in-lineage (sealbot is saturated as an arbiter), so the restart abandons real strength — but strength produced by a training state that demonstrably cannot continue (runaway selfplay distribution, prohibited data, hedge-collapsed value head). ckpt5 + C3 armor + healthy dynamics re-earns it in ~5 epochs; an epoch-15 h2h vs ckpt10 (not just sealbot) is added to the verification plan as the true progress arbiter.
3. **Pre-launch GPU validation batch** — run (see §3 step 5 results in the session log).
4. **Code changes at launch** — C3/C2 enabled, C1 dormant; a config-only restart cannot meet "provably stops the marathon spiral" given H-C is mechanical.
5. **Staged-escalation authority** — C1 triggers and G8 rollback pre-authorized; stage-3 items are NOT (engine-team decisions).
6. **seed 2** — accepted (RNG comparability already broken by warm start + seeding).
7. **epochs=60** kept (→ 55 new epochs; ≥30-epoch improvement goal fits comfortably).

## 6. main_3 disposition

Run dir preserved **read-only** as the forensic record. ep8/9/10 + partial-ep11 shards **prohibited** from any main_4 use (and from anything mtime-adjacent to main_4's selfplay tree) — they remain the golden corpus for C3's tracker tests, C1 decay calibration, and stage-3 design. `length_ema.json` (160.86) must never be copied.

## 7. Artifact index

Evidence: `scripts/_wf_r4_RESULT_*.json` (split package), `_wf_r4_structure*.{py,json,txt}` + `_wf_r4_verify.json` + `_wf_r4_crop.json` + `_wf_r4_pop*.json` (H-C), `_wf_r4_hb_*.{py,json,txt}` (H-B refutation incl. 1024v), `_wf_r4_hd_*.{py,json,txt}` + `_wf_r4_parity*` (H-D refutation), `_wf_r4_traj_ep{7,8,9}lens.json` + `_wf_r4_vtraj_*.json` (lens amplitude), `_wf_r4_arbiter*.{py,json,txt}` (trend table + EMA reconstruction), `_wf_r4_ckpt5_load*` (warm-start gate), `_wf_r4_h2h_*` (GPU head-to-head), `_wf_r4_health.py` (per-epoch gate checker, calibrated: 0 false FAILs ep1–7, 3 FAILs at ep8, 5 at ep9). Prior context: `MAIN3_DEGRADATION_HANDOFF.md`, `MAIN3_RUN_AUDIT.md`.
