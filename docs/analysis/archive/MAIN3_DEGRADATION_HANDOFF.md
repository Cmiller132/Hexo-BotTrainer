# HANDOFF: dense_cnn_restnet_main_3 degradation — root-cause confirmation & intervention design

**Audience:** ultracode multi-agent deep-analysis session.
**Mission:** (1) confirm/complete the causal model of the run's degradation, (2) design and validate the intervention, (3) verify everything adversarially before recommending. The run owner wants evidence-backed compromise fixes, not overcorrection. No intervention has been applied yet — the owner has NOT yet approved any halt/rollback; produce a recommendation package for sign-off.

---

## 1. Current state (as of 2026-06-11 ~22:40 local, epoch 9 complete, ep10 in flight)

- Live run: `E:\Hexo-BotTrainer\runs\dense_cnn_restnet_main_3` (WSL `/mnt/e/Hexo-BotTrainer/runs/...`). Supervisor pid in `driver.pid`/`supervisor.log`; launched 18:08Z; zero relaunches.
- Config: `configs/dense_cnn_restnet_main_3.toml` (T=1.05 flat root temp, eps=0.20, no opening anchor, halflife_fraction 0.12, soft_z=0, PCR 25% full@512 / 75% fast@128 unrecorded, policy-init 25% of games, 384 games/epoch, 60 epochs planned).
- **The ep10 sealbot eval (128 games, every 5th epoch) may have landed by the time you read this — READ IT FIRST** (`runs/.../diagnostics/epoch_000010.json` → metadata.result.evaluation). ep5 baseline: 106/128 = 82.8%. The owner predicts regression. Also compare eval GAME LENGTHS vs ep5's mean_turns=76.3 — if eval lengths jumped too, the stall transfers to full-search play (strengthens H3 over H1).

## 2. Degradation timeline (all numbers verified, full-population where marked)

| epoch | dec/game | rows | policy CE | value CE | P0 share | prior top1 [5,16) (pop) | prior [16,48) (pop) | endgame sign-acc ml[0,5) | sealbot |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 85.2 | 7,986 | 2.659 | 0.933 | .49 | — | — | — | — |
| 5 | 108.4 | 10,229 | 2.686 | 0.699 | .503 | 0.271* | 0.312* | 0.97* | **106/128** |
| 6 | 108.8 | 10,297 | 2.835 | 0.695 | .589 | ~0.268 | — | — | — |
| 7 | 107.5 | 10,280 | 2.663 | **0.684** (< ln2!) | .503 | — | — | — | — |
| 8 | **169.7** | 16,210 | 2.881 | 0.741 | .567 | 0.195 | 0.253 | 0.98 (ckpt8 on ep8) | — |
| 9 | **265.6** | 25,477 | 3.304 | 0.806 | .472 | **0.154** | **0.209** | **0.83** (ckpt9 on ep9) | — |

(* = first-60-games sample, biases top1 LOW/entropy HIGH per audit; pop = all-384-game probe.)
Length is compounding ×~1.56/epoch (108→170→266); ep9 p90=511, max=1024 (= max_actions ceiling, 3 truncations). Temperature is NOT the driver: EMA only 111→126, halflife 13.4→15.1 (+12% vs +56% length).

## 3. Causal model and evidence status

**[CONFIRMED] Proximate mechanism — conversion failure, not equilibrium.**
`scripts/_wf_traj_probe.py 5 9 8 4` (ep5 "healthy lens" replaying ep9 games, every ply, CPU): the 8 longest ep9 games (590–855 plies) show **19–68 "squander events"/game** (|v_P0|>0.6 sustained 3 plies, then the leader's edge < 0.2 within 10 plies). Median-length games: 0 events, decisive_frac 0.03–0.10. Output: `scripts/_wf_traj_ep5lens.{json,txt}`. Marathon games are repeated failures of the leader to close, NOT balanced walls.

**[CONFIRMED] Ignition direction — value misevaluation precedes training contamination.**
ckpt7 evaluating the endgames of the ep8 marathon games IT PLAYED: sign-acc ml[0,5) = **0.83** (`scripts/_wf_m3_x78.json`); ckpt8 (trained on those games) reads the same endgames at 0.98. So ep7's net pushed play into longer/novel structures outside its own value competence; search Q is unreliable exactly there; conversion fails; lengths explode. Each epoch's training catches up on YESTERDAY's distribution while play drifts further — classic non-stationarity runaway.

**[CONFIRMED] Amplifier — data-mix flooding.**
ep9: 65% of rows ≥80 moves from the end (coin flips, sign-acc 0.50); opening/midgame gradient share starved ([5,16) prior 0.27→0.154; [16,48) 0.312→0.209 full-pop). Both heads hedge (mean|v| 0.119 at ep9), search weakens further.

**[CONFIRMED-NEGATIVE] Ruled out:**
- Overfitting: held-out test (ckpt8 on never-trained ep9 rows) scored BETTER than on trained-on rows (CE 0.656/signal +0.034 vs 0.677/+0.015); train CE > fresh CE; no memorization signature; details in session + `scripts/_wf_m3_holdout.json`.
- Temperature feedback (EMA loop): magnitudes don't add up (above).
- Value optimism as a driver: full-pop bias +0.036 (ep8), +0.061 (ep9) — small; was a sample/skew artifact at +0.111.
- main_2-style exploration collapse: root temp flat 1.05; deep-ply sharpening still strongly positive (+0.244 ep8, +0.214 ep9 deep bucket).

**[OPEN] What exactly ignited at ep7→ep8?** ep7's net trained only on ~107-ply games yet generated 170-ply games. Hypotheses to discriminate:
  - H-A: value crossing the hedge bound at ep7 = first real defensive competence → defense outpaces attack structurally in Hexo.
  - H-B: fast-search (128-visit) moves are the conversion bottleneck: 75% of selfplay moves; eval games (512 visits both sides) ran 76 turns vs selfplay 108 same era. **Test: self-play a batch with pcr disabled / all-512 using ckpt7 and ckpt9; if lengths normalize (~100), fast moves are the proximate stall.** Needs GPU (contends with run, or run after halt).
  - H-C: a tactical/structural Hexo property (defensive walls) the search exploits once found — inspect actual marathon games visually (frontend debug screen, dashboard :8080, CPU worker) for repeated structural motifs.
  - H-D: squander events coincide with played moves the SEARCH ranked low (temp-floor sampling blunders) vs moves search ranked top (genuine misevaluation). The traj probe JSON has `played_argmax_at_drop` per event — analyze it (was not yet examined!). Also re-run traj probe with ckpt9 lens for comparison (`_wf_traj_probe.py 9 9 8 4 ...`).

## 4. Intervention options on the table (owner has NOT chosen yet)

1. **Halt + rollback to ep7 weights + quarantine ep8/9 selfplay shards + pcr_full_proportion 0.25→0.40.** Rationale: ckpt7 = best value CE, cleanest weights; quarantine stops the window flooding; more full searches attack conversion directly (eval-game evidence). Cost ~25–30% slower epochs. RISK: ignition lives in ckpt7's playstyle — marathons may re-ignite (mitigated but not removed by stronger search).
2. Same rollback, no config change — cleanest re-ignition experiment, likely repeats ep8.
3. Stage-2 code change (if/when re-ignition): length-aware row weighting — decay `frequency_weight` for rows with moves_left > M (e.g. 80) or from games > L plies, in `samples.py finalize_game_samples`. Caps the flooding amplifier permanently.
4. Possibly needed deeper fix if H-A holds (defense structurally beats attack): game-level adjudication/anti-stall (engine or selection-side), e.g. moves-left-aware tie-break or stall adjudication — design work, engine team decision.
5. NOT recommended: max_actions reduction (truncated games label ALL rows z=0 — poison); temperature knob changes (not the driver); exploration knob changes (exploration is demonstrably healthy).

## 5. Verification standards for this codebase (hard-won; follow them)

- **Population-level or ≥100-game spread samples only** for trend claims; `_m2_probe.py` first-N-games sampling biases visit-target top1 LOW / entropy HIGH (verified ep6: first-60 H 1.682 vs pop 1.542). Deltas of 0.01–0.04 from single 60-game probes are 1–2 SE noise.
- **By-bucket, never aggregate-only**: row-mix composition shifts (ply / moves_left) explain most aggregate moves at ep8/9. Endgame buckets are the control.
- **Time-order cause vs effect** with cross-epoch probes (`_m2_probe.py RUN CKPT_EP ROWS_EP ...` supports ckpt≠rows epoch).
- **CPU-ONLY probes while the run lives**: prefix `CUDA_VISIBLE_DEVICES=` ; venv `/root/.venvs/hexgt-build/bin/python`; run from `/mnt/e/Hexo-BotTrainer-hexgt/scripts`; WSL stdout truncates from Windows — redirect to files.
- **Load trap:** main1 checkpoints epoch ≤23 silently load a RANDOM value head under current heads_v3 code (pre-heads_v2 keys). Always check `load_state_dict` missing keys for `value*`.
- **opp_policy CE reads ~4× low** by construction (PCR full→full mask, zero-rows kept in denominator, coverage 0.236). Not a bug.
- Run dir is READ-ONLY. Scratch prefix: `scripts/_wf_*`. The hourly cron loop writes `docs/analysis/runQuality_<ts>.md` — keep the convention.

## 6. Artifact index

- This session's probes: `scripts/_wf_m3_probe_ep8_full.json`, `_wf_m3_probe_ep9_full.json` (population policy+value by bucket), `_wf_m3_autopsy_ep8.json`, `_wf_m3_holdout.json` (held-out), `_wf_m3_x78.json` (time-order), `_wf_traj_ep5lens.json` (squander events, **per-event `played_argmax_at_drop` unanalyzed**), `_wf_m3_bias_ep{1,3}.json` (bias emergence), `_wf_bias_parity.txt` (mean_z by moves_left: endgame parity +0.4..0.67 last 3 moves).
- Audit (epochs 1–7, 40 adversarially verified findings): `docs/analysis/MAIN3_RUN_AUDIT.md`.
- Collapse forensics of the predecessor + knob grid: `docs/analysis/RESTNET_EXPLORATION_VALUE_WORKFLOW.md`, `scripts/_grid_ep{2,5,11}.json` (+ `_grid_analyze.py`), `scripts/_explore_grid_probe.py` (live-search knob grid harness — reusable for intervention validation).
- Value-head deep review: session memory `restnet-value-head-autopsy` + `scripts/_value_autopsy.py`.
- Tooling: `_m2_probe.py` (policy/value by bucket), `_value_autopsy.py` (hedge test), `_wf_traj_probe.py` (game trajectories), `_main3_health.py` (trends; NOTE bug: reads `train` key, diagnostics use `training`).

## 7. Suggested attack plan (adapt freely)

1. Read ep10 eval (win% AND game lengths) — the external arbiter the owner predicted would regress.
2. Mine `_wf_traj_ep5lens.json` events: `played_argmax_at_drop` distribution; re-run traj probe with ckpt9 lens; classify squanders (search-ranked-low plays = temp-floor blunders vs top-ranked plays = genuine misevaluation).
3. If GPU obtainable (owner halts run, or accept contention): the all-512 PCR-off selfplay A/B with ckpt7/ckpt9 — the decisive H-B test; and a ckpt9-vs-ckpt7 head-to-head match for the true strength delta (hexo_runner match mode).
4. Inspect marathon games structurally (H-C) — dashboard :8080 debug screen or coordinate analysis of the .hxr records.
5. Assemble the intervention recommendation with predicted effects and a verification plan (e.g., post-rollback: dec/game must stay <130 for 3 epochs, [5,16) pop prior recovering toward 0.25, endgame sign-acc >0.95).
6. Adversarially verify every claim per §5 before presenting.
