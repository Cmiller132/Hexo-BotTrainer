# Stage-3 feasibility: moves-left search utility — decision package

**Date:** 2026-06-12. **Method:** offline-only workflow (no run changes): moves_left head-quality audit (3 checkpoints × 3 game sources × 60 games, 33,940 positions), dawdle-discrimination probe (64 real squander positions + 39 controls), full 5-layer integration design, adversarial verification (sustained; corrections folded in below). Artifacts: `scripts/_wf_s3_RESULT_*.json`, probe outputs `_wf_s3*`, verifier re-derivations `_wf_s3v_*`.

## Verdict: CONDITIONAL GO

The **mechanism is validated and safe**; the **current head is not usable yet**. Build the plumbing defaults-off; enable only behind a head-heal re-audit stamp.

### What the probes established

**The utility would work where it matters.** On the 64 engine-verified squander positions (where main_3's wins were measurably thrown away): a calibrated `value − k·moves_left` bonus flips 12.5–29.7% of choices (k 0.2–0.5) toward predicted-faster conversion at median value cost <0.003, with **zero flips in 15 conversion controls (through k=1.0) and 0/24 mid-game controls (k≤0.5)**. All flips reduce predicted moves-left; flip targets have healthy priors (search wouldn't starve them).

**The current main_4 head would steer search backward.** ckpt12's moves_left head is flood-damaged: chance-level within-game ordering (conversion-zone rank corr 0.17 vs 0.58–0.62 for the very weights it started from, *on identical games*), and **wrong-sign sibling gradients** (−3.7/−19.1 — it thinks dawdling moves end games sooner). Its training loss tells the story: 3.19 → 3.65 (flood) → 3.36 by ep13 — **healing, not healed**.

**Healthy heads carry a real but coarse signal.** End-vs-mid discrimination 0.87–0.91, end-drop 97–100%, conversion-zone rank corr ~0.61–0.67, [0,5) median-decode MAE ~10–13 (beats the ≤15 bar on the unbiased sample). But: no per-ply resolution (use stride-level trends), constant output at [60,120), and **blind to marathons** ([120+) bias −111..−194). Scope honestly: this utility speeds conversion inside the <60-decision zone — exactly where the measured squandering lives — it will **not** prevent marathon onset.

### The design (full detail in `_wf_s3_RESULT_design.json`)

Selection-time PUCT bonus (never backprop/Q-shaping — keeps value pure, no virtual-loss interaction):
`U_ml(e) = − w_ml · g(Q_e) · tanh((M_e − M_n)/m_scale)` with `g` a |Q|>0.6 gate (win-side-only by default), w_ml=0.03, m_scale=32. Delta-vs-sibling-baseline form is invariant to the head's absolute bias by construction; median-of-bins decode. Calibrated to sit inside the probe's validated k∈[0.2,0.5] envelope at |Q|∈[0.8,1.0]. ML stats accumulate only on real backups in separate (ml_sum, ml_weight) fields; terminals contribute exact path-distance (with the verifier's off-by-one fix); PCR fast searches steered identically.

Five layers, ~3–5 days effort, every layer defaults-off and byte-equivalent when disabled: L1 Python head export (new forward, `forward_policy_value` untouched) → L2 Rust wire parse (fail-loud when enabled, ignored when not) → L3 tree stats + selection bonus → L4 config knobs (`[selfplay.moves_left_utility]`) → L5 fp16/TRT coexistence (phase 1 refuses TRT/compile adoption when exporting). **Never rebuild the extension while main_4 runs** — off-switch byte-equivalence is the mandatory deploy gate.

### The enablement gate (L0 — blocks enabling, not building)

Re-run the audit (`scripts/_wf_s3_mlaudit.py` + sibling probe, CPU, ~hours) on candidate checkpoints until: conversion-zone within-game Spearman ≥0.6 AND [0,5) median-decode MAE ≤15 AND end-vs-mid pairwise ≥0.85 AND correct-sign sibling decrements. Cheapest heal path: simply later main_4 checkpoints (post-quarantine data; loss already recovering). If stalled ~3 epochs: raise `moves_left_weight` 0.1→0.3–0.5 (one-line config). The verifier **refuted** the suspected plies-vs-decisions unit mismatch — no retraining needed for units.

### Validation ladder before any multi-epoch enable

Rust+Python unit tests → off-switch byte-equivalence smoke → head re-audit stamp → dawdle probe re-run **through the real Rust search** (pass: ≥10% squander flips, 0 control flips at w=0.03) → 1-epoch GPU A/B (400-game anchor match, dec/game by phase, conversion hold rate) → staged enable at 0.03 with predefined aborts (eval drop >3pts, any control flip in the nightly probe, p90 lengths worsening ×2).

### Key risks (verified as disclosed)

Sign-rudder risk: the |v|>0.6 leader read predicted the true winner only 48% at squander points — mitigated by gating on search-averaged Q, win-side-only default, 0.03 cap (empirical-margin argument, not the invalid median-gap one). Policy targets intentionally absorb utility-shifted visits (monitor policy CE). Interaction with C1/C3: all push shorter — hold them fixed in the A/B; if the utility works, *relax* C1 rather than stack. One-ply-probe-to-search gap covered by ladder step 5.
