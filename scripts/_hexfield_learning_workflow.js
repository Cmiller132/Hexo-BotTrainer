export const meta = {
  name: 'hexfield-still-learning',
  description: 'Exhaustive multi-agent determination of whether hexfield_main_1 is still learning (epochs 1-25)',
  phases: [
    { title: 'Diagnostic lenses', detail: '6 independent lenses over existing diagnostics' },
    { title: 'Adversarial refutation', detail: '4 skeptics try to refute "still learning"' },
    { title: 'Synthesis', detail: 'reconcile into a verdict + ladder expectations' },
  ],
}

// ---- Shared, grounded context (extracted from the run diagnostics this turn) ----
const DIAG = 'E:\\\\Hexo-BotTrainer\\\\runs\\\\hexfield_main_1\\\\diagnostics'
const REPO = 'E:\\\\Hexo-BotTrainer-hexgt'

const TRAJECTORY = `Per-epoch trajectory (hexfield_main_1, epochs 1-25), extracted from ${DIAG}\\\\epoch_0000NN.json:
ep  L_pol  L_val  L_ml  L_tot  mean_glen  rootV   entropy trunc gnorm  ev_wr  ev_opp  ml_convSpear
1   2.164  0.598  2.228 4.478  61.2       -0.029  1.000   0     2.561  -      -       -
2   2.084  0.486  2.062 4.171  50.9       -0.021  1.224   0     2.295  -      -       -
3   2.059  0.454  2.086 4.013  58.0        0.004  1.145   0     2.329  -      -       -
4   2.184  0.477  2.279 4.152  67.4        0.014  1.368   0     2.387  -      -       -
5   2.234  0.449  2.184 4.131  60.4       -0.006  1.391   0     2.514  0.562  ep4     -
6   2.259  0.431  2.313 4.133  79.3       -0.002  1.389   0     2.577  -      -       -
7   2.322  0.473  2.392 4.262  87.6       -0.039  1.339   0     2.728  -      -       -
8   2.311  0.443  2.339 4.203  65.6       -0.016  1.219   0     2.792  -      -       -
9   2.358  0.444  2.346 4.254  79.6        0.029  1.315   0     2.808  -      -       -
10  2.369  0.449  2.405 4.283  99.7       -0.024  1.125   0     2.846  0.375  ep9     -
11  2.395  0.504  2.520 4.407  92.4       -0.016  1.165   1     2.730  -      -       -
12  2.347  0.506  2.574 4.357  102.2      -0.004  1.122   4     2.665  -      -       -
13  2.538  0.513  2.612 4.557  109.1       0.012  1.613   10    2.656  -      -       -
14  2.553  0.505  2.594 4.551  90.3       -0.004  1.494   2     2.648  -      -       -
15  2.679  0.506  2.630 4.680  95.9        0.013  1.684   2     2.629  0.438  ep14    0.617
16  2.690  0.507  2.657 4.685  104.3      -0.015  1.313   4     2.713  -      -       -
17  2.866  0.510  2.669 4.872  120.7       0.017  1.744   13    2.674  -      -       -
18  2.863  0.507  2.654 4.857  109.7       0.008  1.583   3     2.783  -      -       -
19  2.943  0.482  2.636 4.918  97.2        0.020  1.691   3     2.804  -      -       -
20  2.988  0.473  2.626 4.958  105.3       0.012  1.685   4     2.825  0.438  ep19    0.732
21  3.162  0.474  2.599 5.135  106.9       0.070  1.888   4     2.861  -      -       -
22  3.168  0.477  2.604 5.143  115.4       0.033  1.675   9     2.852  -      -       -
23  3.164  0.459  2.571 5.120  90.6        0.002  1.618   1     2.879  -      -       -
24  3.155  0.446  2.580 5.095  96.2        0.022  1.708   5     2.984  -      -       -
25  3.067  0.444  2.580 5.003  112.0       0.013  1.504   6     2.980  0.438  ep24    0.669

Notes: clip_fraction=1.0 every epoch (grad_clip=1.0, always clips). max_game_plies=256.
Auto-eval plays ONLY the immediately-prior epoch (16 games, every 5 epochs).
The ONLY fixed-baseline measurement is the eval-v2 multistage pool at ep15:
  ${DIAG}\\\\hexfield.multistage_eval.epoch_000015.json and ${DIAG}\\\\eval_pool.json
  ep15 vs bc_prefit 93.75% (Elo +470), vs ep5 90.6% (+394), vs sealbot 71.9% (+163).
  Pooled BT: cand_ep15 +166 Elo, ep5 -225, bc_prefit -299 (sealbot pinned 0).
  Verdict INCONCLUSIVE only because champion (ep10) had no edge that run.
A background eval-v2 ladder is NOW RUNNING (candidates ep20, ep25, ep10 at 256 visits,
budget 64, no-sealbot, --parts) appending edges to ${DIAG}\\\\eval_pool.json; live log
${DIAG}\\\\_ladder_progress.log. It is NOT finished during this workflow.`

const LENS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension', 'summary', 'key_evidence', 'still_learning_vote', 'confidence', 'caveats'],
  properties: {
    dimension: { type: 'string' },
    summary: { type: 'string', description: '3-6 sentence finding' },
    key_evidence: { type: 'array', items: { type: 'string' }, description: 'specific numbers/file:field citations' },
    still_learning_vote: { type: 'string', enum: ['yes', 'no', 'unmeasurable_from_this_lens'] },
    confidence: { type: 'number', description: '0..1' },
    caveats: { type: 'array', items: { type: 'string' } },
  },
}

const REFUTE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['angle', 'strongest_case', 'verdict', 'evidence', 'what_would_settle_it'],
  properties: {
    angle: { type: 'string' },
    strongest_case: { type: 'string', description: 'the strongest version of the refutation' },
    verdict: { type: 'string', enum: ['refutation_holds', 'refutation_fails', 'partial'] },
    evidence: { type: 'array', items: { type: 'string' } },
    what_would_settle_it: { type: 'string', description: 'the specific ladder result that resolves it' },
  },
}

const SYNTH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verdict', 'confidence', 'headline', 'supporting', 'against', 'ladder_expectations', 'recommendation'],
  properties: {
    verdict: { type: 'string', enum: ['still_learning', 'plateaued', 'degrading', 'inconclusive_pending_ladder'] },
    confidence: { type: 'number' },
    headline: { type: 'string' },
    supporting: { type: 'array', items: { type: 'string' } },
    against: { type: 'array', items: { type: 'string' } },
    ladder_expectations: {
      type: 'object', additionalProperties: false,
      required: ['if_still_learning', 'if_plateaued', 'key_numbers_to_watch'],
      properties: {
        if_still_learning: { type: 'string' },
        if_plateaued: { type: 'string' },
        key_numbers_to_watch: { type: 'array', items: { type: 'string' } },
      },
    },
    recommendation: { type: 'string' },
  },
}

const LENSES = [
  {
    key: 'loss-nonstationarity',
    prompt: `LENS: loss drift vs non-stationarity. Test the hypothesis that the rising policy loss is BENIGN (driven by longer games / harder targets), NOT degradation.
Using the trajectory below, assess the correlation between loss_policy (2.16->3.07) and mean_game_length (61->112) and truncated_games (0->~6-13) across epochs 1-25. Is rising L_pol consistent with non-stationary self-play targets rather than model decay? Cross-check: is loss_value stable? is entropy collapsing (->0) or rising? Read ${DIAG}\\\\epoch_000025.json and ${DIAG}\\\\epoch_000017.json for the full per-epoch breakdown if needed. Conclude whether the loss signal argues for, against, or is silent on "still learning".`,
  },
  {
    key: 'selfplay-health',
    prompt: `LENS: self-play dynamics health. Examine root_value_mean (should hover ~0 for balanced play; collapse = value saturation), root_policy_entropy (collapse->0 = mode collapse; healthy = sustained), truncated_games trend, and the full/fast move mix + early_stops from the selfplay scheduler. Read ${DIAG}\\\\epoch_000025.json (metadata.result.selfplay.scheduler) and ${DIAG}\\\\hexfield.selfplay.epoch_000025.json. Does self-play look like a healthy, exploring, improving agent or a collapsing/degenerate one? Vote on "still learning" from this lens.`,
  },
  {
    key: 'mlh-head',
    prompt: `LENS: moves-left (decisiveness) head trajectory. The MLH head is audited each eval epoch. Compare conv_spearman_mean and full_spearman_mean and the near_end_mae and per-bucket MAE across ep15 (0.617/0.8), ep20 (0.732/0.848), ep25 (0.669/0.8). Read ${DIAG}\\\\hexfield.evaluation.epoch_000020.json and ${DIAG}\\\\hexfield.evaluation.epoch_000025.json. Is the auxiliary head improving, stable-and-healthy, or degrading? Note ml_auto_disabled status. A healthy/improving aux head is weak positive evidence of continued learning; a collapsing one is a red flag.`,
  },
  {
    key: 'autoeval-interpretation',
    prompt: `LENS: per-epoch auto-eval (vs immediately-prior). Win rates: ep5 0.562, ep10 0.375, ep15 0.438, ep20 0.438, ep25 0.438 (16 games each, vs the prior epoch). Explain rigorously WHY this metric is near-uninformative for "still learning" (adjacent highly-correlated checkpoints; 16-game binomial 95% half-width ~0.24). Then extract whatever residual signal exists: is there a consistent <0.5 bias and what could cause it (e.g. prior epoch is a strong/over-fit opponent, first-move/seat effects, draw handling)? Read ${DIAG}\\\\hexfield.evaluation.epoch_000025.json. Vote (likely "unmeasurable_from_this_lens").`,
  },
  {
    key: 'evalv2-pool-audit',
    prompt: `LENS: audit the existing eval-v2 measurement (the ONLY fixed-baseline data so far). Read ${DIAG}\\\\hexfield.multistage_eval.epoch_000015.json and ${DIAG}\\\\eval_pool.json IN FULL. Verify: did the Bradley-Terry fit CONVERGE (max|grad| < 1e-6)? Are the ep15 edges vs bc_prefit/ep5/sealbot statistically clean (pentanomial, pair SE, CI widths)? What does it PROVE about learning up to ep15, and what are its limits (no champion edge -> INCONCLUSIVE verdict; single epoch resolves only ~100-120 Elo)? Also read ${DIAG}\\\\_ladder_progress.log and re-read eval_pool.json to report any NEW edges the running ladder has appended so far (ep20/ep25/ep10 candidates). Vote on what is conclusively established.`,
  },
  {
    key: 'training-stability',
    prompt: `LENS: training-loop stability & optimizer health. Examine grad_norm_mean (~2.3-2.98), clip_fraction (=1.0 every epoch), amp_scale, window_rows/steps, and loss_value/loss_opp_policy/loss_stvalue_* across epochs. Read ${DIAG}\\\\epoch_000025.json and ${DIAG}\\\\epoch_000011.json (epoch 11 had a known optimizer-device resume bug that was fixed). Is the optimizer healthy and is the model still receiving a meaningful, non-saturating gradient signal (consistent with ongoing learning), or are there signs of saturation/instability? Also skim ${DIAG}\\\\events.jsonl for any error/halt/breaker events. Vote.`,
  },
]

const REFUTERS = [
  { key: 'plateau', angle: 'The model PLATEAUED after ep15', prompt: `Argue the STRONGEST case that hexfield stopped improving after ep15: rising policy loss, flat ~0.44 auto-eval, no fixed-baseline gain shown past ep15. Then state honestly whether the available evidence (and the benign-drift counter-argument) refutes your case. Identify the exact ladder result that would settle it (e.g. ep20/ep25 BT Elo vs the fixed bc_prefit/ep5 anchors must exceed ep15's +166).` },
  { key: 'noise', angle: 'Any apparent gain is within statistical noise', prompt: `Argue that per-epoch resolution is only ~100-120 Elo (single-epoch SE(r_L-r_B) ~40-55 Elo), so any ep20/ep25 vs champion difference is likely INCONCLUSIVE and cannot prove learning. Then assess the counter: the rolling pool compounds and the FIXED anchors (bc_prefit, ep5) accumulate games, so the cand-vs-anchor progress curve can be significant even when cand-vs-champion is not. What would settle it?` },
  { key: 'masked-degradation', angle: 'Benign-drift framing masks real degradation', prompt: `Argue that attributing rising loss to longer games could be hiding genuine policy/value degradation. Look for any corroborating red flags in the trajectory (value-loss creep, entropy spikes, truncation storms, root-value drift away from 0). Then judge whether the data supports masked degradation or refutes it. What ladder/diagnostic result settles it?` },
  { key: 'measurement-validity', angle: 'The eval-v2 measurement itself is invalid', prompt: `Argue the ladder measurement could be untrustworthy: SealBot unavailable (scale anchored only via one ep15 sealbot edge + bc_prefit), mixed search visits (ep15 edges at 512 vs new at 256), BT non-convergence, anchor drift, or too few pairs. Read ${DIAG}\\\\eval_pool.json to check edge counts/weights/convergence. Then judge how much these threaten the conclusion and how to mitigate (e.g. rely on cand-vs-fixed-anchor curve; verify max|grad|; note 256 vs 512 makes "still improving past ep15" a CONSERVATIVE bar). What settles it?` },
]

phase('Diagnostic lenses')
const lensResults = (await parallel(
  LENSES.map(l => () => agent(
    `${l.prompt}\n\n=== SHARED CONTEXT ===\n${TRAJECTORY}\n\nReturn the structured finding. Cite specific numbers and file:field. Be rigorous and concise.`,
    { label: `lens:${l.key}`, phase: 'Diagnostic lenses', schema: LENS_SCHEMA }
  ))
)).filter(Boolean)

const lensDigest = JSON.stringify(lensResults, null, 1)

phase('Adversarial refutation')
const refuteResults = (await parallel(
  REFUTERS.map(r => () => agent(
    `${r.prompt}\n\n=== SHARED CONTEXT ===\n${TRAJECTORY}\n\n=== DIAGNOSTIC LENS FINDINGS (phase 1) ===\n${lensDigest}\n\nReturn the structured refutation. Steelman first, then judge honestly.`,
    { label: `refute:${r.key}`, phase: 'Adversarial refutation', schema: REFUTE_SCHEMA }
  ))
)).filter(Boolean)

phase('Synthesis')
const synthesis = await agent(
  `You are the lead analyst. Reconcile the diagnostic lenses and the adversarial refutations into ONE conclusive determination of whether hexfield_main_1 is STILL LEARNING (improving in real playing strength), with calibrated confidence.\n\n` +
  `Ground rules: the per-epoch auto-eval (vs prior) is near-uninformative; rising policy loss alongside growing game length is the documented BENIGN drift; the conclusive signal is the eval-v2 fixed-anchor Elo curve. Up to ep15 learning is already PROVEN (ep15 >> ep5 >> bc_prefit). The open question is whether strength still climbs ep15->ep20->ep25. The eval-v2 ladder producing that data is RUNNING but not finished, so your verdict should be the best call from current evidence AND a precise, falsifiable statement of what the ladder must show.\n\n` +
  `Re-read ${DIAG}\\\\eval_pool.json and ${DIAG}\\\\_ladder_progress.log NOW to fold in any edges produced so far.\n\n` +
  `=== SHARED CONTEXT ===\n${TRAJECTORY}\n\n=== LENS FINDINGS ===\n${lensDigest}\n\n=== REFUTATIONS ===\n${JSON.stringify(refuteResults, null, 1)}\n\n` +
  `Return the structured synthesis.`,
  { label: 'synthesis', phase: 'Synthesis', schema: SYNTH_SCHEMA }
)

return { lensResults, refuteResults, synthesis }
