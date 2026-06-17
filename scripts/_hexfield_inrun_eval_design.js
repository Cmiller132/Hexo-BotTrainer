export const meta = {
  name: 'hexfield-inrun-eval-design-v2',
  description: 'Design a CONSISTENT 512-sim comparable 72-game in-run per-epoch eval + integration + nice graphs',
  phases: [
    { title: 'Design', detail: 'allocation+clean-pool, integration, graphs' },
    { title: 'Review', detail: 'impl-correctness + comparability/stats-honesty' },
    { title: 'Synthesis', detail: 'one implementation-ready spec' },
  ],
}

const REPO = 'E:\\\\Hexo-BotTrainer-hexgt'
const RUN = 'E:\\\\Hexo-BotTrainer\\\\runs\\\\hexfield_main_1'

const LOCKED = `LOCKED DECISIONS (do NOT re-litigate, do NOT optimize for speed — the owner wants CONSISTENCY +
COMPARABILITY + NICE GRAPHS, not minimal wall-time):
- search visits = 512 ALWAYS (the production budget). Every eval, every epoch, every opponent. No 256, no reductions.
- virtual_batch_size = 16 for the eval, FIXED and UNIFORM across all epochs (it only changes MCTS leaf
  parallelism / virtual-loss, a negligible & symmetric strength effect, locked once so all measurements are
  comparable). It must NOT change self-play's SelfplayConfig.virtual_batch_size = 4. (Precedent: the old
  _play_pair hardcodes vbs=8 for eval.) So an EVAL-SPECIFIC vbs must be threaded into the eval search calls.
- 72 total games per eval (= 36 CRN paired games). Fixed roster, no sealbot (unavailable in this env).
- The whole point is a CLEAN, internally-consistent dataset so the progress curve is apples-to-apples across
  epochs -> the existing pool (mixed vbs/visits, sparse) should be RESET and REBUILT uniformly at 512/vbs16,
  and past epochs BACKFILLED at the same config so there is an immediate comparable history to graph.`

const CONTEXT = `GOAL: Make hexfield's IN-RUN per-epoch eval a statistically-meaningful, CONSISTENT, COMPARABLE
fixed-roster eval (so we get nice learning-curve graphs), and INTEGRATE it into the training run. Stay PURE
EVAL (report PROMOTE/REGRESS/INCONCLUSIVE; never gate/halt/promote) and preserve the MLH head audit + heal-gate.

CURRENT CODE (read it; don't trust this blindly):
- Loop: ${REPO}\\\\packages\\\\hexo_train\\\\python\\\\hexo_train\\\\epoch\\\\loop.py calls plugin.evaluate_epoch().
- Plugin: ${REPO}\\\\packages\\\\hexfield\\\\python\\\\hexfield\\\\plugin.py registers evaluate_epoch = evaluation.evaluate_epoch.
- Current eval: ${REPO}\\\\packages\\\\hexfield\\\\python\\\\hexfield\\\\evaluation.py:evaluate_epoch -- 16 GREEDY games vs the
  IMMEDIATELY-PRIOR epoch @128 visits (uninformative); _play_pair HARDCODES vbs=8; runs every eval_every=5 epochs;
  ALSO runs the MLH head audit (audit_moves_left_head) + heal-gate (ML_AUTO_DISABLED_FLAG) -- PRESERVE THIS EXACTLY.
- Reuse (do NOT rewrite): ${REPO}\\\\packages\\\\hexfield\\\\python\\\\hexfield\\\\multistage_eval.py
  (run_multistage_eval / run_multistage_eval_in_parts / aggregate_pool / run_eval_part / select_opponents /
  allocate_budget; 4 stages incl rolling Bradley-Terry pool diagnostics/eval_pool.json, persisted, COMPOUNDS
  across epochs; verdict = candidate-vs-champion BT difference-CI; PURE EVAL). eval_arena.py
  (play_checkpoint_match -- CONCURRENT multi-root batched, CRN paired/pentanomial; uses sp.virtual_batch_size).
  eval_stats.py (converged BT, pentanomial, Wilson). config.py MultiStageEvalSection (enabled, games_budget,
  every_n_epochs, full_search_visits, opponents{permanent_anchors=(bc_prefit,ep5), log_grid=(5,10,20,40,80,160),
  bracket_size=2, sealbot_enabled}, sprt, pool_path, verdict_reference_lag=5, eval_gating/promotion_enabled=False).
- Spec: ${REPO}\\\\docs\\\\specs\\\\hexfield_eval_v2_spec.md (HONEST significance: a single epoch at 128 games resolves
  only ~100-120 Elo; tight resolution is a MULTI-EPOCH ROLLING asymptote. 72 games resolves even less per-epoch;
  significance accrues via the persisted pool. Do NOT overclaim per-epoch.)
- Dashboard (for the graphs): ${REPO}\\\\packages\\\\hexo_frontend\\\\python\\\\hexo_frontend\\\\web.py and static\\\\app.js /
  index.html render run history at :8080. Run diagnostics live in ${RUN}\\\\diagnostics (per-epoch JSONs,
  hexfield.multistage_eval.epoch_*.json, eval_pool.json).
- SealBot UNAVAILABLE (no SEALBOT_PATH/binary) -> eval runs sealbot-off; BT pool anchors on bc_prefit (lowest
  permanent anchor; _choose_anchor handles it).
- Eval runs as a SEQUENTIAL phase after the epoch checkpoint is saved (full GPU); in-process it can reuse
  self-play's compiled forward. Measured (FYI only, NOT to optimize): vbs=16 @512 visits ~20 s/game on a
  worst-case contested edge, anchor blowouts much faster -> 72 games is on the order of ~15-25 min/epoch.

${LOCKED}

DELIVERABLE INCLUDES NICE GRAPHS: the owner wants to SEE the learning curve. Design visualization(s) from
eval_pool.json + the per-epoch multistage JSONs: at minimum (a) pooled Bradley-Terry Elo per candidate epoch
with 95% CIs (the headline "still learning" curve), and (b) candidate win-rate vs EACH fixed anchor (bc_prefit,
ep5) over epochs with CIs. They must be comparable (uniform config) and honest (show CIs/uncertainty).`

const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['area', 'design', 'concrete_details', 'risks'],
  properties: {
    area: { type: 'string' },
    design: { type: 'string', description: 'the design for this area' },
    concrete_details: { type: 'array', items: { type: 'string' }, description: 'exact values / file:function edits / commands' },
    risks: { type: 'array', items: { type: 'string' } },
  },
}
const REVIEW_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lens', 'findings', 'must_fix', 'recommendation'],
  properties: {
    lens: { type: 'string' },
    findings: { type: 'array', items: { type: 'string' } },
    must_fix: { type: 'array', items: { type: 'string' } },
    recommendation: { type: 'string' },
  },
}
const SPEC_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['summary', 'config_changes', 'code_changes', 'allocation_final', 'pool_and_backfill', 'graphs', 'significance_statement', 'test_plan', 'verification', 'open_risks'],
  properties: {
    summary: { type: 'string' },
    config_changes: { type: 'array', items: { type: 'string' } },
    code_changes: { type: 'array', items: { type: 'string' }, description: 'ordered file:function-level edits' },
    allocation_final: { type: 'string' },
    pool_and_backfill: { type: 'string', description: 'reset pool + backfill which epochs at 512/vbs16, exact commands' },
    graphs: { type: 'array', items: { type: 'string' }, description: 'what to plot, from what data, output form (script + dashboard)' },
    significance_statement: { type: 'string' },
    test_plan: { type: 'array', items: { type: 'string' } },
    verification: { type: 'string' },
    open_risks: { type: 'array', items: { type: 'string' } },
  },
}

const DESIGNERS = [
  { key: 'allocation-pool', area: 'allocation + clean comparable pool', prompt: 'Design, at the LOCKED 512 sims / vbs16 / 72 games / no-sealbot: how the 72 games split across the roster (bc_prefit + ep5 fixed anchors = the compounding progress curve; champion = the verdict edge; bracket) for a COMPARABLE progress curve. Decide SPRT on/off. Specify the CLEAN-POOL plan: reset eval_pool.json and BACKFILL which past epochs (candidates among ep5,10,15,20,25) at the locked config so there is an immediate comparable history to graph, with exact runner commands (scripts/_hexfield_run_multistage_eval.py ... --full-visits 512 --no-sealbot --parts, eval-vbs via config). Honest single-epoch-vs-pooled significance.' },
  { key: 'integration', area: 'pipeline integration', prompt: 'Design the integration into evaluation.py:evaluate_epoch at the LOCKED config: reuse run_multistage_eval(_in_parts) verbatim; thread the EVAL-SPECIFIC vbs=16 into the eval search calls WITHOUT changing self-play vbs=4 (find the exact place in eval_arena.py the search vbs is set, and how to plumb an override — e.g. a new MultiStageEvalSection.eval_virtual_batch_size threaded like full_search_visits via _eval_visits); PRESERVE the MLH head audit + heal-gate; gate by config (enabled + every_n_epochs); strictly PURE EVAL; FAIL-SOFT (wrap so an eval exception never breaks the training epoch). CRITICAL: trace whether evaluate_epoch has the candidate epoch CHECKPOINT on disk at eval time (the runner takes a path) or must pass the in-memory model — confirm from loop.py + evaluation.py and specify how the candidate is provided.' },
  { key: 'graphs', area: 'nice graphs / visualization', prompt: 'Design the "nice graphs". Read eval_pool.json + a hexfield.multistage_eval.epoch_*.json to know the exact data shape. Specify: (a) pooled BT Elo per candidate epoch with 95% CIs (headline still-learning curve); (b) candidate win-rate vs each fixed anchor (bc_prefit, ep5) over epochs with Wilson CIs. Recommend output form: a self-contained standalone generator (script reading the pool -> HTML/SVG or PNG) AND whether to add a dashboard panel — read hexo_frontend web.py + static/app.js + index.html to judge how run history is currently rendered and whether a pool-curve panel fits. Make them comparable + honest (always show CIs). Give concrete plotting fields and file outputs.' },
]
const CRITICS = [
  { key: 'impl-correctness', prompt: 'Read the code and verify implementation correctness: (1) EXACT location/mechanism to thread eval-specific vbs into eval_arena.play_checkpoint_match search calls (quote the lines); (2) candidate checkpoint path at eval time in evaluate_epoch (trace it); (3) which existing tests (test_hexfield_multistage_eval, test_hexfield_eval_arena_concurrent, test_hexfield_eval_parts, test_hexfield_eval_harden) are affected and how to keep them green; (4) config wiring for any new MultiStageEvalSection field (parse_hexfield_config + _merge + toml). List concrete must-fixes.' },
  { key: 'comparability-stats', prompt: 'Critique for COMPARABILITY and STATISTICAL HONESTY: is the rebuilt pool truly apples-to-apples (uniform 512/vbs16, same roster, same opening sampling)? Is BT anchored on bc_prefit valid with sealbot absent? Are pair-level/pentanomial/CRN and effective counts used so CIs are not anti-conservative? Does the graph design faithfully show uncertainty (CIs) and avoid implying per-epoch significance the data lacks? Is backfilling past epochs at the new config sound, and does it correctly NOT mix with the old vbs4 edges?' },
]

phase('Design')
const designs = (await parallel(
  DESIGNERS.map(d => () => agent(
    `${d.prompt}\n\nRead the cited code/files before designing. Be concrete and implementable.\n\n=== CONTEXT ===\n${CONTEXT}`,
    { label: `design:${d.key}`, phase: 'Design', schema: DESIGN_SCHEMA }
  ))
)).filter(Boolean)
const digest = JSON.stringify(designs, null, 1)

phase('Review')
const reviews = (await parallel(
  CRITICS.map(c => () => agent(
    `${c.prompt}\n\nRead the cited code to verify.\n\n=== CONTEXT ===\n${CONTEXT}\n\n=== DESIGNS ===\n${digest}`,
    { label: `review:${c.key}`, phase: 'Review', schema: REVIEW_SCHEMA }
  ))
)).filter(Boolean)

phase('Synthesis')
const spec = await agent(
  `Lead engineer: merge the designs + critiques into ONE implementation-ready spec for the CONSISTENT 512-sim / vbs16 / 72-game comparable in-run per-epoch eval, integrated into the run, PURE EVAL, head-audit preserved, fail-soft, gated by config; plus the clean pool reset+backfill and the nice graphs. Resolve every must_fix. Output exact config fields/values, ordered file:function-level code edits, the pool reset+backfill commands, the graph generator(s), a test plan, and an end-to-end verification. Do NOT optimize for speed; 512 sims is fixed.\n\n=== CONTEXT ===\n${CONTEXT}\n\n=== DESIGNS ===\n${digest}\n\n=== CRITIQUES ===\n${JSON.stringify(reviews, null, 1)}`,
  { label: 'synthesis', phase: 'Synthesis', schema: SPEC_SCHEMA }
)

return { designs, reviews, spec }
