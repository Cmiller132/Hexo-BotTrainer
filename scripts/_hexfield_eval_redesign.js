export const meta = {
  name: 'hexfield-eval-resignificance',
  description: 'Redesign the in-run eval to be statistically MEANINGFUL and FAST (few min, concurrent): ~32 vs SealBot + ~32 vs checkpoints, every 5 epochs, applied without interrupting the run',
  phases: [
    { title: 'Design', detail: 'allocation+significance, sealbot+anchor, performance+concurrency, apply-without-interrupt' },
    { title: 'Review', detail: 'stats-honesty + impl-correctness' },
    { title: 'Synthesis', detail: 'implementation-ready spec' },
  ],
}

const REPO = 'E:\\\\Hexo-BotTrainer-hexgt'
const RUN = 'E:\\\\Hexo-BotTrainer\\\\runs\\\\hexfield_main_1'

const CONTEXT = `GOAL: Redesign hexfield's IN-RUN per-epoch eval so it is STATISTICALLY MEANINGFUL, restoring SealBot
as the cross-lineage anchor the way it worked earlier in this run: ~32 games vs SealBot + ~32 games vs
checkpoints (a 1:1 split, NOT the legacy 1:3), every 5 epochs, at 512 visits / eval-vbs 16. The owner found
the current 40-game/no-sealbot eval too thin to be meaningful. Output an implementation-ready spec. HARD
CONSTRAINT: the redesign must NOT require interrupting the LIVE training run now — config/supervisor changes
take effect on the next restart; the spec must say exactly how/when it applies and confirm nothing forces a
restart now.

HOW IT WORKED EARLIER (the SealBot eval to restore — read the code):
- ${REPO}\\\\packages\\\\hexfield\\\\python\\\\hexfield\\\\multistage_eval.py: allocate_budget(total, n_checkpoint_opponents,
  has_sealbot, sealbot_share=0.25) — sealbot_share is a HARDCODED function default (NOT a config knob); at the
  legacy games_budget=128 it gave round(128*0.25)=32 SealBot + 96 checkpoints (this is the "32 vs SealBot"
  the owner remembers). _stage_c_deep / _play_sealbot_opponent / run_eval_part play SealBot UNPAIRED via
  eval_arena.play_sealbot_match (concurrent multi-root; SealBot drained serially per game). The SealBot edge
  is DOWN-WEIGHTED in the BT fit (weight = sealbot_overdispersion = 0.5) because SealBot's 50ms minimax depth
  VARIES UNDER MACHINE LOAD (non-stationary) — it pins the 0-Elo scale but must not carry full difference info.
- ${REPO}\\\\packages\\\\hexfield\\\\python\\\\hexfield\\\\config.py MultiStageEvalSection: sealbot_enabled=True(default),
  sealbot_path=None(falls back to $SEALBOT_PATH), sealbot_variant="current", sealbot_time_limit=0.05,
  sealbot_overdispersion=0.5, games_budget=128(default). Anchors: permanent_anchors=(bc_prefit, ep5),
  log_grid=(5,10,20,40,80,160), bracket_size=2, verdict_reference_lag=5.
- Historical SealBot edge (ep15, pre-lock pool): 23-9 (32 decided games) at 512 visits, weight 0.5.
- eval_arena.play_sealbot_match exists and is wired; it now also accepts the eval-vbs override (added today).
- Spec: ${REPO}\\\\docs\\\\specs\\\\hexfield_eval_v2_spec.md (honest significance: a single epoch at 128 games resolves
  only ~100-120 Elo; the ~15-20 Elo resolution is the MULTI-EPOCH ROLLING-POOL asymptote. SealBot is the
  zero-point ONLY, down-weighted, never the primary verdict comparator. Hexo has NO draws -> binomial base.)

CURRENT STATE (what the redesign must reconcile):
- Config now (${REPO}\\\\configs\\\\hexfield_main_1.toml [model.config.multi_stage_eval]): enabled=true,
  games_budget=40, every_n_epochs=5, full_search_visits=512, eval_virtual_batch_size=16, sprt.enabled=false,
  opponents.sealbot_enabled=FALSE. THE SEALBOT-OFF IS THE GAP TO FIX.
- Pool ${RUN}\\\\diagnostics\\\\eval_pool.json: candidates ep10/15/20/25/26/27/28/29/30 (~36 edges) are ALL
  bc_prefit-ANCHORED (no SealBot edges, because the in-run eval ran --no-sealbot). multistage_eval._choose_anchor
  prefers SealBot whenever ANY sealbot edge exists -> adding SealBot edges for ep35+ will FLIP the whole pool's
  anchor from bc_prefit to SealBot. The redesign MUST address this: do ep10-30 (no sealbot edge) still connect
  to SealBot through shared opponents (bc_prefit/ep5) so the curve stays one comparable scale? Is the re-anchor
  desirable (cross-lineage) or should the pool be reset? State the consequence on the existing learning curve.
- Supervisor ${REPO}\\\\scripts\\\\_hexfield_supervise_main1.sh launches train_model and regenerates
  ${RUN}\\\\_resume_config.toml from configs/hexfield_main_1.toml on each (re)launch. It does NOT export
  SEALBOT_PATH today -> with sealbot_enabled=true but no path, the eval FAIL-OPENS (no sealbot edge, logged).
  SealBot binary is at /mnt/e/SealBot/{current,best} (variant "current"). So enabling SealBot needs the path
  wired (supervisor env SEALBOT_PATH=/mnt/e/SealBot, or config sealbot_path).
- evaluate_epoch is FAIL-SOFT (an eval exception never breaks training) and PURE EVAL (no gating). The in-run
  eval runs as a sequential phase after the epoch checkpoint, sharing the GPU. ENABLING SealBot means SealBot's
  minimax runs concurrently with the (just-finished) training process's GPU residency -> its depth is even more
  load-variable; weigh whether sealbot_time_limit / overdispersion need adjusting for the in-run setting.

PERFORMANCE REQUIREMENT (FIRST-CLASS — the eval must be FAST, target a FEW MINUTES):
- The owner observed the ORIGINAL SealBot eval was fast (a few minutes) because it ran ALL GAMES AT ONCE
  (concurrently batched). The CURRENT in-run eval is SLOW because evaluate_epoch calls
  run_multistage_eval_in_parts, which plays opponents SEQUENTIALLY (one play_checkpoint_match per opponent,
  one part at a time). So wall-clock = the SUM of per-opponent matches, not the MAX. The champion checkpoint
  edge also plays long contested games near the 256-ply cap (the slow tail).
- Within a single matchup, play_checkpoint_match IS concurrent (it batches that matchup's games through one
  multi-root HexfieldMctsSession.search per round); play_sealbot_match batches all SealBot games concurrently
  too. The problem is ACROSS opponents (the parts are serial), so concurrency is wasted between matches.
- Earlier GPU profiling during eval: GPU only ~40-60% utilised while ONE CPU core sat at ~96% — the eval is
  partly CPU-ROUND-BOUND (the per-round Python/Rust orchestration). The number of search ROUNDS per ply is
  visits / virtual_batch_size (512/16 = 32 rounds/ply); a fully concurrent batched run amortises the per-round
  cost across all in-flight games AND makes wall-clock ~= the LONGEST single game's rounds rather than the SUM.
- DESIGN FOR SPEED (the core ask): run the eval CONCURRENTLY across ALL opponents in one batched pass (all ~64
  games in flight; the candidate net's moves batch across every game in one forward, each opponent net's moves
  batch within its own group; SealBot's serial drain stays per-game as today). Determine whether this needs a
  NEW multi-opponent concurrent runner or whether run_multistage_eval's path can be driven concurrently, and
  whether it can REUSE play_checkpoint_match's batched round loop. ALSO weigh, at FIXED 512 visits (kept for
  comparability): raising eval_virtual_batch_size (fewer rounds/ply -> fewer CPU round-trips; applied uniformly
  so comparability holds) and any GPU-saturation win. Give a concrete wall-time estimate for the chosen design.
- Net target: ~32 SealBot + ~32 checkpoints, MEANINGFUL, finishing in a FEW MINUTES (concurrent), every 5 epochs.

DESIGN QUESTIONS TO RESOLVE:
1. The 1:1 split (32 SealBot + 32 checkpoints). allocate_budget's sealbot_share is hardcoded 0.25. Options:
   (a) add a config knob (e.g. sealbot_share or sealbot_games) threaded into allocate_budget; (b) tune
   games_budget; (c) other. Pick one and specify it. Then split the 32 checkpoint games across the roster
   (bc_prefit, ep5, bracket, L-5 champion) to MAXIMIZE information (the fixed anchors compound; the champion
   carries the verdict).
2. Is ~32 SealBot + ~32 checkpoints (per eval, every 5 epochs) statistically MEANINGFUL? Give the honest answer
   (single-epoch resolution vs the rolling-pool asymptote; what the pooled SealBot/anchor curve can resolve over
   N evals). Do NOT overclaim per-epoch significance.
3. SealBot non-stationarity under in-run GPU load: keep overdispersion 0.5? adjust time_limit? Is SealBot even
   trustworthy enough in-run, or should it stay descriptive-only (it already is — down-weighted, never primary)?
4. Anchor: confirm whether enabling SealBot re-anchors the existing bc_prefit pool, whether ep10-30 stay on one
   comparable scale (graph connectivity via shared opponents), and whether to keep/reset/migrate the pool.
5. Apply WITHOUT interrupting the run: which files change (config + supervisor), how they take effect (next
   restart regenerates _resume_config from config; the running process keeps the old in-memory config until
   then), and the recommendation for WHEN to apply (e.g. at the next zero-loss epoch boundary, or let the next
   natural restart pick it up). Confirm NOTHING forces an interrupt now.`

const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['area', 'design', 'concrete_details', 'risks'],
  properties: {
    area: { type: 'string' },
    design: { type: 'string' },
    concrete_details: { type: 'array', items: { type: 'string' } },
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
  required: ['summary', 'allocation_final', 'significance_statement', 'execution_model', 'perf_estimate', 'config_changes', 'code_changes', 'supervisor_changes', 'anchor_pool_decision', 'apply_plan', 'test_plan', 'open_risks'],
  properties: {
    summary: { type: 'string' },
    allocation_final: { type: 'string' },
    significance_statement: { type: 'string' },
    execution_model: { type: 'string', description: 'how the eval runs FAST: concurrency model (all-opponents-at-once vs per-opponent), eval-vbs, which functions change' },
    perf_estimate: { type: 'string', description: 'wall-time estimate for the chosen design vs the current serial parts' },
    config_changes: { type: 'array', items: { type: 'string' } },
    code_changes: { type: 'array', items: { type: 'string' } },
    supervisor_changes: { type: 'array', items: { type: 'string' } },
    anchor_pool_decision: { type: 'string' },
    apply_plan: { type: 'string', description: 'exactly how/when it applies WITHOUT interrupting the run now' },
    test_plan: { type: 'array', items: { type: 'string' } },
    open_risks: { type: 'array', items: { type: 'string' } },
  },
}

const DESIGNERS = [
  { key: 'allocation-significance', area: 'allocation + significance', prompt: 'Design the 32-SealBot + 32-checkpoint allocation and JUSTIFY its statistical meaning. Read multistage_eval.allocate_budget + _stage_c_deep + eval_stats (wilson_ci, bradley_terry, effective_counts). Decide the mechanism for the 1:1 split (config knob sealbot_share/sealbot_games threaded into allocate_budget vs tuning games_budget) and the per-opponent split of the 32 checkpoint games (prioritise the L-5 champion + the two fixed anchors). State honestly what one eval resolves vs what the rolling pool resolves over many evals; do not overclaim per-epoch.' },
  { key: 'sealbot-anchor', area: 'sealbot mechanics + anchor/pool', prompt: 'Design enabling SealBot in the in-run eval: read eval_arena.play_sealbot_match, multistage_eval._play_sealbot_opponent/_choose_anchor, config MultiStageEvalSection sealbot_*, and scripts/_hexfield_supervise_main1.sh. Specify how SealBot is reached (SEALBOT_PATH=/mnt/e/SealBot in the supervisor and/or config sealbot_path; variant "current"), whether overdispersion/time_limit need adjusting for in-run GPU-load non-stationarity, and the ANCHOR consequence: does enabling SealBot re-anchor the bc_prefit pool, do ep10-30 (no sealbot edge) stay on one comparable scale via shared opponents, and should the pool be kept / reset / migrated? Give a concrete recommendation.' },
  { key: 'performance-concurrency', area: 'performance + concurrency (FAST eval)', prompt: 'Design the eval to FINISH IN A FEW MINUTES by running games CONCURRENTLY across ALL opponents (not the current serial --parts). Read multistage_eval.run_multistage_eval / run_multistage_eval_in_parts / _stage_c_deep / run_eval_part AND eval_arena.play_checkpoint_match (its batched multi-root round loop) + play_sealbot_match. Determine: (1) is the slowness the SERIAL per-opponent parts (sum-of-matches) vs a concurrent all-opponents pass (max-of-matches)? (2) Can all ~64 games (candidate vs every checkpoint opponent + SealBot) run in ONE concurrent batched pass — candidate moves batched across all games per round, each opponent net batched within its group, SealBot drained serially per game — reusing the existing round loop, or is a NEW multi-opponent runner required? Specify the concrete execution model + which functions change. (3) At FIXED 512 visits, quantify the gain from raising eval_virtual_batch_size (fewer rounds/ply = fewer CPU round-trips; the eval was CPU-round-bound at ~96% one core / GPU ~50%) and recommend a value. (4) Give a wall-time ESTIMATE for the chosen design vs the current serial parts. The pool/edge outputs and pentanomial/CRN pairing must be UNCHANGED (same statistics), only the execution made concurrent.' },
  { key: 'apply-no-interrupt', area: 'apply without interrupting the run', prompt: 'Trace exactly how the config takes effect: the running train_model holds ctx.config in memory from _resume_config.toml (loaded at process start); evaluate_epoch re-reads ctx.config each epoch but ctx.config does NOT re-read the toml mid-run; the supervisor regenerates _resume_config.toml from configs/hexfield_main_1.toml only on (re)launch. So a config edit affects the run ONLY after a restart. Design the apply plan that does NOT interrupt now: which files to edit, and WHEN the change should land (next natural supervisor relaunch, or a deliberate zero-loss restart at an epoch boundary later — note the next eval is ep35). Confirm no part of the redesign forces a restart now, and that editing the files now is safe (the running process ignores them until relaunch).' },
]
const CRITICS = [
  { key: 'stats-honesty', prompt: 'Critique for statistical honesty + power: is ~32 SealBot + ~32 checkpoints every 5 epochs genuinely meaningful, or still overclaimed? Is the SealBot down-weight (0.5) right? Does the 1:1 split starve the champion edge (the only verdict comparator)? Is the pooled-curve significance argument sound given SealBot is descriptive-only? Are pair-level/Wilson CIs used correctly? Flag any overclaim.' },
  { key: 'impl-correctness', prompt: 'Critique for implementation correctness AND performance feasibility by reading code: (1) if a sealbot_share/sealbot_games config knob is proposed, exactly where in allocate_budget + config.py + the runner _apply_overrides + parse it must be threaded (the runner hand-lists mse_overrides — a new field is silently dropped unless added); (2) does enabling sealbot in-run actually reach the binary (SEALBOT_PATH/variant resolution in eval_arena.play_sealbot_match), and is the supervisor env edit correct; (3) _choose_anchor behaviour when the pool mixes sealbot-anchored (ep35+) and bc_prefit-only (ep10-30) edges; (4) is the proposed CONCURRENT-across-opponents execution actually feasible with the current code (does play_checkpoint_match / the rust multi-root search support multiple distinct opponent nets in one batched pass, or is a new runner truly needed, or is the lowest-risk fast win just raising eval_virtual_batch_size + keeping per-opponent concurrency)? Judge effort vs payoff and flag if the concurrency rewrite risks correctness of the pentanomial/CRN pairing; (5) does the existing test suite stay green; (6) confirm editing config/supervisor now does NOT affect the running process. List concrete must-fixes.' },
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
  `Lead engineer: merge the designs + critiques into ONE implementation-ready spec for a statistically MEANINGFUL AND FAST in-run eval: ~32 vs SealBot + ~32 vs checkpoints, every 5 epochs, 512 visits, SealBot restored as the down-weighted cross-lineage anchor, and execution made CONCURRENT so it finishes in a FEW MINUTES (not the current serial per-opponent parts). Resolve every must_fix. Output: the EXECUTION MODEL (concurrency design + eval-vbs + which functions change) and a wall-time PERF ESTIMATE; exact config values; ordered file:function code edits (incl any sealbot_share/sealbot_games knob threaded through allocate_budget + config + runner, AND the concurrency change); the supervisor SEALBOT_PATH edit; the anchor/pool decision (re-anchor vs reset/migrate); an honest significance statement; a test plan; and an APPLY PLAN that does NOT interrupt the running trainer now (state when/how it lands). Prefer the LOWEST-RISK fast design that the critics judged feasible. Do not overclaim per-epoch significance.\n\n=== CONTEXT ===\n${CONTEXT}\n\n=== DESIGNS ===\n${digest}\n\n=== CRITIQUES ===\n${JSON.stringify(reviews, null, 1)}`,
  { label: 'synthesis', phase: 'Synthesis', schema: SPEC_SCHEMA }
)

return { designs, reviews, spec }
