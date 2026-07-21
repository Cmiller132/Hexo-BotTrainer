# SOLVER_NOTES — current measured state of the TSS solver

Reset 2026-07-22 (full history: `docs/archive/SOLVER_NOTES_2026-07-21_full.md`
and git). Convention going forward: append dated entries per finding; labels
MEASURED (verified data) / CODE-FACT (cite lines) / HYPOTHESIS / RETRACTED
(strike, don't delete, from this reset onward). Entry point and laws:
`docs/HANDOFF.md`.

## 1. Production profile (main_4 line; trainer currently stopped)

`tss_solver_mode=3` (WIN+LOSS), node cap 500, 256 KiB TT, unbounded horizon
(`semantic_horizon=u32::MAX`), wide (`vcf_pair_complete` via
`configure_leaf_profile`: wide + lazy frontier + interior census gate [inert
when unbounded]), `dual_pass=true`, `all_leaves=true` (park 5000 ms
emergency-only), async 12/24, root+interior guards. G2 / ordering hints /
loss_reserve / fragments / zones OFF.

## 2. Measured facts the current work stands on

- **Generation dominates wall.** Winning proof path ≈ 0.0003%. Production
  shape: A_OR_GEN 60.5%, D_FORCED_GEN 20.4%, TT+verify <1%. Deep F19: 82%
  generation (attacker 45.9% + forced defender 36.0%). Post-P7 residuals:
  first-candidate enum ~15%, D_FORCED_GEN ~20%, second_candidates churn
  ~8%. P7 prefilters: 1.42x bit-identical (`2c262e10`).
- **TT at cap 500 ≈ overhead** (hit/entry ≈ 0.01). Deep memory resident =
  WidePnSearch arena + `by_position` (no eviction; admission-rejection
  only) — TT replacement policy cannot affect deep solves (`7c4c04f1`).
- **Grind class** = 73.5% of Unknown wall; at 50k: ~23% provable WIN
  (p50 ~1.7k nodes), ~39% width-exhaust (~2k self-terminate), ~38%
  cap-bound.
- **Loss side**: dual_pass (leftover-budget second pass) = +288 dev
  losses, throughput-neutral; 58 atlas losses need ≥512 dedicated nodes —
  unreachable at cap 500 under any allocation.
- **Ordering family dead**: miss cost 1.8%; ep90 policy priors and threat
  statics both measurably harm df-pn. Bar = proof-participation signals
  only (probe-seeding qualifies; lane in flight).
- **Horizon closed**: unbounded+cap beats h16 on coverage (+26%) and
  strength (+60 Elo h2h); extra wins sit at cert depth 17–22.
- **Reference standing** (quiet 2 GiB gate + matched host): 14/14 corpus
  WINs + 2 LOSSes certified, 0 false claims; 0l4291i = WIN at 512 MiB /
  1,913,955 nodes (the old "memory ceiling" was a 1M rung-cap artifact)
  vs pdspn 256 MB / 1,058 seeded nodes. Real gaps: informed-node
  efficiency (~1,800x on 0l), easy-win latency (~2x), no certified
  refutations (planned).
- **Width boundary**: 3 atlas wins provably outside `vcf_pair_complete`;
  mechanism = free second stone after a forcing first stone (J2near
  candidate, ~1.04x accepted-child multiplier on eligible roots,
  0/248 grinds eligible; A/B mandatory).
- **Rank-two defender boundary**: threat families rank ≤2 ⇒ ≤4 minimum
  cover pairs (exhaustive 33,861-family model check + 229 real pairs,
  0 violations). Stateless plan construction = the D_FORCED_GEN lever.
- **Shallow exact layer**: exact h=2 (win-in-own-turn) and h≤4
  (unanswerable-family forced loss) predicates match engine diagnostics
  on 6,294 roots, 0 mismatches; h2 fires 102 / h4 fires 146 there. Lean
  formalization in flight; h≤6/h≤8 extension lane in flight.
- **Board is unbounded** (sparse axial Z²) ⇒ no global fill deadline
  exists; the parametric deadline-dismissal ladder is NO-GO (`72f68ced`).

## 3. Traps (verified the hard way)

- Cargo from the wrong cwd resolves the main checkout's old crate
  (~68 tests) — always `cd` into the worktree in the same command; expect
  ~257 tests (python feature) / ~172 (plain); serialize (`--test-threads=1`).
- Env-gated features (`TSS_SHARED_FRAGMENTS`, `TSS_CAP_RESUME`,
  `TSS_THRESHOLD_DELTA`, `TSS_TT_REPLACEMENT`, `TSS_K_REPLY_CONSUME`,
  census gate) follow the process environment — every arm must enumerate
  and assert its env gates (harness does this since `41b0d23d`).
- Probe/stats paths that construct fresh solvers describe COLD state by
  construction; batch APIs may omit stats keys entirely — check emission
  before trusting a zero.
- Bench must record the engine binary sha; debug-vs-release .so confusion
  invalidated a round once.
- The `:5979` terminal-Refuted arm is UNREACHABLE via normal producers;
  a debug_assert tripwire + regression test guard it (`8271f696`).

## 4. Open probes / in-flight (details: HANDOFF §7)

probe-seed (PN² init A/B) · candidate-gen (bit-identical generation
rungs) · horizon-r2 (exact h≤6/h≤8) · lean-shallow (h2/h4/rank-two
endpoints) · g2-hostile-review (consume-mode design attack). Queue:
J2near A/B, sibling-transplant shadow, GPU bench close-out, integration
fold.

## 5. Iteration log (from this reset)

- 2026-07-22: document reset to current-facts form (owner-directed
  de-bloat); full history archived.
