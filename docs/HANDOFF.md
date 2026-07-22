# HANDOFF — Hexo solver program

Entry point for an orchestrator session. Technical state document: what
exists, what is measured, what is running, what the rules are. History lives
in git and `docs/archive/`. Updated: 2026-07-22.

## 1. Mission

Maximize *verified* WIN/LOSS coverage of the TSS solver on fixed position
sets — no shortcuts, every improvement proven correct. Strength axes: leaf
solving inside the trainer (main_4), deep/offline solving (atlas, labeling),
and beating the reference solvers (hexo-strix idtt/dfpn/pdspn). Method:
multi-lane Codex delegation (5.6-sol HIGH default, ultra for hardest
design/review/formalization; orchestrator gates every lane with independent
reruns, then commits). Research first, Lean formalization after a result
proves worth licensing. Adoption metric = verified coverage on fixed sets
via the harness; wall/yield are secondary diagnostics.

## 2. Game and search class

- Hexo: Connect6-variant on an **unbounded** hex board (sparse axial Z²;
  the engine's i16 coordinate carrier is encoding, not a rules bound).
  Two placements per turn (opening turn = one). Win = six-in-a-row along
  any of the three axes. D6 symmetry.
- Windows: the 6-consecutive-cell axis segments; the only win geometry.
- TSS solver class `vcf_pair_complete`: every attacker turn must force
  (create threats consuming the defender's whole turn). First stone ∈
  empties of live attacker count-≥2 windows ∪ empties of live defender
  count-≥4 windows; second stone ∈ S(P,a) = (T(P)\{a}) ∪ G₁(P,a) —
  a **proven-exact** characterization. Pair retained iff it creates a
  threat family, answers defender win-now threats, and pins the defender
  (hitting number 2, or no 2-cover ⇒ immediate win). Defender replies =
  minimal 2-cell hitting sets (≤4 pairs at rank-2 boundaries).
- Verdict semantics (keep strictly apart): **WIN/LOSS** = absolute,
  certificate replayed by the frozen verifier before any consumption;
  **width-exhaust** = proven no-forcing-class win (class-relative, NOT a
  game loss); **Unknown** = budget died, no claim.
- Lean (E:\tss-lean): kernel-checked accept⇒won for covered cert classes
  (CP10), census/deadline blocking family, zone theory FHW-T3-R.
- Known width boundary: 3 atlas-certified wins have no forcing-class
  expression. Mechanism identified: the free second stone after an
  already-forcing first stone falls outside S(P,a) — the `J2near`
  extension candidate (RESEARCH_DIVERGENCE_1.md §2).

## 3. Repos, branches, worktrees

| Where | What |
|---|---|
| `E:\Hexo-BotTrainer-hexgt` (main checkout) | detached HEAD, **old engine — never build or test from here** (its crate has ~68 tests; the real one has ~257) |
| branch `claude/main4-integration` (worktree `.claude/worktrees/consolidate-main`) | **production branch**: engine + P7 + all-leaves config + canonical docs (this file, SOLVER_NOTES.md) |
| lane worktrees `.claude/worktrees/<lane>` | one per Codex lane, forked from main4-integration (or research-div); brief = `CODEX_BRIEF.md`, log = `.codex-lane.log`, both deleted at gate |
| branch `claude/research-div` | RESEARCH_DIVERGENCE_1.md + scratch analyzers |
| `E:\tss-lean` | Lean repo (ledger/audit laws local to it; see its handoff) |
| `.claude/worktrees/opening-atlas` | live atlas site (LAN :47021) |
| `.claude/worktrees/resume-run-crash-fdef2b` | run scripts + built infra `.so` needed for trainer relaunch |
| `/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_4` | run data (WSL) |

Cargo (binding): always `cd` into the worktree in the same command; always
`--target x86_64-pc-windows-msvc` + worktree-local `CARGO_TARGET_DIR`;
suite = `cargo test -p hexfield_eq --features python -- --test-threads=1`
(parallel runs flake on env-var tests; plain `cargo test` silently skips
the python-gated `search`/`tree` modules). Expect ~257 tests; ~68 means
you built the wrong crate.

## 4. main_4 production line

**Status: STOPPED (owner-ordered 2026-07-21, stays off during the solver
drive; ran to ~ep25 with an eval). Not a crash; checkpoints intact.**

- Line: fresh-start run, weights-only warm start from main_3 ep90; cosine
  LR 2e-4→2e-5 over 150 epochs + floor. Branch `claude/main4-integration`.
- Solver config: mode 3 (WIN+LOSS), cap 750, `tss_solver_j2near=false`
  (owner reversal 2026-07-21 late — the brief cap-1000/J2near-on point
  was judged not worth the wall; J2near stays wired as a first-class
  default-off key, env path deleted — REPORT_J2NEAR_WIRE.md `80e3ea18`;
  cap 750 = measured strictly-better post-fold point per
  REPORT_J2NEAR_CAP.md: +89 decisions, 45.96 s < old 49.96 s wall; toml
  applies at relaunch, first-epoch gates still bind), 256 KiB TT,
  unbounded horizon, wide profile, `tss_solver_dual_pass=true`,
  `tss_solver_all_leaves=true` (strict solver-first: every leaf solved
  before GPU eval; park 5000 ms = emergency valve only), async 12/24
  workers, root+interior guards on. G2 / ordering hints / loss_reserve OFF.
- Launch/kill procedure: `docs/MAIN4_LAUNCH_PLAN.md`. Supervisor uses the
  INFRA_TREE split (infra packages resolve from resume-run-crash-fdef2b);
  WSL needs rustup cargo ≥1.95. Deterministic stop = kill SUPERVISOR pid
  first, then child; never remove the halt flag while the supervisor lives.
- First-epoch validation gates after any relaunch: `park_bailed==0`,
  `async_dropped==0`, `deep_verify_failed==0`, win/loss backup levels
  (loss stream must stay live), pace ≈ −10% vs pre-all-leaves bench.
- Reference marks: ~27% of training rows carry proofs; policy top-1 broke
  the main_3 plateau (55% → 66.8% by ep20); loss backups ≈ 3.3:1 vs wins
  pre-all-leaves; all-leaves bench = 3x WIN backups at −10% pace.
- ONE-cargo-host-wide law applies only while the trainer is up.

## 5. Measured cost structure (what to optimize and why)

- Candidate **generation dominates** (winning proof path ≈ 0.0003% of
  solve wall) — and is now 1.72x cheaper: candidate-gen rounds 1+2
  folded bit-identically (`2a1bdf97`; REPORT_CANDIDATE_GEN{,_R2}.md).
  Post-fold production shape (cap 500): attacker pair gen 48.7%,
  defender gen 15.6%, second-cand regen 5.7%, TT 0.46%, setup 0.19%,
  outside-expansion residual 32.9% (next unattributed block). Deep F19:
  attacker 36.0%, defender 30.6%. P6 memo BUILT (32,768 slots, 46.2%
  hits).
- TT at cap 500 ≈ overhead but cheap (hit/entry ≈ 0.01; probe/insert
  wall 0.43% — TT-min removal KILLED as not worth it). Deep memory
  resident = WidePnSearch arena + `by_position` index
  (admission-rejection only, no eviction) — TT replacement policy is
  irrelevant to deep solves.
- Grinds (quiet cap-bound Unknowns) = 73.5% of Unknown wall. Anatomy at
  50k: ~23% provable WINs (p50 ~1.7k nodes), ~39% width-exhaust
  (self-terminate ~2k), ~38% still cap-bound.
- Ordering is a dead family: miss cost measured 1.8%; two generic oracles
  rejected. Only proof-participation signals (e.g. probe-seeding) pass.
- Reference standing (2026-07-21 quiet-host pinned rerun; SOLVER_NOTES
  §5): 14/14 corpus WINs + 2 LOSSes certified, 0 failures, 416 s;
  0l4291i = WIN 1,879,612 nodes / 1.73 GB / ~274 s vs pdspn 1,058
  level-1 nodes / 260.9 s — **wall PARITY on 0l; the old "~1,800x
  informed-node gap" was node-accounting** (pdspn nodes each run bounded
  probes; probe-seed import KILLED at matched budget, `ea4170ca`). Real
  remaining gaps: 94gnnol-class disproofs (their No 20.7 s vs our
  cap-bound Unknown at 1M), idtt easy-win latency 2–7x (fresh-ladder tax
  + in-wall cert+verify), no certified refutations yet (v1 design GO —
  pending hostile review).
- **Corpus semantics law**: forcing/puzzle corpus measures search
  efficiency; human corpus (8,698 games) measures prevalence/economics.
  Never quote one for the other's claim.

## 6. Results index

**Proven & live:** exact attacker universe S(P,a); FHW-T3-R zones +
G2 verifier extension (sound, fires, not yet consumed); census blocking
family (DEADLINE_ES, FirstStone); CP10 accept⇒won (base/D17 × tree/DAG);
k*=3 tempo; dual-pass loss stream; P7 prefilters 1.42x bit-identical;
rank-two ≤4 defender cover pairs (model-checked; Lean in flight).

**Killed (with proof/measurement; see IMPOSSIBILITY_LEDGER.md on
tss-vcf-width + lane reports):** B2 tempo pruning (0 non-forcing pairs in
1.09M); 1+ε thresholds (both additive and multiplicative semantics);
work-weighted TT; parametric deadline ladder (board unbounded, D(P)=∞);
pointwise region deadline (open-carrier-ray impossibility, all 248
grinds); census gate under unbounded profile (needs h_rem≤8); generic
ordering oracles (policy priors, threat statics); guessed reply
equivalence (4 real counterexamples); r=2 trim; pairing; ES greedy;
Group-2 consumption at unforced AND nodes on current cohorts (prevalence
~0); warmth/fragments at cap 500 (0 verdict flips).

**Gated this drive (branch, commit, report):** truth-pass `8271f696`;
deep-imports `7c4c04f1`; harness-robust `41b0d23d`
(REPORT_HARNESS_ROBUST.md — SET-DEEP-V1 pin e3e52dc3, disproof metrics,
parallel sweep, contended bench); research-div `274aa3d3`
(RESEARCH_DIVERGENCE_1.md); deadline-ladder R `72f68ced`
(REPORT_DEADLINE_LADDER_R.md — NO-GO + reduced Lean program);
G2 consume design `fcea3c69` (DESIGN_G2_CONSUME.md); probe-seed KILL
`ea4170ca` (REPORT_PROBE_SEED.md — no matched-budget coverage gain,
prototype default-off); g2-hostile-review (g2-cert
`.gate/HOSTILE_REVIEW_G2_CONSUME.md` — SOUND-WITH-REQUIRED-CHANGES,
R1–R6, build NOT authorized); horizon-r2 (deadline-ladder worktree,
REPORT_HORIZON_R2.md uncommitted — exact full-game h≤8 deciders, 76/76
validation, theorem-blocked at h=10); candidate-gen `63b34cbb` folded as
`2a1bdf97` (REPORT_CANDIDATE_GEN{,_R2}.md — 1.72x bit-identical);
lean-shallow (tss-lean `f4315e6` — R-SH0..R-SH3 incl.
`forcedB2PairQuotient`); refute-cert v1 design (refute-design worktree,
DESIGN_REFUTE_CERT_V1.md uncommitted — GO conditional on v1 cut,
~2.5k LOC, hostile review = required next gate).

## 7. Live lanes (as of this writing) and build queue

Live lanes (2026-07-21 evening, third round):

- **refute-build** (claude/refute-build, worktree refute-build):
  RefuteLeafExact/V1 implementation. Authorization chain complete:
  design → hostile review R1–R8 (`d4af5aef`) → amendments `305eeeb8` →
  confirmation re-review `b8699ded` (CLEARED-WITH-REQUIRED-CHANGES) →
  R2-1/R2-2 amendments `a018befb` → narrow confirmation `af5665be`
  (CONFIRMED-CLEARED-FOR-BUILD). Full-tree is NO-GO; final adoption gate
  remains the owner's.
- **horizon-r3** (claude/deadline-ladder): legality-bridge lemma
  (within-8 rule), h≤10 universe shrink, next-rung closure (h13/14),
  h16 feasibility read. Research-primary.
- **lean-horizon** (E:\tss-lean): two-cover lemma, legality bridge,
  h6 decider formalization.
- **refute finisher** (claude/refute-build): implementing the R3-1
  corrected Q=4 golden (design `a24c1f00`; independent recompute of
  pinned bytes mandatory).

Exited and folded: **attacker-gen-r3** (`cc75b304` → merge `b1e7e877`):
1.159x end-to-end bit-identical, all gates incl. WSL Stage-0 34/34;
cumulative solve-wall speedup ≈ 2.0x since drive start. RELAUNCH NOTE:
gate uncovered + fixed a dangling hexo_utils editable install in
hexgt-build (pointed at deleted hexgtfeat; reinstalled from INFRA_TREE
resume-run-crash-fdef2b) — without the fix, main_4 relaunch would have
failed at import.

Exited and gated: **bottleneck-750** (`e08d9da0` on claude/j2near-profile,
REPORT_TSS_BOTTLENECK_750.md) — the 32.85% residual is state make/unmake
(24.10%) + PN bookkeeping (7.71%); UNKNOWN-at-cap rows eat 65.46% of
battery wall (early-stop ceiling 30 s); ranked attacks in SOLVER_NOTES §5.

Landed this session: candidate-gen 1.72x fold `2a1bdf97`; J2near
default-off fold `f5a5c5f0`; cap-750 retune `55f3c2b6` (superseded same
day); **cap 1000 + J2near-ON production wiring `80e3ea18`** (owner
completeness ruling: training target = most complete verified solution
set; wall relaxed); horizon R2+H10 research `e1180970` on
claude/deadline-ladder (h=10 frontier closed); lean-shallow tss-lean
`f4315e6`; triage Phase B `be4bd34a` (pocket detector only); G2 step-zero
screen `75f89cc8` (NOT KILL / STOP — orchestrator recommendation: drop
for production, ceiling 9.8% at the labeling point).

Standing owner rulings this session: completeness ruling above; horizon =
research primarily (no incremental-horizon ladders/consumption now);
certified refutations only at manageable scope (leaf-only cut is the
authorized shape); triage split = classification now (closed) /
re-allocation deferred; hostile reviews launch without waiting for owner.

Queue when slots free: next speed round from the j2near-profile
bottleneck ranking (wall recovered → cap headroom → completeness);
node re-allocation un-deferral proposal (pocket detector as seed);
sibling-certificate transplantation shadow; h6/h8 Lean per horizon-r2
(research track); CapResumeSession promotion (kills fresh-ladder tax,
most of the idtt latency gap); GPU bench close-out; integration round for
remaining lane branches (+ order-prior `a66b707a` park-sweep commit).

## 8. Laws

1. `tss_verify.rs` is frozen. Never weakened; finders conform to it.
   Extensions = owner-gated design + hostile review rounds.
2. Every engine change lands default-off with flag-off bit-identity:
   serialized suite + Stage-0 golden digest + (for default-flips or
   claimed-identical optimizations) the 6,443-position identity battery.
3. Adoption decisions go through the harness on frozen sets; canary per
   declared feature; bench identity echo-to-echo.
4. Shadow-measure with a pre-registered bar before building any
   theory-driven pruning (see Group-2/A1 and B2 precedents).
5. Corpus semantics law (§5).
6. RAM gates: ≥8 GB free before cargo builds; ≥10 GB before ≥512 MiB-TT
   batteries or Lean builds; heavy steps serialize.
7. ONE Lean build host-wide; ledger flips on kernel truth only; audit
   last before exit; manual decl-existence grep on newly-PROVEN rows.
8. Codex sessions never commit. Orchestrator gates (independent rerun of
   load-bearing claims) then commits path-scoped (`git add <paths>`,
   never `-a`). Commit trailer:
   `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
9. Lane hygiene: one worktree per lane; delete `CODEX_BRIEF.md` +
   `.codex-lane.log` at gate; remove worktrees after their branch folds.
10. Paper work deferred (standing owner ruling).

## 9. Document map

Operative (this branch, `docs/`): HANDOFF.md (this file); SOLVER_NOTES.md
(current measurements); TSS_SOLVER_SPEC.md + TSS_SOLVER_PROOF.md (engine
contract); PROOF_TSS_DEFENDER_ZONES.md (zone theory);
PLAN_TSS_SOLVER_UPGRADES.md (upgrade taxonomy); TSS_RUNBOOK.md (flags,
gates, build commands); PLAN_TSS_HARNESS.md + REPORT_HARNESS_ROBUST.md
(harness); PLAN_TSS_MCTS_INTEGRATION.md (§2 = binding trainer soundness
contract); MAIN4_LAUNCH_PLAN.md; STRIX_SOLVER_COMPARISON.md +
INVESTIGATION_PDSPN_IMPORTS.md (reference-engine round; items 2/3 closed,
1 in flight); REPORT_TRUTH_PASS.md, REPORT_DEEP_IMPORTS.md,
REPORT_DEADLINE_LADDER_R.md (lane evidence); ARCHITECTURE.md,
HEXFIELD_EQ_EXPLAINER.md, DERIVATION_D6_EQUIVARIANT_ATTENTION.md,
intro_to_hexo.md (background). On research-div: RESEARCH_DIVERGENCE_1.md.
On g2-cert: DESIGN_G2_CONSUME.md + `.gate/` reports.
Superseded material: `docs/archive/` (INDEX.md inside).

## 10. Pending owner decisions

1. Refute-cert v1: go/no-go on the hostile semantics/grammar review
   round (design verdict is GO conditional on the v1 cut; review is its
   required next gate).
2. G2: hostile review landed (SOUND-WITH-REQUIRED-CHANGES). Owner
   expressed interest in continuing; orchestrator assessment: deep/
   labeling lever only, NOT cap-500. Next step if continued = R1–R6
   amendments + fixed step-zero screen; build auth still not granted.
3. main_4 relaunch timing (currently: leave off).
4. `hunt-cert-support` worktree holds uncommitted crel6 patches —
   keep/commit/discard.
5. Three closed worktrees (group2-zones, hunt-completeness, hunt-gap-raw)
   + junk `claude/hello-*` branches await a permission-approved removal.
6. Triage Phase B (sub-root telemetry instrumentation for the Unknown
   classifier): pursue or park (phase-1 evidence: root-level signals
   insufficient).
