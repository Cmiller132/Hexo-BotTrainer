# PLAN_TSS_HARNESS — the designated solver comparison harness

Owner rulings (2026-07-20): the harness is the standing instrument for
comparing solver changes. Adoption metric = **verified win/loss coverage on
fixed position sets**. Default run ≤ 30 min, ideally 5–10, multithreaded,
**production-shaped** (persistent solver, real batching). Comparison model =
**archive + in-process A/B**. Puzzle/ground-truth set draws on ALL of:
forcing anchors, VCF fixtures, offline deep-solved labels, atlas certified
rows, human tactical picks. Hard to cheat; the harness itself is tested.
Multiple cargo lanes authorized (parallel work, Codex HIGH on build-out).

## 1. Architecture

Three layers:

1. **Rust support** (additive, verifier untouched):
   - `hexfield_eq_solver_manifest(...)` — effective-config echo derived from
     the SAME construction path real solves use (never re-hardcoded
     constants): width flags, tt_enabled + tt_bytes_cap, fragment store
     enabled + reconfigured cap bytes, lazy frontier, interior census gate,
     every env var the solver reads with its resolved value, cert version.
   - Per-solve `stats_*` emission from the ACTUAL solving (persistent)
     solver in the batch API — closes the §SOLVER_NOTES.3 instrumentation
     gap (today: batch emits no stats; probe stats come from a cold
     re-solve).
   - Production-shaped batch entry: persistent solver over a game/sequence,
     identical to the trainer leaf path caps (256 KiB TT, node cap arg).
2. **Python driver** `scripts/tss_harness/` — set loading, arm specs,
   process-parallel sharding by game, canaries, gates, archive, reports,
   paired diffs, A/B mode.
3. **Frozen sets + archive** — hash-pinned position sets; every run writes
   an archive directory; diffs run between any two archives.

## 2. Position sets (versioned, SHA-pinned; new version = new hash, never
   edit in place)

- `SET-SELFPLAY-V1` — the 3,255 frozen V1 selfplay positions.
- `SET-HUMAN-V1` — the 2,720 frozen V1 human-game positions.
- `SET-PUZZLE-V1` — the ground-truth hard-gate set, assembled from:
  (a) 190 forcing-corpus positions with anchors; (b) the 19 VCF-width
  fixtures; (c) offline deep-solved labels — big-cap (≥1M nodes, no park)
  offline solves that definitively label positions, INCLUDING the 248
  cap-bound grinds from V1 (ground truth exactly where production solvers
  struggle); (d) a stratified sample of atlas certified rows (early-game
  proven WIN/LOSS); (e) human-game tactical picks labeled by (c)'s
  protocol. Labeling protocol: verified cert or double-run deterministic
  Unknown at the big cap; labels carry provenance + the labeling solver's
  manifest.

## 3. Anti-cheat / anti-artifact design (each item traces to a real incident)

1. **Manifest assertion** (warmth env-gate incident): every arm declares
   intended features; the harness compares against the Rust manifest echo
   and ABORTS on mismatch. No silent "requested but not effective".
2. **Canaries** (warmth zero-counters incident): tiny fixtures where an
   enabled feature MUST visibly fire — warmth canary (second solve of a
   repeated-structure pair must cost fewer nodes than cold), TT canary,
   census-gate canary (a position inside the h_rem ≤ 8 window), horizon
   canary (a win at depth > h that unbounded finds and bounded must not).
   Canary not firing under a config that claims the feature = run FAILURE.
   Inverted canaries too: feature OFF must make the canary NOT fire (guards
   against always-on bleed).
3. **Ground-truth hard gates**: on SET-PUZZLE, losing a known win/loss or
   claiming a verdict contradicting a label FAILS the run. Everywhere:
   `deep_verify_failed == 0` (fatal), cert_version pinned.
4. **Determinism shard** (solver is deterministic given state+caps+cache):
   one shard re-solved twice per run; any bit difference FAILS the run.
5. **Nodes over wall** (warmth wall-confound incident): node counts and
   verdicts are the primary, load-independent currency; wall is reported
   with a machine-load fingerprint (core count, concurrent GPU/CPU jobs
   noted) and never gates anything by itself.
6. **Schema + environment fingerprint**: every archive records harness
   version, set hashes, wheel build hash, solver git rev, python/rustc
   versions, full env snapshot of TSS_* vars.
7. **The harness is tested**: a self-test suite where each gate is
   deliberately violated (feature forced off against a claiming manifest,
   a corrupted label, a nondeterministic stub) and the harness must FAIL
   each one. A gate that cannot fail is not a gate.

## 4. Run tiers

- `quick` (~2–4 min): canaries + manifest + determinism shard + stratified
  ~15% sample of all three sets. For mid-development iteration.
- `standard` (target 5–10 min, ≤30 hard): full three-set sweep, parallel
  workers (game-sharded, semantics-preserving), full report + diff vs
  chosen baseline. THE default for "made a change, run it".
- `full` (~45+ min): adds warmth arms, both-goal arms, second determinism
  pass, TT-off control. For adoption decisions.

## 5. Comparison model

- **Archive-first**: `harness_runs/<ts>_<label>/` with raws, manifests,
  fingerprint, report.json. `compare A B` produces the paired diff:
  upgrade list (Unknown→WIN/LOSS), downgrade list, verdict churn (both
  directions visible, not netted), node deltas, per-set and per-stratum
  coverage tables, gate status. Cross-build comparisons are first-class.
- **In-process A/B**: two arms, same build, interleaved over the same
  shards (same-process load symmetry) for config-flag changes.

## 6. Report contents (per run and per diff)

Coverage: WIN/LOSS counts per set and per stratum (band, placements, hot,
source). Soundness: verify counter, ground-truth gate results. Economics:
nodes-per-verdict, node distributions, cap-bound share, wall p50/p90/p99
(fingerprinted). Geometry: cert-depth distribution. Deltas: full paired
lists, not aggregates. Canary and determinism status up top.

## 7. Build plan (lanes)

- **Lane A (Codex HIGH, cargo)**: Rust support layer per §1.1. Hard
  constraints: `tss_verify.rs` UNTOUCHED; additive API only; existing tests
  pass unchanged (verdict-identical behavior); targeted new tests: manifest
  reflects env flips, warm batch shows fragment stats > 0 on the
  repeated-structure fixture, cold shows 0. Build discipline: limited
  parallelism (shared host with live GPU eval), no venv installs from the
  lane.
- **Lane B (orchestrator, Python)**: driver + canaries + gates + archive +
  diff + self-test suite, against the existing API first (behavioral
  detection), swapping to manifest/stats when Lane A lands.
- **Lane C (offline, after V2 frees CPU)**: ground-truth labeling runs for
  SET-PUZZLE-V1 (big-cap solves of grinds, atlas sample extraction, human
  tactical picks).

Resource rule for parallel cargo (owner authorized multiple lanes
2026-07-20): separate worktrees/target dirs always; heavy compile/link
steps stagger behind a free-RAM check (≥10 GB) while the GPU eval runs.
