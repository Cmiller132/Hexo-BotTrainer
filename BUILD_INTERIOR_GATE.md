# R-IG1 interior census gate integration memo

Status: **IMPLEMENTED AND LIVE-GATED PASS**. This memo was created as the
pre-implementation integration map, before production code was edited, and
was updated after the completed campaign with exact final code locations and
results. The governing proof is
`PROOF_DTW_CENSUS_BOUND.md`, Contract 8.1/8.2, reviewed at
`ffdd414ad5197444eef44af4f28da376a5d95507`. The NQ6 measurement contract is
`HUNT_REPORT_PN_INIT.md` in this worktree.

## Attempt split and claimant meaning

`TssSolver::solve_goal` splits the requested root goal before entering either
search backend (`packages/hexfield_eq/rust/src/tss_solver.rs:606-734`). Its
primal attempt fixes `claimant = root_player`; its dual attempt fixes
`claimant = root_player.other()`. Each call to `prove_for` therefore asks one
bounded WIN question for that fixed claimant. The result is translated to the
root-facing `Win` or `Loss` only after a certificate is returned. The census
gate must never skip the unsplit `SolveGoal::Both` call or manufacture a root
`Loss`; it can only make one interior claimant proof query return the same
no-certificate result that ordinary restricted-search failure returns.

The environment flag is read once at the start of `solve_goal` (`:615-617`) and the
resulting boolean threaded through both possible attempts. Flag-off therefore
adds only a predictable boolean branch at the two claimant dispatch seams and
does not read the environment in the search hot path.

## Wide proof-number path

`prove_for` selects the wide PN backend at
`packages/hexfield_eq/rust/src/tss_solver.rs:760-798`.
`WidePnSearch::work` recursively reaches every linked position and calls
`expand` for every unexpanded entry (`:2211-2503`). All one- and two-placement
child links re-enter this same `work`/`expand` funnel; there is
no separate recursive WIN query hidden in the move generators.

The one eligible wide query point is `WidePnSearch::expand`, after all of:

- the staged structural depth cutoff and absolute semantic-horizon checks;
- terminal-state rejection; and
- the existing tactical/lambda winner check;

and immediately before the ownership dispatch
`state.current_player() == self.claimant` chooses a `WidePnKind::Choice`
(`packages/hexfield_eq/rust/src/tss_solver.rs:2967-3059`). At that point:

- `claimant` is the fixed attacker for this attempt;
- equality of `state.current_player()` and `claimant` proves this is the
  existential claimant WIN arm, not a defender/universal arm;
- `state.phase()` is exact and will be accepted only for `FirstStone` or
  `SecondStone`;
- the node is known nonterminal and the complete census scan can reject any
  inconsistent on-board six; and
- `h_rem` is derived by checked-wide subtraction of
  `self.semantic_horizon - state.placements_made()`. The former is copied
  directly from `SolveCaps::semantic_horizon` in `prove_for_wide_pn`
  (`:801-824`, `:1768-1796`); the latter is the exact state placement clock.

The implementation additionally requires
`state.placements_made() > self.root_ply`, so depth-zero roots are not gated.
This is the requested interior-only scope. A dismissal sets this wide entry to
the existing `Refuted` bounded-search state and refreshes its PN/DN. It does
not create a certificate, a global LOSS value, or any cache entry stronger
than the entry's existing within-attempt refutation. Because the depth-cutoff
test remains before the gate, staged deepening still reopens cutoffs normally;
only an actually reached eligible stage may dismiss the entry.

`attack_single_children`'s completion/lambda checks (`:3161-3236`),
`position_prior`, and `WideProofMaterializer` are not additional query points:
they respectively classify a move already being generated, initialize visit
order, and materialize an already proven PN entry. No gate belongs there.

## Narrow compatibility path

`prove_for` selects `WidePnSearch::prove_narrow_compat` at
`packages/hexfield_eq/rust/src/tss_solver.rs:760-785`, which constructs one
`NarrowCompatSearch` with exact `root_ply` and `semantic_horizon`
(`:1862-1932`). Every recursive child from both `prove_choice` and
`prove_universal` calls back through `NarrowCompatSearch::prove`
(`:3896-4324`). Thus `prove` is the single narrow query funnel.

The one eligible narrow query point is in `NarrowCompatSearch::prove`, after:

- checked derivation of structural depth from the absolute `ply` clock;
- semantic-horizon, depth, and node-cap checks;
- solve-local/shared TT lookups;
- terminal rejection; and
- the existing tactical/lambda winner check;

but immediately before the ownership dispatch to `prove_choice` versus
`prove_universal` (`packages/hexfield_eq/rust/src/tss_solver.rs:3896-4009`).
Only the `state.current_player() == claimant` / `prove_choice` side is eligible.
The universal side is never evaluated by the gate.

Here exact `h_rem` is checked-wide
`self.semantic_horizon - state.placements_made()`. In absolute-clock
production construction, `prove` already asserts
`state.placements_made() == ply` (`:3903-3906`), and both fields originate
from the same root/cap values at `:1880-1895`. As on the wide path, the gate
also requires `state.placements_made() > self.root_ply`.

A narrow dismissal flows through the existing `node == None` tail. Therefore,
when `pair.is_none()`, pair-complete mode, and no resource limit was hit, it
stores the same `LOCAL_TT_FAILED` sentinel that an exhaustively failed
restricted claimant generator stores (`:4000-4005`). With an active pair
context or any hit limit it stores nothing, also matching existing behavior.
It never enters the shared positive-proof cache.

## Contract checks at both placements

The shared predicate is at
`packages/hexfield_eq/rust/src/tss_solver.rs:111-214`. It runs only when the
per-solve flag is true and all of
the following hold:

1. the node is interior, nonterminal, claimant-owned, and in `FirstStone` or
   `SecondStone`;
2. checked `i64` subtraction produces `0 <= h_rem <= 8`;
3. the proof's checked-wide coordinate guard succeeds for that exact
   `h_rem` (`R = 8 * (h_rem + 1)`, `SAFE = 16383`, checked `q,r,s` and checked
   absolute values);
4. a complete `state.board().windows().entries()` scan computes the maximum
   `ac` satisfying `ac > 0 && dc == 0`, starting from zero, and rejects any
   entry with either player count above five;
5. the exact phase formula is used: FirstStone adds the proved gap only for
   `c <= 3`, SecondStone only for `c <= 2`; the matching tables are
   `[1,2,5,6,9,10]` and `[1,4,5,8,9,12]`; and
6. dismissal uses only strict `LB_plies > h_rem`.

The helper returns only a boolean bounded-WIN dismissal. It mints no theorem
artifact and exposes no result that could be interpreted as global LOSS.

## Locations deliberately not gated

- The pre-attempt root `immediate_winner` seam in `solve_goal` (`:642-670`)
  is a factual/lambda shortcut, not an interior recursive query.
- Opening nodes have no phase map and are rejected.
- All wide `WidePnKind::Universal` and narrow `prove_universal` arms are
  defender obligations and are never evaluated.
- Move-generator tactical classifications, PN priors, TT proof imports, and
  certificate materialization do not initiate a new bounded claimant query.
- Any node whose phase or checked absolute horizon cannot be derived exactly,
  whose coordinate guard fails, or whose complete census violates the
  nonterminal `c <= 5` invariant is not gated.

## Live campaign result

All values below are cold per-root sums from the dedicated live harness. “TT
entries” is the sum of retained exact-key entries at each solve's end. Gate
time covers the complete census scan and bound calculation; wall time also
includes the cheap scope and coordinate guards. Timing is a same-host A/B and
is not a portable benchmark.

| cohort | nodes off -> on | expansions off -> on | TT entries off -> on | tt_hits off -> on | wall ms off -> on | evaluations / dismissals / gate ms |
|---|---:|---:|---:|---:|---:|---:|
| forcing 10k | 89,424 -> 18,928 (-78.83%) | 89,405 -> 18,909 (-78.85%) | 148,308 -> 37,478 (-74.73%) | 25,168 -> 10,540 | 7,808.625 -> 1,133.453 (-85.48%) | 7,553 / 4,980 / 6.708 |
| forcing 100k | 324,182 -> 21,321 (-93.42%) | 324,163 -> 21,302 (-93.43%) | 497,219 -> 41,490 (-91.66%) | 122,790 -> 12,535 | 28,032.003 -> 1,250.028 (-95.54%) | 8,371 / 5,706 / 7.327 |
| double_fork_compact | 409 -> 409 (0%) | 408 -> 408 (0%) | 258 -> 258 (0%) | 51 -> 51 | 33.752 -> 34.126 (+1.11%) | 170 / 0 / 0.138 |
| human 100, 10k | 79,070 -> 46,519 (-41.17%) | 78,970 -> 46,419 (-41.22%) | 203,444 -> 124,702 (-38.70%) | 67,512 -> 48,837 | 10,534.951 -> 5,545.502 (-47.36%) | 8,682 / 5,988 / 23.125 |

The measured scan/bound cost per evaluation was 0.888 us, 0.875 us, 0.812 us,
and 2.664 us in table order.

### Live versus NQ6 trace estimate

| cohort | NQ6 trace-subtree estimate | live expansion saving |
|---|---:|---:|
| forcing 10k | 82.6% | 78.85% |
| forcing 100k | 88.0% | 93.43% |
| double_fork_compact | 0% | 0% |
| human 100, 10k | 53.1% | 41.22% |

The live result is not constrained to equal the trace replay: transposition
reachability and changed PN/DN values alter the generated frontier. The 100k
live saving exceeded the trace replay, while forcing 10k and human were lower.

## Identity, soundness, and official gate evidence

- Flag-off NQ6 identity was exact:
  `PNI_IDENTITY id=0hz3hty status=UNKNOWN nodes=9302 tt_hits=2872 expansions=9301 result=PASS`.
- All frozen flag-off cohort counts matched `HUNT_REPORT_PN_INIT.md` exactly:
  89,405 / 324,163 / 408 / 78,970 expansions, verdict splits 3/16,
  3/16, 1/0, and 23/77, and the same eligible/gated/subtree counts.
- The clean live A/B compared 139 keyed rows (19 forcing roots at each of two
  caps, compact, and 100 human roots): zero verdict differences. Both runs
  independently verifier-accepted every returned certificate; 60 certificates
  were accepted across the two runs. No forcing-NO row returned WIN and no
  WIN-only solve returned LOSS.
- The flag-on official command used a 2 GiB TT and began with 12.970 GB free.
  It finished in 443.17 s with `CORPUS_DONE failures=0` and `1 passed; 0
  failed`. Its semantic horizon is `u32::MAX`, so Contract 8.2 correctly made
  this an enablement/no-regression gate with zero census evaluations.
- The complete non-ignored release library suite passed: 97 passed, 0 failed,
  20 ignored. The focused phase-table and reachable SecondStone-c=3 strict
  boundary tests also passed.

Raw logs are in `.codex-hunt/interior-gate-pni-off.log`,
`.codex-hunt/interior-gate-live-off.log`,
`.codex-hunt/interior-gate-live-on.log`, and
`.codex-hunt/interior-gate-official.log`.

## Regeneration

Run from this worktree root. Check RAM immediately before every Cargo command;
wait below 9 GB and require more than 11 GB for the official 2 GiB run.

```powershell
$env:CARGO_TARGET_DIR='.target-hunt'

# Frozen flag-off identity/shadow comparison.
Remove-Item Env:TSS_INTERIOR_CENSUS_GATE -ErrorAction SilentlyContinue
cargo test --release -p hexfield_eq pn_init_campaign -- `
    --ignored --test-threads=1 --nocapture

# Clean live A/B (run once with the flag absent, once with it set to 1).
cargo test --release -p hexfield_eq interior_gate_live_campaign -- `
    --ignored --test-threads=1 --nocapture
$env:TSS_INTERIOR_CENSUS_GATE='1'
cargo test --release -p hexfield_eq interior_gate_live_campaign -- `
    --ignored --test-threads=1 --nocapture

# Official enablement gate.
$env:TSS_BACKWALK_TT_BYTES='2147483648'
cargo test --release -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture

rustfmt --edition 2021 --check `
    packages/hexfield_eq/rust/src/tss_core.rs `
    packages/hexfield_eq/rust/src/tss_solver.rs `
    packages/hexfield_eq/rust/src/tss_pn_init_hunt.rs `
    packages/hexfield_eq/rust/src/tss_corpus.rs
```
