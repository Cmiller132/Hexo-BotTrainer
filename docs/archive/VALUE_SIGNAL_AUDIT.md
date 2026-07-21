# VALUE_SIGNAL_AUDIT — WIN/LOSS symmetry of the TSS consumption path

Lane: `claude/consolidate-main` (integration mainline + V0 campaign engine).
Scope (owner ruling): the **consumption path only** — leaf hook, root/interior
guards, backprop, training-target emission, telemetry. No MCTS redesign, no
direct training-target swapping (Lever-2 banned), verifier untouched. Soundness
contract v3.1 §2 binding.

Motivation: the incoming dual-pass leaf change (`claude/twopass-leaf`, not
merged here) multiplies certified LOSSes ~2.6x and at production the loss stream
outnumbers wins (bench window: 5,337 L vs 4,714 W). The consumption machinery
was built and tuned in a win-dominated regime. This audit asks, end to end,
whether a proven LOSS is consumed with the same fidelity as a proven WIN so the
new signal is not silently wasted.

All line cites are against the tree state of this worktree at audit time
(function-name anchors given alongside line numbers because small edits shift
absolute lines). Cites use `packages/hexfield_eq/rust/src/…` and
`packages/hexfield_eq/python/hexfield_eq/…` unless noted.

---

## 1. Symmetry table

| Mechanism | Proven WIN | Proven LOSS | Symmetric? |
|---|---|---|---|
| **Value seam** (`HardValue`, `ProofStatus::value`) — tss_core.rs:38-44, 56-67 | `+1` (side-to-move) | `-1` (side-to-move) | **YES** — one sealed ±1 mint; `HardValue::status()` reads sign back verbatim. |
| **λ¹ leaf verdict** `solve_leaf_lambda1` — tss_core.rs:75-77; consumed search.rs `select_pending_leaves`/`select_continuous_leaves` (~2345-2354, 2487-2496) | backup `hard.value()`=+1 for `leaf_player`, GPU eval elided | backup `hard.value()`=-1 for `leaf_player`, GPU eval elided | **YES** — identical code path; sign carried by the mint, perspective is `leaf_player` for both. |
| **Deep leaf backup** (`tss_deep_leaf`, async descent-stop, park) — search.rs ~2333-2338, 2401-2407, 2475-2508 | `backup_virtual(path, leaf_player, hard.value(), …)` | same call, same args | **YES** — one `backup_virtual` call, no branch on sign. |
| **Leaf-solve gate** `has_threats()` — tree.rs:1343,1459; engine `tactics.rs:406-412` | fires on own ≥4 window | fires on opponent ≥4 window (`live_threats` covers **either player**) | **YES** — a loss-danger position (opponent threats, we have none) still enters the solver. Not a filter against losses. |
| **Consume tier gate** `tss_consume_gate` — tree.rs (~1583) | consumed at `mode >= 3` | consumed at `mode >= 2` | **YES at production (mode 3 = Both, both consume).** Mode 2 is a LOSS-only tier — the machinery already *prioritises* loss below full consumption. |
| **Backprop fidelity** `backup_virtual` | exact ±1 into every ancestor visit; no averaging distinct from loss | exact ±1, same routine | **YES** — no sign-dependent attenuation. |
| **Counters — proven** `deep_win`/`deep_loss` — tree.rs:800-801 (inside `tss_solve_verified`) | `deep_win += 1` on every verified WIN, **all modes** | `deep_loss += 1` on every verified LOSS, **all modes** | **YES** — proven stream fully visible even in shadow. |
| **Counters — consumed** `deep_hard_backups` — tree.rs `tss_consume_gate` | counted (combined) | counted (combined) | **WAS ASYMMETRIC IN VISIBILITY → FIXED.** Combined counter hid whether the loss half reached backup. Split added (§3, Fix A). |
| **Root guard** — search.rs `build_continuous_payload` (~4131-4152) | forces the winning move (`deep_forced_move`, `deep_override`), inserts class `+1`, appends the move to policy/π' export support at weight 0 with `Q=1.0`, sets `deep_root_proof=+1` | sets `deep_root_proof=-1` **only** | **ASYMMETRIC — DESIGN-JUSTIFIED (flagged, §4-A).** A proven root LOSS means *every* legal move loses (dual cert exhausts our moves); there is no single move to force, promote, or steer toward. The scalar is recorded symmetrically into `tss_proof` (search.rs ~4260) and reaches training identically to the WIN scalar. |
| **Root move classifier** `classify_root_move` — search.rs:4629-4653 | child is our win → `+1` | child is our loss (opp λ¹ win) → `-1` | **YES** — per-move classes are minted symmetrically for both signs. |
| **Play-time tactical guard** `tactical_guard_weights_from` — search.rs:4659-4683 | any `+1` present → zero all non-winning moves (steer **toward** wins) | else any `-1` present → zero the losing moves (steer **away** from losses) | **YES** — structural mirror; loss verdicts ARE consumed as policy steering. All-zero result falls back to raw weights (never zero the only move). |
| **Training value target** `finalize_game_samples` — samples.py:207-208 | `hard_z = +1` from game outcome, row-player perspective | `hard_z = -1` | **YES** — outcome-derived, sign-symmetric. (Lever-2 proof-corrected values remain unbuilt/banned; neither sign is pinned via Lever-2 yet, so no asymmetry there.) |
| **Trainer-side policy mask/sharpen** `_apply_class_mask` / `_sharpen_target` — selfplay.py:100-141 | winners exist → keep only winners; zero-mass-winner **recovery** spreads mass onto proven-winning move outside visit support (126-135) | else zero proven-losing moves; if all support is losing → fall back to raw target (no recovery) | **STRUCTURALLY SYMMETRIC; WIN-only recovery is DESIGN-JUSTIFIED (flagged, §4-B).** The recovery exists because a root-guard WIN identifies a *known better move* riding the support at weight 0; an all-losing support has no identified better move to spread onto. |
| **Driver telemetry** `ContinuousDriver` — selfplay.py:494-515 | `tss_win_rows`, `tss_win_retained`, `tss_gumbel_win_retained` (retained-mass magnitudes) | `tss_loss_only_rows` (count only) | **ASYMMETRIC — minor telemetry (flagged, §4-C).** Loss rows are counted but the guard's removed-mass magnitude is not measured. |

---

## 2. The upstream driver (not a consumption-path bug; noted for completeness)

`SolveGoal::Both` under the wide `vcf_pair_complete` **leaf profile** gives the
WIN attempt the FULL node budget and the dedicated LOSS attempt ZERO —
tss_solver.rs:1010 (`SolveGoal::Both if self.width.vcf_pair_complete => (remaining, 0)`).
The narrow profile splits 50/50 (line 1011); production uses the wide profile.
Measured (docs/SOLVER_NOTES.md §6 P4, `claude/order-prior`): production mode-3
Both surfaces only the losses the primal win search proves incidentally —
**~64% of provable losses are budget-starved away** (human n=338: Both 15 vs
loss-goal 42; selfplay: Both 0 vs 7). `goal=win` additionally FILTERS loss facts
at the root by construction (`solve_goal_filters_root_facts`,
tss_solver.rs:14114). **In production this filter is NOT reached**: both the leaf
channel at mode 3 and the root guard (search.rs, `SolveGoal::Both`) use the Both
goal, not Win.

This starvation is the FINDER's, and its fix is the incoming
`claude/twopass-leaf` branch — explicitly out of this lane's scope. The audit's
job is to confirm the consumption path is *ready* to absorb the ~2.6x loss
surge that branch delivers. **It is:** at mode 3 the leaf consume gate takes
LOSS and WIN symmetrically, backprop is sign-identical, the value target is
sign-symmetric via `hard_z`, and the guard steers away from proven-losing moves
symmetrically. The one gap the surge would expose is *visibility* of the
consumed-loss stream — fixed below.

---

## 3. Fixes implemented (pure telemetry — no behavior change; play/targets bit-identical)

### Fix A — split the consumed hard-backup stream by outcome
`deep_hard_backups` (bumped only in `tss_consume_gate`, the single chokepoint
for inline, async memo-hit, descent-stop, and park consumption) counted WIN and
LOSS backups combined. In a loss-heavy regime a run could see the *proven* loss
stream (`deep_loss`) but not whether those losses actually reached backup vs
wins. Added `deep_win_backups` / `deep_loss_backups` (their sum == the existing
`deep_hard_backups`).

- Rust: `TssCounters` fields + `add()` (tree.rs); split bump in
  `tss_consume_gate` keyed on `status`; emitted in the payload `tss` dict
  (search.rs).
- Python: aggregated in `ContinuousDriver` per-move loop and epoch-merge
  `int_keys`, emitted in the driver summary (selfplay.py). All reads are
  `.get(key, 0)`, so the pure-Python side is backward-compatible with an
  un-rebuilt extension (reads 0).
- Never consulted for any search decision (`TssCounters` doc). Play and targets
  are bit-identical; no flag required for a read-only counter.

Test: `tree::tests::consumed_hard_backups_split_by_outcome` — a verified WIN
consumed at mode 3 bumps only `deep_win_backups`; a verified LOSS (canonical
forced-loss fixture, dedicated LOSS goal) bumps only `deep_loss_backups`; the
sum invariant holds. Both go through the one gate.

---

## 4. Flagged (asymmetric but deliberate / out-of-scope — NOT changed)

**A. Root-guard LOSS records a scalar, not a steer.** A proven root LOSS has no
single move to force or de-weight (all moves lose; the dual cert exhausts our
legal set). The WIN path's move-forcing has no sound mirror. The value fact is
recorded symmetrically via `tss_proof=-1`. Changing this would require inventing
a per-move loss target the certificate does not designate — contract-unsound.
Leave as-is.

**B. Sharpen zero-mass recovery is WIN-only** (selfplay.py:126-135). Justified
by information availability: a root-guard WIN designates a specific better move
riding the support at weight 0, so mass can be spread onto it; an all-losing
support designates no better move. No sound loss analog exists in the
consumption path. Leave as-is. (A future position-level "forced loss within h"
label — the §7 tactical-class-head candidate — is the right home for a
symmetric loss signal, but that head is unbuilt and out of scope.)

**C. Driver loss-row telemetry is count-only** (`tss_loss_only_rows`), while win
rows also measure retained mass. This is a candidate symmetric telemetry
addition (a `tss_loss_removed_mass` analog), but it touches the retained-mass
merge/`_weighted_mean` plumbing and is lower value than Fix A (the guard's loss
steering is already visible through the sharpen path and the new
`deep_loss_backups`). Flagged for a follow-up; not implemented to keep this
lane's change surface minimal and purely additive.

**D. `SolveGoal::Both` wide-profile loss starvation** (tss_solver.rs:1010) — the
primary asymmetry, but it is the FINDER's, and its fix is the incoming
`claude/twopass-leaf` branch. Out of scope here (§2).

---

## 5. Answers to the audit questions

- **Leaf:** WIN pins `+1`, LOSS pins `-1`, both for `leaf_player`, via one sealed
  mint and one `backup_virtual` call — no node created, GPU eval elided for both.
  Perspective/sign verified verbatim (tss_core `lambda1_wrapper_is_verbatim`
  test; `HardValue::status` round-trips the sign). **Symmetric.**
- **Root/interior guard:** the play-time tactical guard consumes LOSS verdicts
  symmetrically (steers away from proven-losing moves; `classify_root_move`
  mints `-1` for loss children). The *deep root guard* forces a move only for
  WIN — design-justified (a root LOSS has no move to steer toward). The interior
  guard is λ¹ forced-defense narrowing, side-symmetric, not a Win/Loss verdict
  consumer.
- **Backprop:** a pinned loss propagates with identical fidelity (exact ±1, same
  routine, no averaging distinct from a win).
- **Training targets:** the value target is sign-symmetric via `hard_z`. The
  policy target reflects the certified line for LOSSes (guard/sharpen zero
  proven-losing moves) as it does for WINs (keep proven-winning moves). No site
  drops, attenuates, or caps loss facts in the consumption path. The `goal=win`
  root filter is NOT reachable in production (both channels use Both). Lever-2
  value pinning is unbuilt/banned for both signs.
- **Counters/telemetry:** proven losses were already counted in all modes
  (`deep_loss`). Consumed losses were NOT separable from consumed wins — **now
  fixed** (`deep_loss_backups`/`deep_win_backups`).

---

## 6. Test evidence

- Rust: `CARGO_TARGET_DIR=/e/cargo-target-valsig cargo test
  --manifest-path packages/hexfield_eq/Cargo.toml --features python
  --target x86_64-pc-windows-msvc` → **183 passed, 0 failed, 37 ignored**
  (includes the new `consumed_hard_backups_split_by_outcome`). Note: `mod
  search`/`mod tree` are behind the non-default `python` feature — the suite is
  meaningless without `--features python`.
- Python: selfplay.py `ast.parse` OK. The compiled `hexfield_eq._rust`
  extension is **not installed in this worktree**, so the end-to-end Python
  telemetry test is not runnable here without a maturin build (out of the RAM/
  build budget for this lane and not required — the Rust test proves the counter
  logic and the Python additions are `.get(..., 0)` mirrors of 40 sibling
  lines). The tss key set is read by explicit key (no exact-key-set assertion),
  so the additive keys cannot break existing consumers.

## 7. Files touched

- `packages/hexfield_eq/rust/src/tree.rs` — `TssCounters` fields, `add()`,
  `tss_consume_gate` split bump, `verified_loss_fixture_solve` helper + test.
- `packages/hexfield_eq/rust/src/search.rs` — payload `tss` dict emission of the
  two new keys.
- `packages/hexfield_eq/python/hexfield_eq/selfplay.py` — driver init,
  per-move aggregation, summary emission, epoch-merge `int_keys`.
- `docs/VALUE_SIGNAL_AUDIT.md` — this document.
