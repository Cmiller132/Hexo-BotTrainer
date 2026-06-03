# hexgt TSS + MCTS Threat Integration — Independent Code Review

**Branch reviewed:** `impl/hexgt-tss-softvalue` (read-only via `git show`; no
checkout, no build — the stopped hexgt run's worktree and the live dense_cnn run
were not touched).
**Scope:** the TSS/MCTS threat integration — `threats.rs` (threat core +
hitting-set), injection in `mcts_tree.rs` (`split_tactical`, forced edges,
`select_or_materialize_edge`), the leaf override in `mcts.rs`, and the
move-selection guard (`tactical_guard_weights` / `classify_root_move` /
`select_action_from_policy`). PART 2 (soft-Z / global-pooled readout) and the
feature-side changes are separate phases and out of this review's scope.

---

## HEADLINE — how does it look?

**Solid. High-quality, faithful to the finalized design, and the priority concern
is handled correctly.** I traced every path end-to-end and found **no correctness
defects**. Findings are a handful of low-severity perf/coverage notes. The
threat core is exact, phase-awareness is threaded consistently, and the
search/selection plumbing does what the design doc specifies.

### Priority question — multiple hitting sets: **CORRECTLY HANDLED, not a gap.**

Traced with code evidence:

1. **Injection injects the UNION, not one precomputed cover.**
   `threats::tactical_cells` (`threats.rs:182-189`) returns `own_winning_cells ∪
   {empties of EVERY opponent ≥4 window}`, deduped — *all* defensive options, not a
   single chosen hitting set. `split_tactical` (`mcts_tree.rs:835-875`) turns **every**
   one of those cells present in the candidate list into a `forced` edge. So MCTS
   materializes and can explore every block.
2. **The hitting-set SIZE is only a yes/no loss test.** `min_hitting_set`
   (`threats.rs:109-146`) returns `Some(k)`/`None`; the *cells* of any particular
   cover are never used to bias a move. `verdict()`/`forced_loss()`
   (`threats.rs:57-72`) only read `min_hitting_set.is_none()`. The override uses it
   purely as "does any ≤B cover exist?"
3. **The search chooses its favorite defense by value.** Each `forced` edge gets a
   guaranteed first visit (`select_or_materialize_edge`, `mcts_tree.rs:516-520`) so a
   low-prior block is never FPU-starved; after that, normal PUCT distributes visits
   by value across all edges. The played move is sampled from the (raw) visit policy;
   the move-selection guard (`tactical_guard_weights`, `mcts.rs:879-914`) **only**
   zeroes proven-losing moves and forces a proven win — among non-losing defenses the
   choice is by visit weight (value). It is **not** constrained to a precomputed pick.

The implementer's own native-MCTS test `test_hexgt_tss_move_selection.py`
corroborates: with two valid blocks `(2,4)/(3,4)`, across 60 seeds the played move is
always one of the two and **both are selectable** (`chosen <= safe_ids`), i.e. the
search is free to pick either by value — no single-cover lock-in.

**Verdict: it searches among multiple hitting sets and selects by value. No quality
gap on the multiple-hitting-set axis.**

---

## What was verified (positives)

- **Exact hitting set for B≤2** (`threats.rs:109-146`). `k=1` = a cell common to every
  opponent window's empties; `k=2` = exhaustive pair scan over the empties' union;
  else `None`. Since a turn places ≤2 stones, `B≤2` always, so this is an *exact
  minimum*, not a heuristic. Correct early-out to "loss" when ≥3 independent threats
  can't be covered (no false negatives — the pair scan is exhaustive, so it never
  declares a loss when a 2-cover exists).
- **Phase-awareness is threaded correctly everywhere.** `placements_remaining`
  (`threats.rs:31-36`) = 2 at FirstStone, 1 at Opening/SecondStone. Own win-now =
  count-5 for any B, count-4 only at B==2 (`threats.rs:159-162`, `79-90`); the
  defender's hitting-set budget is the same phase-aware B. Matches TEST G exactly. The
  override never returns +1 for a count-4 at a SecondStone leaf.
- **The forced-loss test is sound — no false hard-loss.** `min_hitting_set > B` ⟺ at
  least one opponent ≥4 window is left with all empties empty ⟺ the opponent completes
  it next turn. The 1-ply scope is honest: a non-winning counter-threat can't save the
  defender at depth 1 (opponent completes first), so restricting to own-win-now +
  hitting-set is correct (the counter-threat/quiet-refutation incompleteness is
  fenced into the deferred deep solver, see below).
- **Injection cap fix is correct and the displacement bug was caught.**
  `split_tactical` makes the cap **additive**: `nucleus + |tactical cells ranked
  outside the nucleus|` (`mcts_tree.rs:858-873`), not the earlier wrong
  `max(nucleus,|forced|)` which would have let an out-of-nucleus block *displace* a
  top-prior child. Covered by `injection_is_additive_and_preserves_top_prior_set` and
  `injection_tactical_inside_nucleus_costs_no_extra_cap`.
- **Forced-first-visit mechanism** (`mcts_tree.rs:516-520`) correctly guarantees each
  injected edge one visit before PUCT, with the right pending-skip guard, and only at
  threat nodes (the `T=∅` common path is a literal no-op — `injection_no_tactical_is_noop`).
- **Tactical cells are guaranteed legal candidates.** `candidate_cells` component (A)
  seeds the empties of **every active window unconditionally** (`candidates.rs:158-167`);
  the dead-cell filter only touches radius cells. So a needed block is never silently
  dropped from the candidate set before injection. (`split_tactical` skips any
  absent cell defensively, but for active ≥4 windows none are absent.)
- **Leaf override ordering & perspective are correct** (`mcts.rs:539-557`): terminal →
  existing-node → verdict override → net eval. Terminal detection precedes the
  override, so an already-won (count-6) state is handled by `terminal_value`, never the
  override (the `min_hitting_set` empty-set guard at `threats.rs:115` is thus a defensive
  unreachable). The verdict and backup both use `leaf_player = state.current_player()`,
  so the ±1 sign matches the existing terminal backup path. No node is created (mirrors
  the terminal branch) — accounting is consistent with virtual loss.
- **Training policy is left untouched.** Selection uses the raw `visit_policy` masked by
  the guard (`mcts.rs:640-647`); the **exported** target is `pruned_visit_policy`
  (`mcts.rs:631-635`, `665-670`) — guard-free. Masked (0) weights survive temperature:
  `0.powf(1/τ)=0` and argmax skips them (`select_action_from_policy:1114-1146`), with a
  safety-net fallback if the whole set is masked (`mcts.rs:910-912`). A loser cannot slip
  through noise/temperature.
- **Both selection sites are consistent.** The advance path (`mcts.rs:314-319`) and the
  reported-payload path (`mcts.rs:322→640-647`) both use `move_temps[index]` +
  `seed.wrapping_add(index)` with the same deterministic guard, so the reported action
  equals the advanced one.
- **Deep VCF/VCDT correctly deferred.** `vcf.rs` is explicitly an exploratory benchmark
  (`vcf.rs:1-23`), **not referenced** anywhere in `mcts.rs`/`mcts_tree.rs` (grep-clean),
  and it self-documents the soundness caveat (defender counter-threats not modeled ⇒ can
  over-claim). Matches the design's "(d) deferred."

---

## Findings (prioritized)

**No High or Medium correctness findings.** All Low.

| # | Sev | Finding | Evidence / suggestion |
|---|---|---|---|
| L1 | Low (perf) | The leaf override runs `threats::analyze` — a full `WindowStore::threats()` scan over *all* touched windows — on **every fresh leaf** on the CPU select path (the documented self-play bottleneck). It's cheap relative to the GNN forward it gates, but it is added to the hot loop and `threats()` iterates the whole store (grows late-game), not just ≥4 windows. | `mcts.rs:546`, `threats.rs:151-176`. Suggest measuring the per-leaf overhead at visits=512/active=64, and consider a cheap "store has any ≥4 window" short-circuit before building `opp_empties` (the store could expose a threat-count, or cache it on the incremental update). |
| L2 | Low (selection nuance) | Final move selection samples the **raw** visit policy (forced playouts included), so among multiple *non-losing* defenses a low-prior worse one can occasionally be sampled at high temperature due to forced-playout-inflated visits. This is intentional/pre-existing (opening diversity) and the guard guarantees it is never a *losing* move — but it means "favorite by value" is a strict argmax only at temperature 0. | `mcts.rs:626,640-647`; `forced_root_edge` `mcts_tree.rs:586-609`. Not a defect; flagging because it slightly blunts "pick the best defense by value" during exploratory self-play. Competitive play (τ→0) is unaffected. |
| L3 | Low (test coverage) | The implementer's tests cover *proven win always played*, *proven loss never played*, and *both blocks selectable* — but **not** the owner's exact "two viable defenses, one clearly better by value ⇒ better one chosen" assertion. The code guarantees value-weighted selection, but there is no regression pinning it. | Suggest a test: a position with two non-losing blocks where one leads to a clearly higher-value continuation, asserting (τ=0) the higher-value block is played. |
| L4 | Low (perf) | `tactical_guard_weights` calls `classify_root_move` for **every** root candidate at each move decision, each cloning the full `HexoState` (AHashMap window store) + apply + analyze. Gated on a threat existing, and per-decision (not per-sim), so amortized fine; could be notable for a dense threat position with many candidates. | `mcts.rs:840-914`. Acceptable; note only. Could early-out candidates not adjacent to any threat window. |
| L5 | Low (re-proof) | Override-proven leaves create no cached node, so a repeatedly-selected proven node re-runs `analyze` and re-backs ±1 each visit. Mirrors terminal-leaf handling (consistent), and PUCT abandons proven losses quickly, so impact is negligible. | `mcts.rs:546-556`. Note only. |

---

## Completeness vs the finalized design doc

- **(a) tactical injection at expansion** — implemented (root + interior threat nodes),
  additive cap, forced visits. ✓
- **(b) phase-aware hitting-set leaf override, 1-ply only** — implemented, exact, sound. ✓
- **(c) threat / hot-token features** — present per the branch history; **not reviewed
  here** (out of scope), flagging only that the parity test must still gate it.
- **(d) deep VCF/VCDT** — deferred; `vcf.rs` is an unwired exploratory benchmark with an
  honest over-claim caveat. ✓
- **Guard masks selection only, training target untouched** — implemented as specified. ✓

No TODOs, stubs, or silent shortcuts found in the reviewed files. `vcf.rs` is the only
"prototype" and is correctly fenced off.

---

## Method & caveat

This is a **static** review: end-to-end code trace plus reading the implementer's
native-MCTS regression tests. I did **not** execute a fresh GNN-MCTS two-distinct-defenses
simulation, because the `impl/hexgt-tss-softvalue` branch is not built and rebuilding the
Rust extensions into the shared WSL venv would alter the environment the **stopped hexgt
run** depends on — outside the "don't disturb the runs" constraint. The threat core
(`threats.rs`) is pure and small enough to verify by reading against the previously
engine-verified model (`scripts/_tss_verify*.py`), and the MCTS integration paths are
unambiguous. If a live confirmation is wanted, build the branch in an isolated worktree +
fresh venv and run `tests/test_hexgt_tss_*.py` (they drive the real native MCTS), then add
the L3 "better-defense-by-value" case.

**Bottom line:** the multiple-hitting-set handling is correct (union injection + value
selection; hitting-set is a yes/no loss test only), the threat math is exact and
phase-correct, and the integration is faithful to the finalized plan. Ship-quality, with
only minor perf/coverage follow-ups.
