# TSS Solver Optimization — Build Specification (proof-carrying, round 2)

**Context.** The Stage-3 solver (docs/TSS_SOLVER_SPEC.md, docs/TSS_SOLVER_PROOF.md)
is CORRECT — the full harness, certificate verifier, and production shadow
deployment all agree (deep_verify_failed = 0). It is also **~100–1000× too
slow for leaf consumption**: first contact on the live run (self-play, real
positions) pegged 2 cores and dropped throughput 19 → 4.6 pos/s at full
sampling / node_cap 2000. The release bench
(`cargo test --release -p hexfield_eq tss_bench_report -- --ignored --nocapture`)
reproduces it: threat-dense buckets run at **141–705 nodes/s (~2–7 ms/node)**
while short-circuiting buckets run at 20k–138k nodes/s.

**Owner directive:** dramatically optimize the speed — ambitiously — but, as
before, every optimization ships only with a written correctness argument.
An unproven speedup is rejected; the proven-but-slower form ships instead.

## 1. Diagnosed hotspots (verify with a profile first; do not trust blindly)

1. **Universal (AND) expansion sweeps all legal moves** (~150–250 engine
   apply + analyze + undo per node) — the dominant per-node cost.
2. **OR move ordering applies-and-analyzes every candidate** to build its
   proof-cost tuple before searching anything (O4 in the proof doc).
3. **Per-solve TT**: thousands of solves per move share forcing structure and
   re-derive it cold every time.
4. **The dual attempt always runs**, doubling cost even when the caller
   consumes only one side.

## 2. Required deliverables

1. **A profile** (per-node cost breakdown on the bench corpus + the curated
   fixtures) confirming/refuting §1 before optimizing. Include it in the
   report.
2. **Optimizations** — the expected big levers, plus anything the profile
   justifies:
   - **Persistent/shared TT across solves** exposed as an explicit reusable
     handle the caller owns (`TssSolver` instance state or a `SolverCache`
     parameter): full-canonical-key equality on every value-bearing hit
     (§2.5 discipline, as today), **hard byte cap with replacement**
     (host-memory rule — the run host kills unbounded growth). Determinism
     contract RESTATED, not silently weakened: results may depend on the
     cache's prior contents, but (a) a given (state, caps, cache-state) is
     deterministic, (b) proofs remain certificate-verified individually, and
     (c) callers pin per-leaf results themselves (the search layer's per-move
     memo already guarantees idempotent re-selection).
   - **Incremental candidate generation / ordering from the WindowStore**:
     derive threat-creation features from window membership deltas instead of
     apply-per-candidate; or lazy ordering (order only what gets expanded).
   - **Cheaper universal handling**: at the L1 dispatch boundary skip
     non-hitting enumeration entirely (search-side; the verifier still checks
     per-move); at spare-budget nodes, lazy child materialization ordered by
     hitting-universe-first so refutations surface before the full sweep.
   - **Mode-aware budgets**: `SolveCaps` gains which sides the caller wants
     (win/loss/both); skip the un-consumed dual attempt.
3. **Wall-clock acceptance gate (the criterion round 1 lacked):** on the
   extended bench corpus (add ≥10 curated threat-dense fixtures, including
   the DEEP_WIN/FORCED_DEFENSE/FORCED_LOSS families from
   tests/test_hexfield_eq_tss_shadow.py), every stones-on-board bucket must
   reach **≥ 20,000 nodes/s**, and median full-solve wall-clock at
   node_cap=2000 must be **≤ 10 ms**. Report the final table. If a bucket
   cannot meet it soundly, say so explicitly with the reason.
4. **Proof-doc updates**: one subsection per optimization appended to
   docs/TSS_SOLVER_PROOF.md (shared-TT soundness argument mandatory: why a
   warm hit can never change a *verdict's validity*, only discovery), plus
   updates to any lemma whose code mapping moved.
5. **Harness stays green and grows**: all existing tests pass unmodified
   (except where an interface deliberately changed — flag those); add
   TT-sharing tests (warm-vs-cold same-verdict-validity, cross-solve
   contamination attempts via forced collisions, byte-cap enforcement under
   sustained reuse).

## 3. Scope and boundaries

- Modify `tss_solver.rs` / `tss_verify.rs` / `tss_reference.rs` /
  `tss_bench.rs` and `tss_core.rs` ONLY if the caps/interface needs additive
  fields (flag any such change prominently). **Do not modify** `search.rs`,
  `tree.rs`, `threats_shared.rs` — the caller-side wiring (per-move memo,
  counters, flags) is owned by the reviewer and adapts after acceptance.
- The verifier may gain performance work too, but its INDEPENDENCE contract
  is untouchable: no solver-derived shortcuts, same engine-primitives-only
  rule.
- Determinism: no wall clock anywhere in solve paths (the bench harness is
  the only timing site, as today).
- `cargo test -p hexfield_eq` and `cargo check -p hexfield_eq --features
  python` green; commit in logical commits on your branch.

## 4. Acceptance (reviewer will)

Re-run the harness + bench independently; audit each new proof subsection
adversarially (the shared-TT argument hardest); re-run the production golden
digest gates on the wired build; only then re-tune the live shadow knobs
upward.
