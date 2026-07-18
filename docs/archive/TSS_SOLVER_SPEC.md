# TSS Deep Solver — Build Specification (Stage 3, docs/PLAN_TSS_DEEPENING.md §6)

**Audience:** the delegated implementer (Codex). **Owner constraint:** be
ambitious and optimize aggressively — but every optimization ships only with a
written mathematical/logical correctness argument in the proof document. An
unproven speedup is rejected; the unoptimized-but-proven form ships instead.

## 1. Goal

A deterministic, memory-capped, **proof-carrying** forced-tree solver for Hexo
(Connect6-variant) positions, plus an **independent certificate verifier** and
a **brute-force reference solver**, plus a differential/property test harness,
plus the proof document. The solver searches only the forced tree (threat
sequences); its WIN/LOSS claims must be *game-theoretically sound* — a false
hard value poisons the training pipeline through the MCTS backup, which is why
verification is structural, not advisory.

This build is **standalone**: no integration with the MCTS (`search.rs` /
`tree.rs`) — that is a later stage owned by the caller. You implement against
the frozen seam in `packages/hexfield_eq/rust/src/tss_core.rs`.

## 2. Game semantics you must honor (read these sources first)

- `packages/hexo_engine/rust/src/state.rs` — the engine: `HexoState`,
  `apply_placement`, `apply_with_delta`/`undo` (make/unmake), `TurnPhase`
  (`Opening`/`FirstStone`/`SecondStone`), outcomes. **Win = 6+ in a line.**
- `packages/hexo_models/rust/src/threats_shared.rs` — the λ¹ layer: a THREAT is
  a single-colour length-6 window with count ≥ 4; `B = placements_remaining`
  (2 at FirstStone, 1 at Opening/SecondStone); `analyze()` gives `own_win_now`
  (count-5 any B; count-4 iff B=2), `min_hitting_set` (exhaustive for k ≤ 2),
  `verdict()` (±1/None — **sound post-opening**, see the header).
- `packages/hexfield_eq/rust/src/tss_core.rs` — the seam you implement:
  `DeepSolve` (associated `Cert`), `CertVerify`, `SolveCaps`, `ProofStatus`
  {Win, Loss, Unknown}, `SolveStats`.
- Perspective is by **player identity, not ply parity**: FirstStone →
  SecondStone keeps the same player; naive per-ply negamax sign-flipping is
  WRONG for this game. (`classify_root_move` in `search.rs` shows the correct
  identity mapping.)
- `packages/hexgnn/rust/src/vcf.rs` is the UNSOUND prototype — reference for
  shape only. Its two defects (defender set restricted to hitting cells
  unconditionally; defender list capped at 24) are exactly what this build must
  not repeat.

### The forcedness boundary (the load-bearing subtlety)

At a defender node with live opponent threats and no own win-now:

- **`min_hitting_set == B` (defense consumes the whole turn):** every
  non-hitting move is refuted by a one-ply λ¹ argument (the move leaves the
  threat windows untouched with insufficient remaining budget; the attacker
  completes next turn; the mover's own count-4/5 cannot exist or λ¹ would have
  said win-now). Restricting the searched set to the **hitting-cell universe**
  is sound HERE — with the refutation *stapled*, not assumed.
- **`min_hitting_set < B` (a spare stone exists):** quiet moves and
  counter-threats are genuine options (the classic Connect6 tempo play).
  Restriction is NOT sound; these moves enter the searched set (or the node
  returns Unknown under budget).

"Exhaustive-with-instant-dispatch": every legal defender move is either
(a) instantly refuted by the λ¹ child verdict, or (b) searched. Nothing is
silently dropped, ever, at any cap.

## 3. Semantics of the three results

- **Win** — the side to move at the solved state has a proven winning
  strategy. OR (attacker) nodes may restrict to threat-creating moves: omitting
  attacker options can only MISS wins (safe). AND (defender) nodes must be
  exhaustive-with-instant-dispatch as above.
- **Loss** — the side to move provably loses: this is the **dual certificate**
  — a proven winning strategy for the opponent, whose universal nodes exhaust
  every legal move of the side to move (same machinery, seats swapped). "My
  attack failed" is NEVER Loss.
- **Unknown** — anything else: budget/cap exhaustion, incomplete coverage,
  spare-stone nodes not fully searched. Unknown propagates upward; a capped
  child makes the parent's universal claim Unknown. `SolveStats`/`hit_limit`
  style telemetry is never consulted for soundness.

## 4. Hard requirements

1. **Engine:** df-pn (depth-first proof-number search) or an argued-equivalent
   best-first AND/OR scheme; make/unmake via `apply_with_delta`/`undo` — no
   per-node state clones on the hot path.
2. **Determinism:** the result is a pure function of `(state, caps)` and
   nothing else. No wall clock anywhere in the solve; no HashMap iteration
   order reaching decisions (fix orderings explicitly); no randomness.
3. **Memory (a run-killer on the 29 GB training host):** the transposition
   table and every cache are **hard-capped in bytes** (`SolveCaps.tt_bytes_cap`)
   with an explicit replacement policy, and accounted (`peak_tt_bytes`). A cap
   binding degrades results toward Unknown — never toward a wrong verdict, and
   never unbounded growth. Include a test that solves under a tiny cap and
   asserts correctness of everything still claimed.
4. **Cache identity:** any value-bearing hit (TT or symmetry cache) compares
   the **full canonical position** — occupancy/owners, side to move, exact
   phase including the SecondStone first-placement witness — never a 64-bit
   hash alone. The neural `StateHash` (hexo_utils, history-bearing) must not be
   used. A D6-canonicalized outer cache is optional; if present it obeys the
   same full-representation equality (a canonical collision must not return a
   wrong hard hit).
5. **Certificates:** every Win/Loss carries a compact replayable certificate
   sufficient for an independent checker to verify the claim by replaying
   moves through the engine + λ¹ analysis only — no access to solver
   internals. Think: strategy tree with, at each universal node, the coverage
   argument (searched children + the instant-dispatch rule invoked for the
   rest). Bound certificate size; document the format.
6. **Verifier independence:** `CertVerify` lives in its own module and shares
   only `hexo_engine` primitives + `threats_shared::analyze` with the solver.
   It must reject: wrong-perspective claims, non-exhaustive universal coverage,
   instant-dispatch invocations whose λ¹ premise doesn't hold, cycles, and
   claims about the wrong position.
7. **Reference solver:** plain exhaustive minimax over ALL legal moves to a
   given ply budget, player-identity perspective, **independently written**:
   its own legal-move enumeration and its own direct six-in-line win scan (do
   NOT reuse the window store or the solver's move generation) so a common-mode
   bug cannot pass the differential.

## 5. Proof document — `docs/TSS_SOLVER_PROOF.md`

Lemmas, each mapped to the code (file:line) that embodies it and the test(s)
that exercise it:

- **L1 (instant dispatch).** At a defender node with live threats,
  no own win-now, and `min_hitting_set == remaining budget`, every non-hitting
  placement loses. Full B-accounting across FirstStone/SecondStone and the
  Opening edge case (why it cannot arise post-opening).
- **L2 (OR restriction safety).** Restricting attacker moves only loses
  completeness of WIN, never soundness.
- **L3 (AND completeness).** dispatch(node) ∪ searched(node) = all legal
  moves, at every universal node of every certificate, including under caps.
- **L4 (dual LOSS).** The LOSS certificate is a WIN certificate for the
  opponent whose universal side exhausts our legal moves; perspective mapping
  by player identity.
- **L5 (Unknown monotonicity).** No code path converts a capped/failed/absent
  proof into a verdict; Unknown children poison universal claims.
- **L6 (cache identity).** Why full-representation equality on every
  value-bearing hit suffices, and why `StateHash` is excluded. If a D6 outer
  cache exists: why certificate remapping under the group action is sound.
- **One subsection per optimization** (TT replacement, move ordering, proof
  number initialization, epsilon tricks, whatever you add): the argument why
  it cannot change any verdict, only cost. No subsection → no optimization.

## 6. Harness — definition of done

All under `cargo test -p hexfield_eq` (unit + integration), CPU-only:

1. **Differential vs reference:** randomized positions (seeded playouts across
   all phases; include threat-dense endgames) + curated fixtures (the four in
   tests/test_hexfield_eq_tss_shadow.py, plus counter-threat/tempo positions
   where restriction would over-claim — build positions where the defender's
   spare-stone counter-threat refutes a naive win claim). Agreement rule: on
   solver Win/Loss the reference (at sufficient depth) must agree; solver
   Unknown is always acceptable; any divergence is a failure.
2. **Every certificate verifies:** the verifier accepts all shipped
   certificates; mutation tests (corrupt a cert move / drop a child / flip the
   claim) are rejected.
3. **D6 replay:** for each of the 12 symmetries, solve(g·s) has the same
   status, and certificates re-verify after remapping onto g·s.
4. **TT on/off equality:** identical verdicts with the TT disabled, tiny, and
   large; forced-collision tests (two distinct positions engineered to share
   the 64-bit hash must not cross-contaminate).
5. **Make/unmake integrity:** state round-trips exactly under deep recursion.
6. **Cap behavior:** node/byte caps binding ⇒ Unknown (never a verdict flip);
   peak accounting accurate.
7. **Determinism:** repeated solves bit-identical; solves interleaved with
   other solves bit-identical.

Also provide a small bench harness (`cargo bench` or a test-gated timing
binary) reporting nodes/s and Unknown-rate by stones-on-board on the random
corpus — the caller sizes production caps from this.

## 7. Scope and boundaries

- New modules only: `packages/hexfield_eq/rust/src/tss_solver.rs`,
  `tss_verify.rs`, `tss_reference.rs` (+ optional `tss_bench.rs`), wired with
  `mod` declarations in `lib.rs`. **Do not modify** `search.rs`, `tree.rs`,
  `threats_shared.rs`, or existing behavior. If the frozen `tss_core.rs`
  interface needs adjustment, make the smallest additive change and flag it
  prominently in your summary — the caller re-reviews that seam.
- No Python-side changes. (An optional `#[pyfunction]` bench/diagnostic hook
  following the `hexfield_eq_threat_analysis` pattern is welcome but not
  required.)
- Workspace: your own git worktree/branch; commit granularity is yours; leave
  the tree building (`cargo check -p hexfield_eq --features python`) and green
  (`cargo test -p hexfield_eq`).

## 8. Acceptance (what the caller will do)

The caller re-runs your harness independently, adversarially audits each
lemma→code mapping in the proof doc, attempts to construct counterexample
positions (especially spare-stone counter-threat refutations and phase-edge
B-accounting), and only then wires consumption. Deliberately hostile review —
write the proof doc so it survives one.
