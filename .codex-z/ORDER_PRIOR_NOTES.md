# R-ORDER-PRIOR notes

## Changed

- Added a solve-level `SolveOrdering::{Off, Prior}` mode and included its
  `"off" | "prior"` name in the shared `EffectiveSolveConfig` resolver.
- Added `TssSolver::set_ordering_hints`, accepting position-specific
  `(HexCoord, f32)` weights for exactly the next `solve_goal` call.
- Wide-PN now generates each complete legacy child vector first, then applies
  a stable hinted-first sort. Higher weights precede lower weights; all
  unhinted children retain their previous relative order. Atomic attacker and
  defender pairs rank by their higher constituent weight and then their lower
  constituent weight.
- The same post-generation seam covers attacker Choice children (including
  first/second-stone atomic pairs) and defender Universal children. Hints do
  not participate in legality, threat gates, pair deduplication, candidate
  admission, or any pruning decision.
- When a generated node contains a hinted child, wide-PN drives the first
  still-live child in that stable prior order. Nodes with no matching hint use
  the pre-existing PN/zone selector unchanged.
- Hints are taken and removed at the `solve_goal` boundary, including early
  result paths. A nonempty hinted solve clears persistent TT/fragment state
  before and after the solve so order-dependent warm provenance cannot affect
  either the hinted position or the following unhinted position.
- Extended `hexfield_eq_solver_manifest` with source-compatible
  `ordering="off"` and an `ordering` echo.
- Extended `hexfield_eq_deep_solve_batch` with source-compatible
  `ordering_hints=None`. When present, it must have one list per state; each
  item is a finite `(q, r, weight)` tuple. Each state's hints are installed
  immediately before that state's verified solve.
- Added tests for no-hint verdict/certificate/node identity (win, loss,
  unknown, quiet), non-binding hint verdict parity, constructive node
  reduction, no-leak behavior with shared fragments enabled, root candidate
  set invariance, manifest mode echo, and batch hint validation.

## Constructive result

On the fixed `acly7kb` forcing fixture at cap 500, hinting the certificate's
proving root move `(3, -6)` proves WIN in 68 nodes. An away hint `(0, -6)` also
eventually proves the same verified WIN but uses 70 nodes. This gives a strict,
deterministic signal that the ordering mechanism is live.

## Tests

- Focused ordering tests:
  `cargo test --features python ordering -- --nocapture --test-threads=1`
  - 8 passed, 0 failed.
- Full parallel suite:
  `cargo test --features python`
  - 196 passed, 1 failed, 37 ignored.
  - The sole failure was the documented pre-existing parallel environment
    flake: `tss_cap_resume::cap_resume_discards_on_binding_or_cap_mismatch`
    returned `UnsupportedProfile` while environment-mutating warmth tests ran.
- Full serialized rerun:
  `cargo test --features python -- --test-threads=1`
  - 197 passed, 0 failed, 37 ignored.
- `git diff --check`: clean.
- `git diff --stat -- packages/hexfield_eq/rust/src/tss_verify.rs`: empty.
- `git diff --name-only -- scripts/tss_harness`: empty.
- Toolchain host confirmed as `x86_64-pc-windows-msvc` (`rustc 1.95.0`).

The requested `CARGO_TARGET_DIR=E:/cargo-targets/order-prior` could not be
created because this sandbox denies writes there. `E:/tmp` was also denied
despite being advertised as writable. Tests therefore used Cargo's existing
workspace target directory; the compiler target remained Windows MSVC.

## Flagged concerns / costs

- Cache isolation is intentionally conservative: every nonempty hinted solve
  is cold and discards any fragments it proves afterward. This guarantees the
  requested no-leak/node-identity property but gives up cross-position warmth
  around hinted solves.
- A coordinate hint applies anywhere that coordinate appears in the solve's
  wide-PN tree. This matches the per-solve coordinate-list contract and covers
  both Choice and Universal fanouts; no per-node policy maps are represented by
  the API.
- For an atomic two-stone edge, the API has only per-coordinate weights. The
  implemented deterministic aggregation is descending `(max_weight,
  min_weight)`; it does not invent a pair probability.
- The one-shot setter is consumed by each `solve_goal` call. Consequently, a
  verified horizon-ladder retry is a distinct solve and does not reuse the
  first attempt's hints, consistent with the explicit clear-after-each-solve
  requirement.
- Real network-prior extraction and `scripts/tss_harness/` wiring remain out of
  scope and were not changed.

