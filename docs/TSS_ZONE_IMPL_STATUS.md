# TSS zone-upgrade implementation status

Date: 2026-07-13  
Normative specification: `docs/PLAN_TSS_SOLVER_UPGRADES.md`  
Implementation brief: `docs/TSS_ZONE_IMPL_BRIEF.md`

## Outcome

Phases P0 through P3 are implemented in the working tree. P4 was not started,
as permitted by the brief. The single hard-value mint remains
`tss_core::hard_value_from_verified`; every synchronous and asynchronous deep
solve still reaches that mint only after independent certificate verification.

The requested per-phase Git commits could not be created because this managed
workspace permits writes to the worktree but not to the worktree's real Git
metadata directory. `git add`/`git commit` fails while creating
`E:\Hexo-BotTrainer-hexgt\.git\worktrees\tss-zone-upgrades\index.lock` with
`Permission denied`. All implementation changes therefore remain visible as
ordinary working-tree modifications.

## Phase and upgrade status

| Phase | Upgrade | Status | Implementation notes |
|---|---|---|---|
| P0 | U4 clock plumbing | Complete | `SolveCaps.semantic_horizon` is an absolute placement deadline; production recursion carries absolute ply separately from search depth. `horizon_retry` and preflight telemetry are plumbed through sync, async, Rust diagnostics, and Python epoch aggregation. |
| P0 | U10 fixture scaffolding | Complete | The G1 capped junction and G3 counterfork ownership geometries were ported to Rust `WindowStore` fixtures, including the four-arm/single-pin and three-disjoint-fork assertions. |
| P1 | U3 theorem dispatch | Complete | Production verification derives the dispatch boundary, requires every hitting cell explicitly, and does not construct/replay the legal complement. The old per-move staple remains test-only as a paired oracle. |
| P2 | U2 typed certificates | Complete | Added typed OR-completion, WIN, LOSS, Choice, and Universal nodes; stable `WindowKey` evidence; explicit node/edge/witness/commutation caps; D6 witness remapping; exact resolution arithmetic; final-DAG core derivation; legal-successor replay; nonempty AND, no defender-terminal edge, defender-own-win, Z1/Z2/Z4/Z5, D>=6, Opening, binding, acyclicity, reachability, and replay-memo checks. |
| P2 | U1 zone generator | Complete | Default-off proof-carrying zones implement hitting/A-touched/defender-completion candidates, D>=6 full-legal fallback, deterministic ordering, no defender-count truncation, and the monotone final-core/Z5 closure loop. The stale-area and count-2 heuristics are independent default-off flags. |
| P3 | U4 cache composition | Complete | `CachedProof` carries `resolution_t=max` and `zone_build_t=min`. Imports check resolution and enclosing horizon, composites retain the minimum build horizon, final certificate metadata rechecks global T, and smaller-horizon imports relabel D evidence before preflight. The executable slow-sibling counterexample is rejected atomically. Zoned fragment promotion is re-enabled only with these stamps. |
| P3 | U5 P3 commutation | Complete | Default-off pair generation uses a parent-frozen strict coordinate order, keeps newly legal lower cells, and records state-bound parent/mirror evidence. Verification requires both parent-legal cells, nonterminal singleton successors, exact child bindings, a materialized mirror edge, matching pair outcomes, and valid graph references. |
| P4 | U8/U9 | Not implemented | Optional phase; no flags were flipped and no partial P4 behavior was shipped. |

## Flags and defaults

All new rollout flags are additive and default to `false` in both Rust
divergence profiles and Python `SelfplayConfig`:

| Flag | Default | Purpose |
|---|---:|---|
| `tss_zone` | `false` | Enable proof-carrying zoned AND generation/verification. |
| `tss_zone_stale_filter` | `false` | Enable the exact all-18-windows-two-coloured stale-area filter. |
| `tss_zone_count2` | `false` | Include claimant count-2 windows in the initial search-zone heuristic. |
| `tss_pair_commutation` | `false` | Enable P3 same-turn defender-pair canonicalization. |

New telemetry is `horizon_retry`, `horizon_preflight_failed`, `zone_nodes`,
`pair_omitted`, `zone_verify_failed`, plus the existing fatal
`deep_verify_failed`. A retry that still has a mismatched exact horizon stops
as nonfatal Unknown before the minting verifier. A certificate submitted to
the minting verifier and rejected still increments `deep_verify_failed`.

## Test and benchmark evidence

- `cargo test -p hexfield_eq`: **58 passed, 1 ignored, 0 failed**. The ignored
  item is the opt-in release timing harness.
- The mutation tests accept valid typed WIN/LOSS/OR-completion and valid zone
  obligations, then reject corrupt witness/count/budget/ply/horizon evidence,
  zero-edge ANDs, terminal roots, defender own-win, dropped dispatch coverage,
  late final-core cells, D>=6 omissions, Z5 corridor omissions, Opening zones,
  cycles, orphans, invalid IDs, and oversized arenas.
- The U5 condition matrix covers parent-frozen order, absent/state-mismatched
  mirrors, newly legal second cells, singleton-terminal prefixes, and allowed
  joint-second-win pairs. The generator test confirms lower turn-start cells
  are omitted while higher and newly legal cells remain.
- U3 theorem-vs-per-move paired oracle: zero divergences on the deterministic
  curated/seeded corpus.
- One-sided optimized-vs-`tss_reference` differential: zero asserted
  divergences on the deterministic all-phase seeded corpus, repeated with the
  zone solver enabled at a matched semantic horizon. Optimized Unknown remains
  permitted.
- D6: all twelve certificate remaps replay and verify.
- Release harness:
  `cargo test -p hexfield_eq tss_bench_report --release -- --ignored --nocapture`
  passed. Cap-2000 summary: 82 positions, 9,296 nodes, 28,310 nodes/s,
  median 0.073 ms, p95 29.4898 ms, max 52.2909 ms; the harness's throughput
  and median gates were true. Curated DEEP_WIN was 0% Unknown and curated
  FORCED_LOSS was 0% Unknown in this run.
- Python files changed by the telemetry/flag plumbing parse successfully via
  `ast.parse`.

## Deviations and open validation items

1. **Git phase commits unavailable:** the sandbox's read-only external Git
   metadata prevented every commit. Intended conventional phase boundaries
   were P0 `feat(hexfield_eq): add TSS semantic ply clock`, P1
   `perf(hexfield_eq): verify TSS dispatch by theorem`, P2
   `feat(hexfield_eq): add proof-carrying TSS zones`, and P3
   `feat(hexfield_eq): add horizon-safe TSS cache and pair commutation`.
   *Resolved post-hoc (Claude, same day): committed as a single commit
   aa2b823f with the phase boundaries recorded in the message.*
2. **Python shadow suite unavailable:** `python -m pytest
   tests/test_hexfield_eq_tss_shadow.py -q` could not run because the provided
   `C:\Python314\python.exe` has no `pytest` module. This is an environment
   blocker, not a test failure. Python syntax validation passed.
   *Resolved post-hoc (Claude, same day): with `hexfield_eq` AND
   `hexo_engine` extensions maturin-built from this worktree and
   `PYTHONPATH=packages/{hexfield_eq,hexo_engine,hexo_runner}/python`
   (installed `hexo_utils` unshadowed), the full suite passes:
   **19 passed, 0 failed, 0 skipped** (199 s, CPU). Gotcha for future runs:
   mixing a stale installed `hexo_engine` pyd with a fresh `hexfield_eq`
   pyd aborts with nondeterministic multi-TB allocation failures (native
   ABI/state skew) — always build both extensions from the same tree.*
3. **G1/G3 representation:** the source experiment constructs arbitrary
   ownership maps in its mini-model, not legal engine placement histories.
   Their exact geometry and critical-cell properties are therefore tested via
   Rust `WindowStore` fixtures. Legal-engine matched-horizon coverage is
   supplied by the seeded `tss_reference` differential rather than by turning
   those arbitrary ownership maps into invalid `HexoState` histories.
4. Package-wide `cargo fmt --check` reports substantial pre-existing format
   drift in unrelated Rust files. The touched TSS modules were formatted
   directly with `rustfmt`, and `git diff --check` is clean.

No soundness obligation was weakened to work around these environment or
fixture-representation limitations.
