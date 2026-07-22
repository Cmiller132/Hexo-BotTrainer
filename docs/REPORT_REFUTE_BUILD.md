# RefuteLeafExact/V1 build report

Status: **FINISHED (prototype; not adopted)**. The leaf implementation,
independent verifier, fixtures, three realizable goldens, corpus round-trips,
firewall audits, and release suites are present. Amendment R3-1 resolves the
impossible root-level `Q=1` obligation with the realizable class-level
sole-orientation vector. The economics gates remain failed and continue to
prohibit adoption.

## Scope and module map

- **CODE-FACT** — `packages/hexfield_eq/rust/src/tss_refute_leaf_cert.rs`
  (385 lines) owns data-only wire constants/structs, canonical
  `RootSemanticPreimageV1`, the literal encoder, reachable-root token, compact
  baseline sizing, and dependency-free SHA-256.
- **CODE-FACT** — `packages/hexfield_eq/rust/src/tss_refute_produce.rs`
  (768 lines) owns the separately implemented producer regeneration and the
  explicit post-search entry point. `RefuteLeafModeV1::default()` and unknown
  `TSS_REFUTE_CERT_V1` values are `Off`.
- **CODE-FACT** — `packages/hexfield_eq/rust/src/tss_refute_verify.rs`
  (1,125 lines) owns the strict decoder, private direct state/geometry/window/
  transition/transversal implementation, resource meter, semantic replay, and
  public typed verifier result.
- **CODE-FACT** — `packages/hexfield_eq/rust/src/tss_refute_tests.rs`
  (891 lines) owns R2 fixtures, golden checks, D6, mutation, corpus, discovery,
  and economics harnesses.
- **CODE-FACT** — `scripts/refute_leaf_v1_oracle.py` (431 lines) is the third
  oracle. It uses only Python's standard library and imports no repository
  package.
- **CODE-FACT** — `scripts/check_refute_verifier_firewall.py` (76 lines) checks
  source imports and walks LLVM call edges from the public verifier root.
- **CODE-FACT** — `packages/hexfield_eq/rust/src/lib.rs` only registers the
  three isolated modules and the test module. No solver/search/consumer calls
  the producer.
- **MEASURED** — frozen `tss_verify.rs` hash after all work is
  `d99845c84500b7f480972b43214c6b1aee63a4b9`, identical to the starting hash.
- **CODE-FACT** — no `ProofStatus`, `HardValue`, `TssCertificate`, trainer,
  cache, pruning, or game-value conversion exists in the new modules.

## Identity, wire, predicate, and producer

- **CODE-FACT** — the preimage begins with the exact 25 domain bytes and writes
  ruleset, coordinate, class, wire, profile, shortest ULEB128 stone count,
  raw-sorted five-byte stone records, mover, full phase payload, placement
  clock, terminal byte, and claimant in the specified order.
- **CODE-FACT** — the accepted decoder admits only versions `1`, profile `1`,
  phase `FirstStone`, terminal `0`, claimant equal to mover, tag `0x20`,
  canonical ULEB128, exact payload length/checksum, and no trailing byte.
- **CODE-FACT** — `fail_*` values count ordered occurrences. Pair evaluation is
  streamed; no `T × T` occurrence table is retained. A guarded commuting class
  contributes two occurrences and one quotient class.
- **CODE-FACT** — producer cheap gates check mode, the complete natural-width
  profile metadata, strict `expansions < node_cap`, policy, phase, terminal,
  post-opening clock, reachability binding, and root D6 domain before semantic
  regeneration. Wrong profile returns `IneligibleLeafProfile`; equality cap
  returns `IneligibleNodeCap`.
- **CODE-FACT** — emission occurs only after the producer has checked absence
  of `ClaimantTerminal`, `OwnWinNow_A`, and `ForcedLoss_A`, regenerated all
  `T/G1/S/U` dispositions, observed zero completion/tactical/tight occurrences,
  encoded exact counters, and obtained acceptance from the independent public
  verifier.
- **CODE-FACT** — flag-off returns before root normalization, allocation,
  policy inspection, digesting, or semantic work. The ordinary solve path was
  not edited.

## Independent verifier and firewall

- **CODE-FACT** — the verifier rebuilds the semantic preimage independently
  from strictly decoded values and compares its SHA-256 to the stored digest.
- **CODE-FACT** — it builds a private `BTreeMap` board, derives the 18 literal
  windows through every occupied cell, checks all discovered coordinates for
  D6 safety before conversion/use, and does not call `Board::windows`,
  `WindowStore`, `threats_shared`, `tss_solver`, or the positive verifier.
- **CODE-FACT** — each pair placement is performed both by the private phase
  machine and by the engine transition primitive. Owner insertion, mover,
  phase including `SecondStone.first`, placement clock, and terminal winner
  must agree exactly.
- **CODE-FACT** — disposition order is illegal/nonclaimant terminal,
  claimant completion, no-new, defender-first, loose `0/1`, tight `2`, then
  tactical `>2`. Earlier root constructors are checked before leaf acceptance.
- **CODE-FACT** — externally supplied policy values cannot exceed the section
  2.5 ceilings. Wire/root allocation and count work are rejected before the
  associated operation/allocation; time is checked at deterministic charge
  points. Limit exhaustion returns `UnsupportedPolicyBudget`, never evidence.
- **MEASURED** — source firewall passed. LLVM IR generated with
  `rustc 1.95.0 (59807616e 2026-04-14)`, LLVM `22.1.2`, release/default feature
  set; the audit found one public verifier root and 34 transitively reachable
  LLVM symbols, with no denied module/symbol. Denylist SHA-256 was
  `41ab4f8b8b45abf852854a07186795162c3c100ace0a081bd31d113b0fd3fd86`.

## Fixtures, goldens, mutations, and corpus

### R2-1 fixtures

- **MEASURED** — reachable corpus root `8is963b/prefix91` has
  `ForcedLoss_A`; regeneration with only the earlier-constructor guard disabled
  found zero completion, tactical, and tight occurrences (`T=54`, `Q=3010`).
  The named predicate emitted no bytes and a forged `0x20` artifact was
  rejected as `ForcedLoss_A`.
- **MEASURED** — reachable corpus root `8is963b/prefix13` was rejected as the
  earlier `OwnWinNow_A` constructor; it emitted no bytes and its forged `0x20`
  artifact was rejected.
- **MEASURED** — a realizable terminal claimant root was rejected as
  `ClaimantTerminal` before the leaf cut.
- **MEASURED** — the semantic Q2 leaf under `SearchProfileV1::Other` returned
  `IneligibleLeafProfile`; with `expansions == node_cap == 200` it returned
  `IneligibleNodeCap`. Both branches occur before semantic regeneration or
  public self-verification.

### R2-2/R3-1 realizable goldens

| root | preimage SHA-256 | T | Q | classes | fail no-new/defender/loose0/loose1 |
|---|---|---:|---:|---:|---:|
| `q0_corpus_0hz3hty_prefix3` | `1e0ee42712858b73e46fcfe603a6400bf29676ccb5d5921fbdb52225b26d6167` | 0 | 0 | 0 | `0/0/0/0` |
| `q2_commuting_no_new` | `499d226e46bd418ab44e42819229b09b8ed47f31047856a059445586c73e5b0a` | 2 | 2 | 1 | `2/0/0/0` |
| `q4_sole_orientation_no_new` | `0ef6c6f1d35ff6826d655b3b11a8af537bf1be1da114034a1c7feef2adf2817c` | 1 | 4 | 4 | `4/0/0/0` |

- **MEASURED** — the Python oracle, producer, literal codec/preimage test, and
  independent verifier agree on all three rows, including full preimage hex frozen
  in the script and Rust test.
- **MEASURED** — the Q2 row combines identity and counter expectations in one
  accepted literal artifact and demonstrates two ordered occurrences versus
  one quotient class.
- **MEASURED** — independently replaying the R3-1 nine-placement history in
  the standard-library-only oracle reproduced the design's preimage bytes and
  digest exactly. The preimage is
  `485852464c56313a524f4f542d53454d414e5449433a5631000100010001000100010009ffff000001ffff06000100000000000100000000020000000003000500010400000000050000000006000000010101090000000001`.
- **MEASURED** — the same regeneration confirmed a nonterminal `FirstStone`
  root with `A=P1`; its sole defender count-at-least-four window is axis 0,
  start `(0,0)`, with empty `(3,0)`. No claimant-live window has count at
  least two; the only claimant-live window through `(3,0)` is axis 1, start
  `(3,0)`, with claimant stone `(3,5)`.
- **MEASURED** — the oracle regenerated `T={(3,0)}`,
  `G1={(3,1),(3,2),(3,3),(3,4)}`, the corresponding four-member ordered `U`,
  defender threat family `{{(3,0)}}` with `tau=1`, and no
  `ClaimantTerminal`, `OwnWinNow_A`, or `ForcedLoss_A` constructor.
- **MEASURED** — all four occurrences are `NoNewClaimantThreat` and form four
  singleton quotient classes. Therefore `Q=4=sum fail_*=sum_C |C|`, the four
  failing class counts sum to `quotient_class_count=4`, and selected class
  `{((3,0),(3,1))}` contributes exactly one occurrence; the Rust golden checks
  the same identities through producer, codec, and independent verifier.

### Mandatory Q=1 contradiction and R3-1 resolution

- **CODE-FACT** — under section 2.2, `Q=1` is impossible:
  1. If `|T| >= 2`, choose distinct `a,b in T`. Then
     `b in T-{a} subset S(a)` and `a in T-{b} subset S(b)`, so both ordered
     occurrences exist and `Q >= 2`.
  2. If `T` is empty, `Q=0`.
  3. If `T={a}`, any occurrence must get `b` from `G1(P,a)`. Its witnessing
     live claimant window has no defender stone. If its claimant count is at
     least two, legal `b` is already in `T`, contradicting `T={a}`. Therefore
     its claimant count is exactly one. The six-cell window then has five
     empties; besides `a`, all four other empties are legal because each is at
     hex distance at most five from that claimant stone. All four belong to
     `G1(P,a)`, hence `Q >= 4`.
- **CODE-FACT** — consequently no root can satisfy the required focused shape
  `Q=1`, `quotient_class_count=1`, selected `fail_*=1`. Creating one would
  require changing `T/G1/S/U`, legality, or the required count, which requires
  a new reviewed version.
- **CODE-FACT** — R3-1 accepts that proof and replaces only the impossible
  root-level vector with the realizable class-level sole-orientation vector
  above. The unchanged Q0 and Q2 identities still match their predecessor
  values byte-for-byte; the replacement supplies the required selected
  one-occurrence class without pretending the root itself can have `Q=1`.

### Mutation and closure results

- **MEASURED** — bad magic/version, redundant ULEB128, trailing byte, payload
  checksum, payload length/count, owner, stone order, phase, terminal,
  claimant, semantic digest, tag, cross-root binding, and externally lowered
  budgets all rejected.
- **MEASURED** — one-sided weak-`G1` omission, stale turn-start defender/Q
  omission, quotient class/occurrence conflation, and changed redundant failure
  counters rejected after checksum repair.
- **MEASURED** — private tests exercise exact tau `0`, `1`, `2`, and `>2`
  cases; corrupt `SecondStone.first`, mover transition, and terminal result do
  not agree with the engine state and cannot contribute to acceptance.
- **MEASURED** — all 12 D6 images of the accepted Q2 fixture produce and verify
  with identical counters; original bytes reject against every distinct raw
  image.
- **MEASURED** — source/import and compiled reachability firewall audits pass.
- **HYPOTHESIS** — a future exhaustive bounded-state cross-language campaign
  would provide broader evidence than the fixed third-oracle vectors. It was
  not treated as a substitute for the fixed R2-2/R3-1 golden set.

### Real corpus leaves

- **MEASURED** — `0hz3hty/prefix5`, `0l4291i_live/prefix5`, and
  `8is963b/prefix7` were replayed from the production forcing corpus. For each,
  ordinary pair-complete search at cap 200 and semantic horizon `u32::MAX`
  returned `Unknown`, used one expansion, stayed strictly below cap, and
  reported root `(pn=1_000_000_000,dn=0)` after staged refresh. All three then
  produced and independently verified a leaf artifact.

## Economics measurement (measure only; no adoption)

The release harness used warmed code, a fresh solver/verifier per repetition,
30 serialized repetitions per root, and the same three corpus roots.

| root | bytes / baseline | ratio | emit median/p95/max us | verify median/p95/max us | search median/p95/max us |
|---|---:|---:|---:|---:|---:|
| `0hz3hty/p5` | 125 / 118 | 1.059322 | 895.3 / 932.6 / 1073.7 | 537.6 / 560.9 / 582.6 | 18.7 / 26.3 / 223.1 |
| `0l4291i_live/p5` | 125 / 118 | 1.059322 | 83.8 / 93.1 / 109.9 | 41.7 / 42.9 / 46.1 | 15.6 / 18.1 / 111.6 |
| `8is963b/p7` | 135 / 128 | 1.054688 | 1139.6 / 1195.7 / 1221.4 | 659.3 / 687.3 / 702.4 | 24.6 / 30.9 / 49.3 |

- **MEASURED** — aggregate bytes were `385 / 364 = 1.057692`; the `<=110%`
  leaf-size gate passes on every root and in aggregate.
- **MEASURED** — aggregate 90-call totals were emission `64.060 ms`, standalone
  verification `37.314 ms`, and matched cold search `2.136 ms`. Aggregate
  verification/search was about `17.47x`, so the `<75%` replay gate fails.
  Aggregate emission/search was about `29.99x`, so producer, enabled-workflow,
  and three-audit amortization gates also fail on this tiny one-expansion cohort.
- **MEASURED** — all producer/verifier absolute times were far below 30 s CPU /
  60 s wall. The largest verifier work row had `W=81`, `Q=16`, 90 threat
  memberships, 4,588 pair operations, 75 transversal operations, 168 retained
  state bytes, and a conservative 504-byte internal heap estimate.
- **CODE-FACT** — the harness reports internal checked state/heap estimates,
  not OS-sampled peak RSS. No unmeasured RSS value is claimed.
- **HYPOTHESIS** — larger searches could improve relative economics, but this
  campaign does not establish that and the failed measured gates prohibit
  adoption.

## Commands and outcomes

- **MEASURED — PASS** — `python scripts/refute_leaf_v1_oracle.py` (all three
  frozen realizable vectors and the detailed R3-1 semantic census matched).
- **MEASURED — PASS** — the full refute test module under the release/MSVC,
  local-target, 32 MiB stack, and serialized settings — 9 passed, 7 ignored.
- **MEASURED — PASS** — `cargo rustc -p hexfield_eq --lib --release --target
  x86_64-pc-windows-msvc -- --emit=llvm-ir`, followed by
  `python scripts/check_refute_verifier_firewall.py
  .cargo-target/x86_64-pc-windows-msvc/release/deps/hexfield_eq.ll` (one root,
  34 reachable symbols, no denylist hit).
- **MEASURED — PASS** — with `CARGO_TARGET_DIR=.cargo-target`, target
  `x86_64-pc-windows-msvc`, `RUST_MIN_STACK=33554432`, RAM preflight above
  8 GiB, and `--test-threads=1`:
  `cargo test -p hexfield_eq --lib --release --target
  x86_64-pc-windows-msvc -- --test-threads=1` — 153 passed, 49 ignored.
- **MEASURED — PASS** — the same command with `--features python` — 237
  passed, 50 ignored; only seven pre-existing pyo3 deprecation warnings.
- **MEASURED — PASS** — economics harness — three roots × 30 repetitions.
- **MEASURED — PASS** — `git diff --check`.
- **MEASURED — PASS** — free physical RAM preflights observed 13.07–14.89 GiB
  before the focused and complete release test runs.
- **CODE-FACT** — no commit was created.

## Gate disposition

- **CODE-FACT** — implementation remains default-off and has no consumer.
- **MEASURED** — realizable wire/identity, Q0/Q2/R3-1 sole-orientation,
  occurrence/class sum identities, R2-1, mutation, D6, resource, corpus,
  firewall, and release-test evidence passes.
- **MEASURED** — size passes; replay and producer economics fail on the measured
  leaf cohort, as expected for a measure-only prototype and not adopted.
- **CODE-FACT** — R3-1 resolves the impossible Q1 gate without changing the
  frozen semantics. The requested implementation definition of done is
  complete; the separate failed economics gates still prohibit adoption.

REFUTE_BUILD_DONE
