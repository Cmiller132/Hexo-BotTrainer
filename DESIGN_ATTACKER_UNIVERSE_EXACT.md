# R-B1-EXACT — attacker-universe characterization at `62818b2a`

Date: 2026-07-20  
Branch/worktree: `claude/tss-vcf-width`, `62818b2a12606b60864dca5ac3b07b406448b2b8` plus the shadow-only changes described here.  
Scope: `WidthOptions::vcf_pair_complete()` only. No tightening is shipped.

## Result

The implemented normal-turn attacker universe is an ordered, left-biased union followed by a forcing filter and unordered-pair deduplication. At a claimant FirstStone position it is:

\[
T(P)=C_{\ge2}(P)\cup D_{\ge4}(P),
\]

\[
S(P,a)=G_{\ge2}(P,a)\;\overset{\rm ord}{\cup}\;(T(P)\setminus\{a\})\;\overset{\rm ord}{\cup}\;G_1(P,a),
\]

and the searched pair edges are the unordered pairs \(\{a,b\}\), with \(a\in T(P)\), \(b\in S(P,a)\), that pass the complete-turn forcing classifier. Here \(G_{\ge2}(P,a)\subseteq T(P)\), so the raw second-coordinate **set** simplifies to

\[
S(P,a)=(T(P)\setminus\{a\})\cup G_1(P,a).
\]

The `G_ge2` prefix changes ordering, not membership. The known slight superset is narrower than the old comment may suggest: it is principally frozen turn-start defender-block cells whose block status disappears after `a`. The exact post-apply reference set can therefore be recovered without engine mutation by retaining the claimant-window terms above but recomputing, or statelessly subtracting, dead defender-block-only cells.

The fresh official all-19 gate passed: 14 WIN, 5 non-WIN, strict-verifier failures 0. The measurement flag was also exact on every one of the 34 ladder attempts: off and on produced the same normalized search signature SHA-256, `82343502C50E3F8B4B90BB8A58165CD7849F8E9BCF687BD9DF7F430C55FA9E72`.

## 1. Exact implemented universe

### Definitions

For a position `P`, claimant `A`, defender `D`, and a live six-cell window `w`:

- `owner_P(w)` is its sole stone owner; mixed windows are inactive.
- `n_P(w)` is that owner's stone count.
- `E_P(w)` is its empty-cell set.
- `C_ge2(P) = union E_P(w)` over claimant-pure windows with `n_P(w) >= 2`.
- `D_ge4(P) = union E_P(w)` over defender-pure windows with `n_P(w) >= 4`.
- `T(P) = C_ge2(P) union D_ge4(P)`.
- `G_ge2(P,a) = union (E_P(w) - {a})` over claimant-pure `n_P(w) >= 2` windows containing `a` as an empty.
- `G_1(P,a) = union (E_P(w) - {a})` over claimant-pure `n_P(w) = 1` windows containing `a` as an empty.

`threat_creating_moves_with_threshold(..., 2)` constructs `T(P)`: claimant count>=2 empties are inserted and deduplicated, then every live defender count>=4 empty is inserted or marked as a defender block. See `tss_solver.rs:8620-8709`. `ordered_threat_creating_moves_with_width` ranks that unchanged set; see `tss_solver.rs:8722-8807`. Its two width tiers are narrow/mandatory-block (`defender_block || strength>=3`) and count-two-only; see `tss_solver.rs:8588-8595`.

At a FirstStone OR node, `attack_pair_children` freezes that ordered list as `first_candidates`; see `tss_solver.rs:6317-6407`. For every `a` in that list, `WideTurnGate::second_candidates` emits:

1. `G_ge2(P,a)`, ordered by descending window strength and raw coordinate;
2. the frozen `T(P)` order, excluding coordinates already emitted and `a`;
3. `G_1(P,a)`, raw-coordinate ordered and excluding coordinates already emitted.

This is the literal code at `tss_solver.rs:8915-8973`. Left-biased dedup is by the `seen` set. `WideTurnGate::build` supplies the count>=2, count-1, and defender-threat snapshots at `tss_solver.rs:8857-8902`.

### The forcing filter

The raw pair `(a,b)` is not yet an OR edge. `WideTurnGate::evaluate_pair` retains it only if all of the following hold (`tss_solver.rs:9020-9107`):

1. Placing the pair creates at least one claimant-pure count>=4 window (`family` is nonempty).
2. The pair hits every pre-existing defender count>=4 window; otherwise `defender_win_now` rejects it.
3. The post-pair claimant threat family has minimum hitting-set size exactly the defender budget two, or exceeds that budget. `mhs == Some(2)` becomes a pending forced defender node; `mhs == None` becomes a tactical leaf when the sparse LOSS obstruction fits the horizon, otherwise pending.
4. The two coordinate orders are deduplicated by the unordered raw-coordinate pair. The first accepted ordering is retained (`tss_solver.rs:6513-6531`).

Thus the actual FirstStone OR universe is:

\[
U(P)=\operatorname{dedup}_{\{a,b\}}
\left\{(a,b):a\in T(P),\ b\in S(P,a),\ \operatorname{Force}_2(P,a,b)\right\}.
\]

`Force_2` denotes the four checks above, including horizon-sensitive tactical materialization. A failed restricted attacker universe yields Unknown, never a defender proof.

### Partial-turn and opening roots

If the root is already at SecondStone, the engine uses the freshly regenerated `T(P)` single-coordinate list, then admits only an immediate win/tactical leaf or a completed turn satisfying `turn_created_claimant_threat && turn_forces_small_defender_reply` (`tss_solver.rs:6675-6799`). An Opening root uses the same single-coordinate generator and admits pending children. The corpus contains two such single OR observations across all ladder attempts; the normal engine path is overwhelmingly the atomic pair path.

### Exact post-apply reference relation

Let `R(P+a)` be the historical generator called after applying `a`. Its claimant part is exactly:

\[
(C_{\ge2}(P)\setminus\{a\})\cup G_1(P,a).
\]

The first term persists because adding a claimant stone cannot destroy a claimant-pure window. The second term consists precisely of count-1 windows promoted to count two by `a`. `G_ge2` is already contained in the first term and is only an ordering prefix.

The defender-block part differs:

\[
D_{\ge4}(P+a)\subseteq D_{\ge4}(P)\setminus\{a\}.
\]

The implementation uses the right-hand frozen set. Any cell in the set difference that has no claimant-window membership is stale width. This gives the concrete exactness target:

\[
S_{exact}(P,a)=
(C_{\ge2}(P)\setminus\{a\})\cup G_1(P,a)\cup D_{\ge4}(P+a).
\]

No tightening is made in this round.

## 2. Completeness accounting

“Complete” here means complete for the solver's forcing-VCF contract, not for arbitrary Hex strategy wins containing quiet turns. The latter require the separate round-3 quiet-turn universe and are outside `vcf_pair_complete()`.

The argument factors into explicit obligations:

1. **Attacker placement coverage.** Assume a winning forcing strategy has a normal form in which each claimant turn wins immediately or creates a new live count>=4 threat and leaves the defender at the tight `tau=b` dispatch boundary. Its first move lies in `T(P)`. Its legal second move lies in the post-apply reference set `S_exact(P,a)`, which is a subset of implemented `S(P,a)`. Therefore every normal-form attacker turn occurs in the raw pair enumeration.
2. **Turn filter preservation.** The assumed normal form satisfies the `family != empty`, no surviving defender win-now, and `tau=2`/`tau>2` checks, so `evaluate_pair` retains it. Immediate/tactical outcomes are materialized rather than discarded.
3. **Pair quotient.** The pair represents a complete same-player turn. Deduplicating `(a,b)` and `(b,a)` preserves the resulting board when both orders are legal; the implementation deduplicates only after classification and retains an actually generated order. The plan records P3 as done, but also records its Lean formalization as backlog (`PLAN_TSS_SOLVER_UPGRADES.md:713`).
4. **Defender coverage.** Every descendant defender node is at `min_hitting_set == b`. `forced_defender_replies` uses the extendable-hit kernel (`tss_solver.rs:9294-9352`). The plan records U3 staple-by-theorem as done with lambda-one soundness proven, and U15/T6 kernel calculus as landed (`PLAN_TSS_SOLVER_UPGRADES.md:711,723`). Consequently omitted defender replies are theorem-dismissed, not silently ignored.
5. **Proof acceptance.** Search restrictions can miss wins but cannot manufacture them. Every returned hard certificate is independently replayed by `TssVerifier`; the 19-row harness asserts acceptance for every non-Unknown result (`tss_corpus.rs:267-546`). `tss_verify.rs` was not changed.

### Proven versus asserted

| Claim | Status supported in this repository |
|---|---|
| Strict certificate soundness for returned WIN/LOSS | Implemented and tested; all tip-gate certificates accepted. |
| Defender lambda-one/U3 dismissal | Recorded as proven/done in `PLAN_TSS_SOLVER_UPGRADES.md:711`. |
| Forced-node `K_b`/T6 calculus | Kernel algebra landed; the plan explicitly says the full T6 region contract is still being stated (`:723`). |
| Same-turn pair quotient | Implemented and dispositioned done; Lean P3 row is explicitly unformalized (`:713`). |
| All 14 corpus lines satisfy turn forcing | Externally checked and recorded as proven for this finite corpus in `TSS_VCF_WIDTH_BRIEF.md:221-235`. |
| Count>=2 suffices for all 181 attacker placements in those lines; 53 need it | Finite-corpus external result in `TSS_VCF_WIDTH_BRIEF.md:237-250`. It is not a theorem about every possible game position. |
| Every arbitrary attacker win has the forcing/count>=2 normal form | Not claimed here. `vcf_pair_complete` is complete only under the forcing-VCF normal-form obligation above. |

This distinction is essential: the gate proves empirical WIN-completeness on the designated corpus plus verifier soundness. It does not upgrade the attacker normal-form assumption into a global game theorem.

## 3. Tightness and removal candidates

The checked-in forcing-line file makes the examples below replayable: start from the named position in `rust/corpus/forcing_corpus_moves.txt`, then apply the listed continuation. The component classifications were independently recomputed from six-cell windows; they are line witnesses, not claims that the position has no alternative winning strategy unless an ablation is also stated.

### A. Turn-start claimant count>=3 cells — retain; line witness

For `0hz3hty`, replay continuation prefix:

```text
(5,3) (5,5) (5,1) (5,6) (7,3) (8,2) (3,7) (9,1)
```

The next attacker first move `(8,4)` is count>=3-only at that position (not count-two and not a defender block), followed by `(8,6)`. Removing this component deletes the recorded forcing line. A true necessity result would run a component ablation and establish that no alternate certificate exists; that ablation is proposed, not shipped.

### B. Turn-start claimant count-two cells — retain; strongest line witness

At the root of `0l4291i_live`, attacker move `(1,5)` is count-two-only. The complete recorded continuation begins:

```text
(1,5) (2,5) (-2,5) (4,5) (11,-2) (11,-1) (11,-4) (11,2)
(2,4) (9,-5) (2,1) (10,-6) (9,-1) (13,-1) ... (13,0) (14,0)
```

The finite-corpus proof accounting records 53 attacker placements that need the count-two tier and zero needing a count-one/r3 escalation. This component is not a plausible removal target.

### C. Turn-start defender-block contribution — retain at first ply; tighten stale second-ply cells

Defender blocks are required at the first coordinate to permit threat-plus-block tempo turns; `pair_complete_width_keeps_defender_threat_blocks` pins their inclusion. The removable part is narrower: after choosing `a`, delete a frozen block-only `b` if every defender count>=4 window containing `b` has already died and `b` is not in the post-`a` claimant count>=2 set.

Proposed safety proof: establish the set identity for `S_exact(P,a)` above, then show `second_candidates_exact == ordered_threat_creating_moves_with_width(P+a)` as a property test over seeded legal states and all `a in T(P)`. This is a sound branching-factor reduction because it removes exactly `D_ge4(P) - D_ge4(P+a)` cells with no other membership. Follow with the all-19 gate and a broad randomized differential against the apply/regenerate reference.

### D. `G_ge2` second-ply promotion prefix — set-redundant, safe code-removal candidate

Every cell in `G_ge2(P,a)` is already in frozen `T(P)`. Removing this component from the **set construction** is therefore safe by inclusion, but it must be replaced by an equivalent stable partition if the current strongest-promotion ordering is to remain byte-identical. It saves generation/sort work, not branching, because the same cells remain in `T(P)`.

### E. Frozen turn-start second-ply cells — necessary line witness, except stale blocks

After the first eight `0l4291i_live` continuation moves above, the attacker pair is `(2,4),(9,-5)`. For first `(2,4)`, second `(9,-5)` belongs to the frozen turn-start component only: it is neither a through-first count>=2 promotion nor a through-first count-1 promotion. Removing the whole frozen component deletes this recorded forcing line. The stale defender-block-only subset from C remains removable.

### F. `G_1` weak-window promotion — high-value removal candidate, not yet proved safe

This is the only component that adds claimant cells beyond frozen `T(P)`: after `a` joins a claimant count-1 window, its other empties become post-apply count-two cells. It is therefore part of the exact apply/regenerate reference and cannot be called logically redundant.

However, the 14 checked-in reference lines have zero second moves admitted exclusively by this component in an independent turn-start census. In the live certificate measurement, only 19 of 77,678 winning pair edges (0.0245%) used a weak-only second coordinate, although weak membership overlapped 49,599 winning edges. This makes it the best empirical removal candidate after stale blocks.

Proposed proof/experiment sequence:

1. Add a shadow ablation that marks weak-only edges but never steers; retain concrete certificate paths for the 19 observed carriers.
2. Prove or refute a pair-normalization lemma: every forcing pair with weak-only second `b` has an equally forcing ordered pair using a frozen-turn-start second coordinate, possibly by exchanging the quiet build with the threat-making coordinate. Counterexamples are expected to be threat-plus-build constructions, so this must not be assumed.
3. Only if the lemma holds, run an actual consume ablation in a separate tightening round: all-19 gate, randomized reference differential, and mined human-leaf positives. No consume ablation is included here.

### Ranked payoff

1. Remove stale defender-block-only second cells: exact-reference equality, genuine branching reduction, clearest proof path.
2. Investigate weak-only `G_1`: genuine branching reduction and almost no observed proof carriage, but the normal-form proof is open.
3. Eliminate the redundant `G_ge2` construction while preserving its order as a view/partition: generation savings only.

## 4. Shadow measurement

### Implementation contract

The collector is `cfg(test)`, default off, and enabled only by `TSS_ATTACKER_UNIVERSE_SHADOW=1` (`tss_solver.rs:104-214,4061-4072`). It records:

- exact histograms of turn-start candidate count, per-first second-candidate count, retained forcing/dedup pair count, and partial-turn retained singles;
- overlapping first-coordinate membership `[count>=3, count-two, defender-block]`;
- overlapping second-coordinate membership and left-biased first admission `[G_ge2, frozen turn-start, G_1]`;
- the same component masks for the child that closes each winning Choice node (`tss_solver.rs:5715-5752`).

The masks are stored after child generation and are read only when a node transitions to `pn=0`. No counter or mask is consulted by generation, ordering, selection, proof numbers, materialization, or verification. The corpus harness prints per-cap and aggregate rows only when enabled (`tss_corpus.rs:556-581,731-755`).

### Corpus-wide distribution

Official profile: release, `x86_64-pc-windows-msvc`, `.target-b1`, 2 GiB TT, one test thread, WIN ladder 10k/100k/1M/20M, NO cap 1M.

| Quantity | observations | min | mean | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| turn-start/partial first candidates per OR | 698,407 | 0 | 48.252 | 49 | 62 | 65 | 91 |
| raw second candidates per first | 33,699,676 | 8 | 53.509 | 54 | 66 | 69 | 101 |
| retained forcing/dedup pairs per pair OR | 698,405 | 0 | 10.733 | 4 | 28 | 45 | 415 |
| retained singles per partial OR | 2 | 2 | 2.000 | 2 | 2 | 2 | 2 |

Zero retained pairs occurred at 189,449 pair OR nodes (27.126%). The forcing classifier therefore removes most of the raw Cartesian width, but the generator still pays to build and classify it.

Across all 1,803,229,707 raw second-coordinate slots, first admission was:

| component | slots | share |
|---|---:|---:|
| `G_ge2` promotion prefix | 121,040,122 | 6.7124% |
| frozen turn-start | 1,556,021,824 | 86.2908% |
| `G_1` weak promotion | 126,167,761 | 6.9968% |

### Proof carriage

The solved frontiers closed 77,679 winning Choice edges: 77,678 atomic pairs and one partial-turn single. Membership percentages overlap.

| coordinate/component | winning edges | share |
|---|---:|---:|
| first in count>=3 claimant window | 63,352 | 81.5561% |
| first in a count-two claimant window | 76,345 | 98.2827% |
| first is a defender block | 6,018 | 7.7473% |
| second in `G_ge2` | 67,226 | 86.5445% of pairs |
| second in frozen turn-start | 77,659 | 99.9755% of pairs |
| second in `G_1` | 49,599 | 63.8521% of pairs |

For the mutually exclusive first-admitting second component:

| component | winning pair edges | share |
|---|---:|---:|
| `G_ge2` | 67,226 | 86.5445% |
| frozen turn-start | 10,433 | 13.4311% |
| `G_1` weak-only | 19 | 0.0245% |

These are closure-edge counts across all ladder attempts, not unique certificate coordinates or independent positions. A node proven again in a fresh cap attempt is counted again; that is intentional because the unit of cost is the attempted solve.

### Flag identity and cost

The normalized signature contains every `CORPUS` attempt's id, cap, status, expected class, nodes, expansions, TT entries, TT hits, TT cap, peak TT bytes, stage refreshes, interior-gate counts, and seed scans. Results:

| run | attempts | test-body wall | failures | signature SHA-256 |
|---|---:|---:|---:|---|
| Phase A, pre-instrumentation tip | 34 | 533.96 s | 0 | `82343502...FA9E72` |
| instrumented, flag off | 34 | 539.49 s | 0 | `82343502...FA9E72` |
| instrumented, flag on | 34 | 746.86 s | 0 | `82343502...FA9E72` |

`Compare-Object` was empty for Phase-A vs off and off vs on. Verdicts and all deterministic search fields, including node counts, are identical. The on collector cost 207.37 s / 38.44% relative to the immediately preceding off test body on this loaded machine; that is acceptable only as offline shadow instrumentation.

## 5. Fresh tip gate

### Per-row Phase A walls

Walls sum every ladder attempt for a row. `final cap` and `final nodes` are the last attempt. LOSS and UNKNOWN are both accepted for expected-NO rows.

| id | expected | final | row wall (s) | final cap | final nodes |
|---|---|---|---:|---:|---:|
| 0hz3hty | WIN | WIN | 0.130 | 10,000 | 2,412 |
| 0l4291i_live | WIN | WIN | 347.585 | 20,000,000 | 1,879,612 |
| 8is963b | NO | LOSS | 0.000 | 10,000 | 1 |
| 94gnnol | NO | UNKNOWN | 144.799 | 1,000,000 | 1,000,000 |
| acly7kb | WIN | WIN | 0.009 | 10,000 | 75 |
| dy3dg99 | NO | LOSS | 0.000 | 10,000 | 1 |
| g2xx6wl | WIN | WIN | 0.582 | 10,000 | 4,107 |
| hu01jk4 | WIN | WIN | 0.094 | 10,000 | 380 |
| jh7yo7y | WIN | WIN | 0.252 | 10,000 | 2,119 |
| jnzzmcm | WIN | WIN | 0.860 | 10,000 | 9,798 |
| l9mxn59 | NO | UNKNOWN | 0.046 | 1,000,000 | 226 |
| lz60mfb | WIN | WIN | 20.750 | 1,000,000 | 109,896 |
| mvp2lvc | NO | UNKNOWN | 5.488 | 1,000,000 | 17,957 |
| xsnfyll | WIN | WIN | 0.005 | 10,000 | 82 |
| zrugh2x | WIN | WIN | 5.941 | 100,000 | 41,734 |
| strongloss_a_prefix6 | WIN | WIN | 1.896 | 100,000 | 16,126 |
| strongloss_b_prefix8 | WIN | WIN | 0.068 | 10,000 | 1,099 |
| hayes_20260712_turn16 | WIN | WIN | 2.674 | 100,000 | 11,664 |
| hayes_20260712_placement31 | WIN | WIN | 2.401 | 100,000 | 11,664 |
| **total solve rows** | **14 WIN + 5 NO** | **gate pass** | **533.580** |  |  |

Harness wall was 533.96 s. Cargo process wall was 551.025 s, including a 16.91 s cold release build. Every returned hard certificate passed the strict verifier; harness `failures=0`.

### Historical comparison and load caveat

The prompt labels 177 s as the round-9b full-corpus mark. The repository plan instead identifies 177.7 s as the round-9b **`0l4291i_live` full solve**, compared with pdspn 264 s on that same hardest position (`PLAN_TSS_SOLVER_UPGRADES.md:177-178,424,769`). Both interpretations are reported:

- Current all-19 harness 533.96 s versus prompt-labelled 177 s: +356.96 s, +201.7% (3.02x).
- Current `0l4291i_live` ladder-summed row 347.585 s versus repository round-9b 177.7 s: +169.885 s, +95.6% (1.96x).
- Current `0l4291i_live` row versus reference pdspn 264 s: +83.585 s, +31.7% (1.32x).

These are not controlled benchmark deltas. Immediately before the identity run the host had 10.55 GiB free and low instantaneous Windows CPU, while `vmmemWSL` held 7.56 GiB with multiple WSL processes present: the requested shadow soak was resident. Phase A and flag-off repeated the same deterministic nodes but took 533.96 s and 539.49 s respectively, supporting load stability within this session, not comparability to a clean historical machine.

## 6. Artifacts and verification

Artifacts are under `.codex-b1/`:

- original PowerShell/Tee raw captures: `phase-a-tip-gate.log`, `phase-b-identity-off.log`, `phase-b-identity-on.log`;
- NUL-stripped UTF-8 views preserving the same captured text: matching `.utf8.log` files;
- normalized deterministic signatures: matching `.signature` files;
- `SHA256SUMS.txt` manifest.

The originals are retained because PowerShell appended native output as UTF-16LE to ASCII headers; the `.utf8.log` files remove only NUL bytes to make the capture searchable. No semantic line was edited.

Final checks:

- `cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq`: 107 passed, 0 failed, 36 ignored.
- `rustfmt --edition 2021 --check` on the two touched Rust files: pass.
- `git diff --check` on the two touched Rust files: pass.
- `packages/hexfield_eq/rust/src/tss_verify.rs` SHA-256 before and after: `9990D38618DA2204351E328CA0143BE2AEF98BB3001E4A0462CF346B707F2CE8`; byte-untouched.

