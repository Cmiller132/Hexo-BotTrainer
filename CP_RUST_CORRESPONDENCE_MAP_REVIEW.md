# R-CPRUST-1-REV — hostile review of `CP_RUST_CORRESPONDENCE_MAP.md`

## 1. Method and provenance

This was a hand-only, read-only source audit. I ran no Cargo command, Rust test, Lean command, checker, fixture harness, or game-search executable. I made no Rust or Lean source edit and no commit. Text searches, line extraction, revision checks, and revision-to-revision source diffs were the only inspection aids.

The reviewed map is the unmodified `CP_RUST_CORRESPONDENCE_MAP.md` at workspace HEAD `7ca56a7915b4a488dfe8456d6c7bb62c3673e2c2` on `hunt/completeness`. The required Rust files in this worktree have no diff from Rust baseline `78691bab9b001d637c0d370e1b58d2831518d525`; citations below therefore refer to that Rust source. The frozen Lean source was read only with `git show 0e1bdab:TssZones/CP1.lean` and `git show 0e1bdab:LEDGER_CP1.md` from `E:\tss-lean-cp1`. I did not read that repository's dirty worktree. The resolved Lean commit is `0e1bdabd7216ddbd2d979f05ddf49bd75337d832`.

Corpus inspected:

- `CP_RUST_CORRESPONDENCE_MAP.md`, `COMPLETENESS_CERT_SPEC.md`, and the obligation/profile clauses cited by the map in `COMPLETENESS_SPEC.md`;
- `packages/hexfield_eq/rust/src/{tss_solver,tss_verify,tss_core,tss_async,tss_reference}.rs`, `tree.rs`, `search.rs`, and their module gates in `lib.rs`;
- `packages/hexo_models/rust/src/threats_shared.rs` and the map-cited local HexGNN duplicate;
- `packages/hexo_engine/rust/src/{state,coord,legal,tactics}.rs`;
- additional call-graph evidence in `tss_bench.rs`, `tss_reference_fast.rs`, and the test/corpus modules named by the map;
- frozen `0e1bdab:TssZones/CP1.lean` and `0e1bdab:LEDGER_CP1.md`.

Severity convention: **REFUTED** means a material reviewed conclusion is false; **MAJOR** means a defect changes an obligation or architecture disposition; **MINOR** means a bounded factual, scope, citation, or inventory defect that does not change the disposition; **NOTE** records a confirmed claim or an important limitation on how it may be read. No REFUTED or MAJOR finding was found.

## 2. Numbered findings

### 1. NOTE — the v1 negative pipeline claim is correct, and the release seam is even less connected than the map states

> **Quoted claim:** “Rust has no v1 negative parser, checker, replay-table builder, functional/successful emitter, or sealed negative-result path; `try_emit_no_tss_v1` is an always-error seam.” (`CP_RUST_CORRESPONDENCE_MAP.md:7`)

**Independent source evidence.** `NoTssCertificateV1` is a private opaque byte box, and `RootRefutedCandidate` is expressly pending independent checker acceptance (`packages/hexfield_eq/rust/src/tss_solver.rs:2318-2338`). After the mandatory bottom-up refresh, only `root_entry.dn == 0` attempts emission, and only an `Ok` result could construct that candidate (`tss_solver.rs:4672-4689`). The emitter's exhaustive match returns `Err` for `Unexpanded`, `DepthCutoff`, both `Branch` cases, `Refuted`, `ProvenLeaf`, and `ProvenFragment`; there is no successful expression (`tss_solver.rs:4865-4891`; node variants at `tss_solver.rs:2622-2637`). The only direct byte construction is the invalid test fixture `b"NTSSCP1\0fixture"` (`tss_solver.rs:12911-12923`).

Release code also discards the complete `SearchStop` returned by `run`, and `AttemptResult.search_stop` exists only under `cfg(test)` (`tss_solver.rs:1155-1159,1805-1813`). The existing `TssVerifier` consumes a positive in-memory `TssCertificate`, not v1 bytes (`packages/hexfield_eq/rust/src/tss_verify.rs:106-153,171-227`). Its `ReplayMemo` is positive-node memoization based on indegrees and exact replay keys, not a v1 primary-parent replay-table builder or proposition-uniqueness service (`tss_verify.rs:322-461`). The Rust module/caller sweep found no other negative parser, decoder, checker, or mint.

**Assessment.** Confirmed. “Always-error” is literal for every current `WidePnNode` shape. Merely adding an `Ok` arm would still leave release transport, parsing, primary replay construction, independent checking, and minting absent. The claim should be read as “no sealed v1 `NO_CONTRACT_WIN` path”; Finding 7 addresses the summary's unqualified wording.

### 2. NOTE — default execution is not frozen CP1, and the two TSS-derived Loss meanings are not `NoContractWin`

> **Quoted claim:** “`TssSolver::default` leaves pair-complete width disabled. Production-authoritative Rust hard `-1`/Loss paths are a one-turn threat verdict ... or an opponent-positive certificate; neither is Lean `NoContractWin`.” (`CP_RUST_CORRESPONDENCE_MAP.md:7`)

**Independent source evidence.** `WidthOptions` derives `Default`, so its private `vcf_pair_complete: bool` is false; the explicit `WidthOptions::vcf_pair_complete()` constructor sets it true (`packages/hexfield_eq/rust/src/tss_solver.rs:569-574,620-627`). `TssSolver::default` installs `WidthOptions::default()` (`tss_solver.rs:688-700`), and false routes directly to `prove_narrow_compat` (`tss_solver.rs:1082-1111`). Production tree slots, the root guard, and the async worker instantiate defaults (`packages/hexfield_eq/rust/src/tree.rs:545-556,920-945`; `packages/hexfield_eq/rust/src/search.rs:4079-4095`; `packages/hexfield_eq/rust/src/tss_async.rs:545-583`). The verified wrapper changes solve caps/zone options, not width (`tree.rs:575-624`).

The direct hard producer wraps the one-turn threat analysis (`packages/hexfield_eq/rust/src/tss_core.rs:70-87`). That analysis returns `-1.0` exactly when there is no mover win-now and the opponent threat family cannot be hit within the placements remaining in the turn (`packages/hexo_models/rust/src/threats_shared.rs:46-90,137-183`). Deep `Loss` instead means the side to move loses and requires an opponent winning certificate (`tss_core.rs:24-45`): the dual search uses `root_player.other()` and returns `Loss` only with that positive certificate (`tss_solver.rs:1023-1054`), while verification requires the claimant to be the root mover's opponent (`tss_verify.rs:179-197`). The sole deep hard mint requires a present certificate accepted by the concrete positive verifier; failure becomes Unknown (`tss_core.rs:471-506`; `tree.rs:665-703`).

`tss_reference` can independently return minimax `Loss`, but it returns only `ReferenceResult`, has no certificate or hard mint, and its callers are test/corpus consumers (`packages/hexfield_eq/rust/src/tss_reference.rs:21-44,137-198`; module gates at `packages/hexfield_eq/rust/src/lib.rs:27-29`).

**Assessment.** Confirmed. Existing test-only setters can select pair-complete width, but no production query/profile constructor binds the executed producer to frozen CP1. Neither one-turn impossibility nor an opponent-positive strategy is the frozen proposition `Not (ContractWin R Q P)`.

### 3. NOTE — CP-O14 is genuinely open; the established defect is a process mismatch, not yet a locked missing-edge witness

> **Quoted claim:** “**Verdict: NOT DISCHARGED — the current solver violates the required both-order process** ...” (`CP_RUST_CORRESPONDENCE_MAP.md:340`)

**Independent source evidence.** The v1 contract freezes checker-owned raw signed edge order and requires both legal attacker-pair orders to be examined before quotient (`COMPLETENESS_CERT_SPEC.md:71-83`). Rust obtains one outer `first_candidates` list, derives `second_candidates(first)`, evaluates those encounters, deduplicates by an unordered key, and retains the first encountered orientation (`packages/hexfield_eq/rust/src/tss_solver.rs:6333-6376,6397-6435`). The second list contains strong promotions, the turn-start list, and fresh weak promotions, so a second coordinate need not itself be an outer first candidate (`tss_solver.rs:8777-8835`). There is no reverse evaluation in this loop. The source comment itself warns that candidate membership is non-monotone and only one ordering may be generated (`tss_solver.rs:6397-6401`).

Frozen Lean does not supply a hidden closure theorem: attack regeneration remains a field of an abstract `RegenerationBinding`, with only sorting, no-duplicate, and canonical-shape laws (`0e1bdab:TssZones/CP1.lean:308-339,354-366`). The frozen ledger explicitly leaves the both-order/correspondence work open (`0e1bdab:LEDGER_CP1.md:49-54`).

**Assessment.** The NOT DISCHARGED verdict is correct. The current source violates the literal required procedure and has no extensional theorem proving that evaluating one encountered orientation represents both order-sensitive classifications. Existing candidate and evaluator machinery could reduce the repair: a proof might reuse the evaluator plus explicit reverse probes and raw normalization. It does not already close CP-O14. The evidence here does not by itself prove that a particular semantic attack edge is missing from the final set; the map appropriately distinguishes that from the process defect.

### 4. NOTE — CP-O15 is genuinely open; the D6 fixture is a real list-order counterexample, not a quotient-completeness counterexample

> **Quoted claim:** “**Verdict: NOT DISCHARGED** — sound-looking local checks do not establish an exact quotient, and canonical list order is concretely divergent.” (`CP_RUST_CORRESPONDENCE_MAP.md:361`)

**Independent source evidence.** The planner constructs `completed_pair` keys by sorting the two extra stones, forcing the post-pair player/phase/terminal fields, and using saturating clock arithmetic (`packages/hexfield_eq/rust/src/tss_solver.rs:2991-3024`). Once the reverse directed pair exists, comparing its key with the forward key is therefore order-insensitive by construction, not independent evidence that both actual executions commute (`tss_solver.rs:3676-3695`). Actual forward replay is checked later during positive materialization (`tss_solver.rs:7167-7187`); the positive verifier replays both orders but compares only the final `Option<GameOutcome>`, not exact final state/cache equality (`packages/hexfield_eq/rust/src/tss_verify.rs:741-825`). No source theorem proves all sequential K2→K1 obligations, every `None` fallback, or synthetic-key/full-state equivalence.

The map's D6 derivation is reproducible from source. The fixture asserts K2 coordinates `a=(1,-6)`, `b=(3,-5)`, `c=(4,-6)` (`tss_solver.rs:12340-12373,12843-12868`). Reading the three six-cell axes (`packages/hexo_engine/rust/src/tactics.rs:21-50`) gives threat-empty sets `{(1,-7),a}`, `{a,(1,-1)}`, and `{b,c}`; recomputing K1 after each first therefore retains raw-low pairs `(a,b)` and `(a,c)`. The full-state canonical frame and transform are defined at `tss_solver.rs:9663-9736,9791-9800`; for this fixture symmetry 8 ranks `a,c,b`, and planner rank sorting consequently places `(a,c)` before `(a,b)` (`tss_solver.rs:3698-3720`). Frozen `Edge.CanonicalPair` requires raw-low coordinates and `Edge.before` compares the raw signed identity tuple, so it orders `(a,b)` before `(a,c)` (`0e1bdab:TssZones/CP1.lean:231-251`).

**Assessment.** The witness is genuine for direct ordered-`Vec` equality. It is not a counterexample to pair-set membership or semantic quotient completeness: v1 deliberately owns ordering independently of solver rank (`COMPLETENESS_CERT_SPEC.md:71-83`), and a raw `Edge.before` sorting adapter repairs this isolated mismatch. CP-O15 nevertheless remains open because the difficult converse, fallback theorem, transition equality, and synthetic-key correctness are absent. The map did not wrongly open the obligation, but its witness must not be cited as proving more than ordered-list divergence.

### 5. NOTE — CP-O27 remains a program-critical blocker; partial reusable Rust helpers do not instantiate the frozen contract

> **Quoted claim:** “**Verdict: BLOCKING / PROGRAM-CRITICAL, NOT DISCHARGED.**” (`CP_RUST_CORRESPONDENCE_MAP.md:383`)

**Independent source evidence.** Frozen `GrammarVersion` contains only an ID and profile, and `Query` stores that grammar separately from the semantic regeneration parameter (`0e1bdab:TssZones/CP1.lean:31-72,182-196`). `RegenerationBinding` still supplies root invariants, leaves, attack/defense lists, replay, and tight dispatch abstractly (`0e1bdab:TssZones/CP1.lean:308-339`). `ContractWin`, `NoContractWin`, and `checkNoDag` all take an independent `R` (and the checker also takes `X`) (`0e1bdab:TssZones/CP1.lean:446-459,2604-2614`). Accordingly, `checkNoDag_sound` proves `NoAt R ...` only for the supplied binding (`0e1bdab:TssZones/CP1.lean:2887-2896`). The frozen ledger expressly excludes Rust replay/generator correspondence (`0e1bdab:LEDGER_CP1.md:3-7,47-54`).

Rust does contain useful fragments: `tss_reference::legal_moves` and `direct_winner` rebuild legal cells and wins from occupancy (`packages/hexfield_eq/rust/src/tss_reference.rs:46-135`), `WindowStore::from_placements` rebuilds window masks (`packages/hexo_engine/rust/src/tactics.rs:429-449`), and the positive verifier memoizes exact replay keys for shared positive IDs (`packages/hexfield_eq/rust/src/tss_verify.rs:322-461`). None parses v1, reconstructs a complete phase/clock/root from unordered ownership entries, builds the required primary replay table, supplies exact CP1 attack/defense regeneration, or carries a refinement theorem. The existing state loader instead replays ordered history (`packages/hexo_engine/rust/src/state.rs:372-388`).

The frozen Lean codec is also only the strict node-record suffix: its own header says `NoCertificateDag` lacks the query/header/root fields and leaves the complete header/root codec and round-trip theorem as residual work (`0e1bdab:TssZones/CP1.lean:3225-3232`). The only landed round-trip is one four-node sharing fixture (`0e1bdab:TssZones/CP1.lean:3440-3467`). The frozen ledger separately lists the full 86-byte parser, general inverse, primary-parent replay-table construction, and executed correspondence as unclaimed (`0e1bdab:LEDGER_CP1.md:44-45,55-62`).

**Assessment.** The blocker verdict is correct. The frozen commit proves a checker parametrically in `R`; it does not freeze a concrete frontier merely by naming `CP1`. CP-O27 must connect bytes, checked state reconstruction, all global primitives, transitions, generator lists, profile selection, executed checking, and the unique mint to one concrete binding.

### 6. MINOR — the advertised 74-row exhaustive inventory is not exhaustive as cited

> **Quoted claim:** “the table enumerates every audited Rust path” and “N01–N17 cover every construction and outward consumption of `Loss`, `Refuted`, negative cache, or negative artifact found across production and test modules.” (`CP_RUST_CORRESPONDENCE_MAP.md:47,160`)

**Independent source evidence.** Several already-sealed deep `HardValue` consumers are not in N16's cited search spans: parked lockstep backup (`packages/hexfield_eq/rust/src/search.rs:2037-2048`), async-descent, memo-hit, and inline-deep lockstep backups (`search.rs:2333-2338,2357-2369,2401-2407`), their continuous-search equivalents (`search.rs:2475-2480,2497-2508,2542-2546`), and parked continuous backup (`search.rs:2674-2691`). Some incidentally fall inside N01's broad ranges, but N01 describes the one-turn producer, not these verified-deep consumers; others are outside both N01 and N16 citations.

C13 says it groups all `cfg(test)` certificate/evidence consumers but omits the `tss_bench` module declared at `packages/hexfield_eq/rust/src/lib.rs:38-39`. That module consumes raw solver `ProofStatus::Loss` (`packages/hexfield_eq/rust/src/tss_bench.rs:340-360`) and verified `Loss` (`tss_bench.rs:586-623`). C13's broad solver range stops at line 14793 even though a substantive ignored identity test continues at `tss_solver.rs:14817-15005`, and its verifier range stops at line 2330 before the final assertions at `tss_verify.rs:2331-2332`. C11's substantive “drops the certificate” claim is correct, but its citations stop before the decisive response fields and construction: `SolveResponse` contains binding/status/hard/counters and no certificate (`packages/hexfield_eq/rust/src/tss_async.rs:230-238`), and the worker constructs exactly those fields (`tss_async.rs:603-613`).

**Assessment.** The completeness claim is false as a source enumeration/citation claim. The missing paths are test-only or consumers of an already sealed positive result; they expose no new v1 negative authority and can be absorbed into the existing N16/C13 categories. The 74 semantic-category count need not change, but the artifact must not call its present spans exhaustive.

### 7. MINOR — two gate-summary phrases need explicit v1/TSS scope

> **Quoted claim:** “no sealed negative-result path” and “Production-authoritative Rust hard `-1`/Loss paths are ...” (`CP_RUST_CORRESPONDENCE_MAP.md:7`)

**Independent source evidence.** Rust does have a sealed negative-valued `HardValue` path: `HardValue` has a private field, the one-turn constructor can return negative, and the positive-certificate verifier can mint negative after accepting an opponent certificate (`packages/hexfield_eq/rust/src/tss_core.rs:47-76,471-506`). It also has ordinary game-terminal `-1.0` (`packages/hexfield_eq/rust/src/tree.rs:3168-3173`), consumed separately from TSS hard values (`packages/hexfield_eq/rust/src/search.rs:2328-2338,2470-2480`).

**Assessment.** In context, the map means “no sealed v1 `NO_CONTRACT_WIN` result path” and “TSS-derived hard `-1`/`ProofStatus::Loss` paths”; it later expressly excludes ordinary terminal reporting (`CP_RUST_CORRESPONDENCE_MAP.md:166`). The principal claim is not substantively wrong, but the summary is technically overbroad without those qualifiers.

### 8. MINOR — R02 omits an overflow sub-seam directly inside checker-critical window reconstruction

> **Quoted claim:** R02 identifies unchecked `i16` arithmetic through coordinate/legal-radius generation and gives a radius-overflow witness. (`CP_RUST_CORRESPONDENCE_MAP.md:144,306`)

**Independent source evidence.** `HexCoord::scale`, addition, subtraction, negation, distance, and radius enumeration all use unchecked `i16` arithmetic (`packages/hexo_engine/rust/src/coord.rs:27-39,43-95`). Independently of legal-radius enumeration, every placement constructs 18 `WindowKey`s using unchecked scale/subtraction (`packages/hexo_engine/rust/src/tactics.rs:459-499,526-528`), and later enumerates window cells using unchecked scale/addition (`tactics.rs:64-75`). Those keys feed terminal and threat reconstruction.

**Assessment.** The map finds the representation-domain problem but its inventory/witness discussion does not enumerate this checker-critical B2 path. A boundary coordinate can affect the rebuilt window/threat proposition even if legal generation is replaced. O27-L1/L2/L3's checked arithmetic and global rebuild plan would close it, so this is a missing sub-seam in the inventory rather than a changed verdict.

### 9. MINOR — the “sharpest” non-vacuity headline omits B3

> **Quoted claim:** “The single sharpest endgame risk remains **B2/B4/B5 non-vacuity** ...” (`CP_RUST_CORRESPONDENCE_MAP.md:585`)

**Independent source evidence.** In frozen Lean, `regenerateStateLeavesGlobally`, `regenerateEdgeLocalShapesGlobally`, and `replayPendingEdge` are just as external as attack and defense regeneration (`0e1bdab:TssZones/CP1.lean:314-320`). They directly determine state-leaf, edge-local, and recursive positive constructors (`0e1bdab:TssZones/CP1.lean:404-443`), and the DAG checker consumes them while validating dispositions and node bodies (`0e1bdab:TssZones/CP1.lean:2480-2501,2530-2560`). Empty or incorrectly narrow leaf/edge-local lists and a too-restrictive replay relation can make `NoContractWin R` concern the wrong positive grammar even if B2/B4/B5 are exact.

**Assessment.** The sharpest-risk label should be **B2/B3/B4/B5 non-vacuity**. The map defines B3 (`CP_RUST_CORRESPONDENCE_MAP.md:209`) and includes it in the later dependency plan, so this is an emphasis/prioritization defect, not a missing overall gate.

### 10. MINOR — grammar-ID-to-binding dispatch and all non-profile producer switches need explicit invariants

> **Quoted claim:** B8 binds the executed query to `CP1`/`frozenProfile`, while O27-L9B later constructs “the single concrete `R : RegenerationBinding` and `X : ExecutableRegenerationBinding R`.” (`CP_RUST_CORRESPONDENCE_MAP.md:214,398-399`)

**Independent source evidence.** Frozen `GrammarVersion` contains no `R`, `Query` contains only `grammar/root/claimant/horizons`, and `checkNoDag` receives `R/X` independently (`0e1bdab:TssZones/CP1.lean:60-72,182-196,2604-2614`). The byte policy says any generator, edge-order, or semantic change requires a new grammar ID (`COMPLETENESS_CERT_SPEC.md:199-201`). Therefore exact query bytes alone do not prove which concrete regeneration binding the executed API selected.

There is a related operational switch family not explicitly named in B8/L9: `ZoneSearchCaps` has four booleans (`packages/hexfield_eq/rust/src/tss_core.rs:110-116`), and production root callers populate them from runtime configuration (`packages/hexfield_eq/rust/src/search.rs:4083-4095`). They change zone candidates and commutation behavior (`packages/hexfield_eq/rust/src/tss_solver.rs:8029-8045,8077-8087,9267-9303`), while the normative initial matrix places zone/narrow-compatible profiles outside CP1 (`COMPLETENESS_SPEC.md:643-660`).

**Assessment.** Add a named API/refinement invariant: bytes naming `CP1-a49e8abd-v1` invoke exactly one pinned declarative `R_CP1/X_CP1`, and any semantic change changes the ID; separately, the producer manifest binds every non-profile switch, including all `ZoneSearchCaps`, or proves it irrelevant to emitted CP1 bytes. O27-L9B/X1 and artifact pinning largely anticipate the first condition, and C07/V12 recognize zone divergence, so this is under-specification rather than a contradictory architecture.

### 11. NOTE — the recommended architecture is sound only as the full stated refinement program, not as the diagram alone

> **Quoted claim:** untrusted bytes flow through “strict bounded v1 parser → exact external-root match + checker-local rebuild → canonical B2/B4/B5 regeneration + primary replay table → checkNoDag-equivalent checker → sealed `NO_CONTRACT_WIN`.” (`CP_RUST_CORRESPONDENCE_MAP.md:421-455`)

**Independent source evidence.** The frozen theorem is universally parameterized by the supplied `R/X` (`0e1bdab:TssZones/CP1.lean:2604-2614,2887-2896`), and attack/defense are definitionally whatever that `R` returns (`0e1bdab:TssZones/CP1.lean:354-360`). A parser, a second implementation of regeneration, and Boolean equality with `checkNoDag` would therefore be insufficient if `R` were simply defined from that same implementation. Conversely, the map's detailed plan requires independently declarative attack/defense relations, separate checker and producer refinements, concrete `R/X` assembly, exact-index/replay refinement, and a unique mint (`CP_RUST_CORRESPONDENCE_MAP.md:346-353,367-375,389-407,463-465,577-583`).

**Assessment.** As a target architecture, this is conditionally sound and does not merely relocate the non-vacuity problem—provided those detailed obligations are literal acceptance gates, B3 and Finding 10's ID→`R/X` dispatch are included, and “independent regeneration” means equality to a separately defined declarative frontier rather than implementation diversity alone. The short diagram by itself does relocate the risk; the subsequent proof plan is what resolves it. None of this architecture exists at the audited revisions.

## 3. Requested gap-inventory adjudication

| Map item | Review disposition | Independent source basis |
|---|---|---|
| N03 | **Supported.** The exhaustive reference can say `Loss`, but emits no certificate/`HardValue`; callers found are test/corpus uses. It is reusable comparison machinery, not current authority. | `packages/hexfield_eq/rust/src/tss_reference.rs:21-44,137-198`; `packages/hexfield_eq/rust/src/tss_solver.rs:13553-13732`; `packages/hexfield_eq/rust/src/tss_spare_corpus.rs:735-736,1163-1178,1255` |
| N04 | **Supported.** The fast reference is explicitly test-gated and is not a v1 result route. | `packages/hexfield_eq/rust/src/lib.rs:27-29`; `packages/hexfield_eq/rust/src/tss_reference_fast.rs:88-234` |
| C07 | **Supported, with an exclusion option.** It reconstructs positive-certificate zone labels, not frozen negative syntax. A proved CP1 profile/call-graph exclusion could close this row without modeling it in `R`; no production CP1 binding currently supplies that exclusion. | `packages/hexfield_eq/rust/src/tss_solver.rs:1623-1710`; ranked-zone switch off at `0e1bdab:TssZones/CP1.lean:47-58`; general zone profiles excluded by `COMPLETENESS_SPEC.md:643-660` |
| C09 | **Supported.** The public D6 shaper transforms the positive certificate schema. Current callers found are tests; it is neither a v1 codec nor a raw-order theorem. It may be excluded from the negative TCB by a pinned call graph. | `packages/hexfield_eq/rust/src/tss_verify.rs:1515-1700,2315-2332`; test caller at `packages/hexfield_eq/rust/src/tss_solver.rs:13901` |
| C10 | **Supported.** Positive preflight derives leaf/completion `T` and a zone bit, and tree may retry; this is not the v1 header/H binding. | `packages/hexfield_eq/rust/src/tss_verify.rs:229-320`; `packages/hexfield_eq/rust/src/tree.rs:625-650` |
| C11 | **Substance supported; citation repair required.** Async returns only binding/status/hard/counters and therefore drops the positive certificate. | `packages/hexfield_eq/rust/src/tss_async.rs:230-238,573-613` |
| C13 | **Inventory claim not supported.** The grouped category is valid, but its claimed exhaustive spans omit `tss_bench` Loss consumers and truncate live test code. | `packages/hexfield_eq/rust/src/lib.rs:38-39`; `packages/hexfield_eq/rust/src/tss_bench.rs:340-360,586-623`; `packages/hexfield_eq/rust/src/tss_solver.rs:14817-15006`; `packages/hexfield_eq/rust/src/tss_verify.rs:2315-2334` |
| T09 | **Supported.** The candidate carrier is semantically pending and unreachable through the all-error emitter; the only direct carrier fixture contains invalid synthetic bytes. | `packages/hexfield_eq/rust/src/tss_solver.rs:2318-2379,4672-4689,4865-4891,12911-12923` |
| V12 | **Supported.** This rederives the positive zone grammar; it is outside the initial CP1 matrix and is not negative global regeneration. | `packages/hexfield_eq/rust/src/tss_verify.rs:1019-1276`; `COMPLETENESS_SPEC.md:643-660` |
| V13 | **Supported.** It recomputes positive tight-dispatch/kernel prerequisites, while its optional oracle does not establish exact CP1 represented-list equality. | `packages/hexfield_eq/rust/src/tss_verify.rs:1280-1349,159-168,938-961` |
| G06 | **Supported.** The budget-one/budget-two kernel code is concrete and useful, but frozen Lean still exposes only an abstract defense list; no equivalence theorem is present. | `packages/hexfield_eq/rust/src/tss_solver.rs:9130-9197`; `0e1bdab:TssZones/CP1.lean:308-339,358-366` |

The requested list therefore contains no overlooked existing bridge. C07/C09/C10 and the reference/test paths may ultimately be discharged by precise non-authority/profile/call-graph exclusion rather than full semantic correspondence. C11 needs better citations, and C13 is the one requested inventory row whose exhaustiveness claim fails.

## 4. Per-obligation and architecture verdicts

| Subject | Map verdict | Review disposition | Reason |
|---|---|---|---|
| CP-O14 | NOT DISCHARGED | **UPHELD — NOT DISCHARGED** | The frozen contract requires both legal attacker orders before quotient; Rust's outer-first/per-first-second process can generate only one order, retains first orientation, and has no exact ordered-list/refinement theorem. Existing evaluator/candidate machinery makes repair plausible but is not a discharge. |
| CP-O15 | NOT DISCHARGED | **UPHELD — NOT DISCHARGED** | The D6 fixture really refutes direct list equality, though a sort adapter fixes only that symptom. Sequential quotient completeness, every planner fallback, actual two-order final-state equality, and synthetic-key correctness remain unproved. |
| CP-O27 | BLOCKING / PROGRAM-CRITICAL | **UPHELD — BLOCKING / PROGRAM-CRITICAL** | No v1 parser/root builder/primary replay/checker/emitter/mint chain or Rust-to-model theorem exists; production profile selection also differs. Frozen Lean is parametric in an uninstantiated `R/X`. |
| Recommended architecture | Open target | **CONDITIONALLY SOUND, NOT IMPLEMENTED** | It closes the identified boundary only if it uses an independently defined concrete CP1 frontier, proves B2/B3/B4/B5, binds grammar ID and all runtime switches to that `R/X`, proves exact replay/index and compiled-checker equality, and makes the checker the unique mint. Without those capstones it merely moves the abstract-generator risk. |

## 5. Overall verdict

**The map's principal conclusion holds: the Rust↔Lean correspondence boundary is not closed.** The audited source supports the absence of a functional v1 negative pipeline, the always-error emitter seam, the default narrow/non-pair-complete production profile, and the non-equivalence of existing Rust Loss values to frozen Lean `NoContractWin`.

No CP-O14, CP-O15, or CP-O27 obligation was wrongly declared open or closed. The D6/raw-order witness is not an artifact of misreading the frozen contract, but its force is limited to ordered-list equality; it does not independently establish a missing semantic quotient edge. Partial reference/global-rebuild helpers reduce future implementation work but do not close any of the three obligations.

The most severe defect in the map is bounded: its 74-row exhaustive-inventory claim is false as written. The review also adds three closure details that should be explicit in the campaign gate: B3 belongs in the non-vacuity headline; the grammar ID must select exactly one pinned concrete `R_CP1/X_CP1`; and checker-critical window arithmetic plus all non-profile runtime switches, including `ZoneSearchCaps`, must be covered by checked reconstruction/profile refinement. These corrections do not alter the boundary-not-closed verdict.
