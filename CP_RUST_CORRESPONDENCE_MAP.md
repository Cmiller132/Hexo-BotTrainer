# Rust ↔ Lean correspondence map for CP-O14, CP-O15, and CP-O27

Status: source audit at Rust HEAD `78691bab9b001d637c0d370e1b58d2831518d525`; Lean stable contract at `E:\tss-lean-cp1` HEAD `0e1bdabd7216ddbd2d979f05ddf49bd75337d832`. That Lean worktree is currently dirty from the concurrent R-CP9 full-file codec, which is explicitly treated as in flight rather than landed. This document was produced by read-only source analysis. No Cargo command, Lean build, Rust/Lean source edit, or commit was used.

## Gate summary

The correspondence boundary is **not closed**. The stable Lean work has a proved negative tree checker, a proved shared-DAG checker, and stable node-stream codec declarations with one proved fixture round-trip, but no general node-stream inverse theorem. Rust has no v1 negative parser, checker, replay-table builder, **functional/successful** emitter, or sealed negative-result path; `try_emit_no_tss_v1` is an always-error seam. The live Rust production profile is also not bound to the frozen CP1 profile: `TssSolver::default` leaves pair-complete width disabled. Production-authoritative Rust hard `-1`/Loss paths are a one-turn threat verdict (shared in most lineages but locally duplicated in HexGNN) or an opponent-positive certificate; neither is Lean `NoContractWin`. N03/N04 separately retain non-authoritative reference-solver Loss values.

The inventory below has **74 countable surface rows**. The row is the declared audit/counting unit used in the final class totals. A row may aggregate multiple construction sites when the row documents one operational category or trust boundary; C13 deliberately aggregates test-only certificate fixtures/evidence consumers. All ten `SearchStop` and all six `RunUntilExit` variants remain separate rows because each is a distinct production stop vocabulary case.

### Source aliases

| Alias | Path |
|---|---|
| `SOL` | `packages/hexfield_eq/rust/src/tss_solver.rs` |
| `VER` | `packages/hexfield_eq/rust/src/tss_verify.rs` |
| `CORE` | `packages/hexfield_eq/rust/src/tss_core.rs` |
| `TREE` | `packages/hexfield_eq/rust/src/tree.rs` |
| `SEARCH` | `packages/hexfield_eq/rust/src/search.rs` |
| `ASYNC` | `packages/hexfield_eq/rust/src/tss_async.rs` |
| `REF` | `packages/hexfield_eq/rust/src/tss_reference.rs` |
| `SHARED` | `packages/hexo_models/rust/src/threats_shared.rs` |
| `STATE` | `packages/hexo_engine/rust/src/state.rs` |
| `COORD` | `packages/hexo_engine/rust/src/coord.rs` |
| `LEGAL` | `packages/hexo_engine/rust/src/legal.rs` |
| `TACTICS` | `packages/hexo_engine/rust/src/tactics.rs` |
| `CP1` | `E:\tss-lean-cp1\TssZones\CP1.lean` |
| `CERT` | `COMPLETENESS_CERT_SPEC.md` |

Line spans refer to the audited revisions, not to future edits.

## 1. Surface enumeration

### 1.1 Enumeration architecture (stated before the completeness claim)

The first sweep followed authority from entry points, not names:

1. module gates in `packages/hexfield_eq/rust/src/lib.rs:18-40`;
2. the production deep entry `TREE:559-705` into `TssSolver::solve_goal` (`SOL:842-1057`);
3. the wide and narrow search routes (`SOL:1070-1365`, `SOL:4281-4370`), their generators, typed leaves, proof-number update, and materializers;
4. every positive certificate constructor/mutator into `TssVerifier::verify` (`VER:171-227`), then every accept/reject helper and the sealed hard-value mint (`CORE:471-506`);
5. every arm of the closed `SearchStop`, `RunUntilExit`, and `StageEvent` types and every place those values are retained or discarded;
6. outward consumers through tree/search/async and the Rust-to-Python-to-NPZ trace.

The call-graph walk was cross-checked with repository-wide searches for constructors and decision tokens: `ProofStatus::Loss`, `HardValue`, `Refuted`, `LOCAL_TT_FAILED`, `dn == 0`, `NoTssCertificateV1`, `TssCertificate {`, `CertNode::`, `SearchStop::`, `RunUntilExit::`, `StageEvent::`, `return None`, `return false`, `check`, `verify`, `encode`, `decode`, `bytes`, `exhaust`, and `negative`. The independent second sweep and its additions are recorded in §4 rather than being used silently here.

The completeness claim is therefore scoped precisely: the table enumerates every audited Rust path that (a) creates or consumes a negative/refuted fact, (b) constructs, mutates, transports, or serializes a TSS proof artifact, (c) stops the wide search, or (d) accepts/rejects at the strict verifier. It also includes the state, key, flag, and generator surfaces needed to judge CP-O14/15/27. Test-only paths are retained and labeled; non-authoritative sibling consumers are grouped, not omitted.

### 1.2 Negative, loss, and internal-refutation surfaces (N01–N17)

| ID | Rust span | Exact predicate or state transition | Authority and obligations |
|---|---|---|---|
| N01 | `SHARED:46-90,93-183,245-257`; `CORE:70-87`; `SEARCH:2318-2354,2460-2496,4591-4618`; `packages/hexgnn/rust/src/threats.rs:38-72,119-160` | `ThreatAnalysis::verdict = Some(+1)` if the mover has a win-now window; otherwise `Some(-1)` exactly when the opponent threat family has no hitting set within the mover's remaining placements; otherwise `None`. `solve_leaf_lambda1` seals the shared verdict as `HardValue`; `lambda1_status` separately classifies it as `ProofStatus`. The search backup paths and root-move classifier consume the same λ1 predicate. HexGNN has a local formula-shaped implementation over `windows().threats()` rather than the shared `live_threat_entries()` implementation; their extensional equality is unproved. | Production hard `-1`, but one-turn only. CP-O3, O27, O28; it is **not** CP1 generator exhaustion. |
| N02 | `SOL:940-985,1023-1054,1796-1802,1874-1880`; `CORE:24-45` | Immediate-root preflight may return `ProofStatus::Loss` for an opponent-positive one-node certificate under `SolveGoal::Loss` or `Both`, regardless of pair-complete width. After preflight, `SolveGoal::Loss`, and `SolveGoal::Both` only when `width.vcf_pair_complete` is false, run the separate opponent-positive attempt. Every returned Loss carries an opponent-positive `TssCertificate` and maps to `-1`; pair-complete `Both` disables only the later dual attempt. | Production deep/immediate loss. CP-O3, O27, O28. Not `NoContractWin`. |
| N03 | `REF:21-44,137-198` | At a nonterminal nonempty root with positive remaining horizon, independent depth minimax returns `Loss` when every legal root child recursively returns Loss; opponent terminals also return Loss. Horizon zero or an empty legal list returns Unknown rather than using vacuous all-children loss. | Production-compiled reference, but caller sweep found test/corpus use only; no certificate or hard mint. CP-O27 comparison evidence, not authority. |
| N04 | `packages/hexfield_eq/rust/src/tss_reference_fast.rs:88-125,131-234`; `packages/hexfield_eq/rust/src/lib.rs:28-29` | Fast reference analog returns Loss from bounded exhaustive minimax, with its solve entry able to receive an explicit root player. | `cfg(test)` only. CP-O27 differential oracle candidate. |
| N05 | `SOL:2410-2455,2622-2637,5546-5565,5969-6039,11295-11297,11364-11365,11455-11466,11749-11780` | `WidePnNode::Refuted` has `(pn,dn)=(INF,0)`; Choice uses min/sum and Universal sum/min over children. A Branch whose recomputed `dn` is zero is numerically treated as refuted but retains its Branch tag. Empty Choice is `(INF,0)`; empty Universal is `(0,0)`, though expansion normally intercepts emptiness. Unit fixtures also construct Refuted nodes/children directly to test recurrence. | Internal proof-number fact only; direct fixture constructors are test data, not new semantic causes. CP-O18, O20–O24, O27, O28. |
| N06 | `SOL:2237-2242,2622-2637,6125-6133` | A node deeper than the current staged cap becomes `DepthCutoff`; it deliberately shares numeric `(INF,0)` with Refuted while retaining a distinct tag. | Incomplete, never negative evidence by number alone. CP-O22, O28. |
| N07 | `SOL:1113,2184-2200,4634-4644,6125-6138` | The source contains a post-cutoff check that installs Refuted if semantic clock `placements_made > H`. In production, every staged depth cap is at most `min(H-root_ply, MAX)`, so an admitted root reaches `DepthCutoff` first; the Refuted branch is reachable only if a test or future caller constructs an inconsistent `WidePnSearch`. | Intended out-of-contract closure is currently production-dead and still needs a reachability theorem. CP-O13, O16, O27. |
| N08 | `SOL:6169-6176` | Any already-terminal expanded position, regardless of winner, becomes Refuted. | Claimant-terminal correctness depends on the separate invariant that every such successor was caught edge-locally. CP-O3, O16, O27. |
| N09 | `SOL:6178-6198` | At any non-Opening node, if threat analysis names the claimant as winner, an available typed immediate leaf with resolution `<= H` yields ProvenLeaf and absence/shape failure or excess resolution yields Refuted; a named winner other than the claimant also yields Refuted. This applies regardless of whose turn the state records. | Typed-leaf/global replay boundary. CP-O3, O16, O27. |
| N10 | `SOL:121-214,858-865,6202-6216` | When the environment-derived optional census gate is enabled **and** `state.current_player() == claimant`, `evaluate_interior_census_gate(...).is_some_and(|evaluation| evaluation.dismiss)` makes the interior position Refuted. | Outside frozen CP1 initial flags; still production-compiled/env-controlled. CP-O20, O27. |
| N11 | `SOL:6221-6233` | A defender node not at the exact tight-dispatch boundary becomes Refuted. | Tight dispatcher must correspond exactly. CP-O15, O16, O27. |
| N12 | `SOL:6249-6254` | Expansion producing no children becomes Refuted. | Sound only if the relevant generator was exact and exhaustive. CP-O14, O15, O16, O18, O27. |
| N13 | `SOL:5273-5372,5539-5544` | If `apply_with_delta` cannot apply the selected child move, the child is changed to `WidePnChildResult::Refuted`; parent proof numbers then consume it. A mismatch between the recomputed future key and `child.future_key` is only debug/test asserted on the single and pair paths at `SOL:5286-5295,5361-5370`; a release build otherwise trusts and inserts the stored future key. | Concrete semanticization of move-application failure, plus a separate unchecked future-key premise for B6. CP-O14, O15, O27, O28. |
| N14 | `SOL:6567-6620` | In ordinary defender generation, a replayed terminal won by claimant is `ClaimantCompletion`; any other terminal is `Refuted`; nonterminal is Pending. | Edge-local terminal tagging. CP-O3, O15, O16, O27. |
| N15 | `SOL:489,4281-4370,7666-8155` | Narrow compatibility search collapses cap/depth/terminal/leaf/generator/replay failures to `None`; `LOCAL_TT_FAILED = CertNodeId::MAX` records a selected subset of no-proof returns and makes later calls return `None`. `AttemptResult.search_stop` is always `None`. | Production-compiled, selected whenever base-wide profile is not active; outside frozen CP1. CP-O20–O27, O28. |
| N16 | `TREE:559-705,1300-1580`; `SEARCH:4043-4226,4591-4618`; `CORE:471-506` | A deep hard `-1` exists only after the positive verifier accepts an opponent certificate. Root search merges λ1 and deep scalar values, filling zero with deep result. Tree failures degrade to Unknown. | Actual production semantic boundary. CP-O27, O28. |
| N17 | `SOL:2318-2407,4634-4752,4865-4891,12911-12923` | Negative artifact is opaque bytes. Refreshed `dn==0` calls `try_emit_no_tss_v1`; every current production node shape returns an error, so `RootRefutedCandidate` is unreachable through the production run/emitter. A `cfg(test)` stop fixture constructs synthetic invalid bytes directly. | Fail-closed production negative seam; the fixture is observation data only. CP-O17–O19, O27–O29. |

Additional non-authoritative consumers of the **shared** N01 implementation are `packages/hexfield/rust/src/search.rs:1939-1945,2018-2024,3580-3591`, `packages/hexo_models/hexgt/rust/src/mcts.rs:626-636,928-940`, and `packages/hexo_models/dense_cnn/rust/src/mcts.rs:1340-1357,1434-1440,2098-2110`. HexGNN's consumers are `packages/hexgnn/rust/src/mcts.rs:717-727,1019-1031`, but they call its local implementation cited in N01. All emit the same *nominal* one-turn scalar shape and no CP1 certificate; the unproved shared/local extensional equality is explicitly part of N01 rather than silently assumed.

### 1.3 Certificate construction, shaping, transport, and serialization (C01–C13)

| ID | Rust span | Exact operation | Obligations |
|---|---|---|---|
| C01 | `VER:39-153` | Defines positive in-memory `RootBinding`, `CertNode::{OrCompletion,Win,Loss,Choice,Universal}`, and `TssCertificate`. `CertNode::Loss` is a tactical leaf in a proof that the certificate claimant wins. | CP-O3, O17–O19, O27, O29. |
| C02 | `SOL:940-972,1882-2079` | Preflight immediate leaf construction: accepts exact typed leaf/resolution under goal/horizon, creates a one-node positive certificate, or records precondition rejection. | CP-O3, O13, O16, O27. |
| C03 | `SOL:1237-1290,6801-7293,7295-7317` | Wide materializer requires root `pn==0`; chooses a proved child at Choice, all obligations at Universal, replays moves, expands pair edges, imports leaves/fragments, then assembles/compacts/rebases a positive certificate. Failure returns no certificate. | CP-O8, O14–O18, O20–O27. |
| C04 | `SOL:4281-4370,7666-8160` | Narrow recursive search directly builds positive Choice/Universal certificates; failure collapses to `None`; optional zone extras are added at `SOL:9267-9385`. | Outside frozen base-wide CP1; CP-O20–O27. |
| C05 | `SOL:7132-7266`; `VER:741-825` | A batched defender pair is expanded into sequential nested Universals plus explicit commutation evidence. The materializer replays only the retained raw-low first→second ordering while building nodes (`SOL:7167-7196`) and records reverse commutation metadata (`SOL:7244-7255`). The verifier alone replays both orders, requires nonterminal SecondStone intermediates, and compares the two second-placement `Option<GameOutcome>` values, not exact final `RootBinding`/cache state. | CP-O15, O18, O27. |
| C06 | `SOL:1292-1335,1717-1788,6883-6920,10032-10230` | Shared-fragment import, dominant-label rebasing, and cached-proof import reshape positive certificate subgraphs; promotion requires a strict positive re-verification. | Frozen CP1 disables shared fragments; CP-O8, O19, O24, O26, O27. |
| C07 | `SOL:1623-1710` | Recomputes zone distances/labels for a positive certificate and fails if reconstruction is invalid. | CP-O12, O13, O27. |
| C08 | `SOL:10057-10125,10646-10828` | Offsets/remaps IDs and compacts reachable positive graph in child-before-parent postorder, rejecting cycles/range/resource excess. Cached proof requires `child < parent`. | CP-O17–O19, O27, O29. |
| C09 | `VER:1515-1700` | D6-remaps every coordinate, window, root phase field, zone, and commutation while preserving node IDs/player/clock; rejects invalid transformed artifacts. | Production-public shaper, caller sweep found test use; CP-O9, O27. |
| C10 | `VER:237-320`; `TREE:625-650` | `certificate_horizon_preflight` derives `(max leaf/completion resolution, has_any_zone)`. Full metadata separately records the minimum zone build horizon. Tree retries one zoned mismatch at the derived leaf/completion `T`; a second mismatch becomes Unknown. | CP-O13, O27, O29. |
| C11 | `ASYNC:212-230,545-600`; `packages/hexfield_eq/rust/src/lib.rs:33` | Async worker invokes verified solve, then transports binding/status/hard value/counters but drops the certificate. It cannot mint authority independently. | CP-O27, O28. |
| C12 | `SEARCH:889,3150,3731-3850,4043-4226`; `packages/hexfield_eq/python/hexfield_eq/selfplay.py:439-555,608-625`; `packages/hexfield_eq/python/hexfield_eq/samples.py:167-185,222-240`; `packages/hexfield_eq/python/hexfield_eq/shards.py:44,101-144,216-251,390-425`; `packages/hexfield_eq/python/hexfield_eq/window.py:427-430` | λ1 or deep result is collapsed to signed `tss_proof: i8`, copied to Python, and stored as schema-v5 NPZ `int8`. Both native payload paths use the conversion. No certificate, root, horizon, grammar, source kind, or stop survives; missing/old data reads as zero; `.hxr` and expanded training rows omit the field. | Exact emitted-trace serialization named by CP-O27; also O28. |
| C13 | `SOL:1369-1617,10830-14793`; `VER:1703-2330`; `packages/hexfield_eq/rust/src/tss_cap_resume.rs:108-119`; `packages/hexfield_eq/rust/src/tss_corpus.rs:413,715`; `packages/hexfield_eq/rust/src/tss_leaf_surface_hunt.rs:420,667-675`; `packages/hexfield_eq/rust/src/tss_k_reply_shadow.rs:293-350,792,848`; `packages/hexfield_eq/rust/src/tss_pn_init_hunt.rs:642,697,1171-1253`; `packages/hexfield_eq/rust/src/tss_spare_corpus.rs:1188,1238,1289,1392-1413,1571-1617`; `packages/hexfield_eq/rust/src/tss_turn_quotient_hunt.rs:723,914,939-1100` | Groups all `cfg(test)` certificate fixtures, cap-resume assembly, fingerprints, mutations, preflights, and direct verifier/oracle consumers. Comments state the fingerprint encoders are not a public wire codec; the only negative fixture bytes are `NTSSCP1\0fixture`. | Test evidence only. CP-O17, O27–O29. |

### 1.4 Search termination taxonomy (T01–T17)

The declarations are `StageEvent` at `SOL:2244-2252`, six-way `RunUntilExit` at `SOL:2254-2265`, and ten-way `SearchStop` at `SOL:2327-2368`. Exact construction/precedence is `SOL:4634-4752,4808-4863`; preflight/immediate constructions are `SOL:916-972`. `HUNT_REPORT_CP4_SEAM.md:18-39,54-75,80-123` documents the landed seam and its fail-closed intent.

| ID | Variant | Rust span | Exact trigger / meaning | Obligations |
|---|---|---|---|---|
| T01 | `StageEvent::SelectedCutoff` | `SOL:4712-4726,4791-4803` | A selected cutoff is recorded under `cfg(test)` before requesting staged deepening; failure to compute a strictly larger bounded stage instead terminates `NonAdvancingCutoff`. Not a verdict. | CP-O22, O27, O28. |
| T02 | `RunUntilExit::RootPnZero` | `SOL:4823-4829,4672-4675,4741-4744` | After recompute, root `pn==0`. Outer `run` refreshes again and must see RootProven or reports invariant failure. | CP-O18, O20–O24, O27. |
| T03 | `RunUntilExit::RootDnZero` | `SOL:4823-4832,4676-4689,4735-4739` | After recompute, root `dn==0`. Outer `run` may attempt negative emission; the number itself is never a verdict. | CP-O17–O24, O27–O29. |
| T04 | `RunUntilExit::NodeCap` | `SOL:4817-4821,4690-4703` | Either global or invocation expansion cap reached; carries exact expansions and minimum cap. | CP-O21, O28. |
| T05 | `RunUntilExit::SelectedCutoff` | `SOL:4833-4839,4705-4727,4788-4803` | `work` returns a selected cutoff and staged deepening is enabled. | CP-O22, O28. |
| T06 | `RunUntilExit::CutoffNoProgress` | `SOL:4841-4854,4728-4733` | Selected path reaches a cutoff with `made_progress=false` when no further deepening is requested. | CP-O22, O28. |
| T07 | `RunUntilExit::Stalled` | `SOL:4823-4826,4856-4859,4734` | Missing root entry or `work` reports no progress and no categorized cutoff. | CP-O20–O24, O28. |
| T08 | `SearchStop::RootProven` | `SOL:940-965,4672-4675,1237-1290,12913` | Immediate-leaf preflight constructs a positive proof at stage 0, but its SearchStop push is `cfg(test)` only. The wide run constructs RootProven after mandatory refresh finds root `pn==0`; release callers discard that stop, and later positive materialization may fail. The last span is a direct fixture. | CP-O18, O20–O24, O27, O28. |
| T09 | `SearchStop::RootRefutedCandidate` | `SOL:2370-2379,4676-4684,12915-12923` | Mandatory refresh finds `dn==0` **and** future `try_emit_no_tss_v1` returns complete bytes plus structural boundary count. It is unreachable from the current production emitter; the last span directly constructs an invalid-byte `cfg(test)` fixture. | CP-O17–O19, O27–O29. |
| T10 | `SearchStop::NodeCap` | `SOL:4690-4703,12925-12931` | `RunUntilExit` reports cap or refreshed expansion counter meets cap. Public class is Unknown/Capped; the last span is a direct fixture. | CP-O21, O28. |
| T11 | `SearchStop::CutoffNoProgress` | `SOL:4728-4733,12933-12938` | Like-named `RunUntilExit` after refresh. Public class Unknown/Incomplete; the last span is a direct fixture. | CP-O22, O28. |
| T12 | `SearchStop::NonAdvancingCutoff` | `SOL:4717-4726,12940-12945` | Selected cutoff cannot yield a strictly larger bounded stage. Unknown/Incomplete; the last span is a direct fixture. | CP-O22, O28. |
| T13 | `SearchStop::Stalled` | `SOL:4734,12947-12948` | Like-named exit. Unknown/Incomplete; the last span is a direct fixture. | CP-O20–O24, O28. |
| T14 | `SearchStop::ExhaustionArtifactFailed` | `SOL:4676-4689,4735-4739,12950-12956` | After refresh/emission work and the higher-priority cap cases, an inner `RunUntilExit::RootDnZero` is mapped to this stop. If the refreshed root still has `dn==0`, the saved emission failure reason is used; otherwise the code supplies the default unsupported-artifact reason. Unknown/Incomplete; the last span is a direct fixture. | CP-O17–O19, O27–O29. |
| T15 | `SearchStop::MaterializationFailed` | `SOL:1237-1288,12958-12962` | RootProven could not be built, compacted, rebased, or (for relevant feature paths) strictly positive-verified. The underlying release behavior drops the certificate/returns Unknown; assignment of this typed SearchStop is `cfg(test)` only. The last span is a direct fixture. | CP-O18, O24, O27–O29. |
| T16 | `SearchStop::PreconditionRejected` | `SOL:916-929,940-972,12964-12968` | Zero cap, horizon-before-root, oversized root, immediate beyond H, or goal filter returns Unknown in release; typed SearchStop pushes are `cfg(test)` only. `UnsupportedCP1Root/Profile` are declared but have no construction sites. The last span is a direct fixture. | CP-O1–O3, O27, O28. |
| T17 | `SearchStop::InvariantViolation` | `SOL:4661-4669,4706-4710,4741-4744,12970-12974` | Executed constructors are missing root, refresh/exit disagreement, and stage-depth overflow. `ImpossibleRunUntilFallthrough` is declared but never constructed. All are Unknown/Incomplete; the last span is a direct fixture. | CP-O20–O24, O27, O28. |

Production currently discards the returned wide stop at `SOL:1155-1159`. The complete observation/consumer seam is test-only: retained vectors/getters and solve result pushes at `SOL:668-685,743-751,1007-1008,1041-1042,1356-1365,1805-1813`; materialization downgrade at `SOL:1285-1288`; compatibility comparison at `SOL:2270-2275`; pending-candidate helper at `SOL:2370-2379`; and fixtures/assertions at `SOL:12911-13103`. Narrow returns no stop at `SOL:4356-4367`. Thus the taxonomy is total locally but is not yet a production provenance channel.

### 1.5 Strict positive-verifier accept/reject surfaces (V01–V15)

There is exactly one production verifier entry: `TssVerifier::verify` at `VER:171-176`, delegating to `verify_certificate` at `VER:179-227`. It verifies the positive in-memory schema in C01; no Rust entry parses or checks `NoTssCertificateV1`.

| ID | Rust span | Exact accepted predicate / rejection boundary | Obligations |
|---|---|---|---|
| V01 | `VER:171-227` | Reject Unknown; require certificate root binding exactly equals `RootBinding::from_state(external_state)`; claimant equals root mover for Win and its opponent for Loss; validate arena/metadata; prohibit a zoned terminal root; replay root. | CP-O1–O3, O17–O19, O27–O29. |
| V02 | `VER:1364-1451` | Enforce global positive-arena node/edge/witness/commutation totals and caps, valid node/child IDs, unique Universal moves, whole-arena acyclicity (including orphans), and reachability of every node from root. Node numbering may be any acyclic order. | CP-O17–O19, O29. |
| V03 | `VER:203-207,229-320` | Metadata computes maximum leaf/completion resolution (`derived_t`), whether any zone exists, and minimum zone build horizon; preflight exports only `(derived_t, has_zone)`. Verification requires `derived_t <= certificate.semantic_horizon` and, when a zone build horizon exists, `derived_t <= zone_build_t`; it does not directly compare semantic horizon with zone build and has no independent H argument. | CP-O13, O17, O27, O29. |
| V04 | `VER:322-461,478-482,585-590` | Replay key contains the full `RootBinding` projection plus allowed commutation context, but not private incremental caches. Node-ID-indexed slots accept a repeated ID only under exact key equality; initial/inserted memory-cap overflow, an out-of-range slot, or an occupied insertion rejects. There is no hash/collision oracle. Distinct IDs for the same proposition remain permitted. | CP-O19, O26, O27, O29. |
| V05 | `VER:463-590,512-531` | Dispatch by positive node kind. Choice accepts an anchored, legal claimant placement with a recursively accepted child; it does **not** require membership in `AttackEdges_CP1`. | CP-O14, O16–O18, O27. |
| V06 | `VER:601-615` | `attacker_placement_wf` requires distance ≤8 from either a current claimant stone or any root stone. It does not itself check placement legality or proximity to every occupied stone; edge callers separately use `with_move`, while leaf empty-cell checks use only this proximity predicate. | CP-O1, O14, O27. |
| V07 | `VER:617-637` | `OrCompletion` replays one legal claimant move, requires terminal claimant win in the named pure-six window, and checks `completion_ply == placements.saturating_add(1)`. | CP-O3, O16, O27. |
| V08 | `VER:639-669` | `Win` requires the named active claimant-pure window to have count 5, or count 4 with two placements remaining, and checks resolution by `p.saturating_add(1/2)`. | CP-O3, O13, O16, O27. |
| V09 | `VER:671-739` | Positive `Loss` accepts a supplied nonempty list of active claimant count≥4 witness windows, requires no defender win-now, checks that **that supplied family** has hitting number greater than defender budget, and uses saturating `p+b+2`. It does not require the complete global claimant family. | CP-O3, O13, O16, O27. |
| V10 | `VER:741-825` | Validate recorded commutation structure, require each possible first placement to be legal/nonterminal SecondStone, replay both second placements, and compare only the resulting `Option<GameOutcome>`. Exact final `RootBinding`/state equality is not checked here. | CP-O15, O18, O27. |
| V11 | `VER:827-964` | At defender/nonterminal/no-own-win: explicit edge moves are unique and each replays with `outcome=None` to an accepted child. `allowed_commuted` moves are disjoint/unique and directly checked only for legal application; their nonterminal role is mediated by the parent commutation/mirror structure. The combined represented set is nonempty. Implicit dispatcher requires all kernel moves but accepts a superset; zone form rederives zone obligations; otherwise the combined set equals all legal moves. The test oracle checks omitted nonkernel moves are λ1-safe, not set exactness. | CP-O15, O16, O18, O25, O27. |
| V12 | `VER:1019-1276` | Rederive D14 zone candidates/core/descendant closure and require certificate zone fields equal that derivation. | Outside initial CP1 profile; CP-O12, O27. |
| V13 | `VER:1280-1349` | Recompute tight-dispatch prerequisites and the forced-defender kernel used by V11. The test-only oracle at `VER:159-168,938-961` validates omitted nonkernel moves by λ1; it does not require exact represented-set equality. | CP-O15, O25, O27. |
| V14 | `VER:593-599,905-910,1295-1302,1351-1362` | Replay clones caller engine state and calls `apply_with_delta`; leaf/dispatcher checks read its live incremental window/legal stores. Equality of root scalar fields does not independently bind those caches. | CP-O1–O3, O14–O16, O27, O29. |
| V15 | `CORE:471-506`; `TREE:559-705,1437-1449` | `hard_value_from_verified` requires non-Unknown status, present positive certificate, and V01 acceptance against the exact caller state. Tree drops all rejection/failure to Unknown before hard consumption. | Current fail-closed positive seal. CP-O27, O28, O29. |

### 1.6 State, key, profile, and exact-generator surfaces (R01–R04, G01–G08)

| ID | Rust span | Exact operation/predicate | Obligations |
|---|---|---|---|
| R01 | `STATE:21-112,203-253,283-406`; `packages/hexo_engine/rust/src/snapshot.rs:14-42` | Mutable state stores player/phase/clock/terminal plus board, legal, and tactics caches. `apply_with_delta` checks legal placement, mutates caches, increments clock, resolves terminal/turn schedule; `undo` restores delta. `load_state` replays an **ordered placement history**, not a v1 ownership map. | CP-O1–O3, O5, O6, O27. |
| R02 | `COORD:9-95`; `LEGAL:17-41,123-139`; `SEARCH:4136-4159,4282-4323` | Coordinates are `i16`; addition/subtraction/radius use unchecked machine arithmetic. Packed coordinate zero is the valid `(-32768,-32768)`, while search comments assume ID zero is an illegal sentinel. The fallback returns root action IDs without enforcing nonzero, and its debug assertion checks only root membership, so the sentinel convention is not actually established. | CP-O1, O5, O6, O27 and trace part of O27. |
| R03 | `VER:39-79,185-216` | `RootBinding::from_state` sorts `(q,r,owner)` and copies current player, full phase including `SecondStone.first`, placements, and terminal. Verification compares these fields but does not rebuild state invariants/caches. | CP-O1–O3, O6, O27, O29. |
| R04 | `SOL:2681-2773,2910-3024,9803-9909` | `PositionKey` and `WidePositionKey::from_state` contain sorted occupancy/owners, player, full phase, clock, and terminal; equality, not hash, authorizes reuse. Synthetic completed-pair keys reproduce presumed final state but use saturating clock arithmetic. | CP-O19, O26, O27. |
| G01 | `SOL:562-714,842-929,1070-1159,4392-4413`; `TREE:535-559,940`; `ASYNC:545-589`; `SEARCH:4075-4086`; `CP1:25-74` | Profile routing: production tree, async, and root-guard callers construct `TssSolver::default`; default width has pair-complete false, so it selects narrow compatibility. Environment flags control shared fragments, K-reply, interior census, and lazy frontier; no serialized query binds them. CP1 requires base-wide/pair-complete/batch on and listed refinements off. | CP-O2, O20–O27. |
| G02 | `SOL:8426-8669` | Scan **all** active windows. Candidate set contains claimant-pure windows meeting threshold; at CP1 threshold 2 it also unions all empty cells in defender count≥4 threats. Coordinates dedup with aggregated strength/block flags. `ordered_*` changes only order, using heuristic/D6 keys. | CP-O14, O16, O27. |
| G03 | `SOL:8671-8965` | `WideTurnGate` freezes turn-start strong/weak claimant families and defender threats. `second_candidates(first)` unions strong continuations, frozen candidates, and weak promotions. `evaluate_pair` retains exactly a nonempty claimant count≥4 family that hits every frozen defender count≥4 threat and has hitting value `None` or `Some(2)`; `Some(0/1)` is rejected. `Some(2)` is Pending. For `None`, horizon plus inclusion-minimal-obstruction availability chooses Tactical versus Pending but does **not** decide pair retention; its resolution arithmetic saturates. | CP-O14, O16, O27. |
| G04 | `SOL:6267-6275,6308-6484` | FirstStone attacker route evaluates the ordered encounters generated by its outer first-candidate and per-first second-candidate universes, dedups accepted encounters by unordered raw-coordinate key, and retains the first encountered orientation as `WidePnMove::Pair`. Because candidate membership is non-monotone, one unordered pair need not be generated in both orders (`SOL:6397-6401,8777-8835`). | CP-O14, O16, O27. |
| G05 | `SOL:6486-6565` | Opening/SecondStone attacker route replays each ordered candidate; retains claimant completion within H, exact tactical leaf, forcing completed turn, or pending opening move; other outcomes are omitted. | CP-O3, O14, O16, O27. |
| G06 | `SOL:9130-9197` | Exact defender reply kernel: budget 1 cells hit every threat; budget 2 cells admit a distinct mate hitting every threat; unsupported budgets use the full hitting universe. | CP-O15, O16, O27. |
| G07 | `SOL:3256-3725` | Batch pair planner admits only defender FirstStone, budget 2, live opponent threat, no own win, τ=2; builds canonical-dedup K2, applies each first to derive exact K1 at SecondStone, requires second∈root K2, directed uniqueness and reverse direction, then requires equality of the two **synthetic completed-pair `WidePositionKey`s**. It does not replay each second or compare exact final states here. It keeps a raw-low representative, then sorts pairs by D6 kernel rank. Any failed premise returns `None`. | CP-O15, O18, O27. |
| G08 | `SOL:6567-6799,7132-7266` | Defender routing uses batch pairs only when G07 returns `Some`; otherwise ordinary kernel singles. Release always uses batch planner. Atomic pair children are Pending with synthetic final key, then C05 expands them back to sequential proof nodes. | CP-O15, O16, O18, O27. |

### 1.7 Why this inventory is exhaustive

The 74 declared audit units cover and classify the audited authority graph. They are not disjoint by source span: a site appears in more than one row when it has distinct negative, certificate/test, stop, or verifier roles (for example the synthetic stop fixture is N17/T09 and lies inside C13's broad test boundary).

- N01–N17 cover every construction and outward consumption of `Loss`, `Refuted`, negative cache, or negative artifact found across production and test modules.
- C01–C13 cover every production `TssCertificate { ... }` site (immediate at `SOL:951`, wide root at `SOL:1258`, fragment promotion at `SOL:1316`, narrow at `SOL:4330`, and D6 return at `VER:1694`), plus the test-only cap-resume constructor at `SOL:1561`, every graph/label shaper, every byte-like encoder, async transport, and durable trace.
- T01–T17 are a closed enumeration of the enum declarations; matching the declaration count is mechanically checkable: one event, six inner exits, ten terminal stops.
- V01–V15 descend from the sole verifier entry and account for every node kind and every distinct global acceptance gate.
- R01–R04 and G01–G08 are the state/key/profile/generator dependencies reached by those paths and named explicitly by CP-O14/15/27.

No CP1 Rust CLI or alternate negative parser/checker was found. Ordinary game terminal reporting is excluded because it declares a game result, not CP1 contract refutation. Hunt telemetry and sibling VCF `false` returns are not proof authority; relevant test-only TSS constructors/data are nevertheless recorded in N04, N05, N17/T09, and C13. The second-method resweep in §4 tests these exclusions again.

## 2. Correspondence map

### 2.1 Meaning of the labels

- **ALIGNED** means the cited Rust predicate and cited Lean/spec predicate have the same scoped semantics, and the remaining proof should be a direct representation/branch calculation. It does **not** mean the proof has landed.
- **DIVERGENT** means the two predicates or representations differ now. A bridge theorem stated as equality would be false without changing the Rust path, changing its authority, or inserting a checked adapter.
- **DIVERGENT-SUSPECTED** means source-local behavior differs or has a concrete hazard, but a missing reachability/domain theorem might prove the differing branch irrelevant for frozen CP1.
- **UNMODELED** means Rust has behavior for which the stable Lean contract supplies no concrete implementation counterpart, or Lean supplies an abstract field that has not been instantiated by Rust execution.

All ALIGNED rows cite both sides. Most are deliberately narrow control-taxonomy alignments. There is no ALIGNED claim for a production Rust negative checker, because none exists.

### 2.2 The exact bridge that future work must prove

Let `Pᵣ` be an independently supplied Rust root, `b : Vec<u8>` the exact byte sequence handed to the executed checker, and `ι` a checked, total conversion into `CompletenessCP1.State`. The conversion is defined only after validating every `i16/u32/u64/usize` use and never interprets overflow as a cutoff. Let `Q` be the query decoded from the v1 header. The target is not an informal simulation claim; it is this composition:

```text
executed_check_no(Pᵣ, b) = true
  ↔ strict_v1_decode(b) = some F
   ∧ F.rootEntries = canonical_root_entries(Pᵣ)
   ∧ Q = F.query
   ∧ WellFormedCP1 R Q (ι Pᵣ)
   ∧ build_primary_replay_table(ι Pᵣ, F.dag) = some replay
   ∧ checkNoDag R X Q (ι Pᵣ) 0 0 F.dag replay = true
```

followed by:

```text
checkNoDag_sound                         ⇒  NoAt R Q (ι Pᵣ) 0 0
NoAt at depth 0 + exact external binding ⇒  NoContractWin for that exact root/query
```

The stable Lean endpoints are `NoContractWin`/`NoAt` (`CP1:446-459`), `checkNo`/`checkNo_iff` (`CP1:1694-1748,2094-2108`), and `checkNoDag`/soundness/iff (`CP1:2480-2614,2887-2953,3164-3212`). At pinned HEAD `0e1bdabd…`, the codec endpoint is only the node-stream suffix declarations and one fixture (`CP1:3225-3467`; committed `LEDGER_CP1.md:44-45`); the committed file has no full-file syntax/decoder, general inverse, replay-table constructor, or bytes-to-`NoAt` capstone. The evolving dirty R-CP9 work is excluded regardless of its current progress.

The bridge decomposes into executable proof obligations:

| Bridge | Required exact statement |
|---|---|
| B0 bytes | The future **executed** strict parser and the completed Lean full-file decoder accept exactly the same byte strings and produce field-for-field equal file values: exact 86-byte header, signed `i16`, minimal ULEB32, root 0, forward deltas, all caps, no trailing bytes (`CERT:108-201,344-355`). Only the Lean node-suffix portion is stable today. O29-A0 decides whether the executed parser is compiled Lean, refined Rust, or another machine-checked route. |
| B1 root/state | Successful checked conversion gives `ι(Pᵣ)` with byte-identical sorted occupancy/owners and equal player, phase/first anchor, clock, terminal; external root equality precedes all semantic work. A checker-local builder independently reconstructs the state from v1 entries. |
| B2 global primitives | On every admitted replay state, the executed checker-local global reconstruction of legal cells, windows, terminal, threat families, own-win, hitting data, and phase budget equals the functions installed in `RegenerationBinding`; the Rust engine/producer functions need their own O27 refinement. No solver-maintained incremental cache is checker authority (`CERT:207-215`). |
| B3 transitions/leaves | For each canonical edge, checked Rust replay succeeds iff Lean `R.replayPendingEdge` does, and converted child/clock equals exactly. Rust typed leaf and edge-local classifications equal `R.regenerateStateLeavesGlobally`/`R.regenerateEdgeLocalShapesGlobally`, including claimant, resolution, H/S/C, and terminal cases. |
| B4 attack | For every `NodeInContract` claimant state, after mapping coordinates/edge tags and canonicalizing raw signed `Edge.before`, the **entire ordered, duplicate-free** checker-owned executed regeneration equals `R.regenerateAttackEdgesGlobally P = AttackEdges_CP1 R P`. Before quotienting, it probes both legal attacker-pair orders (or proves an order illegal), includes global defender-block inputs, proves order classifications/final states agree, and emits one raw-low edge satisfying `Edge.CanonicalPair`. The Rust solver-producer enumeration needs a separate equality to the same target for completeness. |
| B5 defense | The checker-owned executed sequential K2→K1 relation equals `R.regenerateDefendEdgesGlobally P = DefendEdges_CP1 R P`. Separately, Rust `forced_defender_pair_plan = Some plan` iff `plan` is the exact raw-canonical `Edge.defenderPairQuotient` relation with both commutations/equal final state, while `None` routes to the complete ordinary single relation. Checker and solver producer are refined separately to the shared target. |
| B6 keys/sharing | `PositionKey::from_state` and `WidePositionKey::from_state` equality iff converted full states are equal; synthetic keys equal two checked applications; hash only selects buckets. Repeated DAG IDs correspond to exactly equal `NoDagProposition` values. |
| B7 tags/clocks | Every Rust node/child tag maps to exactly one Lean constructor/disposition or to explicit incomplete rejection; state and certificate increments equal `Edge.stateDepthIncrement`/`Edge.certificateDepthIncrement`; no numeric PN/DN equality erases the tag. |
| B8 profile | The executed query proves grammar version `CP1` and profile `CP1.profile` (definitionally `frozenProfile`) (`CP1:31-74`): `pairComplete=true`, `baseWideSearch=true`, `canonicalBatchDefense=true`, and every switch represented in `GrammarProfile` has its frozen value. Environment reads cannot mutate or bypass this binding. `TSS_LAZY_FRONTIER` is **not** a `GrammarProfile` field: the first producer manifest must bind it to `0` (`COMPLETENESS_SPEC.md:643-653`), or later prove CP-O24. Likewise `cfg(test)` incremental-defender mode stays outside release authority unless CP-O25/checker-side exact comparison is proved. Checker soundness must be independent of both producer discovery modes. |
| B9 emission | **One-way producer guarantee:** if the emitter reports success, its canonical bytes decode to the claimed DAG/root/query, with declared exhaustive/selected/boundary provenance ready for independent B10 checking. Emitter failure or inability to materialize is Unknown, not a semantic failure. Checker acceptance—not emitter success—establishes `WellFormedNoDag`; any desired `dn==0 → eventual emit` liveness belongs to scheduler/materialization obligations. |
| B10 executed checker | For byte-identical `b` and independently reconstructed `Pᵣ`, the compiled **executed-checker** Boolean equals the Lean `checkNoDag` composition above. `ExecutableRegenerationBinding` (`CP1:1042-1052`) sits here: its tight-dispatch Boolean must be implemented by the B2/B5 checked primitive and its `iff` proof; it is not a license to supply arbitrary solver lists. The implementation language is selected by O29-A0. |
| B11 sealed mint | `NO_CONTRACT_WIN` is constructible only from B10 acceptance for the exact external root. Search `dn`, `Loss`, cache, cap, and emitter provenance cannot bypass it (`CERT:424-428`; `COMPLETENESS_SPEC.md:781-790`). |
| B12 trace | The emitted proof trace serializes an injective source tag plus exact root/query/profile/horizon binding and certificate bytes (or a content hash bound to retained bytes), and deserialization reconstructs the same proposition. A scalar `-1` is insufficient. |

Environment assumptions for all statements are: frozen grammar ID `CP1-a49e8abd-v1`; external root post-opening/nonterminal; claimant is root mover; `H≥p0`; `S=min(H-p0,256)`, `C=256`; checked coordinate/clock/depth arithmetic; no concurrent mutation of the input; deterministic global primitive rebuild; raw signed `(q,r)` ordering; batch defender mode; and no quiet/ranked/shared/round3/census/K-reply/prior/narrow-cache refinement. The first producer profile additionally binds `TSS_LAZY_FRONTIER=0`; `cfg(test)` incremental defender mode is not release authority. Later lazy/incremental admission requires CP-O24/CP-O25 respectively. Discovery caches may vary adversarially, but the checker has no solver arena or cache input.

### 2.3 Row-by-row map: negative/refutation and artifact shaping

| ID | Lean/spec counterpart and exact correspondence statement | Class |
|---|---|---|
| N01 | `StateLeafShape.kind = .defenderAdaptiveLambdaOneLoss` plus `StateLeafShape.OwnerMatches` (`CP1:275-290`) can model the one-turn positive claimant leaf only when `P.currentPlayer ≠ Q.claimant`, and only after B2/B3. A claimant-to-move/root-mover λ1 `-1` has no such positive-leaf counterpart. Neither case is `NoContractWin` (`CP1:446-459`): a no-≤b-hitting-set Rust scalar supplies no exhaustive CP1 attack tree. In addition, the shared and HexGNN-local Rust threat scans require a separate extensional-equality proof before either can stand for the same global primitive. | **DIVERGENT** |
| N02 | Positive `ContractWin R Q_opponent P` (`CP1:370-459`) is the nearest logical analog, not the negative dual for the root claimant. `ContractWin` has no executable-binding argument. Moreover, a normal CP1-well-formed `Q` binds claimant to the root mover, so Rust's opponent-positive Loss at the same root requires an explicitly opponent-oriented, non-CP1 query or a separate semantic wrapper. Even a proved wrapper equivalence would not imply root-claimant `NoContractWin` without a determinacy/grammar theorem. | **DIVERGENT** |
| N03 | `NoContractWin` ranges over the finite CP1 forcing grammar, while REF ranges over all legal moves to a depth. Neither relation contains the other by definition. It may be a differential oracle only after a horizon/game-semantics theorem. | **DIVERGENT** |
| N04 | Same mismatch as N03, additionally outside production. No CP1 declaration models its optimized memoization. | **DIVERGENT** |
| N05 | Negative Choice/Universal duality is represented by `NoCertificate`, `WellFormedNo.sound` (`CP1:605-1040`) and DAG analogs (`CP1:2110-3212`). B7 must preserve the tag/provenance distinction; only CP-O20's recurrence proof together with exact B3 replay/leaves and B4/B5 exhaustion could establish `dn=0` **with complete typed provenance** iff a corresponding negative constructor exists. Scheduler/materialization obligations are additional if eventual emission is claimed. Numeric recurrence alone is weaker, especially because cutoffs share numbers. | **DIVERGENT-SUSPECTED** |
| N06 | CP-2 explicitly says `DepthCutoff` and Refuted share numbers and only tagged, final structural boundaries can enter v1 (`COMPLETENESS_SPEC.md:775-790`; `CERT:333-342`). Rust retains the distinct tag (`SOL:6125-6133`) and N17 rejects it. Scoped statement: `DepthCutoff → no negative disposition emitted` is direct. | **ALIGNED** |
| N07 | `StructuralBoundaryEvidence` (`CP1:647-659`) records `observedStateDepth > S ∨ observedCertificateDepth > C`; its Boolean/iff is at `CP1:1295-1338`, while the no-edge-local conjunction belongs to enclosing `checkEdgeNoWith` (`CP1:1444-1460`). With exact clocks and `S=min(H-p0,256)`, `p>H` must imply an S crossing, but the converse is false when `S=256 < H-p0`. B1/B3/B7 must prove the required implication/exclusion correspondence and, for current production, that the earlier staged DepthCutoff makes this Refuted branch unreachable. | **DIVERGENT-SUSPECTED** |
| N08 | Lean positive grammar has terminal/edge-local constructors (`CP1:370-445`); a claimant terminal cannot be negative. B3 must prove claimant-terminal nodes are unreachable at expansion because every such edge was already tagged completion. A direct claimant-terminal input is a source-local witness against unqualified equality. | **DIVERGENT-SUSPECTED** |
| N09 | `StateLeafInContract`, `NoStateLeafAt`, and Boolean reflection (`CP1:378-384,632-636,1100-1158`) are counterparts. B3 must prove exact global leaf/resolution equivalence and that “unavailable” means logical exclusion, not cache failure or overflow. | **DIVERGENT-SUSPECTED** |
| N10 | Frozen `CP1.profile.interiorCensus = .off` (`CP1:31-74`) has no negative constructor for census dismissal. Exact statement is therefore only that B8 makes this branch unreachable. No production query currently establishes B8. | **UNMODELED** |
| N11 | `ExecutableRegenerationBinding.checkTightDefenderDispatcherGlobally` and its `_iff` field (`CP1:1042-1052`) are abstract. B2/B5/B10 must instantiate them and prove Rust failure iff `R.tightDefenderDispatcherGlobally` is false; no such bridge exists. | **UNMODELED** |
| N12 | `NoCertificate.choiceExhausted`/`.universalCounterexample` (`CP1:605-767`) distinguish “empty complete canonical list” from generator failure. B4/B5/B7 must prove the returned vector is the complete list and its empty case has the correct owner semantics. Without those equalities, empty Rust children cannot be negative evidence. | **DIVERGENT-SUSPECTED** |
| N13 | `checkRecursiveChild_eq_some_iff` and DAG edge checking (`CP1:1216-1265,2480-2614`) reject a failed pending-edge replay; they do not turn it into `LocalNo`. Required B3 equality is `Rust apply/replay failure ↔ Lean R.replayPendingEdge = none`, followed by checker rejection. Rust instead consumes an `apply_with_delta` move-application failure as Refuted. Separately, B6 must prove that every trusted `child.future_key` equals the recomputed child key, because release code does not enforce that equality. | **DIVERGENT** |
| N14 | `EdgeLocalInContract`/`NoEdgeLocalAt` (`CP1:386-393,639-645`) are counterparts. B3 must prove the winner/terminal mapping and edge reachability; opponent terminal may refute a positive claimant edge, but claimant terminal must be completion. | **DIVERGENT-SUSPECTED** |
| N15 | CP1 freezes `CP1.profile.baseWideSearch = true` and `CP1.profile.narrowNegativeCache = .off` (`CP1:31-74`). No Lean counterpart exists for `LOCAL_TT_FAILED` or the conflated `None` causes; B8 should make this path unreachable. | **UNMODELED** |
| N16 | Sealed negative design is B11 and `COMPLETENESS_SPEC.md:781-790`. Current Rust `-1` is N01 or opponent-positive N02 and its production scalar merge erases which. It cannot correspond to `NoContractWin`. | **DIVERGENT** |
| N17 | The current scoped predicate is only “no production Rust negative candidate can cross”: every emitter shape errors and production `RootRefutedCandidate` remains unreachable. The direct `cfg(test)` synthetic fixture is not an emitter result. This matches the landed CP4 fail-closed seam (`HUNT_REPORT_CP4_SEAM.md:18-39,116-123`) and the no-mint rule (`CERT:424-428`). B9/B10 remain absent. | **ALIGNED** |
| C01 | Lean negative schema is `NoCertificateDag` with `NoCertificateDagBody.baseNoConstructor`/`.choiceExhausted`/`.universalCounterexample` and forward dispositions (`CP1:2110-2317`); C01 is a claimant-positive schema with five different node forms. No representation bijection exists. | **DIVERGENT** |
| C02 | Lean `StateLeafShape`/`EdgeLocalShape` (`CP1:281-313`) is the counterpart. B3 must prove exact kind/resolution and, for edge-local forms, `expandedHeight`, plus a separate theorem that Rust's named window evidence is globally regenerated and soundly erased into those shapes. Source predicates look closely related, but the binding is abstract and positive-only. | **DIVERGENT-SUSPECTED** |
| C03 | Negative `NoCertificateDagBody.choiceExhausted` must retain every regenerated attack disposition; positive materialization selects one Choice edge and all Universal edges. The quantifiers are dual and the output schema/ID orientation differs. | **DIVERGENT** |
| C04 | No frozen-CP1 counterpart: narrow is excluded by `CP1.profile.baseWideSearch = true`, yet its positive certificate is production-reachable under defaults. As a current artifact authority it also has the opposite quantifiers from negative DAG. | **DIVERGENT** |
| C05 | `Edge.defenderPairQuotient`, `Edge.CanonicalPair`, and sequential `DefendEdges_CP1` are at `CP1:200-251,354-366`. B5 must prove both orders, exact final state, and quotient expansion. The planner compares synthetic final keys, but the verifier's commutation check compares only final `Option<GameOutcome>`; exact final state/cache commutation therefore still needs the engine correspondence theorem. | **DIVERGENT-SUSPECTED** |
| C06 | Frozen CP1 disables shared positive fragments (`CP1:31-74`); no negative-DAG counterpart models positive fragment relabel/import. B8 must make it unreachable for a CP1 negative checker. | **UNMODELED** |
| C07 | Frozen CP1 has no zone-distance certificate field; the positive zone grammar is outside initial negative grammar. | **UNMODELED** |
| C08 | V1/Lean require root ID 0 and positive forward deltas to larger child IDs (`CERT:100-106`; `CP1:2110-2317`). C08 recursively emits children before parent and cached proof requires `child<parent`; a two-node parent→leaf is a concrete opposite-orientation witness. A new negative compactor is required. | **DIVERGENT** |
| C09 | `Edge.before` uses raw signed identity (`CP1:200-251`); no landed executable theorem maps D6-remapped positive artifacts to the frozen negative byte grammar. It should be outside the minimal checker. | **UNMODELED** |
| C10 | Lean `HorizonTriple`/exact binding and node depth checks (`CP1:112-197,1694-1748`) are counterparts, but the zoned positive retry protocol is not modeled and CP1 negative bytes bind H directly. | **UNMODELED** |
| C11 | No Lean artifact transport counterpart exists. Required B11 theorem must be checked before transport; dropping certificate makes later revalidation impossible. | **UNMODELED** |
| C12 | B12 is false now. Concrete witness: any λ1-undecided root with an accepted deep Loss emits signed byte `0xff`, identical to a direct λ1 forced-loss root, despite proving different propositions. | **DIVERGENT** |
| C13 | Stable Lean has only strict negative node-stream codec declarations/one fixture (`CP1:3225-3467`), while these are test-only positive fingerprints or invalid synthetic stop bytes. No byte equivalence theorem is possible. | **UNMODELED** |

### 2.4 Row-by-row map: stopping surfaces

The Lean file intentionally does not model df-pn scheduling. These rows correspond to the normative control contract in `COMPLETENESS_SPEC.md:723-790` and the landed Rust evidence in `HUNT_REPORT_CP4_SEAM.md:18-123`. ALIGNED here means “same stop/event predicate and public non-authority,” not “search completeness proved.”

| ID | Spec counterpart and exact correspondence statement | Class |
|---|---|---|
| T01 | Spec `StageEvent.SelectedCutoff`: intermediate only, with source stage and encountered depth; never a verdict. Rust records it only as test observation before requesting a deeper stage; failure to compute a strictly advancing stage instead ends as `NonAdvancingCutoff`. | **ALIGNED** |
| T02 | Spec `RunUntilExit.RootPnZero`: refreshed local proof number is zero. Outer driver must recheck after bottom-up refresh. Rust branch is identical. | **ALIGNED** |
| T03 | Spec `RunUntilExit.RootDnZero`: refreshed local disproof number is zero, carrying no semantic negative authority. Rust returns only the tag and N17 gates emission. | **ALIGNED** |
| T04 | Spec `RunUntilExit.NodeCap {expansions,cap}`. Rust tests both global/invocation caps and returns the exact counter/minimum bound. | **ALIGNED** |
| T05 | Spec selected-cutoff inner exit requests deeper stage; Rust does so only when `deepen_after_selected_cutoff`. | **ALIGNED** |
| T06 | Spec cutoff-no-progress exit; Rust requires a selected cutoff with `made_progress=false`. | **ALIGNED** |
| T07 | Spec stalled exit for no selectable/progressing work; Rust uses missing root or `work` stall. It remains incomplete. | **ALIGNED** |
| T08 | Spec `RootProven` covers an immediate admitted positive leaf at stage 0 or wide `pn==0` after mandatory refresh, later subject to positive materialization/verification. The immediate SearchStop observation is `cfg(test)` only; the wide run constructs the stop in release but its caller discards it. The underlying positive/Unknown behavior matches the scoped control contract. | **ALIGNED** |
| T09 | Spec requires emitted v1 bytes and later independent `checkNo` acceptance before NO. Rust has the carrier but no B9 emitter/B10 checker or production consumer; only a direct `cfg(test)` invalid-byte fixture constructs the stop today. | **UNMODELED** |
| T10 | Spec NodeCap→Unknown/Capped. Rust carries stage/counters and never treats it as a proof. | **ALIGNED** |
| T11 | Spec CutoffNoProgress→Unknown/Incomplete. Rust branch and fields match. | **ALIGNED** |
| T12 | Spec NonAdvancingCutoff→Unknown/Incomplete. Rust checks failure of strictly advancing stage selection. | **ALIGNED** |
| T13 | Spec Stalled→Unknown/Incomplete. Rust mapping matches. | **ALIGNED** |
| T14 | Spec failed negative artifact→Unknown/Incomplete with typed reason. After cap precedence, Rust maps the inner `RunUntilExit::RootDnZero` tag to `ExhaustionArtifactFailed`; the refreshed `dn==0` condition selects the saved reason rather than being, by itself, the constructor trigger. Other `dn==0` observations remain non-authoritative NodeCap/Unknown. | **ALIGNED** |
| T15 | Spec failed positive materialization→Unknown/Incomplete. Release Rust drops the certificate/returns Unknown; only the `cfg(test)` seam rewrites the retained stop to `MaterializationFailed` with its typed reason. | **ALIGNED** |
| T16 | Normative precondition includes exact post-opening/nonterminal root and frozen profile (`CERT:209-215`), but release Rust executes only cap/horizon/root-size/immediate filters and returns Unknown; only test builds record `PreconditionRejected`. Declared `UnsupportedCP1Root/Profile` have no constructors. | **DIVERGENT** |
| T17 | Spec invariant violations are Unknown/Incomplete. Rust's three executed typed cases never mint a verdict; the fourth declared code, `ImpossibleRunUntilFallthrough`, has no constructor. | **ALIGNED** |

### 2.5 Row-by-row map: current strict verifier

| ID | Lean/spec counterpart and exact correspondence statement | Class |
|---|---|---|
| V01 | Target is B0/B1/B10 and `checkNoDag` (`CP1:2480-2614`), beginning from bytes with root node 0. V01 starts from a positive in-memory object and caller state, accepts Win/Loss, and has no parser/WellFormedCP1 gate. | **DIVERGENT** |
| V02 | `WellFormedNoDag`/global checker require root 0, forward references, one ID per exact proposition, all reachable (`CP1:2110-2317,2480-2614`; `CERT:100-106`). V02 accepts arbitrary acyclic ID orientation and duplicate propositions under distinct IDs. C08 gives a two-node witness. | **DIVERGENT** |
| V03 | `HorizonTriple`, node clocks, and v1 fixed caps (`CP1:112-197`) are counterparts. Positive metadata uses different leaf/zone fields; exact clock conversion still needs B3/B7. | **DIVERGENT-SUSPECTED** |
| V04 | Lean `NoDagProposition` exactness and `checkNoDagUniquePropositions` (`CP1:2110-2317,2480-2614`) require repeated ID→equal proposition **and** equal proposition→one ID. V04 enforces only the former. Two identical leaf propositions stored under two IDs are a witness. | **DIVERGENT** |
| V05 | CP1 Choice must enumerate every `AttackEdges_CP1` edge in canonical order (`CP1:354-445,605-767`). V05 accepts one arbitrary anchored legal move. Witness shape: at FirstStone with a nonterminal claimant-pure count-5 window, choose a remote legal radius-8 cell outside every CP1 candidate family, then cite the unchanged count-5 Win leaf; V05's grammar permits it while CP1 membership fails. | **DIVERGENT** |
| V06 | B4 requires exact attack membership, not the V06 anchor-proximity predicate plus separate edge replay legality. `RegenerationBinding.regenerateAttackEdgesGlobally` is abstract (`CP1:314-366`), so V06 is a strictly wider eligibility component and leaf callers do not separately query the legal store. | **DIVERGENT** |
| V07 | `EdgeLocalShape.kind = .directOrCompletion` or `.secondPlacementOrCompletion` and `EdgeWinAt` (`CP1:281-313,430-445`) are counterparts. B3 must establish identical global terminal/window checks and prove `p+1` cannot overflow; Rust uses saturation. Current use of live engine caches prevents a mechanical claim. | **DIVERGENT-SUSPECTED** |
| V08 | `StateLeafShape.kind = .claimantLambdaOneWin` and `StateLeafInContract` (`CP1:281-313,378-384`) are counterparts. Count/budget/resolution shapes look related, but B2/B3, post-opening/nonterminal premises, and no-overflow for Rust's saturating `p+1/p+2` are not instantiated. | **DIVERGENT-SUSPECTED** |
| V09 | `StateLeafShape.kind = .defenderAdaptiveLambdaOneLoss` is the counterpart positive tactical leaf, not `NoContractWin`. V09 checks only the supplied nonempty witness subset, not equality with a complete global family, and saturates `p+b+2`; B2/B3 must decide whether the modeled leaf permits that witness form and prove the admitted arithmetic domain. | **DIVERGENT-SUSPECTED** |
| V10 | `Edge.defenderPairQuotient` and its two-placement clock increments (`CP1:200-251,257-271`; ledger rows 21–23) are counterparts. V10 compares only final outcome options, not final RootBindings. B1/B3/B5 must prove that the two legal nonterminal-first executions commute in every modeled field and cache. | **DIVERGENT-SUSPECTED** |
| V11 | `DefendEdges_CP1` is an exact ordered relation (`CP1:354-366`). In implicit mode V11 accepts every kernel member plus extra unique legal nonterminal moves with recursively accepted children. `SOL:12653-12675` constructs the full-universe superset and shows both production verifier and test oracle accept it; the oracle only validates omitted nonkernel moves and is vacuous for a full represented set. | **DIVERGENT** |
| V12 | Frozen CP1 has ranked/unforced zone features off (`CP1:31-74`); no negative checker counterpart for D14 zone certificate data. | **UNMODELED** |
| V13 | `ExecutableRegenerationBinding.checkTightDefenderDispatcherGlobally` and `checkTightDefenderDispatcherGlobally_iff` (`CP1:1042-1052`) are the targets, but no theorem or generated adapter identifies V13 with them. | **UNMODELED** |
| V14 | Normative checker rebuilds all derived primitives and accepts no solver cache (`CERT:207-215`). V14 clones the supplied engine/caches. Equal RootBindings with divergent private cache state would be observationally indistinguishable at binding but could verify differently; a checker-local rebuild is required. | **DIVERGENT** |
| V15 | `checkNoDag_sound` plus B11 is the negative seal. V15 seals only a positive certificate, including opponent-positive Loss; it is safely fail-closed for its own grammar but cannot discharge the negative proposition. | **DIVERGENT** |

### 2.6 Row-by-row map: state, keys, flags, and generators

| ID | Lean counterpart and exact correspondence statement | Class |
|---|---|---|
| R01 | `State`, `State.PhaseScheduleValid`, `State.StoredFirstOccupied`, and `WellFormedCP1` (`CP1:81-197`) are targets. B1/B3 require state reconstructed from unordered v1 ownership entries and exact apply/undo. Rust `load_state` instead requires ordered history and there is no checker-local v1 builder. | **DIVERGENT** |
| R02 | Lean `Cell` has unbounded integer fields `q` and `r`, while v1's checked `i16` representation requires partial checked conversion/arithmetic. Rust uses unchecked `i16`. Concrete in-domain witness: stones `(0,0),(8,0),…,(32760,0)` are fewer than 4097 and can avoid six contiguous; legal-radius generation evaluates `32760+8`, debug-panicking or release-wrapping instead of rejecting. Separately `pack_coord(-32768,-32768)=0` collides with search's assumed sentinel convention, yet no nonzero action-ID invariant is enforced. | **DIVERGENT** |
| R03 | `Query.ExactBinding`/`WellFormedCP1` (`CP1:112-197`) require exact root plus reconstructed global invariants. RootBinding field equality covers the scalar projection but trusts caller caches and cannot be built from v1 bytes independently. | **DIVERGENT** |
| R04 | `NoDagProposition` and exact state equality (`CP1:2110-2317`) are counterparts. `from_state` keys look field-complete and hash is non-authoritative, but synthetic keys use saturation. Witness premise `p=u32::MAX-1` makes a two-placement key saturate rather than reject; B1 may exclude it only after a proved domain gate. | **DIVERGENT-SUSPECTED** |
| G01 | Exact grammar target is `CP1.profile` (definitionally `frozenProfile : GrammarProfile`) (`CP1:31-74`) and B8. `TssSolver::default` uses `WidthOptions::default` with pairComplete false, routes to narrow, and reads unbound env flags. `TSS_LAZY_FRONTIER` is an additional operational producer choice rather than a `GrammarProfile` field; the initial profile must bind it to 0, while later lazy completeness is CP-O24. Thus current production execution is not the frozen grammar/operational profile. | **DIVERGENT** |
| G02 | Target is exact ordered `AttackEdges_CP1` via `RegenerationBinding.regenerateAttackEdgesGlobally` and canonical laws (`CP1:314-366`; ledger rows 17,19-20). Set ingredients include global defender blocks, but final Rust ordering uses heuristic/D6 keys rather than an explicit raw `Edge.before` sort; there is no independent model equality or reachable inversion fixture in this audit. | **DIVERGENT-SUSPECTED** |
| G03 | `Edge.attackerPair`, `Edge.CanonicalPair`, and edge-local/pending classifications (`CP1:200-313`) are targets. B3/B4 must prove frozen-family promotion, τ gates, all threats, and checked H arithmetic. Saturating `start_placements+6` is a concrete boundary hazard. | **DIVERGENT-SUSPECTED** |
| G04 | `CERT:83` requires both legal attacker-pair orders examined before the unordered/final-state quotient. Rust examines only encounters whose first coordinate is in the outer turn-start list; promoted/fresh second coordinates need not be outer first candidates (`SOL:8777-8835`), and `SOL:6397-6401` explicitly records non-monotone membership/one-order hazards. It then retains first-encounter orientation rather than constructing an edge satisfying `Edge.CanonicalPair` (`CP1:200-251`). The process predicate is therefore different even before a final-list inversion fixture is locked. | **DIVERGENT** |
| G05 | `AttackEdges_CP1` plus `R.regenerateStateLeavesGlobally`/`R.regenerateEdgeLocalShapesGlobally` are targets. The route may be extensionally correct, but omission of terminal/nonforcing outcomes and exact H resolution need B3/B4; no Rust/Lean statement exists. | **DIVERGENT-SUSPECTED** |
| G06 | `DefendEdges_CP1` and finite canonical defense laws (`CP1:354-366`; ledger rows 19-20) are abstract targets. Kernel mathematics is suggestive, but no theorem equates Rust threat families/extendable-hit code to the sequential Lean relation. | **UNMODELED** |
| G07 | `Edge.CanonicalPair` and raw `Edge.before` require raw signed pair order. Concrete read-only witness: `xsnfyll_forced_defender_fixture` (`SOL:12843-12868`) has asserted K2 `a=(1,-6), b=(3,-5), c=(4,-6)` (`SOL:12340-12373`). Its residual K1s retain `(a,b)` and `(a,c)`. Replaying `canonical_frame`/`d6_coord_i32` (`SOL:9658-9800`) gives symmetry 8 and kernel ranks `a,c,b`; `SOL:3716-3720` therefore orders `(a,c),(a,b)`, while raw `Edge.before` orders `(a,b),(a,c)`. Add a locked assertion later, but the source/fixture derivation is already an explicit list-order counterexample. | **DIVERGENT** |
| G08 | Target is the exact sequential `DefendEdges_CP1` quotient of B5. The routing fact “Some→atomic, None→singles” matches the design, but preservation of every sequential obligation and synthetic-key equality is exactly the missing CP-O15 theorem. | **DIVERGENT-SUSPECTED** |

### 2.7 Classification count and headline gaps

Counting the 74 rows above:

| Class | Rows | Count |
|---|---|---:|
| ALIGNED | N06, N17; T01–T08, T10–T15, T17 | **17** |
| DIVERGENT | N01–N04, N13, N16; C01, C03, C04, C08, C12; T16; V01, V02, V04–V06, V11, V14, V15; R01–R03, G01, G04, G07 | **26** |
| DIVERGENT-SUSPECTED | N05, N07–N09, N12, N14; C02, C05; V03, V07–V10; R04, G02, G03, G05, G08 | **18** |
| UNMODELED | N10, N11, N15; C06, C07, C09–C11, C13; T09; V12, V13; G06 | **13** |
| **Total** |  | **74** |

The classifications deliberately separate source-local hazards from absent implementations. In particular, the 17 ALIGNED rows do not include any end-to-end No path; fifteen are stop/control taxonomy, one is cutoff non-authority, and one is the current fail-closed emitter gate.

## 3. Obligation verdicts and executable-checker architecture

### 3.1 CP-O14 — exact claimant enumeration

**Normative obligation, verbatim (`COMPLETENESS_SPEC.md:446`):** “Prove exact completeness of claimant candidate and atomic-pair enumeration against `AttackEdges_CP1`, including global defender-block inputs and unordered dedup.”

**Surface subset:** R01–R04, G01–G05, N07–N09, N12–N13, C02–C03, V05–V09, and bridge obligations B1–B4/B7/B8.

**Verdict: NOT DISCHARGED — the current solver violates the required both-order process, while final order/orientation mismatches remain separately suspected pending locked fixtures.** The source contains important high-level ingredients: global defender-threat blocks, frozen turn-start families, generated ordered encounters, and unordered dedup. But fresh/promoted second candidates make membership non-monotone and `SOL:6397-6401` explicitly permits only one generated order, whereas `CERT:83` requires both legal attacker-pair orders examined before quotient. Candidate/pair order is also heuristic/D6 rather than a named raw `Edge.before` sort, and the unordered table keeps first-encounter orientation instead of constructing a raw-low edge satisfying `Edge.CanonicalPair`; reachable final-list inversion fixtures remain to be locked. More fundamentally, Lean `RegenerationBinding.regenerateAttackEdgesGlobally` is still an arbitrary finite function satisfying laws; no independent Rust instantiation exists.

**Sharpest O14 risk:** proving only set inclusion or testing only counts. The checker consumes canonical ordinals and exact exhaustion. A set-equal but differently ordered vector, a reverse-oriented pair, or a missing remote defender block makes byte dispositions refer to the wrong proposition.

**Executable session plan:**

1. **O14-L1 — independent declarative attack spec (Lean).** In a new Lean module, define checked finite engine-shaped primitives over `CompletenessCP1.State`: global active windows, claimant/defender families, raw signed coordinate order, candidate aggregation, and an explicit raw-canonical edge satisfying `Edge.CanonicalPair`. Do not define it by calling `R.regenerateAttackEdgesGlobally`. Prove finiteness, raw ordering, and no-duplicates.
2. **O14-L2 — single-candidate equality (Lean).** Prove that the declarative candidate set is exactly claimant threshold candidates union all defender count≥4 block cells. Split set equality from list canonicalization. Cover the FirstStone pair route and SecondStone single route, terminal completion, tactical leaf availability, and checked H arithmetic; prove the Rust Opening route unreachable from a post-opening CP1 root/replay or classify it out of contract.
3. **O14-L3 — pair gate equality (Lean).** Model `WideTurnGate` as a pure turn-start record. Prove the second-candidate union is complete for strong continuation, frozen candidate, and weak promotion cases. In accordance with `CERT:83`, prove both legal directions are probed before quotient (or one direction is proved illegal), their classifications/final states agree, and the quotient classification equals the CP1 edge-local/pending classification. Current solver pruning does not meet this statement.
4. **O14-L4 — quotient and order (Lean).** Prove unordered dedup has one representative iff a pair is admitted, and prove a **normalizing projection** `(x,y)↦(min_raw,max_raw)` followed by `Edge.before` sort yields exactly `AttackEdges_CP1`. Do not attempt to identify the current raw Rust `Vec` unless O14-R0P's production normalization is an explicit premise; the current source lacks that projection, while a locked reachable inversion fixture remains part of deferred validation.
5. **O14-R0 — DEFERRED-NEEDS-CARGO, conditional refined-Rust checker route.** If O29-A0 selects a Rust checker, implement a new checker-owned CP1 attack regenerator with no call/import from solver generator code. It must globally rebuild inputs, probe both legal pair directions before quotient, reject duplicate identities, raw-low normalize, and sort by `Edge.before` before counts/ordinals are checked. A compiled-Lean route executes the proved Lean regenerator instead and does not create this Rust checker component.
6. **O14-R0P — DEFERRED-NEEDS-CARGO solver/emitter producer refinement.** Separately fix/refine the existing solver producer so CP-O14 search completeness probes both legal directions and its successful negative emitter writes ordinals for the canonical checker relation. It may propose bytes, but cannot be checker authority.
7. **O14-R1 — DEFERRED-NEEDS-CARGO Rust harness.** When the cargo slot is free, add a non-authoritative dump harness that serializes, for exact roots, (a) raw global candidate ingredients, (b) every generated encounter plus independent probes of the reverse evaluator/result tag, (c) checker-regenerator and solver-producer canonical outputs, and (d) final full child states. Include remote-block, one-order-only promotion, reverse-first, coordinate-boundary, terminal, and H-boundary fixtures. Compare against independently evaluated Lean fixtures; never import the solver vector as the Lean expected list.
8. **O14-X — bridge capstone (Lean + generated fixture evidence).** Under O27-A0's machine-checked source-semantics route, prove B2/B3/B4 for the chosen executed-checker regenerator and separately for the named Rust solver producer. Supply the attack function plus sorted/nodup/canonical laws for later O27-L9B assembly; do not pretend an `R` already exists. Fixture equality is regression evidence; the universally quantified refinements are the discharge.

### 3.2 CP-O15 — exact defender sequential quotient

**Normative obligation, verbatim (`COMPLETENESS_SPEC.md:447`):** “Prove exact completeness of batch defender enumeration against the sequential `DefendEdges_CP1`: kernel cells, canonical atomic pairs, reverse-key equality, and commutation expansion form an exact operational quotient, with ordinary singles on planner fallback.”

**Surface subset:** R01, R03–R04, G01, G06–G08, N11–N14, C05, V10–V13, and B1–B3/B5–B8.

**Verdict: NOT DISCHARGED — sound-looking local checks do not establish an exact quotient, and canonical list order is concretely divergent.** The planner is carefully all-or-nothing: it checks τ=2, each K2 first, exact τ=1/K1 after the first, membership of seconds in root K2, reverse directed pair, and equality of synthetic completed-pair keys; fallback exists. But exact equivalence to the Lean sequential relation and synthetic-key-to-full-state equality remain unproved. Independently, the xsnfyll fixture derivation in G07 gives a reachable two-pair counterexample: D6 kernel-rank order emits `(a,c),(a,b)` while raw `Edge.before` requires `(a,b),(a,c)`.

**Sharpest O15 risk:** proving every emitted pair is sound while failing the converse. One omitted K2→K1 direction, or a planner `None` case whose single fallback is not the exact sequential grammar, invalidates a Universal counterexample and therefore the negative checker.

**Executable session plan:**

1. **O15-L1 — independent sequential relation (Lean).** Define the threat family and extendable-hit kernels `K2(P)` and `K1(apply P x)` from global primitives. Define the exact directed sequential relation before quotienting. Prove membership/reflection for Lean `Budget.one` and `Budget.two`; separately prove every other Rust `u8` budget is rejected/unreachable under B7/B8, rather than inventing an unsupported Lean Budget case.
2. **O15-L2 — pair quotient soundness (Lean, after B3/B6).** From planner admission premises, transition correspondence, synthetic-key correctness, and full-key injectivity, prove each retained raw-low pair represents two legal, nonterminal directed paths, both end in the same exact state, and consumes two state/certificate levels. Source reverse-key equality alone is not state equality until B6 is proved. Relate the resulting state to `Edge.defenderPairQuotient`.
3. **O15-L3 — pair quotient completeness (Lean).** Prove every sequential K2→K1 obligation has its reverse, equal final state, and exactly one raw-low quotient representative whenever the planner returns `Some`. Prove the output after raw `Edge.before` sorting is duplicate-free and exact. This is the hard direction.
4. **O15-L4 — fallback theorem (Lean).** Characterize every `None` exit. For each, show ordinary single expansion equals the sequential `DefendEdges_CP1` obligation at that state; do not treat `None` as “no defense.” Prove the dispatcher is nonempty exactly where the positive Universal constructor requires it.
5. **O15-L5 — tight-dispatch executable component (Lean).** Define the tight-dispatch Boolean and prove the iff fact needed later for `ExecutableRegenerationBinding.checkTightDefenderDispatcherGlobally_iff`; do **not** construct `X` before the concrete `R` exists. Positive `CertCommutation` records are not serialized v1 authority; discharge commutation entirely inside the regenerated B5 quotient, then connect only the negative Universal's regenerated count, canonical chosen ordinal, selected disposition, replayed depths, and exact final state to `checkSelectedEdgeDag` (`CERT:167-180`; `CP1:2516-2527`). O27-L9B packages this component into `X`.
6. **O15-R0 — DEFERRED-NEEDS-CARGO, conditional refined-Rust checker route.** If O29-A0 selects a Rust checker, implement a new checker-owned sequential defense regenerator/quotient with no solver-generator import. It must globally rebuild K2/K1, prove reverse/final-state commutation, use raw-low `Edge.defenderPairQuotient` representatives, and sort by `Edge.before` before checking counts/ordinals. A compiled-Lean route executes the proved Lean relation instead.
7. **O15-R0P — DEFERRED-NEEDS-CARGO solver/emitter producer refinement.** Separately normalize/refine the existing planner and successful emitter to the same B5 relation; checker authority stays independent.
8. **O15-R1 — DEFERRED-NEEDS-CARGO Rust harness.** Differentially enumerate root K2, every post-first K1, all directed pairs, reverse pairs, actual two-apply final bindings, synthetic keys, planner result, fallback singles, and both independent-checker/producer canonical lists. Include the xsnfyll two-pair D6/raw inversion, asymmetric/reverse, terminal-first, empty-kernel, hash-collision, clock-boundary, and SecondStone-anchor fixtures.
9. **O15-X — capstone.** Under O27-A0's machine-checked semantics, prove B5 for both the chosen executed-checker relation and the named Rust solver/producer function. Prove B6's `PositionKey`/synthetic-key portion only for the producer; the checker's exact DAG-proposition/replay portion belongs to O27-X1/O29. Show each byte/ordinal path uses its proved function. Source-order independence must be an implementation fact plus theorem, not a modeled projection alone.

### 3.3 CP-O27 — Rust-to-model correspondence

**Normative obligation, verbatim (`COMPLETENESS_SPEC.md:459`):** “Prove Rust-to-model correspondence for state replay, full keys, leaf predicates, generators, tags, atomic edges, clocks, flags, and emitted trace serialization.”

**Surface subset:** all R/G rows; N01–N17; C01–C13; T01–T17; V01, V03–V15; and B0–B12. C13 is test evidence rather than production authority, but its byte/fingerprint paths must be distinguished from the real codec. O14 and O15 are dependencies, not substitutes.

**Verdict: BLOCKING / PROGRAM-CRITICAL, NOT DISCHARGED.** There is no Rust v1 negative parser/checker/root builder, no **functional** negative emitter (the existing `try_emit_no_tss_v1` always errors), no production CP1 profile binding, and no negative sealed mint. Current coordinate arithmetic can panic/wrap on a v1-admissible-size root. Current strict verification starts from a mutable engine state with incremental caches rather than reconstructing global primitives. Positive graph IDs run opposite v1 forward order. Durable `tss_proof` serialization is a provenance-free scalar that conflates two different negative meanings.

**Sharpest O27 risk:** a common-mode Rust checker that calls the same mutable generator/cache code as the solver. Even if it agrees on every test, it would not establish the independent regeneration required by `CERT:207-215`; a shared omission would mint a false completeness result.

**Executable session plan:**

1. **O27-A0 — choose the source-semantics/refinement route.** Before claiming correspondence to any named Rust function, choose a machine-checkable semantics/refinement method for the engine root/state builder, `apply_with_delta`/undo, global primitives, `PositionKey`/`WidePositionKey`, independent checker components (if Rust), solver generators, emitter tags, and trace serializer. Pin the exact function set, Rust revision, toolchain/features, environment semantics, and which functions will be replaced versus refined. Generated fixtures and a pinned revision are evidence, not a universally quantified semantics.
2. **O27-L1 — representation domains (Lean).** Define checked encodings/decoders for Rust `i16`, `u8`, `u16`, `u32`, `u64`, and bounded indices. Prove little-endian two's-complement bytes decode to the intended signed values and canonical root/list order is lexicographic on the **decoded signed** `(q,r)`, not on raw byte lexicographic order. Add explicit failure theorems for addition/subtraction/clock/depth overflow.
3. **O27-L2 — executable checker-local root/global component (Lean).** Define `rootFromV1` and a concrete Boolean root/global-rebuild predicate: validate canonical unique ownership entries, count=clock, schedule/player/phase, `SecondStone.first`, nonterminal status, exact external RootBinding, and globally rebuilt board/window/legal/threat facts with no history requirement. Prove its iff/soundness against the future concrete `rootEngineInvariant`. The complete `R/X` cannot be constructed yet because it also needs L3/L5 and O14/O15 generator fields.
4. **O27-R1 — DEFERRED-NEEDS-CARGO, conditional refined-Rust root builder/harness.** If O29-A0 selects a Rust checker, implement the checker-local root builder in the new negative-checker module, not by `STATE::load_state` and not by cloning solver state. Independently of the checker route, use a later Rust harness to dump reconstructed primitives and test malformed schedule/anchor/terminal/overflow roots. A compiled-Lean checker executes the L2 component directly and does not wait for a Rust checker-local builder.
5. **O27-L3 — transition and undo model.** Prove one checked placement corresponds to `apply_with_delta`, including phase schedule, absolute clock, terminal winner/count, legal anchor, and cache-independent resulting primitives. Prove undo restores the full modeled state. Treat machine-overflow cases as rejection.
6. **O27-L4 — exact full keys.** Prove `PositionKey::from_state` and `WidePositionKey::from_state` are injective over admitted states and equal exactly when model states are equal. Prove synthetic pair keys equal two real checked applications; forbid saturation under WellFormed premises rather than reasoning about saturated values.
7. **O27-L5 — global leaf predicates.** Prove window completeness, win-now, threat family, minimum hitting set, terminal, budget, and resolution equality. Connect Rust tags to `StateLeafShape`/`EdgeLocalShape`; separate N01 tactical Loss from semantic No.
8. **O27-L6/O27-L7 — generator bridges.** Execute the O14 and O15 proof plans and prove the attack/defense fields/laws intended for the concrete binding. These sessions must consume independently defined primitives from O27-L2/L5.
9. **O27-L8 — tag/clock/structural boundary map.** Give an exhaustive Rust tag conversion. Prove each positive/local/pending/refuted/cutoff case maps to a Lean constructor or explicit rejection, with exact state/cert increments. Prove claimant terminals are intercepted or remove N08 as negative evidence; prove replay failure rejects rather than refutes.
10. **O27-L9 — profile and environment binding.** Define one serialized/executed CP1 query object. Prove construction rejects unless its grammar version equals `CP1`, its grammar profile equals `CP1.profile` (definitionally `frozenProfile`), and the first producer manifest binds non-profile operational choices such as `TSS_LAZY_FRONTIER=0`; eliminate default/env ambiguity from authority. Wide search may remain the producer, but the checker rebinds the grammar/profile from bytes and is discovery-mode independent.
11. **O27-L9B — assemble the executable binding.** After L2–L8 and O14/O15, construct the single concrete `R : RegenerationBinding` and `X : ExecutableRegenerationBinding R`; discharge every sorted/nodup/replay/nonterminal/clock/root/tight iff law. This is the non-vacuity point consumed by `checkNoDag`.
12. **O27-L10 — trace contract.** Replace or explicitly de-authorize scalar `tss_proof`. Specify a durable record with source kind, full canonical root and query, grammar/profile, H/S/C, verdict kind, and retained certificate bytes/content address. A finite hash may index retained data but never replace exact root/query recheck. Prove Rust→Python→shard round trip; distinguish absent from Unknown and λ1 from checked No.
13. **O27-R2 — DEFERRED-NEEDS-CARGO differential/mutation campaign.** Run cross-language vectors for state, primitives, keys, leaves, generators, tags, canonical bytes, checker Boolean, and trace. Required hostile cases include the full `CERT:422-428` mutation suite plus coordinate overflow, action-zero sentinel-assumption collision, D6/raw order, reverse pair orientation, cache perturbation, duplicate proposition, reverse node IDs, and env-profile mutation.
14. **O27-R3 — DEFERRED-NEEDS-CARGO negative emitter/serializer.** Implement canonical negative-DAG extraction, provenance classification, root/query header binding, strict v1 serialization, and one-way failure handling. It must use the canonical producer relation proved in O14/O15 and may never turn `dn==0` alone into bytes or a verdict.
15. **O27-R4 — DEFERRED-NEEDS-CARGO durable trace implementation.** Implement the L10 record and Rust→Python→storage round trip, retaining exact canonical root/query and certificate bytes (or exact-rechecked retained content behind an indexing hash). Explicitly de-authorize legacy scalar `tss_proof` for completeness.
16. **O27-X1 — checker refinement capstone.** Under O27-A0/O29-A0, prove the chosen compiled executed checker's complete Boolean equals the structured Lean composition for B0–B5, B6's checker-facing exact full-state/`NoDagProposition` equality, B7's checker-facing edge increments/disposition rejection, B8's checker query/profile binding, and B10. This capstone does **not** claim correspondence for solver `PositionKey`/synthetic keys or solver PN/tag provenance.
17. **O27-X2 — producer refinement capstone.** Prove successful emitter output satisfies one-way B9, calls the actual canonical O14/O15 producer functions, and satisfies B6's solver-key/synthetic-key, B7's solver node/child-tag provenance, and B8's producer operational-profile premises. Failure remains Unknown.
18. **O27-X3 — sealed API capstone.** Prove B11 at the API/call graph: the unique negative mint is reachable only after X1 acceptance for the exact external root, with no alternate constructor/bypass.
19. **O27-X4 — trace capstone.** Prove B12 independently across Rust/Python/storage. Compose X1–X4 only at the end into authority→`NoContractWin`; tests are regression evidence, not the discharge.

### 3.4 CP-O19 shared DAG core and the requested executed-checker architecture

There is a numbering collision in the request that matters for gating. Normative **CP-O19** is (`COMPLETENESS_SPEC.md:451`): “Prove negative DAG sharing sound: exact repeated `(state,Q)` propositions may share, graph is acyclic/well-founded, and unfolding preserves every exhaustion obligation.” The “connect the proved checker to the executed checker” obligation is normatively **CP-O29** (`COMPLETENESS_SPEC.md:461`). The architecture below answers the requested topic while preserving both numbers: CP-O19 supplies the proof-level DAG core; CP-O29 connects that core to compiled execution and depends on CP-O27.

**CP-O19 verdict:** the direct shared-DAG semantic/checker core is landed, but the normative obligation is **not literally fully discharged**. `NoDagProposition`, forward child indices, unique proposition checking, reachability, and well-formedness are at `CP1:2110-2317`; direct reverse-index soundness at `CP1:2319-2465`; executable `checkNoDag` at `CP1:2480-2614`; `checkNoDag_sound` at `CP1:2887-2953`; and `checkNoDag_iff` at `CP1:3164-3212`. The proof deliberately uses direct DAG induction and preserves Choice exhaustion/selected Universal evidence at every exact incoming replay, but no named extensional unfolding theorem proves the spec's literal “unfolding preserves every exhaustion obligation” clause. Under the current normative text, that theorem is required. Amending the spec would be a separate scope change and would leave the original CP-O19 statement unmet rather than proving it.

**CP-O29/executed-checker verdict:** normative CP-O29 says (`COMPLETENESS_SPEC.md:461`) “Prove checker termination/resource bounds and connect the proved checker to the executed checker without importing solver generator code as authority.” The structured Lean tree/DAG checker cores are already structurally fuel/count bounded (`LEDGER_CP1.md:32,40`) and the suffix codec declares deployment caps (`LEDGER_CP1.md:44`). Full-file parser/primary-replay resource bounds, compiled stack/heap behavior, and the executed-checker Boolean/binary/artifact bridge remain open; no production code builds the structured inputs from v1 bytes or proves equality. A Rust-specific refinement boundary applies only if O29-A0 selects that route.

The pinned Lean ledger/journal numbering is not normative: committed `LEDGER_CP1.md:37-43` groups the shared DAG under CP-O18, while committed `LEDGER_CP1.md:62` and `JOURNAL_CP1.md:199-201` call compiled correspondence CP-O19. This map follows `COMPLETENESS_SPEC.md:451,461` and `CERT:385,387`: sharing/primary replay is CP-O19/CP-O26; executed termination/bridge is CP-O29.

#### Minimal architecture and trust boundary

```text
untrusted solver arena / SearchStop / bytes
                  |
                  v
 [strict bounded v1 parser] --reject--> UNKNOWN
                  |
                  v
 [exact external-root match + checker-local WellFormedCP1 rebuild]
                  |
                  v
 [canonical B2/B4/B5 regeneration + primary replay table]
                  |
                  v
 [structured checkNoDag-equivalent bounded checker]
                  |
          accept  v
 [sealed NO_CONTRACT_WIN for this exact root/query/bytes]
```

The solver, proof numbers, transposition table, stage scheduler, incremental caches, positive verifier, and negative emitter are all **untrusted producers**. They may propose bytes or fail; none belongs in the semantic TCB.

The minimal semantic TCB is:

1. the exact CP1 definitions and proofs checked by the Lean kernel, especially `checkNoDag_sound`;
2. the strict byte decoder and bounded scalar/arithmetic implementation corresponding to B0;
3. the checker-local root/global-primitive implementation corresponding to B1/B2, including legal/window/terminal reconstruction;
4. primary-path replay-table construction (CP-O19/CP-O26), exact full-state equality, canonical B4/B5 regeneration, and the structured checker corresponding to B3–B10; its compiled realization is also in CP-O29;
5. the tiny sealed-mint call site corresponding to B11;
6. the compiler/runtime/binary-integrity assumptions for the actual checker artifact. A handwritten Rust checker adds the Rust implementation and compiler to the TCB unless a proof-producing translation or machine-checked refinement theorem closes that gap. A compiled-Lean route still requires a stated compiler/runtime trust story **and** an optimized streaming Lean checker; directly compiling the stable reference `checkNoDag` does not meet the intended deployment bound.

The stable `checkNoDag` is the semantic reference, not yet the minimal-memory deployed algorithm. Its `NoDagProposition` table stores a full `State` per node, uniqueness uses pairwise containment, and reachability scans prior nodes (`CP1:2157-2161,2576-2597`). That can be full-state-per-node and quadratic, unlike `CERT:359-374`'s fixed metadata plus active/primary replay and work bound `N + D + Refs×C + N×Gmax`, where this document renames the spec's repeated-reference count `R` to `Refs` to avoid collision with `RegenerationBinding R`.

There is a sharper unresolved architecture blocker inside that contrast: exact rejection of two IDs for the same `(state,stateDepth,certificateDepth,Q)` needs an exact `seenPropositions` service (`CERT:100-106,225-258`). A finite hash is only a prefilter and cannot authorize equality. Storing a canonical full proposition key is state-sized per node; replaying on hash collisions has adversarial quadratic replay cost; external sort or a persistent exact-state dictionary changes the heap/I/O/work model. Therefore the current fixed-metadata statement and linear-looking envelope are **not yet simultaneously realized**. O29-A1 below must select a concrete exact algorithm and either prove its worst-case resources inside the normative limits or trigger an explicit spec/resource-envelope amendment. Until then, “optimized streaming checker” is an open design problem, not an implementation promise.

What the compiled checker must correspond to is the **whole composition**, not merely the recursive Boolean: strict bytes→file, external-root equality, `WellFormedCP1`, global primitive rebuilding, primary replay table, `checkNoDag`, all-reached/unique proposition checks, exact consumption/limits, and sealed rejection/acceptance. Correspondence to the solver's current positive `TssVerifier` is irrelevant because that verifier implements a different grammar.

#### Effect of the landed codec

The landed node-stream codec changes the proof shape usefully: the executable bridge can be factored into `bytes decoder equality` followed by the already proved structured `checkNoDag` theorem, rather than proving byte parsing and semantic recursion in one monolith. It does **not** shrink the remaining boundary to zero. The pinned stable work covers node-suffix encoder/decoder declarations and one concrete fixture round-trip; no general node-stream or full-file inverse theorem is landed, and the full 86-byte file syntax/parser, replay-table constructor, and bytes-to-`NoAt` capstone are absent from the committed revision. Dirty R-CP9 additions are treated only as concurrent work, not as evidence. Therefore the current deployable theorem still starts at structured syntax, not arbitrary external bytes.

**Recommended checker sessions:**

1. **O19-L1 (required):** state/prove an explicit DAG-unfolding semantic equivalence preserving every Choice exhaustion and Universal selection. A later normative scope amendment may supersede this task, but it is not a proof of the current obligation.
2. **O29-A0 — choose and freeze the checker refinement route.** Choose compiled optimized Lean, formally refined Rust, or a proof-producing translation. Pin source revision, compiler/toolchain, runtime assumptions, build configuration, and eventual artifact hash. This determines what “compiled checker equivalence” means; coordinate it with O27-A0, which separately supplies machine-checked semantics for named Rust engine/producer functions.
3. **O29-A1 — exact proposition-index design gate.** Choose the concrete exact `seenPropositions` algorithm: canonical full-state keys, a proved injective representation/persistent dictionary, external exact sorting, or replay with an honestly enlarged worst-case bound. Specify collision handling, equality witnesses, heap/I/O, and adversarial time. A finite digest alone is forbidden. Either prove the choice within `CERT:359-374` or record the required normative resource amendment before coding.
4. **O29-L1 — full codec.** Define and land a committed deployable codec-domain predicate (the dirty R-CP9 draft currently uses a `CodecWellFormed` name), then prove strict scalar/node/full-file inverse theorems on that domain and rejection of every malformed/out-of-domain byte string.
5. **O19/O26-L2 + O29 bridge — primary replay.** Define and prove primary replay-table construction from root and primary incoming edges; prove exactness, uniqueness, reachability, and bounded termination, then refine its compiled implementation.
6. **O29-L3 — optimized checker refinement.** Define the active/primary-replay checker plus O29-A1's exact proposition-index service. Prove its Boolean equals the structured reference `checkNoDag` on decoded well-formed inputs. Do not claim fixed metadata, no pairwise scans, or linear time until the selected exact service proves those properties.
7. **O29-L4 — formal resource theorem.** Let `RootEntries` count decoded root stones and `Refs` count repeated references. Derive a computable `Gmax` and generator-workspace bound from the O14/O15 finiteness proofs. Bound strict parsing/offsets, root-board construction, every global legal/window/threat rebuild, checked allocation/count arithmetic, the chosen exact proposition index, forward recursion/stack, active/primary replay state, and failure→false. Then prove the selected route's honest worst-case time and heap; if O29-A1 supports the intended envelope, state it as `RootBuild(RootEntries) + N + D + Refs×C + N×Gmax` up to explicitly bounded primitive costs. Measurement is regression evidence, not this proof.
8. **O29-L5 — bytes-to-No.** Compose decoder, executable root/global binding, exact-index/optimized-checker refinement, `checkNoDag_sound`, exact external binding, and `NoContractWin`.
9. **O29-R1 — DEFERRED-NEEDS-CARGO, conditional refined-Rust implementation/evidence.** If A0 selects Rust, implement the independent bounded checker and mutation harness without solver generator state. Stack/heap/limit measurements validate assumptions and catch regressions but are not the resource proof.
10. **O29-E1 — chosen-route build/evidence.** For compiled Lean or a proof-producing translation, build the selected checker with the pinned non-Cargo toolchain and run the same mutation/resource evidence; for refined Rust, this is supplied with R1. This campaign performs none of those builds.
11. **O29-X — named artifact theorem.** Under O29-A0's compiler/runtime assumptions and O29-A1's exact-index/resource theorem, connect the pinned source/toolchain/artifact hash to the proved Boolean/refinement theorem, then place B11 immediately after acceptance and nowhere else.

## 4. Hostile self-review

### 4.1 Independent resweep method

Phase 1 began at production authority and descended the call graph. The hostile resweep did the opposite: it ignored entry points and searched the entire Rust/Python tree for construction sites, sentinel values, enum matches, byte paths, and output fields. The exact search classes were:

```text
ProofStatus::Loss | HardValue | forced_loss | verdict
WidePnNode::Refuted | WidePnChildResult::Refuted | dn == 0
LOCAL_TT_FAILED | return None | exhaust | refut | negative
NoTssCertificateV1 | TssCertificate { | CertNode::
SearchStop:: | RunUntilExit:: | StageEvent::
verify | check | encode | decode | bytes | certificate
tss_proof | proof_status | shard | npz
#[cfg(test)] and module declarations in lib.rs
```

For each hit class, the resweep then searched every constructor, every match arm, and every caller. It separately inspected all 80 Rust source files under `packages`, sibling/reference solvers, Python serialization, and process/CLI exits. This is independent of the Phase-1 call graph because a hidden constructor with no expected caller name still appears in the type/construction sweep.

### 4.2 Surfaces added by the second sweep and reconciliation

| Second-sweep discovery | Why Phase-1-style reasoning could miss it | Reconciled row(s) |
|---|---|---|
| Shared λ1 `ThreatAnalysis` is a production hard `-1` producer used outside `hexfield_eq`. | It is included via a path import, not under a TSS solver filename. | N01 plus listed sibling consumers. |
| HexGNN duplicates the λ1 formula locally over a different window iterator instead of importing the shared implementation. | A shared-module call graph would miss it, and formula-shaped code does not prove Rust↔Rust extensional equality. | N01, with the local implementation/consumers cited and parity left open. |
| Two leaf-search backup loops and per-root-move classification consume λ1 HardValue directly. | They are outside the root deep-proof block and do not carry a certificate. | N01 (`SEARCH:2318-2354,2460-2496,4591-4618`). |
| `tss_reference` and `tss_reference_fast` can return Loss. | They are independent modules, not called by the primary production authority path; fast is test-gated. | N03, N04. |
| Narrow `LOCAL_TT_FAILED` caches a conflated no-proof fact and exposes no `SearchStop`. | Frozen CP1 intends wide search, so a profile-led walk can exclude it too early; production defaults currently route there. | N15, G01. |
| Failed selected-child `apply_with_delta` becomes semantic Refuted. | The constructor occurs in selected-edge work, not expansion; stored future-key equality is only debug/test checked. | N13, with the unchecked-key premise routed to B6. |
| Positive compaction uses child-before-parent IDs; cached proofs enforce `child<parent`. | It is a generic graph shaper far from materialization. | C08, V02. |
| D6 certificate remapper constructs a full certificate. | It is public in verifier code but has only test callers in the current caller sweep. | C09. |
| Test-only cap-resume constructor and two byte-fingerprint encoders. | Names contain “bytes/certificate” but are below test gates and are not v1. | C13. |
| Horizon preflight can cause a retry/downgrade before verification. | It shapes acceptance control rather than nodes. | C10, V03. |
| Narrow zone shapers and positive zone rederivation helpers. | Outside initial CP1 flags; some helpers are production-compiled but only test-called. | C04, C07, V12. |
| Actual tree sealed mint and root search merge; production drops wide stop. | Search-local return values do not show the later authority boundary. | N16, V15, post-T17 note. |
| Async result drops the certificate. | Separate worker/transport module under Python feature. | C11. |
| Durable `tss_proof` is only `int8` and conflates λ1/deep provenance. | The byte path is Rust→PyDict→Python→NPZ, not a TSS codec. | C12. |
| No builder can reconstruct engine state from v1 final ownership entries. | `load_state` sounds suitable until its ordered-history input is inspected. | R01, R03, V14. |
| Unchecked coordinate arithmetic and packed-zero sentinel-assumption collision without a nonzero invariant. | Neither appears under certificate names. | R02. |
| Default solver is narrow and environment flags are unbound. | The grammar divergence is at constructor/configuration, not generator body. | G01, T16. |
| Static replay of the xsnfyll K2 fixture through the source D6 transform gives a two-pair canonical-order inversion. | Comparator choice alone was only suspected until the existing reachable fixture supplied two retained pairs and an inverted rank. No Cargo execution was used. | G07; witness cited in §2.6. |

Every added item is now in the 74-row inventory. No second-sweep hit created a hidden production `NoContractWin` emitter or checker.

### 4.3 Construction-site reconciliation

The hostile type counts reconcile as follows:

- All production `TssCertificate { ... }` constructions are immediate (`SOL:951`), wide root (`SOL:1258`), fragment promotion (`SOL:1316`), narrow (`SOL:4330`), and D6-remap return (`VER:1694`); cap-resume at `SOL:1561` is test-only. These are C02–C09.
- All remaining test-only certificate literal constructors, mutations, fingerprints, preflights, and direct verifier/oracle consumers are grouped in C13 under the complete `cfg(test)` module spans `SOL:10830-14793`, `VER:1703-2330`, and the seven test modules cited there. They are not alternative production authority.
- All ten `SearchStop` variants are declared at `SOL:2327-2368`; constructions occur in preflight/immediate paths `SOL:916-972`, materialization downgrade `SOL:1285-1288`, the pending-candidate helper `SOL:2370-2379`, and wide-run paths `SOL:4661-4747`; one-per-variant fixtures are `SOL:12913-12974` with assertions at `SOL:13033,13078,13103`. T08–T17 contain exactly ten rows.
- All six `RunUntilExit` variants are declared at `SOL:2254-2265` and returned at `SOL:4808-4863`. T02–T07 contain exactly six rows. `run_resumable` (`SOL:4754-4805`) is test-only and returns no stop, so it is recorded as a test driver rather than a seventh variant.
- `StageEvent` has exactly one variant at `SOL:2244-2252`, recorded as T01.
- All source-logic wide Refuted causes found are the post-cutoff horizon check (production-dead under current staged caps), terminal, unavailable/opponent leaf, optional census, invalid tight defender boundary, empty generated branch, selected-child move-application failure, ordinary-defense opponent terminal, and recurrence from refuted children. They are N05, N07–N14. N05 also cites every hand-built Refuted unit fixture. `DepthCutoff` is separately N06 because equal numbers do not make equal provenance.
- The only negative byte carrier is `NoTssCertificateV1` (`SOL:2318-2325`). The only current byte construction is the invalid test fixture `NTSSCP1\0fixture` (`SOL:12918-12920`). The emitter always errors. There is no `parseStrictV1`, `decodeNoTss`, `checkNo`, or negative verifier entry in Rust.
- `PreconditionFailure::{UnsupportedCP1Root,UnsupportedProfile}`, `NoEmitFailure::{StructuralCostMismatch,NegativeDagLimit}`, and `InvariantCode::ImpossibleRunUntilFallthrough` are declarations without construction sites. Their presence must not be mistaken for executed checks.

### 4.4 Hostile attacks on the correspondence claims

**Attack: “ProofStatus::Loss already means no.”** Rejected. N02 requires a positive certificate for the opponent, while N01 is a one-turn no-hitting-set fact. Neither follows the CP1 finite-dual grammar, and C12 merges them into the same scalar.

**Attack: “`dn==0` is enough because proof numbers are exact.”** Rejected. N06 shares `(INF,0)` with Refuted, and N05 can acquire refutation from execution failures or unproved generator exhaustion. The CP4 gate correctly refuses all such arenas today.

**Attack: “The strict positive verifier can be reused.”** Rejected with concrete grammar witnesses. V05 accepts a legal claimant Choice outside exact `AttackEdges_CP1`; V09 checks a supplied threat-window subset rather than complete global-family equality; V11 accepts a strict superset of the implicit defender kernel; V02 accepts the reverse ID orientation and duplicate propositions. Its commutation check compares only outcome options, and global replay trusts incremental caches. Reuse would change both quantifiers and trust boundary.

**Attack: “Lean soundness closes the Rust side because `R` is abstract.”** Rejected. `RegenerationBinding` supplies arbitrary finite attack/defense functions plus laws about their own lists. An empty binding can make the checker sound relative to an empty grammar. B2/B4/B5 are the non-vacuity bridge that identifies those functions with independent CP1/Rust semantics.

**Attack: “Set equality is enough.”** Rejected. V1 stores generated counts, chosen ordinals, and list-aligned dispositions. `Edge.before` is part of the grammar ID. G02/G04/G07 can be set-equal yet byte-incompatible.

**Attack: “RootBinding equality makes cache use safe.”** Rejected. RootBinding omits the internal legal/window caches. V14 may observe state not determined by its bound fields. The normative checker explicitly requires a global rebuild.

**Attack: “Machine range is irrelevant because normal games stay near origin.”** Rejected as an authority theorem. The v1 caps permit a sparse chain with roughly 4096 stones reaching q=32760; radius generation then computes an out-of-range `i16`. Either the grammar must impose/prove a tighter coordinate closure domain or checked arithmetic must reject.

**Attack: “The full Lean codec is landed.”** Rejected. Pinned stable lines through `CP1:3467` cover node-suffix declarations and one fixture only; full-file syntax, general inverse, replay construction, and bytes-to-No are absent from committed `0e1bdabd…`. Dirty R-CP9 state is not gate evidence.

### 4.5 Exclusions revalidated

- Ordinary engine game-terminal output is not a CP1 contract-refutation proposition.
- Sibling VCF Boolean `false` paths conflate refuted/unproved but have no CP1 certificate or hard mint; they are not hidden No authority.
- `tss_reference` is production-compiled but its discovered callers are tests/corpus comparison; it cannot mint a sealed result.
- Hunt/corpus modules below `#[cfg(test)]`, debug dispatcher oracles, round3 shadow helpers, and certificate fingerprints are evidence generators only.
- `.hxr` stores moves/outcome, not `tss_proof`; NPZ `tss_proof` is training metadata and is omitted from expanded training labels. It remains included in C12 because CP-O27 explicitly names emitted trace serialization.
- No CP1 Rust `main`, process exit, alternate FFI verifier, or second negative parser was found.

### 4.6 Cold-reader gate

Keep `NO_CONTRACT_WIN` unreachable unless every item below is evidenced by a named theorem, source revision, and hostile test suite:

1. O14 B4 exact ordered attack equality, including canonical pair orientation and remote blocks.
2. O15 B5 exact sequential defender quotient and fallback equality.
3. O27 B0–B8 state/primitive/key/leaf/tag/clock/profile correspondence with checked arithmetic and independent rebuild.
4. O19 model-side DAG soundness pinned to the stable Lean revision, with the required explicit unfolding-preservation theorem under the current normative statement; O19/O26 primary replay-table correctness also landed.
5. O29 strict full-file inverse, compiled replay-table/checker correspondence, bytes-to-No theorem, named-binary equivalence, and full parser/stack/heap/resource argument.
6. O28/B11 one sealed mint reached only after exact-root checker acceptance.
7. B12 trace distinguishes λ1, opponent-positive Loss, checked No, Unknown, and absence, and binds retained bytes to the exact root/query.

Until those seven gates pass, the correct public result for every negative-search exhaustion, cap, cutoff, cache failure, parser rejection, or checker rejection is Unknown/Incomplete.

## 5. Execution-order recommendation

The shortest dependency-respecting campaign is:

1. land the required O19 unfolding theorem, freeze/finish the full-file Lean codec, and define the primary replay-table reference without waiting for Rust;
2. choose O29-A0's checker route, O27-A0's Rust source-semantics route, and O29-A1's exact proposition-index algorithm; then define checked representation domains and executable root/global primitives (O27-L1/L2), but do **not** construct `R/X` yet;
3. prove the model/reference side of transitions, key injectivity, leaves, tags/clocks, and grammar/operational-profile binding (B3/B6–B8), which is prerequisite to interpreting planner keys and generated children;
4. prove the independent O14/O15 declarative set/quotient/order equalities and finiteness/workspace bounds in parallel, supplying the attack/defense functions and laws for later assembly; no theorem about absent or unfixed production normalization code is claimed here;
5. after transitions, leaves, O14/O15 fields, and all binding laws exist, construct the concrete `R/X`; define the selected exact-index/streaming reference checker, prove its equivalence and honest resource bounds, and compose structured bytes-to-No at the model level;
6. in later build-enabled sessions, implement the Rust producer fixes, negative emitter, trace, harnesses, and—only if O29-A0 selected refined Rust—the independent Rust checker components marked **DEFERRED-NEEDS-CARGO**. A compiled-Lean/proof-producing route instead builds its selected executed checker with the pinned non-Cargo toolchain. Run differential/mutation suites as evidence;
7. only after those implementations exist, prove the pinned source/artifact refinements, bind the source/toolchain/artifact hash, and land checker equality, one-way emitter refinement, B11 call-site uniqueness, and B12 trace theorems together so no intermediate revision exposes an unchecked negative mint.

The single sharpest endgame risk remains **B2/B4/B5 non-vacuity**: accidentally proving the Lean checker sound for an abstract or common-mode generator while never proving that the independently rebuilt Rust frontier is exactly the frozen CP1 frontier. That failure can survive unit tests, valid-looking bytes, DAG soundness, and a fail-closed stop taxonomy while still making “complete” mean the wrong grammar.
