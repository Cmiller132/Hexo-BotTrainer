# R-CP1 completeness specification: certified width-contract exhaustion

**[DOCUMENT STATUS]** AUTHORITATIVE DESIGN, NOT A PROOF

**[CODE FACT]** Engine snapshot: `hunt/completeness` at `a49e8abd97cd49ffb2c653e23e62d51c8103cc38`

**[PRIOR-ROUND FACT]** Snapshots: `hunt/census-deep` at `a70e4b37b6e924a55b1dc3434beb00c6f5bc0a48`; `hunt/cert-support` at `408dc5b6fa2ca923eb9ab50c09ef84418b761a92`

**[LEAN FACT]** Snapshot inspected read-only: `E:\tss-lean` at `120231da50a5d7aeaf1388fcac8cacb2702ba8b3`, including the spine lane's live, uncommitted edits in `LEDGER.md`, `TssZones/ForcedHit.lean`, and `TssZones/Soundness.lean` as observed at final QA; all were left untouched

**[DOCUMENT STATUS]** Date: 2026-07-17

## 0. Claim labels and executive decision

**[DESIGN]** Every substantive assertion in this document is prefixed by one of these labels: **DOCUMENT STATUS**; **CODE FACT** (read from the named engine snapshot); **LEAN FACT** (read from the named spine snapshot and its final ledger `Status` column); **PRIOR-ROUND FACT** (read from a named retained research artifact); **DESIGN** (the proposed formal contract); **PROPOSAL** (an engine/checker change, not implemented here); **ASSUMPTION** (a premise that must later be discharged or kept explicit); **OBLIGATION** (proof or refinement work); **ESTIMATE**; **RISK**; **NON-CLAIM**; or the decision labels **RECOMMENDATION**, **DECISION**, **GATE**, and **KILL**. A qualified label such as `[CODE FACT — MAJOR FINDING]` retains the status of its leading label.

**[CODE FACT — MAJOR FINDING]** The engine cannot currently distinguish a capped `UNKNOWN` from an exhausted `UNKNOWN`. `ProofStatus` deliberately merges “capped / exhausted / unproven” (`packages/hexfield_eq/rust/src/tss_core.rs:24-32`); `unknown()` carries only `Unknown`, no certificate, and generic statistics (`packages/hexfield_eq/rust/src/tss_solver.rs:1725-1730`); and the production result exposes neither a stop reason nor root proof/disproof numbers (`packages/hexfield_eq/rust/src/tss_solver.rs:676-679,1093-1096`). Consequently, the target theorem is not statable against today's public result type.

**[RECOMMENDATION]** Pursue route **(b), an exact-root exhaustion certificate plus an independently specified checker**, as the production authority. Use a small Lean functional grammar to prove the checker sound, then prove operational search refinement in layers. Do not make a paper-only Rust-to-Lean correspondence the authority for a hard no-win conclusion.

**[DESIGN]** The intended result taxonomy is:

| Result | Authority | Meaning |
|---|---|---|
| **[DESIGN]** `WIN(cert)` | existing strict positive verifier | the named claimant has a verified forcing proof |
| **[DESIGN]** `LOSS(cert)` | existing strict positive verifier for the opponent claimant | the root side to move has a verified loss |
| **[DESIGN]** `NO_CONTRACT_WIN(no_cert)` | new exhaustion checker | the exact, versioned width contract contains no positive derivation |
| **[DESIGN]** `UNKNOWN(Capped | Incomplete)` | none | no semantic conclusion |

**[NON-CLAIM]** `NO_CONTRACT_WIN` is not “the game position is dead” and is not “the ordinary strict verifier would reject every possible WIN certificate.” The R-CF1 separation says exactly that a forcing-grammar negative is horizon-free only relative to the current generator, while strict-verifier `Choice` nodes accept arbitrary legal placements and do not re-check `WideTurnGate` (`hunt/census-deep:CENSUS_CANDIDATES.md:9-23`).

## 1. The exact objects

### 1.1 Exact root, claimant, and grammar version

**[DESIGN]** A query is a tuple

```text
Q = (grammar_id, root, claimant, semantic_horizon,
     state_depth_cap, certificate_depth_cap)
```

**[DESIGN]** Here `root` is the complete state: sorted occupied coordinates and owners, current player, exact `TurnPhase` (including `SecondStone.first`), absolute `placements_made`, and terminal outcome. This deliberately matches the fields of `RootBinding` (`packages/hexfield_eq/rust/src/tss_verify.rs:39-77`), not a local support projection.

**[DESIGN]** For R-CP1, `claimant = root.current_player`. The principal theorem covers `SolveGoal::Win` and the identical primal arm of `SolveGoal::Both`. In pair-complete mode, `Both` assigns all post-root budget to that primal restricted-WIN attempt and assigns zero to the independent dual attempt (`packages/hexfield_eq/rust/src/tss_solver.rs:927-937`).

**[DESIGN]** The initial `WellFormedCP1(Q,P)` domain is exact: `P = Q.root`; the canonical `RootBinding` reconstructed from `P` matches the query; `claimant = P.current_player`; `P` is nonterminal and post-opening (`FirstStone` or `SecondStone`, not `Opening`); the board/phase/clock and `SecondStone.first` satisfy the engine's state invariants; root occupancy is within `MAX_CERT_ROOT_STONES`; `H >= p0`; `S` and `C` have the values in Section 1.2; and the frozen flags/grammar ID below match. Terminal and opening inputs retain today's fail-closed/immediate-verdict behavior but are outside the first completeness theorem. A later opening theorem needs a distinct proved domain extension.

**[DESIGN]** `grammar_id = CP1-a49e8abd-v1` freezes the following semantic choices:

- **[DESIGN]** `WidthOptions { vcf_pair_complete: true, quiet_turn_or_edges: Round3Flag::Off, ranked_unforced_defender_zone: Round3Flag::Off }`, exactly the constructor at `packages/hexfield_eq/rust/src/tss_solver.rs:620-627`.
- **[DESIGN]** Base wide-PN search, not the round-3 narrow-compatibility path selected when both quiet-turn and ranked-zone consumption are enabled (`packages/hexfield_eq/rust/src/tss_solver.rs:1018-1047`).
- **[DESIGN]** The exact batch defender-pair plan is the canonical grammar. `TSS_INCR_DEFENDER=1` may implement it only under the structural-identity obligation in Section 3.
- **[DESIGN]** `TSS_LAZY_FRONTIER` is an operational representation choice, not a grammar change: eager and lazy executions must denote the same child relation.
- **[DESIGN]** In the initial covered profile, shared positive fragments, round-3 consume, finite-horizon interior census pruning, K-reply consume, experimental PN priors/deltas, and narrow negative caching are **all off**. Section 6.1 records how each can be admitted later.

**[CODE FACT]** The official deep corpus profile at this snapshot is 1 GiB with `TSS_LAZY_FRONTIER=1` and test-harness `TSS_INCR_DEFENDER=1` (`docs/TSS_RUNBOOK.md:10-25,32-43`). The incremental mode is nevertheless `cfg(test)` in the solver; non-test release code always uses the batch planner (`packages/hexfield_eq/rust/src/tss_solver.rs:2800-2819,3923-3931,6345-6346`). The theorem must state which compilation/profile it covers rather than calling the environment flag a production-wide semantic fact.

### 1.2 Semantic and structural horizons

**[CODE FACT]** `SolveCaps.semantic_horizon` is an **absolute placement index**, not a work budget (`packages/hexfield_eq/rust/src/tss_core.rs:92-107`). A positive certificate carries the caller's cap, the verifier derives the actual maximum leaf resolution `T`, and acceptance requires `T <= semantic_horizon` (`packages/hexfield_eq/rust/src/tss_verify.rs:142-152,200-209`).

**[DESIGN]** Let `p0 = root.placements_made`, `H` be the requested absolute semantic horizon, and `C = MAX_CERT_DEPTH = 256`. A production CP1 query is well-formed only when `H >= p0`, its engine-state depth cap is

```text
S(Q) = min(H - p0, C),                  C = 256,
```

**[DESIGN]** The horizon component of `WellFormedCP1` additionally requires all conversions and additions used by leaf resolution, state depth, and atomic-edge expansion to satisfy their stated checked/saturating arithmetic contracts; an overflowed or inconsistent query is rejected, never exhausted.

**[CODE FACT]** `S(Q)` is the engine's actual **state-placement-depth** cap: `wide_search_final_depth` is `min(semantic_horizon - root_ply, MAX_SEARCH_DEPTH)` (`packages/hexfield_eq/rust/src/tss_solver.rs:2082-2088`), and `MAX_SEARCH_DEPTH = MAX_CERT_DEPTH` (`packages/hexfield_eq/rust/src/tss_solver.rs:458-460`). The verifier's `C=256`, however, bounds recursive certificate-node depth (`packages/hexfield_eq/rust/src/tss_verify.rs:25-26,264-295,468-474`). Those clocks coincide on ordinary one-placement recursive edges but are not definitionally identical for atomic pairs or edge-local leaves.

**[DESIGN]** `ContractWin(Q,P)` therefore carries an exact structural rank from the verifier grammar, not merely `P.placements_made - p0`. A current-state typed leaf costs no further recursive edge; a single placement and an atomic two-placement turn cost exactly what their expansion into `Choice`/`Universal`/`OrCompletion` nodes costs. A positive constructor exists only when its fully expanded strict-certificate path stays at depth `<= C` and its declared resolution stays at `<= H`. `S` bounds admission of a successor as another recursive search state; it does **not** by itself suppress an edge-local completion/tactical leaf whose applied endpoint has placement depth `> S`. Such an edge-local success remains in-contract when it is generated from an admitted parent and satisfies `H` and its exact expanded `C` cost. Equivalently, a contract win is a CP1 forcing derivation that can be expanded into the strict positive grammar at the covered depth—not an arbitrary legal-move proof.

**[NON-CLAIM]** The strict verifier's node/edge/witness/commutation/memo limits are deployment bounds, not semantic ways to make a winning grammar derivation disappear. They are deliberately **not** included in `ContractWin`. The Rust-facing finding theorem therefore needs CP-O31: prove every PN-closed contracted win can be materialized under those limits, continue search after an oversized proof, or change the certificate/checker representation. Without CP-O31, only the abstract derivation-finding theorem is unconditional.

**[CODE FACT — MAJOR FINDING]** The current solver does not enforce that exact structural cost while generating. It cuts off a state only when `state_depth > depth_cap` (`packages/hexfield_eq/rust/src/tss_solver.rs:5725-5733`); `attack_pair_children` ignores its depth parameter, and edge-local completion/tactical results create no child entry on which the cutoff could fire (`packages/hexfield_eq/rust/src/tss_solver.rs:5908-5908,6104-6123,8523-8536`). An atomic proof near depth 256 can thus close PN before its expanded certificate is verifier-admissible. CP-O16/CP-O31 must either prove this unreachable, add an exact edge-height guard, or make materialization failure resume another admissible proof search. Until then Candidate B is conditional at the Rust verdict boundary.

**[NON-CLAIM]** The official `H = u32::MAX` profile is semantically unbounded only in the campaign's sentinel sense. Recursive-state admission is still bounded by `S(Q)` and expanded certificate recursion by `C`; executed verifiers also have fail-closed artifact limits. The theorem must say “within the exact query bounds,” never simply “unbounded game search.”

**[DESIGN]** The primary R-CP1 theorem instantiates the official sentinel horizon and the resulting structural bounds. This query-bounded result is “horizon-free” only in the R-CF1 forcing-grammar sense that it is not a no-six-by-finite-deadline-`T` census statement; it is not literally unbounded. A secondary finite-`H` theorem may decide only derivations whose exact typed-leaf resolution is at most `H`; it must be named `NoContractWinBy(Q)`, not silently presented as an unqualified grammar-negative.

**[PRIOR-ROUND FACT]** R-CF1 separately defined a stage-bounded semantic certificate (“no six by absolute deadline `T`”) and a forcing-grammar certificate (“no positive derivation in this generator”). A proof resolving after `T` is a `late_win`, while any PN=0 subtree refutes a forcing-grammar negative (`hunt/census-deep:CENSUS_CANDIDATES.md:11-23`; `hunt/census-deep:HUNT_REPORT_CENSUS_DEEP.md:220-241`). R-CP1 keeps those two statements separate.

### 1.3 The positive width-contract grammar

**[DESIGN]** Write `ContractWin(Q, P)` for the least structurally ranked inductive relation below, with the root claimant fixed throughout and every atomic edge expanded for the exact verifier-depth checks above. It is a **positive derivation relation**, not a statement about proof numbers and not the strict verifier's larger legal-move grammar.

**[DESIGN — typed leaves]** `StateLeaf(Q,P)` holds only when the existing exact typed-leaf predicates establish a claimant-to-move lambda-one `Win` or defender-to-move adaptive lambda-one `Loss` at `P`, and the leaf's exact resolution is at most `H`. `OrCompletion` is instead an edge-local constructor: it checks a legal claimant placement from a nonterminal Choice state that completes the named window at the declared ply. The positive certificate node forms are documented at `packages/hexfield_eq/rust/src/tss_verify.rs:106-139`; the checker replays their facts from the complete state rather than trusting tags.

**[DESIGN — claimant Choice]** If `P.current_player = claimant`, then `ContractWin(Q,P)` holds when `StateLeaf(Q,P)` holds or when **one** edge in the complete `AttackEdges_CP1(P)` relation is a checked `OrCompletion`/tactical success for the claimant or leads to a child `P'` with `ContractWin(Q,P')`.

**[DESIGN — defender Universal]** If `P.current_player != claimant`, `StateLeaf(Q,P)` is checked first. The non-leaf grammar is available only at the checked post-opening tight-dispatch boundary: there is a live claimant threat, the defender has no own win-now, and the exact minimum hitting number equals the defender's remaining turn budget `b`. The complete `DefendEdges_CP1(P,b)` relation must be nonempty, and `ContractWin(Q,P)` requires claimant success for **every** obligation in it. Outside that boundary, or for an empty defender relation, there is no positive contract constructor; the empty conjunction is deliberately **not** a win because the engine marks an empty generated vector `Refuted`.

**[CODE FACT]** The engine implements the Choice/Universal ownership convention with a fixed claimant rather than negamax; a `FirstStone` placement can leave the same player to move (`packages/hexfield_eq/rust/src/tss_solver.rs:1-13`). Defender nodes are rejected unless `opp_threat_count > 0`, `!own_win_now`, and `min_hitting_set == b` (`packages/hexfield_eq/rust/src/tss_solver.rs:5821-5835`).

**[DESIGN — attacker edges]** `AttackEdges_CP1(P)` is the exact extensional result of the following phase rules:

- **[DESIGN]** Candidate coordinates are empties in claimant-pure count-at-least-two windows, plus cells blocking live defender count-four/five windows (`packages/hexfield_eq/rust/src/tss_solver.rs:8066-8157,8168-8181`).
- **[DESIGN]** At claimant `FirstStone`, edges are complete two-placement turns. The second-coordinate universe is the deduplicated union of promoted count-at-least-two continuations through the first coordinate, the frozen turn-start candidates, and newly promoted count-one windows through the first coordinate (`packages/hexfield_eq/rust/src/tss_solver.rs:8377-8435`). Both legal orderings are examined before unordered-pair deduplication (`packages/hexfield_eq/rust/src/tss_solver.rs:5933-6055`).
- **[DESIGN]** A nonterminal pair is retained only if it creates a nonempty claimant count-at-least-four family, blocks every pre-existing defender count-at-least-four win-now, and leaves exact hitting number two; an already forced-lost family may become a typed tactical edge or remain pending when the leaf cannot materialize inside `H` (`packages/hexfield_eq/rust/src/tss_solver.rs:8437-8564`).
- **[DESIGN]** At claimant `SecondStone`, a nonterminal continuation is retained only when the completed turn creates a new claimant threat and reaches the same tight defender dispatcher; opening single-stone handling follows the exact code branch at `packages/hexfield_eq/rust/src/tss_solver.rs:6086-6164`.

**[DESIGN — defender edges]** The logical `DefendEdges_CP1(P,b)` is sequential. At `b=1`, it is the exact extendable-transversal kernel. At `b=2`/`FirstStone`, its first obligations are the `K2` extendable-hit cells and each surviving `SecondStone` state exposes its exact `K1` obligations. Any unsupported future budget falls back to the full hitting universe (`packages/hexfield_eq/rust/src/tss_solver.rs:8744-8797`). Operationally, the code uses atomic complete-turn children exactly when `forced_defender_pair_plan` returns `Some`, otherwise it falls back to ordinary one-placement children (`packages/hexfield_eq/rust/src/tss_solver.rs:6222-6233,6270-6346`). CP-O15 must prove the atomic plan plus checked commutations is a quotient of this sequential grammar; “may be atomic” is not part of the logical relation.

**[CODE FACT]** The current wide generator is eager even when lazy frontier admission is on. `expand` constructs the whole child vector before installing a `Branch`; an empty completed vector becomes `Refuted` (`packages/hexfield_eq/rust/src/tss_solver.rs:5821-5854`). `TSS_LAZY_FRONTIER` delays arena/TT admission of pending states, not child generation.

**[DESIGN — negative proposition]** Define

```text
NoContractWin(Q,P) := not ContractWin(Q,P).
```

**[DESIGN]** Its constructive dual has these cases: a Choice refutation carries an exact exhaustion proof and refutes **every** generated edge (including the vacuous zero-edge case); a Universal refutation carries **one** generated defender obligation whose child is refuted; a base refutation demonstrates that no typed positive leaf or applicable nonempty internal constructor exists. Transposition sharing is allowed only under exact state/query equality.

**[PRIOR-ROUND FACT]** On nonempty branches this dual matches the R-CF1 ranked grammar: a resolved Universal is dead if one resolved child is dead, while a Choice is dead only when every edge is resolved and every child is dead; tactical, completion, and unresolved edges block Choice death (`hunt/census-deep:CENSUS_CANDIDATES.md:187-205`). R-CF1 required the ranked-lift Choice list to be nonempty because its rank-0 `G0` predicate handled no-pair generation separately. R-CP1 makes that zero-edge base explicit rather than pretending the ranked-lift rule itself covered it.

**[PRIOR-ROUND FACT]** R-CF1 did **not** prove that grammar dual: every forcing-grammar candidate remained `CONJECTURE / SHADOW-SURVIVOR`, with no production refutation (`hunt/census-deep:CENSUS_CANDIDATES.md:289-301`). R-CP1 borrows its contract separation and quantifier shape, not a landed theorem.

## 2. Exact theorem candidates

### Candidate A — mathematical contract decision (model capstone)

**[DESIGN — TARGET THEOREM A]** Let `decideContract` be a total functional evaluator over the finite query-bounded grammar, independent of proof-number ordering and caches. Prove:

```text
theorem decideContract_win_iff (Q P) (h : WellFormedCP1 Q P) :
  decideContract Q P = win <-> ContractWin Q P

theorem decideContract_noWin_iff (Q P) (h : WellFormedCP1 Q P) :
  decideContract Q P = noWin <-> NoContractWin Q P
```

**[OBLIGATION]** Totality follows only after proving finite generation, strict placement-clock progress, the structural depth bound, exact atomic-pair expansion, and well-founded handling of shared states. No existing TssZones theorem proves this negative decision procedure.

### Candidate B — operational df-pn completeness (the prize statement)

**[DESIGN — TARGET THEOREM B]** Let `kappa` range over `AdmissibleCacheSchedule(Q)`: at each lookup it may retain or forget entries and may take a miss, but any hit must carry the exact proposition and may not invent a semantic result. After fixing `kappa`, `stepQ(kappa)` is the deterministic transition relation of the abstract `WideDFPN_CP1` machine with the node-cap guard removed, and `stepsQ(kappa,k,s0,s)` means exactly `k` transitions. Operational outcomes are `FoundWin(cert)`, `Exhausted(no)`, and fail-closed `Failed(reason)`; only the first two are semantic. Define

```text
CP1WinArtifact Q P cert :=
  exists d : ContractDerivation Q P,
    expandCP1 d = cert /\ checkWin Q P cert = true.
```

**[DESIGN]** This dependent semantic predicate, not `checkWin = true` alone, is the reverse-direction evidence that a found artifact belongs to CP1: the strict verifier's `Choice` grammar is wider. The runtime need not serialize `d`; CP-O27 must reconstruct the membership proof from the modeled CP1 generator/execution correspondence. `MaterializationComplete(Q)` says a PN-closed proof cannot terminate the supported run merely because strict-certificate resource limits reject the chosen representation. Quantifying over **every** admissible cache schedule, prove:

```text
theorem wide_dfpn_finds_contract_win (Q P)
    (hwf : WellFormedCP1 Q P)
    (hmat : MaterializationComplete Q) :
  ContractWin Q P ->
    forall kappa, AdmissibleCacheSchedule Q kappa ->
      exists k cert,
        stepsQ kappa k (init Q P) (FoundWin cert) /\
        CP1WinArtifact Q P cert

theorem wide_dfpn_exhausted_sound (Q P no)
    (hwf : WellFormedCP1 Q P) :
  forall kappa, AdmissibleCacheSchedule Q kappa ->
    (exists k, stepsQ kappa k (init Q P) (Exhausted no)) ->
      NoContractWin Q P

theorem wide_dfpn_uncapped_dichotomy (Q P)
    (hwf : WellFormedCP1 Q P)
    (hmat : MaterializationComplete Q) :
  forall kappa, AdmissibleCacheSchedule Q kappa ->
    (exists k cert,
       stepsQ kappa k (init Q P) (FoundWin cert) /\
       CP1WinArtifact Q P cert) \/
    (exists k no,
       stepsQ kappa k (init Q P) (Exhausted no))
```

**[DESIGN]** The first theorem is the prize's finding direction. The second is its direct operational exhaustion form; unlike Candidate C it does not assume checker soundness. The dichotomy includes termination and proves `Failed` unreachable for a supported query once CP-O31 discharges `MaterializationComplete`. Because `CP1WinArtifact` contains a `ContractDerivation`, its reverse direction is definitional rather than borrowed from the wider strict verifier. Thus, for every admissible `kappa`, these results yield the exact corollary `reaches a CP1WinArtifact <-> ContractWin`; otherwise the machine reaches an `Exhausted` result whose emitted witness Candidate C independently checks.

**[DESIGN]** “Uncapped” means the node-expansion guard is absent, not merely that a finite run happened to consume fewer nodes than its configured cap. “Exhausted” means a complete structural negative witness at **any** stage: the root is genuinely refuted, every reachable Choice generator is complete, there is no live `Unexpanded` or unresolved in-contract cutoff, and termination was not stall/materialization/precondition failure. Reaching `S(Q)` is necessary only on a branch whose refutation uses the structural boundary. A shallow refutation with no cutoff dependency is exhaustive for every later stage.

**[DESIGN]** A final-run `DepthCutoff` at child state depth `> S(Q)` is not required to disappear. It may be translated into `StructuralBoundary` evidence after the checker regenerates the crossing edge, expands its exact certificate cost, and first confirms that it has no in-contract edge-local terminal/tactical/completion constructor under `H` and `C`; only recursive-state admission is defeated by `child_state_depth > S(Q)`. An intermediate cutoff, an eligible cutoff at depth `<= S(Q)`, or an unclassified overshoot remains unresolved.

**[NON-CLAIM]** `root.dn == 0` alone is not exhaustion. `DepthCutoff` and genuine `Refuted` both map numerically to `(PN_INFINITY,0)` (`packages/hexfield_eq/rust/src/tss_solver.rs:5596-5605`), while selection deliberately treats a depth cutoff as unresolved (`packages/hexfield_eq/rust/src/tss_solver.rs:5569-5582`).

### Candidate C — certificate-first production theorem (recommended first landing)

**[DESIGN — TARGET THEOREM C]** Define a versioned, independently specified `NoTssCertificate` and a total bounded checker `checkNo`. Prove in Lean:

```text
theorem checkNo_sound (Q P no) (hwf : WellFormedCP1 Q P) :
  checkNo Q P no = true -> NoContractWin Q P
```

**[DESIGN]** Expose only the sealed engine conclusion:

```text
emit NoContractWin only if
  root == exact_bound_root(no)
  and grammar_id == no.grammar_id
  and checkNo Q root no == true.
```

**[DESIGN]** This theorem makes a terminating accepted exhaustion a theorem even before full Rust scheduler refinement is complete. If a generator, lazy-frontier, incremental-planner, TT, or scheduler bug omits a winning branch, the emitted trace must fail independent regeneration rather than mint a false negative.

**[NON-CLAIM]** Candidate C alone does not prove that buggy Rust will make progress, terminate, or find a present win; it proves that Rust cannot successfully label such a run exhausted. Candidate B supplies the stronger liveness/finding claim. The recommended program lands C first, then B.

### 2.1 Verdict coverage and today's behavior

**[CODE FACT]** Today's hard `WIN`/`LOSS` results must carry a certificate in the corpus harness, and every supplied certificate is passed to `TssVerifier` (`packages/hexfield_eq/rust/src/tss_corpus.rs:405-418`). The strict verifier rejects `Unknown` claims at entry (`packages/hexfield_eq/rust/src/tss_verify.rs:179-196`).

**[CODE FACT]** Today's `UNKNOWN` has no negative meaning. Early invalid/cap/horizon cases return it (`packages/hexfield_eq/rust/src/tss_solver.rs:890-924`), and any primal/dual attempt without a materialized certificate falls through to it (`packages/hexfield_eq/rust/src/tss_solver.rs:941-993`). The module header itself says failed restricted attack and resource exhaustion are `Unknown`, never an opponent proof (`packages/hexfield_eq/rust/src/tss_solver.rs:8-13`).

**[CODE FACT]** Corpus expectation `NO` means only “must not return `WIN`”: both `LOSS` and `UNKNOWN` are accepted (`packages/hexfield_eq/rust/src/tss_corpus.rs:1-10,561-570`). The ladder continues on `UNKNOWN` until its last **applicable** configured rung—`NO` rows stop above one million nodes, while positive rows may continue higher (`packages/hexfield_eq/rust/src/tss_corpus.rs:300-311,350,555-558`). Therefore no retained `NO` row is current evidence of contracted exhaustion.

**[DESIGN]** Candidate B/C covers:

| Engine state | Covered conclusion |
|---|---|
| **[DESIGN]** verified `WIN` from the primal CP1 arm | positive certificate soundness plus search completeness |
| **[DESIGN]** verified immediate `LOSS` | existing strict positive authority; not the width-completeness target |
| **[DESIGN]** proposed checked `NO_CONTRACT_WIN` | `not ContractWin(Q,root)` |
| **[DESIGN]** node-cap `UNKNOWN` | no conclusion |
| **[DESIGN]** internal stall, unresolved cutoff, materialization failure, invalid precondition, or checker rejection | no conclusion; fail closed as `UNKNOWN(Incomplete)` |
| **[DESIGN]** non-immediate dual `LOSS` search | outside CP1 theorem; pair-complete `Both` does not run it |

### 2.2 Smallest prerequisite result split

**[PROPOSAL — NO CODE IN THIS ROUND]** First add a default-off, `cfg(test)` observation seam returning an internal stop enum without changing `ProofStatus`:

```text
StageEvent = SelectedCutoff { depth }

SearchStop =
  RootProven
  | RootStructurallyRefuted { stage, uses_boundary }
  | NodeCap
  | NoSelectedCutoff
  | NonAdvancingOrInvalidCutoff
  | Stalled
  | MaterializationFailed
  | PreconditionRejected
```

**[PROPOSAL — NO CODE IN THIS ROUND]** Promote the minimum production distinction only when the observation campaign is stable:

```text
UnknownKind = Capped | Incomplete | ExhaustedPendingVerification
```

**[PROPOSAL — NO CODE IN THIS ROUND]** `ExhaustedPendingVerification` may become `NO_CONTRACT_WIN` only after `checkNo` accepts.

**[CODE FACT]** The required distinction cannot be reconstructed reliably from existing statistics. `run` and `run_until` currently merge proof, root refutation, cap, final depth, missing cutoff, nonadvancing/no-progress cutoff, and stall into loop breaks/`Option<usize>` (`packages/hexfield_eq/rust/src/tss_solver.rs:4359-4401,4451-4491`); a selected intermediate cutoff is normally a stage event, not a terminal stop. `AttemptResult` carries only an optional positive certificate and stats (`packages/hexfield_eq/rust/src/tss_solver.rs:1707-1712`). A `NoSelectedCutoff` exit becomes `RootStructurallyRefuted` only after negative provenance validates it.

## 3. The search discipline that must be modeled

**[DESIGN]** The completeness proof may erase heuristics only after proving that each erased mechanism refines the same finite contract relation. The following table is the minimum operational surface.

| Mechanism | Abstract role | Required invariant |
|---|---|---|
| **[DESIGN]** Choice/Universal PN/DN | selects unresolved obligations and recognizes positive/negative closure | recurrence agrees with the inductive grammar; semantic node tags disambiguate numeric sentinels |
| **[DESIGN]** df-pn thresholds | batches descent work without restarting at the root | second-best, subtraction, and child floor always permit local progress and cannot starve a live derivation |
| **[DESIGN]** complete generators | defines the contracted child relation | Choice cannot be refuted until its generator is complete and every generated child is refuted |
| **[DESIGN]** staged depth | searches the structural horizon incrementally | intermediate cutoffs stay unresolved; all eligible cutoffs reopen; checked crossing edges become structural-boundary evidence at `S(Q)` |
| **[DESIGN]** lazy frontier | delays position admission | same edge set, state, key, prior, depth, and eventual selected admission as eager mode |
| **[DESIGN]** incremental defender plan | avoids batch re-enumeration in the test profile | exact ordered-plan identity with the canonical batch generator or checked fallback |
| **[DESIGN]** TT / fragments | optional state sharing and positive reuse | hits require exact proposition identity; misses, refusal, replacement, or eviction can cost work but never delete an obligation |
| **[DESIGN]** stage refresh | propagates shared-child changes to inactive transposed parents | deepest-first recomputation reaches a fixed point adequate for the next decision |
| **[DESIGN]** interior census | optional bounded-horizon refutation | off, proved inert, or separately proved sound for the exact finite-horizon contract |
| **[DESIGN]** node caps / failures | resource termination | never classified as contract exhaustion |

### 3.1 Node values and Choice/Universal recurrences

**[CODE FACT]** The implementation uses `PN_INFINITY = 1_000_000_000` (`packages/hexfield_eq/rust/src/tss_solver.rs:1982`). If `num(c) = (pn(c),dn(c))`, recomputation is exactly (`packages/hexfield_eq/rust/src/tss_solver.rs:5596-5638`):

```text
Choice:
  pn(n) = min_c pn(c)
  dn(n) = satSum_c dn(c)

Universal:
  pn(n) = satSum_c pn(c)
  dn(n) = min_c dn(c)
```

**[CODE FACT]** Each sum is clamped to `PN_INFINITY`. A proven leaf/fragment is `(0,infinity)`; a refuted node and a staged `DepthCutoff` are both `(infinity,0)`. Edge-local claimant completion/tactical/refutation uses the same solved values (`packages/hexfield_eq/rust/src/tss_solver.rs:5146-5165`).

**[OBLIGATION]** Prove the semantic recurrence invariant, by node tag rather than numbers alone:

```text
node is positively solved  -> ContractWin at its exact state
node is genuinely refuted  -> NoContractWin at its exact state
node is DepthCutoff         -> unresolved at the current stage
```

**[OBLIGATION]** Prove the converses needed for operational closure after complete generation. Saturating arithmetic must not turn a finite large value into a semantic solved/refuted tag.

**[CODE FACT]** Selection already uses semantic child predicates to avoid confusing a saturated sentinel with a finished child (`packages/hexfield_eq/rust/src/tss_solver.rs:5381-5405,5569-5593`). The proof must preserve that distinction rather than idealize the implementation as untagged natural-number df-pn.

### 3.2 Second-best threshold descent, progress floors, and conjunctive subtraction

**[CODE FACT]** Production `work` performs threshold-bounded df-pn descent. The code states that thresholds change visit order only, while recurrences, expansion, refutation, and materialization stay unchanged (`packages/hexfield_eq/rust/src/tss_solver.rs:4632-4643`). With selected child `i`, parent thresholds `(theta_p,theta_d)`, parent values `(p,d)`, child values `(p_i,d_i)`, and production delta one, the thresholds are (`packages/hexfield_eq/rust/src/tss_solver.rs:4763-4828`):

```text
Choice:
  theta_p_i = max(p_i + 1, min(theta_p, secondMin(p_j, j != i) + 1))
  theta_d_i = max(d_i + 1, theta_d - (d - d_i))

Universal, uncommitted:
  theta_d_i = max(d_i + 1, min(theta_d, secondMin(d_j, j != i) + 1))
  theta_p_i = max(p_i + 1, theta_p - (p - p_i))

Universal, committed:
  theta_d_i = max(d_i + 1, theta_d)
  theta_p_i = max(p_i + 1, theta_p - (p - p_i)).
```

**[CODE FACT]** PN/DN recurrence sums clamp at `PN_INFINITY = 1_000_000_000`, while production threshold addition/subtraction uses `u32` saturating arithmetic. Consequently the production `+1` floor at `PN_INFINITY` is the strictly larger `1_000_000_001`; only the `cfg(test)` experimental threshold-delta path reclamps the increment when that option is present (`packages/hexfield_eq/rust/src/tss_solver.rs:4622-4629,4778-4801,4813-4826,5605-5628`). That experiment is outside the covered flag profile.

**[OBLIGATION]** Prove **local progress by arithmetic case split**. For selected values below `PN_INFINITY`, the child floor makes each relevant threshold strictly larger and descent must expand/change a semantic tag, raise a relevant number, expose a deeper cutoff, or consume the node cap. At `PN_INFINITY`, the production floor crosses above the semantic sentinel, but recurrence values remain clamped at the sentinel; numeric threshold growth alone therefore cannot establish semantic progress. Prove that descent across that sentinel eventually changes a tag/resolves the selected obligation, or prove that an unresolved selected child can never carry the saturated value. If neither statement is true, Candidate B requires an engine repair.

**[OBLIGATION]** Prove **unsaturated conjunctive conservation**: subtracting the siblings' exact current aggregate gives a child enough residual threshold to change the parent's aggregate. This lemma is false as stated for a clamped parent sum: after saturation, a child can change while the parent remains `PN_INFINITY`, and subtraction sees only the clamped total (`packages/hexfield_eq/rust/src/tss_solver.rs:4790-4828,5612-5628`). The saturated case needs the separate semantic-progress/non-starvation lemma above. Prove the second-best rule causes parent re-selection after the chosen disjunct/conjunct ceases to be best.

**[CODE FACT]** The selected-child policy also includes a root-only sequential probe for partial/urgent turns, a root width tier, and a high-fanout Universal commitment that begins at four distinct linked obligations (`packages/hexfield_eq/rust/src/tss_solver.rs:2060-2064,4719-4751,5407-5566`).

**[OBLIGATION]** Prove **global non-starvation** for those policies. In particular, a committed Universal must either resolve its obligation or yield a true stall and try each distinct sibling once; a root sequential probe must advance the selected depth cutoff instead of permanently hiding a lower-ranked winning turn; and width/urgency ties must only reorder the complete Choice set.

### 3.3 Generator exhaustion is a semantic event

**[CODE FACT]** There is no generator cursor or exhaustion bit today. One expansion synchronously computes the entire `Vec<WidePnChild>` and only then installs `Branch` or `Refuted` (`packages/hexfield_eq/rust/src/tss_solver.rs:5708-5715,5821-5864`). Thus “generator exhausted” is an implicit fact about successful return from a whole-vector function.

**[DESIGN]** The model nevertheless carries explicit generator state:

```text
GeneratorState = Unstarted | Producing(prefix) | Exhausted(exact_child_set)
```

**[DESIGN]** A Choice node may acquire a negative/exhausted proof only in the last state. An implementation that later makes generation incremental must preserve this state machine.

**[OBLIGATION]** Prove exact set equality between `AttackEdges_CP1`/`DefendEdges_CP1` and the installed vectors, including:

- **[OBLIGATION]** frozen first-candidate membership and the promoted second-candidate union;
- **[OBLIGATION]** both legal pair orders before unordered deduplication;
- **[OBLIGATION]** `WideTurnGate` family construction, defender-win-now rejection, and exact `mhs` classification;
- **[OBLIGATION]** emitted-move legality and exact apply/undo state;
- **[OBLIGATION]** defender kernel/pair canonicalization and commutation expansion;
- **[OBLIGATION]** edge-local terminal/tactical classification, including the invariant that a pending child can never later be an unrecorded claimant terminal win.

**[CODE FACT — RISK]** Expanded terminal states are currently marked `Refuted` regardless of winner (`packages/hexfield_eq/rust/src/tss_solver.rs:5769-5776`). Completeness therefore depends on the invariant that every claimant terminal win is caught and represented edge-locally before it can become a pending expanded state. That invariant must be proved and mutation-tested; it cannot remain an informal reachability belief.

### 3.4 Staged structural depth and refresh

**[CODE FACT]** Wide search starts at depth zero, reopens eligible cutoffs, follows the selected path to its next useful depth, and shares one global node cap across stages (`packages/hexfield_eq/rust/src/tss_solver.rs:4359-4398`). After every stage it recomputes all arena entries deepest-first; reopening cutoffs triggers the same refresh so transposed parents outside active recursion do not retain stale `dn=0` (`packages/hexfield_eq/rust/src/tss_solver.rs:4369-4375,4494-4531`).

**[OBLIGATION]** Prove staged-deepening coverage:

1. **[OBLIGATION]** every encountered pending child beyond the current stage becomes a tagged cutoff at its exact depth;
2. **[OBLIGATION]** a selected in-contract cutoff produces a strictly greater next stage, bounded by `S(Q)`;
3. **[OBLIGATION]** every cutoff with `entry.depth <= new_stage` reopens;
4. **[OBLIGATION]** refresh propagates every changed shared child to all parents before the next root stop test;
5. **[OBLIGATION]** a shallow root refutation supports exhaustion when its witness has no cutoff dependency; otherwise every eligible cutoff reopens through `S(Q)`, and only independently checked crossing-edge `StructuralBoundary` evidence may remain;
6. **[OBLIGATION]** atomic pair overshoots and edge-local terminal/tactical results obey the exact expanded certificate-depth guard rather than bypassing the stage boundary.

### 3.5 Lazy frontier admission

**[CODE FACT]** `TSS_LAZY_FRONTIER` is sampled once when `WidePnSearch` is constructed (`packages/hexfield_eq/rust/src/tss_solver.rs:4129-4140`). Attacker thunks are selection-only; defender thunks virtually expose the exact eagerly admitted state through an existing indexed entry or through the deferred prior (`packages/hexfield_eq/rust/src/tss_solver.rs:2142-2173,5146-5178`).

**[CODE FACT]** Deferred identity preserves the first eager admission's depth and prior; selection removes that deferred record and inserts or reuses the exact key (`packages/hexfield_eq/rust/src/tss_solver.rs:4230-4248,4279-4285`). Defender singles and defender pairs choose between eager insertion and a thunk over the same key/prior (`packages/hexfield_eq/rust/src/tss_solver.rs:6195-6204,6378-6385`). Selected future keys are checked against the applied state, with stronger assertions behind a test flag (`packages/hexfield_eq/rust/src/tss_solver.rs:4882-4900,4957-4973`).

**[OBLIGATION]** Prove an eager/lazy bisimulation with five equalities: generated edge order/set, future position, exact key, first-admission prior/depth, and resolved PN/DN observation. Also prove every selected pending thunk is eventually linked unless the run ends for a separately classified cap/failure reason.

**[DESIGN]** The exhaustion checker does not trust lazy bookkeeping. It regenerates the extensional grammar from full states. A lazy-mode omission therefore makes the certificate uncheckable instead of changing `NoContractWin`.

### 3.6 Incremental defender enumeration

**[CODE FACT]** The `TSS_INCR_DEFENDER` mode, snapshots, incremental planner, and relevant search fields are `cfg(test)` (`packages/hexfield_eq/rust/src/tss_solver.rs:2800-2835,3923-3931`). Non-test builds use the batch planner unconditionally (`packages/hexfield_eq/rust/src/tss_solver.rs:6345-6346`).

**[CODE FACT]** In test `shadow` mode the engine computes both plans and asserts equality. In `consume` mode it trusts the incremental result when a bounded snapshot exists and falls back to batch only when reconstruction supplies no snapshot (`packages/hexfield_eq/rust/src/tss_solver.rs:6270-6344`). The implemented equality is structural—not just extensional final positions—and compares kernel order, pair count/order, coordinates, final keys, and priors (`packages/hexfield_eq/rust/src/tss_solver.rs:3124-3148`).

**[ASSUMPTION — TEMPORARY]** Until proved, any theorem covering the official test profile assumes, for **every reachable eligible state**, that `forced_defender_pair_plan_incremental` returns exactly the canonical batch plan under that structural equality, or takes the batch fallback. Observed row identity is regression evidence, not a universal proof.

**[OBLIGATION]** Replace that assumption with either a Lean refinement theorem over the bounded snapshot transformation or a checker step that independently regenerates and compares the batch plan at every exhaustion node. A consume-mode `None` where batch returns `Some` is a completeness failure even if all produced pairs are individually correct.

### 3.7 TT, cache retention, and the 1 GiB profile

**[CODE FACT — CORRECTION]** There is no special “evict at 1 GiB” rule in the wide search core. One GiB is the official caller-owned profile cap (`docs/TSS_RUNBOOK.md:16-25`). The wide solve-local structure keeps the frontier arena authoritative: an exact-key hit reuses an entry, but a full/disabled index still admits a new arena node and merely refuses to index its key (`packages/hexfield_eq/rust/src/tss_solver.rs:4230-4275`). The wide index does not evict; it admission-rejects.

**[CODE FACT]** The narrow direct-map TT does replace colliding entries at any cap, after exact hash/claimant/full-key checks, and refuses an oversized replacement (`packages/hexfield_eq/rust/src/tss_solver.rs:9520-9628`). Cross-solve stores retain only positive proof fragments, use exact key/claimant checks, and reset on incompatible reconfiguration (`packages/hexfield_eq/rust/src/tss_solver.rs:655-667,9888-9952,10073-10117`). Shared wide fragments are default-off (`packages/hexfield_eq/rust/src/tss_solver.rs:682-691`).

**[DESIGN]** The `kappa` parameter of Candidate B is an adversarial optional cache schedule. At any step it may forget any index/cache entry or choose a miss. A hit may only share an exact `(position, claimant, horizon-compatibility, grammar)` proposition; a miss unfolds another tree occurrence. Candidate B is quantified over every admissible `kappa`, so completeness and a checked negative may not assume that any entry survives, that a 1 GiB index admits every key, or that a stage refresh sees a parent only through the index. The concrete deterministic Rust cache policy is one schedule that CP-O26/CP-O27 must prove admissible.

**[OBLIGATION]** Prove cache erasure/refinement for the covered wide arena. If shared positive fragments later enter scope, prove that every hit is independently verified and horizon/height/path-context compatible before replacing search. If the narrow compatibility profile later enters scope, separately prove the soundness of `LOCAL_TT_FAILED`; it is a negative cached fact and is not covered by “positive cache only.”

### 3.8 Interior census gate

**[CODE FACT]** `TSS_INTERIOR_CENSUS_GATE` is default-off and sampled once per solve (`packages/hexfield_eq/rust/src/tss_solver.rs:832-840`). It examines only non-root claimant-owned `FirstStone`/`SecondStone` nodes whose remaining semantic horizon is in `[0,8]` and whose coordinates pass checked safety arithmetic (`packages/hexfield_eq/rust/src/tss_solver.rs:167-193`); a firing gate directly marks the node `Refuted` (`packages/hexfield_eq/rust/src/tss_solver.rs:5802-5816`).

**[DESIGN]** The initial CP1 theorem keeps this flag off. A small extension may cover `H=u32::MAX` under the explicit reachable-clock precondition `current_ply <= u32::MAX - 9`: then `h_rem > 8`, evaluation returns `None`, and flag-on is definitionally inert. This is the exact R-CF1 inertness observation, not a finite-horizon census theorem (`hunt/census-deep:HUNT_REPORT_CENSUS_DEEP.md:9-39`).

**[NON-CLAIM]** Finite-horizon flag-on completeness remains outside scope until the exact phase/census lower bound, coordinate guard, and Rust-to-model correspondence are machine-checked. T3/T4 establish strong certificate dismissal results in the Lean spine, but they do not by themselves prove that this Rust census implementation equals their premises.

### 3.9 Caps, stalls, and exhausted termination

**[CODE FACT]** The public node cap includes one root examination; the wide attempt receives `node_cap - 1` (`packages/hexfield_eq/rust/src/tss_solver.rs:901-927`). Expansion checks the cap before increment, and both recursive descent and the outer driver stop at it (`packages/hexfield_eq/rust/src/tss_solver.rs:4451-4467,4711-4716,5708-5715`).

**[DESIGN]** If that guard binds, the only correct classification is `UNKNOWN(Capped)`. It is not “exhausted,” even if some subtrees are refuted or the root's current numeric value resembles a closed value.

**[DESIGN]** `Exhausted` requires a structural witness with all of the following:

1. **[DESIGN]** the recorded stage is no deeper than `S(Q)`; if it is shallower, the witness has no dependency on a deeper cutoff;
2. **[DESIGN]** root has genuine-refutation provenance, not only `dn=0`;
3. **[DESIGN]** every negative Choice occurrence has a checked complete generator and negative evidence for each edge;
4. **[DESIGN]** every negative Universal occurrence identifies a checked generated reply with negative evidence;
5. **[DESIGN]** all shared references have exact state/query identity and form a well-founded DAG;
6. **[DESIGN]** no live occurrence is `Unexpanded`, an eligible/intermediate `DepthCutoff`, or a lazy unlinked pending obligation; a cutoff beyond `S(Q)` appears only as checked structural-boundary evidence for its crossing edge;
7. **[DESIGN]** termination was not `NodeCap`, `Stalled`, precondition failure, panic, allocation failure, or positive-certificate materialization failure.

**[CODE FACT — MAJOR FINDING]** The engine records none of that provenance in a return value today. Therefore a theorem consumer must wait for the proposed stop split and a successfully checked exhaustion certificate; heuristics such as `stats.nodes < cap` are not authority.

## 4. Obligations ledger

### 4.1 Status discipline

**[LEAN FACT]** The authoritative Lean implementation status is the final `Status` column in `E:\tss-lean\LEDGER.md`, not the source document's `Doc status`. The ledger defines `UNSTATED`, `STATED`, `PROVEN`, and `AUDITED`, and explicitly distinguishes a source claim from a landed kernel proof (`E:\tss-lean\LEDGER.md:3-21`).

**[LEAN FACT — MAJOR FINDING]** T3, T4, T5, T9, T10, L15, and L16 are `PROVEN`. L17, T11, T11.1, and the T6 soundness capstone are `UNSTATED`. D19-D21 and the T6 reflected region contracts are primarily `STATED`, with substantial named sublemmas `PROVEN` (`E:\tss-lean\LEDGER.md:80-86,92-116,127,131`).

**[LEAN FACT]** The module ownership matches that reading. `Certificate.lean` owns the proof-free D9/D18 graph, leaf grammar, roles, clocks, and structural validator, while leaving T6/D19 semantics downstream (`E:\tss-lean\TssZones\Certificate.lean:3-10`). `Soundness.lean` owns base/D17 compilers and soundness and deliberately imports no forced-hit semantic module (`E:\tss-lean\TssZones\Soundness.lean:3-9`). `ForcedHit.lean` owns the separate T6 region, grammar/checker, kernel calculus, and compiler (`E:\tss-lean\TssZones\ForcedHit.lean:3-10`). `DAGUnfoldingSoundness.lean` transports soundness-owned results through finite unfolding (`E:\tss-lean\TssZones\DAGUnfoldingSoundness.lean:3-9`).

**[DESIGN]** Status categories below are exactly: **AKP** = already kernel-proved in the inspected spine; **PEM** = provable with existing machinery but not currently landed at the claimed strength; **NNM** = needs new machinery. Difficulty is an engineering/proof estimate, not status.

### 4.2 Numbered proof/refinement ledger

| ID | Statement sketch | Difficulty | Honest status and dependency |
|---:|---|---|---|
| CP-O1 | **[OBLIGATION]** Define `CP1-a49e8abd-v1`, exact query binding, phase schedule, `(H,S,C)`, attack/defense edge relations, and `ContractWin`/`NoContractWin` in Lean. | Medium | **NNM.** Existing `Position`, schedule, certificate, and horizon types help, but no declaration denotes the Rust width grammar. |
| CP-O2 | **[OBLIGATION]** Prove the contract is finite and well-founded: finite child sets, every pending edge strictly advances the absolute placement clock, atomic pairs advance by two, and their expanded certificate height is bounded by `C`. | Medium | **NNM.** D18 finite DAG/rank infrastructure is reusable, but exact generator finiteness and atomic operational edges are new. |
| CP-O3 | **[OBLIGATION]** Under the exact post-opening/nonterminal and remainder premises, typed completion/WIN/LOSS leaves imply the claimant's semantic win through their exact resolution. | Low model-side; medium bridge | **AKP semantic ingredients; NNM CP1 bridge.** Lambda-one WIN/LOSS soundness is `PROVEN` but explicitly assumes `PostOpening` and `Nonterminal` (`E:\tss-lean\TssZones\Basic.lean:1157-1162,1393-1399`); D9 leaf predicates are reflected but their ledger rows are `STATED` (`E:\tss-lean\LEDGER.md:32,50-56`). Exact CP1/Rust tag, claimant, clock, and remainder correspondence remains CP-O27. |
| CP-O4 | **[OBLIGATION]** Given `PostOpeningAt` and the theorem's other premises, a valid base tree certificate yields `AttackerWinsBy` through the inclusive horizon and mapped resolution on every maximal history; ordinary omitted replies satisfy the exact inclusive reply guard. | Low | **AKP.** T3 and `T3_soundDismissal` are `PROVEN` with their explicit premises (`E:\tss-lean\LEDGER.md:80`; `E:\tss-lean\TssZones\Soundness.lean:14028-14046,14364-14414`). |
| CP-O5 | **[OBLIGATION]** Exact ranked-zone premises reconstruct base validity and invoke T3. | Low | **AKP.** T4 is `PROVEN` (`E:\tss-lean\LEDGER.md:81`; `E:\tss-lean\TssZones\Soundness.lean:14416-14438`). This does not yet identify any Rust generator set. |
| CP-O6 | **[OBLIGATION]** Under its four explicit short-budget/threat-path premises, the static `r3 ∪ attackerTouchedAliveEmpties` set covers the mandatory zone. | Low | **AKP, conditional.** T5 is `PROVEN` (`E:\tss-lean\LEDGER.md:84`; `E:\tss-lean\TssZones\Soundness.lean:15053-15068`). It is not general search completeness and must not be cited without its premises. |
| CP-O7 | **[OBLIGATION]** Given `PostOpeningAt` and the theorem's other premises, valid D17 tree certificates yield the D17 attacker strategy, mapped resolutions, and ordinary-node dismissal. | Low | **AKP.** T9 and its dismissal corollary are `PROVEN` with their explicit premises (`E:\tss-lean\LEDGER.md:127`; `E:\tss-lean\TssZones\Soundness.lean:15714-15732,15953-16003`). |
| CP-O8 | **[OBLIGATION]** Given `PostOpeningAt` and the source theorem's other premises, finite exact-position DAG sharing unfolds to a sound tree and transports base/D17 strategy, horizons, mapped resolutions, and dismissal back to the DAG. | Low | **AKP.** T10 is `PROVEN` (`E:\tss-lean\LEDGER.md:131`; `E:\tss-lean\TssZones\DAGUnfoldingSoundness.lean:49-103,193-241`). It says nothing yet about df-pn TT behavior or negative exhaustion DAGs. |
| CP-O9 | **[OBLIGATION]** At a D19 forcing gate, the stored-map/kernel split is exact; the kernel is finite/nonempty at exact pressure; every legal reply is exclusively exact-copy or adaptive escape with the stated deadline. | Medium | **AKP for the named local facts.** D19's overall contract remains `STATED`, but classification, finiteness/nonemptiness, compact bounds, and L15 are `PROVEN` (`E:\tss-lean\LEDGER.md:92-100,112`; `E:\tss-lean\TssZones\ForcedHit.lean:551-891,941-1129,1509-1531`). |
| CP-O10 | **[OBLIGATION]** Forced-hit-debited ranks/exposures obey their recurrences and comparisons; D21 omission bounds and all four L16 weighted-hazard inequalities hold. | Medium | **AKP for the listed kernels.** D20/D21 definitions remain `STATED`; recurrence/comparison/omission lemmas and L16 are proved (`E:\tss-lean\LEDGER.md:101-109,113`; `E:\tss-lean\TssZones\ForcedHit.lean:1789-1827,2119-2127,2563-2801`). |
| CP-O11 | **[OBLIGATION]** Construct source-strength L17 traces from raw compiler modes/remainders, including LOSS/off-kernel carry and the D17-authority branch, and prove both first-bad-event clauses. | High | **PEM; `UNSTATED`.** Fixed-window traces, reification, mapped-extension seams, and protection carriers exist; the live raw compiler/run carrier now retains the literal LOSS or gate-escape suffix and proves the completion contradiction. The inherited D17-authority branch, both source L17 clauses, and any source-strength L17 theorem remain absent (`E:\tss-lean\LEDGER.md:114`; `E:\tss-lean\TssZones\ForcedHit.lean:2862-2969,3294-3320,3526-3592`). |
| CP-O12 | **[OBLIGATION]** Prove `T11_exactCopySoundness` and `T11_1_d17Compatibility`: checker-valid FH/FH+D17 suffixes imply the corresponding attacker wins, including LOSS and off-kernel paths. | High | **PEM after CP-O11; both `UNSTATED`.** Policies, validators, exact-copy mapped extension, and reflection exist, but acceptance-to-win theorems do not (`E:\tss-lean\LEDGER.md:115-116`; `E:\tss-lean\TssZones\ForcedHit.lean:3774-3942,4182-4249`). |
| CP-O13 | **[OBLIGATION]** Prove the one-leading-region T6 capstone under explicit `PostOpeningAt`/nonterminal premises: exact kernel interiors, typed terminals, first handoff/escape, direct base/D17/FH/FHD17 suffixes, strategy splices, and horizon transport imply an attacker win. | High | **PEM after CP-O12 plus open T6 splits; capstone `UNSTATED`.** `CertificateValidT6For` does not itself package `PostOpeningAt`; the landed traversal bridge takes it separately (`E:\tss-lean\TssZones\ForcedHit.lean:4429-4436,5128-5158`). Region/checker reflection and kernel/compiler pieces exist, while reverse state splits, first-escape adapter, splice lemmas, positive fixture, and `T6_extendableHitKernel` remain absent (`E:\tss-lean\LEDGER.md:85-86`; `E:\tss-lean\TssZones\ForcedHit.lean:4318-4960,5258-5359,5477-5502`). |
| CP-O14 | **[OBLIGATION]** Prove exact completeness of claimant candidate and atomic-pair enumeration against `AttackEdges_CP1`, including global defender-block inputs and unordered dedup. | High | **NNM.** No TssZones theorem models `WideTurnGate`, candidate ordering, pair promotion, or the Rust `Vec` result. |
| CP-O15 | **[OBLIGATION]** Prove exact completeness of batch defender enumeration against the sequential `DefendEdges_CP1`: kernel cells, canonical atomic pairs, reverse-key equality, and commutation expansion form an exact operational quotient, with ordinary singles on planner fallback. | High | **NNM for Rust correspondence.** T6/D19 kernel mathematics helps semantically, but the engine planner and key construction are unmodeled. |
| CP-O16 | **[OBLIGATION]** Prove terminal/structural edge classification: every in-contract claimant terminal/tactical successor becomes positive; no pending expanded successor hides such a win; atomic overshoots and edge-local results obey exact expanded height. | High | **NNM.** Required by the terminal-always-refuted path (`packages/hexfield_eq/rust/src/tss_solver.rs:5769-5776`) and the current missing generator-side depth guard (`packages/hexfield_eq/rust/src/tss_solver.rs:5908,6104-6123,8523-8536`). |
| CP-O17 | **[OBLIGATION]** Define the constructive dual `NoContractWin` grammar and prove its induction equivalent to negated `ContractWin` over the finite contract. | High | **NNM.** Existing spine machinery proves positive certificate soundness, not negative grammar completeness. |
| CP-O18 | **[OBLIGATION]** Define a finite, versioned `NoTssCertificate` and total checker; prove `checkNo_sound`; include exact generator equality at Choice and one checked refuting reply at Universal. | High | **NNM.** Existing certificate reflection patterns are reusable, but the negative object/checker is new. |
| CP-O19 | **[OBLIGATION]** Prove negative DAG sharing sound: exact repeated `(state,Q)` propositions may share, graph is acyclic/well-founded, and unfolding preserves every exhaustion obligation. | Medium/High | **NNM, reusing a proved pattern.** D18/T10 suggest the finite-DAG construction, but the negative syntax, proposition, and unfolding theorem are all new. |
| CP-O20 | **[OBLIGATION]** Prove PN/DN recurrence correctness for tagged `Unexpanded`, `DepthCutoff`, proven, refuted, Choice, and Universal nodes under saturation. | Medium | **NNM.** No proof-number model exists in TssZones. |
| CP-O21 | **[OBLIGATION]** Prove below-sentinel threshold progress/conjunctive conservation and a separate saturated semantic-progress theorem for second-best `+1`, child floors, subtraction, and committed Universal thresholds; account for production floors crossing above `PN_INFINITY` while recurrence values remain clamped. | High / potentially blocking | **NNM.** No df-pn threshold machinery exists in the spine; the ordinary conservation claim is false on a clamped parent aggregate. |
| CP-O22 | **[OBLIGATION]** Prove global fairness/termination for ordinary selection, root sequential probe, width tier, Universal commitment/yield, and finite stages in the absence of a node cap. | Very high | **NNM.** This is the central scheduler-completeness lemma behind Candidate B. |
| CP-O23 | **[OBLIGATION]** Prove staged-deepening and deepest-first refresh correctness, including all transposed parents, shallow cutoff-free exhaustion, eligible cutoff reopening, and checked structural-boundary overshoots. | High | **NNM.** Existing certificate horizons are semantic, not a model of the mutable stage driver. |
| CP-O24 | **[OBLIGATION]** Prove eager/lazy frontier bisimulation and eventual admission of every selected thunk. | Medium/High | **NNM.** Retained equivalence campaigns are executable evidence only. |
| CP-O25 | **[OBLIGATION]** Prove universal structural equality of the incremental defender plan to batch, or make the exhaustion checker compare against batch at every relevant node. | High | **NNM.** `shadow` equality on observed states is not a universal proof; consume trusts the incremental output. |
| CP-O26 | **[OBLIGATION]** Prove TT/cache erasure and exact-hit sharing for every admissible cache schedule; prove stage refresh independent of retention and the concrete deterministic Rust policy is admissible. | Medium/High | **NNM operationally.** T10 handles semantic DAG sharing, but not arena/index refusal, direct-map replacement, cache warmth, or refresh scheduling. |
| CP-O27 | **[OBLIGATION]** Prove Rust-to-model correspondence for state replay, full keys, leaf predicates, generators, tags, atomic edges, clocks, flags, and emitted trace serialization. | Very high / program-critical | **NNM.** This is the largest trust boundary. The ledger classifies prior engine/model comparison as historical evidence rather than a theorem (`E:\tss-lean\LEDGER.md:146`). |
| CP-O28 | **[OBLIGATION]** Split stage events from capped/exhausted/incomplete terminal causes; distinguish missing/nonadvancing cutoffs; prove return classification matches provenance; prohibit a sealed no-result on cap/stall/materialization/precondition failure. | Medium | **NNM plus small engine instrumentation.** No current field exposes the distinction. |
| CP-O29 | **[OBLIGATION]** Prove checker termination/resource bounds and connect the proved checker to the executed checker without importing solver generator code as authority. | High | **NNM.** A proof-level `noncomputable` reflection alone is insufficient for an executed strict-discharge boundary. |
| CP-O30 | **[OBLIGATION]** Prove the `u32::MAX` interior-census inertness lemma under exact root/depth arithmetic; keep finite-horizon gate-on outside scope until separately related to the Lean theorem. | Low / High if finite | **PEM for inertness; NNM for finite flag-on correspondence.** The code guard is only `h_rem in 0..=8` (`packages/hexfield_eq/rust/src/tss_solver.rs:187-193`). |
| CP-O31 | **[OBLIGATION]** Close positive materialization: prove every PN-closed contracted win can be emitted within strict node/edge/commutation/witness/depth limits, continue searching after an oversized/unmaterializable proof, or adopt a representation whose accepted limits cover every contracted derivation. | Very high / theorem-critical | **NNM.** Materialization can return `None` at hard bounds (`packages/hexfield_eq/rust/src/tss_solver.rs:6401-6417,6878-6888`); verifier limits are 100k nodes and one million edges/witnesses/commutations (`packages/hexfield_eq/rust/src/tss_verify.rs:17-32`). Structural depth alone does not bound strategy width below them. |

### 4.3 The pending capstone chain

**[LEAN FACT]** The honest current dependency is:

```text
L15 + L16                         [PROVEN]
     |
     v
L17 full compiler/run traces      [UNSTATED]
     |
     v
T11 + T11.1 FH/FHD17 soundness    [UNSTATED]
     |                         T6 reverse state splits,
     |                         first-escape adapter, splices
     +-----------------------------+
                                   |
                                   v
T6 one-leading-region capstone     [UNSTATED]
```

**[DESIGN]** When the full T6 capstone lands, it unblocks these R-CP1 obligations:

1. **[DESIGN]** checker-valid one-leading T6 regions, together with the explicit `PostOpeningAt`/nonterminal premises, imply a real attacker win rather than only satisfying a reflected syntax;
2. **[DESIGN]** composition/splicing of the leading T6 region with already-sound base/D17/FH/FHD17 handoff suffixes is semantically sound; T11/T11.1, not the T6 capstone, supplies FH/FHD17 suffix soundness;
3. **[DESIGN]** `AndRegime.t6Kernel` can be included in the machine-checked positive contract grammar without a semantic assumption;
4. **[DESIGN]** the positive-contract semantic interpretation gains proved cases for T6 interior nodes, typed terminals, off-kernel escapes, and direct handoffs; operational discovery remains CP-O14/CP-O15 and CP-O20 through CP-O26;
5. **[DESIGN]** the Rust defender-kernel correspondence can target a proved semantic constructor rather than a paper-level contract.

**[DESIGN]** The authoritative 3.1 target uses the **full** spine capstone, so the L17 → T11/T11.1 → T6 dependency is real: `NonT6HandoffValid` admits base, D17, FH, and FHD17 handoffs (`E:\tss-lean\TssZones\ForcedHit.lean:4342-4348`). A restricted no-FH or base/D17-only CP1 theorem may land earlier, but it must carry a distinct grammar identifier and cannot certify an emitter profile that can use FH/FHD17 handoffs.

**[NON-CLAIM]** T6 does **not** unblock recursive T6 re-entry, which remains explicitly outside the current region scope. It also does not unblock Choice generator exhaustion, df-pn threshold/fairness proofs, lazy or incremental refinement, TT/cap reasoning, negative-certificate soundness, positive materialization, or Rust-to-Lean correspondence; those operational obligations remain CP-O14 through CP-O31.

## 5. Route decision

### 5.1 Comparison

| Architecture | Machine-checked authority obtained | Rust trust gap | Main advantage | Main cost/risk | Honest effort estimate |
|---|---|---|---|---|---|
| **[DESIGN]** (a) Lean search-grammar model + paper Rust-to-model argument | abstract model theorem only | large: generator, tags, thresholds, stages, lazy/incremental paths, TT, and stop classification remain a review argument | fastest path to a publishable mathematical theorem and scheduler understanding | does not make today's Rust `UNKNOWN` a machine-checked conclusion | **[ESTIMATE]** 10-16 focused proof sessions plus 2-4 correspondence/evidence sessions; roughly 4-8 serial weeks, excluding the T6 chain |
| **[RECOMMENDATION]** (b) exact-root exhaustion certificate + verified checker | accepted `NO_CONTRACT_WIN` is independently checkable end to end; later scheduler theorem adds liveness | emitter/search bugs fail closed if checker is independent; checker/execution bridge remains TCB work | closest analogue of the strict positive verifier and directly changes the authority boundary | exact generator regeneration, trace volume, checker independence, negative-DAG formalization, and positive materialization | **[ESTIMATE]** 22-39 focused sessions; roughly 9-18 serial weeks with ±50% uncertainty, excluding the T6 chain; Candidate C alone is plausibly 12-22 sessions |
| **[DESIGN]** (c) full functional correctness of the mutable Rust search core | strongest possible implementation theorem | smallest in principle, but includes Rust/compiler verification stack | subsumes most operational obligations | enormous proof surface: engine mutation, window store, allocation/maps, env flags, serialization, cache policies, panics, and FFI | **[ESTIMATE]** at least 60-100 focused sessions / 9-18 months, likely requiring a verified-core rewrite rather than proof of this implementation |

### 5.2 Route (a): useful, but not final authority

**[DESIGN]** Route (a) would formalize `ContractWin`, the dual, PN/DN tags, stages, and an abstract fair scheduler in Lean, then map each Rust function to one abstract transition in a review document.

**[DESIGN]** The best possible bound on its Rust trust gap is a frozen `grammar_id`; an exhaustive code-to-definition map; exact-state differential fixtures; mutation tests for omitted candidates, stale keys, cutoffs, caps, and TT refusal; and a retained trace whose abstract replay agrees with Rust. Those measures make the gap visible and regression-resistant but do not make it kernel-checked.

**[RISK]** The gap is theorem-sized, not clerical. A one-line omission in second-candidate promotion, an incremental `None`, a lazy future-key mismatch, a `DepthCutoff` misclassification, or an unreported stall can falsify the implementation claim while the Lean theorem remains true.

**[DECISION]** Use route (a)'s functional model as a component of route (b), not as the sealed authority for a no-win result.

### 5.3 Route (b): exact exhaustion certificates

**[RECOMMENDATION]** Route (b) is the authoritative path because it turns a negative claim into data that an independent checker can reject. Search remains an untrusted proof producer, just as the current solver is an untrusted positive-certificate producer.

**[DESIGN]** The logical object is a frontier refutation tree, compacted to an exact-state DAG where useful:

```text
NoTssCertificate {
  format_version
  grammar_id
  exact_root_binding
  claimant
  semantic_horizon
  state_depth_cap
  certificate_depth_cap
  root_node
  nodes
}

NoNode :=
  | NoLeafOrConstructor
  | ChoiceExhausted { exact_edges, child_no_for_every_pending_edge }
  | UniversalCounterexample { generated_edge, child_no }
  | StructuralBoundary {
      exact_state, crossing_edge, expanded_cost,
      no_in_contract_terminal_tactical_or_recursive_case
    }
```

**[DESIGN]** `NoLeafOrConstructor` is accepted only after the checker independently recomputes terminal/typed-leaf status, owner/phase, and the applicable internal boundary. `StructuralBoundary` is accepted only after it replays the parent and crossing edge and expands atomic pairs into the exact strict-certificate grammar. It must reject the boundary claim if the edge has any edge-local terminal/tactical/completion form within `H` and `C`, even when the applied endpoint has state depth `> S(Q)`. Only after excluding those forms may `child_state_depth > S(Q)` discharge recursive-state admission; `C` or `H` may independently exclude any positive form. Thus an overshoot such as an atomic edge from `S(Q)-1` to `S(Q)+1` is evidence only for the recursive child, not automatically for the whole edge. An intermediate/eligible cutoff is never serializable as negative evidence.

**[DESIGN]** For `ChoiceExhausted`, the checker regenerates the complete ordered/extensional raw attack edge relation from the full replay state and requires exact equality with the trace's edge map. It independently filters each edge by semantic and expanded structural admissibility, checks negative or `StructuralBoundary` evidence for every nonpositive edge, and confirms that no in-contract edge is a claimant completion/tactical success. An empty set is acceptable only when regeneration is exactly empty and no positive leaf applies.

**[DESIGN]** For `UniversalCounterexample`, the checker rederives the tight-dispatch boundary and complete defender generator, proves the named edge is one of its obligations, replays it exactly, and checks the child negative. It need not store every defender reply because one failing conjunct refutes a positive Universal derivation.

**[DESIGN]** DAG references are keyed by full replay state plus the entire query/grammar proposition, not by hash alone. The checker validates reachability, acyclicity/rank decrease, placement-clock increments, and exact state equality. Hashes and generator fingerprints may accelerate rejection but are never authority.

**[DESIGN]** The checker must be independent in the same sense as the positive verifier: `tss_verify.rs` deliberately does not depend on `tss_solver` and replays certificate moves through the engine using shared one-turn primitives (`packages/hexfield_eq/rust/src/tss_verify.rs:1-5`). The new negative checker may share board transition and exact primitive definitions, but it must not call `WidePnSearch::expand`, trust installed child vectors, or treat a solver `WideTurnGate` result as an exhaustion proof.

**[RISK]** “Independent” cannot mean duplicating the same complicated Rust generator line-for-line with no proved relation. The preferred authority is an executable Lean checker or a small pure checker generated/connected from the proved specification. A second handwritten Rust checker is acceptable only with a separately proved correspondence path; differential tests alone do not make it verified.

### 5.4 Why C-REL O4 does or does not apply

**[PRIOR-ROUND FACT]** C-REL round 1 rejected a new support-only verifier. Its permitted salvage was a rootless template that materializes an ordinary exact-root `TssCertificate` and submits it to the unchanged strict verifier; the cheap hint is never authority (`hunt/cert-support:DESIGN_C_REL.md:11-25`).

**[PRIOR-ROUND FACT]** The NQ2 adversary has 538 legal `SecondStone` completions. Its unique winning move is `r=(6,-6)`, distance six from the nearest attacker stone and in no live attacker window; each of the other 537 moves leaves `r` for an immediate defender win (`hunt/cert-support:DESIGN_C_REL.md:482-507`; `5e06c29c:PROOF_QUIET_LOCALITY.md:127-220`). C-REL proposed adding a second disjoint defender count-five as a soundness-shaped NQ2 mutation. Separately, the retained NQ3 far-five construction exercised that remote-threat attack class; shifted/rebound strict replay rejected it and the round reported **no verifier soundness finding** (`hunt/cert-support:DESIGN_C_REL.md:497-511`; `hunt/cert-support:HUNT_REPORT_CERT_SUPPORT.md:126-136`). These remain hostile locality tests, not evidence that an unsound checker accepted them.

**[PRIOR-ROUND FACT]** O4 is marked “dissolved” only in C-REL's redesigned strict-discharge architecture because full target replay redoes global analysis, leaf/hitting checks, dispatch, zone exposure, legal-store, and terminal checks. If `HintMatch` becomes sufficient, the disjoint remote count-five restores O4 as fatal (`hunt/cert-support:DESIGN_C_REL.md:619-635,1155-1168`).

**[DESIGN — O4 RULING]** An exact-root checker evades the C-REL transfer/locality failure **only if** it globally regenerates every relevant fact: full legal/generator sets (O3), remote threats/goals (O4), terminal outcomes (O5), and legal-store/WF anchors including `SecondStone.first` (O6). Binding full occupancy while scanning only attacker-local windows still inherits the bug class. These global primitives must themselves be in CP-O27's proved correspondence.

**[DESIGN — O4 RULING]** A support-only frontier, a trusted `exhausted=true` marker, a list checked only for sound membership, or a checker that validates listed children without proving generator equality inherits the combined O3/O4/O5/O6 failure class. At a Choice, 537 valid refutations do not refute a 538th omitted winning move.

**[PRIOR-ROUND FACT — SCOPE CORRECTION]** NQ2's unique `r` is quiet: the retained artifact says the exact forcing predicate is false and recovery requires the full-legal consume fallback (`5e06c29c:PROOF_QUIET_LOCALITY.md:172-179`).

**[CODE FACT — SCOPE CORRECTION]** Frozen CP1 has quiet-turn consumption off, and a nonterminal `SecondStone` continuation is pending only when it creates a claimant threat and reaches the small-defender-reply boundary (`packages/hexfield_eq/rust/src/tss_solver.rs:6125-6136`). Therefore NQ2 is **not** a direct CP1 generator-omission regression; a correct CP1 checker may omit `r`.

**[DESIGN]** Use NQ2 as the contract-vs-game boundary test and as a mandatory regression if a later grammar enables quiet/full-legal consumption. For CP1 itself, first freeze a different exact-root remote fixture whose winning edge is independently proved to belong to `AttackEdges_CP1`; only that fixture may test omission from a CP1 `ChoiceExhausted` trace.

### 5.5 Why route (c) is not realistic for 3.1

**[DESIGN]** Full correctness would have to cover not only the mathematical df-pn loop but also exact `HexoState` apply/undo, incrementally maintained windows, candidate maps/sets and ordering, atomic pair keys, hash maps and allocation failure, environment sampling, staged mutable arena sharing, positive fragment caches, certificate compaction/relabeling, panic behavior, and serialization. It would then need a trusted account of compiled Rust and the Lean/Rust boundary.

**[RISK]** Verifying a freshly designed small functional core and treating the current engine as an optimized refinement might eventually be rational. Verifying the present 14k-line solver in place is a separate program that would delay the negative authority boundary for many months.

**[DECISION]** Route (c) is not the recommended 3.1 architecture. Reconsider it only if route (b) shows that the independent generator checker necessarily recreates nearly the whole solver and a verified-core rewrite becomes cheaper than maintaining two implementations.

### 5.6 Recommended sequence and effort

**[ESTIMATE]** The recommended route-(b) sequence is:

1. **[ESTIMATE]** **Stop taxonomy and capture seam — 1-2 sessions.** Add only the proposed default-off/test-only stop provenance, freeze exact examples of cap, intermediate cutoff, final refutation, stall, and materialization failure, and make no hard negative result.
2. **[ESTIMATE]** **Lean CP1 grammar and finite dual — 4-6 sessions.** Land CP-O1/O2/O17, then a simple unshared `checkNo_sound` over tiny fixtures. This can proceed while the existing L17/T11/T6 lane finishes.
3. **[ESTIMATE]** **Negative DAG/checker and hostile mutations — 3-5 sessions.** Add exact replay, generator equality, a proved-in-contract CP1 remote-edge mutant, NQ2 as an out-of-contract boundary fixture, disjoint-remote threats, acyclicity, limits, and checker independence.
4. **[ESTIMATE]** **Rust emitter and exact correspondence slices — 3-5 sessions.** Emit from completed arena provenance; start eager+batch+TT-disabled; fail closed on every unsupported tag.
5. **[ESTIMATE]** **Operational completeness and positive materialization — 5-10 sessions.** Prove PN/DN, saturated/unsaturated thresholds, staged fairness, cap-free termination, exact structural-edge guards, and CP-O31 for the minimal profile.
6. **[ESTIMATE]** **Optimization refinements — 4-7 sessions.** Add TT sharing/refresh, lazy frontier, then incremental=batch; add features only one at a time.
7. **[ESTIMATE]** **Integration/resource gate — 2-4 sessions.** Bound codec/checker resources, seal `NO_CONTRACT_WIN`, and retain `UNKNOWN` for all rejected/incomplete cases.

**[ESTIMATE]** Summing the slices gives **22-39 focused sessions**, roughly 9-18 serial weeks with ±50% uncertainty, excluding the parallel T6 chain. A first verified-exhaustion-certificate milestone (Candidate C, without scheduler completeness) is plausibly 12-22 of those sessions. Full Candidate B is dominated by exact generator/checker execution, saturation/fairness, and positive materialization—not by the already-landed base/D17 certificate theorems.

### 5.7 Kill criteria for route (b)

**[DESIGN]** The following discoveries kill or materially rescope the recommended route; they are not ordinary bugs to wave through:

1. **[KILL — independence]** The checker cannot rederive exact Choice/Universal sets without calling the same solver generator or trusting solver-produced summaries, and no small proved correspondence layer can separate them. This would leave the central omission bug mirrored.
2. **[KILL — semantics]** The T6 capstone or a hostile formal review produces a counterexample to the exact kernel/commutation semantics used by CP1, with no local contract repair.
3. **[KILL — non-succinctness]** On three representative genuinely exhausted roots, even DAG compaction plus streaming cannot keep extra checker peak below a predeclared 256 MiB or checker wall below 10% of search wall. This kills hot-path integration; an offline-only theorem may remain worthwhile.
4. **[KILL — no target population]** After the stop split, no representative uncapped `UNKNOWN` is genuinely structurally exhausted—every case is capped, stalled, boundary-invalid, or incomplete. Then negative certification has no current consumer; finish scheduler repairs before continuing the certificate pipeline.
5. **[KILL — version churn]** The contracted generator changes faster than a versioned checker/proof can be maintained, and stable backward verification of old traces is rejected as a product requirement. A theorem about an unshippable moving target is not worth sealing.
6. **[KILL — execution bridge]** The only feasible executed checker is a handwritten translation whose Lean correspondence is as large as full search verification. At that point reconsider a small verified functional core (route c') instead of pretending route (b) closed the boundary.
7. **[KILL — positive materialization]** CP-O31 requires proof-size-aware re-search or a new positive artifact format whose implementation/proof surface is comparable to route (c), and the product requires the full finding theorem immediately. Then land Candidate C as a narrower negative-authority milestone or rescope the program before claiming search-discipline completeness.

**[DESIGN]** The 256 MiB/10% resource figures are proposed go/no-go thresholds, not measurements. Freeze them before the first representative exhaustion run; do not tune them after seeing results.

## 6. Boundary map

**[DESIGN]** This table is part of the theorem statement. A consumer may not silently promote a right-hand-column non-guarantee into a conclusion.

| Boundary | What the covered theorem gives | What it does **not** give / failure disposition |
|---|---|---|
| Exact proposition | **[DESIGN]** `NoContractWin(CP1-a49e8abd-v1, exact_root, claimant, H, S, C)` after checker acceptance | **[NON-CLAIM]** no statement about another root, claimant, clock, horizon, structural depth, grammar version, or flag set |
| Contract vs game | **[DESIGN]** absence of a positive derivation in the frozen forcing grammar | **[NON-CLAIM]** no unrestricted Hexo no-win theorem; arbitrary legal quiet wins may exist outside CP1 |
| Contract vs strict verifier | **[DESIGN]** every CP1 derivation is intended to translate to the strict positive grammar | **[NON-CLAIM]** absence of every certificate the strict verifier could accept, because verifier `Choice` does not re-check `WideTurnGate` (`hunt/census-deep:CENSUS_CANDIDATES.md:18-23`) |
| Positive verdicts | **[DESIGN]** existing verifier-backed `WIN`/`LOSS` authority remains unchanged | **[NON-CLAIM]** Candidate C does not re-prove all Rust positive-verifier code; that checker and its Lean/Rust correspondence remain in the trusted bridge until separately verified |
| Node cap | **[DESIGN]** no negative is emitted when the cap binds | **[NON-CLAIM]** a capped `UNKNOWN` says nothing about existence of a contract win |
| Stall/cutoff/failure | **[DESIGN]** rejected as `UNKNOWN(Incomplete)` unless every used cutoff becomes valid checked `StructuralBoundary` evidence or the witness closes shallow without it | **[NON-CLAIM]** `nodes < cap`, `dn=0`, final loop exit, missing/nonadvancing cutoff, or failed positive materialization is not exhaustion |
| Search liveness | **[DESIGN]** Candidate B eventually finds a present contract win in the modeled uncapped machine after CP-O31 discharges `MaterializationComplete` | **[NON-CLAIM]** Candidate C alone guarantees only that an accepted negative is true; an emitter/search bug may loop, cap, stall, or fail closed |
| Structural horizon | **[DESIGN]** completeness under recursive-state cap `S=min(H-p0,256)` and exact expanded certificate recursion cap `C=256`; edge-local successes from an admitted parent may end beyond `S` when within `H,C` | **[NON-CLAIM]** state depth and certificate depth are not interchangeable; no deeper recursive-state derivation is covered even when `H=u32::MAX` |
| Interior census | **[DESIGN]** flag off; optionally flag-on only under proved `u32::MAX` inertness arithmetic | **[NON-CLAIM]** no finite-horizon flag-on completeness before CP-O30's full proof |
| Lazy frontier | **[DESIGN]** included only after CP-O24 eager/lazy bisimulation | **[NON-CLAIM]** current empirical edge/node/certificate identity alone is not a theorem |
| Incremental defender | **[DESIGN]** canonical grammar is batch; test consume may enter only after CP-O25 or per-node independent batch comparison | **[NON-CLAIM]** observed 31/31 identity or shadow equality on visited states does not prove all reachable states |
| TT and 1 GiB | **[DESIGN]** cache/index retention is optional; a full wide index may refuse keys without changing truth | **[NON-CLAIM]** no promise of wall time, memory below process RSS, or successful termination at 1 GiB; narrow/direct-map negative caching is outside the initial theorem |
| Shared fragments | **[DESIGN]** off in the initial covered profile | **[NON-CLAIM]** no completeness claim for warm fragment import until exact hit/context refinement is proved |
| Round-3 flags | **[DESIGN]** `quiet_turn_or_edges=Off`, `ranked_unforced_defender_zone=Off` | **[NON-CLAIM]** no theorem for `round3_consume`, narrow compatibility, K-reply consume, or zone/ranked fallback grammars |
| Experimental/test flags | **[DESIGN]** production `+1` thresholds and frozen CP1 priors | **[NON-CLAIM]** no theorem for threshold-delta experiments, live-ge3 seeding, quotient/closure instrumentation that changes behavior, cap-resume execution, or future flags until versioned/refined |
| TT/hash bugs below the model | **[DESIGN]** an independent checker can reject a bad trace when exact replay/generator equality differs | **[NON-CLAIM]** a common bug in shared `HexoState`, window analysis, legal move generation, or coordinate arithmetic can affect both producer and checker unless those primitives are also related to Lean |
| Apply/undo and future keys | **[DESIGN]** CP-O27 requires exact transition/key correspondence | **[NON-CLAIM]** before CP-O27 lands, memory corruption, wrong undo, or a collision/equality bug remains in the Rust boundary |
| Checker/emitter codec | **[DESIGN]** only a bounded, versioned, exact-root artifact accepted by the proved checker has authority | **[NON-CLAIM]** truncated, oversized, cyclic, stale-version, or parse-failed artifacts yield no fact; a checker implementation bug remains TCB until execution correspondence lands |
| Resource guarantees | **[DESIGN]** checker has explicit node/edge/witness/depth/memory limits and fails closed | **[NON-CLAIM]** no guarantee that every true negative has a certificate within deployment limits; resource refusal stays `UNKNOWN` |
| Positive materialization | **[DESIGN]** existing `WIN` remains hard only after strict verification; Candidate B requires CP-O31 | **[NON-CLAIM]** current PN=0 does not guarantee an emitted certificate below the 100k-node/1M-edge-style bounds, and an oversized proof may mask another admissible proof |
| Root API preconditions | **[DESIGN]** theorem assumes a well-formed exact root and supported post-opening/phase conditions stated by the grammar | **[NON-CLAIM]** zero-cap, expired horizon, oversized-root, malformed history, or unsupported opening/terminal inputs are not silently turned into negatives |
| Dual/LOSS completeness | **[DESIGN]** root-side CP1 WIN completeness | **[NON-CLAIM]** no completeness theorem for independent opponent attack/`SolveGoal::Loss`; pair-complete `Both` normally does not spend a dual budget |
| Version evolution | **[DESIGN]** old artifacts remain checkable only by their frozen checker/specification | **[NON-CLAIM]** no automatic transfer of a theorem to later candidate tiers, generator repairs, or different contract identifiers |
| Lean/compiler/hardware trust | **[DESIGN]** mathematical claims rely on the Lean kernel and the stated execution bridge | **[NON-CLAIM]** no protection from a compromised Lean kernel/toolchain, unsound extraction/FFI/compiler, OS/hardware faults, cosmic rays, malicious process memory mutation, or incorrect binary deployment |

### 6.1 Initial covered flag matrix

| Setting | Initial status | Admission condition for later coverage |
|---|---|---|
| `vcf_pair_complete=true` | **[DESIGN]** in | frozen CP1 definition |
| `quiet_turn_or_edges=Off` | **[DESIGN]** in | consume requires a new grammar/proof |
| `ranked_unforced_defender_zone=Off` | **[DESIGN]** in | consume requires a new grammar/proof |
| `TSS_LAZY_FRONTIER=0` | **[DESIGN]** first emitter/proof profile | none beyond base proof |
| `TSS_LAZY_FRONTIER=1` | **[DESIGN]** staged admission | CP-O24 |
| batch defender planner | **[DESIGN]** canonical/in | CP-O15 correspondence |
| `TSS_INCR_DEFENDER=1` under `cfg(test)` | **[ASSUMPTION]** conditional only | CP-O25 or checker-side exact batch comparison |
| wide TT disabled or arbitrary admission cap | **[DESIGN]** staged admission | CP-O26; truth independent of retention |
| `TSS_SHARED_FRAGMENTS=1` | **[DESIGN]** out | verified exact-hit/context refinement |
| `TSS_INTERIOR_CENSUS_GATE=0` | **[DESIGN]** in | base profile |
| census gate on, `H=u32::MAX` | **[DESIGN]** conditional inert extension | CP-O30 arithmetic and root-clock precondition |
| census gate on, finite `H` | **[DESIGN]** out | full semantic/correspondence proof |
| round-3/K-reply/zone/narrow-compat profiles | **[DESIGN]** out | separate versioned theorem |
| node cap finite and binding | **[DESIGN]** never a negative theorem | remain `UNKNOWN(Capped)` |

## 7. Acceptance gates for the 3.1 program

**[DESIGN]** The program may call the width contract “certified complete” only when all five gates below are green:

1. **[GATE]** **Exact theorem gate:** Candidate A and `checkNo_sound` are kernel-proved for a versioned CP1 grammar. The grammar/checker subgate may land before, but full-profile certification also requires the explicit post-opening T6 semantic cases actually used.
2. **[GATE]** **Termination gate:** Rust distinguishes stage events, cap, shallow or boundary-dependent structural exhaustion, missing/nonadvancing cutoff, stall, materialization, and precondition stops; only provenance-backed structural exhaustion can invoke the no-checker.
3. **[GATE]** **Generator gate:** every negative Choice has independently checked exact child-set equality; a proved-in-contract CP1 remote-edge omission and named disjoint-remote threat regressions are rejected. NQ2 is required only as an out-of-contract boundary fixture, or as a direct omission test for a later quiet/full-legal grammar.
4. **[GATE]** **Operational gate:** Candidate B or an equivalent refinement proves uncapped df-pn progress/fairness for the exact covered flags, including saturation, exact structural costs, and CP-O31 positive materialization; TT retention is erasable and lazy/incremental modes are separately refined.
5. **[GATE]** **Execution gate:** the executed checker/codec/root binding is connected to the proved checker, bounded, fail-closed, hostile-tested, and the sealed mint cannot be reached on cap/rejection.

**[NON-CLAIM]** Passing gates 1, 2, 3, and 5 without gate 4 yields a valuable certified-negative result boundary but not the stronger claim that every uncapped Rust execution finds every present win. The paper and API must call that milestone “verified exhaustion certificates,” reserving “search-discipline completeness” for gate 4.

## 8. Priority order

**[DESIGN]** The top five R-CP1 obligation groups, in risk/value order with prerequisites grouped, are:

1. **[OBLIGATION]** CP-O16/CP-O28/CP-O31: make structural edge costs, capped/exhausted/incomplete stops, and positive materialization exact and fail closed.
2. **[OBLIGATION]** CP-O1/CP-O2/CP-O14/CP-O15/CP-O17/CP-O18: define the finite positive/negative CP1 grammar, independently regenerate the exact attacker/defender frontier, and prove the negative checker sound.
3. **[OBLIGATION]** CP-O20/CP-O21/CP-O22/CP-O23: prove tagged recurrence correctness, unsaturated and saturated progress, global non-starvation, and shallow/boundary-dependent staged coverage.
4. **[OBLIGATION]** CP-O27/CP-O29: close the Rust state/generator/tag/clock/trace and executed-checker correspondence boundary.
5. **[OBLIGATION]** CP-O24/CP-O25/CP-O26: admit the official lazy + incremental + 1 GiB cache profile by refinement, never by empirical identity alone.

**[DESIGN]** CP-O11/CP-O12/CP-O13 (L17 → T11/T11.1 → full T6 capstone) is the parallel semantic prerequisite for including the current full T6 region in group 2's grammar; it is not replaced by any operational obligation above.

**[RISK — PROGRAM MAXIMUM]** The single largest risk is CP-O27: an exact, maintainable correspondence between the mutable Rust generator/search state and the finite Lean grammar. The T6 chain is difficult but already surrounded by substantial proved machinery; the Rust completeness boundary is genuinely new.
