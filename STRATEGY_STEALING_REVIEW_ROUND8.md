# R-ST8-REV — Hostile review of STRATEGY_STEALING_ROUND8.md

## Method and proof boundary

Default posture was REFUTE. I read the required corpus in the prescribed order and in full: STRATEGY_STEALING_HEXO.md; Rounds 2–6 with their reviews, including the binding §35, §44, and §53 formulations; Round 7 with §63 and its review; and finally STRATEGY_STEALING_ROUND8.md. I then read the cited production-rule portions of coord.rs, legal.rs, rules.rs, board.rs, state.rs, and tactics.rs and recomputed the relevant traces from the physical positions.

The rule baseline used in the audit is: radius-eight axial legality; one opening placement followed by paired turns; insertion before win detection; six length-six axial windows; and immediate engine termination after a winning append. Whenever Round 8 uses an associated co-terminal append, I tested it against the inherited §53 atomic paired-final-event convention and required that neither terminal history be continued afterward.

No Cargo command, Lean command, harness, executable search, solver, or generated game census was run. The arithmetic below is hand recomputation. Read-only text and Git metadata inspection established:

- declared proof input 175ca45e resolves to 175ca45e3772659f1026ff8116268f78e3b18a06;
- the artifact landed unmodified at e93d9d74d17019fc3aaf8fdacfae9f4cf07e7452;
- its Git blob is 7e57298fa80f3709a9bce47d3bf2d2031f202c71; and
- its recomputed SHA-256 is 7910d2917c2bfbe10a44cdff0c109ed361622a65ee9999a5a3b6929171ffcc7a.

The review distinguishes three logically different outcomes throughout: a physical game stop, failure of a proposed coupling/transfer obligation, and an outcome-level refutation of an alleged-winning strategy. They are not interchangeable.

## Findings

### Finding 1 — REFUTED: §68 is not an exhaustive dichotomy, and its stated explanation for the remaining gap is false

**Quoted claim.** §68 says:

> “What is still missing is a theorem that forces every alleged winner either into the fast class or through a recurring reserve-admitted controller on every one of its S turns. That missing quantifier is exactly why NL_F remains open.”

The decision table also describes the fast row as “S61 forces S49 misalignment,” while §69.1 later calls the fast shadow-terminal-fidelity class “resolved.”

**Independent recomputation.** Even granting the missing universal fast-or-reserve theorem, the two branches would not settle NL_F:

1. S61 makes the fast branch reach S49’s sixth-shadow-versus-fifth-real event. That is a physical shadow F win with no real F win. It refutes this one-for-one terminal-transfer carrier; it does **not** refute sigma, give F a nonloss, or otherwise decide the original game outcome.
2. S59 handles only histories already admitted to R_1(sigma). It does not prove admission after arbitrary S pairs, canonical F-LOCK, or membership of every alleged winner through its horizon.
3. The §68 predicates do not partition the local state space. For example, a (mu_H,mu_R)=(2,3) pair whose first prescription is quiet but whose second prescription ages a newly created deficit-two window is neither S57’s dynamically quiet pair nor S58’s first-prescription cliff.
4. S62 is a non-disjoint negative cover for a named common-phase handler. Except for S63’s mirror-clean physical stop, it detects a terminal-transfer failure; it does not supply a general winning counterplay. Missing/blocked images, wrong-role or unsupported certificates, phase lag, and high-transversal reconciliation remain outside it.
5. Nonisometric recodings, reverse legality, common-only real wins, and simultaneous legality/P5/P5R maintenance remain independent open obligations, as Round 8 itself concedes in §69.

Thus the unresolved region contains at least the fast S49 outcome branch, slow winners, the complement of the quiet/R_1 admission class, and the unresolved ejection and outer-carrier classes. It is not exactly “slow winner versus quiet boundary,” and the absent quantifier is not exactly why NL_F remains open.

**Proposed repair.** Replace the causal sentence with:

> “A universal fast-or-reserve theorem would connect two local interface results, but would not by itself prove NL_F. The fast S49 branch still needs an outcome-level argument, and all outer-coverage obligations listed in §69 remain binding.”

Relabel §68 as a partial interface map, not an alignment dichotomy or exhaustive synthesis. Likewise change §69’s “FAST CLASS RESOLVED” to “FAST ONE-FOR-ONE TERMINAL TRANSFER FORCED TO FAIL; OUTCOME OPEN.”

### Finding 2 — MINOR: the §68 decision table drops hypotheses that its source theorems require

**Quoted claims.**

> “(mu_H,mu_R)=(2,3), dynamically quiet pair | S57 prepays to the reserve class | Conditional on both actual reached prescriptions being dynamically quiet”

and

> “mu_H=2, mu_R<=2 | Reserve handler maintains CAD; a first-event aging hit is catchable”

**Independent recomputation.** S57 assumes a common-live F FirstStone checkpoint **and tau_E<=1** in addition to (2,3) and sequential dynamic quietness. The table’s caveat omits tau_E<=1. The reserve row is also only a local feasibility/catch statement unless all R_1 admission clauses hold: RES_1; inherited live A_FS2 conditions; a handler-generated nonterminal F pair; a causal, first-safe, certificate-fresh, service-admissible, nonterminal S pair; both real-S cells avoiding W_*; and a common-live exit with tau_E<=1.

The detailed statements of S57 and S59 contain these premises, so this is a synthesis-table scope defect rather than a failure of the local theorems.

**Proposed repair.** Add tau_E<=1 and common-live FirstStone to the S57 row. Split the (2,<=2) row into “one-event CAD/catch feasibility” and “rolling maintenance conditional on every R_1 admission clause.”

### Finding 3 — NOTE: S55 and S56’s feasibility and catch/service censuses are exact

**Quoted claims.**

> “An F-CAD_2^st portfolio exists exactly when mu_H>=3, or mu_H=2 and mu_R<=3, or mu_H=1 and mu_R=1.”

> “The post-event prefix admits F-CAD_2^st before the second query exactly when k is in C_R.”

**Independent recomputation.**

- If mu_H>=3, there is no shadow window in the portfolio domain.
- If mu_H=2, every domain window has deficit two. The one-debt inequality permits assignment to a real window of deficit at most three, hence exactly mu_R<=3.
- If mu_H=1, assigned-window terminal readiness requires a real deficit-one window, hence exactly mu_R=1.

The portfolio definition permits many shadow windows to use the same real window, so assigning all domain members to the fixed-order least real minimum is valid. Fixed window/coordinate orders and the §63 Option/None analysis certificate make the selection causal and totalized.

At (2,2), a nonterminal shadow prescription that meets a deficit-two window changes the shadow minimum to one. A single real append makes the real minimum one exactly when it occupies a hole in some real deficit-two window, namely k in C_R. A mandatory nonterminal service continuation additionally exists exactly when the residual urgent family has a legal nonterminal singleton transversal. If the follow-up instead wins for real F, it is a sound stop rather than a service continuation. No alternate portfolio assignment changes these counts.

**Proposed repair.** None.

### Finding 4 — MINOR: S57 is sound and its named class is nonempty, but the tau_E=0 branch omits a coordinate choice

**Quoted claim.**

> “If the urgent family has singleton transversal s, use s on the second real event unless c=s, in which case service is already complete and a legal padding cell is used.”

**Independent recomputation.** The theorem permits tau_E=0. In that case the urgent family is empty, so the quoted conditional does not select the second real coordinate. A fixed legal filler repairs the omission: it cannot increase mu_R, and a resulting real-F win is a sound stop. The production position is far from board exhaustion, so a legal filler exists.

The displayed nonemptiness witness survives every relevant clause. At the S57 entrance:

- real F occupies (0,0),(1,0),(2,0), so W={(0,0),...,(5,0)} has deficit three;
- shadow F occupies (1,0),...,(4,0), so V={(1,0),...,(6,0)} has deficit two;
- the other q-axis windows through those four shadow stones are blocked by physical Shat@(0,0), and other axes contain too few shadow-F stones, so V is the only shadow deficit-two window;
- the urgent family has the legal singleton service cell (0,5).

The reached prescriptions Fhat@(2,5), Fhat@(3,5) miss V sequentially; the first creates no new shadow deficit-two window for the second to age. Real F@(3,0) prepays W from deficit three to two and F@(0,5) services the urgent family. The unchanged S41 rolling pair 2 then reaches a common-live (2,2), tau_E=0 checkpoint. The finite special cases plus fixed least-legal behavior elsewhere give the round-7 §63 totalized policy; the diagnostic strategy is correctly not claimed alleged-winning.

**Proposed repair.** After the singleton case add: “If tau_E=0, use the fixed least legal filler; if it wins, close at the sound real-F stop.”

### Finding 5 — MINOR: S58’s (2,3) cliff is forced, but “any stronger selector” needs the carrier qualifier

**Quoted claim.**

> “Neither canonical service, the S47 least-choice handler, portfolio reassignment, nor any stronger one-for-one selector can maintain F-CAD_2^st—and hence cannot maintain CAD+LOCK—before it must query z_2.”

**Independent recomputation.** At the first microstep, every real F-unblocked window has deficit at least three. One real append can lower the deficit of a window containing its coordinate by only one, so after **any** legal paired real coordinate mu_R is at least two. The shadow prescription hits a deficit-two window, so mu_H becomes one; it cannot win because no shadow window had pre-append deficit one. The real append also cannot win because no real window had pre-append deficit one. S55’s mu_H=1 row requires mu_R=1. Consequently no portfolio exists at the mandatory pre-query prefix.

This is independent of service selection, window assignment, least-order conventions, and portfolio choice. There is no one-for-one escape. A second unmatched real F append might prepay farther, but §65 correctly observes that doing so leaves S40’s common-phase, one-event-per-microstep carrier. The theorem is not a negative about every possible asynchronous or prepayment architecture.

**Proposed repair.** Replace “any stronger one-for-one selector” by “any query-first, common-phase, one-event-per-microstep one-for-one selector.” Keep the existing warning that no permanent failure after the next S turn follows.

### Finding 6 — MINOR: S59 maintains both CAD and augmented assigned-window readiness, not canonical F-LOCK

**Quoted claim.**

> “Every finite concatenation in R_1(sigma) maintains F-CAD_2^st at every reached F query, passes F-LOCK^+ at every shadow-terminal event, completes its singleton E service on every continuing turn, and otherwise closes at a sound real-F stop.”

**Independent recomputation.** The induction preserves both asserted components:

- Under RES_1, S55 supplies Pi_min.
- If mu_H>=3, one shadow append can lower the minimum by at most one; the preserved real reserve has deficit at most two, so a newly relevant deficit-two shadow family remains CAD-admissible.
- At (2,2), first-event aging is caught by S56. With tau_E<=1, the remaining placement either services the singleton urgent family or closes at a sound real-F win.
- If aging occurs on the second event, the quiet first event has already serviced or filled, and a hole of a real deficit-two window changes both minima to one.
- At (1,1), a nonterminal shadow prescription must miss every shadow deficit-one window. The real reserve remains deficit one unless its unique hole is taken as a sound stop.
- Across the S pair, Shat stones can only block/remove shadow F windows, hence cannot lower mu_H. Requiring both real-S cells to avoid W_* leaves an F-unblocked real reserve of deficit at most two; in the mu_H=1 case that same reserve has deficit one.
- At a terminal shadow-F prescription, RES_1 gives a real deficit-one window already assigned by Pi_min. Appending its physical unique hole gives the co-terminal real win and the claimed F-LOCK^+ incidence.

The complete witness also checks out. Starting from the S57 ingress and unchanged S41 rolling pair 2, the pair

> Fhat@(-8,0), Fhat@(5,0) / F@(1,5), F@(4,0)

has a quiet first event and a second-event catch that changes V/W from deficits 2/2 to 1/1. The following real-S pair at (8,0),(8,1) avoids W; the shadow pair at (4,4),(10,0) does not block V; and the exit has tau_E=0. The final associated pair Fhat@(6,0)/F@(5,0) physically completes V/W. Under §53 it is one atomic coupled-final event, and neither terminal state is continued.

Clause by clause, the displayed R_1 cycle enters common-live with RES_1, tau_E=0, and the folded live A_FS2 facts; its handler-generated F pair is legal and nonterminal; unchanged S41 rolling pair 3 is first-safe, certificate-fresh, service-admissible, and nonterminal; both real-S cells avoid W_*; and the exit is again common-live with tau_E=0. The terminal pair is a closure, not a further nonterminal cycle. Thus the witness satisfies the strict class definition rather than merely its scalar CAD inequality.

This is **not** canonical F-LOCK. Round 8 defines F-LOCK^+ using the reserve handler’s selected assigned window and explicitly leaves canonical F-LOCK open. Accordingly, any summary that calls §65 a proof of unqualified canonical F-LOCK is false; §71.1 itself uses the correct augmented notation.

There are two drafting ambiguities, not counterexamples: “all inherited live A_FS2 clauses” should expressly say that the reserve handler replaces the inherited canonical F-service choice, and “sound terminal closures” should cite §53’s atomic paired-final-event convention. With those readings, the finite-prefix special cases, fixed least choices, and None on class exit satisfy §63 totalization.

**Proposed repair.** State both replacement and §53 closure semantics in the definition of R_1. Preserve “F-LOCK^+ only” in every summary and ledger.

### Finding 7 — MINOR: S60’s horizon gap is correct, but its appeal to S51.1 exceeds that lemma’s formal premise

**Quoted claim.**

> “If such a [local placement seven] placement won, it would itself be a legal counterplay refuting the alleged-winning premise; by S51.1 that event would be the contradiction stop.”

**Independent recomputation.** From S15 the physical ownership/count cadence is:

| Local placement | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Owner | Fhat | Fhat | Shat | Shat | Fhat | Fhat | Shat | Shat |
| Owner’s count after append | 3 | 4 | 4 | 5 | 5 | 6 | 6 | 7 |

Fhat cannot win before local placement six. If every compatible branch wins there, the least uniform depth is six. Otherwise take a legal sigma-consistent branch nonterminal after six. Under alleged-winningness, Shat cannot win on local seven or eight: either winning append, followed by arbitrary off-path totalization of the counterstrategy, directly contradicts alleged-winningness. Hence a nonterminal branch reaches placement eight and the next Fhat opportunity is local nine. Therefore d_sigma(h)=6 or d_sigma(h)>=9.

The arithmetic and logic are sound, but round-7 S51.1 is formally stated at a **genuine common-live coupled node**. S60’s arbitrary shadow branch has not been shown to possess a corresponding live real carrier. The direct shadow-game counterstrategy argument above needs no coupled-node premise and no assumption that sigma has already been refuted.

**Proposed repair.** Replace the S51.1 citation in S60 with the direct alleged-winningness contradiction, or generalize S51.1 separately to every reachable live shadow node. Do not describe this citation as “without crossing S51.1” until repaired.

### Finding 8 — MINOR: S61 is strategy-own and count-exact, but it omits the completion step required by d_sigma

**Quoted claim.**

> “S60 changes d_sigma(h)<=8 into d_sigma(h)=6; hence the reached fourth prescription must win on this particular legal counterplay.”

**Independent recomputation.** S50 supplies actual reached sigma prescriptions and a causal legal Shat continuation; sigma is not asked to choose a favorable response. Because d_sigma is defined over complete/maximal counterplays, one sentence is needed: if S50’s finite prefix were nonterminal after local placement six, it could be extended legally to a complete/maximal counterplay, contradicting d_sigma(h)=6. This establishes that the fourth reached prescription wins on sigma’s own history.

The age-six count is exact. At S15 real/shadow F counts are 1/2. The four post-S15 F events change them successively to 2/3, 3/4, 4/5, and 5/6. Thus the fourth shadow prescription is shadow F’s sixth physical stone, while real F has only five. No earlier real-F stop is possible by stone count. The event therefore satisfies S51 cover item 3—shadow terminal and real nonterminal—regardless of service-cell choice, provided the stipulated one-for-one, legal/nonterminal carrier remains in force. Because S51 is a non-disjoint cover, the proof need not exclude overlap with another cover item.

At root, S15 ends after global placement five. Its local placement six is global eleven. If Fhat misses there, globals twelve and thirteen belong to Shat and cannot be terminal under the alleged-winning premise; Fhat’s next opportunity is fourteen. Hence B_sigma=11 or B_sigma>=14. When B_sigma=11, every reached S15 continuation has

> d_sigma(h) <= B_sigma - 5 = 6,

so the FAST_<14 to FAST_8^{S15} bridge is valid. Fast-class nonemptiness remains open.

S61 proves forced failure of the one-for-one terminal carrier, not an outcome-level refutation of sigma. This distinction is the central defect in §68, not a defect in S61 itself.

**Proposed repair.** Insert the complete/maximal-extension sentence before applying d_sigma. Say that the event “satisfies item 3, possibly alongside another cover item,” and include the displayed root-bound equation.

### Finding 9 — NOTE: S62 and S62.2 give exact negative covers at their named common-phase scopes

**Quoted claims.**

> “Every handler in C_CP^S meets the following exhaustive, generally non-disjoint stop cover.”

> “At an actual common F FirstStone checkpoint with tau_E>2 ... some pre-turn urgent E-window W was missed by both real F placements ... [and] the next C_CP^S turn is covered by S62.”

**Independent recomputation.** At a real-S FirstStone microstep m=2; at SecondStone m=1. If 1<=delta_R^S(W)<=m, real S can fill W’s remaining holes successively. Since W already has at least four real-S stones, each hole is within two line steps of physical support and is legal. After each real append the candidate makes its stipulated one actual legal Shat append. If that append wins, the branch stops physically. If none wins, real S completes W within m placements and the associated shadow state is nonterminal; the applicable P5R duty, or the distinct common-only real-win transfer duty, fails. Cross-window real wins only reach the same cover earlier. These alternatives may overlap, exactly as §63 requires.

For tau_E>2, the two real F coordinates form a set of size at most two and therefore miss at least one urgent hole set. That window remains F-unblocked and retains real-S deficit at most two at the genuine common FirstStone handoff, so S62 applies. This conclusion is conditional on an actual nonwinning pair and common-phase handoff. It neither constructs a Shat win nor supplies permanent fencing or positive reconciliation.

The statement is faithful to the inherited barriers: S45’s explicit S30 stress position still has tau_E=5, lacks candidate-own alleged-winning reachability, and remains only a stress case. S31’s six-blocker fence and its availability/interruption obligations remain open.

**Proposed repair.** None. Retain the words “cover,” “named handler scope,” and “common handoff” whenever citing these results.

### Finding 10 — NOTE: S63 is a physical stop and obeys the terminal grammar

**Quoted claim.**

> “Append the associated physical Shat@T(y). If it wins, stop immediately. Otherwise, after any actual legal real second coordinate r, append Shat@T(x) on the associated final coupled step.”

**Independent recomputation.** The mirror-clean premise gives four actual Shat stones in the six-cell engine window T[W], with exactly the fresh holes T(y) and T(x). No separate Fhat-unblocked premise is needed because those four cells are physically occupied by Shat. Each hole is at line distance at most two from one of the four stones. Thus Shat@T(y) is legal; if nonterminal it leaves five stones, T(x) stays empty and legal, and Shat@T(x) physically completes six independently of the real second coordinate r.

If the first shadow append wins, play stops immediately. If the real second append wins, the second shadow append is the associated §53 atomic final-step reflection; if the real append is nonterminal, the shadow completion directly contradicts alleged-winningness. Neither terminal engine history is continued afterward.

The folded C_shield witness also checks every advertised cell:

- T(q,r)=(q-2,r);
- real S already occupies (0,1),(1,1),(2,1),(3,1);
- shadow Shat already occupies (-2,1),(-1,1),(0,1),(1,1);
- W has holes y=(4,1), x=(5,1), whose fresh images are (2,1),(3,1).

The old debt at (0,5) is unrelated and remains physical. This proves diagnostic physical nonemptiness without claiming the diagnostic sigma alleged-winning. Indeed, by the two-hole counterplay, the alleged-winning intersection of the class is already a direct-refutation stop.

**Proposed repair.** None.

### Finding 11 — NOTE: the restored cross-round caveat ledgers are complete, and no formal NL_F theorem is claimed

**Quoted claim.** §69 says the round-7 review’s twelve obstacles and the earlier obligation ledgers remain binding.

**Independent recomputation.** The caveats specifically lost in earlier rounds are present at their required local ledger sites:

- §69.1 row 1 restores arbitrary nonisometric and non-total zero-lag recodings, not merely fixed-isometry recurrence.
- §69.1 rows 4 and 10 distinguish common-only real wins from P5R and retain simultaneous legality, P2, P3, P5, and P5R terminal maintenance.
- §69.2 agenda row 2 explicitly keeps S13’s fixed-isometry FIFO frontier failure.
- §69.2 agenda row 3 keeps S14’s literal one-cell terminal-lag barrier.
- §69.3’s P5 row separately retains simultaneous legality and terminal maintenance.
- §69.3’s P5R row again retains S14 and the common-only transfer duty.

Round 8 repeatedly states that NL_F remains open and does not promote a local stop cover, diagnostic witness, or carrier failure into a formal nonloss theorem. The problem is narrower: Finding 1 refutes §68’s statement that one missing quantifier is the **exact** remaining reason.

**Proposed repair.** Preserve the ledgers. Amend only the §68 synthesis language and the “fast resolved” shorthand.

### Finding 12 — MINOR: the artifact’s provenance omits its landed commit

**Quoted claim.**

> “Landed artifact commit/hash: not yet known.”

**Independent recomputation.** The file is byte-identical to the artifact at e93d9d74d17019fc3aaf8fdacfae9f4cf07e7452. Its Git blob and SHA-256 are recorded in the method preamble. The artifact correctly records the input commit, authoring HEAD c019400ad14e06fa6f600c5462113a74795e3270, branch, no-commit boundary, and unchanged proof corpus. Its only provenance defect is that the now-known landing identity is absent from the landed artifact.

**Proposed repair.** Add landing commit e93d9d74d17019fc3aaf8fdacfae9f4cf07e7452 and SHA-256 7910d2917c2bfbe10a44cdff0c109ed361622a65ee9999a5a3b6929171ffcc7a to the review record. Do not rewrite the audited artifact merely to self-embed its hash.

## Per-theorem verdicts

| Result | Verdict | Exact scope or repair |
|---|---|---|
| S55 scalar CAD feasibility | **CONFIRMED** | Exact three-case census; many-to-one portfolio semantics respected |
| S56 catch/service intersection | **CONFIRMED** | Exact for the stated nonterminal service definition; sound real wins are stops |
| S57 quiet one-debt prepayment | **CONFIRMED WITH MINOR REPAIR** | Named class is nonempty; specify the tau_E=0 filler |
| S58 first-event readiness cliff | **CONFIRMED AT CARRIER SCOPE** | No one-for-one query-first common-phase choice escapes; asynchronous unmatched prepayment is outside scope |
| S59 reserve-one maintenance | **CONFIRMED AT CONDITIONAL R_1 SCOPE** | Maintains CAD and F-LOCK^+ on every admitted concatenation; canonical F-LOCK and universal membership remain open |
| S60 post-S15 horizon gap | **CONFIRMED WITH MINOR REPAIR** | d=6 or d>=9; replace the formally over-scoped S51.1 citation |
| S61 fast-winner forcing | **CONFIRMED WITH MINOR REPAIR** | Candidate-own history and counts are exact; add completion of the finite S50 prefix |
| Root FAST_<14 bridge | **CONFIRMED CONDITIONALLY** | B=11 or B>=14 and B<14 implies local depth at most six; class nonemptiness is open |
| S62 deadline-deficit cover | **CONFIRMED** | Exhaustive, non-disjoint negative cover for C_CP^S; not a constructive shadow win |
| S62.1 first-unsafe boundary | **CONFIRMED** | Exact common SecondStone no-live-repair boundary |
| S62.2 tau_E>2 handoff barrier | **CONFIRMED AT NAMED SCOPE** | Requires an actual nonterminal F pair and genuine common handoff; S45’s tau_E=5 stress case remains unresolved positively |
| S63 mirror-clean stop | **CONFIRMED** | Two actual legal Shat appends physically complete six; §53 terminal closure respected |
| §68 alignment synthesis | **REFUTED** | Not exhaustive; fast misalignment is still outcome-open and multiple outer classes remain |

## Overall verdict

**REFUTED.** The local mathematical core S55–S63 survives at its narrow conditional scopes, subject to the minor repairs above. The advertised §68 synthesis does not: it conflates forced failure of a particular one-for-one carrier with resolution of the alleged winner, and it omits independent open regions. Consequently the four requested dispositions are:

- **§65 — CONFIRMED WITH QUALIFICATION.** The quiet/R_1 class is genuinely nonempty, §63-totalized, and closed correctly at its displayed terminal trace. Every admitted continuation maintains both F-CAD_2^st and the assigned-window **augmented** F-LOCK^+. Canonical F-LOCK is not proved. S58’s (2,3) first-event cliff is selector-independent and has no one-for-one service/portfolio escape.
- **§66 — CONFIRMED WITH MINOR PROOF REPAIRS.** S60’s cadence gap, S61’s sixth-versus-five count, the own-history quantifier, non-disjoint S51 cover use, and the root FAST_<14 arithmetic are correct. The conclusion is carrier misalignment, not an outcome theorem; fast-class existence and slow winners remain open.
- **§67 — CONFIRMED AT STATED SCOPES.** S63 is a physical mirror-clean Shat stop with correct terminal closure. S62/S62.1 and S62.2 are exact negative covers for the named common-phase handlers; they do not cover phase lag or positively solve high-transversal service.
- **§68 — REFUTED.** Its table is only a partial interface inventory. The remaining open region is not exactly the slow-winner/quiet boundary, and the missing universal fast-or-reserve quantifier is not the sole obstruction to NL_F.

## Exact unresolved obstacles after Round 8

1. **Full per-pair and broader zero-lag coverage:** arbitrary-S recurrence, intra-pair changing isometries, total nonisometric point recodings, non-total/window recodings, and indefinite one-repair-per-placement.
2. **Pre-checkpoint and recurring P3 coverage:** one genuine common-phase, serviceable, terminal-faithful history through arbitrary S pairs, not merely concatenations already admitted to R_1.
3. **Coverage outside strict A_FS2:** missing, blocked, or illegal image cells; unreflected real-S terminals; wrong-role occupancy; unsupported certificates; phase lag; uncertified high-transversal exits; and shadow-F terminals outside admitted alignment.
4. **P5R and common-only real-win transfer through every lag/recode:** including S14’s terminal-lag and S25’s older-surplus obligations, shielding, certification, F blocking, and same-step physical supply.
5. **Universal F service and lock:** canonical F-LOCK, arbitrary service compatibility, recurring nonterminal portfolio admission, and horizon membership beyond the named tau_E<=1 reserve class.
6. **Universal shadow-F terminal fidelity and the fast outcome:** later slow/non-reserve terminals still lack same-event real certificates, while S61’s fast misalignment only defeats the tested carrier and does not decide sigma or NL_F.
7. **Reverse legality for spatial carriers:** every inverse/FIFO construction still owes S18, S13, and current unsupported/collision checks.
8. **Global strategy domain and physical persistence:** every event of a universal construction must remain on one total sigma-consistent append-only history with all old stones retaining all rule effects.
9. **Global causality:** outer backing, recoding, and repair must avoid exposing a future real-F coordinate across an S turn.
10. **Universal window/certificate maintenance:** arbitrary S-created windows and reassignment, plus simultaneous legality, P2, P3, P5, P5R, and the distinct common-only real-win duty in one recurring physical handler.
11. **High-transversal service and permanent fencing:** S30’s exact tau_E=5 position, S31’s six-cell cost, cell availability, interruption, S occupation, reconciliation, and compatibility with P3. S62.2 supplies only a conditional negative handoff cover.
12. **Strategy-specific reachability, class membership, and outcome:** fast-class nonemptiness, a slow-winner controller, preservation or deliberate exit of quiet/R_1 membership, handling of real sound stops, and an outcome-level argument after S49.

These are logically independent enough that proving the universal quantifier proposed in §68 would not erase the list. NL_F remains open.
