# Hostile review of the F+H_W report, round 1

The exact-copy gate theorem survived the hostile review. No coordinate claim
failed. The report is not installable as written: it contains one false
proof sentence, an incomplete D17-composition argument, an undefined allowance
for larger labels which contradicts the claimed shrink inequalities, and no
normative D13/T7 integration text. These defects have local repairs. No item is
UNDECIDED.

## Verdict table

| Item | Verdict | Basis |
|---|---|---|
| Opening `PARTIAL` verdict | CONFIRMED | The exact-copy, target-specific debit is sound; no scalar D14 debit or zero-cost substitution theorem is proved. |
| Claim that this is the “strongest” branch-coherent weakening | REPAIR | No optimality theorem supports “strongest.” Apply R1. |
| §1.1 reachable disjoint-hit race | CONFIRMED | The full history is legal and nonterminal at every setup prefix. The displayed family is the complete threat family, has `tau=2`, and has kernel `{a,b}` disjoint from `W`. The defender wins on the second ignored-threat fill. |
| §1.2 obstruction after coupling divergence | CONFIRMED | The fragment counts, exact ghost threats, kernel, real win, and all eight ordered exact-hit LOSS lines check. The claimed `tau=3` is correctly the transversal number of the three named LOSS witnesses, not of the complete leaf threat family. |
| §1.3 pressure is not a coupling invariant | CONFIRMED | The ghost family has two disjoint empty pairs and `tau=2`; the real `(4,0)` stone kills the first and leaves `tau=1`. |
| §1.4 substituted hit corridor | REPAIR | The masks and both global minimum-distance margins are correct, but “after one shared attacker turn” does not name that turn. Apply R2. |
| D19 protected exact-copy forcing gate | CONFIRMED | Tightness makes `K(Q)` finite and nonempty. Checkpoint carriers cover the complete named masks, including ghost-legal carriers, and the two-phase discharge is sound. |
| D20 exact `f` recurrence | REPAIR | The recurrence and gate cost zero are sound for real-only frontier opportunities. The later unrestricted larger-label allowance is incompatible with `f<=r`. Apply R1, R3, and R4. |
| D20a exact `Q^D` recurrence | REPAIR | The child-coherent gate maximum, `b` escape floor, and dual-purpose indicator are sound. The unrestricted larger-label allowance is incompatible with `Q^D<=E^D`. Apply R3 and R4. |
| D21 debited zones | REPAIR | The four terms are sound for exact D20 labels, but the text does not amend D13/T7 and therefore is not an install-ready closure definition. Apply R3 and R5. |
| L15 gate transfer and escape | CONFIRMED | For `d` outside `K`, the at most `b-1` later placements cannot hit every member of `F_Q` not hit by `d`. A surviving initial empty set remains legal by L1/T1. A defender win elsewhere is explicitly retained as the other branch. |
| L16 weighted hazard bounds | REPAIR | The inequalities and escape coverage are correct for exact labels, but “every suffix capable of producing” is undefined. Apply R3, R4, and R6. |
| L17 protected-occupation part | REPAIR | The conclusion is sound, but the assertion that every placement legalized by a copied gate stone is ghost-legal is false when that cell is ghost-occupied. That case is A2 and creates no `X`. Apply R7. |
| L17 completion part | REPAIR | The split is exhaustive and charges copied hits in `W` in every continuing case. The text should state explicitly that post-coupling fills in the remaining escape turn are inside the full `b` floor. Apply R7. |
| T11 exact-copy soundness | REPAIR | The theorem survived the clock, terminal-order, gate, and DAG attacks. Its installable proof depends on repaired L16/L17, exact labels, and an unambiguous node taxonomy. Apply R1 and R3-R7. |
| T11.1 D17 compatibility | REPAIR | Full undebited D17 tests are sufficient, including inside gated subtrees, but the proof omits D17 condition 8, does not state the provenance split for inherited seeds, and is ambiguous for a checkpoint whose deadline is entry to `C_s` itself. Apply R4 and R8. |
| §4.1 strict exposure sharpness | CONFIRMED | The 15+15 position is nonterminal, radius-8 connected, and has D-alive maximum 3. The named singleton family has `K={h}`. Old exposure is 3, new exposure is 2, and the legal LOSS line attains two `W` fills and count 5. The complete root threat family also contains `{h,(6,0)}`; D19 permits the report to name only `{h}`. |
| §4.2 dual-purpose coefficient sharpness | CONFIRMED | `Q(W')=3` is attained by `h,(6,0),(7,0)`. Removing the indicator gives 2 and undercounts that line. |
| Exact sharpness of every `f` seed reduction left open | CONFIRMED | This is an explicit nonclaim, as required by the originating specification. |
| D14/D15/D16 integration | REPAIR | Keeping the full clocks is correct, but installation must say explicitly that a D19 gate remains an ordinary `+1` AND for `B`, `r`, and `E^D`; only `f` and `Q^D` use gate clauses. Apply R4. |
| D17/T9 integration | REPAIR | A substituted transition must retain its `+1`; §1.4 supports that claim. Nested gated subtrees require the complete R8 proof text. |
| D18/T10 DAG integration | CONFIRMED | Reverse topological evaluation is well-founded on a finite DAG, and unfolding preserves one gate label, child map, clock, and both maxima. |
| Verifier procedure | REPAIR | It is finite for exact labels. Its `Q<=E<=B` query bound and ordinary-node closure story require R3 and R5. |
| GAP 1, target-independent D14 debit | CONFIRMED | No such debit is proved. |
| GAP 2, substituted forced hits | CONFIRMED | No zero-cost substitution is proved, and §1.4 shows why the direct rule fails. Apply the terminology correction in R9. |
| GAP 3, automatic net shrinkage | CONFIRMED | Checkpoint roles can enlarge `Prot`; only the fixed-certificate hazard terms are compared. |
| GAP 4, slack pressure | CONFIRMED | For `tau<b`, T6 makes the extendable kernel all legal cells, so the report proves no debit. |

## Proof attacks

L15 survives blocking by the defender's remaining placements. For an
off-kernel first reply `d`, `tau(F_Q \ d)>b-1`. The set of all later placements
in that turn has size at most `b-1`; therefore one named initial empty set
avoids both `d` and all later placements. It has one or two cells, and permanent
shared attacker stones keep those cells legal. A defender win on any placement
of that turn is not suppressed; it is the alternative later excluded by L17.

No checkpoint carrier can slip into `X` between an ordinary check and gate
entry. A ghost-legal carrier is in `Z_dir` and cannot be dismissed. A
ghost-illegal real placement at a carrier traces to an earlier checked
ghost-legal seed. OR edges add no defender stone. A copied gate edge adds a
shared stone. If a carrier is occupied on a copied exact branch, permanence
prevents the exact child from reaching a later label in which that carrier is
empty. This covers checkpoint carriers which are ghost-legal as well as
ghost-illegal chains.

The missing `+1` in the D20 gate clause is valid only because `f` bounds
opportunities to create `X`, not defender placements. A copied gate stone is
shared. If it supplies legality to a later ghost-empty cell, that later cell is
a newly checked ghost-legal seed. If the later cell is ghost-occupied, the move
is A2 and creates no `X`. Actual defender-placement bounds continue to use
`B`, `r`, and `E^D`.

The D20a gate maximum is branch-coherent. A copied branch contributes exactly
`1[d in W]` plus its own child value; an off-kernel branch contributes the
entire remaining `b` placements. The escape deadline may exceed the old
resolution of the abandoned subtree. This does not break the proof: old `B`
at a `b=2` gate is at least 2 because every nonterminal `D,b=1` child has
budget at least 1, and at a `b=1` gate it is at least 1. Ancestor nesting covers
the defender placements, while the enlarged absolute horizon covers the two
attacker placements.

The L17 completion split covers copied hits. A copied hit in `W` is shared and
cannot be a first real-only fill; it is charged by `H_W`. With no mapped-prefix
`W` dismissal, gate-entry `(MI)`, `(FG2)`, and the full escape floor give at
most `3+2=5` or `4+1=5`, including a first `W` fill later in the escape turn.
If the off-kernel reply itself is the first real-only `W` fill, the same bound
applies. Every other first real-only fill is ordinary and is covered by the
touched/virgin count.

T11.1 needs a joint D17/D21 provenance argument. A transition introduced by
D17 is not justified retroactively by a smaller `f` or `Q^D` value. It is
justified by D17's full transition-inclusive `r` and `E^D` tests and inherited
through the selected subtree. New D21 dismissals inside that subtree use the
debited clocks. Repair R8 states this split and includes entry checkpoints and
LOSS remainders.

## Install-ready repairs

### R1. Scope, node taxonomy, and `F` terminology

Replace “The strongest branch-coherent weakening proved here” with:

> A branch-coherent weakening proved here uses protected exact-copy forcing
> gates.

Replace the definition of “ordinary AND” with:

> For T11, an ordinary node is a D21-governed internal AND node. It may use a
> T11.1/D17 envelope dismissal by dismissal. A D19 gate is not ordinary. A T6
> kernel-region node remains governed by T6 and is outside this extension
> unless a separate equal-position T6 handoff is declared.

Add after D20's path interpretation:

> Here `F` means full-cost placements: every ordinary defender opportunity and
> every placement in the first LOSS or off-kernel escape remainder. An
> ordinary placement which happens to hit a threat is still counted in `F`.
> Only an exact copied gate placement receives the forced-hit debit, and it
> contributes to `H_W` exactly when its cell lies in `W`.

### R2. Complete the §1.4 construction

Replace “After one shared attacker turn” with:

> The shared attacker turn plays `(33,1)` and `(33,2)`. Both cells are legal at
> turn start from the shared stone `(33,0)`; neither prefix is terminal. They
> leave the minimum distances from `(5,8)` equal to 8 in the real position and
> 9 in the ghost position, and leave the pre-relay distance from `zeta=(5,16)`
> equal to 9 in both positions. The real placement `(5,8)` then reduces only
> the real distance to `zeta` to 8.

### R3. Require exact D20 labels

Delete the larger-label allowance in D20a and replace it with:

> The verifier uses the exact reverse-topological values in the displayed
> recurrences. These are the labels for which L16(3), the comparisons
> `f_N(rho)<=r_N(rho)` and `Q_N^D(W)<=E_N^D(W)`, and the `B`-bounded finite
> query procedure are asserted.

Without this repair, assigning a larger finite `Q` to a node with old exposure
2 is called admissible by the prose but immediately falsifies `Q<=E`, the
claimed zone shrink, and verifier step 6. No displayed inequality currently
restricts such a label.

### R4. State which clocks debit a gate

Add after D20a:

> For D14-D16 and for a full D17 envelope, a D19 gate remains an internal AND
> node: `B`, `r`, and `E^D` retain their original `+1` inequalities over every
> `K(Q)` child. Only `f` and `Q^D` use the D20 gate clauses. Consequently the
> full clocks cover all defender placements in an off-kernel escape, while the
> debited clocks measure only their stated hazards.

### R5. Compose D21 with D13, T7, §9, and §12

Add the following normative clause:

> **D13/T7 augmented clause.** At a D21 ordinary node set
> `R_cert^FH(𝒸,N)=Z_dir^FH(N) ∪ Z_seed^FH(N) ∪ Z_touch^FH(N) ∪
> Z_virgin^FH(N)`. Any independently nonempty `S(N)` containing
> `R_cert^FH(𝒸,N)`, together with (Z4) and all reachable D19 checkpoint roles,
> is sufficient by T11. The optional solver superset is
> `R_search^FH=R_cert^FH ∪ hitting(P_N) ∪ 𝒜(P_N) ∪ r3(P_N)`. At a
> D19 gate this clause does not apply: the certified searched-child map is
> exactly `K(Q)`, and heuristic terms are not added to `S(Q)`.

Replace the first §9 bullet, for the augmented case, with:

> For base D9-D18 certificates the ordinary mandatory zone remains
> `Z_dir union Z_seed union Z_touch union Z_virgin`. For a D19-D21 augmented
> certificate, a D21 ordinary node uses `R_cert^FH`, while a D19 gate searches
> exactly `K(Q)`. For a fixed augmented certificate, exact `f<=r` and
> `Q^D<=E^D` shrink the seed, touched, and virgin hazard terms. Checkpoint
> roles can enlarge `Prot`, so no reduction in total searched-set cardinality
> is claimed. T6 remains a distinct equal-position kernel theorem.

Replace §12 item 1 with:

> **Partially resolved -- protected exact-copy `F+H_W`.** D19-D21 and T11
> prove target-specific `f` and `Q^D` debits at protected tight exact-copy
> gates. No theorem debits scalar `B`, grants zero cost to a D17 substitution,
> or proves net searched-set shrinkage; those extensions remain open.

The historical 147-versus-302 measurements remain labelled as predecessor
heuristic measurements and need no change.

### R6. Make L16(2) a defined statistic

Replace L16(2) with:

> For a fixed window `W`, on every continuation before the certificate
> attacker wins or first enters `W`, or before an off-kernel escape resolves,
> count one for each ordinary defender edge, `1[d in W]` for each exact copied
> gate edge, and every remaining defender placement in the first LOSS or
> off-kernel escape remainder. This count is at most `Q_N^D(W)`.

The reverse induction in the existing proof then proves the statement
literally.

### R7. Repair L17's copied-stone sentence and escape wording

Replace the copied-stone paragraph in the protected-occupation proof with:

> A copied gate stone is present in both games. If it supplies legality for a
> later real placement at a ghost-empty cell, that cell is ghost-legal and any
> dismissal is a newly checked seed. If the later cell is ghost-occupied, the
> move is T3 case A2: it cancels a `Y`-stone and creates no `X`-stone. Hence a
> copied gate stone cannot be an internal link of a ghost-illegal real-only
> chain. Every link of the final `X`-chain was created at an ordinary AND
> opportunity.

Add after the off-kernel no-dismissal count:

> Here “no `W`-cell was ever dismissed” concerns the mapped prefix. Any first
> `W`-fill among the later placements after an off-kernel reply is already
> included in the full `b` escape floor.

### R8. Complete T11.1

Add to the theorem premise:

> The selected role union contains every role live at `C_s` and at every node
> reachable from `C_s`, expressly including a checkpoint role whose deadline
> is entry to `C_s` itself. D17 condition 3 therefore forbids the transition
> cell `d` from occupying such a carrier.

Replace the middle of the proof with:

> D17 conditions 2-8 protect all selected-child roles and all three L3
> channels, including LOSS remainders. If the protected-occupation seed or
> first real-only window fill was introduced at a D21 dismissal, L17 applies
> with `f` and `Q^D`. If it was introduced by a D17 transition, D17 conditions
> 3-5 apply with the full transition-inclusive role rank and window exposure;
> condition 7 carries that envelope through later ghost-illegal descendants,
> and condition 8 covers a LOSS remainder. L16(3) gives `f<=r` and `Q^D<=E^D`
> for later D21 steps. This joint D17/D21 induction remains valid through a
> D19 gate because the full clocks use the old AND inequalities there. An
> off-kernel reply abandons the selected subtree and is charged by its gate
> floor.

This also corrects “D17.2--D17.7” to conditions 2-8.

### R9. Mechanical terminology

Replace `L9-prime` by `L9′`. Replace “the full D17 C1--C3 envelope” by “all
three L3 channels under D17 conditions 1-8.”

## Machine-check results

Command: `python scripts/_fhw_review_check.py`

```text
PASS 1.1-history-legality: fixed Opening plus all 18 later placements satisfy radius-8 legality
PASS 1.1-prefix-nonterminal: neither colour has a complete window at any setup prefix
PASS 1.1-threat-family: complete A-threat empty multiset is the four displayed sets (found {frozenset({(-5, 8), (-6, 8)}): 1, frozenset({(6, -4), (5, -4)}): 1, frozenset({(5, -4)}): 1, frozenset({(-5, 8)}): 1})
PASS 1.1-tau-kernel: tau(F)=2 and K={a,b} (tau=2, K=[(-5, 8), (5, -4)])
PASS 1.1-kernel-disjoint-W: the full extendable-hit kernel is disjoint from W
PASS 1.1-defender-two-fill: u and v are turn-start legal; counts are 5 then 6 and D wins only on v
PASS 1.2-fragment-at-N: N has no A-threat, ghost D-alive maximum 3, and W counts ghost 3/real 4
PASS 1.2-divergent-replies: x and y are legal nonwinning replies and yield X={x}, Y={y}
PASS 1.2-gate-threats: Q has exactly E1,E2 as A-threat empties and tau=2 (found {frozenset({(15, 0), (14, 0)}): 1, frozenset({(25, 0), (24, 0)}): 1})
PASS 1.2-gate-kernel: K=E1 union E2 and is disjoint from W (K=[(14, 0), (15, 0), (24, 0), (25, 0)])
PASS 1.2-real-defender-win: real D ignores the disjoint gate, reaches W counts 5/6, and wins with all threat cells empty
PASS 1.2-loss-leaf: all eight ordered exact-hit lines give the three displayed disjoint empty pairs, tau=3, and D-alive max 3
PASS 1.3-mask-count: ghost masks give two disjoint threats/tau 2; real X at (4,0) kills the first and leaves tau 1
PASS 1.4-substitution-masks: substituted hits create exactly X={(5,0),(35,0)} and Y={(4,0),(34,0)}
PASS 1.4-shared-turn-witness: the repair pair (33,1),(33,2) is a legal nonterminal shared A turn
PASS 1.4-first-link-distance: (5,8) is real-legal at distance 8 and ghost-illegal with nearest distance 9 (real=8, ghost=9)
PASS 1.4-second-link-distance: zeta is real-legal at distance 8 and ghost-illegal with nearest distance 9 (real=8, ghost=9)
PASS 4.1-position: the displayed 15+15 position is nonterminal and radius-8 connected
PASS 4.1-D-alive-maximum: maximum count in a D-alive window is exactly 3 (found 3)
PASS 4.1-singleton-gate-threat: named T is a singleton-empty A-threat with tau=1 and K={h} (K=[(5, 0)])
PASS 4.1-exposure-arithmetic: W starts at 3; old E=3 gives 6 while debited Q=2 gives 5
PASS 4.1-loss-contract: after h and the two A moves, the named LOSS pairs are disjoint with tau=3 and D-alive max 3
PASS 4.1-attained-real-line: the legal two-fill LOSS remainder attains two W-hits/count 5, then A completes a surviving witness
PASS 4.2-dual-purpose-arithmetic: W' starts with one D; Q=3 is attained algebraically and deleting 1[h in W'] gives the false value 2
PASS 4.2-dual-purpose-line: h,(6,0),(7,0) are legal W' hits, realize three units of harm, and do not complete W'
```

INSTALLABLE-WITH-REPAIRS
