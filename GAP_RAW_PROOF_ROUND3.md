# GAP-RAW Proof Round 3 — Obligation J research round

**Worktree:** `hunt/gap-raw` at input HEAD
`283348dce09d42b67e364e0b2f2b63166b6b5f4d`  
**Date:** 2026-07-17  
**Disposition:** canonical Obligation J is **REFUTED**; GAP-RAW remains
**OPEN**.  
**Prime directive:** the counterexample is to J's stronger account invariant,
not to perpetual Defender survival.

This document resumes the numbering and conventions of
`GAP_RAW_PROOF_ROUND2.md`. All game terms, `Phi`, `Theta_2`, `B_2`, `I`, `tau`,
`kappa_2`, `S_2`, and `n_1` have exactly the meanings fixed there. In
particular, the normative game is the blanket Maker–Breaker game: only an
Attacker six is terminal, and a normative root need not be the state of an
engine-reachable history.

For state-level displays, write `B_2(P):=Theta_2(P)`. This is shorthand for
round 2's history account `B_2(h)=Theta_2(P_h)` at a history whose current
position is `P`; it does not introduce a second account.

## 14. Executive verdict and exact boundary

**Theorem R3.1 (canonical-renewal counterexample) [PROVEN].** There is a
finite, nonempty, nonterminal Defender-`FirstStone` normative root `P_*` with

`Phi(P_*) = B_2(P_*) = Theta_2(P_*) = 8/9 < 1`

and `I(P_*)=empty` such that, for every legal ordered two-cell Defender reply,
there is a legal ordered Attacker pair returning a nonterminal Defender epoch
with

`B_2 >= 11/9 > 1`.

The exact coordinates and the universal-reply proof are in §§15–16.

**Obligation J [REFUTED].** At the initial epoch of `P_*`, clauses 1 and 2 hold
for every legal Defender reply: `B_2=8/9`, and service is vacuous because
`I=empty`. Nevertheless, every possible actual reply has an Attacker response
violating clause 4. Hence no pure Defender strategy can satisfy J for this
normative root.

**Canonical GAP-GLOBAL-RENEWAL [REFUTED].** The same first-epoch universal
quantifier refutes renewal of the canonical subunit account `B_2=Theta_2` over
all normative roots. This is stronger than the round-2 three-gadget refutation
of O1-prime: `P_*` itself satisfies the original strict `Phi<1` root condition.

**GAP-RAW [OPEN].** The response below raises no alive label above count two:
the charged fresh labels are count two, while additional births can remain
count one. It does not complete a six, and at the resulting epoch `I=empty`
and `tau=0`. Crossing `Theta_2=1` is therefore not an Attacker win. Theorem D2 (`J => GAP-RAW`)
remains a sound conditional, but its antecedent is false.

**GAP-AMORTIZED-ABANDONMENT [OPEN].** The counterexample contains no stored
count-one labels at its root. Its obstruction is a virgin
count-zero-to-count-one-to-count-two birth within one Attacker turn. Thus it
refutes the canonical renewal route, but it neither supplies nor refutes a
fully specified credit/refund account that is allowed to tolerate
`Theta_2>=1`.

## 15. The exact normative root

### 15.1 Eight isolated grade-two labels

Let

`G={(0,1),(1,-1),(-1,0),(3,-2),(0,-4),(-3,3),(1,3),(6,0)}`.

For `i=0,...,7`, put

`b_i=(30i,0)`,

`A_i={b_i, b_i+(1,0)}`,

`D_i=b_i+G`,

and define the Q-axis window

`W_i={b_i+(t,0): 0<=t<=5}`.

All additions here are coordinatewise. The two cells of `A_i` lie in `W_i`,
and no cell of `D_i` does.

**Lemma L9.1 (one-pair isolator) [PROVEN].** For the untranslated gadget
`A={(0,0),(1,0)}`, `D=G`, the only Attacker-alive window is
`W={(t,0):0<=t<=5}`, and it has Attacker count two.

*Proof.* There are thirty-one distinct windows through the two adjacent
Attacker cells: their five common Q-axis windows are counted once. The target
`W` is Q-start 0. Every other one contains a listed blocker:

- The other Q starts are `-5,-4,-3,-2,-1,1`. The starts `-5` through `-1`
  contain `(-1,0)`, and start `1` contains `(6,0)`.
- The R-axis windows through `(0,0)` have R-starts `-5,...,0`; starts `-5,-4`
  contain `(0,-4)`, while starts `-4,...,0` contain `(0,1)`.
- The R-axis windows through `(1,0)` have the same six start parameters;
  starts `-5,...,-1` contain `(1,-1)`, while starts `-2,-1,0` contain
  `(1,3)`.
- Parameterize the QR line through `(0,0)` by `(t,-t)`. Its six starts
  `-5,...,0` are covered by `(-3,3)` for starts `-5,-4,-3` and by `(1,-1)`
  for starts `-4,...,0`.
- Parameterize the QR line through `(1,0)` by `(1+t,-t)`. Its starts
  `-5,...,0` are covered by `(0,1)` for starts `-5,...,-1` and by `(3,-2)`
  for starts `-3,...,0`.

None of those eight blockers belongs to Q-start 0. Thus exactly `W` survives,
with its two displayed Attacker stones. ∎

**Lemma L9.2 (eight-gadget profile) [PROVEN].** For

`A_old = union_{i=0}^7 A_i` and `D_old = union_{i=0}^7 D_i`,

the exact alive profile is

`(n_1,n_2,n_3,n_4,n_5,n_6)=(0,8,0,0,0,0)`.

*Proof.* Lemma L9.1 supplies one count-two label per gadget. Any two Attacker
stones in distinct gadgets have hex distance at least 29, while the diameter
of a length-six window is five. Hence no window contains stones from two
gadgets. A Defender stone in another gadget can only delete a label, and the
eight target sets `W_i` are disjoint and contain no Defender stone. Therefore
the eight displayed labels are exactly the alive family. ∎

### 15.2 Three legal fresh-pair launch sites

For `j=0,1,2`, set

`R_j=100+30j`,

`c_j=(0,R_j)`, `d_j=(1,R_j)`, and `a_j=(0,R_j+8)`.

Put each anchor `a_j` in the Defender set and define

`U_j={(q,R_j):-4<=q<=5}`.

The five Q-axis windows containing both adjacent cells `c_j,d_j` have starts
`q=-4,-3,-2,-1,0`; their union is exactly `U_j`.

**Lemma L9.3 (launch separation and legality) [PROVEN].** The eleven sets
`W_0,...,W_7,U_0,U_1,U_2` are pairwise disjoint. Every `U_j` is disjoint from
all root stones. Moreover `d(c_j,a_j)=8`, and `d(d_j,c_j)=1`.

*Proof.* The `W_i` lie on row 0 in Q-intervals separated by 25 empty cells.
The `U_j` lie on three distinct rows `100,130,160`, while their Q-interval is
always `[-4,5]`. The old Attacker stones have R-coordinate 0; old blocker
R-coordinates lie in `[-4,3]`; and the three anchors lie on rows
`108,138,168`. This proves all disjointness statements. The two distances are
immediate from the axial formula. ∎

Define

`A = A_old`,

`D = D_old union {a_0,a_1,a_2}`,

and let

`P_*=(A,D,Defender,FirstStone)`.

**Lemma L9.4 (normative-root audit) [PROVEN].** `P_*` is finite, nonempty,
disjoint, and blanket-nonterminal; its exact profile is
`(0,8,0,0,0,0)`. Consequently

`Phi(P_*)=Theta_2(P_*)=8/9<1`, and `I(P_*)=empty`.

*Proof.* Finiteness and nonemptiness are explicit. Lemma L9.3 keeps the new
anchors out of every `W_i`, so Lemma L9.2's exact profile is unchanged. No
gadget has more than two Attacker stones, and no length-six window crosses two
gadgets, so Attacker has no six. Each count-two label weighs
`lambda^{-4}=1/9`; there are eight and no other alive labels. Count two is not
imminent. ∎

There are 16 Attacker stones and 67 Defender stones. These counts have no
parity or historical significance: the normative domain fixed in round 2 is
the set of all finite blanket positions with the stated side and phase, not
only positions reachable from `HexoState::new()`.

## 16. Every actual Defender pair loses canonical renewal

Fix an arbitrary legal ordered Defender reply `(x_1,x_2)` at `P_*`. Let `k`
be the number of **distinct** target windows `W_i` met by `{x_1,x_2}`. Because
the eight `W_i` are pairwise disjoint,

`0<=k<=2`.

Similarly, each Defender cell belongs to at most one of the pairwise-disjoint
launch unions. Hence there is an index `j` such that

`x_1 notin U_j` and `x_2 notin U_j`.                         (12)

Choose any such `j` and let Attacker play the ordered pair `(c_j,d_j)`.

**Lemma L9.5 (legal, nonterminal response) [PROVEN].** This ordered pair is a
legal Attacker continuation after the fixed Defender reply, and it returns a
nonterminal Defender-`FirstStone` epoch.

*Proof.* By (12), both launch cells remain empty. The root anchor `a_j` remains
occupied and is at distance eight from `c_j`, making the first placement
legal. The second placement `d_j` is adjacent to the newly occupied `c_j`, so
it is legal under the sequential growth rule. The old components contain two
Attacker stones each and the new component contains two; their mutual
distances exceed a window diameter. Thus no six is completed, and the normal
two-placement phase returns to a Defender epoch. ∎

### 16.1 The four exact transitions

Each `W_i` has `Theta_2`-mass `1/9`. Since `k` counts distinct killed labels,
the two Defender instances of equation (9) telescope exactly to

`B_2(P_*+D@x_1+D@x_2) = 8/9-k/9`.                           (13)

Call this handoff `Q`. It has no alive count-one labels, because `P_*` had
none and Defender placements only delete labels. Also, `c_j` lies in none of
the surviving `W_i`. Therefore, at the first Attacker placement,

`S_2(Q,c_j)=0`, `n_1(Q,c_j)=0`, and `Delta B_2(A@c_j)=0`.    (14)

After `A@c_j`, precisely the five Defender-free Q-axis windows whose union is
`U_j` are count-one labels containing `d_j`. Two distinct cells determine one
axis, so no R- or QR-axis label contains both `c_j` and `d_j`; no old label is
within a window of `d_j`. Hence, immediately before the second placement,

`S_2(Q+A@c_j,d_j)=0` and `n_1(Q+A@c_j,d_j)=5`.              (15)

The second Attacker instance of equation (9) is therefore exactly

`Delta B_2(A@d_j)=5/9`.                                    (16)

Combining (13)–(16) gives the next Defender epoch `P'`:

`B_2(P')=(13-k)/9 >= 11/9 > 1`.                            (17)

This evaluates both exact Defender kills and both sequential Attacker
placements, including the load-bearing `n_1/9` term. It proves Theorem R3.1
and the stated refutations of J and canonical renewal.

The mechanism is not omitted dormant-stock bookkeeping. Equation (9) sees it
exactly: `n_1` is zero at the first trigger and becomes five before the second.
The failure is that `B_2` does not pre-fund a virgin pair birth. Initial slack
is `1/9`; two cells recover at most `2/9`; the fresh source costs `5/9`, leaving
an overshoot of at least `2/9`.

## 17. What remains true about canonical J

The counterexample decides J negatively, but two reductions sharpen the reason
for failure and prevent a successor from carrying unnecessary suppression
obligations.

### 17.1 Debt below one already implies serviceability

**Lemma L9.6 (canonical debt bound) [PROVEN].** At every finite nonterminal
position,

`B_2(P) >= |I(P)|/3 >= tau(P)/3`.                            (18)

In particular, `B_2(P)<1` implies `tau(P)<=2`.

*Proof.* Every member of `I(P)` is a distinct count-four or count-five label.
Its `Theta_2` weight is respectively `1/3` or `1/sqrt(3)>1/3`, proving the
first inequality. Choosing one residual cell separately for each label gives
`tau(P)<=|I(P)|`, proving the second. Strict `B_2<1` then excludes
`tau>=3`. ∎

**Theorem R3.2 (J.3 redundancy) [PROVEN].** Within canonical Obligation J,
clause 3 follows from clauses 2 and 4.

*Proof.* Clause 2 services `I(P)` and leaves an Attacker-`FirstStone` handoff
`Q` with `I(Q)=empty`. By round 2's L1.1, no legal two-placement Attacker turn
from `Q` can complete a window, so every ordered response reaches a
Defender epoch `P'`. Clause 4 gives `B_2(P')<1` for every such response, and
Lemma L9.6 gives `tau(P')<=2`. This is precisely the unripeness statement. ∎

Thus K3's standalone geometry remains **OPEN**, but it is not an additional
load-bearing lemma once canonical robust renewal has been established. The
counterexample shows that robust renewal itself is the false step.

### 17.2 The exact one-pair margin

Let `a=(x_1,x_2)` be one legal ordered Defender reply at epoch `P` which
services `I(P)`, and put

`K(P,a)=kappa_2(P,x_1)+kappa_2(P+D@x_1,x_2)`

and `Q=P+D@x_1+D@x_2`. The sequential definition counts each killed label
once even when both cells lie in it. For a legal ordered Attacker response
`b=(c_1,c_2)`, let `Delta(Q,b)` be the sum of its two exact equation-(9)
increments, with the second `S_2,n_1` evaluated after `A@c_1`. Define

`E(Q)=max_b Delta(Q,b)`.

The service premise gives `I(Q)=empty`, so L1.1 makes every legal two-cell
response nonterminal and both equation-(9) updates are defined. The maximum
exists: a finite nonempty position has finitely many legal first
placements, and each first placement leaves finitely many legal second
placements under the radius-eight rule.

**Lemma L9.7 (strict renewal margin) [PROVEN].** The same actual servicing pair `a`
renews `B_2<1` against every legal Attacker response if and only if

`E(Q) < 1-B_2(P)+K(P,a)`.                                  (19)

*Proof.* Two applications of each exact transition in (9) give

`B_2(P')=B_2(P)-K(P,a)+Delta(Q,b)`.

Demanding this be strictly below one for every `b` is equivalent to (19).
The maximum converts the universal response quantifier without changing the
Defender pair that supplies service. ∎

At `P_*`, every reply has `K(P_*,a)=k/9<=2/9`, while the untouched launch gives
`E(Q)>=5/9`. The right side of (19) is `(1+k)/9<=3/9`. Thus the strict margin
fails by at least `2/9`, exactly as (17) records.

**Corollary R3.2.1 (two-clause form of J) [PROVEN].** Canonical J is equivalent
to the following statement: for every normative root `P_0` with
`Phi(P_0)<1`, one pure Defender strategy has, at every reached epoch, an actual
legal ordered pair which

1. services `I(P)`; and
2. satisfies the strict margin (19).

*Proof.* J implies the two items by clauses 2 and 4. Conversely,
`B_2(P_0)<=Phi(P_0)<1`; item 2 inductively proves clause 1 at all later epochs,
item 1 is clause 2, Theorem R3.2 proves clause 3, and item 2 is clause 4. The
same strategy and the same actual pair occur in both items. ∎

This is a quantifier-clean diagnosis, not a rescue theorem: `P_*` refutes the
second item for every possible pair.

### 17.3 The exact fresh-pair ceiling is already too large

Call an ordered Attacker pair `(c_1,c_2)` *Q-fresh* when neither trigger lies
in an Attacker-alive window of `Q`. This excludes stored count-one and higher
promotions; only windows born from the two new stones can enter `Theta_2`.

**Lemma L9.8 (fresh-pair ceiling) [PROVEN].** A Q-fresh legal ordered pair
injects at most `5/9` into `Theta_2`, and adjacent collinear cells attain the
bound when their five common windows are Defender-free.

*Proof.* A new label contributes to `Theta_2` only if it contains both
triggers. Two distinct cells share a length-six window only when they are
collinear on one lattice axis and have axis distance `d` in `{1,...,5}`. They
then lie together in exactly `6-d` windows. Each new label is count two and
weighs `1/9`; the maximum is therefore `(6-1)/9=5/9`. ∎

The construction does not exceed a supposedly forgotten source bound. It
saturates the exact bound at one of three disjoint launch sites. Eight
disjoint old labels leave only `1/9` slack, and two Defender cells can earn at
most `2/9`, so even the sharp `5/9` ceiling is fatal to canonical renewal.

### 17.4 Pointwise-dominating replacement accounts are excluded

**Corollary R3.3 (no subunit account dominating `Theta_2`) [PROVEN].** The
following assertion is false:

> For every normative root `P_0` with `Phi(P_0)<1`, there exist a (possibly
> root-dependent) pure Defender strategy `S` and numeric account `C` such that,
> against every Attacker continuation and at every `S`-reached Defender epoch
> `h`, `Theta_2(P_h)<=C(h)<1`.

*Proof.* Apply the proposed strategy at `P_*`. Theorem R3.1 supplies a legal
nonterminal response with `Theta_2(P')>=11/9`; pointwise domination forces
`C(h')>1`. ∎

Any replacement that retains a strict subunit invariant must therefore be
permitted to sit below raw `Theta_2` on safe high-grade configurations, with
the credit variables, earning rule, consumption rule, and safety implication
stated explicitly. A dominating account with a different threshold or a
non-threshold structural conclusion is not excluded. Merely renaming a
pointwise upper bound while retaining `<1` cannot repair J.

## 18. Banked tools and claim boundaries

### 18.1 K1, K2, the beta floor, and K3

**Banked results [PROVEN at their round-2 scopes].** K1, K2, the
`beta=(2+sqrt(3))/9` ripe-witness floor, L1.1/L1.2, and Theorems A2/A2-prime
are unchanged by this round.

They do not block the counterexample:

- `tau(P_*)=0`, so K1 and K2 do not select the root reply.
- The chosen response raises no label above count two; the exact legal minimum
  row also has 26 count-one labels. At the next epoch `I=empty` and `tau=0`.
  No ripe witness challenges the beta floor.
- K3 concerns suppressing ripe witnesses. Equation (17) violates an account
  threshold without producing a ripe witness.

**General K3 suppression [OPEN].** Round 2's matching-number-two geometry is
not solved here. Theorem R3.2 says only that canonical clause 4 would have
implied clause 3; it does not prove K3 as an independent statement and does
not make K3 unnecessary for a replacement invariant that permits
`Theta_2>=1`.

### 18.2 D2 and the normative theorem

**Theorem D2 [PROVEN, retained].** The logical implication `J=>GAP-RAW`
remains correct because J.2 is Service in Theorem A2. The newly proved
falsity of J means D2 cannot be used to establish its consequent.

**Theorem A2 [PROVEN, retained].** The exact target for GAP-RAW remains one
strategy whose actual pair services every reached epoch. The account and
unripeness clauses of J were stronger construction discipline, not necessary
conditions for survival. This is why refuting J does not refute GAP-RAW.

### 18.3 Reachability is not an escape from this root

**Normative-domain statement [PROVEN by definition].** Round 2's §1.3
quantifies over every finite, nonempty, nonterminal blanket position with
Defender at `FirstStone` and `Phi<1`. It imposes neither historical
reachability from the empty engine board nor a stone-count parity condition.
`P_*` lies in that stated domain by Lemma L9.4.

A theorem restricted to engine-reachable roots may be interesting, but it is
a strictly weaker theorem and cannot be substituted for GAP-RAW or J as
stated.

## 19. Exact successor resume point

Canonical J cannot be repaired by a different choice rule: the universal
failure occurs before the strategy has made any earlier choice. A successor
round has two honest routes.

**Route A — direct Service construction [OPEN].** Return to Theorem A2 and
construct one strategy whose actual pair services every reached epoch without
requiring `Theta_2<1`. Such a proof must tolerate safe epochs like the
counterexample successor, where `Theta_2>=11/9` but `tau=0`.

**Route B — a non-dominating structural account [OPEN].** Define an explicit
history/state account with exact transitions and a proved safety implication,
allowing credits that make the account smaller than `Theta_2` on harmless
grade-two packings. Corollary R3.3 excludes every pointwise-dominating subunit
account. The counterexample supplies the first mandatory regression: the new
invariant must admit all roots `P_*` and survive the fresh adjacent-pair
transition (13)–(17), rather than excluding it by a reachability premise.

The sharp named remainder is therefore:

**GAP-REPLACEMENT-INVARIANT [OPEN].** Exhibit one pure Defender strategy and
one fully defined invariant, valid from every normative root, which is closed
under the strategy's same actual servicing pair and every legal Attacker
response, and whose Defender-epoch conclusion is strong enough to ensure that
the next actual service remains possible. The invariant must allow
`Theta_2>=1`; if it uses credits, their exact origin and transition must be
formal.

This is not called J. It explicitly drops J's false canonical threshold, and
no equivalence or shrinkage beyond Theorem A2 is claimed.

## 20. Machine regression

### 20.1 New hard gate [VERIFIED]

The test-gated harness now contains
`round3_j_canonical_renewal_refutation`. It independently reconstructs the
coordinate sets and hard-asserts:

- 16 Attacker and 67 Defender stones;
- exact root profile `(n_1,...,n_6)=(0,8,0,0,0,0)`, strict `Phi<1`, exact
  `27*Theta_2=24`, and `I=empty`;
- exactly the eight target window keys and pairwise disjointness of all eight
  `W_i` and three `U_j`;
- root-stone-free launch unions and distance-eight anchors;
- a quotient-complete two-cell defense universe consisting of every empty
  cell in the eleven relevant sets plus two outside sentinels;
- for every quotient pair, an untouched launch, a legal sequential Attacker
  pair, a nonterminal next Defender epoch, and exact
  `27*Theta_2=3(13-k)>=33`; and
- the exact legal minimum row
  `D@(4,0), D@(213,0); A@(0,100), A@(1,100)`, with eleven final count-two
  labels, 26 final count-one labels, and `27*Theta_2=33`.

The quotient is complete for the claimed lower bound. A Defender cell affects
an old target exactly when it belongs to some `W_i`, and affects one of the
five focal birth labels exactly when it belongs to the corresponding `U_j`.
Every cell outside all eleven sets has neither effect, so one and two outside
sentinels represent all off-union cases. The test deliberately admits
possibly illegal Defender representatives, a Defender-favoring
over-approximation; the separately asserted minimum row is fully legal.

The one coordinated run began with zero existing Cargo processes,
`14.765 GiB` free physical RAM, `8.925 GiB` standby cache, and `23.690 GiB`
combined availability. Three detached RAM guards were active. Result:

```text
running 1 test
test gap_raw_hunt::tests::round3_j_canonical_renewal_refutation ... ok
test result: ok. 1 passed; 0 failed
test time: 3.20 s
release compilation: 18.40 s
command wall time: 23.4 s
```

No production rule or strict-verifier source is modified.

## 21. Hostile-review attack surface

1. **Normative reachability.** Attack whether round 2 actually required an
   engine history. It did not: §1.3 quantified all finite blanket positions.
   If the target is changed to reachable roots, name it as a weaker theorem.
2. **Exact profile.** Attack every one of the thirty non-target windows in one
   isolator. Lemma L9.1 lists the complete Q/R/QR pencil cover; spacing 30 is
   used only after that local proof.
3. **Joint quantifier.** The order is
   `for every legal ordered Defender pair, choose an untouched j, then play
   the exact ordered Attacker pair (c_j,d_j)`. No statewise action existential
   is conjoined with a different strategy.
4. **Kill multiplicity.** `k` counts distinct `W_i`; two cells cannot kill
   more than two because the physical cell sets are disjoint. Hitting the same
   label twice earns no second refund.
5. **Launch coverage.** `U_j` is the literal union of all five common Q-axis
   windows. A cell outside `U_j` kills none of those five labels. The three
   unions are physically disjoint.
6. **Legality and phase.** `c_j` has an unchanged root anchor at distance
   exactly eight; `d_j` is made legal by `c_j` in the same turn. The response
   does not complete six, so equation (9)'s nonterminal updates and the next
   Defender epoch are defined.
7. **The `n_1/9` term.** It is zero on the first placement and exactly five on
   the second. Omitting the second sequential evaluation would erase the
   counterexample's entire source term.
8. **Claim boundary.** The final family has `tau=0`; this is not a GAP-RAW
   refutation. It is a refutation of the stronger canonical threshold and of J.
9. **D2.** A false antecedent does not make the proved implication erroneous.
   It makes that implication unusable as an assembly theorem.
10. **Alternative accounts.** Corollary R3.3 excludes only accounts that
    pointwise dominate `Theta_2` while staying below one. A formally specified
    non-dominating credit invariant remains open.

## 22. Authoritative round-3 status ledger

| Claim | Status | Exact basis / scope |
|---|---|---|
| GAP-RAW | **OPEN** | J was sufficient, not necessary; no Attacker win is constructed |
| Theorem R3.1, normative `8/9 -> >=11/9` counterexample | **PROVEN** | Exact construction and universal proof, §§15–16 |
| Obligation J with `B_2=Theta_2` | **REFUTED** | Every first reply at `P_*` fails J.4 |
| Canonical GAP-GLOBAL-RENEWAL | **REFUTED** | Same normative-root universal response |
| GAP-AMORTIZED-ABANDONMENT / non-dominating credit replacement | **OPEN** | No formal replacement account supplied |
| L9.1 one-pair isolator | **PROVEN** | Complete 31-window pencil audit, §15.1 |
| L9.2/L9.3 exact separation and launch geometry | **PROVEN** | Coordinate proof, §15 |
| L9.4 normative-root audit | **PROVEN** | Exact profile and strict potential, §15.2 |
| L9.5 legal nonterminal response | **PROVEN** | Exact radius-eight and adjacency supports, §16 |
| L9.6 `B_2<1 => tau<=2` | **PROVEN** | Imminent-label mass, §17.1 |
| R3.2, canonical J.3 redundancy | **PROVEN** | J.2 + J.4 + L9.6, §17.1 |
| L9.7 exact strict margin / two-clause J form | **PROVEN** | Four equation-(9) transitions, §17.2 |
| L9.8 fresh-pair ceiling `5/9` | **PROVEN** | Exact shared-window count, §17.3 |
| R3.3 no pointwise-dominating subunit account | **PROVEN** | Direct consequence of R3.1, §17.4 |
| General K3 suppression | **OPEN** | Unchanged standalone round-2 residue |
| Theorems A2/A2-prime and D2 | **PROVEN, retained** | No premise or proof changed |
| GAP-REPLACEMENT-INVARIANT | **OPEN** | Exact successor resume point, §19 |
| Round-3 coordinate/quotient regression | **VERIFIED** | 2,016 quotient pairs plus exact legal minimum row, §20 |

No round-2 PROVEN or VERIFIED result is downgraded. The two changed
round-2 OPEN rows are canonical strategy-reachable renewal and J; both are now
REFUTED by a root in the original normative domain.

## 23. Provenance and regeneration

**Input commit:** `283348dce09d42b67e364e0b2f2b63166b6b5f4d` on
branch `hunt/gap-raw`. No commit was created.

**Required corpus read first, in order, and in full:**

1. `GAP_RAW_PROOF_ROUND2.md`;
2. `GAP_RAW_REVIEW_ROUND2.md`;
3. `HUNT_REPORT_GAP_RAW.md`.

`HUNT_REPORT_ADAPTIVE.md` was then read for policy provenance. The bounded
strategy-stealing audit separately read the production engine sources listed
in `STRATEGY_STEALING_HEXO.md`. The mandatory RAM protocol was read from
`E:\tss-lean\RAM_PROTOCOL.md` before any contemplated build.

**Files authored/edited:**

- `GAP_RAW_PROOF_ROUND3.md`;
- `STRATEGY_STEALING_HEXO.md`; and
- `packages/hexfield_eq/rust/src/gap_raw_hunt.rs`, test-gated regression only.

Production rules and the strict verifier are untouched.

The single regression command used was:

```powershell
$env:CARGO_TARGET_DIR = '.target-gr'
cargo test -p hexfield_eq --lib --release `
  --target x86_64-pc-windows-msvc `
  'gap_raw_hunt::tests::round3_' -- `
  --ignored --nocapture --test-threads=1
```
