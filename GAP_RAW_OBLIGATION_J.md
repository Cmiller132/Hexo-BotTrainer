# GAP-RAW obligation J — exact first-epoch obstruction

**Input HEAD:** `88bca52d2a52dbcda5da60db81f00f69ad6cfcd7`  
**Branch/worktree:** `hunt/gap-raw`, `E:\Hexo-BotTrainer-hexgt\.claude\worktrees\hunt-gap-raw`  
**Date:** 2026-07-18  
**Landed artifact hash:** `[ORCHESTRATOR TO FILL]`  
**Method:** hand proof plus read-only source inspection; no Cargo command, build,
test, game-tree search, or generated enumeration was run. This is a new artifact;
no existing file was edited.

**Binding reading order.** The proof corpus was audited in this order:

1. `GAP_RAW_PROOF_ROUND1.md`;
2. `GAP_RAW_PROOF_ROUND2.md`, especially §§1–7;
3. `GAP_RAW_PROOF_ROUND3.md` and `GAP_RAW_REVIEW_ROUND3.md`;
4. `GAP_RAW_PROOF_ROUND4.md` and `GAP_RAW_REVIEW_ROUND4.md`;
5. `GAP_RAW_PROOF_ROUND5.md` through `GAP_RAW_PROOF_ROUND9.md`, each with its
   review and folded errata, ending with round 9 §86;
6. the production rule sources under `packages/hexo_engine/rust/src/`, in
   particular `coord.rs`, `legal.rs`, `rules.rs`, `state.rs`, `board.rs`, and
   `tactics.rs`.

The two concurrently authored round-10 files were not used as evidence.

## 0. The exact round-2 target, before attack

The following is the round-2 §6.2 statement, reproduced exactly.

> **Obligation J (strategy-reachable joint service, suppression, and renewal)
> [OPEN].** For every normative root `P₀` with `Φ(P₀)<1`, exhibit **one** pure
> Defender strategy S such that, against every legal Attacker continuation, at
> every reached Defender epoch h with position P:
>
> 1. **Renewal:** the canonical account satisfies `B₂(h)<1`.
> 2. **Actual service choice:** S's actual sequential two-cell reply is legal and
>    hits every member of `I(P)`. All discretion among alternative covers,
>    nonminimum servicing pairs, order, and fillers is resolved inside S's one
>    actual sequential pair; no separate existential action is conjoined.
> 3. **Unripeness handoff:** for the actual handoff Q produced by that same pair,
>    every legal ordered Attacker pair returns a Defender epoch with `τ≤2`.
> 4. **Account transition under the same choice:** Defender's two exact kill
>    transitions and every possible Attacker response are evaluated by (9), including
>    `n₁/9`, and establish `B₂(h′)<1` at the resulting next Defender epoch h′.

Round 2 immediately fixes the reachability domain as exactly `Hist(S,P₀)`.
That restricts continuations after a root; it does not restrict which normative
roots are quantified.

The following is round 2 §7's theorem and proof, also reproduced exactly.

> **Theorem D₂ (J alone suffices) [PROVEN].** If obligation J holds, then GAP-RAW
> holds.
>
> *Proof.* Fix an arbitrary normative root `P₀` with `Φ(P₀)<1` and take the one S
> supplied by J. Clause 2 says that, against every Attacker continuation, this
> same S's actual ordered pair services every reached Defender epoch. This is
> exactly `Service(S,P₀)`, so Theorem A₂ proves that S blocks forever. Since the
> root was arbitrary, GAP-RAW follows. ∎

Round 2 explicitly notes that clauses 1, 3, and 4 are logically unnecessary
once clause 2 is known on every reached epoch. They were included to bind one
proposed construction. Accordingly, refuting J.4 refutes J but does not refute
the exact service reformulation A₂ or GAP-RAW.

## 1. Verdict and binding-status correction

**Canonical obligation J is REFUTED.** More precisely, there is a normative
root with

`Φ = B₂ = Θ₂ = 2/3 < 1`, `I = ∅`, and `τ = 0`

such that every legal ordered two-cell Defender reply has a legal ordered
Attacker response returning a nonterminal Defender epoch with `B₂ ≥ 1`.
The six-label construction in §§3–6 below sharpens round 3's accepted
eight-label root `P_*`, whose corresponding lower bound is `B₂ ≥ 11/9`.

The obstruction is exact and occurs at the first epoch. Service is vacuous,
the actual handoff is unripe, and the selected response returns `τ=0`; only the
canonical renewal clause fails. Thus this is neither an Attacker win nor a
refutation of GAP-RAW.

This resolves a conflict in the campaign prompt. Round 2 originally labeled J
`[OPEN]`, but the later binding round-3 Theorem R3.1 proved it false, the
round-3 hostile review accepted that proof, and rounds 4–9 retained the
refutation. The round-9 §86 correction `k*=3` is binding in its tempo-plateau
program but does not touch the canonical `B₂=Θ₂` transition or the construction
below.

| Item | Verdict here | Exact reason |
|---|---|---|
| Obligation J as stated in round 2 §6.2 | **REFUTED** | Universal first-epoch response, §§3–6 |
| J.2 actual service at the obstruction root | **PROVEN for every reply** | `I=∅` |
| J.3 unripeness at the obstruction root | **PROVEN for every reply** | the handoff has only count-two graded labels |
| J.4 canonical renewal | **REFUTED after each Defender reply** | response map `ρ` gives `B₂=11/9,10/9,1` |
| Theorem D₂, `J⇒GAP-RAW` | **PROVEN, retained** | valid implication with a false antecedent |
| Classic GAP-RAW | **OPEN** | no Attacker six and no perpetual Defender strategy proved |

## 2. Normative and engine audit

### 2.1 Domain and account

A normative root is any finite, nonempty, Attacker-nonterminal blanket-game
position with Defender at `FirstStone` and `Φ<1`. It need not arise from the
engine's empty opening, have engine-history stone parity, or have a connected
occupied support. In the blanket game only an Attacker six is terminal; a
Defender stone is a blocker. These are the binding round-2 §§1.1–1.3
quantifiers.

For a nonterminal position `P`, an alive window is a six-cell straight window
with at least one Attacker stone and no Defender stone. The imminent family is

`I(P) = {W alive : count_P(W) ∈ {4,5}}`,

and `τ(P)` is the residual hitting number of `I(P)`. With `λ=√3`, the canonical
account is

`B₂(h)=Θ₂(P_h)=Σ_{W alive, count(W)≥2} λ^{-e(W)}`.

For state-level displays, `B₂(P)` abbreviates this same value `Θ₂(P)`; it is
not a second account.

Thus a count-two label has weight `λ^{-4}=1/9`. The exact sequential updates
from round 2 equation (9) are

`B₂(h+D@x)-B₂(h) = -κ₂(P_h,x)`,

`B₂(h+A@c)-B₂(h) = (λ-1)S₂(P_h,c)+n₁(P_h,c)/9`.

The pre-placement state matters in both formulas, especially for the second
Attacker stone.

### 2.2 Production-rule facts used below

The source audit gives the following hand-legality contract.

- Axial distance is
  `max(|Δq|,|Δr|,|Δq+Δr|)` (`coord.rs:76–82`).
- The three window axes are `(1,0)`, `(0,1)`, and `(1,-1)`
  (`tactics.rs:21–52`); a window has six cells (`tactics.rs:13–17`).
- A non-opening placement is legal exactly when its cell is empty and belongs
  to the color-blind union of closed radius-eight neighborhoods of occupied
  cells (`legal.rs:17–18,123–145`; `board.rs:167–171`). Radius eight is
  inclusive.
- The first stone updates that union before the second stone is checked
  (`state.rs:289–335`). Hence the second cell may use the first as its anchor.
- At a normal turn the same player makes `FirstStone`, then `SecondStone`, and
  control passes; a win is checked after each placement (`state.rs:1–10,
  317–335`). A second stone cannot reuse the first (`rules.rs:24–30`).
- Each placement updates exactly eighteen six-windows, six offsets on each of
  three axes (`tactics.rs:451–499`). A mixed-color window is inactive
  (`tactics.rs:171–208`).

The production engine awards a six to either owner, whereas the binding
normative game deliberately ignores Defender sixes. The explicit four-ply
response below in fact creates no six for either owner, so this semantic
difference is not used to hide a cadence or terminal event. The seed root is a
normative position, not a claim of replay reachability from `HexoState::new()`.
From the empty engine board the owner cadence is
`F ; S,S ; F,F ; S,S ; …`; the normative seed begins later at a declared
Defender `FirstStone`, so the local owner cadence audited below is
`D,D ; A,A ; D,…`.

## 3. Enumeration architecture first

No coverage claim below ranges over an unnamed collection. The complete index
architecture is fixed here before the construction.

### 3.1 Seed and launch indices

Let

`M_m={0,1,…,m-1}`, with `0≤m≤8`,

index `m` isolated old count-two labels, and let

`J₃={0,1,2}`

index three prospective fresh adjacent-pair launches. The sharp counterexample
will use `m=6`; the accepted round-3 root is the same family at `m=8`.

For an ordered legal Defender reply `a=(x₁,x₂)`, define its two relevant
incidence sets

`K(a)={i∈M_m : {x₁,x₂}∩W_i≠∅}`,

`L(a)={j∈J₃ : {x₁,x₂}∩U_j≠∅}`,

and put `k=|K(a)|`, `ℓ=|L(a)|`. Section 4 proves that every `W_i` and `U_j` is
physically disjoint from every other one. Consequently each Defender cell is
in at most one relevant region, and the complete finite effect quotient is

`𝒬={(k,ℓ): k,ℓ≥0 and k+ℓ≤2}`

`  ={(0,0),(0,1),(0,2),(1,0),(1,1),(2,0)}.`              (J-1)

This quotient includes replies with one or both cells outside every relevant
region and replies whose two cells hit the same region. Placement order may
affect legality, but legality is already part of the quantified reply; order
does not change these kill/contact effects.

### 3.2 Complete response map

Since `ℓ≤2`, `J₃∖L(a)` is nonempty. Define

`j(a)=min(J₃∖L(a))`,

and define the Attacker response map on **every** legal ordered Defender reply
by

`ρ(a)=(c_{j(a)},d_{j(a)}).`                               (J-2)

Thus the universal Defender-reply quantifier is not discharged by a witness
line. It is discharged by the six cases (J-1), with a deterministic response
in every case. The account value depends only on `k`; the three exhaustive
values `k=0,1,2` will be displayed in §6.3.

### 3.3 Service/suppression/renewal trichotomy

At the initial epoch every case in (J-1) is classified through the same three
questions:

1. **service:** did the actual Defender pair hit all of `I(P_m)`?
2. **suppression:** is its actual Attacker handoff unripe?
3. **renewal:** does every Attacker response, in particular (J-2), return
   `B₂<1`?

Sections 5–6 prove, for every case, `yes`, `yes`, `no`. No fourth case such as
completion, phase truncation, an illegal response, or an unclassified remote
reply remains.

## 4. Explicit seed family

### 4.1 One isolated count-two label

Let

`G={(0,1),(1,-1),(-1,0),(3,-2),(0,-4),(-3,3),(1,3),(6,0)}`.

For every `i∈M_m`, put

`b_i=(30i,0)`,

`A_i={b_i,b_i+(1,0)}`,

`D_i=b_i+G`,

`W_i={b_i+(t,0):0≤t≤5}`.                                (J-3)

**Lemma J.1 (isolator) [PROVEN].** For the untranslated gadget with Attacker
stones `(0,0),(1,0)` and Defender set `G`, the only alive window is `W_0`, and
it has count two.

*Proof.* The two adjacent stones touch `18+18-5=31` distinct windows. On the
Q-axis, the non-target starts `-5,-4,-3,-2,-1` contain `(-1,0)`, while start
`1` contains `(6,0)`. On the R-pencil through `(0,0)`, starts `-5,-4` contain
`(0,-4)` and starts `-4,-3,-2,-1,0` contain `(0,1)`. On the R-pencil through
`(1,0)`, starts `-5,…,-1` contain `(1,-1)` and starts `-2,-1,0` contain
`(1,3)`.

Parameterize the QR-pencil through `(0,0)` by `(t,-t)`. Its starts `-5,…,0`
are covered by `(-3,3)` for starts `-5,-4,-3` and by `(1,-1)` for starts
`-4,…,0`. Parameterize the QR-pencil through `(1,0)` by `(1+t,-t)`. Its starts
are covered by `(0,1)` for `-5,…,-1` and by `(3,-2)` for `-3,…,0`. These
ranges cover every non-target window, overlaps being harmless. No point of `G`
lies in `W_0={(0,0),…,(5,0)}`. Therefore exactly `W_0` survives. ∎

**Lemma J.2 (translated old profile) [PROVEN].** With

`A_old=⋃_{i∈M_m}A_i`, `D_old=⋃_{i∈M_m}D_i`,

the exact alive profile is `(n₁,n₂,n₃,n₄,n₅,n₆)=(0,m,0,0,0,0)`, consisting
of the pairwise-disjoint labels `W_i`.

*Proof.* Translation preserves Lemma J.1. Attacker stones in consecutive
gadgets are at distance at least `29`, while a six-window has diameter `5`, so
no window contains Attacker stones from two gadgets. Other translated blockers
can only delete labels; they do not meet a target `W_i`. Each target occupies
Q-coordinates `[30i,30i+5]`, so consecutive targets have 24 intervening cells
and are disjoint. ∎

The “24 intervening cells” wording incorporates the binding round-3 review's
off-by-one correction.

### 4.2 Three fresh adjacent-pair launch sites

For `j∈J₃`, put

`R_j=100+30j`,

`c_j=(0,R_j)`, `d_j=(1,R_j)`, `a_j=(0,R_j+8)`,

`U_j={(q,R_j):-4≤q≤5}`.                                  (J-4)

The five Q-windows containing both adjacent cells `c_j,d_j` have starts
`q=-4,-3,-2,-1,0`; their literal union is `U_j`.

**Lemma J.3 (separation and anchors) [PROVEN].** For `m≤8`, the sets
`{W_i:i∈M_m}∪{U_0,U_1,U_2}` are pairwise disjoint. Every `U_j` is disjoint from
all root stones defined below. Moreover

`d_hex(c_j,a_j)=8`, `d_hex(d_j,c_j)=1`.                    (J-5)

*Proof.* The old targets lie on row `r=0`, whereas launch unions lie on rows
`100,130,160`. Distinct launch rows are 30 apart. All old Attacker and blocker
R-coordinates lie in `[-4,3]`; the anchors lie on rows `108,138,168`; and an
anchor is not on its launch row. This proves the required disjointness from
root stones. Formula (J-5) follows directly from
`max(|Δq|,|Δr|,|Δq+Δr|)`. ∎

### 4.3 The normative roots

Define

`A_m=A_old`,

`D_m=D_old∪{a_0,a_1,a_2}`,

`P_m=(A_m,D_m,Defender,FirstStone)`.                        (J-6)

**Lemma J.4 (root audit) [PROVEN].** For `0≤m≤8`, `P_m` is finite, nonempty,
disjoint, and Attacker-nonterminal, with exact alive profile
`(0,m,0,0,0,0)`. Therefore

`Φ(P_m)=Θ₂(P_m)=B₂(P_m)=m/9<1` for `m≤8`,

`I(P_m)=∅`, `τ(P_m)=0`.                                    (J-7)

*Proof.* Lemmas J.2–J.3 give the exact profile and disjointness. Each of its
`m` labels has two Attacker stones, four empties, and weight `λ^{-4}=1/9`.
There is no other alive label, so the three displayed sums agree. No label is
imminent or complete. Strictness holds because `m≤8`. ∎

The accepted round-3 root is `P_*=P_8`. The sharper root used here is `P_6`,
with `Φ=B₂=2/3`.

### 4.4 Root terminal audit for the production cadence

Within one untranslated `G`, every Q-, R-, or QR-axis line contains at most two
points of `G`: the only repeated Q rows are `r=0,3`, the only repeated R
columns are `q=0,1`, and the only repeated QR sums are `q+r=0,1`. Translations
are separated by at least 21 cells even across the full Q-span of `G`; anchors
are separated by 30 and lie far from every `D_i`. Hence every six-window has at
most two root Defender stones. Adding the two stones of an arbitrary Defender
reply can raise that to at most four. The root and both reply placements
therefore create no Defender six. Lemma J.4 already excludes an Attacker six.

This audit is stronger than the blanket semantics require, but guarantees that
the production engine's after-each-placement terminal rule would not truncate
the four-placement cadence used next.

### 4.5 Hostile self-review of the seed stage

| Attempted refutation | Outcome |
|---|---|
| “The blockers merely suggest, but do not enumerate, the unwanted pencils.” | **Closed.** Lemma J.1 partitions all 31 windows through the two Attacker stones into the Q, two R, and two QR pencils and gives a blocker range for each non-target label. |
| “A translated, edge-disconnected pair creates a cross-gadget alive label.” | **Closed.** Consecutive Attacker supports are distance 29; a six-window's diameter is 5. |
| “An anchor lies in a target or launch window.” | **Closed.** Targets are on `r=0`, launch unions on `r=100,130,160`, and their anchors eight rows beyond them. |
| “The separation count repeats round 3's off-by-one.” | **Closed.** Consecutive target intervals end at `30i+5` and start at `30i+30`, leaving exactly 24 intervening cells. |
| “The arbitrary normative seed is being called engine-history reachable.” | **Rejected as a scope change.** Round-2 J quantifies every normative root. This artifact states explicitly that `P_m` is not supplied by `HexoState::new()`. |
| “The engine would stop on a Defender six before the response.” | **Closed.** Root Defender occupancy is at most two per six-window; the reply adds at most two. |
| “Coordinates overflow the implementation.” | **Closed for the finite seed.** All displayed construction coordinates lie between `q=-4…216`, `r=-4…168`, far inside `i16`. An arbitrary legal ordered reply remains within two successive closed radius-eight extensions of this finite envelope. |

No seed claim was downgraded by this review.

## 5. Service and suppression for every actual reply

Fix `m≤8`, an arbitrary pure Defender strategy `S` at root `P_m`, and its
actual legal ordered reply

`a=(x₁,x₂)`.

Let `h₀` be the zero-length history at `P_m`, and let `Q_a` be the Attacker
handoff after these two Defender placements.

### 5.1 Actual service is case-complete

By (J-7), `I(P_m)=∅`. Therefore every one of the six reply classes (J-1), and
indeed every legal reply before quotienting, hits every member of `I(P_m)`.
The service condition is vacuous but genuine: S's own sequential pair is the
pair being evaluated. No cover supplied by a different existential witness is
substituted.

Defender placements cannot create or promote an Attacker label. A target
`W_i` survives exactly when `i∉K(a)`. Hence `Q_a` has exact alive profile

`(n₁,n₂,n₃,n₄,n₅,n₆)=(0,m-k,0,0,0,0)`                  (J-8)

and

`B₂(Q_a)=(m-k)/9`.                                         (J-9)

### 5.2 The same handoff is unripe

**Lemma J.5 (uniform suppression) [PROVEN].** Every `Q_a` in (J-8) is unripe.
In fact every legal ordered Attacker pair from `Q_a` returns `τ≤1`.

*Proof.* Any window becoming imminent after two Attacker placements must have
had count at least two at `Q_a`. The only such alive labels are the surviving
pairwise-disjoint `W_i`. Since each begins at count two, it can become imminent
only by containing both new trigger cells. The same two distinct cells cannot
both lie in two physically disjoint `W_i`, so at most one old label becomes
imminent. A previously count-zero or count-one label reaches count at most two
or three and is not imminent. Thus the returned imminent family has at most
one member and hitting number at most one. ∎

This proves J.3 for the actual handoff of **every** reply class, more strongly
than needed. The obstruction below therefore cannot be blamed on unresolved
K3 suppression geometry.

### 5.3 Hostile self-review of service and suppression

| Attempted refutation | Outcome |
|---|---|
| “Vacuous root service was replaced by a different cover.” | **Closed.** The quantified actual pair itself services the empty family; no second witness appears. |
| “Defender placements create a hidden Attacker count-one/count-three label.” | **Refuted by monotonicity.** They only delete Attacker-alive labels, so (J-8) is exact. |
| “One Attacker pair can promote two remote old labels to count four.” | **Closed.** Promotion from two to four requires both triggers in each label; disjoint `W_i` cannot both contain the same two cells. |
| “A virgin or count-one label becomes imminent.” | **Closed.** Two placements raise those starting counts only to two or three. |

This stage proves only the first handoff's universal unripeness. It does not
claim perpetual service; §6 now tests renewal under every actual choice.

## 6. The universal legal response and exact renewal failure

### 6.1 Legality and cadence

Take `j=j(a)` from (J-2). Because neither Defender reply cell belongs to
`U_j`, and `U_j` was root-stone-free, both `c_j` and `d_j` are empty after the
reply. The root anchor `a_j` remains occupied, and (J-5) gives

`d_hex(c_j,a_j)=8`.

Thus `A@c_j` is legal by the inclusive, color-blind radius-eight rule. The
board is updated before the second legality test, and

`d_hex(d_j,c_j)=1`,

so `A@d_j` is then legal. The exact normal-turn phase ledger is

| Placement | Owner/phase before | Owner/phase after if nonterminal | Audit |
|---|---|---|---|
| `D@x₁` | Defender `FirstStone` | Defender `SecondStone` | legal by the quantified strategy action |
| `D@x₂` | Defender `SecondStone` | Attacker `FirstStone` | legal and distinct by the quantified strategy action |
| `A@c_j` | Attacker `FirstStone` | Attacker `SecondStone` | anchor `a_j` at distance exactly 8 |
| `A@d_j` | Attacker `SecondStone` | Defender `FirstStone` | new anchor `c_j` at distance 1 |

The old Attacker components contain at most two stones each. The new launch
component contains one stone after `c_j` and two after `d_j`, and is more than
five cells from every old component. Hence neither Attacker placement completes
six. Section 4.4 excluded a Defender six during the reply. All four placements
occur, and the last one returns a nonterminal Defender epoch `P'_a`.

### 6.2 All four exact account transitions

The two Defender placements kill exactly the `k` distinct target labels in
`K(a)`. Sequential accounting counts a label met twice only once, so

`B₂(P_m+D@x₁+D@x₂)=m/9-k/9=(m-k)/9`.                     (J-10)

Call this handoff `Q_a`. It has no alive count-one labels, and the launch is
far from every surviving old count-two label. Therefore immediately before the
first Attacker placement,

`S₂(Q_a,c_j)=0`, `n₁(Q_a,c_j)=0`,

`ΔB₂(A@c_j)=0`.                                            (J-11)

After `A@c_j`, exactly the five Defender-free Q-windows containing both
`c_j,d_j` are alive count-one labels through `d_j`. Their union is `U_j`, which
the reply did not touch. Two Q-collinear distinct cells share no R- or QR-axis
window, and no old label reaches this launch. Consequently, at the second
Attacker placement,

`S₂(Q_a+A@c_j,d_j)=0`, `n₁(Q_a+A@c_j,d_j)=5`,

`ΔB₂(A@d_j)=5/9`.                                          (J-12)

Combining (J-10)–(J-12) gives the exact next-epoch value

`B₂(P'_a)=(m-k+5)/9`.                                      (J-13)

This is the load-bearing sequential `n₁/9` charge. Treating the Attacker pair
atomically or evaluating `n₁` only before its first stone would erase the
source and be incorrect.

### 6.3 Exhaustive cases at the sharp root

For `P_6`, the entire quotient (J-1) is exhausted as follows.

| Cases `(k,ℓ)` | Untouched launch exists? | Service | Handoff | Exact next `B₂` | Renewal `<1`? |
|---|---:|---:|---:|---:|---:|
| `(0,0),(0,1),(0,2)` | yes | yes | unripe | `11/9` | **no** |
| `(1,0),(1,1)` | yes | yes | unripe | `10/9` | **no** |
| `(2,0)` | yes | yes | unripe | `9/9=1` | **no** |

The response epoch has no label above count two: it contains `m-k` surviving
old count-two labels, five fresh focal count-two labels, and possibly fresh
count-one labels. Therefore

`I(P'_a)=∅`, `τ(P'_a)=0`.                                  (J-14)

The strict equality case is intentional. J.4 requires `B₂<1`, not `B₂≤1`.

### 6.4 Sharp parameter boundary of this architecture

**Theorem J.6 (six-label canonical-renewal obstruction) [PROVEN].** For every
`m∈{6,7,8}` and every pure Defender strategy from `P_m`, the response map `ρ`
reaches a nonterminal next Defender epoch with

`B₂(P'_a)≥(m-2+5)/9=(m+3)/9≥1`.                           (J-15)

Hence no such strategy satisfies obligation J. In particular `P_6` refutes J
from the lower root value `Φ=2/3`; `P_8` recovers round 3's accepted bound
`B₂≥11/9`.

*Proof.* Every strategy has one actual legal pair `a`. Disjointness gives
`k≤2` and `ℓ≤2`; hence (J-2) is defined. Sections 6.1–6.2 prove its response is
legal, nonterminal, and has exact value (J-13). Inequality (J-15) follows.
Clause J.4 universally quantifies all Attacker responses after this same pair,
so the single response `ρ(a)` refutes it. ∎

The threshold `m=6` is exact for the universal lower bound (J-15) under the
response map `ρ`. When `m=5`, the legal reply that hits two distinct old targets
can be taken explicitly as
`D@(b_0+(2,0)), D@(b_1+(2,0))`, i.e. `D@(2,0),D@(32,0)`.
Both cells are empty and adjacent to an old Attacker support. It has `k=2`, and
the guaranteed response `ρ` gives only `8/9`; this particular lower bound no
longer forces renewal failure. Another Attacker response could still do so.
This observation does **not** prove J for `m≤5`; it delimits exactly where the
proved response map stops.

### 6.5 Hostile self-review of the response stage

| Attempted refutation | Outcome |
|---|---|
| “Only a sample reply line is answered.” | **Closed.** The response map is defined for every legal ordered pair through the complete six-element quotient (J-1). |
| “One Defender cell could kill several old targets or touch all launches.” | **Closed.** All `W_i,U_j` are physically disjoint; each cell has at most one relevant incidence. Two cells leave one of three `U_j` untouched. |
| “A reply outside the finite regions is missing.” | **Closed.** It contributes no old-target kill and no launch contact and is represented by the appropriate `k,ℓ` case, including `(0,0)`. |
| “The first launch cell is at radius eight but the rule is strict.” | **Closed.** Production `LEGAL_RADIUS=8` uses a closed radius iterator; equality is legal. |
| “Both Attacker cells were required to be legal before the turn.” | **Closed.** Legality is sequential; `c_j` updates the store and anchors adjacent `d_j`. |
| “The proof skips the mandatory second stone or a first-stone terminal.” | **Closed.** No component has more than two Attacker stones; both placements occur and the cadence table returns Defender `FirstStone`. |
| “The five-window birth charge is an inequality, not an equality.” | **Closed.** The untouched literal union `U_j` contains all five common Q-windows; no other axis contains both adjacent Q cells; `S₂=0,n₁=5` at the exact second pre-state. |
| “The endpoint at `B₂=1` is permitted.” | **Refuted.** J uses the strict inequality `<1`; equality is a failure. |
| “The account crossing is an unblockable pileup.” | **Refuted.** The exact endpoint has only count-one/count-two labels, so `I=∅,τ=0`. |

The response-stage verdict remains a refutation of J.4 only; no game-loss
claim is promoted.

## 7. Strategy reachability, stated as an invariant

Write `Reply_S(h₀)` for S's actual legal ordered two-cell reply at the root.
The relevant counter-invariant is

`∀S: h₀∈Hist(S,P_6), a=Reply_S(h₀) ⇒ h'_a=h₀·a·ρ(a)∈Hist(S,P_6)`

`and B₂(h'_a)≥1 while τ(P_{h'_a})=0`.                      (J-16)

The base is automatic: the empty history at the quantified root belongs to
`Hist(S,P_6)` for every S. The actual pair `a` is S-consistent by definition.
The response `ρ(a)` is a legal Attacker continuation by §6.1, and its endpoint
is nonterminal, so the extended history remains in `Hist(S,P_6)`. Equation
(J-13) maintains the numeric conclusion in every finite reply class.

There is no hidden appeal to the non-reachability of an arbitrary later state:
the obstruction root is the quantified `P₀` itself. Historical reachability
from the engine's forced empty-board opening would define a strictly weaker
root theorem than round-2 J and GAP-RAW. At ordinary engine-reachable complete
turn boundaries the stone counts also obey opening parity, whereas `P_6` does
not; that distinction cannot be imported into the stated normative domain.

### 7.1 Hostile self-review of reachability

| Attempted refutation | Outcome |
|---|---|
| “Strategy reachability can exclude the bad state.” | **Refuted at the base.** `P_6` is itself one of J's universally quantified roots, and the empty history is S-reachable for every S. |
| “The response is chosen before seeing S's actual cover.” | **Refuted.** The quantifier order is `S`, then its actual ordered reply `a`, then `j(a)` and `ρ(a)`. |
| “A different statewise good pair could replace S's pair.” | **Irrelevant.** Every legal actual pair is in the quotient and fails J.4. |
| “The endpoint drops out of `Hist` because it is terminal.” | **Closed.** Section 6.1 proves both Attacker placements legal and nonwinning; the endpoint is a Defender epoch. |

The reachability objection is closed at J's exact normative scope. It would
reopen only after explicitly changing the root theorem to engine-history-
reachable roots.

## 8. What transfers from rounds 3–9

### 8.1 Binding status chain

Round 3 Theorem R3.1 proves the `P_8` version of (J-15), and its hostile review
accepts the root profile, launch pigeonhole, exact equations, legality, and
normative-domain scope. The review's only geometric erratum was the corrected
“24 intervening cells” wording already used here. Round 4 explicitly retains
canonical J as refuted.

Rounds 4–9 pursue a different, non-dominating tempo program. At an unripe
Attacker handoff `Q`, round 4 defines `TEMPO(Q)` to equal the maximum next
`τ` over all legal Attacker pairs. At a Defender epoch `P`, `M(P)` minimizes
that value over the **same actual ordered servicing pairs**. Thus
`TEMPO≤2` and `M≤2` can certify suppression and same-pair service—the content
of J.2–J.3—but they do not imply either canonical threshold clause J.1 or J.4.
They deliberately allow safe states with `Θ₂≥1`, including the endpoints of
the present obstruction.

Round 5's review states that no canonical-J route is revived. Round 6's
authoritative ledger and review retain canonical J as **REFUTED by R3.1**.
No later round changes that status.

### 8.2 Usable service/suppression modules, not a proof of J

The later program supplies exact modules that remain relevant to a replacement
for J:

- low-only handoffs (all alive counts at most two) are suppressible;
- at a finite nonterminal Defender epoch with `τ=0`, if `τ₃≤2`, where `τ₃`
  denotes the residual hitting number of the alive count-three family, its
  displayed one-/two-cell transversal has a legal same-pair suppression map;
- several sealed-pencil and capped plateau classes have exact one-cycle or
  finite-depth service maps; and
- a conditional policy that always has an actual servicing action whose every
  response retains repairability would assemble perpetual service.

None controls `B₂=Θ₂`. The present root is already a low-only class: §5 proves
that every first reply services and suppresses it, while §6 proves that every
first reply loses canonical renewal. This is an exact separation of those
modules.

### 8.3 Binding round-9 §86 correction

The authored plateau transition `k*=4` is refuted. The binding correction is

`k*=3`:

the cap remains exact-risk two through `P_2^pl`, but every action is unsafe at
`P_3^pl`. In §86.1 the legal response `((9,-2),(10,-2))` creates the
three-carrier miss returning `M≥3`; the full `H₃/G₋/G₊` argument covers every
action. Separately, §86.3 adds the missed boundary incidence `(5,-4)` to the
`P_2^pl` quotient and preserves that state's exact-risk-two result. These are
reached-state Q1/Q3 tempo results. No strategy-independent strict-root route to
`P_3^pl` is proved.

The correction intersects a future replacement strategy because such a
strategy must avoid or pre-empt `P_3^pl`. It does **not** intersect (J-3)–(J-15):
those formulas contain no plateau stock, `M`, cap, `H₃`, or disputed response
class, and they use the independently accepted canonical transition (9).

### 8.4 Hostile self-review of corpus transfer

| Attempted refutation | Outcome |
|---|---|
| “Later `M≤2` constructions silently prove J.” | **Refuted.** They certify service/suppression only and do not renew `B₂<1`. |
| “The `k*=3` erratum invalidates the six-label root.” | **Refuted.** It is local to named plateau states and changes no canonical account equation or coordinate incidence here. |
| “Round-3 J refutation was superseded rather than retained.” | **Refuted by the later ledgers.** Rounds 4–6 explicitly retain it, and rounds 7–9 never revive canonical renewal. |
| “A fixed tempo-policy loss is a GAP-RAW refutation.” | **Refuted.** Later Q1 losses concern named policies/reached states; no all-strategy strict-root forcing theorem follows. |

No tempo-ladder theorem is imported beyond its verified service/suppression or
scope-warning content.

## 9. Consequence for D₂ and classic GAP-RAW

Theorem D₂ remains a correct implication:

`J ⇒ GAP-RAW`.

But Theorem J.6 supplies `¬J` at the stated normative domain. An implication
with a false antecedent yields no conclusion about its consequent. Therefore
D₂ does **not** close classic GAP-RAW, and this artifact does not claim
`¬GAP-RAW`.

Indeed the exact blocking endpoint has `I=∅` and `τ=0`. No six is completed,
and the Defender owes no immediate service. The event is fatal only to the
chosen proof invariant `B₂<1`. The exact surviving classic formulation is
round-2 Theorem A₂:

> for every normative `Φ<1` root, construct one pure Defender strategy whose
> own actual ordered reply services every epoch reached under that same
> strategy against every legal Attacker continuation.

Any replacement account must therefore allow the safe transition

`P_6 : Θ₂=2/3  →  P'_a : Θ₂≥1, τ=0`,

so it cannot be a strict subunit account pointwise dominating `Θ₂`. It must
instead encode interaction/deadline structure, history credits, or direct
service availability.

## 10. Authoritative verdict and exact next question

**J verdict:** **REFUTED**, a result strictly sharper than “OPEN with a
partial.” The exact blocking event is the first Attacker response from `P_6`:
after any actual Defender pair, select an untouched one of three launch unions
and play its radius-eight-anchored adjacent pair; the returned value is
`B₂=(11-k)/9∈{11/9,10/9,1}` while `τ=0`.

**Key invariant in one line:** every strategy reaches its own first reply from
`P_6`, and the exhaustive response map preserves nonterminal reachability while
forcing `B₂≥1` without creating service debt.

**What D₂ yields:** nothing further, because its sufficient hypothesis is
false. Classic GAP-RAW remains **OPEN**.

**Sharpest next question.** For every normative `Φ<1` root, can one choose a
root-dependent total pure Defender strategy `S`, together with an
S/history-indexed replacement invariant `V`, such that S's same actual ordered
pair services `I(P)` and returns to `V` after **every** legal Attacker response,
while `V` admits the safe high-`Θ₂` transition (J-16) and excludes the exact
later hub and `P_3^pl` stop states? The first unresolved initialization slice is
`τ=0` with count-three residual hitting number `τ₃≥3`; low-only and `τ₃≤2`
slices are already closed. The renewal step is the same-action, all-response
Bellman closure. Proving both would return to Theorem A₂ and close classic
GAP-RAW without resurrecting false canonical J.
