# GAP-RAW Proof Round 1 — Raw ES Global Claim for Hexo

**Author:** Fable proof lane, GAP-RAW round 1 (worktree `hunt-gap-raw`, branch `hunt/gap-raw`, HEAD 9b32db63)
**Date:** 2026-07-16
**Status:** ROUND-1 COMPLETE — honest partial. GAP-RAW remains open; this round reduces it to two sharply-stated obligations (W3′ + O1′, §3.10) via an exact reformulation (Theorem A), a proven budget kernel for the fully-forced case (K1), and a graded account (Θ₂) proven immune to the Cor-2 obstruction. The status table (§5) is the source of truth for what is claimed at what strength.

---

## 0. Target claim

**Claim (GAP-RAW / raw ES global).** In Hexo (connect-6 variant on the hex lattice, 2 placements per turn, legality = within distance 8 of an existing stone), from any Defender-FirstStone position whose attacker potential satisfies Φ < 1 — where Φ = Σ λ^(−e(W)) over attacker-touched alive windows W, λ = √3, e(W) = number of empty cells of W — the defender has a strategy that holds forever (the attacker never completes a 6-window).

**Known status going in:** OPEN. Fixed greedy defender strategies are REFUTED (both dynamic-greedy and fixed-cohort variants lose on specific lines). The claim itself has never been refuted: two hunts plus sound minimax found zero attacker wins from Φ < 1 roots. The proof reduces to two named gaps (statements in `docs/proof_parts/ES_GLOBAL_BOUNDARY.md` of the main-review worktree):

- **GAP-GLOBAL-RENEWAL** — re-entering the Φ < 1 epoch after Φ exits below-1 territory.
- **GAP-AMORTIZED-ABANDONMENT** — bounding the co-maturation of abandoned births (windows the defender stopped servicing).

## 1. Candidate invariant and strategy

- **(I1)** Every alive attacker-touched window keeps ≥ 3 empties — maintained by blocking the most-imminent cluster whenever any window is ≤ 2 empties from completion.
- **(I2)** Between forced blocks, the defender plays global touched-danger minimization.
- **Anchor strategy R1b** (strategy name, not the zone-radius row): completion-first touched-window greedy with imminence threshold τ = 2. Empirical: holds 98.6–99.9% of randomized episodes; residual breaks require ≥ 3 simultaneous count-4 windows arising in one attacker turn.
- **Greedy dilemma constraint:** dynamic greedy loses the ES cohort line; fixed-cohort greedy loses fresh-birth lines. Any proof-carrying strategy must be adaptive (respond to the current window population, not a frozen cohort).

## 2. Lemma skeleton

| # | Section | Content | Final status |
|---|---------|---------|--------------|
| L1 | §3.1 | Ledger formalization; completion + epoch service criteria; **Theorem A/A′: GAP-RAW ⇔ perpetual unripeness** | PROVEN |
| L2 | §3.2 | Fork floor: τ≥3 pileup needs ≥7 attacker stones inside its footprint | VERIFIED (re-run PASS) |
| L3 | §3.3 | Per-n ceilings, n = 4..12, four columns | VERIFIED (re-run PASS) |
| L4 | §3.4 | One-turn maturation from Φ<1 roots (0 / ≤3 straight-4 / R1b reset) | VERIFIED per-root (re-run PASS) |
| L5 | §3.5 | ΔΦ per turn: per-root exact (2.309/2.591) + general pencil bound | VERIFIED per-root + PROVEN |
| L6 | §3.6 | Service capacity, legality, kill-multiplicity | PROVEN |
| L7 | §3.7 | Ripeness structure: membership, decomposition, one-stone defusal, mass floors (ripe ⇒ mass ≥ 1/3, ≥5 local pre-stones); **Theorem C horizon extension** | PROVEN (floors; C) / CONJECTURED (sharp floor R7.4) |
| L8 | §3.8 | Θ₂ graded account: Cor-2 neutralized, injection bounds, priced reactivation; obligation **O2′** | PROVEN (account facts) / OPEN (O2′) |
| L9 | §3.9 | Budget kernel **Theorem B**: K1 proven, K2/K3 partial; obligations **W3′**, **O1′** | PROVEN (K1) / PARTIAL (K2,K3) / OPEN (W3′, O1′) |
| L10 | §3.10 | **Theorem D**: GAP-RAW modulo W3′ + O1′; residue table | PROVEN, conditional |

## 3. Proof body

*(Sections filled in below as they are completed — see status table.)*

### 3.0 Definitions

- **D0.1 (board, windows).** Infinite hex lattice, axial coordinates. Three window axes `Q, R, QR`; a *window* is a segment of 6 consecutive cells along one axis. Every cell lies in exactly 6 windows per axis, 18 total.
- **D0.2 (positions).** A position is a pair of disjoint stone sets (A, D). A window is *alive* if it contains no defender stone; *touched* if it contains ≥1 attacker stone. For alive W: `count(W) = |W ∩ A|`, `e(W) = 6 − count(W)` (its empties). A *count-k window* is alive with count = k.
- **D0.3 (turns, legality).** Turns alternate, 2 placements per turn, applied sequentially. A placement is legal iff the cell is empty and within hex distance 8 of some existing stone. A *defender epoch* is a position with the defender to move (Defender-FirstStone parity). The attacker wins on completing a count-6 window.
- **D0.4 (potential).** `λ = √3`; `Φ(P) = Σ λ^(−e(W))` over attacker-touched alive windows. Weights: count-1 `λ⁻⁵ ≈ 0.0642`, count-2 `λ⁻⁴ = 1/9`, count-3 `λ⁻³ = 1/(3√3) ≈ 0.1925`, count-4 `λ⁻² = 1/3`, count-5 `λ⁻¹ = 1/√3 ≈ 0.5774`.
- **D0.5 (imminent family, service, τ).** `I(P)` = the family of alive windows with `e ≤ 2` (count-4/5). A defender placement *services* (kills) every alive window through it. For a family F of alive windows, `τ(F)` = the minimum number of cells, chosen among the empties of F's windows, that hit every member (hitting number / vertex-cover number of the residual-empties system).
- **D0.6 (invariant I1).** A position satisfies **(I1)** iff every alive window has ≥ 3 empties, i.e. `I(P) = ∅`.

### 3.1 L1 — Ledger formalization and the exact reformulation [PROVEN]

**Lemma L1.1 (completion criterion).** With the attacker to move at position Q, the attacker can complete a window this turn **iff** some alive window has `e(W) ≤ 2`.

*Proof.* (⇐) Fill W's ≤2 empties with the turn's 2 placements. Legality: every empty of W is within distance 5 of an attacker stone of W (both lie in a 6-segment), hence within 8. (⇒) Two placements raise any window's count by ≤ 2, and dead windows stay dead; a window completed this turn had count ≥ 4 and no defender stone before the turn. ∎

**Lemma L1.2 (epoch service criterion).** At a defender epoch P, the defender survives the attacker's next turn **iff** `τ(I(P)) ≤ 2`; moreover if `τ(I(P)) ≥ 3` the attacker wins outright (with any continuation of best play), and if `τ(I(P)) ≤ 2` the defender can restore (I1).

*Proof.* If `τ(I(P)) ≤ 2`, a ≤2-cell hitting set exists among the empties of I(P)'s windows; those cells are legal placements (distance ≤ 5 from attacker stones in their windows). Placing there kills every member of I(P); every other alive window has ≥3 empties, so (I1) holds at attacker-turn start and Lemma L1.1 blocks completion. If `τ(I(P)) ≥ 3`, whatever 2 cells the defender picks, some W ∈ I(P) receives no defender stone; W is alive with e ≤ 2 at the attacker's turn, and L1.1(⇐) completes it. (Spare placements, when the hitting set has size < 2, are always available: legality is colour-blind — any empty cell within distance 8 of *any* stone qualifies — and the board is infinite.) ∎

A position `τ(I(P)) ≥ 3` is exactly the *unblockable pileup* of the hunt reports (min hitting set ≥ 3 over ≤2-empty alive windows).

**Theorem A (exact reformulation of GAP-RAW).** GAP-RAW holds **iff** from every Defender-FirstStone position P₀ with Φ(P₀) < 1 the defender has a strategy under which `τ(I(P)) ≤ 2` at every defender epoch P of the play.

*Proof.* (⇐) By L1.2 the defender services I(P) at every epoch and never faces a completion; the game lasts forever. (⇒) Suppose a strategy holds forever against every attacker line. If some attacker line reached an epoch with `τ(I(P)) ≥ 3`, then the attacker line that plays to that epoch and then cashes the pileup (L1.2) beats the strategy — contradiction. So no line reaches such an epoch. ∎

**Definition (ripeness).** Let Q be a position with the attacker to move satisfying (I1). Q is *ripe* iff there is a legal ordered placement pair (c₁, c₂) with `τ(I(Q + c₁ + c₂)) ≥ 3`.

**Theorem A′ (ripeness form).** GAP-RAW holds iff the defender can, forever, hand the attacker only unripe positions (servicing I at each epoch as in L1.2 and choosing among covers/spares so that the resulting attacker-turn-start position is unripe).

*Proof.* Immediate from Theorem A and the definitions: the epoch after an unripe position has `τ(I) ≤ 2`; a ripe position handed to the attacker yields an epoch with `τ(I) ≥ 3` under attacker best play, which loses by L1.2. ∎

**The ledger, in these terms.** Service debt at an epoch is `τ(I(P)) ∈ {0, 1, 2, loss}`. Debt does **not** carry across epochs — if ≤ 2 it is fully paid within the turn. Therefore the attacker's only winning route is to generate debt 3 within a single turn, i.e. to be handed a ripe position. The whole of GAP-RAW is an inventory-control problem on ripeness: forced covers consume the defender's 2 placements; ripeness-suppression competes for the same budget. The two named gaps are exactly the two ways this control could fail: the account that certifies "unripe" fails to renew (GAP-GLOBAL-RENEWAL), or dormant abandoned mass makes ripeness cheap late (GAP-AMORTIZED-ABANDONMENT).

### 3.2 L2 — Fork floor [VERIFIED]

**Lemma L2.** Any attacker stone configuration realising a family F of alive ≤2-empty windows with `τ(F) ≥ 3` contains **≥ 7 attacker stones inside ∪F**. Equivalently: no configuration of ≤ 6 attacker stones admits an unblockable one-turn pileup; 7 stones suffice (two perpendicular length-4 arms sharing a cell, 6 count-4 windows, 4 pairwise disjoint).

*Validation.* Exhaustive orderly-growth polyhex enumeration, deduplicated under D6 × translation, with three independent checks: (i) canonical polyhex counts equal OEIS A000228 for n = 1..12 (`1,1,3,7,22,82,333,1448,6572,30490,143552,683101`); (ii) the max-window function is superadditive, so a single connected cluster is optimal and the polyhex sweep covers the global optimum; (iii) a no-connectivity brute force over all n-subsets of the `[0,6]²` rhombus reproduces the maxima for n = 4,5,6. **Re-run at HEAD 9b32db63 on 2026-07-16: PASS (199.6 s), all three validations green** (regen command in §6).

*Locality remark (used by L7).* The floor applies to the sub-configuration of stones lying in ∪F: if the stones inside ∪F numbered ≤ 6, they would themselves be a ≤6-stone configuration realising τ ≥ 3, contradicting the enumeration. So the 7-stone cost is paid *inside the witness footprint*, not merely board-wide.

### 3.3 L3 — Per-n ceilings [VERIFIED]

**Lemma L3.** Absolute per-n ceilings over *all* n-stone attacker configurations (re-verified at HEAD, same run as L2):

| n | max count-4+ | max count-3+ | max count-5+ | max disjoint count-4 | unblockable fork? |
|---|--:|--:|--:|--:|:--|
| 4 | 3 | 5 | 0 | 2 | no |
| 5 | 4 | 8 | 2 | 2 | no |
| 6 | 5 | 12 | 3 | 2 | no |
| 7 | 6 | 16 | 4 | 4 | **yes (first)** |
| 8 | 7 | 24 | 5 | 4 | yes |
| 9 | 9 | 28 | 6 | 6 | yes |
| 10 | 10 | 33 | 7 | 6 | yes |
| 11 | 12 | 38 | 8 | 8 | yes |
| 12 | 18 | 41 | 9 | 12 | yes |

For n ≤ 8 a straight line is count-4+-optimal (max = n−1, all collinear, only 2 disjoint); denser 2-D shapes win from n = 9. The count-3+ column bounds the stock of windows sitting *at* the (I1) boundary — the raw material for one-turn ripeness.

### 3.4 L4 — One-turn maturation from Φ<1 roots [VERIFIED, per-root]

**Lemma L4.** Exhaustive over every attacker 2-placement turn from each root of the hunt battery (all Defender-FirstStone, Φ < 1):

1. From every **1-attacker-stone** root (`es_core`, `blocker_2_0`, `blocker_3_0`): **zero** count-4+ windows creatable in one turn (3 stones cannot fill 4 cells of a window — this sub-claim is a trivial pencil fact, PROVEN, and holds for *all* 1-stone positions).
2. From the **densest 2-stone** Φ<1 roots (`dense_01_10`, `dense_01_1m1`, Φ = 0.9405): at most **3** count-4 windows in one raw turn, and they form a straight-4 — **≤ 2 pairwise disjoint**, hence τ ≤ 2, serviceable (L1.2).
3. After a single R1b defender turn, the one-turn count-4+ ceiling drops to **0 on every root**.

*Scope caution (statement fidelity).* Items 2–3 are exhaustive **per root**, not over all Φ<1 positions. Item 1's zero is a pencil fact for any position with ≤ 1 attacker stone. The general-position analogue of items 2–3 is exactly the ripeness question treated in §3.7–3.9.

### 3.5 L5 — ΔΦ per attacker turn [VERIFIED per-root + PROVEN general bound]

**Lemma L5a (VERIFIED, per-root, exhaustive).** Max one-turn ΔΦ: `4/√3 ≈ 2.309` from every 1-stone root (two isolated clean-escape births, 18 fresh count-1 windows per placement, `18·λ⁻⁵ = 2/√3` each); `2.591` from the dense 2-stone roots. So `4/√3` is **not** a universal per-turn ceiling — it is the 1-stone-root value; denser roots exceed it via promotions.

**Lemma L5b (PROVEN, all positions).** For any position with potential Φ, one attacker placement gains at most `(λ−1)·S + 2/√3`, where S ≤ Φ is the touched alive mass through the placed cell (each of ≤18 windows through the cell either promotes, ×λ, or enters at λ⁻⁵). Hence over a 2-placement turn
`Φ_after ≤ λ²·Φ + (λ+1)·(2/√3)`, i.e. `ΔΦ ≤ 2Φ + 2(1+√3)/√3 ≈ 2Φ + 3.155`.

*Proof of L5b.* A placement at cell c multiplies the weight of each alive touched window through c by λ (e decreases by 1) and adds ≤ 18 fresh count-1 windows at λ⁻⁵ each, `18λ⁻⁵ = 2/√3`. So Φ′ ≤ λΦ + 2/√3; iterate twice. ∎

**Consequence (as in the birth-ledger report).** Φ is *not* the servicing currency: one turn can add > 2 to Φ while adding zero serviceable debt. Any renewing account must grade by count, not by touch (§3.8).

### 3.6 L6 — Service capacity and legality [PROVEN]

**Lemma L6.** At a defender epoch P with `τ(I(P)) ≤ 2`: (i) a hitting set of size ≤ 2 exists among empties of I(P)-windows, and each such cell is a legal placement; (ii) placing there kills every member of I(P) and restores (I1); (iii) any placements left over ("spares") may be placed on any legal cell, in particular on any empty cell of any alive window (distance ≤ 5 from a stone) — spares are the defender's entire discretionary budget. Furthermore a defender stone at cell c kills **every** alive window through c simultaneously (up to 18) — service is set-valued, not per-window.

*Proof.* Contained in L1.2's proof; the kill-multiplicity is D0.5. ∎

### 3.7 L7 — Structure of ripeness: decomposition, defusal, mass floors, horizon

Throughout, Q is an attacker-to-move position satisfying (I1) (the shape every position handed by an L1.2-servicing defender has), (c₁, c₂) is a legal attacker pair, and `F = I(Q + c₁ + c₂)`.

**Lemma L7.1 (membership) [PROVEN].** Every W ∈ F was alive at Q with `count ≤ 3`, and contains c₁ or c₂. If W is count-5 in F, it contains both. If W is count-4 and contains exactly one cᵢ, it was count-3 at Q; if it contains both, it was count-2 at Q.

*Proof.* Dead windows stay dead and the defender did not move, so W was alive at Q; (I1) gives count ≤ 3 there. Its count at F is ≥ 4 and grew only by placements inside it, so it contains ≥ 1 of the placed cells; the case counts follow from "one placement adds exactly 1 to windows through it". ∎

**Definition (clusters, heavy cells).** `F₁ = {W ∈ F : c₁ ∈ W}`, `F₂ = {W ∈ F : c₂ ∈ W, c₁ ∉ W}` (disjoint, F = F₁ ⊎ F₂ by L7.1). A placement cell cᵢ is *heavy* for the pair if `τ(Fᵢ) ≥ 2`.

**Lemma L7.2 (decomposition) [PROVEN].** τ(F) ≤ τ(F₁) + τ(F₂). Hence if Q is ripe with witness pair (c₁,c₂), then (up to swapping) either **(T1)** τ(F₁) ≥ 3, or **(T2)** τ(F₁) = 2 and F₂ ≠ ∅. In both cases at least one placement cell is heavy.

*Proof.* Union of covers is a cover. τ(F) ≥ 3 forces τ(F₁)+τ(F₂) ≥ 3; if neither were ≥ 2 the sum would be ≤ 2. τ(F₂) ≥ 1 ⇔ F₂ ≠ ∅. ∎

**Lemma L7.3 (one-stone defusal of a cluster) [PROVEN].** For any cell c, a defender stone at c kills every alive window through c; in particular, for **any** pair (c₁,c₂) and any i, all pre-images of Fᵢ-windows contain cᵢ, so a defender stone at cᵢ removes the entire cluster Fᵢ from every future maturation. A T1 witness (triple junction at c₁) is annihilated by the single placement D@c₁; a T2 witness loses its heavy half to D@c₁, and what remains (single-window clusters only) can contribute τ ≤ 2, which is serviceable.

*Proof.* Immediate from D0.5 and L7.1 (every W ∈ Fᵢ contains cᵢ as an empty of Q, a legal defender cell). For the last clause: with all heavy cells defused, every remaining pair (c₁′,c₂′) has τ(F₁′) ≤ 1 and τ(F₂′) ≤ 1, so τ(F′) ≤ 2. ∎

*Caution (why L7.3 does not finish the proof).* "Defuse every heavy cell" may cost more than the defender's spare budget: heaviness is a property of *cells*, many cells can be heavy at once (sharing support stones), and forced covers of I(P) compete for the same two placements. Bounding the number of *independently priced* heavy cells is exactly the content of §3.8–3.9.

**Lemma L7.4 (witness mass floors) [PROVEN].** Let Q be ripe with witness (c₁,c₂), F as above, and let `w(Q; c₁,c₂) = Σ_{W ∈ F} λ^(−e_Q(W))` be the *pre-mass* of the witness (weights taken at Q). Then:

1. Every W ∈ F has pre-weight ≥ λ⁻⁴ = 1/9 (count ≥ 2 at Q by L7.1); every W ∈ F₂ has pre-weight λ⁻³ (count-3 at Q).
2. **T1 branch:** |F₁| ≥ 3 (each window needs a cover cell, τ ≤ |family|), so w ≥ 3·(1/9) = **1/3**.
3. **T2 branch:** |F₁| ≥ 2 and |F₂| ≥ 1, so w ≥ 2·(1/9) + λ⁻³ = 2/9 + 1/(3√3) ≈ **0.415**.
4. Hence any ripe Q has `Φ(Q) ≥ 1/3` carried on the witness windows alone.

*Proof.* (1) L7.1 and D0.4. (2,3) count the windows; a family with τ ≥ k has ≥ k members since choosing one empty per window covers it. (4) F's members are distinct alive windows at Q and Φ sums over all alive touched windows (count ≥ 2 ⇒ touched). ∎

**Refinement R7.4 [CONJECTURED, machine-checkable].** In the T1 branch, all-count-2 triples require all three windows to contain both c₁ and c₂ (else they can't gain +2), forcing all three to be same-axis segments through the collinear pair (c₁,c₂) at distance ≤ 5. Worked 1-D cases show these self-reveal or collapse:

- *Adjacent pair.* c₁ = 0, c₂ = 1: the both-cell windows are W_s = {s..s+5}, s ∈ {−4..0}. An all-count-2 T1 triple needs three of them with pairwise-disjoint residuals. The extremes W₋₄, W₀ have automatically disjoint residuals, but every candidate third window shares 4 of its 6 cells with an extreme, and the cases collapse: inward stones (W₋₄-stones {−2,−1}, W₀-stones {2,3}) make the middle window {−2..3} an already-present **count-4** with empties exactly {c₁,c₂} — the witness self-reveals one epoch early as a forced block, and the single cover cell 0 (or 1) kills all three windows at once; outward stones ({−4,−3}, {4,5}) leave every third window at count ≤ 1; mixed choices give count-3 thirds (mass then ≥ 2/9 + λ⁻³, above the conjectured floor anyway) or intersecting residuals. No all-count-2 triple materialised in any assignment tried.
- *Spread pair.* c₁ = 0, c₂ = 3: the three windows containing both ({−2..3}, {−1..4}, {0..5}) have middle cells {1,2} in common; any count-2 assignments leave all residuals ⊆ {1,2} ∪ (two flank cells), and the worked assignments all produced pairwise-intersecting residuals — τ ≤ 1, no heavy cluster.

We conjecture this pattern is exhaustive: pure-collinear T1 witnesses are impossible, and the true ripe floor is `2/9 + λ⁻³ ≈ 0.415`. *Intended check:* exhaust all stone placements on a single 12-cell axis segment for every offset pattern of (c₁,c₂); finite (≤ 2¹⁰ configurations per pattern). Not run this round (budget).

**Lemma L7.5 (local stone floor for ripeness) [VERIFIED via L2].** Any ripe Q has ≥ 5 attacker stones inside the witness footprint ∪F ∖ {c₁,c₂}.

*Proof.* At Q + c₁ + c₂ the family F has τ ≥ 3, so by L2 (locality remark) ∪F contains ≥ 7 attacker stones; removing the two placements leaves ≥ 5 stones already present at Q. ∎

**Theorem C (unconditional horizon extension) [PROVEN, standing on VERIFIED L2].** Let P₀ be any Defender-FirstStone position with a₀ attacker stones (any Φ). Then **every** defender that services I(P) whenever τ(I(P)) ≤ 2 (any completion-first rule; no suppression needed) survives every attacker turn t with `a₀ + 2(t−1) ≤ 6`. Equivalently the attacker's earliest possible completion is on its turn `t* + 1` where `t* = min{t : a₀ + 2t ≥ 7}`:

- a₀ = 1: no six during attacker placements 1–6; earliest six at placements 7–8 (attacker turn 4);
- a₀ = 2: earliest six at placements 7–8 as well (t* = 3).

*Proof.* A completion at attacker turn t requires (L1.1, L1.2) an epoch before turn t with τ(I) ≥ 3, i.e. an unblockable pileup with the attacker owning `a₀ + 2(t−1)` stones; by L2 this needs ≥ 7 stones. While `a₀ + 2(t−1) ≤ 6`, every epoch's imminent family has τ ≤ 2 (again L2: τ ≥ 3 is impossible), so a servicing defender's covers exist and L1.2 blocks the turn. ∎

*Relation to Theorem 2 of `ES_GLOBAL_BOUNDARY.md`.* Theorem 2 certifies the first **5** attacker placements from any Φ<1 position via fixed-cohort greedy, sharp for that strategy class. Theorem C certifies placements 1–6 (first six no earlier than placement 7) for the low-stone roots (a₀ ≤ 2) that dominate the Φ<1 threshold, for **every** servicing defender, with no potential hypothesis at all. For large-a₀ Φ<1 positions Theorem C is vacuous and Theorem 2 remains the only horizon certificate — the two are complementary. This also sharpens the birth-ledger report's "4th–7th placement" phrasing to: *pileup earliest at the epoch after attacker turn 3, six earliest at attacker placements 7–8 (a₀ ≤ 2).*

### 3.8 L8 — GAP-AMORTIZED-ABANDONMENT attack: the graded account Θ₂

**Definition (graded account).** `Θ₂(P) = Σ λ^(−e(W))` over alive windows with `count(W) ≥ 2`. (Φ restricted to count ≥ 2; count-1 births carry zero Θ₂.) Always `Θ₂ ≤ Φ`.

**Lemma L8.1 (clean escape is Θ₂-invisible) [PROVEN].** The clean-escape turn of `ES_GLOBAL_BOUNDARY.md` Lemma 1 — two placements each of whose 18 windows were stone-free beforehand, sharing no window — injects **exactly 0** into Θ₂ while injecting `4/√3` into Φ.

*Proof.* All 36 affected windows go from count-0 to count-1; none reaches count 2. ∎

**Consequence (Cor-2 neutralized) [PROVEN].** Corollary 2's divergent birth source — the argument that kills Φ<1 as a renewable invariant and grounds both named gaps' "repeated clean escape" obstruction — consists entirely of count-1 mass. It places no obstruction on maintaining `Θ₂ < 1`. Likewise the §4 blanket-game divergence (labels stay at count one or are killed) never touches Θ₂. *This is the round's account-design claim: the renewal invariant should be graded at count ≥ 2, where every object the service criterion (L1.2) and the witness floors (L7.4) price actually lives.*

**Lemma L8.2 (Θ₂ injection bounds) [PROVEN].**
1. Per attacker placement at cell c: `ΔΘ₂ ≤ (λ−1)·S₂(c) + n₁(c)/9`, where S₂(c) = Θ₂-mass of alive windows through c and n₁(c) ≤ 18 = number of count-1 windows through c. Hence `ΔΘ₂ ≤ (λ−1)Θ₂ + 2` per placement and `Θ₂′ ≤ 3·Θ₂ + 2(λ+1) ≈ 3Θ₂ + 5.46` per turn (worst case, all geometry saturated).
2. Sparse-injection benchmark: an isolated **adjacent pair** (two stones, same axis, distance 1, far from everything) injects exactly `5·λ⁻⁴ = 5/9 ≈ 0.556` (the 5 shared-axis windows through both stones), all of it remote and dormant.
3. A defender chase stone kills 4 of those 5 windows (one axis-neighbor placement), leaving residue `1/9`; two chase stones clear all 5.

*Proof.* (1) Promotions multiply a window's weight by λ; entries appear at λ⁻⁴; ≤ 18 windows through a cell. (2,3) Direct enumeration of the 5 six-segments containing both cells of an adjacent pair; a defender stone at either outer axis-neighbor lies in 4 of the 5. ∎

**The abandonment mechanism, restated in Θ₂ currency.** The attacker can mint `5/9` of remote dormant Θ₂ per turn indefinitely. The defender's dilemma per remote pair: spend 2 stones (full clear, zero residue, zero spares left — a pure-chase policy, greedy-class, and exactly the completion-blind shape the boundary doc's Theorem 1 defeats on the ES cohort line), spend 1 (residue 1/9 per pair accrues unboundedly — **no fixed threshold θ\* renews against pure accumulation**), or spend 0 (full 5/9 accrues). Naïve Θ₂-thresholding therefore fails, *but not for Cor-2's reason* — the failure is now a **rate race on count-≥2 mass**, which is exactly where the defender's kill-multiplicity (L6) and the witness floors (L7.4) also live. What saves the defender, if anything does, is that dormant mass is harmless until *reconcentrated*:

**Lemma L8.3 (reactivation is local and priced) [PROVEN].**
1. Ripeness requires ≥ 5 attacker stones inside the witness footprint (L7.5), and every witness window passes through c₁ or c₂ (L7.1) — a witness is supported within radius 5 of the two placement cells.
2. Any heavy cell of a witness whose cluster windows lie on ≥ 2 axes requires ≥ 2 attacker placements *inside the cluster's windows* after the windows were last count-≤2 there; a single placement can lift only same-axis overlapping count-2s into a multi-window cluster through a second common empty, and in the worked 1-D cases (R7.4, §3.7) such same-axis structures either share residual empties pairwise (no heavy cluster) or self-reveal as a pre-existing count-4.
3. One defender spare placed at a heavy-in-formation cell c kills every window through c (L7.3) — the *cell* c is dead as a junction forever; the *stones* survive and may support other junction cells, so defusal is per-cell, not per-cluster.

*Proof.* (1) Restates L7.1/L7.5. (2) A placement adds +1 to windows through it; a cross-axis pair of count-3s through a common empty c needs both windows separately promoted to count-3. (3) D0.5; windows through other cells are untouched. ∎

**Obligation O2′ (sharpened GAP-AMORTIZED-ABANDONMENT) [OPEN].** Exhibit an adaptive servicing strategy S and show: in every play from `Θ₂ < 1`, at every epoch, S's spare budget `2 − τ(I(P))` suffices to keep every attacker-turn-start position unripe, where the accounting may charge the attacker's *reactivation* placements (L8.3.2: ≥ 2 per cross-axis heavy cell; L7.5: ≥ 5 stones per witness footprint) against the defender's banked spares. Equivalently (amortized form): find a potential/refund function on dormant clusters — history-sensitive, as §8 of the boundary doc demands — under which (i) minting dormant mass (L8.2.2) does not raise it past the spare rate, (ii) reactivating a dormant cluster raises it by ≥ the spares needed to re-defuse, (iii) it is bounded at ripeness. **What is genuinely new relative to the gap statement:** the currency in which the refund rule must be written is now pinned (count-≥2 graded mass + per-cell junction defusal), the reactivation price is lower-bounded (2 placements per cross-axis heavy, 5 stones per footprint), and the failure of naive thresholds is located precisely (residue accumulation `1/9`/turn, not clean-escape divergence).

### 3.9 L9 — GAP-GLOBAL-RENEWAL attack: the budget kernel

The renewal gap says no proof route returns to a Defender epoch satisfying the original hypothesis (Cor-2 kills Φ<1 renewal from Φ=0 already). This section proves that the *service side* of the induction — one full epoch of coverage plus unripeness — is affordable whenever the graded mass is below 1, and isolates the single covering hole.

Setting: defender epoch P; the defender must pick 2 cells that (a) cover I(P) and (b) make the resulting attacker-turn-start position unripe.

**Theorem B (budget kernel) [PROVEN for τ(I)=2; PARTIAL for τ(I)≤1].** Let P be a defender epoch with `Θ₂(P) < 1` (a fortiori if Φ(P) < 1, since Θ₂ ≤ Φ).

- **K1 (fully forced case, τ(I(P)) = 2): PROVEN.** Some 2-cell cover of I(P) hands the attacker an unripe position.
- **K2 (τ(I(P)) = 1): PARTIAL.** Either some (cover, spare) pair hands unripe, or every witness surviving every spare choice belongs to a *pairwise window-overlapping* family of total mass < 2/3.
- **K3 (free case, τ(I(P)) = 0): PARTIAL.** Either some spare pair hands unripe, or the surviving witnesses form a pairwise-overlapping family of mass < 1 that no two placements defuse.

*Proof.*

K1: `τ(I) = 2 ⇒ |I| ≥ 2 ⇒ mass(I) ≥ 2·λ⁻² = 2/3` (imminent windows weigh ≥ 1/3). Suppose every cover {x, y} of I(P) yields a ripe position Q. Fix any cover; let F be its witness. F's windows are alive at Q, hence alive at P with the same attacker counts (defender stones only kill, never change counts); their counts are ≤ 3 (L7.1), so they are distinct from I(P)'s members. By L7.4(4), mass(F) ≥ 1/3, at weights that agree between Q and P. Then `Θ₂(P) ≥ mass(I) + mass(F) ≥ 2/3 + 1/3 = 1`, contradiction. So some cover hands unripe. ∎(K1)

K2: `mass(I) ≥ 1/3`. Suppose every (cover x, spare y) yields ripe, and fix a cover x₀. For each spare y let F_y be a surviving witness. If some two F_y, F_y′ are window-disjoint, then `Θ₂(P) ≥ 1/3 + 1/3 + 1/3 = 1`, contradiction. So either a good (x₀, y) exists or all surviving witnesses pairwise share windows and their union has mass < 1 − 1/3 = 2/3. ∎(K2)

K3: identical arithmetic with mass(I) = 0: three pairwise window-disjoint witnesses give Θ₂ ≥ 1, excluded; two window-disjoint witnesses are defused by two spares at their heavy cells (L7.3); and a *pool* of ≤ 2 count-3 windows supporting all heavy cells is defused by killing pool windows directly (a defender stone on any empty of a window kills it). The uncovered case is a pairwise-overlapping witness family, mass < 1, needing ≥ 3 distinct defusal cells. ∎(K3)

**Obligation W3′ (overlap defusal) [OPEN; finite-flavored].** Show: at any epoch with Θ₂ < 1, after covering I(P), some placement of the remaining spare(s) defuses every pairwise-window-overlapping witness family. Why this looks closable: overlapping witnesses are geometrically local (every witness window passes through one of the two trigger cells, L7.1; overlap chains propagate within window diameter ~5), so the adversarial configurations live in a bounded neighborhood and are exhaustively enumerable — the same machinery as L2/L3. *Intended check:* enumerate attacker configurations within radius ≤ 8 of a marked cell with Θ₂ < 1 admitting ≥ 2 overlapping witnesses; test whether the available spare(s) always defuse. Not run this round (budget); top target for round 2's harness work.

**Corollary B′ (what the kernel buys, and what it does not) [PROVEN, conditional on W3′].** If W3′ holds, then `Θ₂ < 1` at a defender epoch implies the defender can cover I(P) *and* hand an unripe position — one full safe cycle, with the same certificate available at the next epoch whenever `Θ₂ < 1` again. The induction closes, and GAP-RAW follows, **iff additionally** the defender can keep `Θ₂ < 1` at every epoch:

**Obligation O1′ (sharpened GAP-GLOBAL-RENEWAL) [OPEN].** Exhibit an adaptive rule whose spare placements maintain `Θ₂(P) < 1` at every defender epoch of every play from a `Θ₂ < 1` start. Known pressure, exact numbers: remote minting ≤ 5/9 per turn (L8.2.2; clearable to 0 at 2 stones, to 1/9 at 1 stone), local injection `(λ−1)·S₂(c) + n₁(c)/9` per placement (L8.2.1) — concentrated through the two placement cells, hence chaseable in principle by the defender's two kills. Known failure: fixed R1b does **not** maintain it — its placement-48–60 break epochs have τ(I) ≥ 3, hence Θ₂ ≥ 1 there (three ≤2-empty windows weigh ≥ 1), so Θ₂ crossed 1 strictly earlier on those lines while R1b's spares chased touched-danger Φ_all instead. *Intended check:* add a Θ₂ column to `trace_r1b_breaks`; locate the first Θ₂ ≥ 1 epoch on each break line and what the spares were doing; then test a Θ₂-aware spare rule (argmax killed Θ₂-mass, or minimax of the attacker's one-turn Θ₂ response) on the break lines and the randomized battery.

**Why Cor-2 does not refute O1′ [PROVEN].** L8.1: the clean-escape source that makes Φ-renewal impossible is Θ₂-invisible; the §4 blanket-game divergence concerns count-one labels only. Proposition 3 excludes *static edgewise* factor-three accounts; Θ₂-with-adaptive-spares is neither static nor edgewise. (Equally, it is not yet proven to renew — that is O1′.)

### 3.10 L10 — Assembly: the conditional theorem and the exact residue

**Theorem D (conditional GAP-RAW) [PROVEN modulo W3′ and O1′].** Suppose:

- **(W3′)** overlap defusal: at every defender epoch with Θ₂ < 1, after covering I(P), the spare(s) can defuse every pairwise-overlapping witness family; and
- **(O1′)** Θ₂-renewal: some adaptive rule maintains Θ₂ < 1 at every defender epoch from a Θ₂ < 1 start.

Then GAP-RAW holds: from every Defender-FirstStone position with Φ < 1 the defender holds forever.

*Proof.* Φ < 1 ⇒ Θ₂ < 1 at P₀ (Θ₂ ≤ Φ). By induction over epochs: at each epoch Θ₂ < 1 (O1′ maintains it; base case P₀). By Theorem B (K1 proven; K2/K3 closed by W3′), the defender covers I(P) — possible since Θ₂ < 1 excludes τ(I) ≥ 3, as three imminent windows weigh ≥ 1 — and hands an unripe position, choosing spares consistently with O1′'s rule (O1′ is stated for the same strategy, so the two choices must be exhibited jointly; this is why D is conditional on the *pair*). The attacker's turn from an unripe position yields τ(I) ≤ 2 at the next epoch (definition of ripeness), and by L1.2 no completion ever occurs. ∎

*Consistency note.* O1′ and W3′ are not independent add-ons: the round-2 target is a single rule witnessing both simultaneously (spares must both drain Θ₂ and defuse witnesses; K1 shows fully-forced turns need no spares at all, which is the pressure valve).

**Residual obligations, in order of expected difficulty (ascending):**

| id | statement | flavor | route |
|----|-----------|--------|-------|
| R7.4 | ripe-witness sharp floor (`0.415` vs proven `1/3`); no pure-collinear T1 witnesses | finite, 1-D | short exhaustive check |
| W3′ | overlap defusal under Θ₂ < 1 | finite-flavored, local | bounded-radius enumeration (L2/L3 machinery) |
| O2′ | amortized abandonment: refund rule pricing reactivation ≥ re-defusal cost | infinite-horizon amortization | design on top of L8.3's priced reactivation; the true successor of GAP-AMORTIZED-ABANDONMENT |
| O1′ | Θ₂ < 1 renewal under an adaptive rule | infinite-horizon invariant | strategy design + instrumented traces; the true successor of GAP-GLOBAL-RENEWAL; subsumes O2′ pressure |

**What moved this round.** GAP-RAW is now *exactly equivalent* (Theorem A/A′, unconditional) to perpetual unripeness maintenance; the service arithmetic below mass 1 is closed in the fully-forced case (K1) and cornered into one local covering question elsewhere (W3′); the account that renewal must be written in is identified and Cor-2's obstruction is proven not to apply to it (L8.1); and the certified opening horizon is extended beyond Theorem 2 for the threshold-dominant roots (Theorem C). The two named gaps survive, but strictly smaller: GAP-GLOBAL-RENEWAL has shed its "no account can renew" shadow (that was a fact about Φ, not about accounts), and GAP-AMORTIZED-ABANDONMENT is reduced from "discount indefinitely many births" to "price reactivation of dormant count-≥2 clusters against banked spares" with both sides of the price already lower-bounded.

## 4. Attack surface for review

Where I would strike first, in order:

1. **K2/K3's uncovered overlap case is load-bearing and unpriced (W3′).** The kernel's clean arithmetic closes only the fully-forced case (K1). If pairwise-overlapping witness families of mass < 2/3 that defeat every spare placement are geometrically realizable, Theorem B collapses to K1 alone and Corollary B′/Theorem D lose most of their force. I believe they are not realizable (overlap forces shared kill-cells or pre-existing imminent windows), but this round contains **no proof and no enumeration** — only the locality argument for why the check is finite. Hit here first.
2. **The mass floors lean on `τ(F) ≤ |F|` and per-window weight 1/9.** The 1/3 ripe floor (L7.4) admits in principle the all-count-2 collinear T1 branch, which I argue away only in worked 1-D cases (R7.4, CONJECTURED). If a collinear triple with pairwise-disjoint residuals and *no* self-revealing count-4 exists, the floor stays 1/3 but the conjectured 0.415 sharpening dies, weakening any future tightening of the kernel that relies on it. (Theorem B as written uses only the proven 1/3 — check that no step silently uses 0.415. I believe none does.)
3. **L8.3.2 (reactivation price ≥ 2 placements) is proven only for cross-axis heavy cells built from count-≤2 stock.** Same-axis heavy formation in one placement is excluded only by the R7.4-class 1-D analysis. A reviewer constructing a one-placement same-axis heavy cell with disjoint residuals would cut O2′'s priced-reactivation floor in half.
4. **Theorem D's joint-strategy requirement.** O1′ and W3′ are stated as separate obligations but Theorem D needs one rule satisfying both simultaneously; a reviewer may object that the conjunction could be unsatisfiable even if each holds separately (spares wanted in two places at once). The K1 pressure-valve remark is the sketch of why not; it is not a proof.
5. **Scope fidelity of the VERIFIED imports.** L4/L5a are per-root exhaustive, not universal — the draft marks this, but any later section silently treating "one turn from Φ<1 makes ≤3 count-4s" as universal would be wrong (§3.7/3.9 do not; check independently).
6. **Theorem C's servicing hypothesis.** It certifies any *servicing* defender; a defender that refuses to service loses earlier, and the theorem says nothing about it. Also its interesting content is confined to a₀ ≤ 2 starts; for dense Φ<1 positions it is vacuous and Theorem 2 remains the only horizon result.
7. **Legality details.** All cover/defusal cells are proven legal (within 5 of a stone in their own window); spare legality is colour-blind (within 8 of any stone, board infinite) and holds in every position with ≥ 1 stone — which every Defender-FirstStone position has. I believe this surface is closed; verify the distance-5-inside-a-window claim (6 consecutive cells ⇒ max in-window distance 5) survives the hex-metric definition used by the engine.

## 5. Status table
*(authoritative; every claim labeled)*

| Claim | Label | Evidence / where |
|-------|-------|------------------|
| L1.1 completion criterion | PROVEN | §3.1 |
| L1.2 epoch service criterion | PROVEN | §3.1 |
| Theorem A / A′ exact reformulation (GAP-RAW ⇔ perpetual unripeness) | PROVEN | §3.1 |
| L2 fork floor (τ≥3 ⇒ ≥7 stones in ∪F; ≤6 ⇒ never) | VERIFIED | re-run at HEAD, PASS 199.6 s; §6 cmd 1 |
| L3 per-n ceilings (n=4..12, four columns) | VERIFIED | same run; §6 cmd 1 |
| L4 one-turn maturation (0 / ≤3-straight-4 / R1b-reset-0) | VERIFIED (per-root; item 1 pencil-general) | re-run at HEAD, PASS 608.2 s; §6 cmd 2 |
| L5a ΔΦ per-root ceilings (2.309 / 2.591) | VERIFIED (per-root) | same run; §6 cmd 2 |
| L5b general ΔΦ bound (Φ′ ≤ λ²Φ + 2(λ+1)/√3) | PROVEN | §3.5 |
| L6 service capacity + legality; kill-multiplicity | PROVEN | §3.6 |
| L7.1 membership; L7.2 decomposition; L7.3 one-stone defusal | PROVEN | §3.7 |
| L7.4 witness mass floors (ripe ⇒ Θ₂ ≥ 1/3 on witness) | PROVEN | §3.7 |
| R7.4 sharp floor 0.415; no collinear T1 | CONJECTURED | §3.7; intended 1-D check |
| L7.5 local stone floor (ripe ⇒ ≥5 pre-stones in footprint) | VERIFIED (inherits L2) + PROVEN glue | §3.7 |
| Theorem C horizon extension (six no earlier than placement 7 for a₀≤2, any servicing defender) | PROVEN (on VERIFIED L2) | §3.7 |
| L8.1 clean escape Θ₂-invisible; Cor-2 neutralized for graded accounts | PROVEN | §3.8 |
| L8.2 Θ₂ injection bounds (5/9 remote mint; 1/9 chase residue) | PROVEN | §3.8 |
| L8.3 reactivation locality + pricing (cross-axis case) | PROVEN (cross-axis) / CONJECTURED (same-axis exclusion) | §3.8 |
| O2′ amortized abandonment (refund rule) | OPEN | §3.8 |
| Theorem B kernel: K1 (τ(I)=2) | PROVEN | §3.9 |
| Theorem B kernel: K2, K3 | PARTIAL (overlap hole) | §3.9 |
| W3′ overlap defusal | OPEN (believed true; finite-flavored) | §3.9 |
| O1′ Θ₂-renewal | OPEN | §3.9 |
| R1b break epochs have Θ₂ ≥ 1 (so Θ₂ crossed 1 earlier) | PROVEN (from τ≥3 ⇒ mass≥1) | §3.9 |
| Corollary B′; Theorem D (GAP-RAW modulo W3′ + O1′) | PROVEN, conditional | §3.9, §3.10 |
| No forced pileup/six from Φ<1 roots, plies ≤6 exhaustive | VERIFIED | sibling report `HUNT_REPORT_BIRTH_LEDGER.md` Item 3, same commit 9b32db63 and identical harness file (copied wholesale); **not re-run this round** (projected > 10 min); §6 cmd 3 |

## 6. Regen commands

All from worktree root `E:\Hexo-BotTrainer-hexgt\.claude\worktrees\hunt-gap-raw` (branch `hunt/gap-raw`, HEAD 9b32db63), harness `packages/hexfield_eq/rust/src/gap_raw_hunt.rs` = byte-identical copy of the birth-ledger worktree's superset (uncommitted in both). Deterministic; serial cargo; check free RAM > 9 GB first.

```bash
# 1 — L2/L3 geometry ceilings + A000228 + superadditivity + brute-force validations
#     (re-run this round: PASS, 199.56 s)
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  gap_raw_hunt::tests::birth_ledger_geometry -- --ignored --nocapture --test-threads=1

# 2 — L4/L5a maturation frontier (re-run this round: PASS, 608.24 s)
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  gap_raw_hunt::tests::birth_ledger_maturation -- --ignored --nocapture --test-threads=1

# 3 — pileup minimax (NOT re-run this round; VERIFIED in sibling report at same commit)
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  gap_raw_hunt::tests::birth_ledger_pileup -- --ignored --nocapture --test-threads=1
```

Constants used throughout: `λ = √3`; weights λ⁻⁵ ≈ 0.0642, λ⁻⁴ = 1/9, λ⁻³ = 1/(3√3) ≈ 0.1925, λ⁻² = 1/3, λ⁻¹ = 1/√3 ≈ 0.5774; `18·λ⁻⁵ = 2/√3 ≈ 1.1547`; `4/√3 ≈ 2.3094`.
