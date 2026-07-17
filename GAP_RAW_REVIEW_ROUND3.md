# R-G1-REV — Round-3 hostile review

**Reviewed branch / artifact:** `hunt/gap-raw`, `06a1a649db6f3c7ec517d9835b129fead30d343d`

**Documents:** `GAP_RAW_PROOF_ROUND3.md`, then `STRATEGY_STEALING_HEXO.md`

**Method:** first-principles proof and source audit. `GAP_RAW_PROOF_ROUND2.md`, `GAP_RAW_REVIEW_ROUND2.md`, and `HUNT_REPORT_GAP_RAW.md` were read first, in that order and in full. The Round-3 Rust regression was code-read only. **No Cargo command was run.**
**Overall disposition:** `GAP_RAW_PROOF_ROUND3.md` is **SOUND-WITH-ERRATA**. `STRATEGY_STEALING_HEXO.md` is **SOUND-WITH-ERRATA**. No load-bearing PROVEN claim is refuted.

## Numbered findings

### 1. ACCEPT — Round-3 uses the exact Round-2 account and transition meanings

Round 2 fixes the following meanings:

- `B_2(h)=Theta_2(P_h)`;
- `kappa_2(P,x)` is the current `Theta_2` mass of alive count-at-least-two windows through Defender cell `x`;
- `S_2(P,c)` is the current `Theta_2` mass of alive count-at-least-two windows through Attacker cell `c`;
- `n_1(P,c)` counts alive count-one windows whose empty set contains the legal Attacker cell `c`; and
- equation (9), on a nonterminal placement, is exactly

  `Delta_D B_2=-kappa_2(P,x)` and
  `Delta_A B_2=(sqrt(3)-1)S_2(P,c)+n_1(P,c)/9`.

Round 3's state notation `B_2(P):=Theta_2(P)` merely evaluates that history account at the current state. It does not change the account. Equations (13)–(16) evaluate `kappa_2`, `S_2`, and `n_1` at the correct sequential pre-placement states. No redefinition is smuggled into the refutation.

### 2. ACCEPT — L9.1's complete 31-window pencil has exactly one survivor

For `A={(0,0),(1,0)}`, the independent count is

`18+18-5 = 31`,

because the adjacent cells share exactly five Q-axis windows and share no R- or QR-axis window. The full containment audit is:

| Pencil | Window starts | Blocker containment |
|---|---|---|
| Q-axis union | `-5,-4,-3,-2,-1,0,1` | `(-1,0)` is in starts `-5..-1`; `(6,0)` is in start `1`; start `0` is blocker-free |
| R through `(0,0)` | `-5..0` | `(0,-4)` is in starts `-5,-4`; `(0,1)` is in starts `-4..0` |
| R through `(1,0)` | `-5..0` | `(1,-1)` is in starts `-5..-1`; `(1,3)` is in starts `-2..0` |
| QR through `(0,0)`, `(t,-t)` | `-5..0` | `(-3,3)` is in starts `-5..-3`; `(1,-1)` is in starts `-4..0` |
| QR through `(1,0)`, `(1+t,-t)` | `-5..0` | `(0,1)` is in starts `-5..-1`; `(3,-2)` is in starts `-3..0` |

The group sizes are `7+6+6+6+6=31`. Every listed range is correct, including the overlaps at the range boundaries. None of the eight blockers lies in Q-start `0`, whose cells are `(0,0),...,(5,0)`. An independent enumeration of all 31 keys returned that window as the sole Defender-free label, with Attacker count two.

### 3. ACCEPT — L9.2 and L9.4 give the exact root profile and account

Translations preserve Finding 2. The minimum distance between Attacker stones in consecutive gadgets is `30-1=29`, while a six-cell window has diameter five, so no label contains Attacker stones from two gadgets. Each displayed `W_i` remains Defender-free. The exact profile is therefore

`(n_1,n_2,n_3,n_4,n_5,n_6)=(0,8,0,0,0,0)`.

Each surviving label has four empties and weight

`(sqrt(3))^-4=1/9`.

Consequently `Phi(P_*)=Theta_2(P_*)=B_2(P_*)=8/9<1`, and `I(P_*)=empty`. The support counts are also exact: `|A|=8*2=16` and `|D|=8*8+3=67`. The sets are finite, nonempty, disjoint, and Attacker-nonterminal.

### 4. ERRATUM — L9.3 has an off-by-one in its separation prose

Successive row-zero windows are `[30i,30i+5]` and `[30i+30,30i+35]`. Their closest occupied-window cells are distance 25 apart, but the intervening row-zero cells are `30i+6,...,30i+29`, which is **24** empty cells, not 25.

Exact fix in L9.3: replace

> “The `W_i` lie on row 0 in Q-intervals separated by 25 empty cells.”

with

> “The `W_i` lie on row 0 in Q-intervals whose closest cells are at hex distance 25, with 24 intervening row-zero cells.”

This does not affect disjointness, the minimum cross-gadget Attacker distance 29, or any later inequality.

### 5. ACCEPT — launch geometry, pigeonhole, legality, and nonterminality are complete

For each launch, the five common Q-windows of `c_j=(0,R_j)` and `d_j=(1,R_j)` start at `q=-4,-3,-2,-1,0`; their literal union is `U_j=[-4,5] x {R_j}`. The eight `W_i` and three `U_j` are pairwise disjoint, and every `U_j` is root-stone-free. Thus:

- a physical cell cannot meet two `U_j`;
- a physical cell cannot lie in both some `W_i` and some `U_j`;
- a legal sequential Defender reply has `x_1!=x_2`, because the first cell is occupied before the second placement; and
- two Defender cells meet at most two launch unions, so one `U_j` is untouched.

This closes the ordered-pair edge cases in (12) and simultaneously preserves the old-label bound `k<=2`.

The production predicate is inclusive: `LEGAL_RADIUS=8`, and `coords_within_radius` iterates closed ranges. Hence `d(c_j,a_j)=8` is legal, not excluded by a strict-radius rule. After `c_j` is placed, the legal store is updated before the phase advances, so adjacency makes `d_j` legal. In fact `d(d_j,a_j)=8` as well, giving redundant root support. Every old Attacker component and the new component contains only two Attacker stones, and distinct components are farther apart than window diameter five. No Attacker six is formed, so the blanket turn returns to Defender-`FirstStone` and all four equation-(9) updates are defined.

### 6. ACCEPT — equations (13)–(17) are exact, not merely lower bounds

Let `k` be the number of distinct old targets met by the two Defender cells. Sequential `kappa_2` evaluation kills each such `1/9` label once, even if both cells lie in the same label, so

`B_2(Q)=8/9-k/9` with `0<=k<=2`.

At the untouched launch, `Q` has no count-one label at all, and `c_j` lies in no surviving old count-at-least-two label. Therefore the first Attacker instance has

`S_2(Q,c_j)=0`, `n_1(Q,c_j)=0`, and `Delta B_2=0`.

Immediately before `d_j`, exactly the five common Q-windows are Defender-free count-one labels through `d_j`. Distinct Q-collinear triggers share no R- or QR-axis label, and no old label reaches the launch. Hence

`S_2(Q+A@c_j,d_j)=0`, `n_1(Q+A@c_j,d_j)=5`, and `Delta B_2=5/9`.

Thus the next epoch satisfies the exact identity

`B_2(P')=(8-k+5)/9=(13-k)/9>=11/9`.

The universal quantifier is in the necessary order: fix any legal ordered Defender pair, choose an untouched launch afterward, then make the stated legal Attacker response. This refutes canonical J.4 at the initial epoch for every possible strategy action.

### 7. ACCEPT — the normative domain admits `P_*`; a reachable-only theorem would not

Round 2 defines GAP-RAW roots as **every** finite, nonempty, nonterminal blanket position with Defender at `FirstStone` and `Phi<1`. Neither that definition nor Obligation J imposes engine reachability, a connected radius-eight occupancy graph, or a stone-count parity condition. `P_*` is therefore in the exact quantified domain.

`P_*` itself is not engine-reachable. At every engine-reachable Defender-`FirstStone` epoch after the singleton opening, the cadence gives `|A|=|D|+1`; `P_*` has `16` Attacker stones and `67` Defender stones. In addition, `(0,0)` has the Attacker owner in `P_*`, whereas under the role convention the engine opener there is Defender, and the construction has radius-eight-disconnected components. No engine-reachable equivalent is supplied.

That would be fatal only to a different, strictly weaker reachable-root claim. Round 3 says exactly that and does not present this construction as a refutation of such a weaker theorem.

### 8. ACCEPT — L9.6, R3.2, L9.7/R3.2.1, L9.8, and R3.3 survive

- **L9.6.** Every imminent label contributes at least `1/3` to `Theta_2`, and nonterminality gives every such label a nonempty residual. Selecting one residual cell per label proves `tau<=|I|`; hence `B_2>=|I|/3>=tau/3`, and strict `B_2<1` forces integer `tau<=2`.
- **R3.2.** J.2 leaves a handoff with `I=empty`. Round-2 L1.1 prevents a completion during any legal two-placement reply, J.4 gives `B_2(P')<1`, and L9.6 gives `tau(P')<=2`. This is exactly J.3.
- **L9.7.** Four exact sequential transitions give `B_2(P')=B_2(P)-K(P,a)+Delta(Q,b)`. The legal ordered-response set is finite and nonempty for a finite nonempty position, so maximizing converts the universal response condition exactly into the strict margin (19). At `P_*`, its two sides differ by at least `2/9` in the failing direction.
- **R3.2.1.** `B_2(P_0)<=Phi(P_0)<1` initializes the induction; the same strategy's same actual pair supplies service and the strict margin at every reached epoch. R3.2 restores the redundant J.3 clause. No existential witnesses are conjoined after the fact.
- **L9.8.** A Q-fresh graded birth must contain both triggers. Collinear triggers at axis distance `d in {1,...,5}` share exactly `6-d` windows, each entering at count two with weight `1/9`; the sharp maximum is `5/9` at `d=1`.
- **R3.3.** R3.1 produces a nonterminal reached epoch with `Theta_2>=11/9`. Any account satisfying `Theta_2<=C<1` there is immediately contradictory. The corollary does not exclude non-dominating accounts, different thresholds, or non-threshold structural invariants.

### 9. ACCEPT — the Round-3 harness is the same construction and its quotient is complete

This was a code-read audit only; the historical test run was **not rerun**.

The test constructs the same eight translated blockers, the same three anchors and launches, the same blanket side/phase, and the same `Theta_2` account. The relevant empty-cell quotient has

`8*(6-2)+3*10+2 = 32+30+2 = 64`

representatives: every empty old-target cell, every launch-union cell, and two outside sentinels. It therefore checks

`C(64,2)=2,016`

distinct unordered pairs. This is complete for ordered legal replies because the two final Defender occupancies, old-label kills, and launch intersections are order-independent; every legal ordered pair has one of these unordered effect classes. Two distinct sentinels cover the case of two off-union cells. Admitting illegal representatives only enlarges Defender's tested action set and is a Defender-favoring over-approximation.

The exact arithmetic is also the same: `theta2_ab` stores `27*Theta_2`; every loop value here is rational `3(13-k)`, and the exact comparator checks it against 33. The minimum is 33. The fully legal attaining row

`D@(4,0), D@(213,0); A@(0,100), A@(1,100)`

kills two old targets, leaves six old count-two labels, creates five new count-two labels, and leaves the other `31-5=26` fresh pencil labels at count one. Independent occupancy enumeration gives the asserted final profile `(26,11,0,0,0,0)`.

### 10. ERRATUM — the “authoritative” Round-3 ledger is not a literal inventory of all labeled claims

Section 22 groups some claims transparently, but it omits several claims that the body explicitly labels: the normative-domain statement `[PROVEN by definition]`; the banked K1, K2, beta-floor, L1.1, and L1.2 group `[PROVEN at their round-2 scopes]`; and the separately named Route A and Route B `[OPEN]` items. The omission changes no mathematical status, but “authoritative status ledger” is incomplete if read literally.

Exact fix: add rows for those items, or add an explicit consolidation note identifying the existing rows that subsume each of them. Keep R3.2.1 separately visible or state that the L9.7/two-clause row intentionally groups it.

### 11. ERRATUM — Round-3 provenance records only the input commit

Section 23 records input `283348dce09d42b67e364e0b2f2b63166b6b5f4d` and says “No commit was created.” The reviewed documents and harness are committed at `06a1a649db6f3c7ec517d9835b129fead30d343d`.

Exact fix: retain `283348dc...` as the round's input/base commit, add `06a1a649...` as the reviewed/output artifact commit, and either delete the no-commit sentence or qualify it as “the authoring pass itself made no commit; the artifacts were committed afterward.” The header's phrase “at input HEAD” may remain.

### 12. ACCEPT — the strategy-stealing rule formalization matches production source

All files under `packages/hexo_engine/rust/src/*.rs` were read. No divergence from the formal model was found:

- `HexCoord` is an `i16` axial pair, and `hex_distance` computes `max(|dq|,|dr|,|dq+dr|)`.
- The three window vectors are `(1,0)`, `(0,1)`, and `(1,-1)`; `WINDOW_LEN=6`; a win requires all six cells for one player.
- `HexoState::new()` starts `Player0` at `Opening`; only `(0,0)` is legal; then `Player1` receives `FirstStone`.
- Every nonwinning ordinary turn is `FirstStone` then `SecondStone` by the same player, followed by the other player's `FirstStone`.
- Normal legality is emptiness plus membership in the color-blind closed radius-eight store. The store is updated by the first placement before the second placement is validated.
- A win is checked after each single placement for the current player. A first-placement win prevents the second placement. Terminal states expose no legal actions.
- `GameOutcome` has no draw variant, and the transition code contains no move-cap terminal branch.

The local-crate dependency and capsule path also check out. The document correctly distinguishes the live engine from auxiliary hunt/reference implementations.

### 13. ACCEPT — S1/S1.1 and the six-placement S2/S3 obstruction are exact

For `L(O)=N_8(O)\O` and `x notin O`, set algebra gives

`L(O union {x}) = (L(O)\{x}) union (N_8({x})\(O union {x}))`,

which is exactly S1. For S1.1,

`d((0,0),(-8,0))=8`, `min_{z in O} d((8,0),z)=16`, and `d((8,0),(0,0))=8`.

The inclusive radius predicate therefore makes `x` legal from `O`, makes `y` illegal from `O`, and makes `y` legal after adding `x`.

The S2 phase replay is:

`F opening; S First; S Second; F First; F Second; S First`.

All six coordinates are new. The five normal supports are successively at exact distance eight, including the within-turn support `(-8,0)->(-16,0)`. Through placement six each player owns only three stones, so neither can have a six. After deleting `x=(0,0)`, the four distances from `y=(8,0)` to `(-8,0),(-16,0),(-24,0),(-32,0)` are exactly `16,24,32,40`. S3 therefore proves a real normal move can become illegal under the identity/deletion projection.

### 14. ACCEPT — S4 is airtight at its stated scope, and the document does not exclude cleverer couplings

After `F@(0,0); S@a,S@b`, the real counts are `(F,S)=(1,2)` with F at `FirstStone`. Deleting only F's opening and swapping roles gives shadow counts `(opener=S,second=F)=(2,0)`, whereas the legal shadow prefix before the second player's first action has `(1,0)`. A fixed translation cannot change this count mismatch. More generally, after `k` completed real S turns at an F-`FirstStone` epoch, one-deletion projection gives `(2k,2k-2)` in shadow-role order, while the corresponding legal shadow epoch has `(2k-1,2k-2)`.

The most obvious delayed/translated repair was also attacked. On the S2 prefix through placement five, choose real `S@(-16,0)` as shadow opening, translate by `+(16,0)`, discard real `F@(0,0)` and the other real `S@(-8,0)`, and retain real `F@(-24,0),F@(-32,0)` as shadow `F@(-8,0),F@(-16,0)`. This repairs counts and phases and gives a legal shadow prefix. But the next real `S@(8,0)` maps to `(24,0)`, whose distances from retained shadow cells `0,-8,-16` are `24,32,40`. Its sole mapped radius-eight support is the discarded `F@(0,0)->(16,0)`. The repair of S4 therefore runs directly into S3.

This does not prove that every nonidentity shadow is impossible. Playing/discarding a second stone as a filler, dynamically translating, or encoding frontier-only moves would require a new phase-, legality-, and win-preserving invariant. Sections 4 and 6 expressly leave those possibilities open. The document claims only that the classical one-extra-stone identity/deletion coupling fails, so no outcome or universal-coupling overclaim was found.

### 15. ACCEPT — the unbounded-board idealization is disclosed and does not prop up a finite obstruction

The executable carrier is `i16`, not literal `Z^2`, and arithmetic at the carrier boundary is not modeled by the mathematical idealization. Section 1.2 says so explicitly. Every coordinate used by S1.1–S4 has absolute value at most 32 and is safely inside the carrier. Those PROVEN obstruction claims are finite and valid both in the idealized game and in the literal safe region.

Only the definition of an infinite draw and the OPEN `NL_F` target use the unbounded infinite-play idealization. No PROVEN stealing obstruction silently relies on boundedness or on unbounded play.

### 16. ERRATUM — S4 has an unmatched Markdown backtick

The displayed sentence in S4 ends:

> “`|X_F|=1`, `|X_S|=2`, with F to move at FirstStone.`”

Exact fix: write “`|X_F|=1`, `|X_S|=2`, with F to move at `FirstStone`.”

### 17. ERRATUM — the strategy-stealing provenance has the same stale output statement

Section 8 records only input `283348dc...` and says no commit was created, while the reviewed artifact is at `06a1a649...`.

Exact fix: record both input/base `283348dc...` and reviewed/output `06a1a649...`, and qualify or remove “No commit was created,” exactly as in Finding 11.

## Per-claim verdict — `GAP_RAW_PROOF_ROUND3.md`

This table splits claims that the source ledger groups. It also includes the VERIFIED regression so the machine-evidence boundary is explicit.

| Claim | Source status | Review verdict | First-principles disposition |
|---|---|---|---|
| GAP-RAW | OPEN | **ACCEPT** | Still open; crossing `Theta_2=1` with `I=empty` is not an Attacker win |
| Theorem R3.1, `8/9 -> >=11/9` | PROVEN | **ACCEPT** | Exact universal-reply construction, Findings 2–7 |
| Canonical Obligation J | REFUTED | **ACCEPT** | Every initial actual pair fails J.4 at `P_*` |
| Canonical GAP-GLOBAL-RENEWAL | REFUTED | **ACCEPT** | Same universal first-epoch counterexample |
| GAP-AMORTIZED-ABANDONMENT | OPEN | **ACCEPT** | No formal credit/refund replacement is proved or refuted |
| L9.1 one-pair isolator | PROVEN | **ACCEPT** | All 31 windows independently audited |
| L9.2 eight-gadget profile | PROVEN | **ACCEPT** | Exact `(0,8,0,0,0,0)` profile |
| L9.3 launch separation and legality | PROVEN | **ERRATUM** | Geometry is sound; “25 empty cells” must be 24 intervening cells |
| L9.4 normative-root audit | PROVEN | **ACCEPT** | Exact account, strict threshold, and nonterminality |
| L9.5 legal nonterminal response | PROVEN | **ACCEPT** | Inclusive radius eight, sequential phase, no six |
| L9.6 canonical debt bound | PROVEN | **ACCEPT** | `B_2>=|I|/3>=tau/3` |
| R3.2 J.3 redundancy | PROVEN | **ACCEPT** | J.2 + L1.1 + J.4 + L9.6 |
| L9.7 strict renewal margin | PROVEN | **ACCEPT** | Exact four-transition equivalence |
| R3.2.1 two-clause canonical J | PROVEN | **ACCEPT** | Same strategy and same actual pair remain bound |
| L9.8 fresh-pair ceiling | PROVEN | **ACCEPT** | Exact `6-d` common-window count; sharp `5/9` |
| R3.3 no pointwise-dominating subunit account | PROVEN | **ACCEPT** | Immediate contradiction at the R3.1 successor |
| K1, retained Round-2 scope | PROVEN | **ACCEPT** | Not implicated; `tau(P_*)=0` |
| K2, retained Round-2 scope | PROVEN | **ACCEPT** | Not implicated; `tau(P_*)=0` |
| Beta ripe-witness floor, retained | PROVEN | **ACCEPT** | Successor has no imminent label |
| L1.1 completion criterion, retained | PROVEN | **ACCEPT** | Used with its exact nonterminal Attacker-`FirstStone` premises |
| L1.2 service criterion, retained | PROVEN | **ACCEPT** | Unchanged Round-2 scope |
| Theorem A2, retained | PROVEN | **ACCEPT** | Exact Service target remains necessary and sufficient |
| Theorem A2-prime, retained | PROVEN | **ACCEPT** | No premise or proof changed |
| General standalone K3 suppression | OPEN | **ACCEPT** | R3.2 does not solve the independent geometry |
| Theorem D2, `J=>GAP-RAW` | PROVEN | **ACCEPT** | The implication stays valid despite its false antecedent |
| Normative-domain statement | PROVEN by definition | **ACCEPT** | Round 2 admits non-engine-reachable roots; `P_*` is one |
| Route A, direct Service construction | OPEN | **ACCEPT** | No strategy is supplied |
| Route B, non-dominating structural account | OPEN | **ACCEPT** | R3.3 leaves this class open |
| GAP-REPLACEMENT-INVARIANT | OPEN | **ACCEPT** | Correct successor problem; no solution claimed |
| Round-3 coordinate/quotient regression | VERIFIED | **ACCEPT** | Code matches proof; 2,016-class quotient; not rerun |

## Per-claim verdict — `STRATEGY_STEALING_HEXO.md`

Repeated disposition labels and same-named section/lemma labels are consolidated below; every distinct labeled proposition is named. The three `[GAP]` rows are included even though they are not PROVEN/REFUTED/OPEN labels.

| Claim | Source status | Review verdict | First-principles disposition |
|---|---|---|---|
| Formal rule model in §2 / implemented-rules / Rules verdict | PROVEN | **ACCEPT** | Exact production rule model on the safe carrier region |
| Ordinary delete-extra-stone / Stealing verdict | PROVEN | **ACCEPT** | S3 breaks frontier projection and S4 breaks direct opening alignment |
| Non-loss target / `NL_F` | OPEN | **ACCEPT** | Neither player's outcome strategy is constructed |
| Local `hexo_engine` crate is authoritative | PROVEN | **ACCEPT** | Workspace dependency and live-state capsule checked |
| Coordinate-carrier caveat | PROVEN | **ACCEPT** | `i16` carrier versus `Z^2` idealization is stated honestly |
| Board, distance, and windows | PROVEN | **ACCEPT** | Formula, axes, and length-six win predicate match source |
| State and phase | PROVEN | **ACCEPT** | Singleton opening followed by sequential pairs |
| Normal legality and within-turn growth | PROVEN | **ACCEPT** | Empty plus inclusive color-blind radius eight; first stone updates frontier |
| Winning, termination, and non-loss model | PROVEN | **ACCEPT** | Either color wins after one placement; infinite draw is explicitly meta-level |
| Classical deletion-shadow prerequisites | PROVEN | **ACCEPT** | Correctly identifies the projection lemma the classical identity coupling needs |
| S1 color-blind frontier update | PROVEN | **ACCEPT** | Exact set identity |
| S1.1 strict new frontier | PROVEN | **ACCEPT** | Distances `8,16,8` |
| Reachable six-placement prefix / S2 | PROVEN | **ACCEPT** | Owners, phases, supports, and nonterminality replay exactly |
| S3 deletion-monotonicity failure | PROVEN | **ACCEPT** | Post-deletion distances `16,24,32,40` |
| Consequence for classical proof / unchanged deletion shadow invalid | PROVEN | **ACCEPT** | Narrow identity/deletion conclusion only |
| S4 singleton/pair cadence mismatch | PROVEN | **ACCEPT** | `(2,0)` cannot be the required `(1,0)` shadow prefix |
| GAP-OPENING-ALIGNMENT | GAP | **ACCEPT** | A new coupling must account for both first-turn S stones |
| GAP-FRONTIER-COUPLING | GAP | **ACCEPT** | Nonidentity mapping remains unconstructed |
| GAP-NONLOSS-DETERMINACY | GAP | **ACCEPT** | Logical bridge or direct F strategy remains unproved |

There are no REFUTED or VERIFIED proposition labels in the strategy-stealing document.

## Final document verdicts and mandatory fold-in edits

| Document | Verdict | Why |
|---|---|---|
| `GAP_RAW_PROOF_ROUND3.md` | **SOUND-WITH-ERRATA** | The canonical J refutation and every new PROVEN sharpening survive; separation prose, ledger completeness, and output provenance need edits |
| `STRATEGY_STEALING_HEXO.md` | **SOUND-WITH-ERRATA** | The production-rule model and narrowly scoped classical-coupling obstruction survive; one Markdown typo and output provenance need edits |

A fold-in pass must make exactly these changes:

1. In Round-3 L9.3, change “25 empty cells” to “closest cells at distance 25, with 24 intervening cells.”
2. In the Round-3 status ledger, add or explicitly consolidate the normative-domain claim, retained banked claims, R3.2.1, and Routes A/B so “authoritative” is literal.
3. In both provenance sections, distinguish input/base `283348dc...` from reviewed/output `06a1a649...`; remove or qualify “No commit was created.”
4. In strategy-stealing S4, move the stray backtick so the token is `` `FirstStone` ``.

No theorem-status downgrade, production-source change, harness change, or new machine run is required.
