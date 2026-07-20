# Hostile review of the claimed T3 tightenings — round 1

| Claim under review | Verdict | Short ruling |
| --- | --- | --- |
| Remove Z1 from T3/T4 | **CONFIRMED** | Neither T3 nor any lemma it invokes uses Z1, directly or indirectly. T4 may omit the hitting term, while D9's nonempty searched-child fallback remains mandatory. |
| Replace exact global \(D_N\) by admissible local \(B(N)\) | **CONFIRMED** | D9's grammar makes the stated recurrence placement-exact. T3 needs only an integral upper bound that includes the current AND edge and LOSS remainder and decreases hereditarily. |
| Replace \(8D_N\) by \(8(B(N)-1)\) | **CONFIRMED-WITH-REPAIR** | The numerical bound and off-by-one are correct, but current L9 and D12(iv) cannot merely have their radius edited. They must be replaced by a first-real-only-protected-occupation lemma. |
| Replace full witness windows by witness empty sets | **CONFIRMED** | Shared attacker stones plus protected leaf empties partition every A-alive witness; a \(Y\)-cell cannot occur in such a window. The universal MI anchor does not require full-window protection. |
| Pathwise rather than merely global-\(T\) conclusion | **CONFIRMED** | D9's path clock and the three T3 terminal transfers already give the selected path's declared resolution, unless the real attacker wins earlier. |
| T6 extendable-hit kernel, with no original core | **CONFIRMED-WITH-REPAIR** | The kernel proof and removal of core work when every internal AND node has \(\operatorname{mhs}\le b\). As stated, \(\operatorname{mhs}>b\) makes \(K_b=\varnothing\), violating D9. |
| LOSS witness families of size at most \(3/6\) | **CONFIRMED** | The rank-two transversal argument is correct, and D9's adaptive remainder counts \(b\) defender placements, not turns. |
| Per-window exposure, \(Z_{\rm touch}\), and virgin radius \(8(E_N^D(W)-6)\) | **CONFIRMED** | The recurrence is clock-correct, touched windows are already legal, and a first-ever dismissed fill of a virgin window yields the claimed six-fill tempo bound even for mixed chains. |
| Branch-indexed substitute children | **CONFIRMED-WITH-REPAIR** | The structural substitution is sound only with a current-transition-inclusive envelope. Literal child-only budgets and radii miss the real move \(d\) and admit concrete C2/C3 failures. |
| Full \(F+H_W\) forced-hit accounting remains open | **CONFIRMED** | None of the other tightenings debits compulsory hits from a window's available defender-placement budget; normative §12.1 remains unresolved. |
| Cell-specific deadlines \(r_N(y)\) (mandatory §3 check) | **CONFIRMED-WITH-REPAIR** | The cellwise chain proof works and LOSS empties need protection only through leaf entry, but ranks must be attached to live obligation occurrences and include OR-COMPLETION moves. |
| Internal-AND \(\neg\)own_win_now redundancy (§7 cleanup) | **CONFIRMED** | Under the completion zone and D9's ban on defender-terminal edges, count-5 and count-4/\(b=2\) nodes cannot be internal. This does not remove the LOSS-leaf check or T6's explicit hypothesis. |
| Finite acyclic certificate-DAG extension (§7 cleanup) | **CONFIRMED** | A finite DAG with one exact D9 label and one consistent clock per shared node can be finitely unfolded to a D9 tree; reachable-descendant obligation unions preserve nesting. |

## 1. Review basis and actual T3 dependency trace

The [normative document](PROOF_TSS_DEFENDER_ZONES.md) was read in full before the [claims document](_T3_TIGHTENINGS_REVIEW_CLAIMS.md). The operative source is D1–D13 and L1–L10, especially D9–D12 and T3's Step O, Step A, completion anchor, and leaf transfer. Historical review-log statements are not used as proof substitutes.

Here is every operative dependency in T3.

| T3 proof part | Actual source obligations used |
| --- | --- |
| Coupling setup | D12's synchronized clock/mover/budget, identical attacker stones, canonical \(X/Y\), and \(X\cap\operatorname{Prot}=\varnothing\); protection monotonicity |
| Step O: emptiness | D10/D11 put the designated attacker cell in ancestor protection; D12(iii) excludes \(X\); ghost emptiness excludes \(Y\) |
| Step O: legality | Z4 supplies a shared attacker/root legality witness; D4 transfers that witness |
| Step O: completion | D9 supplies the named OR witness and clock; D10/D12 exclude \(X\) from it; A-completeness excludes \(Y\); L5(b) transfers its mask; L5(d)/L6(b) justify immediate success if the real game completes another window earlier |
| Filler, A1, A2 | D9's \(S(N)\ne\varnothing\), legal exact successors, and no defender-terminal edge; D12's canonical difference updates |
| A3: protected occupancy | Z2 for a ghost-legal cell; L9(b), hence Z5 plus protection monotonicity, for a real-legal/ghost-illegal cell |
| A3: frontier invariant | Z5 directly for a ghost-legal dismissal and L9(a) for later ghost-illegal links |
| Defender-completion anchor | D12's universal MI identity; D9's exact line and no-defender-terminal rule; L4 backward D-aliveness; D11's completion guard; D7 clock synchronization; L7 counting |
| WIN leaf | D9's named own_win_now witness and resolution; D10/D12 for \(X\)-avoidance; ghost A-aliveness for \(Y\)-avoidance; L5(b); L1 and shared A stones for legality |
| LOSS leaf | D9's named family, \(\tau>b\), leaf \(\neg\)own_win_now, adaptive remainder, and clock; equal masks from core/\(X/Y\); leaf-time MI plus the completion anchor for real \(\neg\)own_win_now; L1 and permanence for the completion |

Z1 occurs in none of these rows. The indirect checks also close:

- L9 uses Z5, protection monotonicity, D4's radius-8 legality, and the synchronized defender-placement clock. It does not use Z1.
- The completion anchor uses the D-completion component of protection, not current A-threat hitting cells.
- The LOSS contract uses its named leaf family. A current threat at an ancestor may be absent from that family without affecting the adaptive leaf proof.
- The coupling's filler is well-defined because D9 requires \(S(N)\ne\varnothing\), not because Z1 supplies a hitting cell.

This confirms the claimant's central observation: Z1 is conservative slack in normative T3/T4, not a hidden support for another cited lemma.

## 2. Remove Z1 from T3/T4 — CONFIRMED

At a real dismissal of a current threat hit, one of two things happens. If the ghost continuation later relies on that threat, its completion move or leaf witness empties belong to the protected obligation set and Z2 prevents the relevant blocker from being dismissed. If the continuation never relies on the threat, blocking it in the real game is immaterial to the selected certificate proof. That is precisely why no T3 step needs all current hitting cells.

The repaired T4 zone may therefore omit \(\operatorname{hitting}(P_N)\). It must retain:

1. direct protection of legal obligations and dangerous D-window empties;
2. the applicable frontier/seed guards;
3. Z4 for attacker legality; and
4. an arbitrary legal searched fallback whenever the resulting zone is empty, because D9 and the A2/A3 filler require \(S(N)\ne\varnothing\).

This ruling does not remove hitting cells from the separate T6 kernel analysis; T6 uses the current threat family to build its auxiliary refutation.

## 3. Local defender budget \(B(Q)\) — CONFIRMED

The proposed recurrence matches D9's placement grammar:

\[
\begin{aligned}
B(\text{OR-COMPLETION})&=0, &
B(\text{WIN})&=0,\\
B(\text{LOSS},b)&=b, &
B(\text{OR})&=B(\text{child}),\\
B(\text{AND})&=1+\max_C B(C).
\end{aligned}
\]

D3/D4 make the unit explicit: one AND edge is one defender placement. A defender turn with two stones is represented by two successive defender nodes, first with \(b=2\), then with \(b=1\). At a LOSS leaf, D9's continuation has exactly the remaining \(b\) defender placements unless the game ends earlier. Thus \(b\) is placements, not turns.

Only three properties of normative \(D_N=\mathfrak D(P_N,T)\) are used:

- **Hereditary decrease.** If descendant \(M\) is reached after \(k\) defender placements, \(B(M)+k\le B(N)\).
- **Completion monotonicity.** Consequently
  \[
  \operatorname{cnt}_D(W,P_M)+B(M)
  \le \operatorname{cnt}_D(W,P_N)+B(N).
  \]
- **Anchor coverage.** Starting with a dismissal at \(N\), the selected certificate path, including a LOSS remainder, contains at most \(B(N)\) real defender placements before its declared attacker resolution.

The proof never needs equal budget values on sibling nodes. It needs a maximum over children at their parent so that whichever filler child is selected is covered. Therefore an exact local maximum is optimal for pruning but not necessary for soundness. A verifier may accept any nonnegative integral upper bound satisfying the leaf lower bounds, \(B(N)\ge1+B(C)\) on every AND edge, the corresponding OR inequality, and hereditary decrease.

## 4. Radius \(8(B-1)\) — CONFIRMED-WITH-REPAIR

### Off-by-one and turn accounting

Normative L9 proves the stronger statement that a protected ghost-illegal cell never even becomes real-legal. To make a target \(y\) legal, the last already-placed \(X\)-stone may sit another radius-8 hop away, which explains L9's \(8D\).

T3 only needs to prevent \(y\) from becoming a real-only occupied protected cell. Let \(x_0\) be the first ghost-legal dismissed stone in the causal legality chain and let \(y\) be the \(p\)-th defender placement from \(x_0\), counting \(x_0\). Before \(y\) is placed there are at most \(p-1\) radius-8 links, so D4 gives

\[
d(x_0,y)\le 8(p-1)\le8(B(N_{x_0})-1).
\]

The two placements of a defender turn do not form a radius-16 jump. D4 processes them as two plies and checks legality separately; each can add only one radius-8 link. Equality is attainable by \(x_0\) followed by \(B-1\) successive distance-8 placements, with the target itself last.

### Exact proof repair

Changing \(8D\) to \(8(B-1)\) inside current L9 would make L9(a)/(b) false, because a protected target can become legal on the penultimate event without yet being occupied. Replace L9 and D12(iv) by this weaker lemma:

> **First protected-occupation lemma.** Suppose direct legal protected cells are searched and every ghost-legal dismissal \(x\) at \(N\) lies outside radius \(8(B(N)-1)\) of
> \[
> \operatorname{Prot}(N)\setminus
> (\operatorname{Legal}(P_N)\cup\operatorname{Stones}(P_N)).
> \]
> Then no defender placement creates a real-only stone in the current protected set.

Proof: assume the first violation is \(y\). If \(y\) is ghost-legal, Z2 makes it searched. Otherwise trace real-only legality witnesses backward through current \(X\)-stones to the first ghost-legal dismissal \(x_0\). Protection monotonicity puts descendant \(y\) in \(\operatorname{Prot}(N_{x_0})\). Because ghost legality is monotone for an unoccupied cell, \(y\) was ghost-illegal at \(N_{x_0}\); permanence makes it a non-stone there. The displayed distance inequality puts \(x_0\) in the guarded band, contradicting its dismissal.

D12 should retain invariants (i)–(iii), especially \(X\cap\operatorname{Prot}=\varnothing\), but delete the assertion that every \(X\)-stone is L9-clear. Step A3 then invokes the new lemma when \(d\) is ghost-illegal. No other live T3 user needs L9's stronger “never becomes legal” conclusion: the anchor and both leaf transfers need only \(X\cap\operatorname{Prot}=\varnothing\).

## 5. Obligation compression — CONFIRMED

Let the obligation set contain every future certificate attacker placement, including OR-COMPLETION placements and leaf continuations, plus \(E(W,P_L)\) for every named WIN/LOSS witness at its leaf.

For a WIN/LOSS witness \(W\):

- every ghost A-stone in \(W\) is a real A-stone because D12 keeps attacker stones identical;
- every other cell of \(W\) is in the protected leaf empty set, so no \(X\)-stone can occupy it;
- \(W\) is A-alive in the ghost, so it contains no ghost defender stone and hence no \(Y\)-cell.

Thus \(X\cap W=Y\cap W=\varnothing\), and the complete window masks and empty sets agree even though only leaf empties were explicitly protected. A real-only attacker stone cannot exist under D12.

For an OR-COMPLETION leaf, immediately before the designated move the other five window cells are shared attacker stones. Protecting the designated placement cell is sufficient; ghost emptiness rules out \(Y\) there, and protection rules out \(X\).

The defender-completion anchor's MI formula is a canonical identity for every window:

\[
\operatorname{cnt}_D(W,R)
=\operatorname{cnt}_D(W,G)+|X\cap W|-|Y\cap W|.
\]

It does not assume full-window agreement for named attacker witnesses. The anchor continues to use all current empties of a D-alive window that passes its completion budget test. OR-COMPLETION transfer, WIN transfer, and LOSS leaf-entry equality therefore all survive the compression.

## 6. Pathwise conclusion — CONFIRMED

D9 supplies exact successors, a path-derived clock, a finite typed maximal node, and a declared resolution at each terminal outcome. T3 already establishes:

- immediate success if the real attacker completes another window before the ghost;
- completion on the mapped OR-COMPLETION ply;
- completion no later than a mapped WIN leaf's declared ply; or
- completion no later than \(\text{leaf-ply}+b+2\) at a mapped LOSS leaf.

Unless an earlier real attacker win terminates the game, the coupling therefore descends a finite certificate path and wins by that path's declared resolution. Normative \(T\) is only the maximum of those resolutions. With local \(B\), the defender-completion anchor is run through the selected path's resolution rather than through an unnecessarily later global maximum.

## 7. Cell-specific deadlines \(r_N(y)\) — CONFIRMED-WITH-REPAIR

The same first-protected-occupation proof works one target at a time. If \(y\) is still live and descendant \(M\) is reached after \(k\) defender placements, require

\[
r_M(y)+k\le r_N(y).
\]

Directly search \(y\) when it is ghost-legal. When it is ghost-illegal, guard each ghost-legal seed within radius \(8(r_N(y)-1)\). If \(y\) were the \(p\)-th defender placement from such a seed before its deadline, \(p\le r_N(y)\) gives the same contradiction as §4.

The claimant's LOSS deadline is correct. At leaf entry the selected witness empty sets must agree and have hitting number \(>b\). After entry the adaptive contract intentionally permits the defender to occupy witness empties. For every remainder \(H\) of at most \(b\) defender placements, \(\tau>b\) supplies a witness \(W_H\) with \(E(W_H)\cap H=\varnothing\); only that surviving witness is completed. No \(X/Y\) coupling or protection is needed after leaf entry.

The draft nevertheless needs these exact formal repairs:

1. Attach ranks to live obligation **occurrences or roles**, not only bare cells. If one cell has several roles or branch deadlines, use the maximum live rank.
2. Include every designated attacker move, expressly including the move stored in an OR-COMPLETION leaf; “internal attacker move” is too narrow.
3. Treat WIN/LOSS continuation cells through their witness-empty role, whose deadline is leaf entry.
4. Apply the band only at internal AND nodes while \(r_N(y)\ge1\). Drop the occurrence when its deadline is reached; do not form a negative radius at \(r=0\).
5. Keep defender-completion windows on their separate \(B\) or \(E^D\) clocks.

## 8. Per-window exposure and the virgin-window zone — CONFIRMED

### Clock recurrence and permanence

The recurrence for \(E_N^D(W)\) is consistent with D9:

- an internal AND edge consumes exactly one defender placement;
- an ordinary OR edge consumes none;
- a LOSS leaf contributes its remaining \(b\) defender placements;
- a WIN/OR-COMPLETION leaf contributes none; and
- if the current attacker placement lies in \(W\), \(W\) is permanently unavailable for defender completion.

The last statement means “permanently non-D-alive,” not necessarily “dead” in D5's two-colour sense. D3/D4 permanence and L4 make it sufficient. An overlapping window \(W'\) is a distinct object and retains its own recurrence and guard.

Before an attacker enters \(W\), the recurrence gives the hereditary inequality

\[
E_M^D(W)+k\le E_N^D(W)
\]

after \(k\) defender placements along a descendant path.

### Touched windows

If a D-alive \(W\) has at least one defender stone, L1/F1 puts every empty of \(W\) within distance at most 5 of that stone. Every empty is already legal under D4, so there is no C3 frontier issue for a fill of \(W\).

Moreover, after \(k\) descendant defender placements,

\[
\operatorname{cnt}_D(W,P_M)+E_M^D(W)
\le \operatorname{cnt}_D(W,P_N)+E_N^D(W).
\]

Hence a touched window cannot first become completion-dangerous only after descending. Searching its empties whenever the right side is at least 6 is sufficient.

### Virgin windows, including mixed chains

Assume for contradiction that the real defender first completes a fixed \(W\) before the certificate attacker wins or enters \(W\).

If no ever-dismissed cell of \(W\) exists, MI gives
\(\operatorname{cnt}_D(W,R)\le\operatorname{cnt}_D(W,G)\), so the ghost certificate line also has a defender completion, contradicting D9.

Otherwise take the first ever-dismissed real-only fill \(x\in W\).

- If the ghost already has a defender stone in \(W\), \(x\) is ghost-legal. Before \(x\), real count in \(W\) is at most ghost count by MI. The real placements from \(x\) through completion are at most \(E_N^D(W)\), so
  \(\operatorname{cnt}_D(W,P_N)+E_N^D(W)\ge6\). Thus \(Z_{\rm touch}\) searches \(x\).
- If the ghost window is virgin and \(x\) is ghost-legal, \(d(x,W)=0\) and six W-fills remain in the exposure count, so \(E_N^D(W)\ge6\); \(Z_{\rm virgin}\) searches \(x\).
- If \(x\) is ghost-illegal, trace its current \(X\)-legality witnesses to the first ghost-legal dismissed seed \(x_0\). If the chain has \(j\) radius-8 links before reaching \(W\), then \(d(x_0,W)\le8j\). From the virgin state, completion still consumes all six W placements. Counting \(x_0\), the chain, and those fills gives at least \(j+6\) defender placements before the exposure stop. Therefore
  \[
  E_{N_{x_0}}^D(W)\ge j+6,\qquad
  d(x_0,W)\le8(E_{N_{x_0}}^D(W)-6),
  \]
  so the seed was in \(Z_{\rm virgin}\).

This proof covers the requested mixed cases:

- a certificate attacker or shared defender stone near \(W\) makes a later W-fill ghost-legal, so it is checked directly;
- a searched defender placement in \(W\) makes the descendant window touched, where the monotone touched guard applies;
- multiple independent dismissed seeds reduce to the causal chain supporting the first dismissed W-fill; unrelated placements only consume exposure budget;
- interleaving across searched nodes does not change the one-placement clock; and
- each overlapping candidate completion window has its own first-dismissal anchor.

The obligation zone must separately ensure that the certificate's attacker-in-\(W\) stopping move remains playable in the real game. With that companion condition, the exposure stop is legitimate. The radius is sharp at this level: with \(E=7\), a legal seed at distance 8 can be followed by the first W-fill and the five remaining fills.

## 9. T6 extendable-hit kernel — CONFIRMED-WITH-REPAIR

Let \(\mathcal F\) be the current threat-empty family and

\[
K_b=\{d\in\operatorname{Legal}(P_N):
\tau(\mathcal F\setminus d)\le b-1\}.
\]

Assume first that an internal AND node has \(\tau(\mathcal F)\le b\).

- If \(\tau<b\), deleting the sets hit by any \(d\) cannot increase the transversal number, so every legal \(d\) lies in \(K_b\). There is no pruning.
- If \(\tau=b\) and \(d\notin K_b\), then for \(b=2\) the successor has defender budget 1 and a residual family with \(\tau>1\), giving the adaptive LOSS refutation. For \(b=1\), at least one threat survives and the attacker, now with budget 2, has a WIN refutation.

The defender cannot win first. At a \(b=2\) node, \(\neg\)own_win_now bounds every D-alive window by count 3. The defender can indeed place twice in that turn and jump from 3 to 5, but not to 6. At \(b=1\), the bound is count 4 and the one placement reaches at most 5.

The same-\(T\) argument survives. When \(\tau=b\), a minimum hitting set can be followed through \(K_b\): at \(b=2\), its first member leaves a family hit by the second; at \(b=1\), its sole member is common to all threats. Those searched moves kill every current A-threat, and defender stones cannot create a new A-threat. Therefore the first following attacker placement cannot complete a window: any such window would already have been a current count-5 threat and would have been killed. The original path's clock reaches the second attacker placement required by the auxiliary refutation.

The original core term is genuinely unnecessary in this proof. Before the first reply outside \(K_b\), real play follows searched certificate edges exactly, so \(X=Y=\varnothing\). At that first dismissal the proof abandons the original subtree and uses the current residual threat family directly.

The unqualified statement is false at \(\tau>b\). If \(d\in K_b\), then \(d\) together with a transversal of the residual family of size at most \(b-1\) would hit all of \(\mathcal F\), implying \(\tau\le b\). Hence \(\tau>b\) forces \(K_b=\varnothing\), contrary to D9's nonempty searched set. Three disjoint singleton threat empties at \(b=2\), or two at \(b=1\), exhibit the defect.

Exact repair: require \(\operatorname{mhs}\le b\) at every internal AND node governed by the kernel. A node with \(\operatorname{mhs}>b\) must either remain under a nonempty existing subtree or be converted to a valid D9 LOSS leaf whose conservative \(\text{leaf-ply}+b+2\) clock still fits \(T\). Also replace “strictly refines” by “weakly refines and can strictly reduce”: some forcing families have \(K_b=\operatorname{hitting}(P_N)\).

T6 and T6+ must retain explicit internal \(\neg\)own_win_now unless a completion zone is also imposed; the kernel alone does not search defender-winning cells.

## 10. LOSS-witness sparsification — CONFIRMED

Every nonterminal A-threat has one or two empties.

For \(b=1\), let \(\tau(\mathcal F)>1\). If \(\mathcal F\) contains a singleton \(\{a\}\), choose one set missing \(a\); the two sets have no one-point transversal. Otherwise choose \(E=\{a,b\}\), one set missing \(a\), and one set missing \(b\). Any point hitting \(E\) is \(a\) or \(b\), and the corresponding selected set defeats it. Thus at most three sets suffice. The triangle of two-element sets is sharp.

For \(b=2\), take a maximal pairwise-disjoint subfamily. It cannot have size 1, because its sole set, of size at most 2, would hit the whole family. If it has at least 3, three disjoint sets already force \(\tau>2\). Otherwise take disjoint \(E_1,E_2\). Their two-point transversals are the at most four cross-pairs choosing one element from each. For each cross-pair choose one original set it misses. Together with \(E_1,E_2\), at most six sets exclude every two-point transversal. The six edges of \(K_4\) give the general rank-two sharp example.

D9's LOSS contract quantifies over the remaining \(b\) defender **placements**. Any actual remainder \(H\) has \(|H|\le b\) and therefore cannot hit a selected subfamily with transversal number \(>b\). One named witness survives, exactly as in T3's adaptive leaf transfer. The proof never needs threats outside the selected subfamily.

## 11. Branch-indexed substitution — CONFIRMED-WITH-REPAIR

The nesting idea is valid: once dismissal \(d\) selects substitute child \(C_s\), earlier \(X\)-stones need only have been certified against the full reachable subtree of the previously selected child. Every later selected subtree is nested inside it. A1 may follow any searched real reply inside the current subtree; A2 introduces no new \(X\); and a later ghost-illegal A3 is controlled by the envelope of the earlier ghost-legal seed from which its frontier chain descends.

The claims document, however, says to compute budgets and tests “only for the subtree rooted at \(C_s\).” Taken literally, that omits the current real placement \(d\).

### Concrete C3 failure of a child-only deadline

Let \(N\) be a \(b=2\) defender node. The real defender plays a currently legal dismissal \(d\), while the ghost substitute \(s\) is elsewhere. Let \(y\) be ghost-illegal with \(d(d,y)=8\). In \(C_s\), one defender placement remains before an attacker sequence that first plays a legal setup \(a\) and then designates \(y\); \(a\) supplies Z4 for the attacker move. Thus the child rank is \(r_{C_s}(y)=1\).

A literal child-only band uses \(8(r_{C_s}(y)-1)=0\) and permits \(d\). In the real game \(d\) legalizes \(y\), and the defender uses the remaining placement on \(y\); the ghost uses its searched reply elsewhere. The later designated attacker move is blocked. The parent must charge the current transition, giving radius \(8r_{C_s}(y)=8\).

### Concrete C2 failure of a child-only completion budget

Let a D-alive \(W\) have two defender stones at \(N\), with \(d\in W\) and substitute \(s\notin W\). Suppose \(C_s\) has three further defender placements before resolution. A child-only test sees \(2+B(C_s)=5\) and does not protect \(d\). The real current \(d\), plus those three later W-fills, completes the remaining four cells. The parent-inclusive test sees

\[
2+\bigl(1+B(C_s)\bigr)=6
\]

and correctly forbids the dismissal.

### Exact substitute-envelope repair

For every ghost-legal dismissal \(d\) and named \(s=\phi_N(d)\):

1. Define the transition budget
   \(\widehat B(N,d,s)=1+B(C_s)\), including the current real/ghost defender ply and every LOSS remainder.
2. Use obligations equal to the union over **all reachable descendants** of \(C_s\), not one leaf or one chosen continuation.
3. Require \(d\) itself to avoid those obligation cells and all transition-dangerous completion empties.
4. For a child obligation with rank \(r_{C_s}(y)\), use the parent seed radius
   \(8r_{C_s}(y)=8((1+r_{C_s}(y))-1)\).
5. For completion, test
   \(\operatorname{cnt}_D(W,P_N)+1+B(C_s)\ge6\), or use the analogous transition-aware exposure \(1+E^D_{C_s}(W)\).
6. Preserve an independently nonempty searched fallback \(S(N)\). The rule “search only replies with no safe substitute” is otherwise circular because substitutes must already belong to \(S(N)\).
7. State the transitions: A3 uses \(\phi_N(d)\); A2 may use any searched filler because it creates no \(X\); ghost-illegal A3 uses the inherited earlier envelope.
8. Require the selected envelope to protect LOSS witness empties through leaf entry and count the leaf's \(b\) placements for defender-completion/own-win exclusion.

With these changes, the clock remains synchronized—real \(d\) and ghost \(s\) consume the same single D4 ply—and the normative OR/WIN/LOSS transfers apply inside the selected reachable subtree. The simpler default-child variant needs the same transition-inclusive repair; merely substituting \(\operatorname{core}(\mathcal C,f(N))\) for the whole core is not sufficient.

## 12. Full \(F+H_W\) accounting — CONFIRMED STILL OPEN

Normative §12.1 correctly says that charging quiet placements plus per-window forced-hit capacity requires branchwise worst-case bookkeeping. Local \(B\) only removes delay caused by unrelated certificate branches. \(E_N^D(W)\) stops the clock after the attacker kills \(W\), but still counts every earlier defender placement. Branch-indexed substitution restricts the reachable subtree, but likewise does not prove that compulsory threat hits are unavailable for filling \(W\).

Therefore none of the reviewed claims proves the proposed \(F+H_W\) debit. The claimant's “still open” row is accurate.

## 13. Section-7 cleanup: internal own_win_now — CONFIRMED

This redundancy is conditional on the completion-zone requirement and D9's prohibition of defender-terminal searched edges.

- If an internal AND node has a D-alive count-5 window, its sole empty is legal by L1, at least one defender placement remains before resolution, and the completion guard forces that empty into \(S(N)\). Its exact successor is defender-terminal, forbidden by D9.
- If \(b=2\) and a D-alive window has count 4, both empties are legal and at least two defender placements occur before the attacker can move. Both are forced into the first searched set. After either first fill, the \(b=1\) child has count 5. It cannot be a LOSS leaf because the leaf check is retained; if internal, its last empty is forced searched and yields a forbidden terminal edge.

Thus an otherwise valid completion-zoned certificate cannot contain such an internal node, even without an explicit internal check. Keep the check as a diagnostic. It remains logically necessary at LOSS leaves, where no searched set is expanded. It also remains an explicit premise of T6/T6+, whose kernel has no completion guard.

## 14. Section-7 cleanup: finite DAGs — CONFIRMED

Require a finite acyclic graph in which each shared node has one exact D9 label: the same position, mover/budget, designated OR action or AND searched-successor map, leaf witness data, and one consistent path clock. Unfold the DAG along root-to-node paths. A finite acyclic graph has finitely many such paths, so the unfolding is a finite D9 tree.

Define core/obligations at a DAG node as the union over all reachable descendants. Reachability is nested along an edge, so protection monotonicity survives. The \(X/Y/\widehat X\) history remains path-local in the unfolding. T3 then applies verbatim. A direct topological induction is equivalent but must quantify over every coupling history reaching a merged node.

## 15. Summary rulings

### (a) Does any claim expose an error in normative T3/T4/T6?

**No.** The claims expose conservative hypotheses and nonminimal zones, not an unsound inference:

- T3 remains correct with its larger core, global \(D_N\), radius \(8D_N\), and redundant Z1.
- T4 states a sufficient zone, not a minimal one.
- T6's hitting-plus-core set remains sufficient; the tighter kernel needs an additional internal-node scope condition.

The false literal variants found here—editing L9's radius without changing its conclusion, applying T6's kernel at \(\operatorname{mhs}>b\), or using child-only substitute budgets—are proposals in the claims document, not defects in the normative proof.

### (b) Certification split

**Certifiable at PROVEN quality now:**

- remove Z1 from T3/T4;
- replace global \(D_N\) by an admissible hereditary local \(B\);
- compress named witness protection to future attacker cells plus leaf witness empties;
- state the pathwise resolution conclusion;
- sparsify LOSS witnesses to \(3/6\);
- use per-window exposure, \(Z_{\rm touch}\), and virgin radius \(8(E^D-6)\);
- remove the internal own_win_now check under the stated completion-zone scope; and
- extend to a finite consistently labelled, consistently clocked acyclic DAG.

The statement that full \(F+H_W\) accounting remains open is also confirmed, but it is an open-status ruling rather than a new theorem.

**Needs one repair round before PROVEN tagging:**

- \(8(B-1)\): install the first-protected-occupation lemma and revise D12/A3;
- cell-specific deadlines: formalize live occurrences, maxima, OR-COMPLETION moves, and rank-zero handling;
- T6 kernel/no-core: restrict kernel-governed internal nodes to \(\operatorname{mhs}\le b\) and specify treatment of \(\operatorname{mhs}>b\); and
- branch-indexed substitution: install transition-inclusive budgets/radii/exposures, fallback, and A2/A3 rules.

**Refuted after allowing the stated repairs:** none. The unqualified/literal forms identified in the preceding bullet are unsound until repaired and must not be tagged PROVEN as written.
