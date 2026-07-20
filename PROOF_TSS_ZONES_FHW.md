# R-Z10: sharpened defender-zone budgets (`F + H_W`)

## REPAIR RECORD (R-Z11, 2026-07-20)

This repair was performed at input HEAD `ad606d0e` on branch
`claude/tss-vcf-width`; no commit was made. The older provenance block below
is retained as R-Z10 authoring history, not current artifact identity.

R-Z10-REV correctly withdrew FHW-T3.  The old displayed `kappa_cut`
definition was not a function: on a non-FC transition into an all-empty
target, both `d in W -> 1` and `q<6 -> 0` applied.  The prose selected zero,
so the review's reachable trace cost `1+5=6` while the clock reported five.

R-Z11 makes the cases disjoint and restores only the theorem that is proved:

1. after the non-D-alive stop, classify the edge as exact/FC or non-FC;
2. classify direct incidence `d in W` **before** touched/virgin cuts;
3. every D-alive direct fill costs one, and an all-empty direct-fill edge is
   admissible only if `1+q<6`;
4. the `q<6` and `(WC)` zeroes are available only when `d notin W`; and
5. the C2 proof now inducts on the first real-only fill of `W`, so a nested
   coupling never assumes an unjustified common real/ghost pre-count.

The repaired charge is written `kappa_cut^*` below.  It is sharp for this
predicate language: the direct indicator cannot be reduced (R-Z10-REV and
the new probes in R-Z11-SR), while every stated zero has a separate C1/C2/C3
proof.  This is not a claim of global logical minimality; a stronger,
independently verified certificate-specific cut may justify more zeroes.

The controlling read-only normative source for this repair is
`E:\Hexo-BotTrainer-hexgt\.claude\worktrees\consolidate-main\docs\PROOF_TSS_DEFENDER_ZONES.md`,
2011 lines, SHA-256
`39197460D068CE5442BA0AFFC687F1408DF3F28EEEB26C4DD7192B87A202064B`.
This authority is repository-committed (R-Z11-REV erratum 3, portable
pin): path `docs/PROOF_TSS_DEFENDER_ZONES.md` on branch
`claude/consolidate-main`, commit `6dc08d7a`, pushed to origin; the SHA-256
above binds that exact blob on any machine.
The 899-line local `docs/PROOF_TSS_DEFENDER_ZONES.md` was neither used as
authority nor edited. For R-Z11 this paragraph supersedes section 0's
historical R-Z10 source-reconciliation narrative; every D14--D21/T3/T11
reference in the repair means the pinned 2011-line authority. No source file
was edited and no build/test command was run for R-Z11.

> **Provenance.** Worktree
> `E:\Hexo-BotTrainer-hexgt\.claude\worktrees\tss-vcf-width`, branch
> `claude/tss-vcf-width`, input HEAD
> `7c2706c86a0362f8e9ddff35ddb1e3185fa0670c` (short `7c2706c8`). Written
> 2026-07-18, America/New_York. No Cargo command was used, and no commit was
> made or is authorized. **Landed-hash placeholder:** `UNLANDED (post-review
> folding owner/orchestrator action required)`.
>
> **Status vocabulary.** Every substantive claim below is labelled
> **PROVEN**, **PARTIAL**, or **OPEN**. A counterexample labelled PROVEN is a
> theorem about the proposed weakening, not a claim about solver output.
> **PROVEN-ON-CLASS** means PROVEN only under the class hypotheses stated in
> that theorem; it is not evidence outside them.

## 0. Reading order and source reconciliation

The sources were read in the requested order:

1. `docs/PROOF_TSS_DEFENDER_ZONES.md` at input HEAD (899 lines, SHA-256
   `6A9C10ACD67DE242E10B2E60B2AA79ABF5280711EFD9849D7D2876F3BC7CABBC`);
2. `docs/PLAN_TSS_SOLVER_UPGRADES.md`, especially §§I.2 and I.5, its cited
   Group-2 progress files, and the `hunt/r1b-r2` report;
3. `packages/hexo_engine/rust/src/{coord,board,legal,rules,state,tactics}.rs`.

There is a material source-version conflict. The checked-in proof corpus at
HEAD ends with hostile Round 4 and definitions D1–D13/T1–T8, while the living
solver plan calls its normative theory a rounds-5–8 revision with D9–D21 and
T3–T11. A matching later proof text was found read-only at
`E:\hexo-bot\docs\paper\sources\PROOF_TSS_DEFENDER_ZONES.md` and
`E:\hexo-bot\docs\paper\companion\PROOF_TSS_DEFENDER_ZONES.md` (2,006 lines,
SHA-256
`48D3B0887519681EFF338A6861D81E1E8D4169E86853463EAEDA21DF361118F6`),
and in the older review worktree. It records Rounds 5–7, not Round 8. Per the
campaign instruction to resolve conflicts toward the latest tightening
sections, this document uses the checked-in corpus as the base and the
rounds-5–7 text as the tightening overlay. Neither landed text is edited.

The exact open problem as posed by the checked-in corpus (§12.1, lines
780–784) is:

> 1. **Sharpened budget (F + H_W).** T4 counts *all* defender placements
>    before the horizon. The sharper count — quiet placements F plus
>    per-window forced-hit capacity H_W — requires per-branch worst-case
>    bookkeeping over the certificate DAG and remains open; T3 is sound
>    without it, at the cost of wider zones for deep loose certificates.

**PROVEN (source reconciliation).** The later tightening overlay partially
resolves that historical item: D19–D21 and T11 prove a branch-coherent debit
for protected exact-copy forcing gates, while its revised §12.1 explicitly
leaves a target-independent D14 debit, zero-cost D17 substitutions, and net
searched-set shrinkage open. The task here is therefore not to re-prove T11
under a new name; it is to delimit or enlarge that frontier and quantify it.

## 1. Phase 1 — definitions and finite enumeration architecture

### 1.1 Claim ledger (live)

| ID | Claim | Status |
|---|---|---|
| FHW-A | A branch-coherent per-window capacity must take its maximum over matched child pairs, not independent maxima. | **PROVEN** by FHW-L1 |
| FHW-B | Protected exact-copy gates admit a sound `F + H_W` exposure clock. | **PROVEN-ON-CLASS** by landed-overlay T11/FHW-T1 |
| FHW-C | An unconditional “forced hits cost zero” replacement is sound. | **PROVEN FALSE** by FHW-O1 |
| FHW-D | A genuine substitution is sound without a transition charge whenever the FC/D22 annotation holds; target-local danger cuts permit further target-specific zeroes, but never erase a direct `d in W` fill. | **PROVEN-ON-ANNOTATED-CLASS** by FHW-T2 and FHW-T3-R; logical maximality remains OPEN |
| FHW-E | The sharpening yields a material zone/budget reduction on the corpus worked examples. | **PARTIAL:** one exact example is `1.50x`; three are `1.00x`; no net-zone theorem |

### 1.2 Enumeration-before-completeness obligation

No completeness claim will precede the following finite index set.

For a fixed finite certificate tree, or the finite unfolding of a D18 DAG,
fix an ordinary node `N`, a target window `W`, and (where applicable) one live
obligation role `rho`. The reverse-topological cases to enumerate are:

1. typed terminal: OR-COMPLETION, WIN, or LOSS(`b`);
2. ordinary OR whose placement enters `W`, and ordinary OR which does not;
3. ordinary AND placement;
4. a forcing-gate kernel reply copied exactly, split by `d in W` / `d notin W`;
5. a forcing-gate off-kernel reply, split by `b=1` / `b=2` and by whether D
   terminates during the escape turn;
6. a substitution transition `(d,s)`, with the following finite refinement:
   exact `d=s`; genuine `d!=s` with global FC pass/fail; on FC failure, each
   live role's finite (RC) pass/fail and each D-alive window split into
   **first** `d in W` / `d notin W`, and only on the `d notin W` side into
   already touched, all-empty with child `q<6`, and all-empty `q>=6` with
   finite (WC) pass/fail. A D-alive `d in W` edge always pays one; when `W`
   is all-empty it must also pass `1+q<6`. The residual non-FC case pays a
   full transition unit and its D22-N test. For each window also enumerate
   the four incidence pairs `(1[d in W],1[s in W]) in {0,1}^2`, the
   non-D-alive case, exact/FC versus non-FC, and whether the relevant
   touched/virgin completion inequality passes. These axes are evaluated in
   that order; no later occupancy or cut row can overwrite direct incidence;
7. a shared DAG node reached by each incoming path (handled by finite
   unfolding; histories remain path-local while node labels are fixed).

This list is exhaustive because D9/D18 gives exactly the three terminal
types and the two internal mover types; D19 partitions every gate reply into
kernel/off-kernel; L3 gives exactly occupancy C1, window-mask C2, and legality
frontier C3; global FC either removes all C3 danger or fails; and the target
cuts then give a Boolean result for each finite role/window query. The gate
kernel, descendant roles, 18 windows through `d`, and the 217 cells of
`B_8(d)` are all finite. For (WC), `q<=B`; it is enough to enumerate the
three orientations and bounded translations of length-six `W` satisfying
`B_8(d) intersect B_{8(B-6)}(W) != empty` (no query exists for `B<6`). The
game has no pass, capture, or third mover. The proof below must fill every
row before claiming a general recurrence.

## 2. Phase 1 result

### 2.1 The only sound scalar is a joint branch maximum

Fix a finite D9 tree, or first unfold a finite D18 certificate DAG as in T10.
Fix a node `N` and a window `W`. A **`W`-exposure trace** starts at `N` and
ends at the first of:

- a mapped OR move which wins or first places in `W`;
- a typed WIN/OR-COMPLETION resolution;
- the end of the defender remainder named by a LOSS leaf; or
- the end of the defender's current turn after the first rejected forcing
  gate, immediately before the adaptive escape attack.

For such a trace `gamma`, define:

```text
F(gamma) = (# ordinary AND placements on gamma)
         + (the whole remaining b at its first LOSS leaf, if any)
         + (the whole remaining b at its first off-kernel escape, if any),

H_W(gamma) = # { protected exact-copy gate edges on gamma whose copied
                 defender cell lies in W }.
```

An ordinary placement which happens to hit a threat is still in `F`; only a
placement copied at a verified tight gate is debited. An escape or LOSS base
charges the whole remainder, including the placement which selected the
escape. Thus `F` is more accurately a **full-cost hazard count** than a
purely semantic count of quiet-looking cells.

The precise per-window forced-hit object is the finite joint set

```text
P_N(W) = { (F(gamma), H_W(gamma)) : gamma is a W-exposure trace from N }.
```

The sharpened exposure is

```text
Q_N^D(W) = max { f+h : (f,h) in P_N(W) }.
```

This is the promised per-window forced-hit capacity: `H_W` is trace-indexed,
and the stateable budget is the maximum of its **joint sum** with `F`. There
is generally no sound reconstruction from the two marginal scalars
`max_gamma F(gamma)` and `max_gamma H_W(gamma)`: those maxima may occur in
different children. A verifier may use their sum as a conservative upper
bound, but it is not the sharpened value.

**FHW-L1 (finite recurrence). PROVEN.** For the protected exact-copy class,
the preceding maximum is exactly the reverse-topological recurrence

```text
Q_N^D(W) = 0                              at WIN/OR-COMPLETION;
Q_N^D(W) = b                              at LOSS(b);
Q_N^D(W) = 0                              at an OR entering W;
Q_N^D(W) = Q_C^D(W)                       at any other OR;
Q_N^D(W) = 1 + max_C Q_C^D(W)             at an ordinary AND;
Q_N^D(W) = max { b,
                 max_{d in K(N)}
                   (1[d in W] + Q_{C_d}^D(W)) }
                                             at a D19 forcing gate.
```

Set the value to zero when `W` is already non-D-alive.

*Proof.* The finite cases are exactly rows 1–5 and 7 of §1.2. The three leaf
and two OR clauses are the trace stops. An ordinary AND prepends one
full-cost opportunity. At a tight gate an off-kernel reply prepends/contains
the complete `b`-placement escape turn and stops; a continuing kernel reply
is copied, changes neither `X` nor `Y`, and contributes exactly its direct
incidence indicator before taking its own child. Taking the maximum inside
each child expression preserves branch coherence. Reverse induction proves
the equation on a tree. A finite DAG has finitely many root-to-node paths;
D18 fixes one node label and T10's unfolding preserves every child expression,
so the same reverse-topological value applies. ∎

**FHW-T1 (baseline sharpened zone). PROVEN-ON-CLASS.** Under all hypotheses
of the rounds-5/6 T3, plus D19's checkpointed tight-gate grammar, D21 may
replace `E_N^D(W)` by `Q_N^D(W)` in the touched and virgin terms and may
replace the role rank `r` by the analogous ordinary-opportunity clock `f` in
the seed term. The target window is protected whenever

```text
cnt_D(W,P_N) + Q_N^D(W) >= 6.
```

This is exactly the tightening overlay's T11, independently recovered from
the exhaustive recurrence above. It extends T3; it changes none of T3's
hypotheses or proof. The scalar D14 `B` remains unchanged because it measures
actual resolution length, not target-specific harm.

### 2.2 Frontier-covered substitution: a broader annotated theorem

This subsection adds a new grammar; it does not modify D17, D19, T3, T9, or
T11. For a position `P`, write

```text
St(P)      = its occupied cells,
Lambda(P) = union { B_8(x) : x in St(P) }.
```

Thus an empty cell is legal exactly when it lies in `Lambda(P)` (apart from
the already discharged empty-board opening case). At a protected tight gate
`Q`, retain D19's named threat family, checks `tau(F_Q)=b` and
`not own_win_now(P_Q)`, checkpoint masks, kernel `K(Q)`, and off-kernel escape
contract. Choose a nonempty representative set `R subseteq K(Q)`, a
retraction `phi:K(Q)->R` (`phi(s)=s` for `s in R`), and the exact certified
child `C_s` for every `s in R`.

For a real kernel reply `d` mapped to `s=phi(d)`, call the transition
**frontier-covered** (`FC_Q(d,s)`) when

```text
B_8(d) subseteq Lambda(P_Q + s).                         (FC)
```

The test is inclusive at distance eight. Exact replies `d=s` are treated as
frontier-covered. A **D22 mixed protected gate annotation** validates every
pair `(d,s)` as follows.

1. It retains `Bhat(d,s)=1+B(C_s)`, the full D14/D15/D16 inequalities, every
   descendant and checkpoint role, every reachable LOSS remainder, nesting,
   both completion channels, D17's WF-legality and A2/A3 inheritance rules,
   and the independently nonempty searched set `R`. No scalar-`B` debit is
   made. The full `r/E^D` labels remain available for horizon, LOSS, and
   own-win validation; the smaller clocks below are additional proved labels.
2. The real `d` avoids every obligation carrier reachable below `C_s`.
   Completion danger is specified completely by the touched/virgin formulas
   in clauses 3–4; no undefined extra cell class is assumed.
3. If `FC_Q(d,s)` holds, no transition-radius unit is required. For each
   D-alive touched window `W` containing `d`, validate

   ```text
   cnt_D(W,P_Q) + 1 + Q^D_{C_s}(W) < 6,
   ```

   and for each all-empty `W` containing `d`, validate

   ```text
   1 + Q^D_{C_s}(W) < 6.
   ```

   Windows not containing `d` receive no direct transition unit. These are
   exactly the 18 length-six windows through `d`.
4. If `FC_Q(d,s)` fails, use a new **D22-N transition-inclusive envelope**,
   rather than claiming landed D17 verbatim. For every ghost-illegal child
   role `rho` carried by `y`, require

   ```text
   dist(d,y) > 8 f_{C_s}(rho),
   ```

   in addition to direct avoidance. For every D-alive touched parent window
   `W` with `d in W`, require

   ```text
   cnt_D(W,P_Q) + 1 + Q^D_{C_s}(W) < 6.                 (N-touch)
   ```

   For every all-empty parent window `W`, put `q=Q^D_{C_s}(W)`; whenever
   `1+q>=6`, require

   ```text
   dist(d,W) > 8(1+q-6).                               (N-virgin)
   ```

   Thus the current transition is fully charged, while later exact/FC gates
   may still use their already proved child debits. This is a new inductive
   envelope with explicit C1/C2/C3 tests, not an unannounced weakening of D17.

For completeness, for any fixed `W` the four transition-incidence cases are

```text
(d in W, s in W): (0,0), (0,1), (1,0), (1,1).
```

Writing `c=cnt_D(W,P_Q)`, the real and ghost post-transition counts are
`c+1[d in W]` and `c+1[s in W]`. Hence

```text
cnt_real(W) <= cnt_ghost(W) + 1[d in W].                (INC)
```

This is the only direct mask asymmetry used by an FC edge. A ghost-only `s`
never increases the real defender count.

Define edge charges

```text
epsilon(d,rho) = 0,                 if d=s or FC_Q(d,s),
                 1,                 otherwise;

kappa(d,W)     = 1[d in W],         if d=s or FC_Q(d,s),
                 1,                 otherwise.
```

For this mixed grammar, extend the trace definition in §2.1 as follows:
`H_W(gamma)` counts both exact and FC-protected kernel transitions whose
**real** reply `d` lies in `W`; `F(gamma)` counts a non-FC substituted
transition as one full-cost opportunity. Escape and LOSS remainders keep
their previous whole-turn charges. Thus the same precise object
`P_N(W)={(F(gamma),H_W(gamma))}` and its joint maximum remain stateable; the
classification is by verified edge annotation, never by a global estimate.

The new gate clocks are the paired recurrences

```text
f_Q(rho) = max_{d in K(Q)}
               (epsilon(d,rho) + f_{C_phi(d)}(rho)),

Q_Q^D(W) = max { b,
                 max_{d in K(Q)}
                    (kappa(d,W) + Q^D_{C_phi(d)}(W)) }.
```

All non-gate clauses remain D20/D20a. The `b` floor is indispensable even
when every kernel edge is frontier-covered.

**FHW-L2 (frontier-coverage lemma). PROVEN.** Couple the real transition
`P_Q+d` to the ghost transition `P_Q+s`. If `FC_Q(d,s)` holds, `d` cannot be
the last ghost-legal ancestor of a later ghost-illegal real-only placement.

*Proof.* A first later cell `z` whose real legality is supplied by `d` lies in
`B_8(d)`. By (FC), `z` lies in `Lambda(P_Q+s)`. If it is empty in the ghost,
it is already ghost-legal and therefore starts a freshly checked dismissal;
if it is ghost-occupied, it is the A2 cancellation case. These exhaust
occupancy. Hence no unmatched legality-frontier chain can inherit from `d`.
The direct-avoidance test supplies C1; the 18 direct window tests supply C2;
and (FC) supplies C3 in the corpus's L3 classification. Equality at distance
eight is covered because both legality and `B_8` are inclusive. ∎

**FHW-T2 (mixed-gate sharpened zone). PROVEN-ON-ANNOTATED-CLASS.** Replace
any D19 gate by a verified D22 annotation and use the displayed paired
`f/Q^D` recurrences. Under every other hypothesis of rounds-5/6 T3 and T11,
D21's seed, touched, and virgin terms remain sound. In particular, a
frontier-covered genuine substitution pays zero role-transition units and
only the direct incidence `1[d in W]`; a non-frontier-covered substitution
pays D22-N's full transition unit.

*Proof.* Unfold a finite D18 DAG. At an ordinary node, terminal, OR edge,
exact gate edge, or off-kernel escape, use FHW-L1 and overlay L17. On a
genuine FC edge, FHW-L2 removes the only C3 reason for a transition unit.
Condition 2 covers C1. For C2, (INC) proves that the only direct real-count
increment is `1[d in W]`; condition 3 and the paired child clock exclude both
touched and virgin completion.

This C2 statement is path-inductive, not an assertion that nested real and
ghost masks always have one common pre-count. Fix `W` and split at its first
real-only fill. Before that event, (INC) gives real count at most ghost count
(a ghost-only `s in W` is harmless). At and after that event, keep the
envelope of its earliest ancestor: its child clock contains every later
edge/child charge. A later nonincident transition is unnecessary to legalize
the remaining real `W` empties, while every later incident transition still
pays `1[d in W]`. Thus the earliest-fill guard, rather than a reset child
pre-count, controls a nested completion.

On a non-FC edge, D22-N supplies a separate transition-inclusive induction.
If the current `d` were the last checked seed of a first later role occupation
at `y`, at most `f_{C_s}(rho)` child opportunities give
`dist(d,y)<=8f_{C_s}(rho)`, contradicting the role test. For a touched window,
all empties are already ghost-legal and only direct `d in W` changes the real
count, so (N-touch) is exact. For an all-empty completion causally descending
from `d`, after charging `d` and the `q` child hazards, six fills force
`dist(d,W)<=8(1+q-6)`, contradicting (N-virgin). A later ghost-legal seed is
checked afresh; a later ghost-illegal seed inherits this envelope. Thus the
three L3 channels close with one current unit plus the already proved child
`f/Q^D` bounds. LOSS and off-kernel branches retain their whole remainders,
and the old full clocks validate every terminal and horizon obligation.
Taking each edge charge together with its own child before maximizing
preserves branch coherence. Reverse induction proves the claim on the
unfolding; D18/T10 then folds identical labels back into the DAG. ∎

Global FC is deliberately simple, but stronger than necessary for one
particular role or window. The following is a **separate target-local cut
clock**, not the `max(F+H_W)` quantity of §2.1: a non-FC transition remains a
unit of global `F` even when a proof shows it harmless to one named target.
Put

```text
GI(G) = Z^2 \ (St(G) union Legal(G)),
```

the ghost-empty illegal cells. Process cut labels in reverse topological order.
For `G=P_Q+s`, a ghost-illegal role `rho` carried by `y`, and
`k=f^cut_{C_s}(rho)`, define the finite role cut

```text
GI(G) intersect B_8(d) intersect B_{8(k-1)}(y) = empty,   (RC)
```

where the last ball is empty for `k=0`. For an all-empty parent window `W`
with `d notin W` and `q=Q^cut_{C_s}(W)>=6`, define the finite window cut

```text
GI(G) intersect B_8(d) intersect B_{8(q-6)}(W) = empty.   (WC)
```

Define the role charge as before and the repaired window charge by a decision
tree, not an overlapping list:

```text
epsilon_cut(d,rho) = 0,  if d=s, global FC holds, or (RC) holds;
                     1,  otherwise.

kappa_cut^*(d,W) =
  0,  if W is non-D-alive;

  // From here W is D-alive.
  1[d in W],  if d=s or global FC holds;

  // From here the edge is genuine non-FC.
  1,  if d in W;
  0,  if d notin W and W is already D-touched;
  0,  if d notin W and W is all-empty and q<6;
  0,  if d notin W and W is all-empty and q>=6 and (WC) holds;
  1,  otherwise.
```

Here `q=Q^cut_{C_phi(d)}(W)` for the current edge/window pair. "Touched" and
"all-empty" are evaluated in the ghost parent `P_Q`; for a D-alive
length-six window they are exhaustive (`cnt_D>0` or `cnt_D=0`). The `(WC)`
predicate is not queried unless `d notin W`, `W` is all-empty, and `q>=6`.
Thus `d in W` is a terminal decision-tree row: no later `q<6` or `(WC)` fact
can overwrite its unit charge.

At a cut-annotated gate use

```text
f_Q^cut(rho) = max_{d in K(Q)}
    (epsilon_cut(d,rho) + f_{C_phi(d)}^cut(rho)),

Q_Q^cut(W) = max { b,
    max_{d in K(Q)}
      (kappa_cut^*(d,W) + Q_{C_phi(d)}^cut(W)) }.
```

All other clauses are the corresponding FHW-T2 clauses with cut child labels.

**FHW-T3-R (repaired target-specific danger-cut extension).
PROVEN-ON-ANNOTATED-CLASS.** D21 remains sound with `f^cut/Q^cut` and
`kappa_cut^*` on the preceding annotated class, provided every mapped edge
passes the following mutually exclusive verifier table. This is a further
C1/C2/C3 refinement; it must not be reported as the exact global trace
capacity `max(F+H_W)`.

Concretely, the repaired target budget at a gate is the displayed
`max{b, max_d(kappa_cut^*(d,W)+Q_child^cut(W))}`; at an ordinary parent D21
uses `cnt_D(W,P_N)+Q_N^cut(W)>=6` for touched protection and the corresponding
`Q_N^cut>=6`, radius `8(Q_N^cut-6)` virgin term. The scalar `B`, LOSS bases,
and escape horizons are unchanged.

For a non-FC edge, the verifier applies the following target-specific
alternative to D22-N:

```text
role rho:                         (RC) permits epsilon_cut=0;
                                  otherwise use the D22-N radius;

D-alive touched W, d notin W:     kappa_cut^*=0;
D-alive touched W, d in W:        kappa_cut^*=1 and require (N-touch);

D-alive all-empty W, d in W:      kappa_cut^*=1 and require 1+q<6;
D-alive all-empty W, d notin W,
  q<6:                            kappa_cut^*=0;
D-alive all-empty W, d notin W,
  q>=6 and (WC):                  kappa_cut^*=0;
D-alive all-empty W, d notin W,
  q>=6 and not (WC):              kappa_cut^*=1 and require (N-virgin).
```

Every zero is target-local; another role/window on the same edge may still
pay one. Exact and FC edges retain FHW-T2's direct touched/all-empty guards;
in particular an exact/FC edge entering an all-empty `W` also requires
`1+q<6`.

*Proof.* Unfold a finite D18 DAG, and order its transition events
chronologically on each unfolded path. We prove simultaneously that the role
clock covers C1/C3 and the window clock covers C2. Terminals, ordinary
OR/AND nodes, LOSS remainders, off-kernel escape floors, exact edges, and FC
edges are FHW-T2. Only a genuine non-FC mapped edge needs new analysis.

For a role occupation causally enabled by the transition, let `z` be the
first post-`d` ghost-illegal real placement. Then
`z in GI(G) intersect B_8(d)`. With at most `k` child opportunities including
`z`, reaching carrier `y` forces `dist(z,y)<=8(k-1)`, contradicting (RC).
If (RC) fails, the one transition unit and D22-N radius give the original
transition-inclusive proof. This closes C1/C3.

For C2, fix `W` and induct on the first real-only fill of `W` on the coupled
path. This is the point omitted from the withdrawn exposition.

- If an earlier real-only `W` fill exists, retain the envelope opened at its
  earliest such ancestor. That stone makes every remaining `W` cell
  real-legal. A later nonincident `d` is not needed as a causal seed for a
  `W` fill, while a later incident `d` is still charged by the terminal
  `d in W` row. The ancestor's branch-paired child clock includes all later
  direct charges, ordinary opportunities, LOSS remainders, and escape floors.
  No common real/ghost pre-count is assumed.
- Otherwise no earlier real-only `W` fill exists: real count is at most
  ghost count before the current edge, and a ghost-only `s in W` can only
  make that inequality safer. Exactly one of the following three subcases
  applies — the nesting is load-bearing (R-Z11-REV erratum 1): each subcase
  is reachable ONLY under this no-earlier-fill hypothesis, never as an
  independent peer of the first bullet.
  - For a touched ghost `W`, every empty is already ghost-legal, so a
    nonincident `d` changes neither count nor the needed legality. An
    incident `d` changes the real count by exactly one and (N-touch) uses
    `1+q`.
  - If the ghost parent `W` is all-empty and `d in W`, the current placement
    is the first real fill and costs exactly one. A completion would require
    five more child fills, so the edge is admissible exactly on the safe
    side `1+q<6`. When `q>=5`, the strict guard fails; equivalently N-virgin
    would demand `dist(d,W)>8(1+q-6)` at distance zero and cannot pass.
  - If the ghost parent `W` is all-empty and `d notin W`, all six real fills
    remain in the child. For `q<6` the child hazard bound makes completion
    impossible. For `q>=6`, if the transition were the first causal source
    of a ghost-illegal fill, its first such seed `z` lies in
    `GI(G) intersect B_8(d)`; spending six of the `q` child hazards in `W`
    leaves at most `q-6` radius-eight links before reaching `W`, hence
    `z in B_{8(q-6)}(W)`, contradicting (WC). If (WC) fails, the one unit
    and (N-virgin) restore D22-N's transition-inclusive envelope.

These rows are mutually exclusive and exhaust every D-alive window because
incidence is Boolean and the ghost window is touched or all-empty. The
non-D-alive stop is permanent. Each edge charge is combined with its own
child before the maximum, so no branches are spliced. Reverse induction
proves the clocks on the unfolding, and D18/T10 folds fixed labels back into
the DAG. ∎

The verifier's new index is finite: `K(Q)` lies in the finite union of named
threat empties; each pair tests the finite descendant-role union, 18 direct
windows, and `B_8(d)` (217 axial cells). The danger intersections are bounded
by the finite child clocks (`f^cut,Q^cut<=B`) and enumerate the same 217 offsets at
their first factor. Consequently rows 1–7 of §1.2 are now all filled.
FHW-T3-R is the largest class **proved in this campaign**, not a claim of
logical maximality: certificate-specific domination facts may prove further
safe mappings. Outside these annotations, the ordinary D17 `+1` remains
mandatory unless another independently proved C1/C2/C3 envelope is supplied.

### 2.2a R-Z11 counterexample replay and new overlap probes

The review's explicit reachable trace lands in exactly one repaired row:

```text
W={(10,r):0<=r<=5}, d=(10,0), s=(9,0),
W D-alive and all-empty, genuine non-FC, d in W, q=5.
```

Therefore `kappa_cut^*=1`, the edge expression is `1+5=6`, and the mandatory
direct guard asks `1+5<6`, which is false. The verifier rejects the mapping.
The escape floor `b=2` is irrelevant (`max{2,6}=6`) and cannot hide the
direct fill. This agrees with the review's actual six-fill continuation.

The following three new adversarial verifier-level instances exercise the
same formerly overlapping class. Each `q=t` child can be realized by a chain
of `t` ordinary AND opportunities whose defender moves take the named
remaining `W` cells, with attacker OR fillers on a disjoint supported chain.
The displayed support stone makes those fills legal; before the sixth fill D
has no completed `W`. As in the review, these local paths test the charge and
guard and need not purport to annotate every alternate branch of a full D9
certificate.

1. **R11-A, exact edge at the unsafe boundary.** Let
   `W_A={(20,r):0<=r<=5}`, `d=s=(20,0)`, and put a shared support stone at
   `(21,0)`. Let the child hazard word fill `(20,1)..(20,5)`, so `q=5`.
   The exact/FC row pays one and its retained all-empty guard rejects
   `1+5=6`. Thus exact copying removes frontier divergence, not the direct
   window fill.
2. **R11-B, genuine FC edge at the unsafe boundary.** Let
   `W_B={(30,r):0<=r<=5}`, `d=(30,0)`, `s=(29,0)`, and include the shared
   stone `t=(31,0)`. The edge is genuine and FC because
   `B_8(d) subseteq B_8(s) union B_8(t)`: for a relative axial cell `(a,b)`
   with `max(|a|,|b|,|a+b|)<=8`, at least one of
   `max(|a+1|,|b|,|a+b+1|)` and
   `max(|a-1|,|b|,|a+b-1|)` is at most eight. Indeed, failure of the first
   requires `a=8` or `a+b=8`, while failure of the second requires `a=-8`
   or `a+b=-8`; every pairing is contradictory or forces `|b|=16`. With the
   five-cell child word, `q=5`; the charge is again one and the direct guard
   rejects six.
3. **R11-C, genuine non-FC safe boundary.** Let
   `W_C={(40,r):0<=r<=5}`, `d=(40,0)`, `s=(39,0)`, with shared support
   `(40,-1)` and no other shared frontier cell covering `z=(48,0)`.
   Then `z in B_8(d)` but `dist(z,s)=9` and `dist(z,(40,-1))=9`, so FC fails.
   Give the child four ordinary `W` fills, `q=4`. The repaired row reports
   the attained hazard `1+4=5` and accepts the sharp strict guard `5<6`.
   The withdrawn zero would report four; it happened not to permit a win in
   this boundary case, but it was still not the correct branch capacity.

R11-A/B show that direct incidence dominates both exactness and FC. R11-C
shows that the repaired unit is not merely a reject-all patch: the last safe
integer boundary remains admissible.

### 2.2b Hostile SELF-REVIEW of the repaired accounting

**Enumeration architecture (stated before completeness).** For each edge,
role, and target, enumerate in this order: (A) non-D-alive/D-alive; (B)
exact-or-FC/non-FC; (C) `d in W`/`d notin W`; (D) touched/all-empty; (E), only
for non-FC, nonincident, all-empty targets, `q<6`/`q>=6`; and (F), only on
the latter side, `(WC)` pass/fail. Independently cross every role with `(RC)`
pass/fail, and record all four `(d in W,s in W)` incidence pairs. Terminal,
OR, ordinary AND, LOSS, off-kernel escape, and DAG-unfolding cases remain the
seven outer grammar rows in section 1.2.

The resulting table is the complete **charge partition** (R-Z11-REV
erratum 2): it is exact for charge rows; the exact verifier leaves
additionally split each row by the pass/fail outcome of its retained
mandatory guard (touched guard, N-touch, direct `1+q<6`, WC, N-virgin):

| edge/window leaf | charge | required check |
|---|---:|---|
| non-D-alive | 0 | permanence stop |
| exact/FC, `d notin W` | 0 | retained FHW-T2 guards |
| exact/FC, `d in W` | 1 | touched guard or, if empty, `1+q<6` |
| non-FC, touched, `d notin W` | 0 | all remaining cells ghost-legal |
| non-FC, touched, `d in W` | 1 | N-touch |
| non-FC, empty, `d in W` | 1 | `1+q<6` |
| non-FC, empty, `d notin W`, `q<6` | 0 | fewer than six child hazards |
| same, `q>=6`, WC pass | 0 | finite danger intersection empty |
| same, `q>=6`, WC fail | 1 | N-virgin |

Only now is completeness claimed. D-alive windows have defender count zero
or positive, incidence is Boolean, FC is Boolean after the exact case, the
integer `q` is below six or not, and WC is Boolean where defined. Therefore
the leaves are disjoint and exhaustive. `s in W` does not add a real stone;
its two values are both covered by the real-at-most-ghost inequality before
the first real-only fill. RC is an independent role axis and cannot change a
window charge.

Hostile attacks and outcomes:

1. **Repeat the review's first-match attack.** No first-match convention is
   needed: all-empty zero rows syntactically include `d notin W`; the direct
   row is terminal. **PASS.**
2. **Try `q=5` under every edge class.** The review trace, R11-A, and R11-B
   all compute one plus five and fail the strict direct guard. **PASS.**
3. **Try the safe neighboring integer.** R11-C computes one plus four and is
   accepted, so the correction has not moved the threshold to `q<=3`.
   **PASS; sharp boundary.**
4. **Hide a prior real-only fill in a nested coupling.** The proof retains the
   earliest-fill ancestor envelope; it never resets to a common child count.
   Later incident edges remain charged. **PASS.**
5. **Use `s in W,d notin W`.** This can only increase the ghost count; (INC)
   remains conservative for real completion. **PASS.**
6. **Use WC at `q=6`.** Its last ball is `B_0(W)`, so it explicitly checks
   whether the first transition-enabled illegal seed can already lie in W;
   WC is never queried on an incident edge. **PASS.**
7. **Refuse a gate or terminate at LOSS.** The unchanged `b` escape floor and
   full LOSS base remain in the same branch maximum. **PASS.**
8. **Splice a cheap edge with another child's cheap clock.** Each
   `kappa_cut^*+Q_child` pair is formed before the outer maximum. **PASS.**
9. **Claim global minimality.** Rejected. The theorem is predicate-sharp for
   the stated RC/WC/FC language, not logically maximal over arbitrary future
   certificate facts. **SCOPE HELD.**

### 2.3 General flat debit obstruction with a reachable coupled trace

The following construction refutes the tempting recurrence “charge the one
ordinary quiet fill, then charge zero for the next tight two-hit gate because
its kernel is disjoint from `W`.” Unlike a bare static mask assignment, both
the real and ghost prefixes below are legal engine histories.

Let

```text
W  = {(0,r) : 0 <= r <= 5},
T1 = {(q,-4) : 0 <= q <= 5},    a = (5,-4),
T2 = {(q, 8) : -5 <= q <= 0},   b = (-5,8).
```

Use proof role `D = engine Player0` and `A = engine Player1`. The common
prefix is:

| ply | phase owner | placement | earlier legality witness | distance |
|---:|:---:|:---:|:---:|---:|
| 0 | D Opening | `(0,0)` | opening rule | — |
| 1 | A FirstStone | `(0,-4)` | `(0,0)` | 4 |
| 2 | A SecondStone | `(1,-4)` | `(0,-4)` | 1 |
| 3 | D FirstStone | `(0,1)` | `(0,0)` | 1 |
| 4 | D SecondStone | `(-1,-4)` | `(0,-4)` | 1 |
| 5 | A FirstStone | `(2,-4)` | `(1,-4)` | 1 |
| 6 | A SecondStone | `(3,-4)` | `(2,-4)` | 1 |
| 7 | D FirstStone | `(0,2)` | `(0,1)` | 1 |
| 8 | D SecondStone | `(0,-8)` | `(0,-4)` | 4 |
| 9 | A FirstStone | `(4,-4)` | `(3,-4)` | 1 |
| 10 | A SecondStone | `(-4,8)` | `(0,2)` | 6 |
| 11 | D FirstStone | `(8,0)` | `(0,0)` | **8** |
| 12 | D SecondStone | `(1,8)` | `(-4,8)` | 5 |
| 13 | A FirstStone | `(-3,8)` | `(-4,8)` | 1 |
| 14 | A SecondStone | `(-2,8)` | `(-3,8)` | 1 |
| 15 | D FirstStone | `(-8,0)` | `(0,0)` | **8** |

At the resulting ordinary node `N`, D has budget one and ghost
`cnt_D(W)=3`. Split the second D placement:

| ply | branch | placement | witness | distance |
|---:|:---:|:---:|:---:|---:|
| 16 | real | `x=(0,3)` | `(0,2)` | 1 |
| 16 | ghost | `s=(8,-8)` | `(0,-8)` | **8** |

Both branches then use the same A turn:

| ply | owner | placement | witness | distance |
|---:|:---:|:---:|:---:|---:|
| 17 | A FirstStone | `(-1,8)` | `(-2,8)` | 1 |
| 18 | A SecondStone | `(0,8)` | `(-1,8)` | 1 |

Every coordinate is new. The owner sequence is exactly
`D; A,A; D,D; ...`; the bold boundary distances show that legality is
inclusive at radius eight and color-blind. No prefix is terminal:

- all A stones lie on the two Q-lines `r=-4` and `r=8`; each has at most a
  five-stone consecutive run, and a non-Q line meets each such line in at
  most one cell;
- the real D branch has only the four-cell run `(0,0)..(0,3)` in `W`; the
  ghost has three there; the other D cells are separated by gaps of at least
  eight on their repeated Q/R lines, and no QR line contains six.

At ply 19 both branches are D FirstStone (`b=2`). In the ghost, every D-alive
window has count at most three, so `own_win_now` is false. In the real branch
`cnt_D(W)=4`, solely because `x` is real-only. The two cells `x` and `s` meet
neither `T1` nor `T2`, so the gate pressure agrees in both branches. Its
complete A-threat empty-set family is

```text
{{a}, {a,(6,-4)}, {b}, {b,(-6,8)}}.
```

The singleton sets force `tau=2`; the extendable-hit kernel is exactly
`K={a,b}`, and `K` is disjoint from `W`. Both kernel cells are legal at the
gate: `dist(a,(4,-4))=1` and `dist(b,(-4,8))=1`.

The real defender may reject that kernel and play `u=(0,4)` followed by
`v=(0,5)`. Both were already legal at turn start (distance at most three from
`(0,2)`), `u` raises the real W-count to five without ending the game, and
`v` completes `W` on the second placement. The cadence therefore gives D the
win before A can exploit either ignored threat.

**FHW-O1 (flat-debit obstruction). PROVEN.** At `N`, a flat rule charges the
real-only `x` as `F=1`, assigns the later disjoint “forced” hits
`H_W=0`, and reads

```text
3 + 1 + 0 < 6.
```

The displayed legal real continuation attains

```text
3 + 1 + 2 = 6.
```

Thus the entire remaining gate turn must appear as an escape alternative in
the **same branch maximum**. An exact-copy debit is sound only with the
`max{b,...}` floor; pressure and a disjoint kernel do not make the real reply
mandatory. This is an obstruction to the proposed counting weakening, not a
counterexample to landed T3: old T3's full exposure guard searches `x`, and
T11's escape floor also gives `1+2`, so both landed theorems reject the bad
dismissal.

The same replay also pins the tempting slack-pressure generalization. A
finite residual automaton may label a state by `(F,k)`, where `F` is the
current threat-empty family and `k` is the remaining defender budget, and
update

```text
F down d = { E in F : d notin E }.
```

It may escape when `tau(F down d)>k-1`; otherwise it continues. But if
`tau(F)<k`, then for every `d`,

```text
tau(F down d) <= tau(F) <= k-1.
```

So pressure rules out no current reply. In FHW-O1 name only the subfamily
`F={{a}}` at the final `k=2` node. Its transversal number is one. The legal
reply `(0,4)` reaches a tight `k=1` state, then the off-kernel `(0,5)` both
uses the final placement and completes `W`. Hence the generic slack exposure
is still `1+1=2`, exactly the old value.

**FHW-O2 (generic slack debit obstruction). PROVEN.** A residual-family
automaton can certify branch-specific reductions when its transition is
independently frontier/window-inert, but `tau(F)<k` alone never justifies a
smaller generic `Q^D(W)`. Reusing one residual family after a genuine
`d->s` substitution additionally needs equal threat-incidence vectors (or a
paired real/ghost residual state); otherwise the two masks have different
pressure.

### 2.4 Quantitative payoff on the worked examples

Three prior tightening-overlay examples have exact arithmetic. The fourth row
is this campaign's new obstruction and is reported separately rather than
mislabelled as corpus input. Static G1/G3 geometry does not carry a D9 horizon
or an `F/H_W` label, so assigning it a ratio would be invented data.

| source / target | old full exposure | sharpened `Q^D` | old/new ratio | disposition |
|---|---:|---:|---:|---|
| overlay strict-debit line, `W={(q,40):0<=q<=5}` | `E=1+2=3` | `Q=max{1,0+2}=2` | **1.50x** | material: 33.3% exposure reduction; `cnt=3` guard changes `6` to `5`, omitting W's three empties from this component |
| overlay dual-purpose `W'={(q,0):5<=q<=10}` | `E=3` | `Q=max{1,1+2}=3` | **1.00x** | no gain; the copied hit lies in W' and the unit `H_W` coefficient is sharp |
| overlay reachable disjoint-hit escape | `D=2` | escape floor `Q=2` | **1.00x** | no sound gain on the refusing branch; naive `0` is false |
| new FHW-O1 coupled obstruction at `N` | `D=1+2=3` | `Q=1+2=3` | **1.00x** | no sound gain on the escape-maximizing branch; flat `1` is false |

The prior corpus/overlay ratios are therefore `1.50x, 1.00x, 1.00x`; the new
obstruction is `1.00x`. The first line is a genuinely material local
improvement, but the effect is selective. No total searched-zone ratio is
PROVEN. Checkpoint roles can enlarge `Prot`, and the tightening overlay
expressly disclaims monotone net cardinality shrinkage. The Group-2 witness's
`62/478` and `18/479` zone/legal ratios measure ranked zones, not an FHW A/B,
and must not be presented as this theorem's payoff.

### 2.5 Hostile self-review of Phase 1

The review tried to falsify each proof hinge before Phase 2.

1. **Version laundering.** The checked-in corpus does not contain D19–D21.
   Outcome: caught; §0 fingerprints both sources, calls the later text a
   tightening overlay, and makes no claim that it is at input HEAD.
2. **Independent marginal maxima.** Maximizing `F` and `H_W` separately can
   splice two children. Outcome: rejected; FHW-L1 and FHW-T2 maximize each
   edge charge with its own child before the outer maximum.
3. **Missing refusal branch.** A tight family does not compel a legal
   defender to stay in `K`; the defender may win during the ignored-threat
   turn. Outcome: the explicit FHW-O1 replay proves the failure and pins the
   `max{b,...}` floor.
4. **Engine-rule audit of FHW-O1.** The 19-ply prefixes, all four continuations,
   inclusive radius-eight witnesses, owner cadence, distinct coordinates,
   absence of an earlier six, complete four-member threat family,
   `tau=2`, `K={a,b}`, and the `3+1+2=6` completion were independently
   re-enumerated from `coord.rs`, `legal.rs`, `state.rs`, `rules.rs`, and
   `tactics.rs`. Outcome: **PASS**. The construction proves a local counting
   obstruction; it is deliberately not advertised as a full false T3
   certificate.
5. **Frontier equality and occupancy.** The FC proof was attacked at exact
   distance eight and at a cell occupied only in the ghost. Outcome: the
   closed `B_8` handles equality; ghost-empty gives a legal checked seed and
   ghost-occupied is exactly A2. No third occupancy case exists.
6. **Hidden role/window channels.** Direct role occupation, touched windows,
   virgin windows, LOSS remainders, and checkpoint roles were each deleted
   in turn. Outcome: every deletion recreates a corpus C1/C2 counterexample;
   D22 retains all of them. Only the C3 transition radius and nonincident
   C2 unit are debited under FC. FHW-T3-R's further zeroes use a separately
   defined target-local cut clock and are not relabelled as global `F+H_W`.
7. **Silent scalar debit.** Replacing `Bhat=1+B(child)` by a debited clock
   breaks horizon and terminal accounting. Outcome: rejected; D22 keeps full
   `B`, full resolution inequalities, and escape deadlines.
8. **DAG aliasing.** A shared node can have different path histories.
   Outcome: finite unfolding keeps histories path-local, while every folded
   node retains one fixed annotation and clock label. The theorem does not
   permit path-dependent labels on a shared node.
9. **Maximality and net-size overclaim.** FC is sufficient, not necessary;
   FHW-T3-R weakens it only on finite target-specific danger cuts, and protected
   roles may outweigh local exposure savings. Outcome: claim downgraded to
   **PROVEN-ON-ANNOTATED-CLASS** and payoff to the four exact local budget
   ratios only. Further certificate-specific domination and monotone
   total-zone shrink remain OPEN.

**Phase-1 disposition.** **PROVEN-ON-CLASS + PROVEN obstruction.** T11's
exact `F+H_W` class extends to D22's frontier-covered mixed gates; FHW-T3-R then
adds a separately named target-local danger-cut clock. The fully general flat
`F + H_W` slogan is false unless its values are branch-paired and retain the
off-kernel/LOSS floors. A target-independent scalar-`B` debit, unconditional
zero-cost substitution, and guaranteed net shrink remain OPEN.

## 3. Phase 2 result

### 3.1 Pick and exact target

The selected second ledger item is the checked-in §12.2 item, quoted exactly:

> 2. **Sharper frontier bands.** The (Z5) band radius 8·D_N is worst-case;
>    chains cost the defender tempo the completion guard already restricts,
>    and a joint tempo-and-distance accounting should shrink the band
>    substantially. Unwritten.

This item has the highest leverage after Phase 1 because the later overlay
already replaces `D_N` by the role rank `r` and, at protected gates, by the
ordinary-opportunity rank `f`. The remaining loss is the scalar maximum:
it forgets which certificate branches and which still-illegal support cells
can actually form a causal chain. The proof-only target is therefore an exact
finite backward support set. No empirical MV-P3 claim is made.

### 3.2 Enumeration architecture before completeness

Fix a finite D9 tree or a finite unfolding of a D18 DAG. Give every D10 role
(attacker placement or leaf witness empty) and D19 checkpoint role a distinct
finite ID `rho`, carrier `y_rho`, and deadline. Put

```text
I_N     = Z^2 \ (St(P_N) union Legal(P_N)),
Empty_N = Z^2 \ St(P_N).
```

The first-protected-occupation witness index is

```text
(rho, pi, U_1<...<U_k, z_0,...,z_k),
```

where:

1. `rho` ranges over the finite role set;
2. `pi` is a directed role-carrying path to its deadline, with an exact
   certificate child at each ordinary node and a mapped `K(Q)` child at an
   exact-copy gate;
3. `U_1<...<U_k` is a subsequence of the ordinary AND nodes of `pi`;
4. each post-seed `z_i` is ghost-empty and ghost-illegal at `U_i`,
   `z_k=y_rho`, consecutive support cells have distance in `1..8`, and a
   predecessor remains ghost-empty until used.

This index is finite. The DAG is finite and acyclic, the number of ordinary
opportunities is finite, and with at most `h` causal links every coordinate
lies in the axial ball `B_{8h}(y_rho)`, of size
`1+3(8h)(8h+1)`. It is exhaustive for the relevant failure mechanism:
recursive selection of a real-legality witness strictly decreases placement
time; OR and exact copied-gate moves add no real-only defender stone; hence
every link is born at one of the enumerated ordinary AND nodes. Off-kernel
escape abandons the old roles and is not a carrying path.

### 3.3 Backward support-reach recurrence

Define `SR_N(rho)` to be the set of ghost-empty locations where one extant
real-only support stone at entry to `N` can participate in an enumerated
causal trace that occupies `y_rho` before its deadline. Compute it in reverse
topological order:

```text
SR_N(rho) = {y_rho}
    at rho's deadline check;

SR_N(rho) = empty
    if rho is not live/reachable at N;

SR_N(rho) = Empty_N intersect SR_C(rho)
    at an ordinary OR with exact certificate child C;

SR_N(rho) = Empty_N intersect
    union { SR_{C_d}(rho) : d in K(N), rho reachable from C_d }
    at a D19 exact-copy gate;

SR_N(rho) = Empty_N intersect
    union_C [ SR_C(rho) union B_8(SR_C(rho) intersect I_N) ]
    at an ordinary AND node, over role-carrying exact children C.
```

The deadline clause has precedence: D10 keeps `rho` live through that check
and discharges it only afterward.

The last line has exactly two cases. `SR_C` preserves an existing support.
If a hazardous child support cell is illegal before the current defender
placement, the real move can reach it only from a predecessor within eight;
the ball adds precisely those predecessors. A punctured ball gives the same
set after the persistence term. `Empty_N` deletes a support which a later
shared filler or attacker move has occupied.

At an ordinary node define the replacement seed zone

```text
Z_seed^SR(N) = Legal(P_N) intersect
  union {
    SR_C(rho) :
      C is an exact searched child of N,
      rho is reachable from C,
      y_rho in I_N
  }.
```

Roles with different deadlines remain separate through the recurrence.

**FR-L1 (finite exactness and scalar containment). PROVEN.** The recurrence
enumerates exactly the finite witness language of §3.2. Moreover,

```text
SR_N(rho) subseteq B_{8 f_N(rho)}(y_rho)
```

under D20, and the same inclusion holds with `r_N` under D15. Consequently,
at a current ordinary node,

```text
Z_seed^SR(N)
  subseteq union_rho Legal(P_N) intersect
             B_{8(f_N(rho)-1)}(y_rho)
  subseteq Z_seed^FH(N),
```

and likewise for the D15 `r` zone.

*Proof.* Reverse induction proves exactness: the deadline is the only terminal
support failure; OR and exact-copy edges add no `X`; an ordinary AND either
preserves a support or adds exactly one illegal radius-eight link. Children
are unioned, never spliced within one witness. The deadline radius is zero;
OR/gate propagation adds zero; and an ordinary AND expansion adds at most
eight. These are precisely D20's `0`, copy/max, and `1+max` rank clauses.
For a child of the current ordinary node, `f_C<=f_N-1`, giving the displayed
seed containment. D15 is identical with `r`. ∎

**FR-T1 (support-reach seed replacement). PROVEN-ON-CLASS.** At every
ordinary global-zone node of T3, or T11 with protected exact-copy D19 gates,
replace only `Z_seed`/`Z_seed^FH` by `Z_seed^SR`. Retain `Z_dir`, touched and
virgin completion zones, every checkpoint role, `(Z4)`, the independently
nonempty fallback, D14, LOSS checks, and every horizon. The resulting zone
theorem is sound.

*Proof.* Suppose the first protected-set failure is a real-only defender
stone at carrier `y`. If `y` was ghost-legal, `Z_dir` made it searched, so it
would be shared. Otherwise trace real-legality witnesses backward through
`X` to the first ghost-legal dismissed seed `x_0` at ordinary node `N_0` and
selected child `C_0`. Every later link occurs at an ordinary AND, is illegal
there in the ghost, and lies within eight of an earlier still-present `X`
stone. Exact copied gates add none; an escape drops the role. Reverse
induction along this actual path therefore puts `x_0` in `SR_{C_0}(rho)`.
Ghost legality of an empty cell is monotone, so `y`, illegal at failure, was
already illegal at `N_0`. Hence `x_0 in Z_seed^SR(N_0)`, contradicting its
dismissal. The unchanged zone components then supply every other T3/T11
channel. ∎

**FR-C1 (early shared legalization). PROVEN.** Let `y` be illegal at ordinary
node `N`. If `y` is legal at every later ordinary AND on every carrying path
below every child of `N` until its deadline, its role contributes nothing to
`Z_seed^SR(N)`. Once a shared move makes empty `y` legal, legality is
permanent; thus no later recurrence expansion sees `y` in `I`, every child
support set stays `{y}`, and its intersection with `Legal(P_N)` is empty.

### 3.4 Reachable strict-containment fragment

Here is a complete local replay showing that FR-T1 can delete a seed which
the scalar band includes. Proof roles are `A` (attacker, engine Player0) and
`D` (defender, engine Player1).
The reachable prefix is engine ply 0 `A:(0,0)`, ply 1 D FirstStone `(0,1)`;
the displayed node `N` is D SecondStone (`b=1`). Split that placement into
real dismissal `x=(7,0)` and ghost searched filler `e=(0,2)`, then use the
same continuation:

```text
A FirstStone  a=(8,0)
A SecondStone h=(0,-1)
D FirstStone  g=(0,3)
D SecondStone k=(0,4)
A FirstStone  y=(15,0)      // future attacker-placement role
```

| placement | earlier legality witness | distance |
|---|---|---:|
| real `x=(7,0)` | `(0,0)` | 7 |
| ghost `e=(0,2)` | `(0,1)` | 1 |
| shared `a=(8,0)` | `(0,0)` | **8 inclusive** |
| shared `h=(0,-1)` | `(0,0)` | 1 |
| shared `g=(0,3)` | real `(0,1)` / ghost `e` | 2 / 1 |
| shared `k=(0,4)` | `g` | 1 |
| planned `y=(15,0)` | shared attacker `a` | 7 |

All coordinates are distinct. The cadence is exactly
`A; D,D; A,A; D,D; A`, and each placement is legal color-blind. Before `a`,
`dist(y,(0,0))=dist(y,(0,1))=15`, so `y` is illegal at `N`, while
`dist(x,y)=8`. No prefix can contain a six: before `y`, A has three stones
and each D board has four; after it, A has four. The designated shared A
moves also have legal support at distances 8, 1, and 7.

The role rank at `N` is three defender placements (the current split, then
`g,k`), so D15's scalar radius is `8(3-1)=16` and contains `x`. Shared `a`
legalizes `y` before the next defender opportunity. The exact child support
set therefore stays `{y}`, whose intersection with `Legal(P_N)` is empty;
`Z_seed^SR` excludes `x`. This is a legal local certificate fragment proving
strict set containment, not a standalone WIN certificate.

### 3.5 Limits, payoff, and hostile self-review

The landed tightening overlay's L9-prime already proves the exact-rank
distance inequality sharp: a legal dismissed seed can be followed by
`r-1` successive distance-eight defender links with the protected carrier
last. FR-L1 can therefore equal the scalar boundary on a corridor. No smaller
uniform multiplier follows; the gain is certificate- and state-specific.

No empirical payoff is claimed. The published Group-2 double-fork witness
reports `Z_seed=0` at both instrumented unforced nodes, so FR-T1 would save
zero there. Its intended value is a deep quiet certificate whose illegal
roles become shared-legal before their deadlines. Measuring frequency and
cardinality is deferred to Phase 3.

Hostile review outcomes:

1. **Infinite-board objection:** rejected; every `(node,role)` set is bounded
   by its finite `8f`/`8r` ball before enumeration.
2. **Missing causal branch:** rejected for the declared witness language;
   the finite index includes every carrying path, ordinary-node subsequence,
   and radius-eight link. Child sets are unioned without cross-child splicing.
3. **Occupied/cancelled support:** repaired by `Empty_N` at every recurrence
   clause. A support consumed by a shared placement cannot persist.
4. **“Exact zone” overclaim:** downgraded. `SR` is exact for the single-chain
   causal witness language sufficient to T3, not the globally minimal
   real-play zone, which would require `(node,X,Y,filler-policy)` product
   states.
5. **D17 interaction:** excluded. FR-T1 proves the global-zone and protected
   exact-copy-gate classes. Combining it with arbitrary D17 histories remains
   OPEN unless a separate mixed-history recurrence is verified.
6. **Coordinate fragment:** cadence, inclusive radius eight, color-blind
   support, uniqueness, and prefix nonterminality all pass the hand audit.
7. **Universal improvement:** refuted by landed L9-prime sharpness. The
   theorem is a subset refinement, not a better universal radius constant.
8. **Unmeasured benefit:** recorded; no Cargo run was made and no numerical
   reduction is inferred from geometry alone.

**Phase-2 disposition.** **PROVEN-ON-CLASS.** The scalar seed band may be
replaced by a finite, branch-aware support-reach set at ordinary global-zone
nodes, including certificates with protected exact-copy gates. The D17
mixed-history extension and empirical prevalence are OPEN.

## 4. Global caveat ledger

1. **Source-version caveat (OPEN as repository hygiene).** The current branch
   does not contain the rounds-5–7 proof source that the living plan cites.
   This document fingerprints and crosswalks it but does not silently replace
   the checked-in landed theorem.
2. **Reachability caveat.** Abstract D3 positions need not encode a reachable
   history. Every new coordinate obstruction in this document must include an
   ordered replay rooted at `(0,0)`, the color cadence, an inclusive radius-8
   witness for every placement, and a per-prefix no-win check.
3. **Engine/model caveat.** The mathematical board is `Z^2`; engine
   coordinates are `i16`. All constructions stay far from the representation
   boundary, so this distinction is inert here.
4. **Budget caveat.** `f` and `Q^D` are target-specific hazard clocks. None of
   FHW-T1/T2/T3 debits D14's scalar resolution length `B`, terminal remainders,
   or absolute escape horizons.
5. **Branch caveat.** All maxima pair a transition charge with its own child.
   Separate marginal maxima are safe only as an explicitly conservative
   overestimate, never as the claimed sharp value.
6. **DAG caveat.** Shared nodes require one fixed role set, gate kernel,
   representative map, FC/cut verdict, and clock label. If those are
   path-dependent, the verifier must split the node before folding.
7. **Substitution caveat.** Phase 1 proves D22's annotated mixed gates. Phase
   2's `SR` recurrence is not yet proved across arbitrary D17/D22 mixed
   histories; its theorem deliberately stops at exact-copy gates.
8. **Measurement caveat.** All numerical ratios in Phase 1 are exact local
   arithmetic from existing worked examples. No runtime, corpus frequency,
   or total-zone reduction was measured in this no-Cargo campaign.

## Review erratum and R-Z11 disposition

**Reviewed artifact:** this document, landed unmodified at `ded361c1`
(supplies the `UNLANDED` landed-hash placeholder). **Review:**
`PROOF_TSS_ZONES_FHW_REVIEW.md` (committed this fold). **Verdict:**
REFUTED IN PART / MAJOR REPAIR REQUIRED at R-Z10.

- **Historical FHW-T3 refutation (accepted).** Its overlapping `κ_cut` cases permit an
  unsound zero charge for a direct window fill: a reachable trace costs
  `1 + 5 = 6` while the stated rule counts only `5`. The old theorem remains
  withdrawn and must not be cited.
- **R-Z11 repair: PROVEN-ON-ANNOTATED-CLASS.** FHW-T3-R uses the disjoint
  `kappa_cut^*` decision tree, charges every direct fill, requires
  `1+q<6` on an all-empty direct edge, and proves the nested first-real-only-
  fill induction. Section 2.2a rejects the review trace and checks three new
  overlap probes; section 2.2b gives the exhaustive hostile enumeration.
- **G2-Z1 UPHELD — sound-on-success** (as originally labelled PROVEN-ON-CLASS;
  not a completeness claim).
- **Flat-debit refutation UPHELD.** **FR-T1 UPHELD ON CLASS** (ordinary/
  exact-copy; mixed D17/D22 stays OPEN). **λ² UPHELD ON DESIGN CLASS.**
- **Design bars:** their paper definitions are repaired in
  `DESIGN_GROUP2_NEXT.md`; every empirical outcome remains
  DEFERRED-NEEDS-CARGO.
- Total searched-zone shrink and generic zero-cost D17 substitution remain OPEN.

**Current proof disposition:** repaired on the stated D22/RC/WC annotated
class. No result extends to arbitrary D17/D22 histories or debits scalar `B`.
