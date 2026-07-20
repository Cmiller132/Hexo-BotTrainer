# Forced-hit budget debit (`F + H_W`)

VERDICT: PARTIAL

The blanket rule “subtract placements which answer attacker threats from
`B(N)` or `E^D_N(W)`” is false, even when the selected hitting universe is
disjoint from `W`.  Ignoring the threats leaves the defender's whole current
turn available, and the defender can complete `W` before the promised attacker
reply.  In a coupled real/ghost play, the ghost can also pass
`¬own_win_now` while an earlier real-only fill makes the same turn winning in
the real position.

A branch-coherent weakening proved here uses protected exact-copy forcing
gates.  At such a gate a named current threat family has
transversal number equal to the defender's remaining placement budget.  A
reply in the extendable-hit kernel is copied exactly.  Every other reply
abandons the old certificate and invokes a finite adaptive threat contract.
The resulting harm clock is

```
max { full remaining-turn escape charge,
      max over copied kernel children
          (dual-purpose charge + child harm clock) }.
```

This yields a proved debit in the ranked seed, touched-window, and
virgin-window terms.  It handles dual-purpose hits, the two placements of a
`b = 2` turn, LOSS remainders, prior coupling divergence, and finite DAGs.
It does not debit D14's target-independent scalar `B(N)`, and it gives no
zero-cost rule for a substituted rather than copied hit.  Those two extensions
remain open.

---

## 1. Counterexamples to an unconditional debit

### 1.1 A reachable disjoint-hit race

Let

```
W  = {(0,r) : 0 <= r <= 5},
T1 = {(q,-4) : 0 <= q <= 5},   a = (5,-4),
T2 = {(q, 8) : -5 <= q <= 0},  b = (-5,8).
```

The following legal history starts with `D` as Player 0.  Each displayed pair
is one two-placement turn.

| plies | mover | placements |
|---:|:---:|:---|
| 0 | D | `(0,0)` (Opening) |
| 1--2 | A | `(0,-4)`, `(1,-4)` |
| 3--4 | D | `(0,1)`, `(-1,-4)` |
| 5--6 | A | `(2,-4)`, `(3,-4)` |
| 7--8 | D | `(0,2)`, `(0,-8)` |
| 9--10 | A | `(4,-4)`, `(-4,8)` |
| 11--12 | D | `(0,3)`, `(1,8)` |
| 13--14 | A | `(-3,8)`, `(-2,8)` |
| 15--16 | D | `(-8,0)`, `(8,-8)` |
| 17--18 | A | `(-1,8)`, `(0,8)` |

Every setup placement is within distance 8 of an existing stone.  The final
setup has no complete window: the only collinear five-stone A runs are the
displayed parts of `T1` and `T2`, and the only four-stone D run is the displayed
part of `W`.  It is now D's FirstStone position, so `b = 2`.

The complete current attacker threat-empty family is

```
F = {{a}, {a,(6,-4)}, {b}, {b,(-6,8)}},        tau(F) = 2.
```

The singleton members force every transversal to contain `a` and `b`, so its
extendable-hit kernel at `b = 2` is exactly `{a,b}`.  This complete available
hitting kernel is disjoint from `W`.  Nevertheless D may ignore the threats
and play

```
u = (0,4), then v = (0,5).
```

Both placements are legal at turn start.  After `u`,
`cnt_D(W) = 5`; after `v`, `cnt_D(W) = 6`, so D wins immediately on the
second placement.  A never receives the turn in which the ignored threats
would be completed.

Translations and axial symmetries give the same local counterexample, and
arbitrarily many isolated legal filler pairs can be added without meeting the
three named windows.  Thus this is an infinite family, not a boundary effect
of the Opening.

This refutes all rules which subtract the two disjoint “forced” hits without
charging the branch on which D refuses them.  The defect is not dual purpose:
neither hit is in `W`.  It is not a failure to split a `b = 2` pair: the two
placements are displayed separately.  It is the terminal ordering in D4.

### 1.2 The same obstruction after coupling divergence

An explicit `¬own_win_now` check in the ghost position does not repair a
flat debit.  The following local certificate fragment shows why.  At an
ordinary `D,b = 1` node `N`, let

```
W = {(q,0) : 0 <= q <= 5},

A_gate = {(10,0),(11,0),(12,0),(20,0),(21,0),(22,0)},

A_fork = {(-3,20),(-2,20),(-1,20),(3,20),
          (0,17),(0,18),(0,19),
          (-3,23),(-2,22),(-1,21)},

D = {(0,0),(1,0),(2,0),(9,0),(19,0)}.
```

Radius-8 support stones may be added away from all named windows; they do not
change any count below.  There are no A-threats at `N`, and every D-alive
window has D-count at most 3.  Let the searched ghost reply be
`y = (0,8)` and let the real dismissed reply be `x = (5,0)`.  Both are legal
and neither wins.  The coupling now has `X = {x}` and `Y = {y}`.

The shared attacker turn plays `(13,0)` and `(23,0)`.  At the resulting
`D,b = 2` node `Q`, the ghost has exactly the two relevant threats

```
E1 = {(14,0),(15,0)},
E2 = {(24,0),(25,0)}.
```

The blockers `(9,0)` and `(19,0)` kill the other windows through the two
four-stone runs.  Hence `tau({E1,E2}) = 2`, its extendable-hit kernel is
`E1 union E2`, and that kernel is disjoint from `W`.  The ghost still has
`cnt_D(W) = 3` and passes `¬own_win_now`.  The real position has
`cnt_D(W) = 4` because of `x`.

The real defender ignores the two threats and plays `(3,0)`, then `(4,0)`.
The counts in `W` are 5 and then 6; the second placement wins before A moves.
All four threat cells remain empty.

The searched ghost child can be made a finite genuine A-win certificate.
Force one exact hit in each of `E1,E2`.  A then plays `p = (3,0)`, which
enters `W`, and `c = (0,20)`.  At the resulting `D,b = 2` LOSS leaf name

```
U1 = {(-3,20),(-2,20),(-1,20),(0,20),(1,20),(2,20)},
U2 = {(0,17),(0,18),(0,19),(0,20),(0,21),(0,22)},
U3 = {(-3,23),(-2,22),(-1,21),(0,20),(1,19),(2,18)}.
```

Their empty sets are respectively

```
{(1,20),(2,20)}, {(0,21),(0,22)}, {(1,19),(2,18)}.
```

They are pairwise disjoint, so their transversal number is 3.  No gate hit
meets these windows, and the ghost D-alive maximum remains 3; the mandatory
LOSS `¬own_win_now` check passes.  Every two-placement remainder misses one
named pair and A fills it in the following turn.

A recurrence which charges `x` but assigns zero to the disjoint gate reads
`3 + 1 < 6` at `N` and licenses the losing dismissal.  The correct recurrence
below assigns the gate its escape floor `b = 2`, so the same test reads
`3 + 1 + 2 = 6` and searches `x`.

### 1.3 Pressure itself is not a coupling invariant

Threats used for a debit must also be protected at gate entry.  For example,
let the shared A-stones be

```
{(0,0),(1,0),(2,0),(3,0),
 (30,0),(31,0),(32,0),(33,0)}
```

and let common D blockers be `(-1,0)` and `(29,0)`.  The ghost additionally
has `Y = {(0,8)}`, while the real position has `X = {(4,0)}` instead.  At
`b = 2` the ghost threat-empty family is

```
{{(4,0),(5,0)}, {(34,0),(35,0)}}
```

and has transversal number 2.  In the real position the first window is
already killed by the `X`-stone `(4,0)`, so the corresponding real family has
transversal number 1.  The real defender may hit the second window and use the
other placement quietly.  A debit computed only from the ghost label has
over-debited one placement.  Entry checkpoint roles in D19 below exclude this
mask mismatch.

### 1.4 A substituted hit is not a zero-cost hit

The zero cost proved below is limited to an exact copied placement.  A D17
substitution can answer a threat and still create a harmful real-only frontier.
The following is a local mask/frontier configuration; inert radius-8 support
chains and balanced fillers may be added away from the named cells.  Use the
two threat empty sets

```
E1 = {(4,0),(5,0)},   E2 = {(34,0),(35,0)}
```

supported by shared A-runs at `q = 0..3` and `q = 30..33`, with blockers at
`(-1,0)` and `(29,0)`.  Let the ghost hit `(4,0)` and `(34,0)`, while D17
substitutes the real hits `(5,0)` and `(35,0)`.  Thus the placements hit the
same pressure components, but they create

```
X = {(5,0),(35,0)},   Y = {(4,0),(34,0)}.
```

Add a shared A support stone `t = (5,25)` and a future protected carrier
`zeta = (5,16)` with four defender opportunities before its deadline.
The shared attacker turn plays `(33,1)` and `(33,2)`. Both cells are legal at
turn start from the shared stone `(33,0)`; neither prefix is terminal. They
leave the minimum distances from `(5,8)` equal to 8 in the real position and
9 in the ghost position, and leave the pre-relay distance from `zeta=(5,16)`
equal to 9 in both positions. The real placement `(5,8)` then reduces only
the real distance to `zeta` to 8.
Consequently a rule which removes the substituted hits from the frontier
rank misses a real-only two-link corridor.  D17's transition `+1` and envelope
inheritance are mandatory.

---

## 2. Formal exact-copy weakening

The following text is ready to install as a conditional extension of the
normative proof.
For T11, an ordinary node is a D21-governed internal AND
node. It may use a T11.1/D17 envelope dismissal by dismissal. A D19 gate is
not ordinary. A T6 kernel-region node remains governed by T6 and is outside
this extension unless a separate equal-position T6 handoff is declared.

### D19 (protected exact-copy forcing gate)

**D19 (protected exact-copy forcing gate).** At an internal AND node `Q`
with defender budget `b in {1,2}`, a *forcing gate* names a finite family
`H_Q` of current A-threat windows.  Write

```
F_Q = { E(U,P_Q) : U in H_Q }.
```

The verifier checks

```
tau(F_Q) = b,                  (FG1)
not own_win_now(P_Q).          (FG2)
```

For a cell `d`, put

```
F_Q \ d = { E in F_Q : d notin E },
K(Q) = { d in Legal(P_Q) : tau(F_Q \ d) <= b-1 }.
```

For every `U in H_Q` and `e in E(U,P_Q)`, D19 extends D10 by a third role type,
the checkpoint role `(Q,U,e)`.  Its deadline is the gate-entry mask check
immediately before the defender reply.  It is in every strict ancestor's
reachable obligation union.  At `Q`, let `Prot^-(Q)` be the incoming protected
set before the check; it still includes all checkpoint carriers, and D12's
`X intersect Prot^-(Q) = empty` invariant is maintained by the coupling and
proved in L17.  The verifier checks the checkpoint roles' ancestor
coverage/ranks and the named ghost masks; it does not inspect path-local real
`X`.  Immediately after the masks are checked, discharge those roles and call
the resulting set `Prot^+(Q)`.  The gate then evaluates `S(Q) = K(Q)` in this
post-check phase and is not an ordinary D21 zone node.  This two-phase notation
is only needed at a gate; ordinary nodes continue to use `Prot(N)`.

The searched set at the gate is exactly the nonempty set `K(Q)`.  Every
`d in K(Q)` has its exact D4 child `C_d`, and a continuing real reply is copied
on that edge.  No D17 substitution is permitted for a kernel reply.  A real
reply outside `K(Q)` abandons the original subtree and uses the adaptive
escape contract of L15.

Let `p(Q)` be the absolute index of the last placement before entry to `Q`,
using D9's path-derived clock.  At a root gate it is the last placement index
of the history defining the root position (equivalently, one less than the
root's next-ply clock).  The escape deadline is

```
p(Q) + b + 2.
```

Indeed the remaining D placements have indices `p(Q)+1` through `p(Q)+b`,
and the following A placements have indices `p(Q)+b+1` and `p(Q)+b+2`.
The certificate horizon is the maximum of its old declared resolutions and
all reachable escape deadlines.  A verifier wishing to preserve the old
horizon checks each escape deadline against that old maximum instead.

Every `K(Q)` is finite and nonempty.  If `d` hits no member of `F_Q`, then
`F_Q \ d = F_Q` and `d notin K(Q)`, so `K(Q)` is contained in the finite
union of the named empty sets.  If `T` is a size-`b` minimum transversal,
then every `d in T` satisfies that `T \ {d}` hits `F_Q \ d`; hence
`d in K(Q)`.  T1 makes all such cells legal.

For compact data, one threat suffices at `b = 1`.  At `b = 2`, L13's
`b = 1` selection argument applied to a family of transversal number 2 gives
a subfamily of at most three threats still having transversal number 2.

### D20 (the branch-coherent `F + H_W` clocks)

**D20 (forced-hit-debited role ranks).** For a role `rho` live at `N`, define
`f_N(rho)` in reverse topological order:

```
f_N(rho) = 0
    at rho's deadline or when rho is no longer reachable;

f_N(rho) = f_C(rho)
    across an ordinary OR edge while rho remains live;

f_N(rho) = 1 + max_C f_C(rho)
    at an ordinary AND node;

f_N(rho) = max_{d in K(N)} f_{C_d}(rho)
    at a forcing gate.
```

A child where `rho` is not reachable contributes zero.  As in D15, put

```
f_N(y) = max { f_N(rho) : rho is live at N and is carried by y }.
```

The unit of `f` is an opportunity to add a real-only defender stone.  An
ordinary AND edge costs one.  An exact gate edge costs zero because its
defender stone is shared; an off-kernel reply abandons every old role.

**D20a (forced-hit-debited window exposure).** For every window `W`, define
`Q_N^D(W)` by the D16 leaf and OR clauses and the following AND clauses:

```
Q_N^D(W) = 0
    at WIN and OR-COMPLETION;

Q_N^D(W) = b
    at a LOSS leaf with remaining defender budget b;

Q_N^D(W) = 0
    at an OR whose designated A-placement enters W;

Q_N^D(W) = Q_C^D(W)
    at every other ordinary OR;

Q_N^D(W) = 1 + max_C Q_C^D(W)
    at an ordinary AND;

Q_N^D(W) = max {
                    b,
                    max_{d in K(N)}
                       ( 1[d in W] + Q_{C_d}^D(W) )
                  }
    at a forcing gate with remaining budget b.
```

As in D16, set `Q_N^D(W) = 0` when `W` is already non-D-alive.  The maximum
inside the gate must be taken child by child.  Replacing it by
`max_d Q_{C_d} + 1[K intersect W nonempty]` is admissible but can combine
different branches and overcount by one.

The path interpretation is exact for the abstract gate bookkeeping.  On a
continuing mapped path let `F` count:

1. every ordinary defender opportunity;
2. every remaining defender placement at a LOSS leaf; and
3. all `b` placements of the first off-kernel escape turn.

Let `H_W` count exact copied gate placements whose cell lies in `W`.  Then
`Q_N^D(W)` is the branch-coherent maximum of `F + H_W` generated by these
rules.  A copied hit outside `W` is debited.  A copied hit in `W` is charged
once.  A quiet placement of a `b = 2` split is charged through `F`.  An
ignored gate is charged through the `b` escape floor.

Here `F` means full-cost placements: every ordinary defender opportunity and
every placement in the first LOSS or off-kernel escape remainder. An
ordinary placement which happens to hit a threat is still counted in `F`.
Only an exact copied gate placement receives the forced-hit debit, and it
contributes to `H_W` exactly when its cell lies in `W`.

The verifier uses the exact reverse-topological values in the displayed
recurrences. These are the labels for which L16(3), the comparisons
`f_N(rho)<=r_N(rho)` and `Q_N^D(W)<=E_N^D(W)`, and the `B`-bounded finite
query procedure are asserted.

For D14-D16 and for a full D17 envelope, a D19 gate remains an internal AND
node: `B`, `r`, and `E^D` retain their original `+1` inequalities over every
`K(Q)` child. Only `f` and `Q^D` use the D20 gate clauses. Consequently the
full clocks cover all defender placements in an off-kernel escape, while the
debited clocks measure only their stated hazards.

### D21 (debited zones)

**D21 (ordinary debited zone).** At an ordinary internal AND node define

```
Z_dir^FH(N) = Prot(N) intersect Legal(P_N),

Z_seed^FH(N) = union {
    Legal(P_N) intersect B_{8(f_N(y)-1)}({y}) :
    y in Prot(N) \ (Legal(P_N) union Stones(P_N)),
    f_N(y) >= 1
},

Z_touch^FH(N) = union {
    E(W,P_N) :
    W is D-alive at P_N,
    cnt_D(W,P_N) >= 1,
    cnt_D(W,P_N) + Q_N^D(W) >= 6
},

Z_virgin^FH(N) = {
    c in Legal(P_N) :
    some all-empty window W has
    Q_N^D(W) >= 6 and d(c,W) <= 8(Q_N^D(W)-6)
}.
```

The node searches an independently nonempty superset of

```
Z_dir^FH union Z_seed^FH union Z_touch^FH union Z_virgin^FH.
```

The obligation set includes all reachable checkpoint roles from D19.  Clause
(Z4) is unchanged.  A forcing gate instead performs its entry checks and
searches exactly `K(Q)`; every off-kernel reply uses the escape contract.

For a fixed augmented certificate,

```
f_N(rho) <= r_N(rho),       Q_N^D(W) <= E_N^D(W).
```

Thus every displayed radius or completion threshold is no larger than its
D15/D16 predecessor.  Adding checkpoint roles can enlarge `Prot` at strict
ancestors; the theorem makes no claim that the total number of searched cells
must decrease in every certificate.

**D13/T7 augmented clause.** At a D21 ordinary node set
`R_cert^FH(𝒸,N)=Z_dir^FH(N) ∪ Z_seed^FH(N) ∪ Z_touch^FH(N) ∪
Z_virgin^FH(N)`. Any independently nonempty `S(N)` containing
`R_cert^FH(𝒸,N)`, together with (Z4) and all reachable D19 checkpoint roles,
is sufficient by T11. The optional solver superset is
`R_search^FH=R_cert^FH ∪ hitting(P_N) ∪ 𝒜(P_N) ∪ r3(P_N)`. At a
D19 gate this clause does not apply: the certified searched-child map is
exactly `K(Q)`, and heuristic terms are not added to `S(Q)`.

---

## 3. Proofs

### L15 (gate transfer and escape)

**L15 (protected gate dichotomy). [PROVEN]** On entry to a D19 gate, every
named threat window has identical real and ghost masks.  For every real legal
defender reply `d`, exactly one of the following holds.

1. `d in K(Q)`.  The cell is shared-empty and ghost-legal.  Both games place
   at `d`, take the exact child, and leave `X` and `Y` unchanged.
2. `d notin K(Q)`.  If D does not win during the remaining current turn, A
   completes a named surviving threat in at most two placements of the
   following turn, by `p(Q)+b+2`.

*Proof.* Let `U in H_Q`.  Its A-stones are shared.  Every ghost empty
`e in E(U,P_Q)` is a checkpoint carrier and hence is not in `X` at the entry
check.  Since `U` is ghost A-alive, no ghost D-stone lies in `U`, so no such
cell is in `Y`.  The complete real and ghost masks of `U` agree.

If `d in K(Q)`, then `d` belongs to a named empty set, is shared-empty, and is
legal by T1.  D19 supplies its exact child, so this is case 1.

Otherwise

```
tau(F_Q \ d) > b-1.
```

Let `H` be the set of the at most `b-1` later defender placements in the
current turn.  It cannot hit every member of `F_Q \ d`.  A named window
therefore avoids both `d` and `H`, remains A-alive, and retains its one or two
initial empties.  L1 puts those empties within distance 2 of permanent shared
A-stones, so they are legal and A completes the window in the next turn.  D4
allows D to terminate first during its current turn; that is the alternative
explicitly retained in the statement and charged by D20a.  The two cases are
exclusive by definition.  ∎

### L16 (clock bounds and nesting facts)

**L16 (weighted hazard bounds). [PROVEN]** For every D19--D21 certificate:

1. On a continuing mapped path, the number of ordinary real-only frontier
   opportunities before a live role's deadline is at most `f_N(rho)`.
2. For a fixed window `W`, on every continuation before the certificate
   attacker wins or first enters `W`, or before an off-kernel escape resolves,
   count one for each ordinary defender edge, `1[d in W]` for each exact copied
   gate edge, and every remaining defender placement in the first LOSS or
   off-kernel escape remainder. This count is at most `Q_N^D(W)`.
3. `f_N(rho) <= r_N(rho)` and `Q_N^D(W) <= E_N^D(W)`.
4. At a gate, `B(Q) >= b`.  Consequently D14/L11's ancestor budget covers
   every escape remainder even though `B` is not debited.

*Proof.* Items 1 and 2 follow by reverse induction on the finite certificate.
An ordinary AND placement can add one `X`-stone and is charged one by both
recurrences.  An exact gate placement changes neither `X` nor `Y`; it is
charged to a chosen window exactly when its cell lies in that window.  An
off-kernel reply ends the old role contract and permits at most the current
turn's `b` defender placements before L15's A reply.  A LOSS leaf likewise
permits all `b` placements, so its base cannot be debited.  OR stops and
ordinary OR propagation are exactly D16's.

For item 3, induct against D15 and D16.  The role inequality is immediate:
an old AND edge adds one where a gate edge now adds zero.  At a `b = 1` gate,
the old D16 exposure is `1 + max_C E_C`, which dominates both the escape floor
1 and every `1[d in W] + Q_{C_d}`.  At a `b = 2` gate, every exact child is a
nonterminal `D,b = 1` node, hence has old exposure at least 1 for every still
D-alive `W`.  Therefore the old `1 + max_C E_C` is at least 2 and also
dominates every continuing child term.  Leaf and OR comparisons are equal.

For item 4, `K(Q)` is nonempty.  If `b = 1`, the D14 AND inequality gives
`B(Q) >= 1`.  If `b = 2`, every kernel successor is a nonterminal `D,b = 1`
position.  It is either a LOSS leaf, whose budget is at least 1, or another
internal AND node, whose budget is at least 1.  Hence
`B(Q) >= 1 + B(C_d) >= 2`.  D14 nesting then covers an escape from every
ancestor.  ∎

### L17 (joint protected-occupation and completion safety)

**L17 (debited first-bad-event lemma). [PROVEN]** Under D19--D21:

1. while the real play remains mapped to the original certificate, no defender
   placement creates a real-only stone in the current protected set; the old
   roles are abandoned when an off-kernel escape begins; and
2. before the mapped certificate attacker or a gate escape attacker resolves,
   no real defender play completes a window.

*Proof.* Suppose the first failure of the applicable type occurs.  Every
earlier gate entry on the mapped prefix has exact named masks by the checkpoint
roles and the minimal choice of the failure.

For a protected occupation at carrier `y`, the direct ghost-legal case is in
`Z_dir^FH` and is searched.  Otherwise trace real-only legality backward from
`y` to the last ghost-legal dismissed `X`-seed `x_0`.  A copied gate stone is
present in both games. If it supplies legality for a later real placement at a
ghost-empty cell, that cell is ghost-legal and any dismissal is a newly checked
seed. If the later cell is ghost-occupied, the move is T3 case A2: it cancels a
`Y`-stone and creates no `X`-stone. Hence a copied gate stone cannot be an
internal link of a ghost-illegal real-only chain. Every link of the final
`X`-chain was created at an ordinary AND opportunity.

If the chain has `j` stones,

```
d(x_0,y) <= 8(j-1),       j <= f_{N_0}(rho)
```

for the still-live role `rho` carried by `y`.  Hence `x_0` was in
`Z_seed^FH(N_0)`, contradicting its dismissal.  This proves the checkpoint
invariant as a special case, because a checkpoint role remains live through
its entry check.

Now suppose the first failure is a real D-completion of `W`.  If no W-cell was
ever dismissed, (MI) gives the real D-count at most the ghost D-count.  On a
continuing exact path the ghost has no defender-terminal edge.  At a LOSS
remainder the mandatory ghost `¬own_win_now` check and the fully charged base
`b` give at most five stones.  At an off-kernel gate, the explicit gate check
gives ghost count at most 3 for `b = 2` and at most 4 for `b = 1`; (MI) and the
escape floor `b` again give a real maximum of five.  Thus this case cannot
complete.

If the first real-only W-fill occurs on an off-kernel reply and there was no
earlier real-only W-fill, gate-entry (MI), (FG2), and the fully charged escape
floor bound the final real count by five exactly as in the preceding
no-dismissal case.  It cannot complete.

Here “no `W`-cell was ever dismissed” concerns the mapped prefix. Any first
`W`-fill among the later placements after an off-kernel reply is already
included in the full `b` escape floor.

Therefore every remaining completion case has a first real-only W-fill
anchored at an earlier ordinary node: a continuing gate reply is copied, and
the only non-copied gate reply terminates the mapped line during its charged
escape turn.

Take that first ordinary W-fill and its last ghost-legal dismissed seed.  If
ghost `W` is already touched at the first fill, pre-fill (MI) bounds the real
count by `cnt_D(W,P_N)`.  L16 charges every later ordinary placement, every
copied gate hit in `W`, every LOSS remainder, and any final escape turn.  A
completion would imply

```
cnt_D(W,P_N) + Q_N^D(W) >= 6.
```

The first fill was therefore in `Z_touch^FH(N)`, a contradiction.

If ghost `W` is virgin and the first real-only fill is ghost-legal, completion
requires six charged W-fills, so `Q_N^D(W) >= 6`; distance zero puts the fill in
`Z_virgin^FH(N)`.  If it is ghost-illegal, trace from the last ghost-legal
dismissed seed to `W`.  Let `j` be the number of real-only radius-8 links before
the chain first reaches `W`.  Before a final escape or LOSS remainder, only
ordinary placements can form those links; copied gate placements outside `W`
are shared, while copied gate placements in `W` are charged through `H_W`.
Every approach link or W-fill in a final LOSS remainder or escape turn is
charged by its full `b` base or floor.  The chain and the six real W-fills give

```
Q_{N_0}^D(W) >= j + 6,       d(x_0,W) <= 8j.
```

Thus `x_0` lies in the radius `8(Q_{N_0}^D(W)-6)` virgin term, again
contradicting dismissal.  This also covers a completion during an escape: the
gate's entire remaining turn is in the `b` floor.  ∎

### T11 (soundness of the exact-copy debit)

**T11 (exact-copy `F + H_W` soundness). [PROVEN]** Let a finite D9 tree or
D18 DAG use D21 at every ordinary AND node and D19 at every forcing gate.
Include every checkpoint role in the reachable obligation unions, retain
(Z4), retain D14's `B`, and include the D19 escape deadlines in the global
horizon.  Then the compiled attacker wins against every real defender play by
that horizon.  The debited `f` and `Q^D` values are sound replacements for
`r` and `E^D` in the D21 seed, touched, and virgin terms.

*Proof.* At an ordinary node run T3's A1--A3 coupling, using D21 and L17 in
place of L9′ and L12.  At a gate, first check and discharge the checkpoint
roles.  L15 transfers the named family.  A real reply in `K(Q)` is shared-empty,
is copied on its exact child, and leaves `X,Y` unchanged.  A reply outside
`K(Q)` abandons the old subtree.  L17 excludes a D-completion in the remaining
turn, and L15 supplies an adaptive surviving threat which A completes by the
declared escape deadline.

The WIN, LOSS, OR-COMPLETION, and ordinary OR transfers are unchanged.  The
LOSS branch remains sound because D20a charges its whole remainder and D10
still protects every witness empty through leaf entry.  A finite tree must
reach a typed terminal or a first escape.  For a DAG, unfold as in T10; the
extra labels and max recurrences are preserved.  ∎

**T11.1 (D17 envelope compatibility). [PROVEN]** T11 remains valid when an
ordinary node's global D21 dismissal tests are replaced by a valid D17
envelope, provided the selected reachable-role union includes every future
checkpoint role and the envelope retains D17's original transition-inclusive
`B-hat`, role ranks, and `E-hat` tests.  No debited D17 envelope is claimed by
this corollary.

The selected role union contains every role live at `C_s` and at every node
reachable from `C_s`, expressly including a checkpoint role whose deadline
is entry to `C_s` itself. D17 condition 3 therefore forbids the transition
cell `d` from occupying such a carrier.

*Proof.* Before the selected child, D17's real `d` and ghost substitute `s`
create the same canonical `X,Y` transition as in T9, so its current `+1` is
mandatory.

D17 conditions 2-8 protect all selected-child roles and all three L3
channels, including LOSS remainders. If the protected-occupation seed or
first real-only window fill was introduced at a D21 dismissal, L17 applies
with `f` and `Q^D`. If it was introduced by a D17 transition, D17 conditions
3-5 apply with the full transition-inclusive role rank and window exposure;
condition 7 carries that envelope through later ghost-illegal descendants,
and condition 8 covers a LOSS remainder. L16(3) gives `f<=r` and `Q^D<=E^D`
for later D21 steps. This joint D17/D21 induction remains valid through a
D19 gate because the full clocks use the old AND inequalities there. An
off-kernel reply abandons the selected subtree and is charged by its gate
floor.

The remainder of T9's nested-envelope proof is unchanged.  ∎

---

## 4. Sharpness

### 4.1 A strict exposure debit attained by a real line

Consider a `D,b = 1` position with the following stones.  The two colors have
15 stones each, the position is nonterminal, and the displayed support chain
makes every module radius-8 connected.

```
A = {(q,0) : 0 <= q <= 4}
    union {(-3,20),(-2,20),(-1,20),
           (0,17),(0,18),(0,19),
           (-3,23),(-2,22),(-1,21)}
    union {(0,-7)}.

D = {(-1,0),
     (8,0),(8,8),(8,16),(8,24),(8,32),
     (0,40),(1,40),(2,40),
     (-8,0),(0,-8),(-8,8),(8,-8),(16,-8),(16,0)}.
```

Direct enumeration of the 18 incident windows per stone gives no complete
window and a maximum D-count of 3 in a D-alive window.  Name the singleton
root threat

```
T = {(0,0),(1,0),(2,0),(3,0),(4,0),(5,0)},
h = (5,0).
```

Then `F = {{h}}`, `tau(F) = 1`, and `K = {h}`.  D copies `h`.  A plays
`(3,20)` and then `(0,20)`, reaching a `D,b = 2` LOSS leaf with the three
windows `U1,U2,U3` from section 1.2.  Their empty pairs are pairwise disjoint,
so `tau = 3`, and the D-alive maximum is still 3.

Let

```
W = {(q,40) : 0 <= q <= 5}.
```

Initially `cnt_D(W) = 3`, and `h notin W`.  The old D16 exposure is

```
E_root^D(W) = 1 + 2 = 3,
```

so the old touched test reads `3 + 3 = 6` and searches all W-empties.  The
new branch-coherent value is

```
Q_root^D(W) = max { 1, 0 + 2 } = 2,
```

so the new test reads `3 + 2 = 5` and omits them.

The value 2 is attained.  Follow the copied hit `h`; at the LOSS remainder D
plays `(3,40)` and `(4,40)`.  Exactly two defender placements enter `W`, its
count reaches only 5, one of the pairwise-disjoint LOSS witnesses survives,
and A fills that witness.  Thus this one-unit debit and the LOSS base `b = 2`
are both sharp on a real legal line.

### 4.2 The dual-purpose coefficient is independently sharp

Use the complete certificate position and line of section 4.1, but query the
different D-alive window

```
W' = {(q,0) : 5 <= q <= 10}.
```

Initially it contains the D-stone `(8,0)` and no A-stone.  The copied kernel
hit `h = (5,0)` lies in `W'`.  At the later LOSS remainder let D play `(6,0)`
and `(7,0)`.  The three placements `h,(6,0),(7,0)` are legal, all lie in
`W'`, do not complete it, and are followed by A's certified LOSS-witness win.
The recurrence gives

```
Q_root^D(W') = max { 1, 1[h in W'] + 2 } = 3,
```

exactly the achieved harm.  Deleting the dual-purpose indicator would give
`max {1, 0+2} = 2`, which is false on this line.  Thus `H_W`'s unit
coefficient is independently sharp.  Sections 1.1 and 1.2 separately show
that the full `b = 2` escape charge is sharp when the defender splits or
ignores the asserted pressure.

The exact sharpness of every possible `f`-based seed-radius reduction is open.
On all-ordinary paths `f = r`, so L9′'s existing radius sharpness remains
the boundary case.

---

## 5. Integration notes

### D14 and D15

`B(N)` is unchanged.  It measures total future defender placements needed for
resolution, not target-specific harm.  Exact copied hits still consume plies,
and every LOSS or escape remainder still consumes up to `b` placements.  L16
proves `B(Q) >= b` at a gate, so D14 nesting and the selected-path part of L11
remain valid after adding gate-escape paths.

The role rank used in the seed radius is `f`, not a replacement scalar `B`.
The old inequality `r_M + k <= r_N` for *all* defender placements has no
`f` analogue.  Its replacement is the hazard statement: only ordinary
`X`-creating opportunities count, and copied gate edges create no `X`.  Any
code or later proof which needs the number of actual defender placements must
continue to use `r` or `B`, not `f`.

Checkpoint roles are D19's explicit third extension of the D10 role grammar.
They enter `Omega` and `Prot` at strict ancestors, have deadline zero at gate
entry, and are discharged after the incoming `Prot^-` mask check.  They form no
negative-radius band and impose no post-check direct zone at the gate.

### D16 stop conditions and LOSS leaves

The D16 stop is unchanged: the relevant window clock stops when A wins or
first enters `W`.  `Q^D(W)` uses exactly that stop on continuing branches.  An
off-kernel branch has its separate D19 A-resolution; before it, the entire
remaining D turn is charged by the gate floor.

A LOSS leaf remains `Q^D(W) = b`.  Its `tau > b` contract says that A has a
surviving witness after the D remainder; it does not prevent D from placing
all `b` stones in a chosen harm region before then.  Assigning zero to this
base reproduces the counterexample in section 1.

At ordinary AND nodes, L14 remains derivable with `Q`: a `b = 1` ordinary
node has `Q >= 1`, and a `b = 2` ordinary node has `Q >= 2` because its
nonterminal `D,b = 1` child has exposure at least 1.  At forcing gates the
completion zone is intentionally absent, so (FG2) is an explicit mandatory
check, as in T6.

### T9 and D17 envelopes

Every D17 substitution is ordinary for the debit calculus.  In particular:

1. a transition-inclusive role clock is `1 + f_child`, not `f_child`;
2. a transition-inclusive window/frontier clock is `1 + Q_child(W)`, not a
   zero-cost forced hit;
3. all reachable future checkpoint roles belong to D17.2's child obligation
   union; and
4. D17.4, both D17.5 completion tests, and D17.7 inheritance remain mandatory.

The example in section 1.4 shows why geometric disjointness from a newly
shrunk band cannot remove the transition `+1`.  A real hit and a different
ghost hit create `X,Y`; only an exact copy receives the zero frontier cost.

The existing full `B-hat`, role-rank, and `E-hat` D17 tests are the proved
conservative implementation in T11.1.  A future debited-envelope theorem would
at minimum have to set the selected transition rank to `1 + f_child` and the
selected window clock to `1 + Q_child(W)`, then rerun both touched and virgin
D17.5 tests with those transition-inclusive values.  This report does not
claim that additional theorem.  If a node itself is a forcing gate, kernel
replies are exact A1 edges and off-kernel replies escape; no substitution
annotation is used there.

### T10 DAGs

For a shared D18 node, the forcing-gate flag, named threat family, computed
kernel, exact child map, checkpoint roles, `¬own_win_now` result, and escape
deadline are part of its one consistent label.  The `f` and `Q` inequalities
are checked on every outgoing edge.  Unfolding preserves these labels and the
branch-coherent maxima; `X,Y` histories remain path-local.  T10 therefore
applies without a new graph argument.

### Verifier procedure

All checks are finite.

1. Recompute each named current threat window and its one- or two-cell empty
   set.
2. Check `tau(F_Q) = b` by the existing `mhs <= 2` enumeration and check
   `¬own_win_now`.
3. Enumerate `K(Q)` from the finite union of the named empty sets and test
   `tau(F_Q \ d) <= b-1` for each cell.
4. Check `S(Q) = K(Q)`, exact D4 successors, and the absence of a substituted
   kernel edge.
5. Add the checkpoint roles to all strict-ancestor reachable-role unions and
   check their deadline masks.
6. Compute the least `f` values and the queried `Q` values in reverse
   topological order, using the child-coherent gate maximum.  The queried
   windows are finite: touched windows come from the finite position; for the
   virgin term, `Q <= E <= B`; if `B < 6` there is no query, and otherwise for
   each finite legal candidate enumerate only window starts within radius
   `8(B-6)` of that candidate, exactly as in D11's inverted virgin query.
7. At ordinary nodes enumerate the four D21 zone terms from those finite
   queries.  At gates check only the two-phase gate grammar and kernel.
8. Recompute each absolute escape deadline and include it in the horizon.

No step quantifies over an unbounded continuation.  The universal off-kernel
reply set is compressed by the finite transversal inequality in L15, exactly
as a D9 adaptive LOSS contract compresses its defender remainder.

---

## 6. Remaining gaps

**GAP 1 -- target-independent D14 debit.** No sound scalar
`B_FH(N) < B(N)` is proved.  `B` is used for actual resolution length, LOSS
remainders, transition envelopes, and coarse nesting.  A hit copied outside
one window may be inside another window or may still consume a deadline ply.
The proved quantities are hazard-specific `f` and `Q^D(W)`.

**GAP 2 -- substituted forced hits.** No zero-cost D17 forced-hit rule is
proved.  Section 1.4 refutes the direct proposal.  A stronger domination or
frontier-inertness certificate might debit some substitutions, but it must
prove all three L3 channels under D17 conditions 1-8 rather than only threat
incidence.

**GAP 3 -- automatic net shrinkage.** Checkpoint roles are necessary and can
enlarge ancestor obligation zones.  The theorem proves that the ranked hazard
terms shrink for a fixed augmented certificate; it does not prove that adding
gates always reduces the cardinality of the final searched sets.

**GAP 4 -- slack pressure.** A family with `tau < b` does not force the next
placement: its extendable-hit kernel is all legal cells, as in T6.  This report
proves no debit there beyond later tight gates.  Treating a whole slack turn by
a more precise finite residual-state automaton may yield further reductions.

These gaps are why the overall verdict is PARTIAL.  The exact-copy gate
theorem itself, including the `F + H_W` recurrence and all three zone
replacements, is proven.
