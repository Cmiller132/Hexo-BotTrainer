# Claimed strengthenings of T3/T4/T6 — UNDER HOSTILE REVIEW, NOT NORMATIVE

Provenance: produced by an external frontier model given the uploaded
PROOF_TSS_DEFENDER_ZONES.md. Notation was mangled in transit: `(...)` wraps
inline math, `[ ... ]` wraps display math, and stray `====` underlines are
artifacts of display blocks. Read charitably as LaTeX.

Claimed summary: the present T3 is sound but materially stronger in its
hypotheses and zones than its proof needs. Six claimed tightenings:

1. Z1 is not a hypothesis of T3.
2. The frontier radius can be reduced from (8D_N) to (8(D_N-1)).
3. The global horizon budget can be replaced by node-local, cell-local, and
   window-local budgets.
4. The full witness windows in `core` can be replaced by only their relevant
   empty cells.
5. T6 can be strengthened substantially: neither the original core nor all
   hitting cells need be searched.
6. Dismissals can be checked relative to a selected substitute branch instead
   of the union of the entire subtree.

Claimant's own assessment: 1–5 proved by modifications of the existing
argument; 6 is a larger generalization deserving another hostile review.

The current theorem protects the union of all witness-window cells and all
future attacker placements, uses the absolute-horizon count
(D_N=\mathfrak D(P_N,T)), requires Z1, and applies an (8D_N) frontier band.

# 1. Immediate strengthened version of T3

## Local defender budget

For each certificate node (Q), define (B(Q)) to be the maximum number of
future defender placements before the certificate's declared attacker
resolution, over continuations below (Q), including the defender remainder of
a LOSS leaf.

It can be computed recursively:

[
\begin{aligned}
B(\text{OR-COMPLETION leaf})&=0,\\
B(\text{WIN leaf})&=0,\\
B(\text{LOSS leaf with budget }b)&=b,\\
B(\text{internal OR node})&=B(\text{child}),\\
B(\text{internal AND node})&=1+\max_{\text{children }C}B(C).
\end{aligned}
]

Thus, if descendant (M) is reached from (N) after (k) defender placements,

[
B(M)+k\le B(N).
]

This is all the proof needs. The current quantity (\mathfrak D(P_N,T)) is
merely a possibly much larger admissible upper bound. Consequently, the
"exact (D_N)" caveat in T3 can be weakened to:

> The verifier supplies a sound, hereditarily decreasing upper bound on the
> remaining defender placements. Exact local maxima improve pruning but are
> not required for soundness.

This is especially useful when a long branch elsewhere determines global (T),
while the subtree below (N) resolves much sooner.

## Compressed obligation set

Replace the current `core`, which includes all six cells of every named
witness window, by

[
\operatorname{Obl}(\mathcal C,N)
=
\{\text{future attacker placement cells below }N\}
\cup
\bigcup_{\substack{L\text{ a WIN/LOSS leaf below }N\\ W\text{ named at }L}}
E(W,P_L).
]

Here (E(W,P_L)) is the empty set of the witness window at the leaf.

An OR-completion window adds nothing beyond its designated attacker
placement: immediately before a one-placement completion, the other five
cells are already attacker stones, and attacker stones are identical in the
real and ghost games.

This compression is sound because, for an A-alive witness (W):

* all existing attacker stones are automatically shared;
* a ghost-empty cell cannot be a (Y)-cell, since (Y)-cells are ghost defender
  stones;
* therefore it suffices to ensure that (X) avoids (E(W)).

So entire witness masks agree as soon as the witness's empty cells avoid
(X). The current proof protects complete windows only to establish exactly
this mask agreement.

Define

[
\operatorname{Prot}^{+}(N)
=
\operatorname{Obl}(\mathcal C,N)
\cup
\bigcup
\left\{
E(W,P_N):
\begin{array}{l}
W\text{ is D-alive at }P_N,\\
\operatorname{cnt}_D(W,P_N)+B(N)\ge6
\end{array}
\right\}.
]

## Tight frontier radius

The current (8D_N) radius prevents a protected cell from even becoming
real-legal while ghost-illegal. That is stronger than necessary. T3 only
needs to prevent the defender from **occupying** that cell as a real-only
move.

Suppose (x_0) is the first ghost-legal dismissed stone in a legality chain,
and a protected cell (y) is eventually occupied as the (p)-th defender
placement from (x_0), counting (x_0). There are at most (p-1)
frontier-expanding links:

[
d(x_0,y)\le8(p-1).
]

Since (p\le B(N_{x_0})),

[
d(x_0,y)\le8\bigl(B(N_{x_0})-1\bigr).
]

Thus the exact uniform band is (8(B(N)-1)), not (8B(N)).

The strengthened frontier condition is

[
S(N)\supseteq
\operatorname{Legal}(P_N)\cap
B_{8(B(N)-1)}
\left(
\operatorname{Prot}^{+}(N)
\setminus
(\operatorname{Legal}(P_N)\cup\operatorname{Stones}(P_N))
\right).
]

This is geometrically sharp absent additional structure: after the first
dismissed stone, (B-1) further placements can advance the frontier by exactly
(8(B-1)).

The existing L9 proves the stronger "never even becomes legal" property with
radius (8D_N). Replacing that conclusion by "is never occupied as a real-only
protected stone" saves exactly one complete radius-8 hop.

## T3(^+): revised statement

> **T3(^+) — local-budget dismissal soundness. `[PROVEN-DRAFT]`**
> Let (\mathcal C) be a valid D9 certificate. At every internal AND node (N),
> let (B(N)) be any verifier-certified budget satisfying the upper-bound and
> hereditary-decrease conditions above. Assume:
>
> 1. (S(N)\ne\varnothing);
> 2. (S(N)\supseteq\operatorname{Prot}^{+}(N)\cap\operatorname{Legal}(P_N));
> 3. (S(N)) contains the (8(B(N)-1)) frontier band displayed above;
> 4. every internal attacker placement has a legality witness among the root
>    stones or prior attacker stones.
>
> Then the certificate compiles into a total attacker strategy against the
> full legal defender move set. For every real play, either the attacker wins
> earlier than the ghost certificate, or the real play is mapped to a
> certificate path and the attacker wins no later than that path's declared
> resolution. In particular the attacker wins by the global certificate
> horizon (T).

Notably, **Z1 does not appear**.

The uploaded proof itself describes Z1 as "belt and braces" and explicitly
says that T3's leaf transfer does not require current threats to be searched.
The actual T3 proof invokes Z2, Z5, Z4, the coupling invariants, and the
completion anchor, but never invokes Z1.

Accordingly, the corresponding T4 should become:

[
Z^{+}(N)
=
\operatorname{Legal}(P_N)\cap
\left[
\operatorname{Prot}^{+}(N)
\cup
B_{8(B(N)-1)}
\left(
\operatorname{Prot}^{+}(N)
\setminus
(\operatorname{Legal}(P_N)\cup\operatorname{Stones}(P_N))
\right)
\right].
]

Any **nonempty** (S(N)\supseteq Z^{+}(N)) is sufficient. If
(Z^{+}(N)=\varnothing), the verifier still needs one arbitrary searched
fallback child for the coupling.

# 2. A stronger, channel-specific completion refinement

The previous theorem still treats defender completion and future attacker
obligations through one common protected set. The channels can be separated
further.

## Window exposure budget

For each window (W), define (E_N^D(W)) as the maximum number of defender
placements below (N) before either:

* the attacker wins, or
* the certificate attacker first places in (W), permanently killing (W) as a
  defender window.

A canonical recurrence is:

[
\begin{aligned}
E_Q^D(W)&=0
&&\text{at a WIN or OR-COMPLETION leaf},\\
E_Q^D(W)&=b
&&\text{at a LOSS leaf with defender budget }b,\\
E_Q^D(W)&=0
&&\text{at an OR node whose move lies in }W\text{ or resolves},\\
E_Q^D(W)&=E_{\text{child}}^D(W)
&&\text{at any other OR node},\\
E_Q^D(W)&=1+\max_C E_C^D(W)
&&\text{at an internal AND node}.
\end{aligned}
]

This can be strictly smaller than both global (D_N) and local (B(N)).

## D-touched windows need no frontier band

If a D-alive window (W) already contains at least one defender stone, then
every empty of (W) is within distance at most (5) of that defender stone.
Hence every empty is already legal.

Therefore search

[
Z_{\mathrm{touch}}(N)
=
\bigcup
\left\{
E(W,P_N):
\begin{array}{l}
W\text{ is D-alive},\\
\operatorname{cnt}_D(W,P_N)\ge1,\\
\operatorname{cnt}_D(W,P_N)+E_N^D(W)\ge6
\end{array}
\right\}.
]

No Z5-style frontier expansion is needed for these windows.

## Virgin windows have a much smaller approach radius

Now let (W) be all-empty. Suppose a currently legal dismissed move (d) is to
initiate a frontier chain that eventually completes (W).

Starting with (d):

* reaching and placing the first stone in (W) requires at least
  (\lceil d(d,W)/8\rceil) additional placements;
* after the first W-stone, five more placements are required.

Thus completion requires at least

[
6+\left\lceil\frac{d(d,W)}8\right\rceil
]

defender placements counting (d). Therefore (d) can matter only if

[
d(d,W)\le 8\bigl(E_N^D(W)-6\bigr).
]

The virgin-window seed zone is consequently

[
Z_{\mathrm{virgin}}(N)
=
\left\{
d\in\operatorname{Legal}(P_N):
\begin{array}{l}
\exists\text{ all-empty window }W,\\
E_N^D(W)\ge6,\\
d(d,W)\le8(E_N^D(W)-6)
\end{array}
\right\}.
]

This is a substantial tightening:

* a future attacker/core cell gets radius (8(B-1));
* a virgin defender window gets only radius (8(E^D-6)).

For example, with seven remaining defender placements, the generic current
band is radius (56); the one-hop-corrected obligation band is radius (48);
but a virgin six-window requires only a radius-(8) seed guard.

The proof is the same first-chain argument as L9, except the chain must
reserve six placements for the six window cells. This gives the joint
tempo-and-distance accounting anticipated in the document's open problem.
The document currently identifies the (8D_N) band as worst-case and leaves
this combined accounting unwritten.

A more refined finite zone is therefore

[
Z_{\mathrm{ranked}}(N)
=
Z_{\mathrm{obligation}}(N)
\cup
Z_{\mathrm{touch}}(N)
\cup
Z_{\mathrm{virgin}}(N),
]

where (Z_{\mathrm{obligation}}) uses the compressed obligation cells and
their (8(B-1)), or preferably cell-specific, deadlines.

Claimant considers this proof complete at paper level, but because it
changes both the completion anchor and L9's invariant, tags it
`[PROVEN-DRAFT]` pending one independent hostile pass.

# 3. Cell-specific deadlines

The uniform (B(N)) can itself be eliminated from the attacker-obligation
frontier.

For every obligation cell (y), let (r_N(y)) be the maximum number of defender
placements before the point at which (y) must still be empty:

* for an internal attacker move, until that move is played;
* for a WIN witness empty, until the WIN leaf is reached;
* for a LOSS witness empty, until the LOSS leaf is reached.

For a LOSS leaf, the rank does **not** need to include the leaf's remaining
(b) defender placements. After the leaf is reached, the adaptive hitting
argument expressly allows the defender to occupy witness empties; it only
needs the family to have the correct masks at leaf entry.

Require the hereditary condition

[
r_M(y)+k\le r_N(y)
]

whenever (M) is reached after (k) defender placements and (y) is still an
obligation.

Then use

[
Z_{\mathrm{obligation}}(N)
=
\bigl(\operatorname{Obl}(N)\cap\operatorname{Legal}(P_N)\bigr)
\cup
\bigcup_{\substack{y\in\operatorname{Obl}(N)\\ y\notin
\operatorname{Legal}(P_N)\cup\operatorname{Stones}(P_N)}}
\left[
\operatorname{Legal}(P_N)\cap B_{8(r_N(y)-1)}(y)
\right].
]

The same chain proof works one target cell at a time. This is likely the
best non-branch-sensitive frontier formula.

# 4. Branch-indexed substitution

The current `core(C,N)` is a union over every searched child below (N). But
after an unsearched real move, the coupling is free to select one searched
ghost filler and therefore one child. Only that selected child's future
strategy must survive the new real-only stone.

This gives a stronger theorem schema.

For each ghost-legal unsearched reply (d), let the certificate name a
substitute

[
\phi_N(d)=s\in S(N).
]

Let (C_s) be the child reached by the ghost filler (s). Compute:

* obligations only in the subtree rooted at (C_s);
* local budgets only for that subtree;
* window exposures only for that subtree.

Then (d) may be dismissed whenever it satisfies the obligation, completion,
and frontier tests relative to (C_s).

Equivalently,

[
d\text{ is dismissible}
\quad\Longleftarrow\quad
\exists s\in S(N)\; \operatorname{SafeSub}(N,d,s).
]

Only a reply having **no** safe substitute needs to be searched.

Why this works (claimed):

* in A3, the ghost uses (\phi_N(d)) as its filler rather than an arbitrary
  child;
* the new (X)-stone only needs to avoid the descendant requirements of that
  child;
* earlier (X)-stones remain safe because an earlier selected child envelope
  already contained the entire current subtree;
* later searched real replies may branch arbitrarily inside (C_s), which is
  why the whole (C_s)-subtree, not merely one leaf, is retained.

This is the natural certificate-level counterpart of domination: the
searched child is not merely a clock filler but a certified substitute
strategy.

Claimant sees no missing C1, C2, or C3 channel in this extension but calls
it `[PROVEN-DRAFT]`, not immediately overwriting normative T3.

A simpler implementation is to name one default fallback child (f(N)) for
all dismissals. That already replaces `core(C,N)` by `core(C,f(N))` for
newly introduced (X)-stones at (N).

# 5. T6 can be tightened much further

The current T6 searches

[
\operatorname{hitting}(P_N)
\cup
\bigl(\operatorname{core}(\mathcal C,N)\cap\operatorname{Legal}(P_N)\bigr).
]

But its own proof says that the auxiliary refutation does not rely on the
original subtree's core. Once the first dismissed move is encountered, the
proof abandons the original continuation and wins from the current threat
family.

Therefore the core term can be removed.

More strongly, not every hitting cell needs to be searched.

Let

[
\mathcal F_N
=
\{E(W,P_N): W\text{ is an A-threat}\}.
]

For a legal defender reply (d), define the surviving threat family

[
\mathcal F_N\setminus d
=
\{E\in\mathcal F_N:d\notin E\}.
]

Define the extendable-hit kernel

[
K_b(N)
=
\left\{
d\in\operatorname{Legal}(P_N):
\tau(\mathcal F_N\setminus d)\le b-1
\right\},
]

where (\tau) is minimum hitting-set size.

Interpretation:

* (b=1): (K_1) consists exactly of cells hitting **every** current threat.
* (b=2): (K_2) consists exactly of cells extendable by one second cell to a
  complete two-cell hitting set.

Any (d\notin K_b) is immediately refutable:

* if (b=2), the successor has defender budget (1) and residual hitting
  number (>1), hence is a LOSS leaf;
* if (b=1), at least one A-threat survives and the attacker, now with budget
  (2), has a WIN leaf.

The defender cannot win first:

* at a (b=2) node with `¬own_win_now`, every D-alive window has count at
  most (3); two defender placements reach at most (5);
* at a (b=1) node, every D-alive window has count at most (4); the current
  placement reaches at most (5).

The same-(T) argument also survives. If (K_b\ne\operatorname{Legal}), then
the node has (\operatorname{mhs}=b). Following a genuine minimum hitting set
kills all current A-threats, so the original certificate cannot resolve on
the first following attacker placement. Its horizon therefore reaches the
second attacker placement needed by the auxiliary refutation.

This gives:

> **T6(^+) — extendable-hit kernel. `[PROVEN-DRAFT]`**
> At every internal defender node with `¬own_win_now`, searching exactly
> (K_b(N)) is sufficient. No original-core term is required. If
> (\operatorname{mhs}<b), then (K_b(N)=\operatorname{Legal}(P_N)), so the
> theorem simply performs no pruning at that node. If (\operatorname{mhs}=b),
> it strictly refines the current T6 searched set.

This also removes T6's global hypothesis "(\operatorname{mhs}=b) at every
AND node." The formula automatically degenerates to the full legal set at
non-forcing nodes.

# 6. LOSS witnesses can be sparsified

Every A-threat empty set has size one or two. Therefore a LOSS leaf never
needs an arbitrarily large witness family.

## Budget (b=1)

If a family of one- and two-element sets has hitting number (>1), it has a
subfamily of at most three sets with hitting number (>1).

Proof: choose (E=\{a,b\}). Since neither (a) nor (b) hits everything, choose
one set missing (a) and one missing (b). Those three have no one-point
transversal. If (E) is a singleton, two sets suffice.

The bound three is sharp for the triangle (\{a,b\},\{b,c\},\{a,c\}).

## Budget (b=2)

If the hitting number is (>2), either:

* there are three pairwise disjoint sets, which already witness it; or
* a maximal disjoint matching consists of two sets (E_1,E_2).

Any two-point transversal of (E_1,E_2) chooses one point from each, giving
at most four candidate pairs. For each candidate pair, choose one family
member it fails to hit. Together with (E_1,E_2), at most six sets witness
hitting number (>2).

The bound six is sharp for general rank-two set systems: the six edges of
(K_4) have vertex-cover number three, while deleting any edge lowers it to
two.

Thus D9 may require:

[
|\mathcal T|\le
\begin{cases}
3,&b=1,\\
6,&b=2.
\end{cases}
]

Combined with obligation compression, a LOSS leaf contributes only the union
of at most three or six one-/two-cell empty sets, rather than all six cells
of every threat window.

# 7. Other claim cleanups

First, the explicit `¬own_win_now` check at every **internal** AND node is
logically redundant under the completion-zone requirement and the
prohibition on defender-terminal searched edges. A count-5 defender window
forces its winning empty into the searched set; a count-4 window at (b=2)
forces both empties through two searched steps; either produces a prohibited
defender-terminal edge. Keep the explicit check because it is cheap and
diagnostic, but it is not an independent theorem hypothesis. The check
remains necessary at LOSS leaves, where no searched set is expanded.

Second, T3 extends unchanged from a finite tree to a finite acyclic
certificate DAG with consistent node clocks. The induction becomes induction
on topological rank, and obligations are unions over reachable descendants.

# Recommended revised status (claimant's table)

| Claim | Recommended status |
| --- | --- |
| Remove Z1 from T3/T4 | `[PROVEN]` |
| Replace global exact (D_N) by admissible local (B(N)) | `[PROVEN]` |
| Replace (8D_N) by (8(B(N)-1)) | `[PROVEN]` |
| Replace full witness windows by witness empty sets | `[PROVEN]` |
| Pathwise rather than merely global-(T) conclusion | `[PROVEN]` |
| T6 extendable-hit kernel, no core | `[PROVEN-DRAFT]`; likely promotable after review |
| LOSS witnesses of size at most (3/6) | `[PROVEN]` |
| Per-window exposure and virgin radius (8(E^D-6)) | `[PROVEN-DRAFT]` |
| Branch-indexed substitute children | `[PROVEN-DRAFT]` |
| Full (F+H_W) forced-hit accounting | Still open |

The strongest practical verifier rule suggested: no longer the current T4
union but

[
Z_{\mathrm{ranked}}
=
Z_{\mathrm{obligation}}
\cup
Z_{\mathrm{touch}}
\cup
Z_{\mathrm{virgin}},
]

with no mandatory hitting term, local/cell/window deadlines, and optionally
a substitute-child annotation for every dismissal.

The remaining genuinely difficult tightening is the document's (F+H_W)
problem: proving a branchwise upper bound on how many defender placements
can be devoted to a particular window after accounting for compulsory threat
hits. None of the results above assumes that unresolved claim.
