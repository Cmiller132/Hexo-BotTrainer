# R-ST10-REV — Hostile review of `STRATEGY_STEALING_ROUND10.md`

## Method and proof boundary

**Reviewed artifact.** The review is pinned to branch `hunt/gap-raw` at
landing commit `1bdaf02a9ea631f399fd99d9d8049c973de5898c`. Its immediate
parent is `8424e9a474f160e4d30ec73cf6b655e1da741554`, and the task-named
input `88bca52d2a52dbcda5da60db81f00f69ad6cfcd7` is an ancestor. The
Round-10 artifact first appears at that landing. Its landed and current Git
blob is `3634d5fec9de8b3f1d8038535294f118fbe83fc7`; the worktree copy has
SHA-256
`c6073fa57b1208dba09c36a552db0473397019588817bbf30c338f0450a635aa`.
During final assembly the shared branch advanced externally to
`52ae3e05d94a3a5a38e3d7fd6a4704cd7fb0e86b`. The requested landing remains
its ancestor, the Round-10 blob stayed identical, and an explicit path-limited
diff found no change to any required Round5–Round10 artifact or review. This
audit therefore remains pinned to `1bdaf02a`.

Default posture was **REFUTE**. I read the required corpus in the prescribed
order and in full:

1. `STRATEGY_STEALING_ROUND5.md`, including binding section 44, then
   `STRATEGY_STEALING_REVIEW_ROUND5.md`;
2. `STRATEGY_STEALING_ROUND6.md`, including binding section 53 and
   Definition 46.1 terminal closure, then
   `STRATEGY_STEALING_REVIEW_ROUND6.md`;
3. `STRATEGY_STEALING_ROUND7.md`, including binding section 63, then
   `STRATEGY_STEALING_REVIEW_ROUND7.md`;
4. `STRATEGY_STEALING_ROUND8.md`, including rewritten section 68 and
   binding section 73, then `STRATEGY_STEALING_REVIEW_ROUND8.md`;
5. `STRATEGY_STEALING_ROUND9.md`, including binding section 84, then
   `STRATEGY_STEALING_REVIEW_ROUND9.md`; and only then
6. `STRATEGY_STEALING_ROUND10.md`.

No `GAP_RAW_*` file was opened or used as evidence. No Cargo command, Lean
command, proof harness, test harness, solver, or executable game search was
run. All cadence transitions, stone counts, window intersections, exclusion
bounds, representative ordered pairs, and obstruction cases below were
recomputed by hand. No commit was created, and unrelated worktree entries
were left untouched.

The binding baseline is retained throughout: append-before-win terminality;
no continuation after a terminal append; S49's sixth-shadow-stone versus
fifth-real-stone barrier; S58's one-event common-phase \((2,3)\) cliff; S59's
restricted corridor; the section-68 interface map rather than causal
synthesis; and S70/S70.1/S70.2 as the reviewed finite first-cycle results.
In particular, “carrier succeeds,” “carrier is ejected,” and “physical
closure occurs on a selected history” are not treated as claims about the
outcome of a strategy \(\sigma\).

**Overall verdict: REFUTED AS STATED (LOCALIZED).** S71 really crosses the
forced cliff on its delimited asynchronous carrier class. At the fixed
post-\(k_4\) S70 bridge, S73 proves that for every immediate two-move sleeve
portfolio there exists a legal \(S\) turn that blocks the full near-window
family or wins. S75 supplies one genuine post-ingress second cycle on its
expressly conditional nonempty class, and S76/S77 give the claimed local
ordered-pair classifications. S77.1 itself is false as printed: its
\(\tau\leq1\) gate is exact for a generic strict S59 entrance, but not for
“S70's full strict current-\(R_1\) entrance,” whose advertised disposition
requires \(\tau=0\). The repair is one gate or one target name. The artifact
also omits its actual landing identity.

## Numbered findings

### Finding 1 — NOTE: S71 crosses the S58 cliff by a legal asynchronous history

> “Freeze the physical shadow board and advance only the real board
> according to the following finite schedule.”

**Independent recomputation.** Immediately after a nonterminal
\(z_3/k_3\), the real board has \((|F|,|S|)=(4,4)\), the shadow board has
\((|\widehat F|,|\widehat S|)=(5,5)\), and both boards are at \(F\)
`SecondStone`. Freezing the shadow board and advancing only the real board
gives:

| Physical append | Real counts after append | Real next phase | Shadow state |
|---|---:|---|---|
| \(k_4\) | \((5,4)\) | \(S\) `FirstStone` | \((5,5)\), \(F\) `SecondStone` |
| \(y_1\) | \((5,5)\) | \(S\) `SecondStone` | frozen |
| \(y_2\) | \((5,6)\) | \(F\) `FirstStone` | frozen |
| nonwinning \(r_1\) | \((6,6)\) | \(F\) `SecondStone` | frozen |

Thus common owner and common microphase are restored before the final paired
event. The real append \(r_2\) and pending shadow append \(z_4\) are then both
legal \(F\)-role `SecondStone` appends. They leave counts
\((7,6)\) real and \((6,5)\) shadow and, on the live
nonwinning-\(r_1\) co-terminal subbranch, both terminate at that same paired
event. This is the first common phase after the pause;
there is no hidden phase advance, third stone, or continuation past
terminality.

This construction genuinely reaches the disposition unavailable to all four
previously named conversions. The reserve, ordinary-two-append,
section-53-closure, and terminal-moment-S63 mechanisms tried to repair the
same common-phase stream without enough real physical history. S71 instead
adds an outer real \(k_4,Y,r_1\) history, restores common phase, and supplies
six real \(F\) stones before the final event; the seventh real append
\(r_2\) then completes a six-window while the sixth shadow append \(z_4\)
completes its own. It bypasses S58's synchronous hypotheses; it does not
contradict S58.

**Proposed repair:** none.

### Finding 2 — NOTE: S75 is a genuine second \(R_1\) cycle, not S70's entrance relabeled

> “One complete further reserve-handler-generated \(R_1\) cycle.”

**Independent recomputation.** On disposition 3, S70 ends after constructing
the first strict current-\(R_1\) entrance. S75 starts at that endpoint and
executes the new block
\[
J_2=(p_1,p_2,y_5,y_6)
\]
with paired shadow appends \(z_5,z_6,c,T(y_5)\). At its starting entrance the
real/shadow counts are \((5,6)\) and \((6,7)\), with both boards at \(F\)
`FirstStone`. The complete phase ledger is:

| Paired event | Real counts / next phase | Shadow counts / next phase |
|---|---|---|
| \(p_1/z_5\) | \((6,6)\), \(F\) second | \((7,7)\), \(F\) second |
| \(p_2/z_6\) | \((7,6)\), \(S\) first | \((8,7)\), \(S\) first |
| \(y_5/c\) | \((7,7)\), \(S\) second | \((8,8)\), \(S\) second |
| \(y_6/T(y_5)\) | \((7,8)\), \(F\) first | \((8,9)\), \(F\) first |

The endpoint is therefore a new current-\(R_1\) entrance reached after a
whole post-ingress \(F\)-pair and \(S\)-pair. The recurrence is temporal and
physical; it is not S70's first entrance with a new label. It is, however,
exactly one further cycle, not indefinite recurrence.

**Proposed repair:** none. For maximal clarity, future citations should say
“one full \(R_1\) cycle after S70's constructed ingress.”

### Finding 3 — NOTE: S75's legality bounds close, and nonemptiness is honestly conditional

> “Theorem S75 (one full S70 second cycle) [PROVEN on the nonempty
> \(\mathcal Q_{70}^{\,2}\) class].”

**Independent recomputation.** The spare line \(W_1\) has one old real-\(F\)
anchor and no other real-\(F\) stone on its full axis. Choosing its two holes
\(p_1,p_2\) off \(W_{\rm in}\) is legal within the radius-eight store. For
\(p_1\), every other engine axis through it misses the old anchor \(x\).
For \(p_2\), every other axis through it meets the \(W_1\) axis only at
\(p_2\), so it misses both \(x\) and \(p_1\). Such an axis therefore has at
most the four old off-axis \(F\) stones plus the current \(p_i\), while
\(W_1\) itself has at most three \(F\) stones. Neither append can make six.
The prior deficit-one reserve window \(W_{\rm in}\) is untouched, so
\(\mu_R=1\) and \(\mathrm{RES}_1\) persist.

For the next bridge cell \(y_5\), six real \(S\) stones give
\(6\cdot18=108\) incidences. A window with at least four of them consumes at
least four incidences, so there are at most \(27\) such windows and at most
\(54\) non-\(S\) cells in their union. Eight shadow \(S\) stones give
\(8\cdot18=144\) incidences; at most
\(\lfloor144/5\rfloor=28\) shadow five-\(S\) windows can make their missing
cells forbidden for \(c=T(y_5)\). The round excludes at most
\[
13+15+6+6+54+28=122<217
\]
cells: real and shadow occupancy, the two six-windows to preserve, unsafe
real cells, and unsafe translated cells. Thus an admissible \(y_5\) exists.

After \(y_5/c\), real/shadow occupancy is \(14/16\), and real \(S\) has seven
stones. Its \(7\cdot18=126\) incidences support at most \(42\) windows with
at least three \(S\) stones; their non-\(S\) cells contribute at most
\(42\cdot3=126\) exclusions. Hence
\[
14+16+6+6+126=168<217
\]
leaves an admissible \(y_6\). The chosen cells and their translated copies
are fresh/legal, the real pair stays nonterminal, \(T(y_5)\) is nonwinning,
and no urgent endpoint window remains, giving \(\tau_E=0\). Because
\(y_5,y_6\) avoid \(W_{\rm in}\) and \(W_1\), S69's scalar reserve witness
survives.

The class definition still assumes that \(z_5,z_6\) are nonterminal and
distinct from \(c\), and that the inherited append \(c\) is nonterminal.
Those are selected quiet-history premises, not facts forced for every play.
The diagnostic coordinates supplied in the round instantiate the physical
class, so its board-geometric nonemptiness is real; the text correctly does
not claim that every alleged winning strategy reaches that witness. In the
displayed \(T(q,r)=(q+2,r)\) trace, \(W_0\) is the r-zero interval
\(0\leq q\leq5\) with unique hole \((3,0)\), and \(W_1\) is the q-zero
interval \(-5\leq r\leq0\) with old anchor \((0,0)\). The new real pair
\((0,-5),(0,-4)\), shadow pair \((-6,0),(-5,0)\), inherited shadow
certificates \((12,5),(15,7)\), and returning real pair
\((13,7),(16,8)\) are mutually fresh at their append times, supported, and
nonterminal; the next image \((18,8)\) is fresh/legal. This directly
rechecks the advertised physical nonempty member.

The complementary event ledger also respects Definition 46.1. A terminal
\(z_5\) or \(z_6\) is paired only with the handler's already-associated
real unique-hole append and both traces stop; section 53 supplies no extra
stone. A nonterminal \(z_i=c\) leaves persistent wrong-role
\(\widehat F\)-occupation and ejects the literal recurrence. A
certificate-created \(\widehat S\) win is an immediate physical stop and is
never continued. Only the all-nonterminal row enters S75.

**Proposed repair:** none. Do not drop “nonempty \(\mathcal Q_{70}^{\,2}\)”
or promote this one cycle to arbitrary recurrence.

### Finding 4 — NOTE: S76 exactly classifies ordered pairs at the empty-debt entrance

> “An ordered pair \((u,v)\) gives a live literal inherited-\(T\) guarded
> prefix ... if and only if:”

**Independent recomputation.** At \(p^{(0)}\), after S70's first line seed,
the real board has \(F=3,S=2\), the shadow board has
\(\widehat F=4,\widehat S=3\), both are at \(S\) first, and the debt register
is empty. The literal inherited-\(T\) schedule is therefore
\[
(u,f(u)),\quad(v,T(u)),
\]
with the index recomputed after the first real append. All old and new
counts are below six on the relevant append roles, so the gates need only
test sequential real legality, translated freshness/legality, the protected
line restriction, and the endpoint transversal condition.

Because the endpoint contains exactly the four real \(S\) stones, an urgent
window is precisely an \(F\)-unblocked six-window containing all four.
Writing that family as \(\mathcal D_0(u,v)\), the advertised
\(\tau_E=0\) condition is exactly
\(\mathcal D_0(u,v)=\varnothing\).

The alternative-line reserve formula also recomputes exactly. Let
\(\mathcal A_3\) be the pre-pair real-\(F\)-unblocked deficit-three windows
and
\(\mathcal B_u=\{W\in\mathcal A_3:u\notin W\}\). After the ordered pair, an
adequate alternative window survives if and only if some member of
\(\mathcal B_u\) also avoids \(v\), equivalently
\[
\mathcal B_u\neq\varnothing
\quad\text{and}\quad
v\notin\bigcap\mathcal B_u.
\]

Three representative entries check the classification rather than merely
its notation:

- With \(T(q,r)=(q+2,r)\),
  \(u=(3,3),v=(5,2)\), filler \(f(u)=(0,1)\),
  \(T(u)=(5,3)\), and \(T(v)=(7,2)\), the four real \(S\) stones are not
  collinear, so \(\mathcal D_0=\varnothing\); all freshness and
  off-\(W_0\) gates pass.
- For \(u=(0,3),v=(0,4)\), the q-constant window
  \((0,1),\ldots,(0,6)\) contains all four real \(S\) stones. It leaves two
  holes and gives \(\tau_E=1\), so the service gate correctly rejects it.
- For \(u=(3,0),v=(4,1)\), the named \(W_0\) restriction rejects \(u\), even
  though the r-zero interval starting at \(-3\) contains the three old real
  \(F\) stones and avoids both new \(S\) stones. The alternative-line
  reserve test therefore passes, exactly as the round records.

**Proposed repair:** none.

### Finding 5 — NOTE: S77's scalar ordered-pair test is necessary and sufficient

> “\(\mathrm{RES}_1\) holds at the exit if and only if
> \(\mathcal B_u\ne\varnothing\) and
> \(v\notin\bigcap\mathcal B_u\).”

**Independent recomputation.** At the protected ingress, the real/shadow
counts are \(F=5,S=4\) and \(\widehat F=6,\widehat S=5\), both boards are at
\(S\) first, the debt is \(e\), \(T(e)\) is fresh/legal, and the protected
real window \(W_*\) has deficit one. The only literal singleton-debt
schedule is
\[
(u,T(e)),\quad(v,T(u)).
\]
S77's six gates correctly test, in sequential order, real first safety,
real-second legality/nonterminality, old-certificate
freshness/legality/nonwinning, current-copy
freshness/legality/nonwinning, next-certificate freshness/legality,
\(\tau_E\leq1\), and the scalar reserve formula. S77.1 separately adds
\(u,v\notin W_*\) for preservation of the designated reserve.

Let \(H_u\) be the shadow state after the two prescribed shadow appends.
Then \(a(u)=1\) when \(\mu_H(H_u)=1\), and \(a(u)=2\) when
\(\mu_H(H_u)\geq2\). For the pre-pair real family
\(\mathcal A_{a(u)}\), set
\(\mathcal B_u=\{W\in\mathcal A_{a(u)}:u\notin W\}\). An adequate real
window remains after \(v\) exactly when some \(W\in\mathcal B_u\) also
avoids \(v\). This is equivalent to the displayed nonempty/intersection
test, so the reserve formula is set-theoretically exact rather than only
sufficient.

At the round's diagnostic state, \(u=(8,4),v=(10,5)\) gives
\(T(e)=(7,2),T(u)=(10,4),T(v)=(12,5)\); the pair is legal and nonterminal,
\(T(e)\) and \(T(u)\) are fresh/legal/nonwinning, \(T(v)\) is fresh/legal
for the next certificate, \(\tau_E=0\), and both real cells avoid \(W_*\).
Conversely, with the same \(u=(8,4)\) but
\(v=(3,0)\), the sequential and translated gates still pass and
\(\tau_E=0\), while \(a(u)=2\). Every relevant deficit-two axis interval
starting at \(-1,0,1\) contains index \(3\), so
\(v\in\bigcap\mathcal B_u\) and the scalar reserve correctly fails.

**Proposed repair:** none.

### Finding 6 — REFUTED (localized): S77.1 overidentifies a generic strict entrance with S70's stronger endpoint

> “The literal schedule reaches S70's full strict current-\(R_1\) entrance
> predicate if and only if gates 1–6 hold and the seventh gate ... also
> holds.”

**Independent recomputation.** Binding S70's disposition 3 advertises
\(\tau_E=0\). S77.1 instead retains S77 gate 5, which requires only
\(\tau_E\leq1\). The round itself notes that replacing this gate by
\(\tau_E=0\) yields the stronger S70 endpoint, but its corollary title and
iff sentence still call the weaker seven-gate target exactly “S70's.”

The mismatch has a concrete witness at the section-87.5 diagnostic state.
Take \(T(q,r)=(q+2,r)\), debt \(e=(5,2)\), protected
\(W_*=\{(q,0):0\le q\le5\}\), and
\[
u=(0,3),\qquad v=(0,4).
\]
The real first append is legal and first-safe; the second is legal and
nonterminal. The old certificate \(T(e)=(7,2)\), current copy
\(T(u)=(2,3)\), and next certificate \(T(v)=(2,4)\) are fresh/legal, and
the appends that occur are nonwinning. Both \(u,v\) avoid \(W_*\), so the
protected deficit-one witness survives. But the q-constant window from
\((0,1)\) through \((0,6)\) has four \(S\) stones and two holes, hence the
endpoint has \(\tau_E=1\), not \(0\). All seven printed gates pass while
S70's advertised endpoint does not.

This does not refute S77's generic strict-\(R_1\) classification. It refutes
only the exact identification made by S77.1.

**Proposed repair:** either rename S77.1 as the exact classification of a
generic S59 strict current-\(R_1\) entrance, for which
\(\tau_E\leq1\) is correct, or replace gate 5 by \(\tau_E=0\) everywhere
the target is called S70's full endpoint.

### Finding 7 — NOTE: “both literal schedules” is exhaustive only—and exactly—over the claimed schedule space

> “S76–S77.1 classify all ordered pairs under the two literal fixed-\(T\)
> schedules at their fixed S70 S-turns.”

**Independent recomputation.** The inherited-\(T\) controller has two
register states at the two advertised entrances. With empty debt at
\(p^{(0)}\), the first shadow append is the prescribed filler \(f(u)\) and
the second is \(T(u)\). With singleton debt \(e\) at the protected ingress,
the first shadow append must discharge \(T(e)\) and the second is \(T(u)\).
The ordered-pair index is recomputed after \(u\), so these are the two
literal schedules, not two simultaneous-board approximations.

This is exhaustive for the declared \(\mathcal Q_{\rm TPAIR}\) schedule
space. It does not classify a direct copy on the first microstep, a different
filler policy, changes of \(T\), nonisometric or partial recodings, longer
lag, or a controller with a different debt state. Round 10 expressly leaves
those outside its theorem, so no silent universalization occurs.

**Proposed repair:** none beyond the S77.1 target-name correction in
Finding 6.

### Finding 8 — MINOR: the provenance ledger does not record the landed artifact

> “`LANDING_COMMIT=<POST-REVIEW-FOLD>`”

**Independent recomputation.** Section 91.1 retains a placeholder instead
of the artifact's actual landing identity. Git history shows that
`STRATEGY_STEALING_ROUND10.md` first appears at
`1bdaf02a9ea631f399fd99d9d8049c973de5898c`, with immediate parent
`8424e9a474f160e4d30ec73cf6b655e1da741554`; the task-named input
`88bca52d2a52dbcda5da60db81f00f69ad6cfcd7` is an ancestor. The landed
blob and current worktree blob are both
`3634d5fec9de8b3f1d8038535294f118fbe83fc7`.

This is a reproducibility defect, not a mathematical one.

**Proposed repair:** replace the placeholder with the exact landing,
immediate parent, task input, blob identity, and SHA-256 above.

### Finding 9 — NOTE: the fast outcome and \(NL_F\) remain open without hidden outcome inflation

> “The global post-cliff outcome remain **OPEN**.”
>
> “Global target: \(NL_F\) remains **OPEN**.”

**Independent recomputation.** S71 proves existence and correctness of a
bounded carrier on its selected asynchronous class. S73 proves the
\(\forall\)-portfolio/\(\exists\)-blocking-or-winning-turn negative only for
immediate two-move outer sleeves; it does not say every \(S\)-pair blocks. S75 performs
one further selected quiet cycle. S76/S77 classify local ordered inputs
under fixed literal schedules. None supplies a causal response to every
opponent history, forces entry into its local class, proves indefinite
recurrence, or converts a carrier terminal into a theorem about the alleged
strategy \(\sigma\).

Whenever a shadow append wins while its real mate does not, the text records
carrier failure. Whenever an actual candidate-owned real append wins, the
text uses that physical terminal only on the selected history. The round
does not use the interface map to explain a global outcome and does not
infer \(NL_F\). The declared open status is therefore internally consistent.

**Proposed repair:** none.

### Finding 10 — NOTE: S71's positive/negative split is exhaustive and its scope stays bounded

> “\(\mathrm{ASYNC\mbox{-}CYCLE}_4\) reaches either a sound earlier
> real-\(F\) stop or a common-phase section-53 co-terminal closure if and
> only if \(\mathcal C_2(R_Y)\) is nonempty.”

**Independent recomputation.** After the bridge pair \(Y\), let
\(\mathcal C_2(Y)\) be the real \(F\)-unblocked six-windows of deficit at
most two.

If a selected window has deficit two, \(r_1\) fills one supported hole. If
that append wins, the carrier has already reached a sound earlier terminal
disposition. Otherwise \(r_2\) fills the unique remaining hole and wins. If
the selected window has deficit one, preserve its hole and choose \(r_1\)
elsewhere so that it is nonwinning. Before \(r_1\), eleven cells are
occupied. With five real \(F\) stones, at most two cells are immediate
winning cells. Any such cell would complete a window containing all five
existing \(F\) stones, so those five must occupy five positions of one
six-window on one engine axis. An interior gap gives one completion; five
consecutive stones give the two endpoint extensions, the maximum. The
stated support ball has 217 cells; the eleven occupied cells, at most two
winning cells, and the one preserved hole exclude at most fourteen cells.
A legal nonwinning filler therefore exists. The final \(r_2\) fills the
preserved unique hole.

On the other horn, every real \(F\)-unblocked window has deficit at least
three immediately before \(R\). Two \(F\) appends reduce any one deficit by
at most two, so neither \(r_1\) nor \(r_2\) can produce a real six. The
pending \(z_4\) can then terminate only the shadow side, ejecting this
carrier. The split is exact for the fixed selected bridge pair and the
declared local support conditions.

No step quantifies over arbitrary \(S\) play. The theorem assumes the
post-\(z_3/k_3\) live state, a legal nonterminal \(k_4\), a selected legal
nonterminal bridge pair \(Y\), local supported holes/fillers, and the stated
pending terminal \(z_4\) with its final paired event. The real terminal on
the positive horn is then proved from the selected window, not assumed.
Those hypotheses define the
delimited \(\mathcal Q_{\rm ASYNC}^4\) class. The witness in S71.1 is also
physical: real \(F\) at \((0,0),(1,0),(2,0),(3,0)\), then
\(k_4=(4,0)\); \(S\) at \((6,2),(7,2)\); filler \(r_1=(8,2)\); and final
\(r_2=(5,0)\), paired with shadow \(z_4=(6,0)\). The \(S\)-pair is
nonterminal, \(r_1\) is off the reserve and nonwinning, and each side gets
its q-axis six only at the asserted physical append.

**Proposed repair:** none. Preserve the theorem's explicit selected-history
and bounded-support qualifiers in any later citation.

### Finding 11 — NOTE: S73's two-move immediate-sleeve obstruction is exhaustive

> “Real \(S\) has a legal turn which either wins physically for \(S\) or
> blocks every sleeve in every disjoint or partially overlapping
> \(\mathrm{OUTER\mbox{-}SLEEVE}_2\) portfolio.”

**Independent recomputation.** Let \(W_0=[0,5]\) on its engine axis and let
the five real \(F\) stones occupy
\(P=[0,5]\setminus\{g\}\). Any \(F\)-unblocked window of deficit at most two
contains at least four members of \(P\). A distinct engine line intersects
the axis of \(W_0\) in at most one cell, so every such window is a same-axis
length-six interval. Direct enumeration gives:

| Gap \(g\) | admissible interval starts | hole sets | minimum transversal \(H_g\) |
|---:|---|---|---|
| 0 | \(-1,0,1,2\) | \(\{-1,0\},\{0\},\{6\},\{6,7\}\) | \(\{0,6\}\) |
| 1 | \(-1,0,1,2\) | \(\{-1,1\},\{1\},\{1,6\},\{6,7\}\) | \(\{1,6\}\) |
| 2 | \(-1,0,1\) | \(\{-1,2\},\{2\},\{2,6\}\) | \(\{2\}\) |
| 3 | \(-1,0,1\) | \(\{-1,3\},\{3\},\{3,6\}\) | \(\{3\}\) |
| 4 | \(-2,-1,0,1\) | \(\{-2,-1\},\{-1,4\},\{4\},\{4,6\}\) | \(\{-1,4\}\) |
| 5 | \(-2,-1,0,1\) | \(\{-2,-1\},\{-1\},\{5\},\{5,6\}\) | \(\{-1,5\}\) |

If a listed transversal cell is already \(S\)-owned, every window relying
on it is already blocked; delete that coordinate from the transversal and
delete the already blocked intervals from the live near-window family. The
remaining zero-, one-, or two-cell transversal consists of empty legal cells
through the five-\(F\) post-\(k_4\) position. A two-cell transversal is a
legal \(S\)-pair unless its second append wins. A one-cell transversal can
be followed by a nonwinning filler, available after excluding occupied cells
and at most two immediate \(S\)-winning cells from the 217-cell support
ball. If no retained transversal cell remains, the full near-window family
is already \(S\)-blocked; any fresh legal padding pair either wins on its
second append or leaves that empty family empty. If a transversal or padding
\(S\)-append wins, the real game terminates against this carrier. If it does
not, every deficit-at-most-two window is blocked and S71 lands on its
negative horn.

This exhausts the physical carrier family actually claimed:
\(\mathrm{OUTER\mbox{-}SLEEVE}_2\), including partial sleeves, disjoint
choices, and post-hoc reassignment. A purported single immediate
outer-sleeve carrier surviving the obstruction would need a near window
outside the enumerated same-axis intervals or an unhit hole set; neither
exists. In quantified form, for every such portfolio
\(\mathcal P\) (a subset of the enumerated full family), the displayed
\(H_g\), with the stipulated padding in the zero- and one-cell cases,
supplies one legal real-\(S\) turn that hits the full family and therefore
\(\mathcal P\), or wins earlier. Longer lag, more than two sleeve stones, a
different vehicle, or deeper strategic response is outside S73 and remains
open.

**Proposed repair:** none.

### Finding 12 — NOTE: S71/S73 preserve the carrier-versus-\(\sigma\) boundary

> “This is a carrier success and a real-\(F\) outcome on that history; it
> is not a defeat of \(\sigma\).”

**Independent recomputation.** On S71's positive horn the constructed
physical real history reaches either an earlier terminal \(r_1\) append or
the paired \(r_2/z_4\) terminal event. This proves that the delimited
asynchronous vehicle crosses the S58 carrier cliff. It does not prove that
the vehicle can be selected causally against every opponent pair, that every
\(\sigma\)-history enters its domain, or that the fast game is an \(F\)
win. On S73's negative horn the chosen \(S\) transversal/padding pair either
wins the real game or blocks all near real windows so that the pending shadow
terminal ejects the vehicle. That defeats the immediate-sleeve carrier; it
is not a theorem about every strategy.

The Round-8 section-68 error is not repeated. Round 10 does not infer that
one interface's success or failure causally explains the global outcome.
Its own caveats explicitly separate finite carrier maps, selected histories,
and actual outcome. Its local explanation—“It crosses by adding legal real
history and returning to a common phase”—is supported by the recomputed
append ledger above and is not a claimed explanation of why the fast outcome
stays open.

**Proposed repair:** none.

## Per-theorem verdicts

| Theorem | Audit disposition | Exact boundary or repair |
|---|---|---|
| S71 | **CONFIRMED as PARTIAL** | The bounded \(\mathcal Q_{\rm ASYNC}^4\) vehicle legally restores common cadence and supplies the missing real physical history; it crosses S58 on that delimited selected class, not for arbitrary \(S\) |
| S73 | **CONFIRMED** | At the fixed post-\(k_4\) S70 bridge, every \(\mathrm{OUTER\mbox{-}SLEEVE}_2\) portfolio admits a legal blocking-or-winning \(S\) turn; the same-axis census and all zero/one/two-cell transversal cases are exhaustive |
| S75 | **CONFIRMED conditionally** | One genuine full post-S70 \(R_1\) cycle on nonempty \(\mathcal Q_{70}^{\,2}\); quiet inherited appends and the selected spare-line premises are assumptions, and no indefinite recurrence follows |
| S76 | **CONFIRMED locally** | Necessary-and-sufficient ordered-pair test for the literal empty-debt inherited-\(T\) schedule, with sequential recomputation of the second index |
| S77.1 | **REFUTED AS STATED; LOCAL REPAIR** | Exact for a generic S59 strict entrance with \(\tau_E\leq1\); false for the stronger S70 endpoint unless gate 5 is changed to \(\tau_E=0\) |

## Overall verdict

**REFUTED AS STATED (LOCALIZED).** The headline result survives hostile
recomputation: S71 is the first named bounded asynchronous carrier in this
campaign that actually crosses the S58 forced cliff, rather than merely
rephrasing or approaching it. S73 proves the exact
\(\forall\)-immediate-portfolio/\(\exists\)-blocking-or-winning-turn
obstruction without pretending that every \(S\)-pair blocks or that longer
carriers are excluded. S75 is a real second cycle, while remaining conditional and
finite. S76 is exact at its local schedule, and the underlying S77
classification is exact. But S77.1's printed iff admits the concrete
\(\tau_E=1\) counterexample in Finding 6 and is therefore refuted for its
claimed S70 target; replacing gate 5 by \(\tau_E=0\), or renaming the target
as generic S59, repairs it. The missing landing provenance in Finding 8 is
the only additional erratum.

The fast outcome and \(NL_F\) remain **OPEN**. A carrier crossing is not a
\(\sigma\)-outcome proof, a carrier obstruction is not a universal game
obstruction, and a diagnostic nonempty selected history is not a forced
strategy history.

## Exact unresolved obstacles

1. **Arbitrary-\(S\) cliff crossing.** S71 handles only its delimited
   post-\(z_3/k_3\) asynchronous class. S73 rules out arbitrary-\(S\)
   coverage for every immediate two-move portfolio by supplying at least one
   blocking-or-winning \(S\) turn. Longer lag, deeper real threat families,
   and different outer vehicles remain unclassified.
2. **Forced access to the S71 class.** No causal controller is proved to
   reach the required live \(k_4,Y,R\) geometry against every opponent
   response while retaining all freshness, support, and nonterminal
   premises.
3. **General bridge terminality and P5R.** S71 selects a nonterminal real
   \(S\)-bridge. A general carrier still owes protection against an earlier
   real \(S\) terminal and common-only real-win transfer at arbitrary
   bridge states.
4. **More than one post-ingress recurrence.** S75 completes exactly one
   second \(R_1\) cycle. A third cycle, indefinite iteration, and closure of
   the selected quiet class under its own handler are not proved.
5. **Forced membership in \(\mathcal Q_{70}^{\,2}\).** Nonterminal,
   distinct \(z_5,z_6,c\), the spare line, the old certificate, and the
   nonwinning inherited append are conditional premises, not consequences
   of an alleged winning strategy.
6. **Ordered pairs rejected by S76/S77.** The exact tables diagnose failure
   but do not supply a replacement response for every rejected pair.
7. **Schedule space beyond the two literal inherited-\(T\) cases.** Direct
   copying, other filler rules, changing isometries, nonisometric or partial
   recodings, different debt cardinality, and longer phase lag remain open.
8. **Pre-checkpoint P3 and coverage outside strict
   \(\mathcal A_{\rm FS2}\).** Missing, blocked, wrong-role, illegal,
   unsupported, phase-lagged, unreflected-terminal, and high-transversal
   states are avoided by the selected classes, not handled. Recurring P3
   before an admitted checkpoint remains global work.
9. **Reverse legality and persistent support.** Any broader or inverse
   spatial carrier still owes collision, support, S13, S18, and
   append-before-terminal checks at the actual current board.
10. **Universal service, lock, and window maintenance.** A recurring
    controller must combine P2/P3, P5/P5R, reserve preservation,
    reassignment, arbitrary opponent-created windows, high transversal
    number, and permanent fencing. Canonical \(F\)-LOCK and service
    compatibility remain unproved; the local classifications do not
    construct them.
11. **Global causality and strategy domain.** The finite choices in S71 and
    S75 are causal on their selected histories, but no global rule maps
    every legal \(S\) history into a covered carrier state.
12. **Universal shadow-\(F\) fidelity, the fast outcome, and \(NL_F\).**
    Post-cliff play, the S49 terminal mismatch outside the bounded vehicle,
    and the outcome against a full \(\sigma\) remain undecided. Nothing in
    Round 10 proves either an \(F\)-win or the global negative \(NL_F\).
