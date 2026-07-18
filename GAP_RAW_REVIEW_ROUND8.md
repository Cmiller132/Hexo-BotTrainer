# R-G6-REV — Round-8 hostile review

**Reviewed branch / artifact:** `hunt/gap-raw`,
`c57da44286f75feb236e6da6c55cdd53e5ec2e68`

**Reviewed document:** `GAP_RAW_PROOF_ROUND8.md`

**Artifact's named authoring input:**
`a8a0b92d641b690b63d43f049d2b4c2fa0d4e9c1`

**Artifact identity:** Git blob
`6c460713444ff94758b5416debdfa5b27fa878ef`; SHA-256
`4d8f376c4644ddc9f2987873febb554fbea4f2e495364aca8f8d316e62f42996`.

**Method.** First-principles proof audit with the requested default
presumption against the positive all-response bound. I read, in the required
order and in full, `GAP_RAW_PROOF_ROUND6.md` including binding §58,
`GAP_RAW_REVIEW_ROUND6.md`, `GAP_RAW_PROOF_ROUND7.md` including binding §67,
`GAP_RAW_REVIEW_ROUND7.md`, and then the reviewed round-8 artifact. I consulted
only the needed pure-count-two and sealed-pencil material in rounds 4–5. I did
not read any `STRATEGY_STEALING_*` file as evidence. Every inventory, response
effect, residual, transversal, legality chain, `tau`, `TEMPO`, and `M` value
below was recomputed by hand. No Cargo command, Lean build, harness,
game/search program, or machine enumeration was run.

**Overall verdict:** **SOUND-WITH-MINOR-ERRATA.** The central positive theorem
survives: no legal Attacker response after the cap was found with `M>=3`, and
the proof's effect quotient really covers the infinite response set, including
remote and split responses. R8.1 and R8.2 are confirmed with exact risk two;
§71's one-level deviation tree and §72's exact-input landing theorem also
survive. Four localized repairs are needed: one false adjacency sentence, an
unsafe-filler omission in the exact `C_cap` membership argument, a ledger row
that incorrectly leaves the final `P_stock` cap open despite binding R7.2, and
the absent landed provenance.

## Numbered findings

### 1. NOTE — the cap is legal, servicing, non-hub, and has the exact `(n_1,n_2)=(8,12)` inventory

**Quoted claim:**

> “At `P_0^pl`, the non-hub ordered pair
> `a^dagger=((0,-1),(1,1))` is legal” and after it “the local alive profile is
> exactly `n_1=8,n_2=12`, with no higher count.” (R8.1, L14.1)

**Independent recomputation.** At `P_0^pl`, `I=empty`, so every legal pair is
in `Serv`. The first cap cell `(0,-1)` lies in four live common windows on
`q=0`. Its placement does not touch `q=1`, so `(1,1)` remains in four live
common windows there; it is also at hex distance three from the first cap cell.
Both placements are therefore sequentially legal. Neither is the hub
`h=(10,0)`.

Before the cap there are five count-two windows on each of
`q=0,q=1,s=0,s=1`, two count-one extremes on each of those four lines, and six
count-one windows on each bridge row `r=1,r=-1`. The first cap deletes four
`q=0` common windows, one vertical extreme, and five `r=-1` bridge windows.
The second makes the disjoint reflected deletions. The survivors are exactly:

```text
count two:  five on s=0, five on s=1,
            q=0 with 0<=r<=5, q=1 with -5<=r<=0;
count one:  four diagonal extremes, two vertical outer extremes,
            r=1 with -5<=q<=0, r=-1 with 1<=q<=6.
```

Thus `n_2=20-4-4=12` and `n_1=20-6-6=8`. The anchors, old ray, and axial
cleanup cells meet none of these surviving windows, so this is the complete
global alive inventory, not merely an isolated local count. Its graded tier is
pure count two. Either intact diagonal pencil gives the standard demand-two
flank witness, while L10.4 gives the reverse bound, so the cap handoff has exact
`TEMPO=2`. Since L12.2 gives value two to every action at `P_0^pl`, the cap is
also an immediate-value minimizer.

**Proposed repair:** none.

### 2. NOTE — §71 absorbs the root reply and classifies the complete raw action quotient

**Quoted claim:**

> “Every initial Defender pair leaves one `B_j` untouched, and Attacker can
> legally reach an exact raw adjacent-pair epoch there.” (L14.7)

> “This exhausts the infinite legal-pair universe by its finite survivor-block
> effect.” (L14.8)

**Independent recomputation.** Every point in the translated launch set `K`
has row coordinate between `30j-1` and `30j+1`. A six-window meeting it stays
inside the band

```text
30j-6 <= r <= 30j+6.
```

The three bands at spacing thirty are disjoint, and the selected launch's
anchor `(0,30j+8)` is outside its band. Each initial Defender cell can therefore
meet only one enlarged footprint, so two cells leave one complete `B_j`
untouched. The first launch cell is at distance eight from its anchor and the
second is adjacent. Every window through the launched adjacent pair belongs to
the untouched footprint. The returned inventory is exactly five common
count-two row windows plus twenty-six count-one windows, with `tau=0`.

At that raw epoch, a row reply at `-4,-3,-2,-1` deletes a prefix of respectively
one through four common starts; a reply at `2,3,4,5` deletes a suffix of four
through one; an off-row or farther row cell deletes none. The survivor set is
therefore one consecutive block. For a block of size one the exact future
demand is one. Two adjacent starts also have value one because any pair
maturing both uses two of the three empty cells in their five-cell physical
intersection, leaving a common residual hit. A block of size at least three
contains starts differing by two, and one of the three displayed trigger pairs
in L14.8 leaves two disjoint residuals. L10.4 supplies the upper bound. This
recomputes the exact value row `0,1,1,2,2,2` for survivor-block sizes
`0,1,2,3,4,5`.

The ten unordered two-cell covers listed in §71.2, and their two orders, are
the complete value-zero class. Each contains one negative and one positive row
cell, so it also kills the two endpoint-exclusive row labels. None meets a
transverse pencil away from occupied `c,d`. Thus every minimizing raw reply
followed by `(p,p')` gives the same exact `(n_1,n_2)=(20,20)`,
`tau=0,M=2` diamond endpoint.

**Proposed repair:** none.

### 3. MINOR — one §71 lozenge legality sentence asserts a false adjacency

**Quoted claim:**

> “The first placement is adjacent to old Attacker stock and the second is
> adjacent to the first.” (L14.9)

**Independent recomputation.** This is correct for the four one-corner
templates: after playing `p'`, either `x_0` or `x_2` is adjacent to it, and the
reflected statement holds after `p`. It is false for the no-corner template
`(p,p')`. Here

```text
p=(0,1), p'=(1,-1),
d(p,p')=max(1,2,1)=2,
```

not one. The ordered pair remains legal: `p` is adjacent to unchanged old
stone `c=(0,0)`, while `p'` is adjacent to unchanged old stones `c` and
`d=(1,0)` before and after `p` is placed. No inventory, phase, or theorem value
depends on the erroneous adjacency sentence.

**Proposed repair:** split the legality explanation. State that both `p` and
`p'` are independently adjacent to old stock in the no-corner case, while the
wing cell is adjacent to the first response in each one-corner case.

### 4. NOTE — every non-seal lozenge endpoint has exact `tau=0,M=2`, and the seal is honestly unresolved

**Quoted claims:**

> “If the raw Defender occupancy is not `{p,p'}`, Attacker has a legal
> nonterminal pair forming a four-stone lozenge ... [with]
> `tau(P_a^lozenge)=0,M(P_a^lozenge)=2`.” (L14.9)

> “The second class is the exact earliest unresolved Q2 obstruction.” (R8.3)

**Independent recomputation.** The five templates in §71.3 have the stated
five adjacent-pair central lines. In every row of its table, each nonparallel
line intersection is one of the four Attacker lozenge vertices. Therefore an
empty Defender cell damages at most one pencil. In the no-corner branch the
two raw cells damage at most two pencils. In a one-corner branch, the occupied
corner damages exactly the identified pencil and the other raw cell damages at
most one more. At least three pencils are intact in every non-seal case.

No template has three collinear Attacker stones, so every alive count is at
most two and `tau=0`. Any next Defender pair touches at most two of the five
central lines and leaves an intact pencil. Its two flank triggers realize exact
demand two; L10.4 gives the reverse bound. Consequently every next action has
exact handoff `TEMPO=2`, and the endpoint has exact `M=2`.

If both corners are occupied, no unit lozenge containing the old adjacent edge
`cd` can be completed: every such lozenge contains a triangle on that edge and
hence one of its two common neighbors `p,p'`. The local state is the exact
sealed profile `n_1=6,n_2=5`, with `tau=0,TEMPO=2`. Earlier root cells are
outside `B_j`. A response outside `B_j` is virgin relative to every window
meeting `c,d`; one local and one remote response can promote only the local
sealed stock, while two remote responses create at most count-two labels.
Thus R5.1's first all-response `M<=2` cycle extends to the full legal set here.

The source does not turn that safe first cycle into a perpetual escape. It also
does not call the generalized lozenges local stop states. Section 71.4 and the
ledger leave both successor trees uniterated. The double-corner label is
therefore honest, and no Q2 or GAP-RAW conclusion leaks from R8.3.

As a check on the optional sealed-current-demand census, the consecutive
four-rank pairs `{-2,-1}`, `{-1,2}`, and `{2,3}` yield the familiar three-path
residual families and exact `tau=2`; every other span-at-most-five pair has a
common residual hit, while span above five gives `tau=0`. The displayed
service occupancies cover the maximal graded start ranges. The artifact
carefully assigns no unproved exact `M` value to the remaining responses.

**Proposed repair:** only the legality wording in Finding 3.

### 5. MINOR — R8.4 needs a safe-filler clause to guarantee exact `C_cap` membership

**Quoted claims:**

> “Every legal Attacker response returns an epoch with a legal servicing pair
> whose actual handoff belongs to `C_cap`.” (R8.4)

> “Fill an unused placement only after the effective cells.” (§69.5; compare
> review item 8, “A filler ... can only delete labels.”)

**Independent recomputation.** Defender deletion is monotone for the scalar
`TEMPO` bound, but arbitrary deletion does not preserve the exact nested-pair
clause in the definition of `C_cap`. In a positive diagonal derivative, for
example, the high residual is `{3,4,5}` and the containing low residual is
`{3,4,5,6}`. A filler at the low-only cell `6` kills the count-two containing
label while leaving the count-three high label. The scalar handoff is no more
dangerous, but it no longer contains an *exact nested derivative* as required
by clause 3; the surviving high label also violates clause 2 if it is removed
from the named-derivative list.

This case occurs precisely when one hard axis supplies one effective
stabilizer and the second Defender placement is unused. The proof establishes
existence, and the missing restriction is easy to meet in the two exact finite
positions: choose the standard max-`q` filler outside every surviving
derivative support. If an easy second-axis deletion is used instead, choose its
residual cell away from the at-most-one low-only outer cell of a retained
derivative. In the actual hard/hard combinations the fixed stabilizers are
coordinate-disjoint.

**Proposed repair:** add the safe-filler selection just described. Alternatively,
if a filler kills a derivative's high label, remove that derivative from the
named list; do not rely only on the statement that fillers delete labels. This
repairs exact class membership without changing R8.1, R8.2, or the
`TEMPO<=2` conclusion.

### 6. NOTE — with the safe-filler repair, R8.4's one-cycle bookkeeping and initialization scope are exact

**Quoted claim:**

> “R8.4 is the promised honest renewal step ... [but] does not prove that every
> response from an arbitrary `C_cap` member can be serviced back into
> `C_cap`.” (§72)

**Independent recomputation.** The phase ledger from either exact input is:

```text
low-only Defender epoch P_0^pl or P_1^pl
  -> immediate-minimizing cap pair
  -> pure-count-two Attacker handoff with TEMPO=2
  -> arbitrary legal Attacker response b
  -> response-dependent legal servicing/stabilizing pair
  -> C_cap Attacker handoff with TEMPO<=2.
```

For a same-central-axis response, or for the sole hard stock-singleton family,
the at-most-two-cell rank cover services the complete current family and
deletes every high label. This lands in the zero-derivative, pure-count-two
subclass. For a distinct-axis response, `tau=0`; one cell per high axis deletes
an easy family or leaves exactly one of (90)/(98), with at most two derivatives.
The tail and connector censuses prove clause 4 and split every future pair into
no-tail, one-tail, or two-tail demand at most `2`, `1`, or `1+1`.

This use of the inherited initialization results is faithful. The exact inputs
are low-only L12.6 states. A distinct-axis successor also has `tau=0` and an
at-most-two-cell high transversal, consistent with L13.6. A same-axis successor
may have `tau=1` or `2`, so L13.6 is not invoked there; the rank cover supplies
the required actual servicing pair directly. The theorem has the local order
“for every `b`, there exists a servicing pair” only at the two exact inputs.
It neither initializes the general count-three class nor proves closure from
an arbitrary `C_cap` member.

**Proposed repair:** incorporate Finding 5's filler restriction; otherwise no
scope or bookkeeping repair is needed.

### 7. NOTE — L14.2.1–L14.3 exhaust every response, including remote, split, bridge, and separated same-axis pairs

**Quoted claim:**

> “For every legal Attacker response `b`, either both cells lie on one shield
> central axis ... or the returned epoch has exact `tau=0` and all high labels
> lie on at most two physical axes.” (L14.3)

**Independent recomputation.** There are two exhaustive effect classes.

First, if both response cells lie on one of `q=0,q=1,s=0,s=1`, every live high
interval on that line contains at least three of the four axial Attacker ranks:
the old adjacent pair and the two responses. L11.4's rank-triple construction
therefore supplies at most two empty cells meeting every such interval. It
also covers current count-four labels and hence is a servicing pair. This
remains true when the two response effects do not share an old window. For
example, parameters `-4,5` on `s=0` leave only starts `-4,0` high, and the two
cells at parameters `-1,2` delete both. No second high axis can arise: two
distinct response cells determine a unique physical line, and every
intersection with another shield line is an old occupied diamond vertex.

Second, suppose the responses do not share one shield central axis. No old
count-two window receives both cells, so no count-four label is born; old
count-one labels reach at most three and virgin labels at most two. Thus
`tau=0` exactly. Each empty response cell belongs to at most one shield line,
again because all nonparallel shield intersections are occupied. A possible
third high axis would have to be one of L14.1's eight old count-one labels.
The diagonal and vertical extremes lie on a shield central line and collapse
to the first class. On the surviving row `r=1`, the empty residual is
`q=-5,...,-1`, and only `(-1,1)` lies on any shield pencil, namely `s=0`.
On `r=-1`, only `(2,-1)` similarly lies on `s=1`. Hence a bridge row can coexist
with at most one promoted shield axis. There are never three high axes.

This inventory quotient also disposes of every would-be dihedral fan image.
The round-7 south and north pairs themselves use the now-Defender-occupied
cells `(0,-1)` and `(1,1)`. The only noncentral count-one axes that could have
supplied a third pencil are the two bridge rows just classified; no third fan
remains.

The following hand checks stress cases not used as the theorem's equality
witness:

| Response `b` | Effect class | Servicing/stabilizing cells | Returned high stock |
|---|---|---|---|
| `(-4,4),(5,-5)` | separated same `s=0` axis | `(-1,1),(2,-2)` | none |
| `(-1,1),(-2,1)` | hard `s=0` plus the sole `r=1` bridge | `(2,-2),(-3,1)` | only `R_0^-` |
| `(0,2),(-1,2)` | capped `q=0` singleton plus hard `s=1` | `(0,3),(2,-1)` | only `R_1^-` |
| `(-1,1),(2,-1)` | opposite hard diagonals | `(2,-2),(-1,2)` | `R_0^-` and `R_1^+`, with no connector |
| `(-16,9),(-17,9)` | fully remote, supported from old `D@(-16,8)` | no high cleanup needed | virgin count-two stock only |

A one-local/one-remote pair is the corresponding one-trigger case: the remote
cell promotes no old label and together the two new cells create at most a
virgin count-two label. Thus remote responses do not escape the proof merely
because the legal-pair universe is infinite.

After the constructed service, every easy high family is deleted and a hard
diagonal family leaves only its exact nested outer tail. All other graded
labels have count two, so the remaining question is precisely the tail
isolation checked in Finding 8.

**Proposed repair:** no mathematical repair. An optional clarity sentence in
L14.3 could state explicitly that both-remote responses are pure count two and
one-local/one-remote responses are one-trigger cases.

### 8. NOTE — tail isolation, the exact witness, and the universal risk floor all recompute

**Quoted claims:**

> “Consequently the resulting Attacker handoff has `TEMPO<=2`.” (L14.4)

> “`R_1(P_0^pl,a^dagger)=2`” and “`B_1(P_0^pl)=2`.”
> (equations (86)–(87))

**Independent recomputation.** A hard diagonal trigger at parameter `-1`,
stabilized at `2`, leaves high residual `{-4,-3,-2}` inside the low residual
`{-5,-4,-3,-2}`. The positive reflection leaves
`{3,4,5}` inside `{3,4,5,6}`. Each derivative has exact `h=g=1`.

For one tail on `s=0`, neither off-diagonal old diamond stone `(0,1),(1,0)` is
axis-collinear with any tail cell; the `s=1` statement is reflected. Apart from
the response trigger already on the tail's diagonal, only one response stone
remains, so no transverse line through a future tail trigger is pre-count-two.
For two parallel tails, the complete possible connectors are fixed `q` in
`[-4,-2]` or `[3,5]`, and same-side rows `r=3,4` or `r=-4,-3`. Those connectors
contain no old Attacker stone, and the response triggers at parameters `-1` or
`2` lie outside them. Opposite-side tails have no axial connector. Therefore a
future pair avoiding tails invokes only the pure-count-two bound; a pair using
one tail contributes one demand with no count-two bridge; and a pair using two
tails contributes exactly `1+1`, with no third contribution. This proves the
claimed upper bound.

For the exact lower witness,

```text
b^dagger=((-1,1),(-1,2))
```

makes four consecutive-triple count-three windows on each of the parallel
lines `s=0,s=1`, with no current imminent. A next Defender pair concentrated
on one line leaves the other with demand two. A split pair cannot kill all four
windows on either line, so one surviving count-three label on each parallel
line gives two residual-disjoint singleton demands. Conversely,
`((2,-2),(2,-1))` leaves the two isolated negative derivatives above, of exact
`TEMPO=2`. Hence the witness epoch has exact `M=2`, while Finding 7 supplies
`M<=2` for every response.

Finally, an arbitrary initial Defender pair at `P_0^pl` touches at most two of
the four shield lines. One exterior trigger on each of two untouched pencils
creates two consecutive-triple families with `tau=0`. Every next pair either
leaves one family untouched, giving demand two, or touches both once and leaves
one residual-disjoint demand in each. Thus every initial action has risk at
least two, and the cap attains the floor. Equations (86)–(87) are exact.

**Proposed repair:** none.

### 9. NOTE — the `U^-` stock census and all stock-assisted response classes preserve exact risk two

**Quoted claim:**

> “The same ordered action (85) is legal at the exact `P_1^pl` and has
> `R_1(P_1^pl,a^dagger)=2`.” (R8.2)

**Independent recomputation.** The old blocker `D@(2,0)` kills stock-row start
`2` and the left extreme at start `1`. The live row-zero stock labels are
therefore the four count-two starts `3,4,5,6` and the one count-one start `7`.
Each of `(6,0),(7,0)` is otherwise alone on its fixed-`q` and fixed-`s` axes,
adding twenty-four count-one labels. Adding these to L14.1 gives

```text
n_2 = 12 + 4 = 16,
n_1 =  8 + 1 + 24 = 33,
```

with no higher count. The residuals in (96) follow directly. Responses
`{5,8}` and `{8,9}` make consecutive four-rank subfamilies with hitting number
two; every other co-contained response pair has a common residual hit, as the
source states.

The live count-two central axes are `q=0,q=1,s=0,s=1,r=0`; every nonparallel
intersection is an occupied diamond or stock stone. The two surviving bridge
rows still coexist with at most one promoted diagonal. A count-one singleton
pencil through `q=6,q=7,s=6`, or `s=7` cannot add a third high axis: its
intersections with live central residuals are occupied or lie beyond those
residual ranges. A consecutive triple on such a singleton pencil may need the
full two-cell rank cover, but then it is the only hard axis. A nonconsecutive
three-rank family has an empty internal-gap cover.

The stock-row table is exact. Only trigger `8` leaves a derivative: after
`D@5` the high residual is `{9,10,11}` inside low residual
`{9,10,11,12}`. Its transverse `q`/`s` levels are `9,10,11`, whereas old
Attacker coordinates and levels are only `0,1,6,7`; no transverse count-two
bridge exists. It is also noncollinear with every diagonal tail. For the mixed
stress response `(8,0),(-1,1)`, the stabilizers `(5,0),(2,-2)` leave exactly
the stock and negative `s=0` derivatives, with no connector, so the future
demand is at most `1+1`.

The response `b^dagger` and service `((2,-2),(2,-1))` from Finding 8 do not
meet any `U^-` label. Their lower-bound contact census is unchanged, while the
extended isolation gives the matching upper bound. Hence
`R_1(P_1^pl,a^dagger)=2`. The same two-untouched-shield-pencils argument gives
the universal floor and confirms `B_1(P_1^pl)=2`.

**Proposed repair:** none.

### 10. MINOR — the later-plateau ledger conflicts with binding R7.2 at `P_5^pl=P_stock`

**Quoted claims:**

> “No later plateau is classified here.” (§70.3)

> “Dual near-trigger cap at later plateaux ... **OPEN** ... No `V^-`, `W`,
> `V^+`, or `U^+` incidence audit is claimed.” (§73 ledger)

**Independent recomputation.** The first sentence is harmless when read as a
statement about new round-8 calculations. The authoritative ledger row is not
correct as a status statement. Binding §67 retains R7.2, which says that at

```text
P_5^pl=P_stock,
for every legal Defender action a there is a legal response b with M(P_{a,b})>=3.
```

The cap action is one such `a`. Therefore its all-response risk at `P_5^pl` is
already known to exceed two; it is not OPEN there. R8.1 and R8.2 prove safety
at indices zero and one. No theorem locates the first failure among indices
two, three, four, and five, but the final endpoint is fixed by R7.2.

This does not contradict either new exact theorem: `P_0^pl` and `P_1^pl` are
separate exact states, and §68.1 expressly says the cap is not replayed on one
history. It is a localized inherited-status error in the otherwise
authoritative ledger.

**Proposed repair:** split the row into:

- `P_2^pl,P_3^pl,P_4^pl`: **OPEN**;
- `P_5^pl=P_stock`: cap action **UNSAFE / `R_1>=3`** by R7.2.

Add that the earliest loss index in `{2,3,4,5}` remains unknown.

### 11. MINOR — §75 omits the now-landed output identity

**Quoted claim:**

> “Input commit: `a8a0b92d...` ... no landed output commit exists during this
> no-commit pass ... Once the artifact is landed and reviewed, its landed
> identity must be added.” (§75)

**Independent recomputation.** The authoring-time statement is properly scoped
and the named input is correct. At review time the file is landed at

```text
c57da44286f75feb236e6da6c55cdd53e5ec2e68
```

with Git blob `6c460713444ff94758b5416debdfa5b27fa878ef`, identical to the
working-tree file reviewed here. Section 75 necessarily lacks this later
identity, so the fold requested by its own provenance rule is now due.

**Proposed repair:** record input `a8a0b92d...` and reviewed/output artifact
`c57da442...` together, optionally with the blob and SHA-256 from this review's
preamble.

### 12. NOTE — the Q1/Q2/Q3 boundaries and the meaning of “REPAIRABLE-AT-P0” remain honest

**Quoted claim:**

> “Q2 root forcing, perpetual Q3 renewal, and GAP-RAW remain OPEN.” (§1
> disposition)

**Independent recomputation.** R8.1 has the local exact-state order

```text
there exists a^dagger at P_0^pl such that
  for every legal response b, M(P_{a^dagger,b})<=2.
```

It answers round-7 equation (84) and refutes only the negative alternative that
the first diamond plateau itself is a one-ply stop. It does not define one
policy on every strict-root history. R8.2 repeats the same statewise statement
at the separate exact `P_1^pl`; it does not replay already occupied cap cells.
R8.4 has the response-dependent order `for every b, there exists a service`
from those two inputs and explicitly withholds arbitrary-member closure.

R8.3 absorbs the initial root action and partitions the next raw action, but it
stops after one level. It neither iterates the seal's banked safe cycle nor
classifies the next arbitrary action from every generalized lozenge. Thus it
does not produce the Q2 strict-root forcing order. Finally, the artifact keeps
R7.2's universal final `P_stock` stop theorem intact in a separate ledger row.
Apart from Finding 10's later-cap row, no local result is promoted to Q2,
perpetual Q3 repair, or GAP-RAW.

**Proposed repair:** none beyond the ledger split in Finding 10.

## Per-theorem verdicts

| Claim | Source status | Review verdict | Disposition |
|---|---|---|---|
| R8.1 first-plateau repair | Q3, PROVEN | **CONFIRMED** | The cap is legal, servicing, non-hub, and immediate-minimizing; the complete response quotient gives `M<=2` for every legal response |
| `R_1(P_0^pl,a^dagger)=2` | exact | **CONFIRMED EXACT** | Remote, split, bridge, shield, and separated same-axis cases are covered; `b^dagger` has exact returned `M=2` |
| `B_1(P_0^pl)=2` | exact | **CONFIRMED EXACT** | Every action leaves two shield pencils for a risk-two witness; the cap attains the floor |
| L14.1 exact capped inventory | PROVEN | **CONFIRMED** | Complete global profile is `(n_1,n_2)=(8,12)`, with no higher label |
| L14.2.1 same-axis cover | PROVEN | **CONFIRMED** | The rank-triple cover includes current imminents and separated effects such as parameters `-4,5` |
| L14.3 high-axis census | PROVEN | **CONFIRMED** | Outside the same-axis class, exact `tau=0`; the two surviving bridge rows can add at most one axis to one shield promotion |
| L14.4 tail isolation | PROVEN | **CONFIRMED** | One-tail and two-tail future pairs have no transverse or connector count-two leak |
| R8.2 one-stock test | Q3, PROVEN at exact `P_1^pl` | **CONFIRMED** | Exact census `(33,16)`; stock-row and singleton-pencil cases are exhaustive; exact risk and minimum are two |
| L14.5–L14.6 stock inventory/stabilizer | PROVEN | **CONFIRMED** | Four row count-two starts survive; only trigger `8` leaves the isolated stock derivative |
| R8.3 earliest Q2 tree | Q2, PROVEN at one-level scope | **CONFIRMED-WITH-MINOR-WORDING** | Initial reply is absorbed; every raw action enters a generalized lozenge or the exact seal; one legality sentence needs correction |
| L14.7 enlarged untouched launch | PROVEN | **CONFIRMED** | Three disjoint row bands exclude their anchors, leaving one exact raw launch footprint untouched |
| L14.8 raw action quotient | PROVEN | **CONFIRMED EXACT** | Consecutive survivor-block table and all twenty ordered value-zero covers recompute |
| L14.9 non-seal lozenges | PROVEN | **CONFIRMED** | Every template has exact `tau=0,M=2`; `(p,p')` is legal from old stock despite not being adjacent internally |
| L14.10 double-corner seal | PROVEN at one-level scope / continuation OPEN | **CONFIRMED** | Exact `(6,5)` sealed profile and full first-cycle `M<=2`; no later escape or loss is claimed |
| R8.4 exact P0/P1 landing | Q3-repair, PROVEN at exact domains | **CONFIRMED-WITH-MINOR-BOOKKEEPING** | One-cycle landing and `TEMPO<=2` are sound; exact `C_cap` membership needs the safe-filler clause |
| Arbitrary-member `C_cap` renewal | OPEN | **CONFIRMED OPEN** | Neither the definition nor the exact-input proof supplies closure from every abstract member |
| Cap at `P_2^pl`–`P_4^pl` | OPEN | **CONFIRMED OPEN** | No corresponding stock-incidence enumeration is supplied |
| Cap at `P_5^pl=P_stock` | ledger says OPEN | **CORRECTED: UNSAFE** | Binding R7.2 gives a response with returned `M>=3` against every action, including the cap |
| Q2 root forcing / perpetual Q3 repair / GAP-RAW | OPEN | **CONFIRMED OPEN** | All new theorems are exact-state or one-level results |
| Provenance | record | **MINOR ERRATUM** | Input is present; landed/output `c57da442...` is absent from the proof artifact |

## Overall verdict

**SOUND-WITH-MINOR-ERRATA.** The default presumption against the positive
maximum bound did not produce a counterexample. R8.1 is **CONFIRMED**. The two
cap cells are legal members of `Serv(P_0^pl)`, are not the hub, and remove the
two round-7 near triggers. More importantly, the proof does not rely only on
those two named fans: its complete eight-label count-one inventory and
same-axis rank cover exhaust every response effect. Remote pairs stay in the
pure-count-two class; split pairs create at most one high family per local
trigger; bridge pairs create at most two axes; and separated same-axis effects
use the full two-cell rank cover. Tail isolation then proves the upper bound,
while `b^dagger` proves exact equality. No `b` with `M>=3` was found.

R8.2 is **CONFIRMED** at the exact one-stock state. The new row pencil and
twenty-four singleton labels are fully accounted for, including their possible
use as three-rank material. The only stock-row tail is isolated from every
diagonal tail, and the same exact witness gives risk two.

R8.3 is **CONFIRMED AT ITS ONE-LEVEL SCOPE**. The initial response is absorbed,
the raw effect quotient is complete, every non-seal branch reaches exact
`tau=0,M=2`, and the double-corner seal is honestly left unresolved after its
banked first cycle. The false claim that `p'` is adjacent to `p` is MINOR
because `p'` is independently adjacent to unchanged old stock.

R8.4 is **CONFIRMED-WITH-MINOR-BOOKKEEPING**. The one-cycle scalar landing is
sound, but exact membership in `C_cap` requires choosing unused fillers away
from the low-only outer cells of surviving derivatives. No arbitrary-member
renewal follows. The most consequential documentary defect is the later-cap
ledger row: `P_2^pl`–`P_4^pl` remain open, but `P_5^pl=P_stock` is already
known unsafe by binding R7.2. The landed hash must also be folded into §75.

No theorem is refuted and no MAJOR defect was found.

## Exact unresolved obstacles

1. **Loss threshold inside the stock phase.** The cap is safe at exact
   `P_0^pl` and `P_1^pl`, and every action is unsafe at `P_5^pl=P_stock`.
   Its risk at `P_2^pl,P_3^pl,P_4^pl`, and therefore the earliest failure
   index, remains unknown.
2. **Arbitrary-member capped renewal.** The exact P0/P1 construction lands in
   `C_cap`, but no theorem services every response from every member back into
   the class. R7.3's all-history availability premise remains open.
3. **Double-corner Q2 continuation.** The exact sealed state has one banked
   all-response `M<=2` cycle; its subsequent arbitrary Defender/Attacker tree
   is not iterated to either a stop or a perpetual defense.
4. **Generalized-lozenge Q2 continuation.** Every non-seal raw deviation gets
   one exact `M=2` landing, but the next arbitrary Defender action at those
   generalized plateaux is unclassified.
5. **Continuation after the minimizing raw branch.** Reaching exact
   `P_0^pl` and repairing one cap cycle does not provide a full Q2 forcing
   continuation or a Q3 policy.
6. **Other repair geometries.** Shared and nonshared fanouts outside the exact
   cap audit, cross-hull interactions below R5.2 separation, later nested
   derivatives, and alternative forced-service entrances to the transverse
   seal remain open.
7. **General count-three initialization.** L13.6 still excludes strict
   `tau=0` states whose count-three residual family has hitting number at
   least three.
8. **Root-level conclusions.** No strict root forces a losing state against
   every Defender strategy, and no single Defender policy is initialized and
   renewed on every reached history. Q2, positive Q3 repair, and GAP-RAW remain
   open.
9. **Ancillary separation sharpness.** The minimum separation for R5.2 remains
   unknown; radius 21 is still only envelope-sharp.
