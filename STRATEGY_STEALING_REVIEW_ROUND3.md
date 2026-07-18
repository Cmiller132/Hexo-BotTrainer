# R-ST3-REV — Hostile review of `STRATEGY_STEALING_ROUND3.md`

**Reviewed artifact:** branch `hunt/gap-raw`, reviewed HEAD `890aa531`.

**Required reading order completed:** `STRATEGY_STEALING_HEXO.md`,
`STRATEGY_STEALING_ROUND2.md` (including its folded errata),
`STRATEGY_STEALING_REVIEW_ROUND2.md`, then
`STRATEGY_STEALING_ROUND3.md`.

**Method boundary:** proof and document audit only. No Cargo command, Lean
build, harness, executable proof search, or production-source edit was used.
No `GAP_RAW_*` file was touched.

**Overall verdict: CONFIRMED-WITH-ERRATA.** Ranked outcome (c) survives as a
finite, genuine-game, strategy-consistent opening synchronization plus a
conditional common-live cadence invariant. Ranked outcome (b) survives for
the expressly narrow static class `C_static^2`. Neither result proves a
continuing proxy coupling or `NL_F`.

There is no fatal error in S15–S20. There is, however, one **MAJOR** omission
from the specialized terminal/resume ledger: if a dynamic repair ever leaves
a real-only S stone in `E`, a later move can copy legally on both boards and
still complete a real S six that has no shadow `Ŝ` six. Definition 18.2(7)
states the generic terminal duty and round 2's S14 is mentioned later, but
§22.3 and `GAP-PROXY-RETIRE-OR-RECODE` fail to carry this distinct obligation
into their allegedly sharp checklist. The named resume point is therefore
incomplete as written.

## Numbered findings

### 1. NOTE — the two proxies are genuine legal placements and the first synchronization is real

> “There is a legal role-swapped shadow prefix with exactly one invented
> stone of each shadow color”

> “`q_1,q_2` are exactly `σ`'s sequential first-turn prescriptions”

Start from real

`F@x ; S@a,S@b`, with `x=(0,0)`.

Real legality gives `a,b≠x`, `a≠b`, `d(a,x)≤8`, and
`min(d(b,x),d(b,a))≤8`. In the shadow, `Ŝ@(0,0)` is the compulsory genuine
opening. A legal second-player strategy `σ` supplies a fresh first coordinate
`q_1` supported by the origin and then a fresh `q_2` supported by
`{0,q_1}`. The first prescription leaves `F̂` with one stone and cannot win;
the second leaves it with two and cannot win. Thus the entire pair is really
reached in one legal history consistent with `σ`.

With `q_1` declared the `F̂` proxy and `q_2=T(x)`, collision avoidance is the
only remaining issue. Once it is established, isometry gives

`d(T(a),q_2)≤8`

and

`min(d(T(b),q_2),d(T(b),T(a)))≤8`.

So `T(a),T(b)` are a legal sequential `Ŝ` pair. The two boards then have
real counts `(F,S)=(1,2)` and shadow counts `(F̂,Ŝ)=(2,3)`, both with the
represented F role at `FirstStone`; neither owner has enough stones to win.
This proves a genuine strategy-domain prefix, not a virtual board.

**Proposed repair:** none. S15's legality, nonterminality, strategy
consistency, and unconditional finite scope are confirmed.

### 2. MINOR — all 12 orientations were counted correctly, but the causal selector should be made functional

> “Each equation in (19.4) excludes at most two choices of `g`. There are
> four equations, so their union excludes at most eight of the twelve
> choices. Choose any remaining `g`.”

For fixed `c∈{a,b}` and `p∈{0,q_1}`, the forbidden equation is

`g(c)=p-q_2`.

Both vectors are nonzero: `a,b≠x=0`, while legal occupancy makes
`q_2≠0,q_1`. If the equation has one solution, its full solution set is a
coset of `Stab(c)`. No nonidentity D6 rotation fixes a nonzero vector, and a
nonzero vector lies on at most one reflection axis. Hence
`|Stab(c)|≤2`, including the worst reflection-axis degeneracy. Each of the
four equations therefore removes at most two symmetries; overlaps, unequal
norms, or common orbits only reduce the union. At least four of the twelve
orientations remain.

These are all occupied-cell constraints. Avoidance handles `0` and `q_1`;
injectivity and `a,b≠x` handle `q_2=T(x)`; injectivity also gives
`T(a)≠T(b)`. Thus avoidance is always possible, not merely generic.

The only formal blemish is “choose any.” A later stealing strategy must be a
single causal function, rather than a per-history existential relation.

**Proposed repair:** fix an enumeration of D6 and choose the first surviving
orientation. This preserves the proof and makes the claimed causal history
map unambiguous.

### 3. NOTE — the opening gadget and its legal-update halo are safely inside `i16`

> “`||q_1||_h≤8`, `||q_2||_h≤16`, `||a||_h≤8`, and
> `||b||_h≤16`.”

The bounds follow successively from radius-eight support. Since D6 preserves
hex norm and `T(c)=q_2+g(c)`, triangle inequality gives
`||T(a)||_h≤24` and `||T(b)||_h≤32`. A radius-eight update reaches norm at
most 40. All axial components, sums, and differences used by the displayed
finite construction are far from an `i16` boundary.

**Proposed repair:** none.

### 4. MINOR — permanent cadence is correct only as a common-live invariant; “conditional only” minimizes the premise

> “permanent cadence repair, conditional only on successful future placement
> transfer”

> “A win terminates rather than creating a cadence mismatch.”

The parity induction is correct. From the synchronized checkpoint, the first
cycle is:

| Checkpoint | Real `(F,S)` | Shadow `(F̂,Ŝ)` | Common phase |
|---|---:|---:|---|
| synchronized | `(1,2)` | `(2,3)` | F role `FirstStone` |
| after matched nonwinning F first | `(2,2)` | `(3,3)` | `SecondStone` |
| after matched F pair | `(3,2)` | `(4,3)` | S role `FirstStone` |
| after matched nonwinning S first | `(3,3)` | `(4,4)` | `SecondStone` |
| after matched S pair | `(3,4)` | `(4,5)` | F role `FirstStone` |

The general formulas `(2k-1,2k)→(2k,2k+1)` and
`(2k+1,2k)→(2k+2,2k+1)` in `(real role counts)→(shadow role counts)` retain
exactly one additional shadow stone per role. Therefore S4's raw count
mismatch cannot recur while legal one-for-one transfer continues and both
histories remain nonterminal.

The qualifier is load-bearing. “Successful future placement transfer”
assumes essentially P2–P5: a legal shadow append, a legal real inverse,
collision avoidance or repair, and enough terminal control to have another
common-live checkpoint. A proxy-assisted unilateral shadow win after a first
placement is a terminal-transfer failure, not a new cadence failure. The raw
engine phase fields can differ after such unilateral termination because a
winning placement bypasses phase advancement; there is simply no later
*common-live* checkpoint to which the induction applies.

Unconditionally, S15 still supplies the finite legal prefix, the one-stone
offset for each role, the common `FirstStone` checkpoint, and a legitimate
next query to `σ`. The conditional does not swallow that theorem. It does
swallow every claim that future transfers actually exist.

**Proposed repair:** retitle S15.1 “count/phase invariant under an assumed
successful one-for-one continuation”; define “live” as both histories
nonterminal; and replace “exactly a legal checkpoint” by “the required count
pattern, with legality supplied by the transfer hypothesis.”

### 5. MINOR — the status of `GAP-OPENING-ALIGNMENT` is sound only with its displayed component qualifier

> “`GAP-OPENING-ALIGNMENT`, cadence/history-domain component [PROVEN].”

Round 1 asked for a legal role-swapped prefix that accounts for the real
opening and both members of S's first pair. S15 now supplies exactly that.
It also removes S4 as an independent future *count* obstruction through the
conditional induction in S15.1.

It does not establish `P2 REAL→SHADOW`, `P3 SHADOW→REAL`, `P4 COLLISION`, or
`P5 TERMINAL`. Consequently the shorter ledger entry
“`GAP-OPENING-ALIGNMENT` [PROVEN]” is safe only because its exact-scope column
immediately narrows it to the cadence/legal-prefix component. Detached from
that column, the status would overstate the result.

**Proposed repair:** name the discharged result everywhere as
`GAP-OPENING-ALIGNMENT/CADENCE-PREFIX`; leave global placement transfer open.

### 6. MINOR — `C_static^2` is natural and nonvacuous, but “covered” must not permit voluntary truncation

> “at every covered live prefix”

> “The promise is against every legal real S continuation from the
> synchronized prefix.”

The intended class is coherent: S15 supplies at least one genuine
synchronization; the class then fixes its two proxies and bijective isometry
and promises immediate exact copying. It differs genuinely from round 2's
`C_iso` because its two invented stones violate `C_iso`'s load-bearing
no-invention premise. It is narrow, but it is not gerrymandered: fixed proxies
plus a fixed exact encoder are the minimal static continuation of S15.

The word “covered” is nevertheless undefined. Read literally, a purported
member could declare no successor prefix covered, stop at synchronization,
and satisfy the numbered continuation clauses vacuously. The sentence about
every legal S continuation and §23's “total exact immediate copy” make the
intended totality clear, so this is a definition defect rather than a
counterexample to S16.

**Proposed repair:** require coverage of every coupled nonterminal successor
from synchronization until one of Definition 18.2's justified stops occurs;
explicitly forbid voluntary truncation.

### 7. NOTE — S16's proxy-preimage fork is exhaustive

> “either one of `σ`'s next two shadow replies has no legal real inverse, or
> after both inverses are placed the real S has a legal nonwinning first
> placement whose shadow image is an occupied proxy.”

Let `r=T(x)` be the nonproxy member of `σ`'s first shadow-`F̂` pair.

- If `r=q_1`, it was legal from the sole earlier stone `p_S`, so
  `d(r,p_S)≤8`.
- If `r=q_2`, it was legal from `{p_S,q_1}`, where `q_1=p_F`, so
  `d(r,p)≤8` for at least one proxy `p`.

For that proxy set `c=T^{-1}(p)`. Proxy/represented disjointness and
bijectivity make `c` empty on the real board, and

`d(c,x)=d(p,r)≤8`.

Thus `c` is already legal at the synchronized real-F `FirstStone` checkpoint.
The next `σ` first reply cannot win because it gives `F̂` only its third
stone, so the second reply is genuinely reached. If either inverse placement
is illegal at its sequential phase, item 5 of `C_static^2` has failed. If both
are legal, neither can consume `c`, since its shadow image `p` is already
occupied and neither legal `σ` reply can equal it. The resulting counts are
real `(F,S)=(3,2)` and shadow `(F̂,Ŝ)=(4,3)`, so neither board is terminal.
The permanent real stone `x` still supports `c`; `S@c` gives S only three
stones and is nonwinning; its required shadow copy is the occupied cell `p`.

These branches are exhaustive. There is also a legal second coordinate for
S: after `S@c` only five other real cells are occupied, while `c` has six
radius-one neighbors, so at least one is empty; playing it leaves S with only
four stones. The proof therefore gives a bona fide adversarial continuation,
not an incomplete real turn.

**Proposed repair:** none to the collision proof. S16 is confirmed under the
intended total reading repaired in Finding 6.

### 8. MINOR — “one full ordinary round” is ambiguous, and the losing continuation is existential

> “the continuation fails before it can copy the next real S turn”

> “the minimal static exact-copy repair cannot survive one full ordinary
> round.”

In the legal-inverse branch, the coupling successfully transfers the whole
next F pair and then fails on the first coordinate of a specifically chosen
legal S continuation. It does not fail for every possible S coordinate; it
fails the class's universal promise because there exists this continuation.
Also, earlier documents use “ordinary turn” for one owner's ordered pair,
while “ordinary round” is not defined. The construction can survive one full
F turn, but it cannot complete the following F-turn/S-turn cycle.

**Proposed repair:** say: “For every legal synchronization, either the next F
pair cannot be lifted, or there exists a legal S continuation on which copying
fails at the first placement of S's next turn. Hence no member transfers the
complete F-turn/S-turn cycle following synchronization.”

### 9. NOTE — S17 is an exact and exhaustive one-coordinate legality ledger

> “`Fail_{R→H}=C_{R→H} ∪ U_{R→H}`”

> “The `C` terms are occupied-coordinate collisions. The `U` terms are empty
> coordinates whose only support is on the source-only side.”

For `y∈L(A∪E)`, there are exactly two cases under the stated pairwise
disjointness assumptions.

1. If `y∈P`, it is empty and legal in the real board but occupied in the
   shadow, giving `P∩L(A∪E)`.
2. If `y∉P`, it is empty on both boards. Shadow failure is then precisely
   `y∉N_8(A∪P)`. Since real support exists and support from `A` is thereby
   excluded, support must come from `E`, giving
   `(N_8(E)\N_8(A∪P))\(A∪E∪P)`.

Swapping `E` and `P` proves the reverse formulas. This exhausts a real move
into a shadow proxy, a shadow reply into a real-only stone, and both kinds of
unsupported frontier move at the instant of exact copying. The `C` and `U`
parts are disjoint by their occupancy conditions.

After a successful common first placement, adding it to `A` and recomputing
is correct. A unilateral first placement belongs in `E` or `P` for the
occupancy calculation, but that calculation alone does not repair the already
asymmetric phase, owner, or terminal state. The document correctly leaves
such repair open.

**Proposed repair:** none to S17. Retain the phrase “one-coordinate exact-copy
legality ledger”; it is not by itself a complete terminal ledger.

### 10. MAJOR — the terminal/resume checklist omits real-S sixes containing real-only surplus

> S19 assumes that “every real stone is represented under one
> translation/D6 isometry `T`, and the only additional shadow stones are
> proxies.”

> §22.3 requires “transfer of every shadow-`F̂` terminal window containing a
> proxy.”

> `GAP-PROXY-RETIRE-OR-RECODE` requires control of the two unsupported sets,
> collisions, and “transferring a proxy-assisted shadow-`F̂` win.”

S19's premise sets `E=∅`. That is legitimate for S19, but a surviving dynamic
scheme may use a filler, queue, lag, or unilateral recode and thereby leave a
real-only S stone in `E_S`. S17 only diagnoses legality of the current copied
coordinate. It does not reflect a terminal window containing an *older*
member of `E_S`.

Here is the exact missed configuration. At an S/`Ŝ` `FirstStone` checkpoint,
let the common S-role stones include

`A_S={(2,1),(2,2),(2,3),(2,4)}`.

Keep the existing proxies away from that line. On the first placement, let
real S play `e=(2,5)` while the shadow `Ŝ` uses a legal off-line filler, for
example `(0,1)` from opener proxy `(0,0)`. Both histories are nonterminal and
enter `SecondStone`; `e` is now in `E_S` and the filler is shadow-only. On the
second placement, copy `y=(2,6)` on both boards. It is legal on each board,
already supported by the common stone `(2,4)`, so the current coordinate lies
in neither failure set. Nevertheless the real board now contains the r-axis
window

`{(2,1),(2,2),(2,3),(2,4),(2,5),(2,6)}`,

while the shadow lacks `(2,5)` and need not have any `Ŝ` six. Phase alignment
and one-coordinate legality therefore coexist with terminal-reflection
failure.

Definition 18.2(7) does state generically that every real S win must
contradict `σ`, and §23.2 remembers that round-2 S14 is a mandatory lag test.
So the issue was known at the outermost level. The defect is that the
specialized §22.3 list and the named “sharp” resume point drop it. A proposed
repair could satisfy every displayed resume bullet and still fail on the
configuration above.

**Proposed repair:** add an independent obligation
`P5R REAL-S-TERMINAL-REFLECTION`: every real S six meeting `E_S` must produce
a legal shadow-`Ŝ` win no later, or the invariant must forbid such a window by
representing/reconciling every relevant `E_S` cell. Include terminal real
moves colliding with an opposite-role proxy; an ordinary collision repair is
not enough. Until this is added, the named resume point is incomplete.

### 11. MINOR — S19 covers both proxy colors, but item 3 should state its strategy-domain premise

> “Consider an owner-faithful coupling whose representation invariant holds
> at every live checkpoint...”

> “a legal shadow-`Ŝ` win, proxy-assisted or not, contradicts the premise
> that `σ` is a winning strategy for shadow `F̂`”

Items 1–4 rederive correctly under a genuine live coupling:

- if every real stone is represented and its final placement transfers, a
  real six maps to the corresponding shadow six unless the shadow had already
  terminated;
- a proxy-free shadow six maps back to a real six;
- a proxy-assisted `Ŝ` six is still a legal shadow-opener win, so it is a
  counterplay to a purported winning `σ`; and
- a proxy-assisted `F̂` six can be fabricated because the proxy's real
  preimage may remain empty.

This treats terminal fabrication for either proxy color. A proxy that would
occupy the image of a later real-six cell is also accounted for: the attempted
copy has already hit `C_{R→H}` and broken the representation premise before
S19 item 1 can be invoked. Same-role backing/reclassification is a separate
open repair; opposite-role occupancy is a genuine block.

The proof of item 3 uses two assumptions not repeated in S19's opening
sentence: the shadow prefix is legal, and every past `F̂` action agrees with
`σ`. “Coupling” likely refers back to Definition 18.2 and therefore imports
them, but the local theorem should not depend on that implicit reading.

**Proposed repair:** begin S19 with “Consider a genuine legal live coupling
for `σ`, with every shadow-`F̂` move prescribed by `σ`...” The four directions
then remain proved.

### 12. NOTE — the reverse-frontier and second-placement fabrication gadgets recompute exactly

> “The shadow coordinate `z=(-8,0)` belongs to `Γ(A,P)`, while its real
> inverse `T^{-1}(z)=(-10,0)` is illegal.”

For S18, `d((-8,0),(0,0))=8`. Its distances to
`A={(2,0),(2,1),(2,2)}` are respectively `10,11,12`. Thus `z` is fresh and
shadow-legal solely through the opener proxy. Translation back by `(-2,0)`
gives `(-10,0)`, whose distances to real
`{(0,0),(0,1),(0,2)}` are again `10,11,12`; it is real-illegal. All opening
placements are distance one, and optional shadow second coordinate `(-9,0)`
is legal from `z`. This is a real rule-level reply-lift failure, although it
does not force an alleged winning `σ` to choose that reply.

> “The final placement in (22.2), which is the second placement of its turn,
> completes a shadow-`F̂` six. The corresponding final real placement in
> (22.1) does not complete a real F six.”

For S20, immediately before the last shadow placement, `F̂` owns exactly
`(1,0),...,(5,0)`; adding `(6,0)` completes the q-axis six. `Ŝ` has only five
stones. The corresponding real endpoint gives F only
`(0,0),...,(4,0)`, five stones, while S has four. Every listed normal
placement is fresh and distance one from existing occupancy, and no earlier
owner has six stones. The missing inverse of `p_F=(1,0)` is `(-1,0)`; before
the final real pair, the desired inverse window lacks `(-1,0),(3,0),(4,0)`,
three cells for a two-placement turn. The second-placement timing is therefore
substantive.

S18's maximum displayed norm is 10 and halo 18; S20's are 6 and 14. The
gadgets are safely inside the literal carrier.

**Proposed repair:** none. S18 and S20 are confirmed at their expressly
rule-level, non-winning-strategy-forcing scopes.

### 13. MINOR — genuine proxies cannot literally be moved or erased

> “retire, move, or real-back a proxy before its live preimage can be played”

> “`GAP-PROXY-RETIRE-OR-RECODE`”

A genuine-game proxy is an immutable stone in the legal shadow history.
“Retire” can coherently mean reclassify it as backed/represented or cease
using it as the unmatched bookkeeping stone while continuing to include its
occupancy, frontier support, and terminal-window effects. “Move” can mean
change the encoder or bind a different shadow stone as the active proxy. It
cannot mean removing or relocating the old engine stone.

The survivor disjunction says no listed mechanism is already proved, so this
wording does not create a false construction. It does matter to the next
obligation: an old proxy remains capable of causing collisions and fabricated
wins after its bookkeeping role changes.

**Proposed repair:** define retirement and movement as representation-level
operations only, and require all physically persistent shadow stones to stay
in the legality and terminal ledgers.

### 14. NOTE — rounds 1–2 and Gale–Stewart are used at their accepted scopes

> “Round-2 Theorem D2 is inherited at its exact scope:
> `NL_F ⇔ S has no winning strategy`.”

Round 3 does not resurrect `C_iso`: S15 deliberately adds one invented stone
of each role, so it lies outside round 2's no-invention class. S3 is used only
as the inherited frontier warning, and S4 is answered only at the
opening/cadence level. S9/S11 remain negative results for their original
classes. The Gale–Stewart bridge is invoked only for the unbounded macro-game
and only converts “S has no winning strategy” into `NL_F`; round 3 never
claims to have refuted an arbitrary S winning strategy.

**Proposed repair:** none. `NL_F` and the global proxy coupling remain OPEN.

## Per-result verdicts

| Result | Source status | Review verdict | Exact disposition |
|---|---|---|---|
| Inherited engine rule contract and D6 equivariance | PROVEN | **CONFIRMED** | The opening, radius-eight growth, sequential pair, immediate-win, and safe-carrier facts used here match the inherited contract |
| D2 non-loss bridge | PROVEN from CITED | **CONFIRMED** | Used only for the unbounded Hexo macro-game; it does not select the determinacy alternative |
| S15 genuine-proxy strategy domain | PROVEN | **CONFIRMED** | The forced proxy opening and both sequential `σ` prescriptions form one legal nonterminal shadow history |
| S15 D6 avoidance | PROVEN | **CONFIRMED-WITH-ERRATA** | Four fibers of size at most two leave at least four orientations; fix a deterministic tie-break for a causal strategy map |
| S15 opening synchronization | PROVEN | **CONFIRMED** | Every legal real first S pair is represented without collision at a common F-role `FirstStone` checkpoint |
| S15.1 cadence invariant | PROVEN | **CONFIRMED-WITH-ERRATA** | Exact before the first unilateral terminal event under assumed legal one-for-one transfer; it proves no transfer existence or terminal fidelity |
| `GAP-OPENING-ALIGNMENT` cadence/legal-prefix component | PROVEN | **CONFIRMED-WITH-ERRATA** | The component is discharged; use the qualified name everywhere so it is not read as global coupling |
| Definition 20.1 `C_static^2` | Definition | **CONFIRMED-WITH-ERRATA** | Natural nonempty synchronization family; make total coverage and justified stopping explicit |
| S16 static collision obstruction | PROVEN | **CONFIRMED-WITH-ERRATA** | The fork is exhaustive under the intended total class; state the adversarial S continuation existentially and remove “ordinary round” ambiguity |
| S16.1 window-exact variant | PROVEN | **CONFIRMED** | Accepted S8.1 reduces it to a bijective translation/D6 isometry |
| S17 two-sided failure formulas | PROVEN | **CONFIRMED** | Exact and exhaustive for one-coordinate legality with pairwise-disjoint `A,E,P` |
| Sequential-pair recomputation | PROVEN | **CONFIRMED** | Recompute after the first placement; a unilateral update diagnoses occupancy but does not repair coupling state |
| S18 proxy-frontier gadget | PROVEN | **CONFIRMED** | Distances `8` versus `10,11,12` prove the claimed legal/illegal split |
| S19 terminal-direction ledger | PROVEN | **CONFIRMED-WITH-ERRATA** | All four directions hold when the local premise explicitly includes a legal `σ`-consistent live coupling and `E=∅` |
| S20 proxy-win fabrication gadget | PROVEN | **CONFIRMED** | The final second placement creates shadow `F̂@(1..6,0)` while real F has only five stones |
| Specialized global terminal/resume ledger | OPEN agenda | **DOWNGRADE** | It omits real-S terminal windows meeting `E_S`; S17 legality can succeed while terminal reflection fails |
| Survivor boundary | PROVEN consequence | **CONFIRMED-WITH-ERRATA** | A survivor must violate a static premise, but “retire/move” must be representation-level because genuine stones persist |
| Ranked outcome (b), `C_static^2` | PROVEN | **CONFIRMED-WITH-ERRATA** | Genuine sharp obstruction to fixed proxies + fixed bijective isometry + total immediate two-way copying; not all bounded-proxy couplings |
| Ranked outcome (c) | PROVEN | **CONFIRMED-WITH-ERRATA** | Genuine finite synchronization is unconditional; permanent cadence is only the common-live conditional invariant |
| Global proxy coupling / outcome (a) | OPEN | **CONFIRMED** | No dynamic retirement, recoding, queue, filler, collision, or terminal theorem is supplied |
| `NL_F` | OPEN | **CONFIRMED** | No arbitrary S winning strategy is refuted |
| `GAP-PROXY-RETIRE-OR-RECODE` as the exact resume checklist | OPEN | **DOWNGRADE** | Right obstruction family, but incomplete until real-surplus terminal reflection and persistent-proxy semantics are added |

No theorem S15–S20 is **REFUTED**. The two **DOWNGRADE** entries concern the
claimed completeness of the forward agenda, not a proved game theorem.

## Overall disposition

**CONFIRMED-WITH-ERRATA.** The hostile re-derivation confirms the round's two
ranked results at their narrow stated scopes:

1. **Ranked (c): CONFIRMED-WITH-ERRATA.** Two genuine proxies always give a
   legal, `σ`-consistent synchronized opening. The 12-isometry count is
   exhaustive. S4's count mismatch stays gone along any assumed successful
   common-live one-for-one continuation. This is not a continuing coupling.
2. **Ranked (b) for `C_static^2`: CONFIRMED-WITH-ERRATA.** At least one fixed
   proxy has a real-empty preimage supported by `x`; either the next `σ` pair
   fails to lift or an adversarial legal S placement collides with that proxy.
   The class needs an explicit total-coverage clause, but the collision proof
   is sound.
3. **Global proxy coupling: OPEN.** S18 and S20 validate the reverse-frontier
   and proxy-fabricated-terminal dangers. Dynamic repairs remain unexcluded.
4. **`NL_F`: OPEN.** D2 remains only the accepted logical bridge.

The most severe finding is Finding 10. The exact missing obstacle is
**real-S terminal reflection through real-only surplus**: a coordinate can be
legal and successfully copied on both boards, while an older `E_S` stone
completes a real six absent from the shadow. Add that obligation before
`GAP-PROXY-RETIRE-OR-RECODE` can be called the exact next checklist. No guess
is made about whether a dynamic repair exists.
