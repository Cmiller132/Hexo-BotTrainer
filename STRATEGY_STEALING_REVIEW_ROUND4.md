# R-ST4-REV — Hostile review of `STRATEGY_STEALING_ROUND4.md`

## Method and review boundary

**Reviewed artifact:** `STRATEGY_STEALING_ROUND4.md` at commit
`8fb688649ee72b3a3c7035949335aec87c7b8eb5` on branch `hunt/gap-raw`.

**Named input:** the artifact identifies its authoring input as
`5023169f886bc85b32324a50fb67a58e25e98dff`. That commit is an ancestor of
the reviewed artifact, though not its immediate Git parent. The six cited
engine-rule files are unchanged between the named input and the reviewed
artifact.

**Required reading completed in order and in full:**

1. `STRATEGY_STEALING_HEXO.md`;
2. `STRATEGY_STEALING_ROUND2.md`, including folded errata §16;
3. `STRATEGY_STEALING_REVIEW_ROUND2.md`;
4. `STRATEGY_STEALING_ROUND3.md`, including folded errata §24;
5. `STRATEGY_STEALING_REVIEW_ROUND3.md`; and
6. `STRATEGY_STEALING_ROUND4.md`.

I then read the cited ranges in
`packages/hexo_engine/rust/src/{coord,legal,rules,board,state,tactics}.rs`.
This was a first-principles source and proof audit. I ran no Cargo command,
Lean build, harness, executable search, or proof search. I did not read or
modify any `GAP_RAW_*` file. All coordinate, cadence, window, transversal,
and hitting-set checks below were recomputed by hand.

**Overall verdict: SOUND-WITH-ERRATA.** The support cut and its second
application are sound for the deliberately narrow zero-lag total-exact class.
The deadline shield is also a sound sufficient P5R invariant, and the service,
fork, and permanent-fence calculations recompute correctly. The most serious
defect is not in either ranked theorem: the allegedly exact resume point in
§34.1 has the wrong logical shape and overstates when checkpoint (28.1) is
forced. Several smaller definition, nonvacuity, citation, and ledger repairs
are also required.

## Numbered findings

### 1. NOTE — the production rule facts used by the proofs are confirmed

> “A normal placement is legal exactly when it is empty and belongs to
> `L(O)`. The first nonwinning placement of a pair is inserted and updates the
> legal store before the second is checked.”

The underlying facts are exact. `rules.rs:11-44` rejects terminal play,
enforces the origin opening, rejects occupied cells, and checks the maintained
legal store at both normal phases. `board.rs:83-105` inserts the stone into the
owner map and occupied list before updating the radius-eight legal halo and
the incident windows. `legal.rs:17-18,123-145` uses inclusive radius eight and
tests only physical emptiness, not owner. `state.rs:302-337` checks the window
result before advancing phase, so a first-placement win suppresses the second.
`tactics.rs:13-17,21-75,205-208,451-485` gives length six, three axes, six
offsets per axis, the all-six predicate, and the eighteen updated windows.

The cited predicates therefore support the legality, sequential-update,
cadence, and immediate-terminal uses in S21–S31. Every displayed gadget lies
well inside the literal `i16` carrier.

**Proposed repair:** none to the rule model. The local citation omissions are
listed separately in Finding 12.

### 2. NOTE — S21/S22 survive every rebinding that Definition 26.1 actually permits

> “Every committed exact binding exposes a proxy.”

For a genuine forward shadow history, attach every nonopening placement to
one earlier physical stone at distance at most eight. This includes a pair's
second placement because the first has already been inserted. The resulting
edges form a spanning tree of `G_8(O_H)`, so S21 is exact.

Now fix any committed triple satisfying

`O_H = A disjoint-union P`, with `A=T[O_R]`,

where `T` is a translation followed by D6. At the relevant checkpoints both
parts are nonempty. Connectivity forces an edge `a--p` crossing the
partition. With `x=T^{-1}(a)` and `c=T^{-1}(p)`, bijectivity and `p∉A` give
`c∉O_R`, while `x∈O_R` and

`d(c,x)=d(p,a)<=8`.

Thus `c` is genuinely empty and legal on the real board, including at
`SecondStone` (the stored first coordinate is already occupied and hence
cannot equal `c`). Its exact shadow target is the occupied physical proxy
`p`. This is the load-bearing “real preimage empty and legal” step, and it is
airtight.

Retirement, backing, proxy movement, or designation changes merely replace
one partition by another partition of the same connected physical board.
Backing `p` moves it into `A`, but the one-stone owner-count offset moves some
other physical stone into `P`; the new partition has another cut edge. A
many-to-one merge of two real cells into one shadow cell is not an allowed
rebinding: it violates `A=T[O_R]` and the injectivity of the global isometry.
It must remain outside the theorem's advertised scope.

The terminal refinement is also correct. If a terminal real-S cut coordinate
mapped to a same-role `Shat` proxy, the other five real cells would map to five
`Shat` stones and that proxy would already complete `T[W]`, contradicting the
common-live premise. Such a terminal cut can therefore hit only an
opposite-role proxy unless a different physical shadow-win certificate is
supplied.

**Proposed repair:** none to S21/S22. In summaries, retain all four premises:
normal common-live checkpoint, total exact translation/D6 binding, and
nonempty represented and proxy parts. Do not describe noninjective merging as
covered.

### 3. MINOR — S23 is sound, but “exactly one append” needs the immediate-transfer premise

> “Otherwise phase alignment allows exactly one actual `Shat` placement in
> that repair.”

At synchronization the next `sigma` pair raises shadow `Fhat` from two to
four stones and real F from one to three. Neither side can win with those
counts. Clause 26.4(4) therefore gives an exhaustive fork: either one of the
two prescribed inverses or the required exact restoration fails, or the
candidate reaches (28.1), live at S/`Shat FirstStone`, with

`real (F,S)=(3,2)` and `shadow (Fhat,Shat)=(4,3)`.

S22 then gives `c_1`. Real S has only two stones, so `S@c_1` is nonwinning.
After a successful repair, one associated physical `Shat` append gives counts

`real (F,S)=(3,3)` and `shadow (Fhat,Shat)=(4,4)`

at common `SecondStone`. A new committed total-exact partition again has one
proxy of each role, so S22 gives an empty legal `c_2`. It differs from the now
occupied `c_1`; S reaches only four stones, and the associated shadow append
could give `Shat` only five. No terminal escape exists. A second
coordinate-dependent repair is therefore required.

The quoted sentence is slightly under-argued. Common phase alone permits
`1+4k` additional shadow placements before returning from `Shat FirstStone`
to `Shat SecondStone`. For `k>0`, however, the shadow necessarily traverses a
full `Fhat` pair. Definition 26.4(4) requires those prescriptions to be
immediately realized as real F placements, which is impossible while the
real engine is at S's `SecondStone`. Waiting for the next real coordinate
would also violate the rule that the charged episode closes before that
coordinate. Thus every *successful* escape really does have exactly one
physical `Shat` append, but the proof must invoke the immediate-transfer and
real-turn premises, not phase equality alone.

**Proposed repair:** replace the quoted sentence with: “A successful escape
can append exactly one `Shat` stone. Any additional four-placement shadow
cycle contains a prescribed `Fhat` pair that cannot be transferred while the
real game remains at S `SecondStone`, and is therefore an earlier failure of
items 2 or 4.” This also makes the count and literal-carrier bounds explicit.

### 4. MINOR — the reactive mechanism is genuinely new, but “class nonempty” needs a syntax/semantics distinction

> “Relative to `C_static^2` at the shared S15 checkpoint, it relaxes the
> fixed-map/fixed-proxy restrictions.”

The second cut is not merely round 3's static theorem restated. One reactive
repair can really occur. A hand witness is:

```text
real S15:   F@(0,0); S@(-1,0),S@(-1,1)
shadow:     Shat@(0,0); Fhat@(1,0),Fhat@(1,1);
            Shat@(0,1),Shat@(0,2)
T(c)=c+(1,1)
```

Here the proxies are `Shat@(0,0)` and `Fhat@(1,0)`. Let the next legal
`sigma` pair be `(2,1),(3,1)`, transferred to real F at `(1,0),(2,0)`.
Then `c_1=(-1,-1)` is legal and maps to the same-role opener proxy `(0,0)`.
Back that proxy and append the legal `Shat` filler `(-1,0)`, keeping `T`.
The exact binding and common phase are restored, with the filler as the new
proxy. Its real preimage `c_2=(-2,-1)` is fresh, adjacent to `c_1`, and forces
the second escape. All displayed distances are at most two. Static
`C_static^2` forbids this first backing/filler repair, whereas Definition 26.4
admits it. S23 therefore proves a genuinely stronger local obstruction.

There is a terminology issue. If “member” means a coupling that actually
fulfils all five total promises, S23 proves that the extensional class has no
members; it cannot simultaneously be called nonempty. What is nonempty is the
syntax class of candidate update algorithms and their legal partial
executions, as witnessed above. Nor is Round 4 literally a strict superclass
of every Round-3 synchronization: Definition 20.1 allowed either member of
the first `Fhat` pair to be the proxy, while S15 itself fixes `q_1` as proxy
and `q_2=T(x)`. The source's phrase “at the shared S15 checkpoint” is the
correct narrower comparison.

**Proposed repair:** define a nonempty candidate-rule family separately from
the success property, embed the `q_1`-proxy static subfamily by choosing zero
rebindings and zero escapes, and state S23 as failure of every such candidate.
Alternatively broaden the start condition to every Definition 20.1
synchronization before claiming inclusion of all `C_static^2` candidates.

### 5. NOTE — S24 correctly blocks the inference to unbounded total rebinding

> “It does not prove that zero lag itself is necessary, does not exclude every
> finite total budget `K>=2`...”

After fixing `sigma` and a finite shadow checkpoint, retain the nonterminal
prefixes compatible with `sigma`. Each node has finitely many legal
single-placement children. If compatible nonterminal prefixes had unbounded
depth, the finite-branching path argument would select an infinite branch on
which `sigma` never wins, contrary to its assumed winning property. Hence the
tree has bounded depth. Since a bounded-depth finitely branching tree is
finite, any deterministic candidate doing finitely many representation
operations per placement also has a finite, `sigma`-dependent maximum over
that tree.

This establishes only a strategy-dependent finite horizon. It gives no
uniform budget across all `sigma`, and it does not construct a repair with
budget two or more. The design map and ledgers correctly leave those claims,
lag, and nonisometric representations open.

**Proposed repair:** none.

### 6. NOTE — S25/S26 correctly separate terminal memory from current-coordinate legality

> “The final real placement wins on `{(2,1),...,(2,6)}`. The shadow remains
> nonterminal: it lacks `(2,5)` and has no other `Shat` six.”

Every nonopening coordinate in S25 has hex norm at most eight, so the physical
origin supports it. The owner cadence is exact. Before the divergent pair,
real S and shadow `Shat` each have only the run `(2,1),...,(2,4)` on the
relevant R-axis. The real first placement `e=(2,5)` and shadow filler
`f=(0,1)` are legal and nonwinning. The common second coordinate `(2,6)` is
supported on both boards by `(2,4)`, but only the real board has all six cells
`(2,1),...,(2,6)`. The claimed hidden-terminal failure is therefore reachable
on two legal physical histories. It is not visible in S17's legality sets.

For a fixed next real coordinate, S26 is then taut but exact: if `y` is an
immediate real win through an already present member of `E_S`, the associated
actual shadow append reflects that win no later exactly when it lies in the
physical shadow immediate-win set `D_H`. A label or isometry change alone
cannot change either physical owner masks or the engine verdict. The finite
census is also correct: one cell lies in `3*6=18` distinct engine windows, and
at a fixed occupancy each window contributes at most one missing terminal
cell.

**Proposed repair:** none to S25, S26, or S26.1. Keep S26's scope to surplus
already in `E_S`; a terminal coordinate newly admitted as surplus is instead
caught by the post-placement rule in Definition 30.3.

### 7. NOTE — the deadline shield and S-turn induction are sound

> “`delta(W)>m for every E-live W`.”

For an E-live window there is no real F stone, so

`delta(W)=6-|W intersect X_S|`

is literally its number of empty cells. At S `FirstStone`, `m=2`; at
`SecondStone`, `m=1`; after the ordered pair, the bookkeeping value is zero.
If an old E-live window contains the next S placement, both `delta` and `m`
fall by one. If it does not, only `m` falls. Thus strict inequality is
preserved in either case. A newly surplus stone can create new E-live windows,
so testing precisely those windows in the post-placement state is both the
missing and sufficient admission condition. Valid physical certification can
only remove windows from the E-live family, and a physical F blocker does the
same permanently.

The strict inequality rejects the review witness at the right instant:
immediately after `e=(2,5)`, the target window has `delta=1` and S still has
its second placement, so `m=1` and `1>1` fails.

S28 follows. At every S-role step an E-live full window would have
`delta=0`, contradicting `delta>m>=0`. Therefore a real S terminal window
cannot meet `E_S`. All six cells are in `C_S`, and the retained current
translation/D6 certificate supplies six actual `Shat` stones in the image
window. Definition 30.3 expressly carries that certificate through a
possibly terminal append, closing the otherwise dangerous terminal-state
boundary.

**Proposed repair:** none to S27 or S28. The theorem is a sufficient
conditional invariant, not a proof that a global coupling can always maintain
the admission and service rules.

### 8. MINOR — the displayed “nonvacuity witness” establishes a state, not by itself a complete member

> “Nonvacuity witness from an S15 prefix [PROVEN].”

The displayed prefixes are legal and nonterminal. Under `T(c)=c+(-2,0)`, the
three claimed common S cells have actual `Shat` images, while `(0,5)` does not.
The only other real S cell on an axis through `e=(0,5)` is `(1,5)`, so every
window through `e` has at most two real S stones and hence `delta>=4>2` at the
next S `FirstStone`. This proves that the shielded state space contains a
reachable checkpoint with genuine nonempty surplus.

It does not, as written, resolve whether “member of `C_shield`” means a full
finite/infinite continuation satisfying the rules until a justified stop, or
merely a module state plus update contract. If the former is intended, the
shown endpoint alone is not yet a complete member. The gap is evidentiary,
not substantive: an explicit terminating extension exists. Keep the same `T`
and append the following owner turns from the witness endpoint:

```text
real F:    (-1,0),(-2,0)      shadow Fhat: (3,0),(4,0)
real S:    (0,1),(3,1)        shadow Shat: (-2,1),(1,1)
real F:    (-3,0),(-4,0)      shadow Fhat: (5,0),(6,0)
real S:    (4,1),(5,1)        shadow Shat: (2,1),(3,1)
```

All cells are fresh, within origin support, and nonwinning until the final
second placement. That placement completes the common-only real window
`r=1, q=0..5` and its shadow image `r=1, q=-2..3` simultaneously. During the
extension, an E-live window through `(0,5)` contains at most two S stones: on
the three axes the only possible companions are respectively `(1,5)`,
`(0,1)`, and `(4,1)`. Thus `delta>=4` throughout and the surplus remains
nontrivial until the sound terminal stop.

**Proposed repair:** define whether `C_shield` contains states, handlers, or
complete traces. If complete traces are intended, add the extension above (or
an equivalent one). Then the advertised nontrivial conditional class is
literally nonempty.

### 9. NOTE — S29's service cost and S30's 2:2-cadence fork recompute exactly

> “A completed nonwinning F pair restores the deadline shield ... exactly
> when its two coordinates contain a transversal of the urgent hole family.”

With `C_S,E_S` fixed during the F turn, an urgent E-live window has one or two
empty holes. An F pair restores the next-turn condition `delta>2` exactly when
it places an F blocker in every such hole set. That is precisely the
transversal condition. Every chosen hole is at line distance at most two from
one of the four or five S stones already in the window, hence is legal under
radius eight. If the minimum transversal has size zero or one, a fresh legal
padding coordinate completes the pair; an intervening F win is a sound
terminal alternative. Conversely, if `tau_E>2`, any nonwinning pair misses an
urgent window, and S can fill its one or two still-empty holes on the next
turn, winning on the first coordinate or on the second.

The S30 real history respects the exact `F ; S,S ; F,F ; ...` cadence, and
every post-opening coordinate has norm at most seven, so the origin supplies
legality. At its final F `FirstStone` checkpoint, the three displayed windows
through `e=(0,1)` have the disjoint hole pairs

```text
Q:  {(4,1),(5,1)}
R:  {(0,5),(0,6)}
QR: {(4,-3),(5,-4)}.
```

Thus `tau_E>=3`. The nonterminal check also closes:

- S has length-four runs on `r=1`, `q=0`, and `q+r=1`; every other S axis
  line has at most two stones.
- All F q-coordinates are distinct. The largest `q+r` multiplicity is two.
  On `r=0`, F is at q-coordinates `-7,0,7`, so no length-six interval contains
  more than one of those three. Every old F window therefore contains at most
  two F stones, and the service pair cannot win for F.
- Whichever displayed hole pair remains untouched, its offset-four cell is
  adjacent to S's offset-three stone. Its two cross-axis lines contain no
  hidden S five, so it is nonwinning. The adjacent offset-five cell is then
  legal and completes the selected E-meeting six.

This is a valid abstract labeled real-board counterexample to blanket
two-stone service. It is not a strategy-specific coupling obstruction because
no full shadow certificate or forced reachability under an alleged winning
`sigma` is supplied.

S30 is an example, not an exhaustive classification of `tau_E>2`. Three
spatially separated urgent windows with disjoint singleton holes can also have
transversal number three without forming a three-axis fork through one cell.
The formal Round-4 text uses S30 only as a counterexample, so it remains sound;
the shorthand “`tau_E<=2` unless three-axis fork” must not be read as an
if-and-only-if taxonomy.

**Proposed repair:** none to S29 or S30. In every summary of S29 retain its
fixed-`C_S,E_S`, no-reconciliation premise; in every use of S30 retain its
abstract labeled-state scope.

### 10. NOTE — the eighteen-window census and permanent-fence hitting number are exactly six

> “Every physical permanent F fence of all windows through one S-owned cell
> contains at least six F stones. On an otherwise available neighborhood, six
> suffice.”

There are six placements of `e` within a length-six interval on each of three
axes, hence exactly eighteen incident windows. Fix an axis vector `v`. Its two
extreme windows are

`{e-5v,...,e}` and `{e,...,e+5v}`.

They meet only at the S-owned cell `e`, so one F blocker cannot hit both. Each
axis therefore needs one blocker on each side. The three axis lines meet only
at `e`, making those six off-center blockers distinct. This proves the lower
bound six.

The six adjacent cells `e-v,e+v` over the three axes hit every incident
window: if `e` is an endpoint, the interval contains the adjacent cell on its
only side; if `e` is internal, it contains both. Thus six suffice whenever
those cells are available. Each is individually legal from `e` while empty.
This is an unconstrained geometric hitting number, not a claim that F can
install the set before S occupies a candidate cell or across three interrupted
F turns.

**Proposed repair:** none.

### 11. MAJOR — §34.1's resume point has the wrong quantifiers and logical connective

> “The next exact checkpoint is (28.1) ... A successor must do one of the
> following rigorously:”

The six numbered entries are not six interchangeable alternatives. Items 1–3
are alternative ways to handle an exposed real-S coordinate:

1. restore a zero-lag total/window-faithful representation on that placement;
2. carry an explicit shield-admissible lag or queue; or
3. supply a same-step physical shadow terminal certificate.

Items 4–6 are common obligations: service/reconciliation, persistent physical
stone accounting, and the S18/S20/S12/S25 regression tests. Under the literal
“do one” connective, merely retaining old proxies as item 5 requests would
satisfy the stated resume condition. Under a conjunctive reading, items 1 and
2 demand incompatible zero-lag and lagged invariants. The list therefore does
not state a valid exact successor obligation.

Checkpoint (28.1) also is not forced on every dynamic candidate. It is forced
for every `C_react^<=1` candidate that successfully transfers `sigma`'s next
pair and restores a total exact binding; inability to do so is S23's first
failure branch. A broader P3 recode or lag can diverge before that exact
representation checkpoint. Such a successor must solve the next `sigma` pair,
but it need not instantiate (28.1) as a total-exact binding.

Finally, a future “finite configuration” obstructs a strategy-specific
successor only if, for some alleged winning `sigma`, the adversary can force it
on that candidate's own legal, `sigma`-consistent coupled history. An arbitrary
legal labeled real-board state such as S30 is insufficient by itself; the
artifact correctly says this about S30, but §34.1's final question drops the
reachability quantifier.

**Proposed repair:** replace the list by the following quantifier structure:

> For every alleged-winning `sigma`, every relevant S15 synchronization or
> later strategy-generated live prefix, and every legal real-S continuation,
> the candidate must select one valid branch — (A) zero-lag repair for the
> current single placement, (B) an explicit lag satisfying `delta>m` at every
> intermediate phase, or (C) a same-step physical shadow terminal certificate
> — **and**, in all branches, discharge items 4–6. Before this fork, either
> transfer `sigma`'s next F pair with a genuine restored invariant or give the
> corresponding earlier P3/terminal repair. Any negative gadget must be forced
> on the candidate's own coupled history.

This defect downgrades the claimed exactness of the resume checklist, not S23
or S28.

### 12. MINOR — the design map, ledgers, citations, and provenance need local scope repairs

Several summaries drop premises that the formal results retain.

> “Leave a first-placement S divergence unresolved through `SecondStone` |
> **OPEN only under the shield**.”

`C_shield` is one sufficient preventive route, not the only P5R route. S26
also permits an actual same-step shadow terminal certificate. Say “OPEN subject
to P5R; `C_shield` is one sufficient module, and S26 terminal supply is the
other stated route.”

The following ledger rows should also be narrowed:

- S21 applies to a **nonempty** genuine legal prefix.
- S22 applies at a normal common-live checkpoint with a total exact
  translation/D6 binding and nonempty `A,P`.
- The P2 phrase “every committed total isometry” and P6 phrase “every
  precommitted map” must carry those S22 premises. Non-total, nonisometric,
  lagged, and window-certificate maps remain open.
- S29's design-map row must retain fixed `C_S,E_S` and no reconciliation during
  the F turn.
- The P5R row should mention S26's physical terminal-certificate alternative,
  not only preventive reconciliation.

Two source attributions are locally incomplete. `state.rs:289-337` proves
application and phase advancement but not that a new state starts with
`Player0` in `Opening`; add `state.rs:149-160`. Lines `360-369` expose a public
undo API but do not by themselves prove the prose label “search-only.” Add the
MCTS/search context at `state.rs:283-288` and describe undo as an analysis API
outside legal forward `Placement` transitions, rather than as an
access-enforced search-only operation.

Finally, §34.2's list headed “Required documents read first” omits
`STRATEGY_STEALING_REVIEW_ROUND2.md`. Its input hash is accurate, but it should
also identify the reviewed/output artifact commit, as this review's preamble
does.

**Proposed repair:** apply the scope and citation changes above; insert the
Round-2 review between Round 2 and Round 3 in §34.2; record both
`5023169f886bc85b32324a50fb67a58e25e98dff` and
`8fb688649ee72b3a3c7035949335aec87c7b8eb5`.

### 13. NOTE — the global outcome and unbounded-rebinding boundaries are honest

> “This round does not produce a global dynamic coupling.”

> “It does **not** prove that a survivor needs infinitely many total
> rebindings.”

Those boundaries are respected. S23 is limited to a successful zero-lag,
total-exact, owner-faithful start followed by the tested S pair and a budget of
one charged episode in that pair. S24 expressly leaves finite budgets
`K>=2`; the design map leaves per-placement repair, lag, nonisometric finite
encodings, and window certificates open. S28 proves P5R only for traces that
meet the shield's admission and service contract. It does not solve strategy
domain, reverse transfer, collision, fabricated `Fhat` wins, or causal
preannouncement. Both ledgers leave the global dynamic coupling and `NL_F`
open. No statement leaks to an outcome theorem.

**Proposed repair:** none beyond the §32–§34 wording corrections in Findings
11–12.

## Per-theorem verdicts

| Ranked theorem | Review verdict | Exact disposition |
|---|---|---|
| Ranked (b): S23 for `C_react^<=1` | **CONFIRMED-WITH-ERRATA** | S21's connected support graph, S22's cut edge, real-empty legal preimage, forced successful checkpoint, and second cut are sound. Explicitly derive the one-shadow-append fact from items 2/4, and distinguish the nonempty candidate grammar from the empty class of globally successful members. The result is new relative to the shared S15 static subfamily, but not a theorem about merges, lag, nonisometric maps, budgets `K>=2`, all Round-3 proxy designations, or unbounded total rebinding. |
| Ranked (c): S28 for P5R module `C_shield` | **CONFIRMED-WITH-ERRATA** | The strict `delta>m` invariant prevents every E-meeting real S six and maps every common-only six physically to the shadow. S29, S30, and S31 recompute exactly. Define the membership horizon and extend the finite nonempty-surplus checkpoint to a complete trace (an explicit extension is given in Finding 8) before calling the class literally nonempty. Universal maintenance with P0–P6 remains unproved. |

The §34.1 successor checklist separately receives **DOWNGRADE** until Finding 11's
quantifiers are repaired. That downgrade does not change either ranked
theorem's proof status.

## Exact unresolved obstacles after review

The corrected open agenda is:

1. **Pre-checkpoint P3 transfer.** For every alleged winning `sigma` and every
   S15 synchronization, realize `sigma`'s next sequential `Fhat` prescriptions
   as a legal real F pair with a genuine restored invariant, or supply an
   explicit earlier recode/lag/terminal repair. Checkpoint (28.1) is conditional
   on success of the total-exact branch.
2. **P2/P4 at each real-S coordinate.** After the coordinate is observed,
   choose either a genuine zero-lag repair, a lag/queue, or a physical terminal
   certificate. The tested zero-lag total-exact branch needs separate repairs
   for both coordinates; S23 excludes only a one-episode budget in that pair.
3. **P5R during every lag or recode.** Every newly or previously real-only S
   cell must satisfy `delta>m` at every intermediate S phase, be physically
   reconciled, be protected by an actual F blocker, or be accompanied on the
   terminal step by a legal physical shadow-`Shat` win. Arbitrary filler labels
   do not suffice.
4. **F-service compatibility.** Before the next S turn, reconcile enough
   surplus or realize a P3-compatible transversal of every urgent hole family.
   S29 solves only fixed-label physical service with `tau_E<=2`; S30 shows that
   `tau_E>2` is legally possible, while multiple separated debts can also push
   the transversal above two. No strategy-specific invariant is known to
   exclude or reconcile every such service state.
5. **Permanent-fence installation.** Six available blockers are geometrically
   necessary and sufficient around one surplus cell, but cell availability,
   installation over interrupted turns, opponent occupation, and compatibility
   with `sigma`'s prescribed F moves are unresolved.
6. **Reverse shadow-to-real legality (P3).** A successor must handle S18's
   proxy-supported shadow reply and all updated unsupported/collision sets
   after each first placement, not merely the forward proxy cut.
7. **Shadow-`Fhat` terminal fidelity (P5).** Prevent or transfer S20's
   proxy-assisted second-placement `Fhat` win; a fabricated shadow win cannot
   be treated as a real F win without a physical certificate.
8. **Strategy domain and physical persistence (P0/P1).** Every filler, proxy,
   backing, retirement, or recode must remain one legal `sigma`-consistent
   append-only shadow history with the required phase relation. Every old
   proxy/filler continues to affect occupancy, support, blocking, and windows.
9. **Causality (P6).** No repair may expose a still-empty future real F
   coordinate across an intervening S turn under S12's premises.
10. **Strategy-specific reachability and the outcome step.** Any new negative
    gadget must be forced on the candidate's own `sigma`-consistent history for
    some legal S continuation. No rule-level labeled configuration alone
    refutes all dynamic couplings. Until one mechanism satisfies all items for
    every alleged winning `sigma`, or every mechanism is excluded, the global
    coupling and `NL_F` remain open.

## Overall verdict

**SOUND-WITH-ERRATA.** Neither ranked theorem is refuted. Theorem (b) is a
valid two-cut obstruction for its narrow dynamic zero-lag class, and theorem
(c) is a valid nontrivial conditional P5R invariant with correct service and
hitting costs. The leading repair is the MAJOR quantifier correction to
§34.1; the remaining issues are local definition, proof-explication, scope,
citation, and provenance errata.
