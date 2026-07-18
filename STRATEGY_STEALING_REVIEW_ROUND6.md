# R-ST6-REV — Hostile review of `STRATEGY_STEALING_ROUND6.md`

## Method and review boundary

**Reviewed artifact.** `STRATEGY_STEALING_ROUND6.md`, added at commit
`7c09dee43842bdb73cd3fdfc9e144d51b3b9b62f` on branch `hunt/gap-raw`.
Its SHA-256 is
`214a1476235f36f1f345d8b27e4e14c412af69c853a93b6941939725a29ccce5`
and its Git blob is `109fcc54b324b28698ef17ee4380b20ae7f5c9ba`.
The worktree copy is byte-identical to that committed blob.

**Named input.** The artifact names
`3000a117d10a2148f744412aae26e053cf6babbc`. Git confirms that this is the
sole parent of `7c09dee4`; the round-6 artifact does not exist in the parent
and is added in `7c09dee4`. The same commit also adds only the authoring prompt.
Thus the repository evidence supports the stated input and landing lineage.
It cannot independently preserve or compare a pre-commit authoring buffer, so
"landed unmodified" is confirmed to the observable extent that the current
artifact exactly matches the first committed artifact.

**Required reading completed first, in order and in full:**

1. `STRATEGY_STEALING_HEXO.md`;
2. `STRATEGY_STEALING_ROUND2.md`, including folded errata, then
   `STRATEGY_STEALING_REVIEW_ROUND2.md`;
3. `STRATEGY_STEALING_ROUND3.md`, including folded errata, then
   `STRATEGY_STEALING_REVIEW_ROUND3.md`;
4. `STRATEGY_STEALING_ROUND4.md`, including binding section 35, then
   `STRATEGY_STEALING_REVIEW_ROUND4.md`;
5. `STRATEGY_STEALING_ROUND5.md`, including binding section 44, then
   `STRATEGY_STEALING_REVIEW_ROUND5.md`; and
6. `STRATEGY_STEALING_ROUND6.md`.

I then read and checked the cited ranges in
`packages/hexo_engine/rust/src/{coord,legal,rules,board,state,tactics}.rs`.
This was a first-principles proof audit. I ran no Cargo command, Lean build,
harness, executable search, or proof-search program. Every distance, cadence,
window, deficit, cut, and transversal calculation below was recomputed by
hand. I did not read or use a `GAP_RAW_*` file as mathematical evidence.
Filename and Git-tree metadata were used only for the provenance check.
Unrelated workspace entries were left untouched.

**Overall verdict: SOUND-WITH-MINOR-ERRATA.** S40–S45 survive at their exact
conditional scopes. S41 is a legal, terminal-aligned physical trace; S42 is a
candidate-own negative control; the per-pair reset and colored cut are sound;
and the S30 urgent family has exact transversal number five. Two documentation
repairs are required: the terminal closure of `A_FS2^ET` should explicitly
state how its final paired event relates to round-5 clauses 5–6, and section 50
does not literally carry every required open-ledger entry forward. Neither
issue proves a theorem false or changes `NL_F`, which remains open.

## Numbered findings

### 1. NOTE — the production rule contract used by round 6 is exact

> “A normal placement is legal exactly when its coordinate is physically empty
> and belongs to the color-blind radius-eight legal store.”

**Independent recomputation.** `coord.rs:76-95` implements the displayed
axial maximum-norm distance and the inclusive ball. `legal.rs:17-18,123-145`
fixes radius eight, removes the occupied coordinate, and adds every empty halo
cell without testing owner. `rules.rs:11-44` rejects terminal play, enforces
the origin opening, rejects occupied cells, and checks the maintained legal
store at both normal phases. `board.rs:83-105` inserts occupancy before legal
and window updates. `state.rs:289-357` then records the append, tests the win,
and advances phase only on a nonwin; `state.rs:203-252` exposes no terminal
continuation. `tactics.rs:13-17,21-75,205-208,451-485` gives the three axes,
six cells, eighteen incident windows, and the physical all-six predicate.

These predicates justify the color-blind supports, sequential pairs, physical
window certificates, and first-placement suppression used in S40–S45. The
largest displayed round-6 coordinate has hex norm ten and the largest claimed
update halo has norm eighteen, far inside `i16`.

**Proposed repair:** none.

### 2. NOTE — S43's per-pair reset escapes exactly the lifetime charge used by S32

> “It cannot force a third charged episode before the reset.”

> “S32's coordinate `c_3` is the first coordinate of the next S pair and is
> charged to the fresh counter.”

**Independent recomputation.** The full checkpoint accounting is:

| Checkpoint | Real `(F,S)` | Shadow `(Fhat,Shat)` | Actor/phase | Charge state |
|---|---:|---:|---|---|
| S15 synchronization | `(1,2)` | `(2,3)` | F `FirstStone` | no S-pair counter active |
| first transferred `sigma` pair | `(3,2)` | `(4,3)` | S `FirstStone` | fresh budget 2 |
| `c_1` and successful repair | `(3,3)` | `(4,4)` | S `SecondStone` | one used |
| `c_2` and successful repair | `(3,4)` | `(4,5)` | F `FirstStone` | two used; old counter discarded |
| next `sigma` first event | `(4,4)` | `(5,5)` | F `SecondStone` | no S charge |
| next `sigma` second, if nonwinning | `(5,4)` | `(6,5)` | S `FirstStone` | fresh budget 2 |

At every restored total-exact checkpoint the one-stone offset per role leaves
the represented and proxy parts nonempty, so S22 applies. In the first tested
pair, `c_1` raises real S only to three stones and `c_2` raises it only to four;
their associated shadow appends raise `Shat` only to four and five. Those two
cuts are necessarily nonterminal. A nonwinning second placement advances
directly to F `FirstStone` (`state.rs:330-333`), so S has no third coordinate
before the old counter is discarded.

Crossing the actual intervening F pair is essential. Its first placement gives
real/shadow F-role totals four/five and cannot win. On the second event either
the sixth `Fhat` stone wins while real F has only five, which is already a P5
failure, or both boards remain live and S reaches a new `FirstStone` with a
fresh counter. S32's `c_3` is then the first charge of the new pair. This
proves a limitation of S32's pigeonhole argument, not existence of either
repair.

**Proposed repair:** none.

### 3. NOTE — S43.1's colored case split is exhaustive and candidate-own

> “For every alleged-winning `sigma` ... real S has a legal continuation of
> length at most two coordinates on which the candidate fails...”

**Independent recomputation.** After the candidate's final pre-turn
commitment, fixed `T` and the one-stone offset force

`O_H=A disjoint-union {p_S,p_F}`, with `A=T[O_R]`.

There are exactly two cases.

1. If `p_F` has an `A` neighbor, `c=T^{-1}(p_F)` is real-empty and supported
   within radius eight by the inverse of that neighbor. A nonwinning `S@c`
   cannot be restored owner-faithfully under fixed `T`, because its required
   physical target is permanently `Fhat`, not `Shat`. If `S@c` wins, P5R
   requires an actual same-step legal `Shat` win. Such an append cannot exist
   on this genuine `sigma`-consistent node: extending it by least-legal
   off-path moves would be a counterstrategy to alleged-winning `sigma`.

2. Otherwise connectedness of the genuine physical support graph and the
   two-vertex proxy side force a shortest path

   `p_F -- p_S -- a`, with `a in A`.

   Put `c_1=T^{-1}(p_S)`. It is empty and supported. It cannot be terminal:
   the other five real-S cells of its terminal window would already map to five
   physical `Shat` cells, and physical `p_S` would be the sixth, contradicting
   the common-live shadow premise. If the first episode succeeds with its
   actual fresh `Shat` filler `w`, fixed `T` forces

   `A_1=A union {p_S}` and `P_1={p_F,w}`.

   The physical edge `p_S--p_F` persists. Therefore
   `c_2=T^{-1}(p_F)` is still real-empty, differs from occupied `c_1`, and is
   legal at `SecondStone` through `c_1`. Its target has the wrong physical
   color. A nonterminal append cannot restore owner fidelity; a terminal
   append fails P5R by the same winning-`sigma` argument.

The data are adaptive and candidate-own: `T,p_S,p_F` come from the candidate's
commitment, `w` is its actual filler, and only then is `c_2` selected from the
persistent edge. No S30-style labeled state is imported.

Fixed `T` is genuinely load-bearing. With an intra-pair change of isometry,
the forced identities for `A_1,P_1` need not remain true; a persistent edge
endpoint can change sides in the new cut. S43.1 therefore correctly leaves the
full `G_A^{2/pair}` success question open.

**Proposed repair:** optional exposition only—add “after the candidate's final
pre-turn commitment” to S43.1's theorem sentence. The definition already
supplies it.

### 4. NOTE — S44 and S44.1's no-discount inequalities close at fixed debt

> “Then `K` intersects `H_W` for every `W in U_E`.”

> “Even postponing the certificate check until after S's next first placement
> does not repair this fixed-debt branch...”

**Independent recomputation.** Take a pre-F urgent window `W` missed by the
completed real F pair `K`. Initially `W` contains no F stone. Its cells are
either S-owned or in `H_W`; because K misses `H_W`, neither legal F placement
enters `W`. With `X_S` and `E_S` fixed, the successor therefore still has

`W` E-live and `delta_R'(W)=delta_R(W)<=2`.

At the genuine common `Shat FirstStone` node, any Definition 39.1 selector
would give

`delta_H(nu(W)) <= delta_R'(W) <= 2`,

while S39 at `m=2` gives

`delta_H(nu(W)) > 2`.

This contradiction is pointwise for every successor selector, so dynamic
reselection of `nu` does not help. The two real F cells themselves must form a
transversal, establishing the necessity of `tau_E<=2` under S44's fixed-debt,
no-reconciliation, common-phase premises.

For S30, a two-stone K misses one of the three displayed positive-direction
hole pairs. Its offset-four cell remains empty, adjacent to S's offset-three
cell, and nonwinning by S30's cross-line census. Adding F blockers elsewhere
cannot create an S win. After that real first placement the missed window has
`delta_R=1` at common `SecondStone`, so a certificate would imply
`delta_H<=1`, contradicting S39 at `m=1`. The listed alternatives—physical
reconciliation, an actual same-step `Shat` stop, an earlier sound real-F stop,
or leaving the common-phase/Definition-39.1 premises—are exhaustive at this
conditional module scope.

**Proposed repair:** none.

### 5. NOTE — S45 enumerates all seven urgent S30 windows and proves `tau_E=5`

> “At S30's abstract labeled checkpoint the full urgent family has
> `tau_E=5`.”

**Independent recomputation.** Parameterize each of the three axis lines
through `e=(0,1)` by the increasing coordinate shown below. All six window
starts through the parameter-zero copy of `e` were checked.

| Axis line | S parameters | F-blocked starts | Urgent starts and hole sets | Exact local cost |
|---|---|---|---|---:|
| Q: `r=1`, parameter q | `{0,1,2,3}` | none | `-2:{(-2,1),(-1,1)}`; `-1:{(-1,1),(4,1)}`; `0:{(4,1),(5,1)}` | 2 |
| R: `q=0`, parameter r with `e` at 1 | `{1,2,3,4}` | starts `-4,-3,-2,-1,0`, all by `F@(0,0)` | `1:{(0,5),(0,6)}` | 1 |
| QR: `q+r=1`, parameter q | `{-4,0,1,2,3}` | none | `-2:{(-2,3),(-1,2)}`; `-1:{(-1,2),(4,-3)}`; `0:{(4,-3),(5,-4)}` | 2 |

On each Q/QR line the first and third urgent pairs are disjoint, so one cell
cannot suffice; the shared middle pair's two cells hit all three. The R family
needs one cell. The three axis lines meet only at S-owned `e`, which appears in
no hole set, so no blocker can serve two axes. Every transversal therefore has
size at least `2+1+2=5`.

Conversely, for example

`{(-1,1),(4,1),(0,5),(-1,2),(4,-3)}`

consists of five empty cells and hits every listed hole set. Thus five suffices
and every four-cell set fails. The inherited three-window argument was only a
subfamily lower bound; S45's sharpened exact value is correct.

**Proposed repair:** none.

### 6. NOTE — S40's event selector is causal, including the second prescription

> “computes (46.1) from the observed real prefix”

> “if both first placements are nonterminal, queries the now-reached
> `SecondStone` prescription `z_2`”

**Independent recomputation.** At the covered F `FirstStone` checkpoint, the
preceding S pair is complete. The physical real prefix therefore already fixes
`E_S`, the urgent family, all hole sets, the first minimum transversal, its
order, and the first eligible padding coordinate. A causal implementation is:

| Microstep | Real board | Shadow board | Information used |
|---|---|---|---|
| pre-first | select `k_1` from the post-S real prefix | query `z_1=sigma(h_H)` | only the two observed prefixes |
| first | append `F@k_1` | append `Fhat@z_1` | current physical boards |
| pre-second, only after two nonwins | select/validate `k_2` in `R+k_1` | query `z_2=sigma(h_H,z_1)` | the actual post-first prefixes |
| second | append `F@k_2` | append `Fhat@z_2` | current physical boards |

The shorthand “computes (46.1)” may also deterministically anticipate the
post-`k_1` state: whether `k_1` wins and the resulting legal set are functions
of the known board and `k_1`, not future opponent information. No S action
occurs between selection and use of either service coordinate. In particular,
the service enumeration is evaluated after S has had its chance to occupy an
earlier candidate; it skips whatever is then occupied. This does not meet
S12's intervening-S premise.

**Proposed repair:** optional clarity only—say “select `k_1` now and select
`k_2` lazily after a nonwinning first event,” matching the proof's ordering.

### 7. NOTE — S18 independence is real, while P5 is physical but conditional

> “S18-type proxy-only support of `z_i` cannot make `k_i` illegal, because no
> inverse-coordinate legality claim is made.”

> “every reached shadow-`Fhat` terminal placement ... has a same-step physical
> real-F terminal certificate”

**Independent recomputation.** Each `z_i` is tested only on the physical
shadow board. Each transversal member `k_i` is a real hole in a one- or
two-hole S window and is within line distance at most two of real physical
support; padding is selected from the current real legal set. No step invokes
`T^{-1}(z_i)`, common support, common vacancy, or an F-stone point map.
S18 therefore cannot invalidate the current real service append. A later
old-debt certificate can become occupied or illegal, but that later prefix then
fails `A_FS2` membership; no reverse-legality premise was used to justify the
current event.

The terminal bridge is also physical. At the actual pre-event boards,

`z_i in D_H^F => k_i in D_R^F`

means that the actual shadow append fills an actual `Fhat` window only when the
actual paired real append fills an actual F window. It is not a relabeling or
an event-index convention. S40 does not derive this implication for arbitrary
`sigma`; event-terminal alignment includes it as a trace-class premise, and
S41 proves that this premise is physically realizable on one complete trace.
The artifact discloses that conditional boundary repeatedly.

**Proposed repair:** none. Preserve “proved on terminal-aligned traces,” not
“universal P5 theorem,” in every detached summary.

### 8. MINOR — make the terminal closure's precedence over round-5 clauses 5–6 explicit

> “Let `A_FS2^ET(sigma)` be the `A_FS2` trace segments whose F-role steps are
> generated by Definition 46.1 and are event-terminal aligned.”

> S41's final row pairs real `F@(5,0)` with shadow `Fhat@(6,0)`, both winning
> first placements.

**Independent recomputation.** Round-5 Definition 38.2(5) says that the real
service operator stops on an earlier real-F win, and clause 6 requires a
genuine **nonwinning** shadow `Fhat` pair only unless real F has stopped the
trace. Under round 6's intended paired-event semantics, S41 is consistent:

- the real and shadow winning appends belong to one coupled microstep;
- the real append wins, so clause 6's “unless real F has stopped” antecedent
  is false at the completed event;
- neither engine receives a second placement; and
- there is no post-terminal continuation.

The prior handler, however, was written in real-service-then-shadow-pair order.
Calling the new class simply “the `A_FS2` trace segments” leaves a literal
ordering ambiguity: “stops the trace” can be read as forbidding even the
associated shadow append, while taking clause 6 first would forbid a winning
shadow append. The round-6 simultaneous-event reading resolves the issue, but
the definition should say so rather than relying on that reading.

**Proposed repair:** define `A_FS2^ET` as the terminal closure of live
`A_FS2` segments, with Definition 46.1 replacing clauses 5–6 on the final
paired F microstep. State explicitly that both associated physical appends
occur in that coupled event before the trace closes. This is a definition and
ordering clarification; it does not alter S41's coordinates or terminal
calculation.

### 9. NOTE — S41 is a legal, fresh, cadence-exact `A_FS2` handler trace

> “This is a complete finite member of `A_FS2^ET(sigma_star)`.”

**Independent recomputation: physical legality.** Every real append is fresh.
The following earlier stones supply radius-eight legality; the listed support
is not assumed to have the same owner.

| Real stage | Appends | Earlier support and distance |
|---|---|---|
| opening/S15 | `F@(0,0)` | compulsory opening |
|  | `S@(0,1),(0,2)` | `(0,0)` at 1; then `(0,1)` at 1 |
| seed service | `F@(1,0),(2,0)` | `(0,0)` at 1; then `(1,0)` at 1 |
| rolling pair 1 | `S@(0,3),(0,4)` | `(0,2)` at 1; then `(0,3)` at 1 |
| service 1 | `F@(0,5),(1,5)` | `(0,4)` at 1; then `(0,5)` at 1 |
| rolling pair 2 | `S@(1,4),(2,4)` | `(0,4)` at 1; then `(1,4)` at 1 |
| service 2 | `F@(3,0),(4,0)` | `(2,0)` at 1; then `(3,0)` at 1 |
| rolling pair 3 | `S@(8,0),(8,1)` | `(4,0)` at 4; then `(8,0)` at 1 |
| terminal service | `F@(5,0)` | `(4,0)` at 1; winning first placement |

Every shadow append is also fresh and legal on its own physical board:

| Shadow stage | Appends | Earlier support and distance |
|---|---|---|
| opening/first `sigma_star` pair | `Shat@(0,0)`; `Fhat@(1,0),(2,0)` | compulsory; then distances 1, 1 |
| S15 opponent pair | `Shat@(2,1),(2,2)` | `(2,0)` at 1; then `(2,1)` at 1 |
| seed `sigma_star` pair | `Fhat@(3,0),(4,0)` | distances 1, 1 along `r=0` |
| rolling pair 1 | `Shat@(0,1),(2,3)` | `(0,0)` at 1; `(2,2)` at 1 |
| service 1 prescriptions | `Fhat@(2,5),(3,5)` | `(2,3)` at 2; then distance 1 |
| rolling pair 2 | `Shat@(2,4),(3,4)` | `(2,3)` at 1; then distance 1 |
| service 2 prescriptions | `Fhat@(-8,0),(5,0)` | `(0,0)` at 8; `(4,0)` at 1 |
| rolling pair 3 | `Shat@(4,4),(10,0)` | `(3,4)` at 1; `(5,0)` at 5 |
| terminal prescription | `Fhat@(6,0)` | `(5,0)` at 1; winning first placement |

The owner/phase census is exact:

| Completed stage | Real `(F,S)` | Shadow `(Fhat,Shat)` | Next common role/phase |
|---|---:|---:|---|
| S15 prefix | `(1,2)` | `(2,3)` | F `FirstStone` |
| seed service | `(3,2)` | `(4,3)` | S `FirstStone` |
| rolling pair 1 | `(3,4)` | `(4,5)` | F `FirstStone` |
| service 1 | `(5,4)` | `(6,5)` | S `FirstStone` |
| rolling pair 2 | `(5,6)` | `(6,7)` | F `FirstStone` |
| service 2 | `(7,6)` | `(8,7)` | S `FirstStone` |
| rolling pair 3 | `(7,8)` | `(8,9)` | F `FirstStone` |
| paired terminal firsts | `(8,8)` | `(9,9)` | both terminal; seconds suppressed |

Thus every nonterminal first placement is followed by its same-owner second,
and each nonterminal second passes control. No cadence step is inferred from
stone counts alone.

**Independent recomputation: all six inherited `A_FS2` clauses.** The rolling
debt and service census is:

| Handler point | New first-coordinate test | Final debt | Urgent family / `tau_E` | Canonical real service |
|---|---|---|---|---|
| initial S15 state | no debt | empty | empty / `0` | `(1,0),(2,0)` padding |
| rolling pair 1 | `(0,3)`: at most 3 S cells per incident window, so `delta>=3` | `(0,4)` | sole window `q=0,r=1..6`, holes `(0,5),(0,6)` / `1` | `(0,5)` plus padding `(1,5)` |
| rolling pair 2 | `(1,4)`: at most 2 S cells, so `delta>=4` | `(2,4)` | every incident window has at most 3 S cells / `0` | padding `(3,0),(4,0)` |
| rolling pair 3 | `(8,0)`: at most 1 S cell on each incident line | `(8,1)` | every incident window has at most 2 S cells / `0` | padding `(5,0)`, which wins |

The second coordinates `(0,4)`, `(2,4)`, and `(8,1)` are nonwinning. The
old-debt certificates are, in order,
`T(0,3)=(2,3)`, `T(0,4)=(2,4)`, `T(1,4)=(3,4)`,
`T(2,4)=(4,4)`, and `T(8,0)=(10,0)`; each is fresh and shadow-legal at its
actual append. The only empty-queue filler is `(0,1)`, also fresh and legal.
None of these `Shat` appends wins, so the round-5 section-44 filler-terminal
erratum is obeyed.

At each old-debt rotation there is a physical microstate after real `S@y` and
before shadow reconciliation where both old `e` and new `y` are unmatched.
For every old E-live window,

`delta' - m' = (delta-m) + 1 - 1_{y in W} > 0`.

The separately checked first-safe/nonwinning conditions cover newly incident
windows through `y`. Thus the transient two-debt microstep is not skipped.
The selected services are exactly the first minimum transversals plus the
stated enumeration padding. All shadow F pairs before the last event are
genuine, sequential, and nonwinning. Subject only to Finding 8's terminal
closure wording, S41 satisfies clauses 1–6 on every step.

**Proposed repair:** none to the trace or handler proof.

### 10. NOTE — S41's inverse failures and simultaneous proxy-assisted windows are exact

> “Its second service directly realizes an S18-type prescription whose inverse
> is illegal, and its final paired first placements simultaneously complete a
> real F window and a proxy-assisted shadow-`Fhat` window.”

**Independent recomputation.** Immediately before service 2,
`z_1=(-8,0)` is fresh and

`d((-8,0),(0,0))=8`.

The next closest occupied shadow candidate is `(0,1)`, at distance nine; all
others are farther. Thus the persistent opener is its sole radius-eight
support. For `T(q,r)=(q+2,r)`, its inverse is `(-10,0)`. Every then-occupied
real coordinate has nonnegative q-coordinate, so every distance from
`(-10,0)` is at least ten. The inverse is genuinely illegal, while the paired
service `F@(3,0)` is fresh and adjacent to `(2,0)`.

At the last event, `T^{-1}(6,0)=(4,0)`, already real-F occupied. The carrier
instead uses the fresh, adjacent service cell `(5,0)`. Immediately beforehand:

- real F has the five-cell run `(0,0),...,(4,0)` and no other run longer than
  two;
- shadow `Fhat` has `(1,0),...,(5,0)` and no other run longer than two;
- real S has maximum run four on `q=0`; and
- shadow `Shat` has maximum run four on `q=2`.

The actual paired appends complete

`W_R={(0,0),(1,0),(2,0),(3,0),(4,0),(5,0)}`

and

`W_H={(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)}`.

The latter includes the original invented `Fhat` proxy `(1,0)`. Both are
first-placement wins; both seconds are suppressed. Because the preterminal
boards contain no six, every earlier board, being an owner-wise subset, is
also nonterminal. This is actual same-event P5 transfer, not point relabeling.

**Proposed repair:** none.

### 11. NOTE — S42 reaches the obstruction on the named selector's own history

> “There is one legal pure strategy `sigma_dagger`, one S15 synchronization,
> and one rolling first-safe/two-serviceable continuation on the carrier's own
> genuine `sigma_dagger`-consistent history...”

**Independent recomputation.** With `T(q,r)=(q+2,r)`, the initial real pair
`(0,2),(2,1)` maps to the physical shadow pair `(2,2),(4,1)` after the genuine
proxy/strategy prefix `Shat@(0,0); Fhat@(1,0),(2,0)`. All are fresh and the
successive support distances are at most two. The first canonical service is
real `(1,0),(2,0)` and the reached `sigma_dagger` pair is shadow
`(3,0),(4,0)`; both are legal and nonwinning.

For the rolling turn, real `(3,3)` is supported by `(2,1)` at distance three.
None of its Q, R, or QR axis lines contains an earlier S stone, so it is
first-safe. Filler `(0,1)` is adjacent to the shadow opener. Real `(5,2)` is
supported by `(3,3)` at distance two and is nonwinning; its old-debt
certificate `(5,3)=T(3,3)` is fresh and within distance three of `(4,1)`.
For final debt `(5,2)`, only `(0,2)` shares an incident axis line, at line
distance five, so every incident window has at most two S stones:
`delta>=4` and `tau_E=0`.

The fixed selector must therefore use its next padding pair `(3,0),(4,0)`.
It leaves real F with only `(0,0),...,(4,0)`, five stones and no terminal
window. The actual reached shadow prescriptions `(5,0),(6,0)` extend
`Fhat@(1,0),...,(4,0)`; the first is nonwinning and the second completes the
six. Every listed coordinate is fresh and legal, and the off-path least-legal
rule makes `sigma_dagger` a total legal pure strategy. Hence the witness is
selected from the promised selector and strategy behavior, not substituted
after the fact.

S42 attacks exactly Definition 46.2: a fixed selector, universal promise over
all legal pure strategies, and no physical P5 bridge. Its `sigma_dagger` is not
proved globally winning, so the theorem correctly does not attack an
alleged-winning-only carrier or any different selector.

**Proposed repair:** none.

### 12. MINOR — section 50 is not a literal complete carry-forward of the authoritative ledgers

> “This table uses the round-5 hostile review's list as the authoritative input
> state.”

> “Round-4 review's ten-item agenda, carried forward”

**Independent recomputation.** The twelve rows and ten rows are present in the
correct order, their global statuses are conservative, and the missing duties
remain acknowledged elsewhere in round 6. Four local omissions nevertheless
make the claim of exact carry-forward false as written:

1. Twelve-item obstacle row 1 mentions intra-episode total rebinding,
   non-total/window recoding, and indefinite per-placement repair, but not the
   separately open class of **total nonisometric zero-lag point recodings**.
   Definition 47.1's total grammar still uses translation/D6 isometries, so
   “intra-episode total rebinding” does not name that survivor.
2. Twelve-item obstacle row 10 drops the still-open duties for common-only real
   wins and for simultaneous legality/terminal maintenance. Those duties were
   explicit in the round-5 review's window-certificate item.
3. Ten-item agenda row 2 omits the S13 fixed-isometry FIFO regression, although
   folded round-5 section 44 requires S13 to be named in agenda items 2 and 6.
   Row 6 does name it.
4. Ten-item agenda row 3 omits S14's unguarded literal-lag terminal regression,
   although section 44 requires it there and in the P5R cross-ledger.

This is ledger incompleteness, not silent theorem inflation. Round 6 still
names S13/S14 in sections 45.3 and 45.4, the twelve-item rows 4 and 6, the
P5R cross-ledger, the attack surface, and the regression matrix. It calls the
global systems open. Common-only terminal transfer is also not claimed solved.

**Proposed repair:** add to obstacle row 1 that total nonisometric zero-lag
point recodings remain open; add to row 10 that common-only wins and
simultaneous legality/terminal maintenance remain open; restore the S13 caveat
to agenda row 2 and the S14 caveat to agenda row 3, including why guarded
`A_FS2` traces fall outside the unguarded regressions.

### 13. NOTE — section 35 quantifiers, prior regressions, provenance, and `NL_F` boundaries are otherwise honest

> “These are alternatives only for the current real-S placement. In every
> branch the candidate must also provide recurring P3 transfer ...”

> “`NL_F` remains OPEN.”

**Independent recomputation.** Section 45.4 preserves folded round-4 section
35 exactly: A/B/C are alternatives for one observed real-S coordinate, while
P3, service or reconciliation, persistence, P5/P5R, causal selection, and
strategy-domain legality remain conjunctive. It also retains the requirement
that a negative configuration be selected on the candidate's own legal,
`sigma`-consistent history. S42 meets that requirement for its universal-over-
legal-strategies named class; S43.1 meets it for every alleged-winning
`sigma` in the pair-static subclass; S30 remains explicitly abstract.

The inherited S13/S14 regressions are not claimed solved: event pairing falls
outside S13's inverse/FIFO premises, while S40 inherits guarded `A_FS2`
membership rather than proving unguarded lag safe. S18 is bypassed only for the
nonspatial F event interface. S20 is transferred only by the class premise and
one S41 instance. S25, S30, and S31 remain conditional tests. No sentence
promotes a carrier-class obstruction to an outcome theorem.

Git proves the stated parent/landing relation and byte identity recorded in
the method preamble. The only `GAP_RAW` occurrence in the artifact is its
provenance denial; no theorem, coordinate, citation, or commit-side dependency
uses such a file. Repository evidence cannot prove the historical mental fact
that an author never opened a file, but there is no textual or Git evidence of
mathematical dependence on one.

`NL_F` is marked open in the header, theorem ledger, obstacle ledgers, attack
surface, and resume point. D2 is used only as the logical bridge. Neither
determinacy alternative is selected.

**Proposed repair:** none beyond Finding 12's ledger additions.

## Per-theorem verdicts

| Result | Source status | Review verdict | Exact disposition |
|---|---|---|---|
| Production rule contract | PROVEN | **CONFIRMED** | Physical emptiness, radius-eight support, sequential insertion, immediate wins, terminal no-continuation, and append-only forward histories match the cited production predicates |
| Definition 46.1 P3 event carrier | Definition | **CONFIRMED** | A causal temporal pairing of two independently legal physical events; no inverse point map is imported |
| `A_FS2^ET(sigma)` terminal closure | Definition | **CONFIRMED-WITH-MINOR-ERRATA** | Intended simultaneous final event is coherent, but its precedence over round-5 clauses 5–6 should be explicit (Finding 8) |
| S40 causal service-event carrier | PROVEN at scope | **CONFIRMED AT CONDITIONAL TRACE SCOPE** | Genuine sequential `sigma` queries, canonical real legality, local S12 avoidance, and S18 independence are exact; P5 alignment and continuing membership are premises, not universal conclusions |
| S41 complete terminal-aligned trace | PROVEN | **CONFIRMED-WITH-MINOR-DEFINITION-ERRATA** | All real/shadow coordinates, cadence, handler clauses, transient debt, urgent families, illegal/occupied inverses, and simultaneous proxy-assisted terminal windows recompute exactly |
| S42 named terminal-blind obstruction | PROVEN | **CONFIRMED** | Legal total `sigma_dagger`, fixed-selector own history, `tau_E=0`, shadow second-placement win, and real five-stone nonterminal endpoint are exact |
| Definition 47.1 per-pair reset grammar | Definition | **CONFIRMED** | It changes only the lifetime counter and retains the total-exact, zero-lag, P3/P5/P5R obligations |
| S43 two-cut saturation/reset escape | PROVEN | **CONFIRMED** | Two nonterminal S coordinates consume two charges; the intervening F turn precedes the next fresh counter, so S32's lifetime count does not cross the reset |
| Definition 47.2 pair-static `T` subgrammar | Definition | **CONFIRMED** | Fixed `T` forces the two-proxy complement and is explicitly narrower than full intra-pair rebinding |
| S43.1 colored two-proxy cut | PROVEN | **CONFIRMED** | The `p_F`-touches-A / `p_F--p_S--A` dichotomy is exhaustive, adaptive, and physical; wrong-color restoration or P5R fails within the pair |
| Full `G_A^{2/pair}` success class | OPEN | **CONFIRMED OPEN** | An intra-pair binding change can move a persistent edge endpoint across the new cut; no repair or impossibility theorem is supplied |
| S44 no common-phase certificate discount | PROVEN | **CONFIRMED** | Fixed debt plus a missed urgent hole yields `delta_H<=delta_R<=2`, contradicting S39's `delta_H>2` at common `FirstStone` |
| S44.1 S30 certificate barrier | PROVEN at conditional scope | **CONFIRMED** | Three disjoint selected hole pairs defeat a real F pair; the delayed first-placement check gives the exact `m=1` contradiction; abstract-label scope is retained |
| S45 exact S30 transversal | PROVEN | **CONFIRMED** | Full urgent family has 3 Q, 1 R, and 3 QR windows with additive local costs `2+1+2=5`; an explicit five-cell transversal attains the bound |
| Section 50 twelve-item and ten-item ledgers | ledger maintenance | **CONFIRMED-WITH-MINOR-ERRATA** | Statuses are conservative, but four required survivors/regression cross-references are locally omitted (Finding 12) |
| Global P0–P6 plus P5R coupling | OPEN | **CONFIRMED OPEN** | No theorem gives universal `A_FS2^ET` coverage or an alternative complete branch system for every alleged-winning `sigma` |
| `NL_F` | OPEN | **CONFIRMED OPEN** | D2 remains only the determinacy bridge; no arbitrary alleged-winning S strategy is refuted |

No theorem receives **REFUTED** or **MAJOR**. The two **MINOR** findings concern
definition ordering and ledger completeness, not a failed coordinate, window,
strategy-domain, or terminal proof.

## Exact unresolved obstacles after review

The exact open state, including the entries omitted locally from section 50,
is:

1. **Full per-pair and broader zero-lag branch (A).** S43.1 excludes only a
   total-exact translation/D6 binding fixed within a pair. Intra-pair changing
   isometries, total nonisometric point recodings, non-total/window recodings,
   and one repair per placement indefinitely remain open.
2. **Pre-checkpoint and recurring P3 coverage.** For every alleged-winning
   `sigma`, every reached first and second `Fhat` prescription must be paired
   with a legal real F event or receive an earlier sound repair. S40 proves the
   interface only after membership in `A_FS2^ET` is known.
3. **Coverage outside `A_FS2`.** First-unsafe real-S coordinates, real-terminal
   coordinates, unavailable or occupied old-debt certificates, and
   `tau_E>2` states need branch (A), physical reconciliation, branch (C), or a
   different guarded invariant.
4. **P5R through every lag/recode.** Every real-only S cell must stay shielded,
   become physically certified, receive a real F blocker, or obtain an actual
   same-step shadow-`Shat` terminal certificate. S14 and S25 remain mandatory;
   common-only real wins remain an outer physical-transfer duty.
5. **Canonical F-service compatibility.** The post-S service selector is legal
   and causal when `tau_E<=2`, but no theorem proves that it remains fresh and
   serviceable on every winning-`sigma` history, especially after unrelated
   physical `Fhat` events.
6. **Universal shadow-`Fhat` terminal fidelity.** Event-terminal alignment is
   assumed by S40 and witnessed by S41. A first- or second-placement S20-type
   terminal prescription must be made co-terminal with real F for every
   alleged-winning `sigma`, or terminal misalignment must be forced on that
   carrier's own winning-strategy history.
7. **Reverse legality for spatial carriers.** Event pairing avoids inversion,
   but every inverse-map or fixed-FIFO proposal still owes S18 and S13 plus the
   sequentially updated unsupported/collision sets.
8. **Strategy domain and physical persistence.** Every filler, proxy, service
   cell, queue rotation, rebinding, and certificate change must remain one
   genuine append-only `sigma`-consistent history. Old stones retain occupancy,
   support, blocking, and every terminal-window effect.
9. **Global causality.** Definition 46.1 avoids S12 locally. Any outer branch,
   reconciliation plan, or future real-F backing coordinate must also avoid
   fixing an empty coordinate across an intervening S turn.
10. **Universal window-certificate maintenance.** S37 covers one fixed-selector
    append and S44 blocks one common-phase missed-service route. Dynamic
    selectors, newly admitted debt, phase-lagged/event-level certificates,
    common-only wins, and simultaneous legality plus terminal maintenance lack
    a universal physical handler.
11. **High-transversal service and permanent fencing.** S30 has exact
    `tau_E=5`, and S31's permanent fence costs six blockers geometrically.
    Reconciliation, blocker availability, interrupted installation, S
    occupation, and P3 compatibility remain open.
12. **Strategy-specific reachability and outcome.** S42's strategy is legal but
    not globally winning; S43.1 is strategy-specific only for its fixed-T
    subclass; S30 remains an abstract labeled state. Until every alleged-winning
    `sigma` is refuted or one global complete carrier is built, the global
    coupling and `NL_F` remain open.

## Overall verdict and objective dispositions

**Overall verdict: SOUND-WITH-MINOR-ERRATA.** Hostile recomputation found no
false theorem, illegal witness coordinate, missed earlier terminal window,
incorrect reset charge, bad colored cut, or erroneous transversal. The source
should repair the terminal-closure ordering and the locally incomplete ledgers.

1. **Section 46 / P3 objective — CONFIRMED AT CONDITIONAL SCOPE.** S40 is
   genuinely causal and does not use reverse shadow-to-real legality. Its P5
   implication is an actual same-step physical certificate but remains a class
   premise. S41 is a complete nonempty trace satisfying the inherited handler
   clauses and both S18/S20 stress demands. S42 kills the named terminal-blind
   selector on its own legal trace. Universal alleged-winning-`sigma` coverage
   remains open.
2. **Section 47 / per-pair `K=2` objective — CONFIRMED PARTIAL.** The reset
   really escapes S32's lifetime episode count. The pair-static-isometry
   subclass is adaptively refuted for every alleged-winning `sigma`; the fixed
   `T` premise is load-bearing. Full intra-pair rebinding remains open.
3. **Section 48 / S30 objective — CONFIRMED AT CONDITIONAL MODULE SCOPE.** At
   fixed physical debt and common phase, Definition 39.1 cannot discount a
   missed blocker. S30's full urgent family has exact `tau_E=5`, with both the
   five-cell upper bound and four-cell impossibility proved. Candidate-own
   reachability of the S30 label remains open.

**Most severe finding:** Finding 8, **MINOR**. The intended simultaneous final
event makes S41 coherent, but `A_FS2^ET` should explicitly state that this
terminal closure supersedes the sequential wording of round-5 clauses 5–6.
