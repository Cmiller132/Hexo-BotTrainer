# Horizon R3: true rules, quotient universes, and the next attacker rung

## Executive verdict

**CODE-FACT.** Non-opening placements are legal exactly on empty cells within
hex distance eight of an existing stone.  The decision path is
`HexoState::write_legal_moves` -> the incremental radius-eight legal store; the
EQ support builder consumes that list and applies the same `dist <= 8` filter.
The load-bearing sources are `constants.rs:5`, `state.rs:216-231`,
`legal.rs:135-138`, and `support.rs:106-108`.

**PROOF-SKETCH.** The old unbounded game is a conservative opponent-action
over-approximation, while every retained root-anchored cell is at distance at
most five from a root stone.  Deadline-relevant moves therefore have legal
representatives, and an illegal remote opponent move is outcome-inert or is
dominated by the corresponding retained legal action.  Positive strategies,
negative bounded refutations, and forced-loss strategies from R2/H10 all
transfer to the radius-eight game.  Section 2 gives the quantified lemma.

**MEASURED -- PASS.** The R3 h<=8 quotient returned identical current-WIN and
forced-loss Booleans on every one of the 6,443 frozen rows and on the requested
248-row grind audit: zero mismatches.  The grind rows overlap self-play, so
this is 6,443 unique frozen rows plus a separately checked 248-row slice.

**MEASURED.** Endpoint clock truncation plus incidence-twin quotienting reduced
the all-phase self-play current universe from `29/76/130` physical cells to
`17/52/91` quotient cells (p50/p90/max), and its loss universe from
`45/88/130` to `21/53/92`.  The prior 6.94-second self-play and 33.62-second
human loss tails became 0.45 and 0.46 seconds in this run.

**MEASURED -- PASS.** The true-rule H10 entry point caught all 78 certificates
which H10 had actually completed: 76 depth<=8 shortcuts and the two fresh
depth-10 witnesses.  Their witness cells are respectively at root-stone
distances `[2,1]` and `[2,2]`, so both turns are directly legal.

**PROOF-SKETCH.** The wholly remote constant extends to the next rung.  Before
the last attacker placement(s), at most six newly placed attacker stones can
support a root-empty threat.  Two distinct hex lines containing at least four
of those stones would need at least seven stones; hence all such threats lie
on one axis.  Exhaustion of 259 translation/reflection classes found a
one/two-cell cover for every class, with zero failures.

**MEASURED -- RUNTIME BOUNDARY, NOT VERDICT.** The finite H13/H14 endpoint was
implemented as `exists A1, forall D1, exists A2, forall D2, exists A3` followed
by a one-cover (fresh h13) or two-cover (h14 and SecondStone h13) test.  All 155
new depth-13/14 certified WIN roots reached the 250 ms harness boundary; none
returned a negative Boolean.  Three representative five-second runs also
timed out.  Thus R3 closes the semantic/infinite-universe obstruction, but it
does **not** close the Python runtime frontier or claim a new completed h14
verdict.

## 1. Artifacts and scope

**CODE-FACT.** The implementation is
[`horizon_r3.py`](../.scratch/horizon_r3.py).  Evidence is split so a long
universal tail cannot destroy completed measurements:

- **MEASURED:** [`horizon_r3_h8.json`](../.scratch/horizon_r3_h8.json) is the
  complete all-row equivalence, universe, node, and wall frame.
- **MEASURED:** [`horizon_r3_h10.json`](../.scratch/horizon_r3_h10.json) is the
  78-certificate H10 replay.
- **MEASURED:** [`horizon_r3_legality.json`](../.scratch/horizon_r3_legality.json)
  is the radius-eight/anchored-distance audit.
- **MEASURED:** [`horizon_r3_next.json`](../.scratch/horizon_r3_next.json) is the
  complete 155-root bounded next-rung frame and cohort bite floors.
- **MEASURED:** [`horizon_r3_boundaries.json`](../.scratch/horizon_r3_boundaries.json)
  contains three five-second boundary probes.
- **MEASURED:** [`horizon_r3_lemmas.json`](../.scratch/horizon_r3_lemmas.json)
  contains the clock table and six-stone exhaustive lemma.

**CODE-FACT.** All new implementation work is Python.  No engine, verifier,
Lean, Cargo, or configuration file was edited, no Cargo command was run, and
no commit was created.

## 2. The true-rules transfer bridge

### 2.1 Rule statement

**CODE-FACT.** For a non-opening position `P`, define

```text
L(P) = { x | x is empty and exists stone z in P, hexDist(x,z) <= 8 }.
```

**CODE-FACT.** `L(P)` is exactly the engine single-placement action set.
Opening is the separate forced action `{(0,0)}`.  A placement adds its own
radius-eight disk to the incremental legal store, so legality is evaluated at
each physical prefix, including between the two stones of a turn.

### 2.2 Transfer lemma

**PROOF-SKETCH -- true-rules transfer lemma.** Let `Q_infinity(P,s,T)` be any
R2/H10 deadline game on schedule `s` and target `T`, using the old unbounded
empty-cell rule, after its relevance/inert normalization.  Let
`Q_8(P,s,T)` use `L` at every prefix.  Assume:

1. every retained non-inert action has a legal ordering at its prefix;
2. every removed action changes no window completable by the deadline; and
3. each inert class has a concrete legal representative whenever it is used.

Then `Q_infinity(P,s,T) = Q_8(P,s,T)`.

**PROOF-SKETCH.** Induct on the schedule.  A retained move has the same window
transition in both games by condition 1.  A removed move is replaced by the
legal inert representative from conditions 2-3.  If that representative was
reserved for a later move, exchange the two placements: stones are permanent,
earlier ownership of a relevant cell can only advance the mover's live windows
and block the opponent's.  The induction preserves every first terminal
prefix, so existential and universal nodes have the same value.

**PROOF-SKETCH -- direction-level reading.** An unbounded positive attacker
strategy already beats a superset of opponent actions; if its witnesses are
legal, it remains positive.  An unbounded forced-loss strategy likewise beats
a superset of attacker actions and transfers when its defender witnesses are
legal.  For a negative current-attacker result, a hypothetical legal win can
be normalized into the same retained legal classes; every extra unbounded
opponent action is inert or dominated by one of those legal classes.  It would
therefore contradict the old exhaustive negative.  The dual negative follows
symmetrically.

### 2.3 Why the hypotheses hold through H10

**PROOF-SKETCH.** A length-six window containing a root stone has diameter five.
Every root-empty cell in that anchored window is therefore within distance
five of an existing stone and is legal before any future move.  At h<=8 every
completable window is root-anchored because each player has fewer than six
placements.  Conditions 1-3 therefore reduce exactly to R2's
root-window-ancestry and inertness lemmas.

**MEASURED.** Across both endpoint targets on all 6,443 unique frozen rows,
474,525 anchored-cell references were checked.  Their maximum minimum distance
to a root stone was four and there were zero violations.  Across the fresh
portion of the 78-row H10 test cohort, 7,417 anchored-cell references had
maximum distance five and zero violations.

**PROOF-SKETCH.** H10's old infinite branch consisted of translated root-empty
windows outside the finite interaction halo.  Under the real rule, first moves
come from the finite `L(P)` carrier; there is no infinite translation choice.
Moreover, every final two-cover cell is within five of an attacker stone and
is legal when the cover is played.  The old remote constant remains LOSS, but
the translation machinery is now semantically vacuous rather than needed for
finiteness.

**MEASURED.** H10's two fresh witness pairs are
`(-4,-5),(12,-12)` for `human_b132a09ccb4eb829_p101` and
`(6,-12),(10,-19)` for `sp_20_p77`.  Their per-cell minimum root distances are
`[2,1]` and `[2,2]`; both cells in each pair are root-legal even though the
first human pair's cells are far from each other.

## 3. Universe shrink through H10

### 3.1 Exact pruning and quotient lemmas

**PROOF-SKETCH -- endpoint-clock pruning.** For target `T`, truncate the
schedule immediately after `T`'s last placement.  A window for player `p`
requires at most `count(p,truncated_schedule)` root-empty cells.  A cell found
only in a window above that threshold cannot contribute to a first terminal
prefix and is inert.  This fixes the prior `SecondStone` h8 loss model: the
defender wins no later than ply seven, so the attacker has three relevant
placements, not four.

**PROOF-SKETCH -- node-local quota pruning.** At a node with `k_p` remaining
placements for player `p`, discard a live `p` window whose residual has more
than `k_p` cells.  Candidate actions are drawn only from surviving own
residuals and opponent residuals completable within the opponent's remaining
quota.  The discarded cells cannot lie in any deadline terminal window.

**PROOF-SKETCH -- incidence-twin quotient.** Give every physical cell its
complete tagged incidence vector over all retained player-0 and player-1
windows.  Cells with equal vectors are exchangeable.  Keep the class
multiplicity, and represent an action only by how many members (zero, one, or
two) it takes from each class.  Any permutation inside a class preserves every
window residual and first terminal prefix.  This is an exact quotient; it does
not incorrectly merge two required cells into one.

**CODE-FACT.** R3 still uses arbitrary-precision Python integer masks.  Same-turn
pairs are unordered, terminal first-placement prefixes are retained, and
node-local pair streams have exact exhaustive fallbacks after tactical
prefixes.  These are the pair-commutation and transposition quotients already
present in H10, now combined with incidence classes.

**PROOF-SKETCH -- legality pruning at H10.** A fresh H10 first pair is retained
only if one cell is root-legal and the other is root-legal after that prefix.
Later endpoint cells lie in a window containing a root or newly placed stone,
so their distance is at most five.  Filtering the old first-pair stream is
therefore exact under the true rule.

**CODE-FACT.** D6 is used where it has a nontrivial generic payoff: axes and
reflections of the remote shape lemmas.  The frozen roots are generally
asymmetric, so no unsupported global-board D6 stabilizer quotient is assumed.

### 3.2 H8 universe distributions

**MEASURED.** Values are p50 / p90 / max over all phases.  `endpoint` is the
physical universe after clock truncation; `quotient` is the number of retained
incidence classes with multiplicities stored separately.

| cohort | current before physical | current endpoint | current quotient | loss before physical | loss endpoint | loss quotient |
|---|---:|---:|---:|---:|---:|---:|
| self-play (3,255) | 29 / 76 / 130 | 29 / 76 / 130 | 17 / 52 / 91 | 45 / 88 / 130 | 33 / 76 / 130 | 21 / 53 / 92 |
| human (2,720) | 34 / 70 / 231 | 34 / 70 / 231 | 22 / 50 / 151 | 47 / 82 / 260 | 37 / 72 / 260 | 25 / 51 / 143 |
| puzzle (468) | 25 / 60 / 132 | 25 / 60 / 132 | 20 / 44 / 99 | 41 / 81 / 151 | 37 / 72 / 151 | 30 / 56 / 100 |
| grinds (248) | 42 / 78 / 128 | 42 / 78 / 128 | 25 / 57 / 86 | 74 / 108 / 130 | 67 / 106 / 130 | 46 / 73 / 92 |

**MEASURED.** Current clocks do not shrink physically because the prior current
specializations already stopped at the target endpoint.  The additional loss
shrink comes from the corrected partial-turn endpoint; incidence classes then
reduce both sides.

### 3.3 H8 wall before/after

**MEASURED.** Times are CPython wall from the predecessor JSON versus the R3
run on the same machine.  They are research-frame timings, not Rust latency.

| cohort | current mean before -> after | loss mean before -> after | current max before -> after | loss max before -> after |
|---|---:|---:|---:|---:|
| self-play | 4.52 -> 3.94 ms | 21.53 -> 4.85 ms | 0.47 -> 0.18 s | 6.94 -> 0.45 s |
| human | 5.60 -> 4.13 ms | 55.05 -> 6.20 ms | 0.70 -> 0.30 s | 33.62 -> 0.46 s |
| puzzle | 3.66 -> 2.98 ms | 44.64 -> 4.88 ms | 0.14 -> 0.07 s | 5.34 -> 0.12 s |
| grinds | 9.66 -> 6.73 ms | 30.85 -> 10.51 ms | 0.05 -> 0.05 s | 0.35 -> 0.12 s |

**MEASURED -- PASS.** Verdict totals remained 101/17 self-play current/loss,
157/201 human, 20/22 puzzle, and 0/0 grinds.  Every opening, `FirstStone`, and
`SecondStone` row was compared directly; mismatches were zero.

### 3.4 H10 shrink and exactness boundary

**MEASURED.** The H10-tested certificate cohort is less compressible at the
root because overlapping root-empty halo windows distinguish most cells.

| cohort reference set | rows | before `U/V` p50/p90/max | incidence classes p50/p90/max |
|---|---:|---:|---:|
| self-play | 8 | 185 / 666 / 819 | 159 / 666 / 819 |
| human | 24 | 442 / 608 / 1,339 | 441 / 608 / 1,336 |
| puzzle | 7 | 210 / 555 / 896 | 159 / 555 / 896 |
| all 78 unique certificates | 78 | 342 / 634 / 1,339 | 337 / 634 / 1,336 |

**MEASURED.** On fresh tested references, the unbounded first-pair stream was
`10,439/28,733/178,540`; true legality reduced it to
`10,424/28,670/178,471`.  On the two non-shortcut witnesses, pair-conditioned
live carriers peaked at only 145 and 135 cells despite root `V` sizes 896 and
819.  This branch-local shrink, not raw incidence compression, is the useful
H10 result.

**MEASURED -- PASS.** All 78 completed H10 certificates were caught with zero
misses.  The human witness used 34,703 nodes and 1.30 seconds; the self-play
witness used 689,531 nodes and 7.83 seconds in the final frame.  H10's recorded
predecessor walls were 1.18 and 9.78 seconds; ordering noise dominates the
first comparison, while verdicts, nodes, and witness pairs are unchanged.

## 4. The actual rung clocks

**CODE-FACT.** A horizon counts physical placements, and a first-placement win
terminates before the nominal second placement.  Quotas alone therefore do not
make h13 equal h14.

**CODE-FACT.** The relevant schedules are:

| root phase | h10 | h11 | h12 | h13 | h14 | h15 | h16 |
|---|---|---|---|---|---|---|---|
| fresh `FirstStone` | `AADDAADDAA` (6/4) | `...D` (6/5) | `...DD` (6/6) | `...DDA` (7/6) | `...DDAA` (8/6) | 8/7 | 8/8 |
| `SecondStone` | `ADDAADDAAD` (5/5) | 5/6 | 6/6 | 7/6 | 7/7 | 7/8 | 8/8 |
| opening | `ADDAADDAAD` (5/5) | 5/6 | 6/6 | 7/6 | 7/7 | 7/8 | 8/8 |

**CODE-FACT.** Each `(a/d)` pair is attacker/defender placement quota from the
root mover's perspective.  The opening schedule has the special forced origin
as its first `A` placement.

**PROOF-SKETCH -- fresh collapse.** Fresh h10 ends with attacker placements
9-10.  Placements 11-12 belong to the defender, so
`WinWithin10 = WinWithin11 = WinWithin12` for the current attacker.  H13 adds
the first placement of the next attacker turn and h14 adds its second;
therefore h13 and h14 are both genuine rungs and need not be equal.  H15-16 add
only defender placements, so current-attacker `WinWithin14 = WinWithin16` at a
fresh root.  The loss dual does not collapse across those defender additions.

**CODE-FACT -- partial/opening clocks.** At a `SecondStone` or opening root,
h10-h11 have no new attacker placement; h12 and h13 are the next attacker
pair.  H14-h15 add only defender placements, and h16 adds the first placement
of the following attacker pair.  The six depth-13 certificates in the registry
are all `SecondStone`, matching this shifted clock; all 149 depth-14
certificates are fresh.

## 5. Next-rung normalization and endpoint

### 5.1 Six-stone two-cover lemma

**PROOF-SKETCH.** Let `X` contain at most six attacker stones and no completed
six.  Let `F(X)` be residual sets of size one or two from length-six windows
containing at least four stones of `X`.  Any two supporting geometric lines
which are distinct intersect in at most one cell, so they would contain at
least `4+4-1=7` stones.  Hence every member of `F(X)` lies on one axis line.

**PROOF-SKETCH.** Normalize one supporting interval to `[0,5]`.  A stone which
can participate in another supporting interval lies within five of a base
stone, so the finite carrier is `[-5,10]`.  Enumerate sets of size four through
six, require at least four base-interval stones, and quotient translation and
reflection.  Every residual family has a hitting set of size at most two.
D6 maps the one-axis result to all board axes.

**MEASURED -- PASS.** The executable enumeration contains 259 canonical
classes and zero failures.  This strictly extends H10's ten four-stone line
shapes.

### 5.2 Interaction normalization

**HYPOTHESIS -- proof-ready, not Lean-accepted.** Define `U+` from every
root-pure window containing a same-color root stone and completable within the
player's rung quota.  Define `N` as root-empty windows meeting `U+`, and
`V = U+ union union(N)`.  A terminal window outside `V` is wholly remote.  Its
pre-final threat family is covered by the six-stone lemma.  The at-most-two
attacker placements outside that terminal component can be normalized into
the anchored interaction; intervening defender pairs answer them before the
last pair is reserved for the remote cover.  Conversely, a wholly remote
defender construction is dominated by spending the same pair immediately on
the live interaction family.  Thus all nonconstant interaction lies in `V`.

**HYPOTHESIS -- exactness status.** This is a complete paper proof sketch and
the code exhausts the resulting finite quotient, but the new mixed
anchored/remote normalization has not been independently formalized.  R3 does
not promote it beyond the same proof-relative exactness standard used by H10.

### 5.3 Endpoint formula

**CODE-FACT.** For fresh h14 the implementation searches

```text
exists A1, forall D1, exists A2, forall D2, exists A3:
  A3 wins now
  or (D has no completion pair and tau(A final threats) > 2).
```

**CODE-FACT.** Fresh h13 uses the same nesting but the final attacker capacity
is one; after D's last pair, more than two distinct singleton completion cells
are required.  `SecondStone` h13 starts with one attacker placement and ends
with a final pair, so it uses the two-cover endpoint.

**CODE-FACT.** The implementation applies remaining-quota window thresholds,
root-empty seeding thresholds, exact defender-cover action generation,
node-local incidence quotients, unordered pair commutation, arbitrary-precision
integer masks, memo-free early exits, and an exhaustive fallback.  A harness
deadline raises `HarnessTimeout`; it never returns `False`.

## 6. Next-rung measurements and scaling

### 6.1 Registry boundary

**MEASURED.** The registry has 2,941 unique known WIN roots: 123 are eligible
at depth<=10, six have depth 13, and 149 have depth 14.  There are no depth-11
or depth-12 certificates, as the fresh clock predicts.

**MEASURED -- BOUNDED.** All 155 new eligible roots were attempted with 250 ms
per-root deadlines.  Results were 0 completed WIN, 0 completed negative
mismatches, and 155 timeouts.  Their root `V` sizes were
`418/564/898` physical cells and exactly the same incidence-class counts;
normalized first-action counts were `37,494/64,043/212,691`.

**MEASURED -- BOUNDED.** Five-second probes ended as follows:

| root | phase/depth | nodes at boundary | A3 endpoint pairs | `V` |
|---|---:|---:|---:|---:|
| `human_41e78c67c2ac8570_p20` | SecondStone / 13 | 207,872 | 190,730 | 448 |
| `atlas_full_oa-c515cddcef6134b3` | fresh / 14 | 150,528 | 128,731 | 468 |
| `sp_0_p51` | fresh / 14 | 167,936 | 164,195 | 745 |

**MEASURED.** The cost is no longer an infinite-carrier problem; it is the
finite existential A3 pair scan nested below universal D2 replies.  The root
incidence quotient has no bite on these near-window-dense positions, so a SAT,
BDD, or direct hypergraph endpoint solver is the next implementation step.

### 6.2 Certified bite floors

**MEASURED.** These are all-row lower bounds: exact production h8 positives
union engine certificates at the stated depth.  They are not R3 h14 firing
rates, because the new endpoint produced no completed new verdict.

| cohort | rows | exact h8 wins | h10 certified floor | h14 certified floor | delta floor |
|---|---:|---:|---:|---:|---:|
| self-play | 3,255 | 101 | >=102 | >=107 | >=5 |
| human | 2,720 | 157 | >=163 | >=176 | >=13 |
| puzzle | 468 | 20 | >=21 | >=23 | >=2 |
| grinds | 248 | 0 | >=0 | >=0 | >=0 |
| forcing-19 | 19 | 0 | >=0 | >=0 | >=0 |

**MEASURED.** As at H8/H10, the next rung has no demonstrated bite on the
248-row grind target.  Human data supplies the largest certified delta.

### 6.3 H16 feasibility

**CODE-FACT.** The production seam accepts semantic horizon zero or at least
16.  At a fresh root, current-attacker `WinWithin16` equals `WinWithin14`
because h15-16 are defender placements.  A completed fresh H14 decider would
therefore wire directly at the first accepted bounded seam value without
changing its current-WIN semantics.

**HYPOTHESIS.** Fresh current-WIN feasibility at h16 is exactly the unfinished
h14 runtime problem, not a larger semantic universe.  Fresh forced-loss at h16
is harder because the defender grows from six to eight placements.  At
`SecondStone` and opening roots, h16 adds a genuine attacker placement and
requires another endpoint case.  Engine integration is therefore premature
and was not attempted.

## 7. Reproduction and hashes

**CODE-FACT.** Completed commands were:

```powershell
python .scratch\horizon_r3.py --h8-battery --out .scratch\horizon_r3_h8.json
python .scratch\horizon_r3.py --h10-cohort --out .scratch\horizon_r3_h10.json
python .scratch\horizon_r3.py --legality-audit --out .scratch\horizon_r3_legality.json
python .scratch\horizon_r3.py --next-rung --next-root-ms 250 --out .scratch\horizon_r3_next.json
python .scratch\horizon_r3.py --next-boundaries --boundary-seconds 5 --out .scratch\horizon_r3_boundaries.json
python .scratch\horizon_r3.py --lemmas-only --out .scratch\horizon_r3_lemmas.json
```

**MEASURED.** The worktree base was
`e118097075a2f46afcb30f8c38b0c2c98666eab0`, using Python 3.14.0.  Final
SHA-256 values are recorded in `.scratch/horizon_r3_hashes.json` after all
artifacts and this report were finalized.

## 8. Bottom line

**MEASURED.** Goal 1 is complete: exact h<=8 equivalence passed the full
battery, H10 passed its complete predecessor test cohort, and the search
universes/tails shrank without a verdict mismatch.

**CODE-FACT.** The owner-rule correction strengthens rather than invalidates
the predecessor results: anchored witnesses are legal, remote opponent choices
were conservative, and the H10 infinite-translation premise is absent from the
real game.

**MEASURED / HYPOTHESIS.** Goal 2 is semantically advanced but not runtime
closed.  The true rung clocks, the six-stone remote cover, the finite nested
endpoint, all eligible registry accounting, and honest floors are complete.
The Python endpoint did not finish a new depth-13/14 certificate inside the
reported boundaries; claiming a new rate or negative verdict would be
fabrication.
