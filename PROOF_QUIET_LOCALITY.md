# NQ2 quiet-locality proof, round 1

Repository state examined: `a5c9c69b7fb2095371c1bad72d7170bf1c1943d7`.
No commit was made.

## Verdict

**REFUTED.**  On the production-coherent SecondStone reading used by the hunt
and by the proposed per-Choice pruning, `join_live`, `join_adj2`, and
`join_adj1` are all incomplete.  A frozen position has a verifier-accepted win
whose unique surviving completion is the quiet move `(6,-6)`.  That cell is in
no live attacker window and is hex-distance **6** from the nearest attacker
stone.  The root has 538 legal completions; the other 537 all permit an
immediate defender win.

The strongest unconditional position-only universe proved complete for every
quiet win remains the full legal set.  A smaller exact restriction is proved
for **urgent SecondStone** nodes: retain attacker wins-now and cells that hit
every current defender count-4/count-5 window.  On the frozen counterexample
that kernel is the singleton `{(6,-6)}`.  It is not a proof of locality at
non-urgent nodes.

## Claim ledger

| ID | Claim | Status | Result |
|---|---|---|---|
| Q0 | The report's “quiet gate fires at OR node `P`” describes production literally. | **REFUTED** | Quiet is a post-SecondStone edge property; the consume fallback is unconditional after the ordinary frontier fails. |
| Q1 | The seven previously frozen distinct placements hit `join_adj1`. | **VERIFIED-EXHAUSTIVE** | True only for those seven records; it is empirical coverage, not completeness. |
| Q2a | A move outside `join_live` changes no old live-window count and births only count-1/delta-5 windows. | **PROVEN** | Window-level lemma. |
| Q2b | Such a move creates only new overlap-connected families. | **REFUTED** | A born window can overlap and merge into an old component. |
| Q2c | “No useful remote seed”: every remote quiet move can be pulled into the local universe without losing the strategy. | **REFUTED** | The required remote defensive block is a counterexample under general branching defense. |
| Q3 | `join_live` is complete for quiet-required wins. | **REFUTED** | Frozen witness; 141 local candidates all lie among 537 losing alternatives. |
| Q4 | `join_adj2` is complete for quiet-required wins. | **REFUTED** | Frozen witness; 75 candidates, all losing. |
| Q5 | `join_adj1` is complete for quiet-required wins. | **REFUTED** | Frozen witness; 38 candidates, all losing. |
| Q5a | Adjacency-only `adj_stone_k1` or `adj_stone_k2` is complete. | **REFUTED** | The unique winning completion has `d_stone=6`. |
| Q6 | Exhausting one of the refuted universes licenses “no unrestricted win.” | **REFUTED** | The witness would be a false negative; current exhaustion is `UNKNOWN` in any event. |
| Q7 | The stated finite alternative elimination and positive continuation check close. | **VERIFIED-EXHAUSTIVE** | 538/538 root moves classified; one winning, 537 immediate losses; certificate verified. |
| Q8 | The urgent SecondStone reply-survival kernel below is complete. | **PROVEN** | Direct one-turn argument, independent of the solver. |
| Q9 | Full legal is complete. | **PROVEN** | Identity universe; no locality shrink. |

No `PROVEN` label in this document is inferred from a cap-limited miss.

## 1. Statement fidelity and production scope

### 1.1 Verbatim target

From `HUNT_REPORT_QUIET_LOCALITY.md`:

> **NQ2 locality conjecture.** Let `P` be an attacker OR node that is *unforced*
> (the quiet-turn gate fires). If the attacker has a win from `P` whose winning
> line begins with a quiet turn, then it has such a win whose quiet placement
> `c` satisfies **both** (i) `c` lies in a live attacker window — an active
> length-6 window with ≥1 attacker stone and 0 defender stones — and (ii)
> `dist(c, nearest attacker stone) ≤ 1`. Hence the certified quiet universe
> `C(P) = { legal c : (i) ∧ (ii) }` is complete for quiet-required wins.

The phrase “gate fires at `P`” is not a literal production predicate.  In
`tss_solver.rs`, `prove_choice` first tries the ordinary attacker frontier.  If
consume is enabled, it then enumerates `write_legal_moves` unconditionally
(`tss_solver.rs:3414-3575`).  This happens at Opening, FirstStone, and
SecondStone Choice nodes.  The function
`turn_forces_small_defender_reply` (`tss_solver.rs:3990`) is evaluated on the
**post-placement** state.  The shadow walker calls a turn quiet only when the
pre-move phase was SecondStone and the post-move predicate is false
(`tss_solver.rs:5346-5359`).

The hunt's five constructed quiet specimens also start at SecondStone; their
FirstStone was pre-root.  Therefore the production-coherent and empirically
used target is:

> At a reachable attacker SecondStone Choice state `P`, may the full-legal
> fallback omit every nonterminal quiet completion outside `C(P)` without
> losing a verifier-admissible winning strategy?

The frozen witness answers **no**.  If instead `P` is newly required to be the
FirstStone state before the whole ordered pair, that is a different, stronger
pair-normal-form statement.  It remains **OPEN**, and it cannot license the
current proposed per-Choice pruning at SecondStone.

### 1.2 Exact quiet predicate

Let `A` be the fixed certificate claimant, let `c` be played from a
SecondStone state, and let `Q=P+c` be nonterminal, with the defender to move.
Production defines

```text
Forces_A(Q) :=
    winner_from_analysis(Q) = A
    OR
    (Q is not Opening
     AND opp_threat_count(Q) > 0
     AND NOT own_win_now(Q)
     AND min_hitting_set(Q) = b(Q)).

Quiet_A(P,c) := NOT Forces_A(P+c).
```

At an ordinary post-turn state `b=2`.  `strict_quiet`—no active attacker
count-4-or-higher window—is a different predicate.  The frozen witness is
loose-quiet: attacker threats exist, but their hitting number is one, below
the defender budget two.

### 1.3 Candidate sets

All sets are computed from the pre-placement state `P`:

```text
Live_A(P) = { active length-6 W : count_A(W)>=1 and count_D(W)=0 }.

join_live(P) =
  { c in Legal(P) : some W in Live_A(P) contains c }.

join_adjk(P) =
  { c in join_live(P) : min_{A-stone s in P} hex_dist(c,s)<=k }.
```

The report's conjectured universe is `join_adj1`.  Round 1 also audits
`join_adj2` separately.

## 2. Frozen required-remote counterexample

### 2.1 Replay and root

The exact legal replay is:

```text
[[0,0],[-1,0],[1,-1],[1,0],[2,0],[2,-2],[3,-3],[3,0],
 [4,6],[4,-4],[5,-5],[1,3],[2,3],[2,1],[5,5],[3,3],
 [0,4],[6,2],[-1,5],[0,5],[0,6],[7,6],[1,6],[5,7],
 [6,7],[6,6],[3,6],[7,7],[5,6],[-1,6],[1,4],[6,5],
 [7,4],[7,3],[7,5],[6,0]]
```

After these 36 placements:

- claimant/attacker `A` is Player 0;
- the phase is `SecondStone { first: (6,0) }`;
- `P` is nonterminal;
- `|Legal(P)| = 538`;
- Player 1 owns `(1,-1),(2,-2),(3,-3),(4,-4),(5,-5)`;
- `(6,-6)` is the empty completion of that Player-1 count-five; and
- Player 0 at `(0,0)` blocks the opposite extension of the same five-stone
  segment.

Set `r=(6,-6)`.

### 2.2 Locality failure [VERIFIED-EXHAUSTIVE]

At `P`, `r` is engine-legal.  Its measured properties are:

```text
d_stone(r)       = 6
r in join_live   = false
r in join_adj2   = false
r in join_adj1   = false
```

The candidate sizes at this exact root are:

| universe | size | contains `r` |
|---|---:|---:|
| full legal | 538 | yes |
| `join_live` | 141 | no |
| `join_adj2` | 75 | no |
| `join_adj1` | 38 | no |
| `adj_stone_k2` | 93 | no |
| `adj_stone_k1` | 39 | no |

After `A` plays `r`, the state is nonterminal and the exact engine forcing
predicate is false, so the completed turn is quiet.

`r` is an urgent defender-block candidate in the ordinary frontier, but the
ordinary SecondStone path recurses only when the completed pair both creates a
new attacker count-four-or-higher threat and passes the forcing gate.  This
quiet block is therefore recovered only by the full-legal consume fallback—the
exact fallback `join_adj1` was intended to shrink.

### 2.3 Necessity [VERIFIED-EXHAUSTIVE]

The ignored test enumerates all 538 legal completions at `P`.

- For each of the 537 cells `c != r`, `A` does not win on `c`.
- The turn then passes to Player 1.
- `r` is still legal for Player 1.
- Player 1's placement at `r` immediately completes six and ends the game.

Thus every `c != r` is losing under one explicit legal defender continuation.
This is a game-rule proof, not a restricted-search failure.  In particular,
every member of all three proposed locality tiers loses.  If `r` wins, it is
the unique winning root placement.

### 2.4 Sufficiency against general branching defense [PROVEN]

The test applies `r` and asks `round3_consume` for a `SolveGoal::Loss`
certificate from the resulting defender-to-move state.  “Loss” is relative to
the new root player; its fixed claimant is still Player 0.  The result was:

```text
status                   Loss
search nodes             4,957
child certificate nodes  3,857
absolute horizon         66
verifier(child, Loss)    accepted
```

The harness then prepends the exact parent
`Choice { mv: (6,-6), child: old_root }`, rebinds the certificate to `P`, and
runs the independent verifier again:

```text
parent certificate nodes 3,858
verifier(P, Win)          accepted
```

The Universal nodes in that certificate quantify over general defender
branching using the production verifier rules.  Combining this positive
certificate with §2.3 proves that `r` is the unique winning completion.

### 2.5 Counterexample family

The canonical witness above is frozen.  Hex geometry, window ownership,
distance, legality radius, the gate, and certificates are D6-covariant, so its
rotations/reflections form the corresponding D6 counterexample orbit.  The
canonical member—not all orbit images—was the member counted in the round-1
machine tally.

This family is the named danger case from the attack plan: **defensive tempo**.
The remote move does not advance an old attacker window; it prevents the
opponent's immediate win while the first stone `(6,0)` supplies the attacker's
build progress.

## 3. Why the proposed build potential does not prove a swap

### 3.1 What is true at window level [PROVEN]

For a live attacker window `W`, define

```text
delta_P(W) = |E_P(W)| = 6-count_A(W).
```

If legal `c` is outside `join_live(P)`, then no old live attacker window
contains `c`; therefore every old `delta_P(W)` is unchanged.  Every
defender-free window through `c` had attacker count zero before the move and
has count one afterward, hence delta five.  This proves exactly:

```text
old live-window progress = 0
every born live window's delta = 5.
```

It does **not** rank the whole game state and does not imply dominance.

### 3.2 “Only new families” is false [REFUTED]

The harness defines a family as a connected component of windows under any
cell overlap.  A new count-one window can overlap an old live window at an
empty cell and merge into its existing component.

Frozen legal replay:

```text
[(0,0),(0,8),(1,7),(1,0),(2,0),(2,7),(3,7)]
```

Player 0 is at FirstStone.  The legal move `(5,3)` is in no old Player-0 live
window.  The old horizontal window `(0,0)..(5,0)` is live.  After `(5,3)`, a
new vertical count-one window `(5,0)..(5,5)` overlaps it at `(5,0)`.  The
machine check found 44 old live windows and 16 born delta-five windows and
asserted an overlap merge.

### 3.3 Strategy substitution fails [REFUTED]

A valid swap would have to transform one root action uniformly across every
Universal branch.  “Play the `C(P)` move used later” is not defined when
different defender branches use different later moves.  More importantly, the
completion-distance multiset omits three state-difference channels:

1. **Occupancy:** after removing remote `c`, the defender may occupy `c`.
2. **All windows through both cells:** `c` may block a defender count-four or
   count-five even though it advances no attacker window.
3. **Legality frontier:** a move can seed radius-8 legality for later moves.

The frozen required block realizes channel 2 exactly.  Any substitution into
`join_live`, `join_adj2`, or `join_adj1` loses immediately.  General branching
therefore does not preserve the strategy.

The T3/T4 defender-zone proof cannot be transplanted here: its anchor is a
forced reply and its coupling carries protected roles, local clocks, and
defender-completion exposure.  D17 permits a substitution only with a
transition-inclusive, all-reachable-descendant envelope and an independently
nonempty fallback.  A static attacker build delta supplies none of those
obligations.

### 3.4 The only elementary distance bounds

If `c` lies in a live length-six window containing `k` attacker stones before
placement, then

```text
dist(c, nearest attacker stone) <= 6-k.
```

Consequently `join_live` implies only distance at most five; extending count
two implies at most four; a placement that itself turns a count-three window
into a count-four threat implies at most three; extending count four implies at
most two; only a count-five completion forces adjacency one.  No elementary
geometry yields adjacency one for an arbitrary nonterminal quiet move.

## 4. Strongest proved restriction

### 4.1 Unconditional tier

For the report's all-position quiet-completeness question, the strongest tier
proved here is

```text
C_full(P) = Legal(P).
```

The three proposed strict tiers are refuted.  No claim is made that another
unstudied static locality tier is impossible; none is proved in this round.

### 4.2 Urgent SecondStone reply-survival kernel [PROVEN]

There is a useful exact restriction for one phase and one tactical condition.
Let `P` be nonterminal with attacker `A` to place its SecondStone and defender
`D=A.other()`.  Let

```text
T_D(P) = { active D window W : count_D(W)>=4 and count_A(W)=0 }.

Win1_A(P) = { c in Legal(P) : playing c immediately wins for A }.

BlockAll_D(P) =
  { c in Legal(P) : for every W in T_D(P), c is in E_P(W) }.

K_reply(P) = Win1_A(P) union BlockAll_D(P).
```

Use `BlockAll_D(P)=Legal(P)` when `T_D(P)` is empty.

**Theorem.** Every winning placement from `P` belongs to `K_reply(P)`.

**Proof.** Let `c` be winning.  If it wins immediately, it is in `Win1_A`.
Otherwise suppose `c` misses some `W in T_D(P)`.  Since `c` is not in `W`, the
active defender window is unchanged.  It has one or two empties.  The turn
passes to `D`, which owns two placements.  Those empties are legal because
they lie within distance at most five of a defender stone in `W`.  `D` fills
them and completes six before `A` moves again, contradicting that `c` is
winning.  Hence `c` hits every `W` and belongs to `BlockAll_D`. ∎

This theorem is player-symmetric, handles arbitrary later branching, and does
not depend on quietness or a semantic horizon.  At the frozen witness there is
no attacker win-now and all urgent defender windows share only `(6,-6)`, so

```text
K_reply(P) = {(6,-6)}             (538 -> 1).
```

This is the weakest sound repair established by round 1: at urgent
SecondStone nodes, keep the reply-survival kernel even when its member is
remote.  It does **not** prove `K_reply union join_adj1` complete at non-urgent
nodes; when `T_D(P)` is empty, the theorem deliberately falls back to full
legal.

## 5. Consumption-soundness clause

### 5.1 Positive certificates

Restricting attacker OR generation cannot create a false positive.  A `WIN`
may be consumed only after the independent verifier accepts an exact-root
certificate.  The verifier checks claimant ownership, legal replay, terminal
leaves, Universal coverage/zone evidence, and the absolute semantic horizon;
it need not know why the generator selected a move.

### 5.2 What restricted exhaustion means now

For any refuted tier `C`, clean enumeration can establish at most:

> no certificate was found through the searched `C` edges under the stated
> horizon and other generator restrictions.

It cannot establish “no unrestricted win.”  The frozen root is a direct false
negative: all 38 `join_adj1` moves lose, while omitted `(6,-6)` wins.

Production already enforces this epistemic boundary.  Exhausting a restricted
attacker set returns `None`/`UNKNOWN` (`tss_solver.rs:3573-3575`); resource
exhaustion is never interpreted as an opponent proof (`tss_solver.rs:12-13`).

### 5.3 What would be required to lift exhaustion

Even a future true locality theorem would lift an exhaustive miss only to “no
full-consume certified win by the same absolute horizon `T`,” unless the full
consume profile itself were proved game-complete.  A certificate-grade
negative artifact would have to verify all of the following:

1. every retained attacker option was exhausted or has a verified opponent
   win—no `UNKNOWN` child;
2. every defender response is represented or theorem-dismissed with valid
   zone evidence;
3. there was no node, depth, certificate-size, time, memory, or horizon cutoff;
4. terminal completions were retained;
5. Opening, FirstStone, and SecondStone were treated by the theorem actually
   proved for that phase;
6. claimant pair ordering/`PairContext` was rederived if quotienting ordered
   pairs; and
7. gate-true fallback moves that did not create a new count-four this turn
   were retained or covered by a separate theorem.

The current positive certificate format stores one attacker Choice edge and
cannot certify failed alternatives.  Therefore round-1 consumption is:

- **allowed:** verified positive wins under any restricted generator;
- **allowed:** `K_reply` pruning at SecondStone after recomputing its premises;
- **forbidden:** turning `join_live`, `join_adj2`, or `join_adj1` exhaustion
  into unrestricted absence; and
- **forbidden:** restricting the whole `quiet_turn_or_edges` fallback from the
  verbatim conjecture, because that fallback also runs at FirstStone and for
  non-quiet/non-new-threat cases.

For `K_reply`, a verifier can rederive `T_D(P)`, `Win1_A(P)`, and every omitted
cell's missed defender window.  It then checks the one- or two-placement
defender completion used in the proof above.  This is the exact local
consumption contract.

## 6. Machine checks and tallies

Two ignored tests were added to
`packages/hexfield_eq/rust/src/tss_quiet_locality_hunt.rs`:

- `quiet_locality_adversarial_family_geometry` freezes the overlap-family
  counterexample; and
- `quiet_locality_adversarial_required_remote` constructs a deterministic
  adversarial catalog, validates all root alternatives exactly, obtains the
  positive continuation certificate, prepends the remote Choice, and verifies
  the complete root WIN.

Final passing run:

```text
tests                                      2 passed, 0 failed
catalog cases declared                    5
cases reached before first witness        1
structural remote candidates checked      1
quiet remote candidates checked           1
solver UNKNOWN                            0
hard non-Loss continuation                0
frozen required-remote witnesses          1

witness legal completions                 538
witness losing alternatives               537
child search nodes                        4,957
child certificate nodes                   3,857
parent certificate nodes                  3,858

family old live windows                   44
family born delta-5 windows               16
family overlap merge                      true
```

The passing test body took 0.48 seconds; compile plus tests took 6.6 seconds.
Free RAM at setup was 12.95 GiB.  An earlier constructor run was intentionally
rejected by the exact elimination assertion because it left an immediate local
attacker completion; that failed candidate was removed before the passing run.
No Cargo processes were run concurrently.

## 7. Residue

The following statements remain **OPEN**:

1. A whole-turn FirstStone ordered-pair normal form.  It must constrain both
   placements, preserve pair legality/order, and quantify over all defender
   branches.  The singular-`c` conjecture is insufficient.
2. Any nontrivial static universe complete at non-urgent quiet SecondStone
   nodes.  The counterexample proves only that the three proposed tiers fail.
3. Completeness of `K_reply union join_adj1` or `urgent blocks union join_live`.
   The witness motivates these repairs but does not prove them.
4. A D17-style certificate-relative attacker substitution.  Such a theorem
   would need all-descendant roles, transition-inclusive legality/occupancy and
   opponent-completion guards, and a branch-independent root substitute.
5. A negative/exhaustion certificate format capable of establishing that all
   retained attacker alternatives fail without any resource gap.

## 8. Attack surface for hostile review

- **Phase/gate mismatch.** Quiet is an outgoing SecondStone property; no
  “unforced OR node” predicate gates full-legal enumeration.
- **Fallback scope.** Consume also covers FirstStone and gate-true completions
  that fail `turn_created_claimant_threat`; NQ2 covers neither.
- **Whole-turn ambiguity.** A reader who silently changes `P` to FirstStone is
  changing the target and still does not justify per-Choice pruning.
- **General defense.** Principal-variation reordering is not a strategy-tree
  transformation.  The positive witness uses a verifier-accepted Universal
  strategy.
- **Remote defensive tempo.** Completion distance ignores opponent windows;
  this is the actual counterexample mechanism.
- **Legality frontier.** Moving a stone changes radius-8 legality for both
  players; ordinary Maker monotonicity does not justify a swap.
- **Family definition.** Because overlap-connected windows through one tested
  cell intersect at that cell, `live_families_through_cell >= 2` and exact
  `in2fam_k0` are impossible by construction.  The old zero connector tally is
  tautological for that definition.
- **Cross-branch “served” metric.** `subtree_winning_windows` unions witnesses
  across Universal branches before building components.  `d_used` is not a
  single-line, uniformly promotable later move.
- **Raw branching ratio.** `node_full_legal` ignores PairContext filtering and
  duplicate ordinary/fallback work.  The reported 534→36 is a raw universe
  ratio, not exact production work.
- **“Quiet-required” wording.** A VCF non-WIN under a cap is `UNKNOWN`, not a
  proof that no pure-forcing win exists.
- **Verifier-admissible versus all legal play.** The theorem target must state
  whether it concerns arbitrary game strategies or the verifier's anchored
  claimant moves.  The frozen remote move is verifier-admissible and the full
  root certificate was accepted, so this distinction does not weaken the
  counterexample.
- **Opening/no attacker stone.** The report does not assume an existing
  attacker stone.  At the opponent's first normal turn, adjacency can be empty;
  any repaired theorem needs an explicit scope.
- **Prompt/report displacement mismatch.** The task prompt names `d_win=6`.
  The checked report and JSON have no `d_win` field; the report's named outlier
  is `d_used=4` for `double_fork_ordered`.  Round 1 does not silently equate
  them.  Independently, the new frozen counterexample has `d_stone=6`.
- **Caps and horizons.** A verified positive certificate is sound; a miss at
  200,000 nodes or horizon 66 would not have refuted or proved locality.

## 9. Files and regeneration

Changed files:

- `PROOF_QUIET_LOCALITY.md` — this standalone verdict and proof record.
- `packages/hexfield_eq/rust/src/tss_quiet_locality_hunt.rs` — corrected the
  stale module-level quiet definition, added `join_adj2`, and added two ignored
  adversarial tests.

Formatting and final machine run:

```powershell
rustfmt --edition 2021 packages/hexfield_eq/rust/src/tss_quiet_locality_hunt.rs

do {
  $free = Get-CimInstance Win32_OperatingSystem |
    ForEach-Object { $_.FreePhysicalMemory / 1MB }
  if ($free -lt 9) { Start-Sleep -Seconds 30 }
} while ($free -lt 9)

$env:CARGO_TARGET_DIR='.target-hunt'
$env:QL_ADV_CAP='200000'
$env:QL_ADV_HORIZON_SLACK='30'
$env:QL_TT_BYTES='268435456'
cargo test --release -p hexfield_eq quiet_locality_adversarial -- --ignored --test-threads=1 --nocapture
```

The pre-existing empirical aggregation can be regenerated independently with:

```powershell
python merge_specimens.py QUIET_LOCALITY_SPECIMENS.jsonl QL_SPECIMENS.jsonl QL_HUMAN.jsonl QL_LEAFWIDTH.jsonl
python aggregate_quiet_locality.py QUIET_LOCALITY_SPECIMENS.jsonl
```

Those empirical files retain their historical value, but their 7/7 coverage
does not survive as a universal theorem after this frozen counterexample.
