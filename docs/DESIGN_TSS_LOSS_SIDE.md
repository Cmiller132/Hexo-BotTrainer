# The TSS loss side: claimant semantics, proof obligations, and budget policy

Status: implemented design, frozen prediction, and quick-tier measurement
record, 2026-07-20. This work changes budget scheduling only. It does not
change the verifier or any proof rule.

## 1. Scope and notation

Fix an exact, reachable, nonterminal engine state `s`.  Write

- `M = s.current_player()` for the player who must place next at the root;
- `C` for the identity whose win the certificate claims; and
- `Win_C(s)` for the game-theoretic proposition that `C` has a winning
  strategy from `s`, with the engine's current mover and phase left unchanged.

The root status is a perspective mapping, not a different proof system:

```text
status(s, C) = Win    if C = M
             = Loss   if C = other(M).
```

Therefore the precise loss claim is

```text
Loss(s)  :=  Win_other(M)(s).
```

The opponent becomes the fixed claimant, but the state is not seat-swapped and
the opponent is not artificially put on move.  The original mover `M` still
places next.  A dual certificate consequently begins at a universal node in
the ordinary case: it must defeat every legal root move by `M`.  A root
adaptive lambda-one `CertNode::Loss` is the special case in which that whole
obligation is discharged by a checked threat-family argument.

Player identity, rather than placement-depth parity, selects existential and
universal nodes:

```text
current_player(s) = C       => existential claimant node
current_player(s) != C      => universal defender node.
```

This distinction is necessary in Hexo because `FirstStone -> SecondStone`
does not change the mover, while the transition after `SecondStone` does.
`TssSolver::solve_goal` starts the loss attempt with
`prove_for(state, root_player.other(), ...)`; `status_for_claimant` maps the
result back to root perspective.  Independently, `verify_certificate` requires
`cert.claimant == state.current_player().other()` for a claimed `Loss`
(`packages/hexfield_eq/rust/src/tss_verify.rs:179-227`).

### 1.1 Loss-soundness theorem

**Theorem.** If

```text
TssVerifier.verify(s, cert, ProofStatus::Loss) = true,
```

then `Win_other(M)(s)` holds, so the root mover `M` is lost.

**Reason.** The verifier first binds `cert` to the exact root and fixes the
claimant to `other(M)`.  It then checks an acyclic claimant-winning strategy:
one legal continuation is sufficient at every claimant node, whereas every
opponent continuation is either represented by a recursively winning child or
dismissed by a verifier-rederived theorem obligation.  The accepted leaves are
engine completions or independently checked lambda-one facts for that same
claimant.  Induction over the accepted acyclic arena therefore proves a
winning strategy for `other(M)` from the unchanged state `s`.

The converse is deliberately not claimed: the TSS move language is incomplete,
and a cap can stop it.  In particular,

```text
failure to prove Win_M(s)       does not imply Loss(s);
failure to prove Win_other(M)(s) does not imply Win(s).
```

Either failure is `Unknown`.  A solver-produced status also cannot directly
enter the tree as a hard value.  `hard_value_from_verified` is the sole deep
mint and returns a value only when the concrete `TssVerifier` accepts the
certificate (`tss_core.rs:524-558`); `tss_solve_verified` routes every hard
result through that mint (`tree.rs:860-884`).

## 2. Width restrictions: the exact monotonicity direction

Let `G` be the full game from `s`.  At every claimant position `p`, let
`A_C(p)` be a subset of the claimant's legal moves, and form `G|A_C` by
restricting only the claimant to those moves while leaving all defender moves
available.  Then

```text
C wins G|A_C  =>  C wins G.
```

The same restricted strategy is legal in `G`; the claimant simply continues
to select the certified move.  Restricting an existential choice makes a
claimant-WIN proposition harder to prove and is therefore a pure
strengthening.  This argument is independent of player colour.  It applies
unchanged when `C` is the original opponent and the resulting root status is
`Loss`.

### 2.1 Restrictions that remain sound for either claimant

The following current restrictions all select a subset of claimant strategies
and therefore affect completeness, not proof soundness:

1. The historical narrow generator admits claimant extensions of active
   windows with at least three claimant stones.
2. The pair-complete/wide generator admits count-two extensions in addition to
   the narrow tier and retains urgent defender-window blocks.  It still omits
   arbitrary quiet moves.
3. Wide VCF requires a completed claimant pair to create a new count-four (or
   stronger) claimant window and to leave a small, theorem-supported forced
   reply.  Rejecting a looser pair deletes claimant strategies only.
4. The certificate grammar's attacker-placement well-formedness condition
   restricts relied-upon claimant placements to the radius-eight locality
   needed by the zone coupling.
5. Ordering, node limits, and depth limits may stop discovery. They are safe
   only because an interrupted or exhausted attempt returns `Unknown` and
   emits no hard claim. A bounded loss-first probe would have the same
   soundness property, but section 5 rejects it as a budget policy.

The current production-wide loss attempt is consequently best described as a
restricted opponent-VCF WIN search.  A positive result is enough to establish
root `Loss`; a negative or exhausted result is not a game value.  The solver
states the latter rule directly after claimant candidate exhaustion in
`prove_choice`: the restricted generator found no proof, not a disproof.

### 2.2 Restrictions that are not justified by width monotonicity

Restricting a universal node removes choices from the player opposing `C`.
That makes the claimant's task easier and does **not** transfer to the full
game.  For a root loss attempt this opposing player is the original root mover,
so silently narrowing its replies would be exactly the unsound shortcut that
a loss proof must avoid.

An accepted `Universal` may omit replies only under one of the following
separate, checked rules:

- **Full enumeration:** the represented set equals the engine's complete legal
  move set.
- **Instant-dispatch kernel:** the position is post-opening, the defender has
  no own win now, and the claimant threat family's minimum hitting size equals
  the defender's remaining turn budget.  The verifier independently rebuilds
  the extendable-hit kernel and requires it to be represented.
- **Certificate-relative zone:** the verifier independently reconstructs the
  local defender clock, protected obligations, and mandatory zone, then
  requires every zone cell to be explicit.
- **Same-turn commutation:** an omitted second placement is supported by the
  checked mirror ordering, child identities, legality, and equal pair outcome.

These are proof theorems, not "defender width."  Outside them, a partial
universal frontier cannot certify either `Win` or `Loss`.  Similarly, exhausting
a restricted claimant search can never be recycled as evidence for the other
claimant.  The primal and dual attempts must construct independent positive
proofs.

## 3. The immutable verifier contract for `Loss`

There is no weaker loss verifier.  `ProofStatus::Loss` changes only the expected
claimant identity; the arena is then replayed by the same claimant-WIN checker.
The following obligations are already enforced by
`packages/hexfield_eq/rust/src/tss_verify.rs`, which is outside the scope of
this change.

| Layer | Obligation checked by the verifier | Source |
|---|---|---|
| Root and perspective | Reject `Unknown`; require the canonical full `RootBinding` to equal the supplied state; require claimant `other(current_player)` for root `Loss`. | `verify_certificate`, lines 179-227 |
| Arena bounds | Bound root stones, nodes, explicit edges, witness identities, commutations, replay depth, and memo bytes; reject empty arenas and bad IDs. | constants at lines 17-34; `validate_arena`, lines 1364-1450 |
| Graph structure | Reject duplicate explicit universal moves, cycles even in a disconnected component, and every orphan node; shared nodes may be reused only at an identical full replay state and commutation context. | lines 391-459 and 1364-1499 |
| Time | Derive `T` as the maximum exact leaf completion/resolution; require `T <= semantic_horizon` and `T` no later than every zone build horizon. | `certificate_metadata`, lines 237-319; checks at lines 200-209 |
| Claimant choice | Require claimant to move, nonterminal state, attacker-placement WF, a legal engine transition, no hidden terminal child, and a recursively valid claimant proof. | `verify_node`, lines 463-552; `attacker_placement_wf`, lines 601-615 |
| Immediate completion | Require claimant to move; the named witness contains the move; the ply is exactly `current_ply + 1`; engine application terminates with claimant; the named window is uncontested count six. | `verify_or_completion`, lines 617-637 |
| Claimant lambda-one WIN leaf | Require claimant to move and nonterminal state; recompute remaining budget; require an uncontested count five, or count four only at budget two; require exact `+1`/`+2` resolution and WF witness empties. | `verify_win_leaf`, lines 639-669 |
| Adaptive lambda-one LOSS leaf | Require the defender, not claimant, to move; nonterminal state and a nonempty named family; reject defender `own_win_now`; require every named window active for claimant with claimant count at least four; require WF empties; prove no set of at most `b` cells hits the family; require exact resolution `current_ply + b + 2`. | `family_hitting_exceeds` and `verify_loss_leaf`, lines 671-739 |
| Universal node | Require defender to move, nonterminal state, and no defender own win now; reject duplicate/illegal/terminal explicit edges; replay every explicit child as a claimant win. | `verify_universal`, lines 827-963 |
| Universal coverage | Require full legal equality, independently rebuilt instant-dispatch kernel coverage, or an independently rebuilt mandatory zone.  Validate any commutations separately. | lines 741-825, 875-911, 1216-1330 |

### 3.1 The adaptive loss leaf in equations

At a nonterminal leaf `p`, let `D = p.current_player()`, where `D != C`; let
`b` be `D`'s remaining placements this turn; and let the named claimant-threat
family be

```text
F = { E(W, p) : W is a named active C-window with count_C(W) >= 4 },
```

where `E(W,p)` is the set of empty cells of `W`.  The verifier establishes

```text
not own_win_now_D(p)    and    tau(F) > b,
```

with an exhaustive singleton/pair hitting test for Hexo's only budgets,
`b in {1,2}`.  Every complete defender remainder `H`, with `|H| <= b`, therefore
misses at least one `E(W,p)`.  That claimant window remains alive with at most
two empties, so `C` completes it on the following turn.  The checked worst-case
resolution is

```text
T_leaf = placements_made(p) + b + 2.
```

This node is named `Loss` because the *local mover* is lambda-one lost; inside
a root `ProofStatus::Loss` certificate it is still positive evidence for the
fixed opponent claimant. The strict verifier checks every member of the
nonempty family named by the certificate and proves `tau(F) > b` for that
family. It neither requires the certificate to name every active threat nor
trusts or validates the solver's sparse-family selection/minimality heuristic.
It enforces the global witness-resource bound independently.
Mutation tests explicitly reject the cited fixture's truncated family, a wrong
resolution, and a too-short external horizon (`tss_verify.rs:2156-2257`).

### 3.2 Trust boundary

The verifier is independent of solver search and candidate generation, but two
shared theorem dependencies should be stated rather than hidden:

1. adaptive `Loss` leaves and dispatch/zone premises use the shared
   `threats_shared::analyze` lambda-one primitive (`OrCompletion` and `Win`
   leaves are checked directly); and
2. zone-omitted replies are theorem-dismissed after independently reconstructing
   the zone obligations, rather than replayed one by one.

For `implicit_dispatch`, production verification independently derives the
boundary and kernel.  The per-omitted-move lambda-one replay is a test-only
debug oracle (`verify_universal`, lines 938-961), not an additional production
obligation.  Soundness therefore rests on the proved kernel theorem plus the
checked premises, as documented in `PROOF_TSS_DEFENDER_ZONES.md` T6.

## 4. Zone and deadline machinery on the loss side

The zone theory names the claimant "attacker A" and the other player "defender
D".  Those are roles, not fixed colours.  For a root loss search,

```text
A = C = other(root mover)
D = root mover.
```

Thus the same code and theorem apply without reversal.  It is the original
opponent's proposed completion that supplies each leaf deadline, and the local
budget counts placements available to the original mover before that opponent
completion.

### 4.1 Exact certificate-relative local budget

For a node `N` in a claimant-winning certificate, the implemented D14 clock is

```text
B(OrCompletion) = 0
B(Win)          = 0
B(Loss at defender budget b) = b
B(Choice(child))             = B(child)
B(Universal(children))       = 1 + max B(child).
```

The unit is a defender placement, not a turn, search node, or wall-clock unit.
The two placements of one defender turn therefore contribute two universal
edges.  The adaptive loss leaf contributes the unexpanded remainder `b`.
`verifier_zone_summary` independently replays the certificate and recomputes
this recurrence (`tss_verify.rs:1024-1087`).  A stored `ZoneInfo.d` is evidence
only and must equal the derived local budget; `ZoneInfo.build_horizon` must
equal the certificate horizon (`verify_zone_node`, lines 1216-1275).

The certificate's actual semantic deadline is also claimant-relative:

```text
T(cert) = max exact claimant completion/resolution over all leaves.
```

This is why opponent completion deadlines work on the loss side without new
theory.  They bound how many root-mover placements can interfere with the
opponent claimant's named future moves and windows.

### 4.2 Mandatory uniform zone rederived by the verifier

Let `Prot(N)` be the union of reachable descendant claimant placements and
WIN/LOSS witness-empty roles.  Let `L(N)` be the engine legal set, and let

```text
Pending(N) = Prot(N) minus (L(N) union Stones(N)).
```

With the conservative uniform role rank `B = B(N)`, the current verifier builds

```text
Z_dir   = Prot(N) intersect L(N)

Z_seed  = { x in L(N) : distance(x,y) <= 8 max(B-1, 0)
                         for some y in Pending(N) }

Z_touch = union of empty cells of active D-windows W satisfying
          count_D(W) >= 1 and count_D(W) + B >= 6

Z_virgin = L(N) if B >= 6, otherwise empty

Z(N) = Z_dir union Z_seed union Z_touch union Z_virgin.
```

If that union is empty, one deterministic legal fallback is still required.
Current hitting cells are an ordering heuristic, not a term of this ordinary
zone.  `verify_zone_node` accepts only an independently nonempty explicit set
covering every cell of `Z(N)`, checks every explicit move is legal, and rejects
Opening, claimant-to-move, defender-own-win-now, and a known
`min_hitting_set >= b` position.  The distinct forced boundary
`min_hitting_set == b` uses the instant-dispatch kernel.

The proof obligations and symmetry are developed in
`docs/PROOF_TSS_DEFENDER_ZONES.md`: D9/D14-D16 at lines 188-367, dismissal
soundness T3/T4 at lines 471-571, and the forced kernel T6 at lines 645-680.
Nothing in those statements depends on which engine player is the claimant.

### 4.3 Where the machinery is inert

Zone soundness does not imply zone usefulness.  There are four important inert
regimes.

1. **Quiet claimant nodes.** A zone reduces branching only at universal nodes
   of an already selected claimant attack.  It cannot invent a quiet claimant
   move at an existential node.  If the restricted opponent-VCF search has no
   threat-building continuation, the loss attempt remains `Unknown` before a
   zone can help.
2. **Spare/quiet defender nodes.** In the forced-kernel theorem,
   `min_hitting_set < b` retains all legal defender moves; only equality can
   prune.  Ordinary zones may still be sound, but weak deadlines make them
   large.
3. **Slack local clocks.** The current conservative implementation sets
   `Z_virgin = Legal` for `B >= 6`.  Coverage is then full legal and pruning is
   exactly zero.  The finder likewise returns the full legal candidate set for
   `d >= 6`.
4. **Unbounded/slack semantic horizons.** The current finder supports only the
   short useful budget band.  If counting defender placements to the supplied
   horizon exceeds that band, `remaining_defender_placements_for_horizon`
   declines to construct a zone and the solver falls back to full legal search.

The V1 evidence matches these mechanisms.  The comment at `tree.rs:736-748`
records `zone_nodes = 0` across roughly 33,000 solves under slack budgets and
explains the later tight `+8` diagnostic pass.  The raw summaries report zero
zone-carrying positions and zero zone nodes for the flat, ladder, and unbounded
zone arms on all recorded groups:

- self-play, `n=3,255`: `raws/summary_selfplay.json:635-681`;
- human, `n=320`: `raws/summary_human.json:579-624`;
- forcing, `n=19`: `raws/summary_forcing.json:493-538`; and
- spare-tempo, `n=2`: `raws/summary_spare.json:385-430`.

The zone-on wall deltas in those summaries conflate the tight-horizon wrapper
with actual zone use; with `zone_nodes=0`, they are not evidence of pruning.
V1 also found that 99.7% of cap-bound grind wall had at most one opponent
threat, with median zero (`docs/SOLVER_NOTES.md:39-48`).  Consequently the
existing zone theory transfers fully and symmetrically to an opponent-claimant
loss proof, but it offers no measured leverage on the present quiet, slack
frontier.  Budget scheduling is the appropriate first experiment.

## 5. Fixed-cap budget policies

Let the aggregate per-position node cap be `N=500`.  The solver performs one
shared root fact check before deeper attempts.  Let `r` be that charged root
cost, `w` the actual primal claimant-WIN work, and `l` the actual opponent-WIN
work.  Every policy must maintain

```text
r + w + l <= N.
```

Attempt caps are discovery policy only.  They do not alter certificate
semantics or verifier acceptance.

### 5.1 Current policy: leftover-only dual pass

The wide `Both` primal receives the whole post-root budget.  If it returns a
proof, the solve is done.  If it returns `Unknown` early and
`tss_solver_dual_pass=true`, the dual cap is

```text
q_dual = N - (r + w).
```

This policy preserves the primal's maximum discovery budget.  It is excellent
when the primal width-exhausts cheaply: all unused work can be transferred to
the loss attempt.  It is blind when the primal is cap-bound, because then
`q_dual=0`.  The existing cap-500 standard run nevertheless added 288 verified
losses with wins unchanged and never exceeded the cap
(`docs/SOLVER_NOTES.md:208-216`).

### 5.2 Bounded loss probe first

With configured allowance `q`, first attempt the opponent claimant with cap
`q`. If it proves root `Loss`, that verified positive proof is sufficient and
the expensive primal can be skipped. Otherwise debit the probe's actual work
`p <= q` and give the primal the remaining aggregate budget:

```text
q_primal = N - (r + p).
```

The V1 median dedicated probe was only 16--22 microseconds, so this looked
attractive before paired accounting. It is nevertheless the wrong policy for
the current implementation. A capped `WidePnSearch` is not resumable: the
post-primal dual pass constructs a fresh search. A failed first probe followed
by that pass therefore duplicates opponent work rather than donating reusable
progress. Recorded proof costs predict puzzle-dev loss coverage
`92 -> 88/87/86` for `q=32/48/64`, and quick puzzle coverage `10 -> 8` for
every one of those bounds. Prior full-budget loss runs on the exact cap-bound
quick cohorts found no new losses, so the predicted incremental quick yield is
zero.

A valid first-probe proof would still be sound in this deterministic
perfect-information game under Hexo's turn scheduler; placements need not
strictly alternate. The rejection is economic and coverage-based, not a proof
objection. No `loss_probe_nodes` option was shipped.

### 5.3 Reserved loss floor after the primal

For configured reserve `rho`, let `R=N-r` be the post-root allowance. The
implementation clamps the effective reserve to

```text
rho' = min(rho, R.saturating_sub(1)).
```

Thus any nonempty post-root allowance always leaves at least one node for the
primal; no configuration can silently turn `Both` into a loss-only solve. The
initial split is

```text
q_primal = R - rho'
q_dual   = rho'.
```

If the primal proves `Win`, the solver returns immediately. If it is
undecided, a positive effective reserve `rho' > 0` schedules the opponent-WIN
attempt even when `dual_pass=false`. With `dual_pass=true`, the existing
leftover block is left unchanged and upgrades the dual cap to

```text
q_dual = N - (r + w) = rho' + unused(q_primal).
```

This reaches the opponent claim after a reduced primal cap-binds without
duplicating a discarded loss search. It can still hide a near-cap win, so it
must be judged on paired win coverage. Every accepted result remains a
positive claimant proof; exhaustion remains `Unknown`. Since `r+w+l <= N`,
aggregate node accounting is unchanged.

### 5.4 Pre-implementation Lane C prediction

These predictions were frozen before solver-policy code was edited. They use
the observed standalone `loss_pass.deep_nodes` in
`raws/lanec_labels.jsonl`, not a claim of mathematical minimum proof size, and
assume the same width, ordering, and cache regime. A Lane pass's node count
includes its root fact check, whereas `loss_reserve_nodes` is a post-root
attempt allowance. None of the selected thresholds has a proof on that
one-node boundary, so the tabulated classifications are unaffected.

Lane C has 889 rows: 271 `Win`, 192 `Loss`, and 426 `Unknown`. The frozen
puzzle dev split has 152 certified losses: 116 atlas, 34 human, and 2 forcing.
Gross dedicated-loss recall is:

| Standalone loss cap | All Lane C losses | Puzzle-dev losses | Puzzle-dev composition |
|---:|---:|---:|---|
| 32 | 24 / 192 | 20 / 152 | 1 atlas, 17 human, 2 forcing |
| 48 | 29 / 192 | 24 / 152 | 1 atlas, 21 human, 2 forcing |
| 64 | 29 / 192 | 24 / 152 | 1 atlas, 21 human, 2 forcing |
| 500 | 110 / 192 | 92 / 152 | 58 atlas, 32 human, 2 forcing |

Of the 426 Lane C `Unknown` rows, 402 dedicated loss attempts terminate in
two nodes; p50 and p90 are both two. This agrees with V1's 16--22 microsecond
median loss probe (`docs/SOLVER_NOTES.md:45-49`). More importantly, independent
cap-500 loss passes on the V1 cap-bound paired cohort yielded 0 / 87 losses;
the matching current quick cohorts also yielded human 0 / 22, selfplay 0 / 26,
and puzzle 0. The cheap typical cost therefore does not imply useful yield on
the cap-bound class that scheduling is meant to reach.

The current dual archive proves 58 of the 116 atlas losses and misses 58. For
the missed tail, the recorded dedicated-loss costs are:

```text
minimum = 512 nodes
median  = 1,064 nodes
p90     = 2,856.3 nodes (Hyndman--Fan type 7 / inclusive interpolation)
maximum = 19,536 nodes.
```

The number of these 58 reachable by a standalone loss cap is 27 at 1,000, 48
at 2,000, 56 at 4,000, 57 at 5,000, and 58 at 20,000. Therefore no
redistribution of the same 500 nodes can recover any of them under the
unchanged search algorithm. That is an empirical conclusion from deterministic
recorded runs; resumable proof reuse, a better move/search algorithm, or a
larger cap would be a different experiment.

The frozen policy predictions are:

| Policy at cap 500 | Frozen coverage prediction | Frozen cost / win-risk prediction |
|---|---|---|
| Leftover-only, current | Keep puzzle losses at 92 / 152, including 58 / 116 atlas; no dual work after a cap-bound primal. | Preserve the full primal allowance; pay loss work only after early primal exit. |
| Loss-first, `q=32/48/64`, then fresh leftover dual | Puzzle losses regress from 92 to 88 / 87 / 86; quick puzzle regresses from 10 to 8; zero recovery in the missed atlas tail. | Usually cheap, but failed opponent work is discarded and repeated. Near-cap primal wins are also exposed to the debit. Rejected before implementation. |
| Reserved floor, `rho=32/48/64`, plus leftover dual | Preserve all 412 current standard-dev loss records: incidental losses cost one root node, and every dual-added loss follows a two-node primal. Zero incremental quick loss is predicted. | Zero quick win-risk candidates. Standard records with current win cost above the reduced primal cap number 11 / 13 / 16 respectively. No atlas-tail recovery. |

The reserve prediction selected `rho=32`: it is the smallest requested policy
band, has no known quick win risk, and avoids the provable coverage regression
of a fresh first probe. It was not predicted to satisfy the desired loss-gain
criterion; the experiment tests whether unmodeled ordering/cache effects
falsify the zero-yield evidence.

## 6. Predictions versus harness measurements

Two arms using the same native-release build ran through the real quick-tier
harness: the current dual-pass anchor (`rho=0`) and the selected reserve
(`rho=32`). The host lacked a WSL distribution and the hard-coded Linux/GPU
benchmark venv, so both archives used `--no-bench`. All harness gates passed,
and the separate paired-comparison reports contain no changes or
contradictions. Wall time is therefore diagnostic only; node counts are the
comparable cost measure.

| Frozen quick set | Positions | Anchor W / L | Reserve-32 W / L | Paired verdict changes | Nodes, anchor -> R32 | Maximum / over-cap | Verify failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| `human_v1` | 338 | 48 / 42 | 48 / 42 | 0 upgrades, 0 downgrades | 19,068 -> 18,572 (-496) | 500 / 0 | 0 |
| `puzzle_v3` | 48 | 11 / 10 | 11 / 10 | 0 upgrades, 0 downgrades | 17,412 -> 16,916 (-496) | 500 / 0 | 0 |
| `selfplay_v1` | 343 | 18 / 7 | 18 / 7 | 0 upgrades, 0 downgrades | 19,844 -> 19,110 (-734) | 500 / 0 | 0 |
| **Total** | **729** | **77 / 59** | **77 / 59** | **0 / 0** | **56,324 -> 54,598 (-1,726; -3.1%)** | **500 / 0** | **0** |

The result matches the frozen prediction exactly: no new verified loss and no
lost win. Fifty-five records used 31 fewer nodes and one used 21 fewer. Those
are quiet opponent attempts that width-exhaust before spending their reserved
allowance after the primal is clipped; unused budget is not itself a success.
All harness gates passed, and the manifest echoed
`loss_reserve_nodes=0` / `32` from the same resolver used by the solve.

Archives:

- anchor:
  `scripts/tss_harness/harness_runs/20260720_234546_loss_reserve0_gated`;
- reserve 32:
  `scripts/tss_harness/harness_runs/20260720_234604_loss_reserve32_gated`.

The promotion gate required a verified-loss increase with positionwise win
parity. Reserve 32 fails the first condition. A standard run is also predicted
to put 11 currently verified win records beyond the reduced primal allowance.
It should not be promoted. Reserve 48/64 have no plausible loss upside at cap
500 and strictly larger predicted win risk, so they were not run.

## 7. Implementation and decision record

The shipped experimental option is the default-zero production divergence
`tss_solver_loss_reserve_nodes` and harness key `loss_reserve_nodes`. It is
propagated through `SelfplayConfig`, both divergence maps, strict Rust override
parsing, root/inline/async solver construction, the persistent batch API, and
the benchmark overlay/scorecard echo. The batch manifest obtains the value
from the shared `effective_solve_config` resolver. Existing harness gates are
unchanged; the archived arms explicitly declare `dual_pass` and
`loss_reserve_nodes`, so the existing manifest-subset gate checks both echoes.

The scheduling effect is confined to pair-complete/wide `SolveGoal::Both`.
`Win`, `Loss`, and narrow `Both` calls are bit-identical when it is nonzero.
Zero is bit-identical in wide `Both` with `dual_pass` both off and on, including
status, certificate arena, and stats. Changing the option clears positive
fragment caches so A/B provenance cannot leak. A positive reserve alone runs
its fixed opponent floor; enabling `dual_pass` additionally donates every
actual primal leftover. The pure initial-split helper clamps pathological
values so a nonempty primal allowance is never skipped.

No result bypasses `tss_solve_verified`. No negative inference was added,
universal coverage was not narrowed, and
`packages/hexfield_eq/rust/src/tss_verify.rs` has zero diff. Unit tests cover
zero identity, standalone reserve scheduling, an independently verified loss,
combined cap accounting, the no-skip clamp, inert scopes, strict config
parsing, and shared-resolver manifest echo.

**Decision:** keep the option experimental/default-zero, but do not promote
reserve 32 to standard and do not run the larger reserve arms. It saves some
failed-search nodes but does not deepen loss coverage. For the 58 unreachable
atlas losses, unchanged-search cap requirements predict 0 recovery at 500, 27
at 1,000, 48 at 2,000, 56 at 4,000, 57 at 5,000, and all 58 only by 20,000.
The next credible fixed-cap direction is search efficiency or resumable proof
reuse, not another ordering of the same nonresumable primal/dual budgets.
