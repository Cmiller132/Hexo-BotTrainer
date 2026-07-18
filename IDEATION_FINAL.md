# Final solver ideation gate

Date: 2026-07-17

Worktree: `tss-vcf-width`

Audited HEAD: `28eb5ac816aae96c13a67713c26b84a3d1daddef`

## Verdict

**LEVER EXHAUSTION FAILS. Three genuinely unposed, realistic levers remain.**

| Rank | Lever | Profile it can move | Prospective size | Confidence |
|---:|---|---|---|---|
| 1 | Resume the same proof-number frontier across increasing node-cap rungs | 1 GiB lazy official/deep campaigns; later, repeated exact trainer queries | 30.67% of the measured official solve wall is repeated lower-rung work | High |
| 2 | Prior-scale-aware df-pn threshold increments | Deep official; possibly Phase-3 leaf | Current priors span 1..37 while thresholds still advance by 1; same-game literature makes a >=5% A/B plausible, but local sizing counters are missing | Medium |
| 3 | Opening-root stabilizer orbit pruning | Opening atlas A-0 | Nontrivial stabilizers cover 62.92% of games; corpus-weighted root-child-removal ceiling 32.39%, with end-to-end share still unmeasured | Medium-low |

This count is deliberately narrow. Two other realistic experiments are **not
new ideas** and therefore are not re-ranked here:

1. lazy/incremental child ordering (including revealing only children that
   are searched) was already posed in `docs/TSS_SOLVER_OPT_SPEC.md` and the
   register's RZOP item 8 remains `UNCLEAR -- re-profile`; and
2. the `live_ge3` PN seed was already posed in `HUNT_REPORT_PN_INIT.md`, whose
   explicit next step is a live test-only A/B.

The main register records the census-gate half of NQ6 but never disposes the
`live_ge3` recommendation. These are **existing closure debts**, not discoveries
from this round. They independently prevent an honest claim that every
realistic lever has been dispositioned, but counting them as newly posed would
violate this round's rule.

No production code was changed. No Cargo build or solve was needed: the merged
tip already contains the exact complete 1 GiB transcript and the requested
wide-generator timers. A Windows system CPU trace was attempted, but host
policy denied the WPR CPU profile before any test started; no trace or partial
run was produced.

## Audit basis

I read the complete `docs/PLAN_TSS_SOLVER_UPGRADES.md` first, followed by
`MERGE_RESOLUTION.md` and every local hunt/build report it cites. I also read
the relevant read-only sibling reports for certificate support, corpus
frequency, R1b/R2, domination, and leaf width, plus the paper's complete
`RZOP_SOLVER_OPTIMIZATION.md` and `RZOP_COMPARISON.md`. The source audit covered
the wide df-pn driver, pair gate, defender-pair plan, TT/frontier lifetime,
certificate materializer/verifier seam, official corpus harness, leaf surface,
MCTS memo, and async worker pool.

The literature sweep used primary or author-hosted sources:

- Kishimoto et al., [Game-Tree Search Using Proof Numbers: The First Twenty
  Years](https://webdocs.cs.ualberta.ca/~mmueller/ps/ICGA2012PNS.pdf);
- Pawlewicz and Lew, [Improving Depth-First Proof-Number Search: 1 + epsilon
  Trick](https://www.mimuw.edu.pl/~lew/files/epsilon_trick.pdf);
- Zhang, Iida, and van den Herik, [Deep df-pn and its application to
  Connect6](https://dspace.jaist.ac.jp/dspace/bitstream/10119/15854/1/23404.pdf);
- Gao, Mueller, and Hayward, [Focused Depth-first Proof Number Search using
  CNNs for Hex](https://webdocs.cs.ualberta.ca/~hayward/papers/fdfpnscnnhex.pdf);
- Gao, [On Computation Complexity of True Proof Number
  Search](https://arxiv.org/abs/2102.04907); and
- Winands, Uiterwijk, and van den Herik,
  [PDS-PN](https://dke.maastrichtuniversity.nl/m.winands/documents/PDSPNCG2002.pdf).

“Realistic” below means a plausible >=5% end-to-end gain on the official deep
or Phase-3 leaf profile, or an opening-atlas/soundness capability the owner
explicitly values. A component ceiling by itself is not counted as a lever.

## 1. Telemetry residue

### 1.1 Exact 1 GiB profile

The evidence is the complete merged-tip run in
`.codex-group2/codex-consolidate.log`: 34 solves, 495.592 s summed solve wall,
and 495.940 s test wall. It used the recommended 1 GiB TT and lazy frontier.
The transcript has the interior gate off. That does not hide gate work on this
unbounded-horizon query: `evaluate_interior_census_gate` exits before scanning
unless `h_rem` is in 0..=8, so the official `u32::MAX` horizon produces zero
evaluations. Prior gate-on coverage also records zero evaluations on this
forcing corpus. Thus gate-on has only an inert branch check here; the exact
gate-on regeneration command is included below.

All timer shares use the 495.940 s test wall. Timers marked nested must not be
added to their parent.

| Component | Time | Share | What the number means |
|---|---:|---:|---|
| Attacker pair generation | 216.228 s | 43.600% | Nested in expansion |
| Defender-pair enumeration | 175.837 s | 35.455% | Nested in expansion; disjoint from attacker-pair nodes |
| Pair + defender | 392.065 s | 79.055% | The two mutually exclusive generation paths |
| Expansion, inclusive | 397.325 s | 80.116% | Contains both rows above |
| Expansion excluding pair/defender | 5.260 s | 1.061% | Terminal/gate/bookkeeping, simple defender paths, top-level analysis |
| Prior/second-candidate “regen” | 43.036 s | 8.678% | Nested, primarily in pair generation |
| Full bottom-up stage refresh | 23.557 s | 4.750% | Separate stage pass |
| Direct `insert_position` | 3.283 s | 0.662% | Does not include every key-construction cost |
| Solve wall outside expansion and full refresh | 74.710 s | 15.064% | Descent/scheduling, TT reads, state traversal, deferred realization, materialization/rebase, root setup |
| Verifier plus all harness overhead | <=0.348 s | <=0.070% | Conservative upper bound: test wall minus summed solve timers |

Threat analysis cannot be given an honest standalone share. Much of it is
nested inside `WideTurnGate::evaluate_pair` and
`forced_defender_pair_plan`. Likewise, the 0.662% insertion timer excludes key
construction charged to generation. A fabricated split would be worse than an
unresolved bucket.

TT traffic is not the remaining wall: the run has 102,267 hits over 4,507,328
expansions (2.27%), direct insertion is 0.662%, and the hardest row peaks at
about 549 MiB under the 1 GiB cap. The verifier is at most 0.070%. Zone builds
are zero on this forcing/unbounded profile. On the selected Phase-3 D profile
at cap 500/horizon 8, the census gate costs 0.571 ms of 76.513 ms total
(0.75%) while dismissing all 692 evaluated nodes; there is no new zone-build
hotspot there either.

### 1.2 The hidden cross-call tax

The 34 official solves include 15 non-final lower rungs whose work is thrown
away before the same root is retried. Exact transcript arithmetic:

| Quantity | Fresh official ladder | Final attempt per root | Repeated lower-rung work |
|---|---:|---:|---:|
| Nodes | 4,507,362 | 3,108,953 | **1,398,409 (31.02%)** |
| Expansions | 4,507,328 | 3,108,934 | **1,398,394 (31.02%)** |
| Solve wall | 495.592 s | 343.604 s | **151.988 s (30.67%)** |
| Pair generation | 216.228 s | 146.987 s | 69.241 s |
| Defender enumeration | 175.837 s | 122.525 s | 53.312 s |
| Prior/regen | 43.036 s | 29.307 s | 13.729 s |
| Full refresh | 23.557 s | 17.164 s | 6.393 s |

`0l4291i_live` alone repeats 1,109,997 expansions and 121.864 s before
starting the rung that closes. `94gnnol` repeats 109,998 expansions/13.497 s;
`lz60mfb`, 109,998/9.601 s; and `mvp2lvc`, 27,955/3.148 s.

This observation produces candidate 1. It also explains why looking only at a
single final-cap solver profile missed a large trainer/harness-facing lever.

## 2. Candidate 1 -- same-query node-cap continuation

### Name and mechanism

**Resumable `WidePnSearch` session across a monotone node-cap ladder.** Keep
one exact root's arena, position index, deferred keys, pn/dn values,
commitment state, and staged-depth state alive at 10k, then continue it to
100k, 1M, and 20M. Root, claimant, semantic horizon, width, feature flags, TT
cap, and ordering remain unchanged. Only the expansion ceiling rises.

Today `tss_corpus.rs` constructs a fresh `TssSolver` inside the rung loop, and
`prove_for_wide_pn` constructs and drops a local `WidePnSearch` on every call.
Even reusing a `TssSolver` would retain only finished positive fragments, not
the unresolved frontier.

### Why the program missed it

The program studied three nearby but different ideas:

- semantic-horizon laddering, which reopens a different bounded problem and
  was correctly refuted;
- U22/shared fragments, which persist only independently verified positive
  proofs across roots; and
- U23 residual re-attack, which emits blocking replies for routing on a later
  pass.

None resumes the exact unfinished search for the same query. The register's
“persistent proof-number frontier” is persistent only for one `solve` call.
Fresh-per-rung construction was treated as benchmark methodology, not as a
call-surface cost.

### Prospective size

The exposed ceiling is **30.67% of current official solve wall** and 31.02% of
expansions. This is not claimed as achieved speedup. A correct pause/resume API
may pay unwind/re-entry overhead or alter a small amount of local scheduling.
It needs to recover only 16.3% of the exposed repeated tax to clear the 5%
end-to-end bar, which is a low bar for retaining the exact arena.

Memory does not multiply across roots: the session is released after that
root closes/exhausts, and its peak is the same high-cap arena the final fresh
solve already constructs.

### Soundness obligation

- Bind the session to the exact `RootBinding`, claimant, goal, semantic
  horizon, width, all search flags, hash mode, and TT cap.
- Permit only a monotonically larger node cap. A changed horizon or profile
  creates a new session.
- Preserve the current staged-depth cursor. Calling `run()` naively again is
  insufficient because it initializes local stage state.
- Treat every lower-cap result as `Unknown`, never as a cached game fact.
- A hard result still uses the ordinary materializer and unchanged strict
  verifier. Unfinished pn/dn values never cross the finder/verifier boundary.
- On panic or binding mismatch, discard the session rather than attempting a
  partial recovery.

### Exact next hunt

Build a cfg(test)-only session API, tentatively `advance_to_node_cap`, with
milestones 10k -> 100k -> 1M -> 20M.

1. At each milestone compare resumed root pn/dn, status, and cumulative
   expansions with a fresh solve at that cap.
2. Compare the final resumed outcome with one uninterrupted final-cap solve.
   Certificate bytes may differ if pause unwinding changes an otherwise legal
   tie; status must not contradict and every hard certificate must verify.
3. Run `0l`, `94gnnol`, `lz60mfb`, `mvp2lvc`, both Hayes rows, and a simple
   10k closer first.
4. Run the full official 1 GiB lazy+gate profile. Report cumulative wall,
   nodes, pair/defender timers, peak bytes, and re-entry count.
5. Stop if the full-profile gain is below 5%. If green, expose an in-process
   session handle; serialization is a separate proposal and is not required.

## 3. Candidate 2 -- prior-scale-aware df-pn thresholds

### Name and mechanism

**Dynamic delta threshold increments matched to immutable child priors.** The
wide engine initializes proof numbers from fork degree in the range 1..37 and
disproof numbers from tau, but its standard df-pn descent still gives the
selected child a second-best-sibling threshold of `second + 1` (with the
necessary `child + 1` progress floor). Test a small fixed delta and a dynamic
delta derived from the siblings' immutable initialized scale before testing a
small 1+epsilon grid.

The proof-number survey identifies exactly this failure mode: heuristic
initial values larger than one can make `+1` increments too small and cause
excessive internal re-traversal; constant or mean-initialization deltas are the
published remedy. The multiplicative 1+epsilon trick addresses a related
seesaw/thrashing problem.

### Why the program missed it

Round 9's major scheduling landing correctly replaced single-expansion root
redescent with standard threshold-bounded df-pn. Subsequent rounds optimized
generation, frontier admission, and commitment, but never revisited the unit
threshold after non-unit fork/tau priors became normative. The proof document
still explicitly says there is no epsilon, and the register has no threshold-
scale row.

### Prospective size

Local achieved size is unknown, so this is a measured A/B candidate rather
than a build recommendation.

- If expansion order were unchanged, the directly relevant scheduler/
  traversal/materialization residue is at most 15.06% of wall; cutting roughly
  one third of that clears 5%.
- Thresholds can also change which subtrees expand before a cap, so the true
  effect is not limited to the non-expansion residue.
- In a Connect6 relevance-zone df-pn study, best-per-position tuned 1+epsilon
  reduced node count by 29.2% on average and a deeper prior shaping reduced it
  by 43.5%. Those are **plausibility only**: eight positions, per-position
  tuning, different generator, and some rows regressed.
- The original 1+epsilon work is strongest when a search exceeds its TT. That
  premise is absent in the 1 GiB official profile, so dynamic delta is the
  first test; a broad epsilon sweep would be unjustified.

### Soundness obligation

Thresholds affect scheduling only. Keep the min/sum recurrences, child
universe, terminal classification, and certificate construction unchanged.
Every child threshold must remain strictly above the child's current number,
arithmetic must saturate at `PN_INFINITY`, and commitment-domain behavior must
remain unchanged in the first round. Finite-cap statuses may change only
toward a differently discovered verified proof or `Unknown`; every hard result
must pass the existing verifier, with no WIN/LOSS contradiction.

### Exact next hunt

This hunt earns a grid only after a cheap counter pass.

1. Add default-off counters for threshold-cross returns, same-parent sibling
   switches/reselections, recursive node visits, expansions per residency,
   and time in descent/state apply-undo outside expansion.
2. If avoidable re-traversal cannot account for 5% of full wall, stop and mark
   the lever dry.
3. Otherwise A/B `+1`, delta 2, delta 4, and delta equal to mean immutable
   sibling prior. Only then try epsilon `{1/8, 1/4, 1/2}`.
4. Use the full 1 GiB lazy+gate corpus plus isolated `0l`, `94gnnol`, and
   `lz60mfb`. Add the selected Phase-3 D h8/h16 cells to catch small-cap
   regressions.
5. Report wall, expansions, TT hits, stage refreshes, sibling switches, and
   verifier results. Require >=5% aggregate deep improvement without a
   material Phase-3 regression; do not choose a different parameter per root.

## 4. Candidate 3 -- opening-root stabilizer quotient

### Name and mechanism

**One representative per D6 stabilizer orbit at a primal opening-atlas Choice
root.** For a root state `S`, compute the subgroup of the 12 D6 transforms
that fixes the complete semantic binding. Partition complete root children
under that subgroup and retain one deterministic representative from each
orbit.

A-0 normally solves for the current player, so the root is a claimant Choice
node. An automorphism maps a winning child to an isomorphic winning child;
one representative preserves the existential truth. The ordinary certificate
records the concrete retained move and the unchanged verifier replays it.

### Why the program missed it

Two different symmetry questions were already closed or used:

- the opening census canonicalizes **between roots**, reducing 863 raw P2
  replies to 262 families; and
- NQ5 tested D6 folding **inside midgame search TT state** and found zero
  useful duplicates.

Neither quotients the immediate children by an automorphism that fixes one
high-symmetry opening root. `canonical_frame` currently orders ties only; the
wide position key remains raw. This is therefore distinct from the dead NQ5
proposal and from the already-consumed atlas family canonicalization.

### Prospective size

A read-only census of all 6,902 human games gives:

| Root-family stabilizer size | Canonical families | Games | Game share |
|---:|---:|---:|---:|
| 1 | 201 | 2,559 | 37.08% |
| 2 | 53 | 4,088 | 59.23% |
| 4 | 8 | 255 | 3.69% |

Nontrivial stabilizers cover **62.92%** of games. Assuming generic orbits,
the game-weighted root-child-removal ceiling is **32.39%**. The top three
opening families all have stabilizer 2; ranks 12 and 18 have stabilizer 4.

This is not an end-to-end claim. Choice proofs may already find one winning
representative before exploring its symmetric mate, while UNKNOWN/refuted
roots can pay for more of the duplicated orbits. The candidate remains
realistic because the highest-frequency atlas roots are symmetric and each
omitted root child represents an entire isomorphic subtree, but A-0 profiling
must show >=5% wall.

### Soundness obligation

- Compute the stabilizer from the complete root binding: owner-labelled
  occupancy, current player, claimant, phase and pending first stone,
  placement clock, terminal semantics, and rules/profile identity.
- Scope the first build to primal claimant **Choice roots**. Do not prune a
  Universal node; the verifier would require every defender obligation.
- Orbit complete edge semantics. In particular, do not identify two ordered
  pair applications merely because their unordered coordinate sets match
  unless the existing complete-turn contract licenses it.
- Retain one actual raw representative and emit a normal certificate. No
  symmetry fact is trusted by the verifier.
- Differential all 12 transforms and fail closed to the full child list if
  the stabilizer/orbit construction is inconsistent.

### Exact next hunt

1. Add shadow-only root telemetry to the ranked A-0 families: stabilizer,
   raw children, orbits, fixed children, root-generation wall, and expansions/
   wall below each orbit.
2. Start with the top 10 families at 10k and 100k under the 1 GiB lazy+gate
   profile. Extend caps only after a root survives the small rungs.
3. Consume only at primal Choice roots. Compare all 12 transformed roots and
   strict-verify every hard result.
4. Report family-weighted and game-frequency-weighted wall, not only child
   counts. Stop if the atlas gain is below 5%.

## 5. Algorithm-class sweep

| Shape | Current disposition | Why it is or is not a candidate |
|---|---|---|
| Standard df-pn threshold descent | Already landed | `work` uses second-best thresholds, conjunctive budget subtraction, and a child-progress floor. |
| Prior-scale-aware delta | **New candidate 2** | Non-unit 1..37 priors still use a unit increment; exact published mismatch. |
| Pure 1+epsilon for TT thrashing | Not independently nominated | The 1 GiB position index fits; test only after local seesaw counters justify it. |
| Weak/evaluation-informed initialization | Already partly implemented/posed | Fork/tau priors are live. `live_ge3` has an explicit unrun A/B recommendation; see closure debt below. |
| Delayed evaluation / dynamic widening | Already posed, not disposed | Old optimization spec says “lazy ordering (order only what gets expanded)”; register RZOP item 8 says re-profile. Current code still eagerly classifies all attacker pairs, so this debt remains real but is not a new idea. |
| Proof-set search | Dry | Exact set unions were approximated by prior DAG-frontier work and removed; the survey says full proof-set search has not succeeded in practice. It fights the 1 GiB/256 KiB memory goals. |
| WPNS/source-node/DAG-aware pn/dn | Dry on current evidence | Exact TT hits are 2.27% overall and ~1.97% on `0l`; prior graph/DAG Choice-PN experiments lost. Exact true PN/DN on arbitrary DAGs is NP-hard. Reopen only if shared-descendant mass, not raw hits, exceeds 5%. |
| GHI, df-pn(r), cycle TCA | Inapplicable | Every edge adds a stone, so the search graph is acyclic. The exact key includes occupancy/player/phase/clock and the omitted history is proved future-irrelevant. |
| PN2/PDS-PN or fixed-depth leaf probes for `0l` | Dry | Current solver beats pdspn wall on `0l`; earlier bounded probes/racer paid the same generator cost and regressed. The `94gnnol` reference differential remains separate. |
| Proof tree vs proof DAG storage | Already done | Search uses exact-position DAG entries; materialization compacts/reuses certificate nodes; finished cross-solve fragments were measured and found uneconomic. |
| Neural/Monte-Carlo initialization | Unrealistic here | Current Phase-3 median is around 0.1--0.2 ms; inference would require a demonstrated batching surface and a cheap score has not beaten fork/tau. |

### Existing closure debt A -- dynamic child reveal

The current consolidated engine still builds `WideTurnGate`, generates every
second candidate, evaluates/deduplicates every attacker pair, and returns a
full `Vec` before df-pn selects a child. Lazy frontier delays arena/TT
admission, not this classification. Attacker pair generation is 43.6% of wall;
avoiding 11.5% of it would clear 5% end to end. Dynamic widening is
correctness-preserving when hidden children are eventually revealed and a
Choice node cannot be refuted before generator exhaustion.

This is a credible lever, but it is not newly posed: “lazy ordering (order only
what gets expanded)” appears verbatim in the old optimization spec. The
register should either run its already-called-for re-profile/A-B or mark it
uneconomic. The next existing round should count candidate pairs evaluated,
accepted, selected, linked, and expanded; winning-child rank; and split
gate-build/second-candidate/evaluation/dedup wall. A cursor must use a
hidden-universe sentinel, allow a visible pn=0 proof, and forbid refutation
before exhaustion.

### Existing closure debt B -- `live_ge3` seed

`HUNT_REPORT_PN_INIT.md` reports an outcome-labelled replay of 1,093 -> 355
nodes on solved forcing roots and 652 -> 408 on solved human roots. It also
reports slightly negative global Spearman correlation and warns that the
replay freezes outcomes/subtree costs. The report therefore recommends only a
live test-only A/B after the census gate. The register cites that report for
U26 but does not dispose this seed.

Do not ship from replay. The already-posed round is: default-off seed,
separately timed scan outside the census domain, full 1 GiB and selected leaf
cells, expansions/wall/TT/sibling switches, unchanged recurrence and strict
verification. A negative live A/B closes it; an aggregate >=5% gain promotes
it.

## 6. Domain-structure sweep

| Angle checked | Disposition |
|---|---|
| Attacker threat sequencing | Atomic attacker pairs, forcing-first/quiet-after-exhaustion, width tiers, fork/tau priors, root sequential probes, and commitment domains already cover the credible structures. Literal commutation was <=0.16%; the racer was wall-negative. No new sequence law emerged. |
| Defender reply typing beyond K_reply | Exact `K_b`, K2 pair planning, sparse LOSS witnesses, ranked zones, K_reply, parked b=2 domination, and live U24 macromove/class collapse cover the identifiable classes. “Same winning continuation” is U24, not a new lever. |
| Window-graph decomposition | Nominally separate windows remain coupled by the two-stone reply budget, counterwins, attacker switching, and radius-8 frontier growth. A sound product needs a frontier-equivalence/bisimulation theorem close to U24/D17, with no frequency or wall evidence. Not realistic now. |
| Small-residue/endgame database | Hexo has an unbounded sparse board and a radius-8 legal frontier; there is no monotone “few empty cells left” residue. NQ3 found 0/180 current-format support transfers, and the remote quiet witness defeats a naive local signature. Immediate tactical residues are already leaves. |
| Root D6 | **New candidate 3.** Between-root canonicalization and mid-search TT folding do not consume a root stabilizer quotient. |
| Search D6 | Dead NQ5 remains dead. Midgame exact duplicates were zero; do not reopen it through the atlas candidate. |
| C_rel, b=2 domination, GAP-RAW | Remain parked under their recorded reopening conditions. No new evidence crosses one. |

## 7. Cross-surface sweep

### Node-cap ladders

This is candidate 1. It is the one material cost hidden by single-solve
profiling.

### Leaf horizon height

The h8/h16 question is an explicit owner budget/capability decision, not an
unposed solver lever. At cap 500, selected D moves from 16/300 verdicts and
76.513 ms at h8 to 39/300 and 704.706 ms at h16: 2.44x verdicts for 9.21x
aggregate wall. Broad semantic-horizon laddering was already refuted. A live
trainer value study may choose h16, but this ideation round must not relabel
that choice as efficiency.

### MCTS batching and caching

The trainer already has:

- same-leaf Pending/Done memoization;
- decided-proof retention across moves;
- a persistent solver per async worker;
- FIFO/LIFO queue modes, parking, scale-up, timeout/bail, sampling, and inline
  share controls; and
- a six-nearby-state persistent-solver leaf campaign.

Shared fragments had zero h8 hits and 22/875 h16 hits, no added verdict, and a
wall tax. The selected h8 campaign totals only 76.513 ms over 300 solves. No
new batching/cache interface has a measured >=5% solver opportunity. The only
material persistence omission is the exact unfinished frontier in candidate 1.

## 8. Reference-solver capstone check

The historical columns are useful but are **not a matched-budget experiment**.
`docs/TSS_VCF_WIDTH_BRIEF.md` records neither runnable reference command,
pinned binary/commit, hardware, retained log, nor total nested leaf work. No
idtt/dfpn/pdspn implementation or driver exists in this worktree; the local
`tss_reference` modules are independent exhaustive oracles, not those
algorithms.

What the existing numbers say:

- On the 13 easy reference-WIN rows, every cited idtt wall is below the current
  1 GiB final-rung wall. The current rows sum to **19.091 s** versus **3.466 s**
  cited, a nominal 15.625 s gap. That is 3.15% of the fresh official solve
  wall, or 4.55% of the final-attempt-only wall after removing cap-rung repeat.
  It does not meet the 5% program bar even if the unmatched numbers are taken
  literally.
- `0l4291i_live` is the opposite: current closes in 197.485 s in this
  instrumented 1 GiB run (177.7 s in the earlier headline) versus pdspn's
  cited 264 s. Its 1,879,612 expansions are not comparable with pdspn's 1,058
  top-level nodes plus 733 leaf solves.
- `94gnnol` is the only material apparent deficit: current is still UNKNOWN
  after the 1M rung (125.022 s final rung; 138.519 s cumulative) while pdspn is
  cited as NO in 21 s/108 top-level nodes. Nominally this is >20% of the full
  profile, but the meaning of NO and the uncounted leaf work are not pinned.
- `l9mxn59` and `mvp2lvc` have dfpn node references but no comparable wall
  provenance; current also treats these as one-sided NO controls, not game-
  loss certificates.

Therefore no reference gap is promoted **by definition** yet: the “matched”
premise is absent. The required next action is a `94gnnol` matched-reference
differential, not an invented port. Obtain the actual reference binary/source;
run the identical position on the same host; align whether NO means restricted
VCF exhaustion, global loss, or something else; and log all second-level leaf
nodes, memory, and wall. If the 21 s result reproduces with matching semantics,
that gap immediately becomes a top-ranked lever. Until then, neither a
PDS-PN port nor a disproof certificate is soundly specified.

## 9. Exact regeneration and next-round commands

### 9.1 Requested 1 GiB lazy+gate official profile

Run alone from this worktree. This is the exact requested gate-on profile; the
audited existing transcript differs only in the inert gate flag.

```powershell
$ErrorActionPreference = 'Stop'
if (Get-Process cargo -ErrorAction SilentlyContinue) {
    throw 'another cargo process is running'
}
$os = Get-CimInstance Win32_OperatingSystem
$free = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
if ($free -le 11) {
    throw "free RAM $free GiB is not above 11 GiB"
}
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }

$env:CARGO_TARGET_DIR = '.target-codex'
$env:TSS_BACKWALK_TT_BYTES = '1073741824'
$env:TSS_LAZY_FRONTIER = '1'
$env:TSS_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS = '0'
$env:TSS_CORPUS_EXPECT_LAZY_FRONTIER = '1'
$env:TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CORPUS_EXPECT_K_REPLY_CONSUME = '0'

cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture
```

### 9.2 Recompute the rung-repeat exposure from the retained transcript

```powershell
$runs = @()
$current = $null
Get-Content .codex-group2\codex-consolidate.log | ForEach-Object {
    if ($_ -match 'CORPUS_MODE .*tt_bytes_cap=1073741824') {
        $current = @()
    } elseif ($null -ne $current -and
              $_ -match '^CORPUS id=(\S+) cap=(\d+) status=(\S+).* nodes=(\d+).* expansions=(\d+).* ms=([0-9.]+)$') {
        $current += [pscustomobject]@{
            Id = $Matches[1]
            Nodes = [int64]$Matches[4]
            Expansions = [int64]$Matches[5]
            Ms = [double]$Matches[6]
        }
    } elseif ($null -ne $current -and $_ -match '^CORPUS_DONE ') {
        $runs += ,$current
        $current = $null
    }
}
$run = $runs[-1]
$last = @{}
foreach ($row in $run) { $last[$row.Id] = $row }
$freshNodes = ($run | Measure-Object Nodes -Sum).Sum
$finalNodes = ($last.Values | Measure-Object Nodes -Sum).Sum
$freshExpansions = ($run | Measure-Object Expansions -Sum).Sum
$finalExpansions = ($last.Values | Measure-Object Expansions -Sum).Sum
$freshMs = ($run | Measure-Object Ms -Sum).Sum
$finalMs = ($last.Values | Measure-Object Ms -Sum).Sum
[pscustomobject]@{
    FreshNodes = $freshNodes
    FinalAttemptNodes = $finalNodes
    RepeatedNodes = $freshNodes - $finalNodes
    RepeatedExpansions = $freshExpansions - $finalExpansions
    RepeatedMs = $freshMs - $finalMs
    RepeatedPct = 100 * ($freshMs - $finalMs) / $freshMs
} | Format-List
```

### 9.3 Recompute opening stabilizers

```powershell
function D6Key([int]$q, [int]$r, [int]$s) {
    if ($s -ge 6) { $r = -$q - $r }
    for ($i = 0; $i -lt ($s % 6); $i++) {
        $nq = -$r
        $nr = $q + $r
        $q = $nq
        $r = $nr
    }
    '{0},{1}' -f $q, $r
}

$counts = @{}
Get-Content -LiteralPath E:\Hexo-BotTrainer-hexgt\data\hexo-bootstrap-corpus\hexo_human_corpus.jsonl |
ForEach-Object {
    $game = $_ | ConvertFrom-Json
    if ($game.moves.Count -ge 3) {
        $a = $game.moves[1]
        $b = $game.moves[2]
        $best = $null
        for ($s = 0; $s -lt 12; $s++) {
            $ka = D6Key $a[0] $a[1] $s
            $kb = D6Key $b[0] $b[1] $s
            $key = ((@($ka, $kb) | Sort-Object) -join ';')
            if ($null -eq $best -or [string]::CompareOrdinal($key, $best) -lt 0) {
                $best = $key
            }
        }
        if ($counts.ContainsKey($best)) { $counts[$best]++ }
        else { $counts[$best] = 1 }
    }
}

$classes = foreach ($item in $counts.GetEnumerator()) {
    $parts = $item.Key -split ';'
    $a = @($parts[0] -split ',' | ForEach-Object { [int]$_ })
    $b = @($parts[1] -split ',' | ForEach-Object { [int]$_ })
    $stabilizer = 0
    for ($s = 0; $s -lt 12; $s++) {
        $pair = @(
            (D6Key $a[0] $a[1] $s),
            (D6Key $b[0] $b[1] $s)
        ) | Sort-Object
        if (($pair -join ';') -eq $item.Key) { $stabilizer++ }
    }
    [pscustomobject]@{
        Key = $item.Key
        Games = $item.Value
        Stabilizer = $stabilizer
    }
}

$classes | Group-Object Stabilizer | Sort-Object { [int]$_.Name } |
ForEach-Object {
    $games = ($_.Group | Measure-Object Games -Sum).Sum
    [pscustomobject]@{
        Stabilizer = [int]$_.Name
        Families = $_.Count
        Games = $games
        GamePct = [math]::Round(100 * $games / 6902, 2)
    }
} | Format-Table -AutoSize

$weighted = ($classes | ForEach-Object {
    $_.Games * (1 - 1 / $_.Stabilizer)
} | Measure-Object -Sum).Sum
"families=$($classes.Count)"
"games=$(($classes | Measure-Object Games -Sum).Sum)"
"weighted_root_child_reduction_pct=$([math]::Round(100 * $weighted / 6902, 2))"
```

### 9.4 Commands after the test-only hunts exist

The following names are proposed harness names, not commands that work on the
current tree. Each round must add only default-off cfg(test) instrumentation,
check RAM immediately before Cargo, use `.target-codex`, the MSVC target, one
Cargo process, and one test thread.

```powershell
# Candidate 1
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_cap_resume_campaign -- `
    --ignored --test-threads=1 --nocapture

# Candidate 2
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_threshold_scale_campaign -- `
    --ignored --test-threads=1 --nocapture

# Candidate 3
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_root_stabilizer_atlas_campaign -- `
    --ignored --test-threads=1 --nocapture
```

## Final stopping statement

The efficiency program cannot close at this register landing. Candidate 1 is
a measured 30.67% call-surface exposure and should be hunted first. Candidate
2 has a concrete local algorithm/literature mismatch and earns a tightly
bounded counter-first A/B. Candidate 3 is an exact symmetry quotient on the
owner's planned atlas and earns a top-family sizing run. Separately, the
already-posed dynamic-child and `live_ge3` rounds need explicit dispositions
before the register can call itself exhaustive.

If all five lever rounds fail their >=5%/capability gates **and** the matched
reference differential either closes or identifies and dispositions its
mechanism, the remaining telemetry, algorithm, domain, and trainer angles
audited here are dry under the current evidence. Today, however, **“nothing
realistic remains” would be false.**
