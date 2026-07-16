# NQ2 — λ²-connector locality: empirical hunt report

Working dir: `E:\Hexo-BotTrainer-hexgt\.claude\worktrees\hunt-quiet-locality`
(branch `hunt/quiet-locality`, HEAD `5536b2bb`, round-3 engine with the
quiet-turn machinery). Test-gated harness only; no production code changed; no
commits. `CARGO_TARGET_DIR=.target-hunt`, one cargo process, `--test-threads=1`,
TT ≤ 256 MiB, deterministic fixed seeds.

---

## Headline

- **Quiet turns exist ONLY in quiet-REQUIRED wins.** A position whose win can be
  proved by the fast forcing profile (`vcf_pair_complete`) yields an all-forcing
  certificate with **zero** quiet turns. All **122/122** leaf-width
  "wide-only-win" records and **1236/1237** human winning positions across two
  independent sweeps (different seeds, caps, tail windows) produced forcing wins
  with no quiet turn. Quiet turns appear only where **no pure-forcing win
  exists** and the engine must select a non-forcing move
  (e.g. `double_fork_compact`).
- **The quiet moves are single-family fork-extensions, not two-family
  connectors.** Every observed quiet placement (7/7 distinct) **joins a live
  attacker window** (100%) and sits at **hex-distance exactly 1 from a
  pre-existing attacker stone** (100%). The λ²-**connector** hypothesis — that a
  quiet move bridges ≥2 distinct threat families — is **refuted**: 5/7
  placements serve a single family (`d_two = NA`); only 2/7 have any second
  family, at distances 3 and 7.
- **They are all "loose-quiet".** 7/7 are engine-quiet because the defender
  keeps *slack* (min-hitting-set `< b`), not because no threat is made: the turn
  typically raises a window to count-4/5 (a fork the defender covers with one
  dual-purpose stone). None are passive sub-4 builds.
- **Strongest certified-universe candidate C(P): "legal cells within hex-distance
  1 of an attacker stone that also lie in a live attacker window."** Coverage
  **100%**, median size **36 of 534 legal ≈ 0.067×** (a ~15× shrink of the quiet
  OR universe).

Specimen count: **7 distinct quiet placements** across **6 distinct positions**
(5 spare-corpus + 1 natural human) plus 2 D6-image covariance copies.
Outliers on the primary law: **0** (adjacency + join-live are 100%). One
`d_used` outlier (`double_fork_ordered`, `d_used=4`) discussed below.

---

## Definitions used (stated explicitly)

Certificate = one verifier-accepted winning strategy for the claimant (the
attacker). A claimant **turn** is two placements (FirstStone then SecondStone).
Nodes: `Choice`/`OrCompletion` = attacker placement, `Universal` = defender.

**Quiet turn (primary criterion — the engine's own gate).** A turn is quiet iff,
at the post-turn position (defender to move), the engine's
`turn_forces_small_defender_reply` is **false**. That predicate is true (turn is
FORCING) iff the attacker wins now, OR (not opening) the attacker has ≥1 active
≥4 window (`opp_threat_count>0`), the defender cannot win now, and the defender's
minimum hitting set equals its budget `b` (a *tight* forced reply). This is
exactly the gate under which the engine's `quiet_turn_or_edges` machinery fires
and enumerates the full legal completion universe — the universe C(P) must
cover — so the hunt keys to it. The harness replicates it and cross-checks
against the engine's `round3_shadow_certificate` (`quiet_mine == quiet_engine`
on every specimen).

**strict_quiet (task's stricter reading).** `true` iff after the two placements
there is **no** attacker window with `count_A ≥ 4` and `count_D = 0` (no ≥4
threat at all). Recorded per turn. **All observed quiet turns are loose
(strict_quiet = false).**

**Sub-classes (per placement).**
`connector` = the cell lies in live windows of ≥2 distinct *served* families;
`remote_seed` = the cell touches no pre-existing attacker-active window
(`pre_active_windows = 0`); `pair_build` = raises a live window to attacker-count
2–3 (`max_new_count ∈ {2,3}`); else `other`. Priority connector > remote_seed >
pair_build > other.

**Per-placement measures.** `d_used` = hex distance to the nearest cell of any
window family *completed later* in the same certificate (families = connected
components, under window-cell overlap, of all `OrCompletion`/`Win`/`Loss`
witness windows in the subtree below the turn). `d_two` = distance to the
second-nearest such family (`-1`/NA if <2). `d_stone` = distance to the nearest
existing attacker stone at placement time. `window incidence` = per-count
buckets of `count_D=0` windows through the cell. `node_full_legal` = |legal
moves| at the OR node. Candidate C(P) sizes are computed **from the current
position only** (never from certificate knowledge), so they are usable by
Group-2 at solve time.

---

## Data sources and specimen counts

| Source | Positions solved | Method | WINs | with quiet turn | quiet placements |
|---|---:|---|---:|---:|---:|
| Spare-corpus family | 27 (+2 D6) | consume, cap 100k–400k, adaptive horizon | 15 | **5** | 5 (+2 D6) |
| Leaf-width records | 122 | VCF (fast; = the profile they were mined with) | 122 | **0** | 0 |
| Human sweep #1 | 800 sampled (seed 0xC0FFEE, cap 20k, tail 16) | two-stage VCF→consume | 568 | **1** | 2 |
| Human sweep #2 | 1000 sampled (seed 0xBEEF, cap 60k, tail 20) | two-stage VCF→consume | 669 | **0** | 0 |

Spare-corpus specimens with a quiet turn (all verifier-accepted):
`double_fork_compact`, `double_fork_dense`, `double_fork_ordered`,
`compact_urgent_spare`, `urgent_uncapped_junction`. The remaining spare IDs were
either trivial forcing wins (2–4 nodes, no quiet turn) or stayed UNKNOWN even at
cap 400k (`double_fork_spare`, `deep_urgent_spare`, `deep_quad_block`,
`deep_pruned_latents`, `shared_target_block_endpoints`, `human_2a94/feaa/5801`,
`human_6a5a_block_q/spare_edge`) — genuinely deeper, not converted by more nodes.

Natural human specimen: `2e9f13bcc909a44d:ply49` (WIN at 5,619 consume nodes,
verified) — the winning line's only unforced turn, both stones captured.

**Rarity is a finding.** Quiet-required wins are ~**0.08%** of human winning
positions (**1 of 1,237** wins across two independent sweeps with different
seeds, caps, and tail windows). Pure-forcing (VCF/pair-complete) wins dominate —
including 100% of the leaf-width wide-only-win corpus.

---

## Distance histograms (7 distinct placements)

```
d_stone :  1:7                         median 1     (100% adjacent to a stone)
d_used  :  0:3  2:1  3:2  4:1          median 2     (100% ≤ 4)
d_two   :  NA:5  3:1  7:1              median NA    (single family for 5/7)
```

Per-placement detail:

| position | placement | role | strict | d_stone | d_used | d_two | families served | sub | legal | \|adj1\| |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|
| double_fork_compact | (4,0) | 2nd | no | 1 | 3 | 7 | 2 | other | 478 | 35 |
| double_fork_dense | (-1,0) | 2nd | no | 1 | 3 | NA | 1 | other | 395 | 23 |
| double_fork_ordered | (-1,0) | 2nd | no | 1 | 4 | NA | 1 | other | 481 | 36 |
| compact_urgent_spare | (3,0) | 2nd | no | 1 | 2 | NA | 1 | other | 534 | 34 |
| urgent_uncapped_junction | (1,2) | 2nd | no | 1 | 0 | 3 | 2 | other | 672 | 47 |
| human 2e9f…:ply49 | (5,-10) | 1st | no | 1 | 0 | NA | 1 | pair_build | 563 | 71 |
| human 2e9f…:ply49 | (5,-4) | 2nd | no | 1 | 0 | NA | 1 | other | 562 | 71 |

The spare specimens are all the *turn-completing* stone (the turn's first stone
is pre-root or forcing); the natural human specimen captures *both* stones of the
quiet turn — and its first stone is a genuine count-3 `pair_build`, the second
the count-4 fork completer. Both still obey `d_stone=1`, `d_used=0`.

---

## The empirical locality law (with outliers shown honestly)

Over all 7 distinct quiet placements:

- **L1 (adjacency):** `d_stone = 1` — 7/7 = **100%**. Every quiet winning
  placement is one hex from a pre-existing attacker stone.
- **L2 (join):** the cell lies in a live attacker window — 7/7 = **100%**.
- **L3 (served-proximity):** `d_used ≤ 4` — 7/7 = **100%**; `d_used = 0` for 3/7.
- **¬L4 (connector refuted):** `d_two ≤ 3` — 1/7 = **14%**. Quiet moves serve a
  single family; they extend one threat structure, not bridge two.

**Outlier investigation.** No placement violates L1/L2. The only measure with a
tail is `d_used`: `double_fork_ordered` sits `d_used = 4` from the nearest family
it ultimately completes. This is **not** a locality violation on the primary law
(it is still `d_stone=1`, join-live) — it reflects that its immediate fork
threats are *decoys the defender is forced to cover*, so those windows are never
completed and do not count as "served"; the real served family is 4 hexes away.
The certificate is verifier-accepted and the quiet move is the *unique* unforced
OR node, so there is no closer-quiet alternative *within this strategy*. Re-solve
at higher cap (100k→400k) returned the identical certificate (the position closes
at 2,530 nodes, far below cap), so no shorter/closer certificate is hiding under
the cap for this specimen. **Caveat:** certificates are one winning strategy, not
all; `d_used` is a lower bound over the found strategy, and a *different* winning
strategy could place the quiet move nearer or farther. The adjacency law L1,
being a property the move shares with any threat-extension, is the robust
invariant; `d_used` is strategy-dependent.

---

## Candidate C(P) definitions — coverage vs universe size

Each rule is evaluated at the quiet OR node from the current position only
(median over the 7 placements; `shrink` = median |C| / median |legal| = 534).

| candidate C(P) | rule | coverage | median \|C\| | shrink |
|---|---|---:|---:|---:|
| **join_adj1** | in a live attacker window **and** `dist≤1` to a stone | **100%** | **36** | **0.067×** |
| adj_stone_k1 | `dist≤1` to an attacker stone | 100% | 36 | 0.067× |
| adj_stone_k2 | `dist≤2` to an attacker stone | 100% | 83 | 0.155× |
| join_live | in some live attacker window (extends a threat) | 100% | 156 | 0.292× |
| nearpair_k1 | `dist≤1` of a live family with a ≥2 window | 100% | 259 | 0.485× |
| in2fam_k0 / near2fam_k1 / near2fam_k2 | in / near ≥2 live families (connector rules) | **0%** | 0 | — |

The connector rules (`in2fam`, `near2fam`) capture **nothing** — direct
confirmation that these quiet moves are not two-family bridges. `join_live`
(the "extend an existing threat" rule) captures all at 0.29×. Intersecting with
1-hex adjacency (`join_adj1`) tightens to **0.067×** with no coverage loss.
`adj_stone_k1` alone is essentially identical (an attacker stone's immediate
neighbourhood is almost entirely inside its own live windows), so the two are
interchangeable in practice.

---

## Conjecture (the certified quiet universe C(P))

> **NQ2 locality conjecture.** Let `P` be an attacker OR node that is *unforced*
> (the quiet-turn gate fires). If the attacker has a win from `P` whose winning
> line begins with a quiet turn, then it has such a win whose quiet placement `c`
> satisfies **both** (i) `c` lies in a live attacker window — an active length-6
> window with ≥1 attacker stone and 0 defender stones — and (ii)
> `dist(c, nearest attacker stone) ≤ 1`. Hence the certified quiet universe
> `C(P) = { legal c : (i) ∧ (ii) }` is complete for quiet-required wins, and
> `|C(P)|` is empirically ≈ **0.07×** the full legal completion set.

This is the attacker-side analog of the T3 defender zones: a
position-computable, theorem-shaped restriction of the wide quiet universe.

**Intended proof obstacle (named): the anchor / remote-seed problem.** The T3
defender-zone proof anchors locality on *forced* replies — the defender must
answer a ≥4 threat, and a budget/potential argument bounds where the reply can
sit. A quiet attacker move has **no such anchor**: it does not force, so the
forced-reply potential does not apply. Completeness of `C(P)` therefore needs a
new potential over the attacker's *build*: define the completion-distance of the
families the move serves and show a **remote seed** (a legal cell in no live
attacker window, condition (i) failing) never strictly reduces it more than an
adjacent join does — i.e. a strategy-preserving transformation that pulls any
hypothetical remote quiet move onto an adjacent live-window cell without losing
the win ("no useful remote seed" lemma). The empirical evidence for the lemma is
strong (0 remote seeds observed, every quiet move reduces exactly one live
family's completion distance by 1), but the transformation must be proved for
the general branching defense, which is the open obstacle.

---

## Effect of adopting C(P) on `quiet_turn_or_edges`

`quiet_turn_or_edges` in CONSUME mode currently enumerates the **complete legal
completion set** at an unforced attacker OR node (no locating theorem exists).
On the specimen quiet nodes that set has **395–672** cells (median 534).
Restricting to `C(P) = join_adj1` cuts it to **22–71** cells (median 36) — a
median **~15× reduction (0.067×)** of the quiet OR fan-out — while retaining the
actual winning move in **100%** of observed cases. On `double_fork_compact` the
478-move quiet frontier collapses to 35 candidates; on
`urgent_uncapped_junction`, 672 → 47. Group-2's next round would consume `C(P)`
exactly as it consumes the T3 defender zones, subject to the completeness proof.

---

## Caveats (respected)

1. **Found-certificate limitation.** A locality law over found certificates
   supports a *completeness* conjecture only if the engine finds a win whenever
   one exists at the caps used. Coverage numbers are over the single winning
   strategy the engine returned per position, not over all winning strategies.
2. **Outlier re-solve mitigation.** The `d_used` outlier
   (`double_fork_ordered`) and the closing specimens were re-solved at up to
   400k nodes (≥10× their closing node count); all returned the identical
   verifier-accepted certificate, so no closer-quiet certificate is hidden under
   the base cap for these positions. UNKNOWN spare positions did not convert at
   400k — deeper wins that a future higher-cap pass could add to the population.
3. **Small N.** 7 distinct quiet placements is a thin population because
   quiet-required wins are intrinsically rare (~0.1% of human wins; 0/122
   leaf-width). The *consistency* is the evidence: L1 and L2 hold at 100% across
   five structurally distinct hand-built forks (compact/dense/ordered double
   forks, an urgent spare, an uncapped junction) plus one natural human game.
4. **strict vs loose.** No strict-quiet (sub-4 build) quiet turn was observed;
   the engine-quiet universe in quiet-required wins consists of loose-quiet fork
   shots. A strict-quiet preparation regime, if it exists, lives in *longer*
   wins than the caps here reach and is unmeasured.

---

## Files

- `packages/hexfield_eq/rust/src/tss_quiet_locality_hunt.rs` — the `#[cfg(test)]`
  mining harness (3 ignored tests). Wired via one line in
  `packages/hexfield_eq/rust/src/lib.rs`. `mining_candidate` in
  `tss_spare_corpus.rs` was made `pub(crate)` (test-only module) so the harness
  can reach the spare-corpus family.
- `QUIET_LOCALITY_SPECIMENS.jsonl` — machine-readable, one line per quiet
  placement (position replay, placement, all measures, candidate sizes/hits).
- `QL_SPECIMENS.jsonl`, `QL_HUMAN.jsonl`, `QL_LEAFWIDTH.jsonl` — per-source raw.
- `aggregate_quiet_locality.py`, `merge_specimens.py` — analysis scripts.
- `QUIET_LOCALITY_AGG.txt` — full aggregation dump.

## Regeneration commands

```powershell
$env:CARGO_TARGET_DIR='.target-hunt'; $env:QL_TT_BYTES='268435456'
# specimens (spare-corpus family, consume mode, adaptive horizon)
$env:QL_CAP='400000'
cargo test --release -p hexfield_eq quiet_locality_specimens -- --ignored --test-threads=1 --nocapture
# leaf-width full 122 (fast VCF; confirms 0 quiet turns)
$env:QL_CAP='100000'; $env:QL_LIMIT='200'
cargo test --release -p hexfield_eq quiet_locality_leafwidth -- --ignored --test-threads=1 --nocapture
# human corpus tail-window sweep (two-stage VCF→consume); two independent sweeps
$env:QL_CAP='20000'; $env:QL_SAMPLE='800'; $env:QL_SEED='12648430'; $env:QL_TAIL='16'; $env:QL_MIN_PLY='14'
cargo test --release -p hexfield_eq quiet_locality_human -- --ignored --test-threads=1 --nocapture
$env:QL_CAP='60000'; $env:QL_SAMPLE='1000'; $env:QL_SEED='48879'; $env:QL_TAIL='20'; $env:QL_MIN_PLY='12'
$env:QL_OUT_HUMAN='...\QL_HUMAN2.jsonl'
cargo test --release -p hexfield_eq quiet_locality_human -- --ignored --test-threads=1 --nocapture
# aggregate
python merge_specimens.py QUIET_LOCALITY_SPECIMENS.jsonl QL_SPECIMENS.jsonl QL_HUMAN.jsonl QL_LEAFWIDTH.jsonl
python aggregate_quiet_locality.py QUIET_LOCALITY_SPECIMENS.jsonl
```
