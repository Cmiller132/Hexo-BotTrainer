# Domination b=2 experiment round

> **Incremental experiment log.** Worktree `hunt/domination`, source
> `17a6c6de519185066668e2b22be3217a3d85af32`. This round changes only the
> test-gated domination hunt harness and this report; it does not change
> production code. Machine outcomes are DATA, not proofs.

## 1. Repaired protocol, fixed before solving

### 1.1 Scope and atomic comparison

The parent `P` is a reachable, nonterminal defender-`FirstStone` position
(budget `b=2`) with `!own_win_now(P)` and a nonempty complete family `T` of
attacker-alive count-4/5 windows. A candidate is a legal ordered defender turn
`M=(u,v)`: `u` is legal in `P`, `P+u` is nonterminal, `v` is legal in `P+u`,
and `P+u+v` is nonterminal. The completed child `C_M` has the attacker to move
at `FirstStone`. A pair covers when `{u,v}` intersects every member of `T`.
Old/old pairs are quotiented only after independently checking P3's hypotheses;
directed newly-legal second cells retain their order and both legal old/old
first-action aliases remain attached to the canonical child.

At K1 (`mhs=1<b=2`), `H(P)` is the intersection of all threat-empty sets.
Covered turns are H-containing old/old pairs, old/old split covers with neither
cell in `H(P)`, or directed hit-first pairs with a newly legal spare. At K2
(`mhs=b=2`), covered turns are two-cell hitting sets. Uncovered turns are
B2-COVER controls: if the smallest uncovered initial window has `r` remaining
empties, the attacker must return `Win` by exact depth `r` (`r=1` or `2`).

The oracle value is exact stopped-horizon `ProofStatus` relative to the
attacker at `C_M`, ranked for the defender as `Loss > Unknown > Win`. A
completed `Unknown` means no forced terminal result through the horizon;
deadline or work exhaustion is `INCOMPLETE` and adjudicates nothing. First
stones are compared only after exhaustive second-stone aggregation
`F_d(u)=max_v rho_D(O_d(u,v))`; a sampled completion never establishes a
first-stone domination.

### 1.2 Repairs 2--6 folded in

- **Repair 2**: “Add the analytic baseline `O_0=O_1=O_2=Unknown` for every
  covered b=2 candidate. Recast `d=1,2` as uncovered controls/smoke tests, not
  discriminatory evidence.” Accordingly every covered row is asserted
  `Unknown` through d=2 without counting that equality as evidence. Oracle
  d=1,2 work is limited to B2-COVER, phase/order smoke tests, and selected
  metamorphic replays.

- **Repair 3**: “Make at least `d=3` mandatory for every nontrivial
  covered-candidate comparison, and state which later depth is mandatory for
  quiet/frontier classes. If the complete audit cannot finish that depth,
  report the associated directional subclaim as `NOT ADJUDICATED`.” Every
  covered comparison admitted to the outcome table must therefore complete at
  d=3. Quiet, dead, newly-legal, positive-support-delta, and other
  frontier-sensitive classes additionally require d=4. A sampled panel may
  falsify a universal, but only a complete all-second-move audit may adjudicate
  K1 first-stone dismissal or use “all/any” wording.

- **Repair 4**: “Give DRQ-LIFT and K2-P2-LIFT explicit eligible-pair quotas,
  or require a manifest count and report `NOT TESTED` when it is zero.” This
  round uses manifest counts: all discovered eligible DRQ and P2 pairs in the
  selected parents are counted; zero is reported as `NOT TESTED`, never as a
  passing control. Solving may be deterministically capped before outcomes are
  known, with the shortfall reported.

- **Repair 5**: “For each lifted coverer witness, assert at the preceding
  parent that `!own_win_now`, K1, and both recorded cells are in `H(P)`; retain
  the P3 reverse aliases. Otherwise label it a generic stress fixture, not a
  HIT-ANY control. Do not promise a mismatch by `d<=6` when the historical
  solve used a horizon up to 40.” The harness prints all four eligibility
  assertions and both order aliases. An ineligible lift remains only a named
  stress fixture. No depth-six mismatch is presumed.

- **Repair 6**: “Correct the K1/shipped-b2 terminology and Panel 3's
  spare-identity wording.” K1 is called the **b=2 spare-stone pruning** study;
  shipped b=2 `implicit_dispatch` is the distinct K2 `mhs=b=2` path. Panel 3
  fixes a common spare and varies hits, so it attacks hit identity and
  hit/spare interaction, not fixed-hit spare identity.

### 1.3 Frozen families and result-blind selection

The source universe is the decisive-game human corpus used by the existing
domination hunt plus its deterministic generators and named directed
fixtures. The seed is `7766554433221100`. Before solving, the dry run records
the corpus identity, every eligible b=2 parent (including the first eligible
prefix per corpus row), threat masks, legal width, K1/K2 classification,
candidate coordinates/order, coverage class, H membership, dead/quiet/G3 and
support-delta flags, DRQ/P2 eligibility, and lifted-fixture eligibility. The
manifest is sealed with SHA-256; solve rows bind that digest. Outcomes cannot
refill or reorder it.

The repaired panels retain §7's K1 fixed-hit/spare strata, K1 split-cover
comparisons, common-spare/multiple-hit counterfork strata, K2 multiple hitting
sets, uncovered-deadline controls, directed/lifted fixtures, D6/P3
metamorphics, and small complete K1/K2 audits. Panel quotas may be downscoped
result-blind to keep each invocation below ten minutes; exact shortfalls and
unadjudicated universal claims are reported.

### 1.4 Depth, qualification, taxonomy, and stopping

The depth ladder is: analytic/smoke d=0..2; mandatory discriminatory d=3 for
all covered comparisons; mandatory d=4 for quiet/frontier classes; d=5,6 only
for surviving or frozen targeted cases. A “through d=N” statement requires a
contiguous completed matrix. Every exact mismatch is rerun without D6
canonicalization and with a second TT size. The stock/fast oracle qualification
must add attacker-`Loss` coverage before any b=2 `Loss` is used; a shortfall
blocks that use and is reported rather than silently weakening the gate.

Each comparison is classified as:

- **dominated**: the retained candidate has defender rank at least that of the
  omitted candidate at every completed required depth;
- **non-dominated with witness**: an exact completed depth reverses the claimed
  inequality (or differs for an equality claim); the deterministic seed,
  corpus hash/prefix, full position dump, ordered moves, values, and oracle
  configuration are frozen;
- **analytically-Unknown**: covered d=0..2 equality only, or a required
  discriminatory solve is incomplete. The latter is also labelled `NOT
  ADJUDICATED`; it is not a draw and not affirmative evidence.

The round stops a conjecture's deeper universal work at its first exact
refutation. Otherwise the finite result remains empirical and is handed to a
pencil-proof round.

### 1.5 Ambiguity resolutions fixed before running

1. “The 19-corpus rows” means the 19 checked-in rows of
   `rust/corpus/forcing_corpus_moves.txt`. The harness asserts that row count,
   scans every legal prefix of each row, and reports which rows actually expose
   b=2 defensive parents and the first such prefix. The broader human corpus
   remains the existing hunt's corpus generator and is inventoried separately.
2. The campaign-decision conjecture is K1 first-stone spare pruning, not
   coverer equivalence and not the already-distinct K2 shipped arm: at a
   `b=2,mhs=1,!own_win_now` parent, every legal first stone outside `H(P)` is
   dominated, after complete second-stone aggregation, by at least one first
   stone in `H(P)`. K2 hitting-set identity and DRQ/P2 lifts are boundary and
   harness controls.
3. “Deeper if budget allows” means d=4 is compulsory for any quiet/frontier
   claim admitted to the verdict; d=5,6 are optional targeted follow-up.
4. The host rule “free RAM >9 GB” supersedes §7's older >8 GiB wording. Every
   cargo invocation is separately gated, serialized, uses
   `CARGO_TARGET_DIR=.target-hunt`, and uses `--test-threads=1`.

## 2. Execution log

### 2.1 Harness and manifests

The existing `tss_domination_hunt` test module now has repaired-round runners
for:

- the full human-corpus b=2 inventory and sealed candidate coordinates;
- first-occurrence inventory over the checked-in 19-row forcing corpus;
- explicit DRQ-LIFT/P2-LIFT eligible-pair manifest counts;
- stock/fast attacker-`Loss` qualification with `Complete | Incomplete` stock
  backup semantics;
- one exact exhaustive `F_d(u)` first-action aggregation per invocation; and
- one exact completed macromove per invocation, including the full replay,
  ordered cells, coverage class, support deltas, status, nodes, TT data, D6
  setting, and wall time.

All are registered under the crate's existing `#[cfg(test)]` module. Every
cargo invocation used `.target-hunt`, `--test-threads=1`, and a measured
free-RAM value greater than 9 GiB. No cargo invocations overlapped in this
worktree. `rustfmt --edition 2021` and `cargo test -p hexfield_eq --release
--no-run` passed.

The human-corpus dry manifest contains 42,664 eligible b=2 parents and is
sealed as:

```text
SHA-256 4015CEDD327563DB945C77161E66EC2F5F6182B4C58D521A50B646E9C8EC3CCF
```

Its measured population is:

| Measure | Count |
|---|---:|
| decisive human-corpus rows with at least one b=2 parent | 6,687 / 6,902 |
| b=2 parents | 42,664 |
| K1 (`mhs=1<b=2`) | 20,307 |
| K1 with a split cover | 3,323 |
| K2 (`mhs=b=2`) | 22,357 |
| K2 with multiple minimum hitting sets | 21,643 |

This broad inventory corrected the pre-run ambiguity: the requested
“19-corpus” is the separate checked-in forcing corpus, not a claim that only
19 human games contain b=2 nodes.

### 2.2 Checked-in 19-row forcing corpus

The final forcing manifest SHA-256 is
`9EB8EB4C6EAE30C7A292D8F9ECD7BDEF3B8EED39049D1009530E10F40CA57B3B`.
Thirteen rows expose a qualifying b=2 defensive parent and six do not:

| Row | b=2 nodes | First prefix | First mhs | First legal width |
|---|---:|---:|---:|---:|
| `0hz3hty` | 0 | — | — | — |
| `0l4291i_live` | 2 | 49 | 1 | 543 |
| `8is963b` | 7 | 11 | 2 | 323 |
| `94gnnol` | 4 | 7 | 1 | 296 |
| `acly7kb` | 5 | 11 | 2 | 323 |
| `dy3dg99` | 5 | 15 | 1 | 355 |
| `g2xx6wl` | 16 | 27 | 1 | 392 |
| `hu01jk4` | 13 | 27 | 1 | 373 |
| `jh7yo7y` | 3 | 13 | 1 | 312 |
| `jnzzmcm` | 1 | 15 | 1 | 358 |
| `l9mxn59` | 0 | — | — | — |
| `lz60mfb` | 4 | 15 | 1 | 332 |
| `mvp2lvc` | 6 | 11 | 2 | 345 |
| `xsnfyll` | 0 | — | — | — |
| `zrugh2x` | 1 | 37 | 2 | 496 |
| `strongloss_a_prefix6` | 0 | — | — | — |
| `strongloss_b_prefix8` | 0 | — | — | — |
| `hayes_20260712_turn16` | 1 | 31 | 1 | 531 |
| `hayes_20260712_placement31` | 0 | — | — | — |

### 2.3 Repair-4 control eligibility

The first four K1 and first four K2 audit parents, sorted exactly by
`(legal width, game_hash, prefix)`, produced these manifest counts:

| Control | Parents | Eligible pairs | Protocol result |
|---|---:|---:|---|
| K1 DRQ-LIFT | 4 | 0 | **NOT TESTED** |
| K2 P2-LIFT | 4 | 0 | **NOT TESTED** |

This is not a zero-mismatch pass. It is the explicit non-vacuity disposition
required by repair 4.

### 2.4 Lifted coverer controls

The preceding-parent assertions from repair 5 gave:

| Historical witness | Parent | Result at parent |
|---|---|---|
| canonical #1 | `d7e1b56c925b7f32:19` | `!own_win_now`, K1, both recorded cells `(-2,3),(-1,2)` in `H(P)`; eligible HIT-ANY stress control |
| doubly-proven #2 | `1b73025a7265899c:37` | K2, not K1; generic stress fixture only |
| doubly-proven #3 | `41c4a1056d405fc7:89` | K2, not K1; generic stress fixture only |
| doubly-proven #4 | `24d8dc7181b59042:39` | `!own_win_now`, K1, both recorded cells `(2,-1),(4,-3)` in `H(P)`; eligible HIT-ANY stress control |

Old/old reverse orders remain separately expressible by the ordered pair
runner; no mismatch by d=6 was assumed.

### 2.5 Qualification and budget stops

Q0 did not turn green. The first registered nine-minute pass completed zero
of the required 16 stock-`Loss` rows: every attempted tactical row hit the
one-million-stock-node cap and was `INCOMPLETE`. A second pass restored the
stock oracle's direct-extension ordering, but an unbounded fast recheck then
continued beyond the intended cooperative deadline; it was terminated and
the entire invocation was discarded. Therefore **no attacker-`Loss` result is
admitted in this report**.

The first exact complete K1 aggregation was the smallest K1 split parent,
`32f44c499244b611:9` (297 legal cells, `H={(2,1)}`, split first cells
`{(-2,1),(4,1)}`). `F_3((2,1))` did not finish in 574 seconds and was stopped;
it is `INCOMPLETE`, not `Unknown`. The targeted d=4 canonical lifted case also
did not finish in 574 seconds and is `INCOMPLETE`. No run was allowed to
manufacture a stopped-horizon value from either cutoff.

## 3. Results

### 3.1 Analytic d=0..2 baseline

For every covered child in the manifest,
`O_0=O_1=O_2=Unknown`. These are **analytically-Unknown smoke values**, not
domination evidence. Uncovered B2-COVER retains its proved deadline values;
the round found no harness disagreement but did not spend oracle budget
re-proving the analytic control across all 42,664 parents.

### 3.2 Completed d>=3 matrix

Eleven covered completed turns finished at d=3. All eleven returned exact
attacker `Unknown`; node counts ranged from 121,593 to 173,714, with D6
canonicalization enabled and a 512 MiB TT.

| Family / parent | Covered children completed | d=3 statuses | Required later depth | Classification |
|---|---:|---|---|---|
| K1 audit #1 `32f44c499244b611:9` | 1 split + 2 H-containing | 3 `Unknown` | d=4 (support delta 17 on spare/split cells) | analytically/exact-Unknown; directional claim **NOT ADJUDICATED** |
| K1 audit #2 `19b085e7aa9f6215:9` | 1 split + 1 H-containing | 2 `Unknown` | d=4 | analytically/exact-Unknown; **NOT ADJUDICATED** |
| K1 audit #3 `498a61ae0b5cf4ef:9` | 1 split + 1 H-containing | 2 `Unknown` | d=4 | analytically/exact-Unknown; **NOT ADJUDICATED** |
| K1 audit #4 `fd688f189544bf72:9` | 1 split + 1 H-containing | 2 `Unknown` | d=4 | analytically/exact-Unknown; **NOT ADJUDICATED** |
| eligible lifted canonical control `d7e1b56c925b7f32:19` | 2 H-containing | 2 `Unknown` | d=4 | exact-Unknown; HIT-ANY **NOT ADJUDICATED** |
| **Total** | **11** | **11 `Unknown`; 0 `Win`; 0 admitted `Loss`** | — | **0 dominated claims; 0 admissible non-domination witnesses** |

The representative exact rows include:

| Parent | Ordered pair | Class | d=3 attacker status | Nodes |
|---|---|---|---|---:|
| `32f44c499244b611:9` | `(-2,1);(4,1)` | split | `Unknown` | 133,234 |
| `32f44c499244b611:9` | `(2,1);(-2,1)` | H-containing | `Unknown` | 121,593 |
| `32f44c499244b611:9` | `(2,1);(4,1)` | H-containing | `Unknown` | 121,619 |
| `d7e1b56c925b7f32:19` | `(-1,0);(-2,3)` | lifted H-containing | `Unknown` | 173,714 |
| `d7e1b56c925b7f32:19` | `(-1,0);(-1,2)` | lifted H-containing | `Unknown` | 173,714 |

There is no frozen non-domination witness: no completed admissible comparison
reversed a direction or split an equality. The exact position dumps for all
completed rows are emitted by `B2_PAIR`; the manifest contains their parent
replays and candidate coordinates. Cutoff rows are not witnesses.

## 4. Verdict

**NO b=2 proof round should be launched from this experiment, and the b=2
extension is not proved dead. The repaired experiment is NOT ADJUDICATED.**

This is the only truthful verdict. Zero counterexamples among eleven exact
d=3 children is not the protocol's option (a): all values were `Unknown`, the
first complete `F_3` audit did not finish, the mandatory d=4
quiet/frontier case did not finish, and Q0 did not qualify the only possible
d=3/d=4 discriminating value (`Loss`). Option (b) also does not apply because
there is no admissible reversed inequality.

The exact additional data that settles the campaign decision is:

1. a sealed Q0 manifest with four completed stock-reference `Loss` rows in
   each player-identity × phase bucket (16 total, stock nodes <=1,000,000),
   followed by matching fast-reference `Loss` results, one row per <=540 s
   invocation;
2. completed d=3 `F_d` values for every hitter first action in the four frozen
   K1 audit parents (the smallest outstanding case is
   `32f44c499244b611:9`, `F_3((2,1))`); and
3. completed d=4 values for every quiet/frontier-sensitive comparison admitted
   to those audits, including the frozen canonical lifted control.

One exact reversal after those gates makes the K1 b=2 spare-stone extension
**DEAD** and freezes that row. Zero reversals on the completed frozen matrix
would justify a proof round, whose exact conjecture would be:

> For every reachable nonterminal defender-`FirstStone` parent `P` with
> budget `b=2`, `!own_win_now(P)`, nonempty complete attacker count-4/5 family
> `T`, and `mhs(P)=1`, let `H(P)` be the legal one-cell transversal. For every
> finite stopped horizon `d` and every legal first action `c` outside `H(P)`,
> exhaustive second-stone minimax satisfies
> `F_d(c) <= max_{h in H(P)} F_d(h)` in defender order.

Consumption would require an enforceable branch on the **computed** pair
`(budget=2,mhs=1)` and would retain all H-first actions; it would not collapse
coverers. The existing proven b=1 prune remains separately fenced at
`(budget=1,mhs=1)`. The shipped K2 arm is the distinct
`(budget=2,mhs=2)` extendable-kernel case and receives no new theorem here.
The inherited coordinate-carrier check remains reject-not-certify at every
production boundary. No production code was changed in this round.

## 5. Regeneration commands

Run each cargo command only after the shown RAM gate; keep invocations serial.

```powershell
$freeGiB = Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
if ($freeGiB -le 9) { throw "Need >9 GiB free RAM; found $freeGiB" }
$env:CARGO_TARGET_DIR='.target-hunt'
cargo test -p hexfield_eq --release --no-run
```

```powershell
$freeGiB = Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
if ($freeGiB -le 9) { throw "Need >9 GiB free RAM; found $freeGiB" }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_DOM_B2_MANIFEST="$PWD/.codex-hunt/dom_b2_inventory.jsonl"
cargo test -p hexfield_eq --release dom_hunt_b2_inventory -- --ignored --nocapture --test-threads=1
Get-FileHash -Algorithm SHA256 $env:TSS_DOM_B2_MANIFEST
```

```powershell
$freeGiB = Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
if ($freeGiB -le 9) { throw "Need >9 GiB free RAM; found $freeGiB" }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_DOM_B2_FORCING_MANIFEST="$PWD/.codex-hunt/dom_b2_forcing_inventory.jsonl"
cargo test -p hexfield_eq --release dom_hunt_b2_forcing_inventory -- --ignored --nocapture --test-threads=1
cargo test -p hexfield_eq --release dom_hunt_b2_control_inventory -- --ignored --nocapture --test-threads=1
```

One completed-pair row:

```powershell
$freeGiB = Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
if ($freeGiB -le 9) { throw "Need >9 GiB free RAM; found $freeGiB" }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_DOM_B2_HASH='32f44c499244b611'
$env:TSS_DOM_B2_PREFIX='9'
$env:TSS_DOM_B2_FIRST='-2,1'
$env:TSS_DOM_B2_SECOND='4,1'
$env:TSS_DOM_B2_DEPTH='3'
$env:TSS_REFERENCE_FAST_TT_BYTES='536870912'
$env:TSS_REFERENCE_FAST_D6='1'
cargo test -p hexfield_eq --release dom_hunt_b2_pair_exact -- --ignored --nocapture --test-threads=1
```

One complete first-action aggregation uses the same variables without
`TSS_DOM_B2_SECOND` and invokes `dom_hunt_b2_exact`. The frozen smallest case
did not complete under the 540-second campaign allowance, so regeneration
must preserve `INCOMPLETE` unless a future cooperative fast-reference wrapper
returns an exact value inside the cap.

## 6. Adjudication (DOM-B2R2, 45-minute computations)

The three granted computations ran serially with a fresh `>9 GiB` RAM gate,
`.target-hunt`, release mode, a 512 MiB TT, D6 canonicalization enabled, and
`--test-threads=1`.  A test-only bounded fast-reference entry point propagates
`INCOMPLETE` and never caches it.  Thus the two 45-minute cutoffs below report
visited nodes rather than manufacturing `Unknown`.

### 6.1 Q0 stock/fast Loss qualification

Q0 ran for 2700.16 s.  It visited 3,593,600 stock nodes and 71,827,669 fast
nodes over the attempted rows.  Three rows qualified; thirteen of the sixteen
required bucket slots remained genuinely unqualifiable before the shared
deadline.  Q0 is therefore still failed closed and no b=2 attacker-`Loss`
value is admitted.

Every attempted candidate row was classified as follows:

| Player / phase | Position | d | Stock | Fast | Classification |
|---|---|---:|---|---|---|
| 1 / Second | `0f8c6bdfc55e7f6f:26` | 1 | `Unknown`, 387 nodes | not run | disqualified: stock is not `Loss` |
| 1 / Second | `0f8c6bdfc55e7f6f:26` | 2 | `Unknown`, 176,335 | not run | disqualified: stock is not `Loss` |
| 1 / Second | `0f8c6bdfc55e7f6f:26` | 3 | `Loss`, 1,159 | `Loss`, 70,610 | **qualified**, slot 1 |
| 1 / Second | `0f8c6bdfc55e7f6f:26` | 4 | `Loss`, 1,159 | `Loss`, 37,007,260 | **qualified**, slot 2 |
| 1 / First | `f9f50871d4efa2d9:33` | 1 | `Unknown`, 455 | not run | disqualified: stock is not `Loss` |
| 1 / First | `f9f50871d4efa2d9:33` | 2 | `Unknown`, 236,183 | not run | disqualified: stock is not `Loss` |
| 1 / First | `f9f50871d4efa2d9:33` | 3 | incomplete, 1,000,000 | not run | genuinely unqualifiable: stock node cap |
| 1 / First | `f9f50871d4efa2d9:33` | 4 | incomplete, 1,000,000 | not run | genuinely unqualifiable: stock node cap |
| 1 / Second | `f9f50871d4efa2d9:34` | 1 | `Unknown`, 454 | not run | disqualified: stock is not `Loss` |
| 1 / Second | `f9f50871d4efa2d9:34` | 2 | `Unknown`, 235,276 | not run | disqualified: stock is not `Loss` |
| 1 / Second | `f9f50871d4efa2d9:34` | 3 | `Loss`, 3,172 | `Loss`, 66,618 | **qualified**, slot 3 |
| 1 / Second | `f9f50871d4efa2d9:34` | 4 | `Loss`, 939,020 | incomplete, 34,683,181 nodes, 1319.876767 s | genuinely unqualifiable: fast deadline, slot 4 |

The required 16 manifest slots have this explicit disposition:

| Required slot | Disposition |
|---|---|
| player 0 / FirstStone / 1 | genuinely unqualifiable: no qualified row before 45-minute deadline |
| player 0 / FirstStone / 2 | genuinely unqualifiable: no qualified row before 45-minute deadline |
| player 0 / FirstStone / 3 | genuinely unqualifiable: no qualified row before 45-minute deadline |
| player 0 / FirstStone / 4 | genuinely unqualifiable: no qualified row before 45-minute deadline |
| player 0 / SecondStone / 1 | genuinely unqualifiable: no qualified row before 45-minute deadline |
| player 0 / SecondStone / 2 | genuinely unqualifiable: no qualified row before 45-minute deadline |
| player 0 / SecondStone / 3 | genuinely unqualifiable: no qualified row before 45-minute deadline |
| player 0 / SecondStone / 4 | genuinely unqualifiable: no qualified row before 45-minute deadline |
| player 1 / FirstStone / 1 | genuinely unqualifiable: no qualified row before 45-minute deadline |
| player 1 / FirstStone / 2 | genuinely unqualifiable: no qualified row before 45-minute deadline |
| player 1 / FirstStone / 3 | genuinely unqualifiable: no qualified row before 45-minute deadline |
| player 1 / FirstStone / 4 | genuinely unqualifiable: no qualified row before 45-minute deadline |
| player 1 / SecondStone / 1 | **qualified**: `0f8c6bdfc55e7f6f:26`, d=3 |
| player 1 / SecondStone / 2 | **qualified**: `0f8c6bdfc55e7f6f:26`, d=4 |
| player 1 / SecondStone / 3 | **qualified**: `f9f50871d4efa2d9:34`, d=3 |
| player 1 / SecondStone / 4 | genuinely unqualifiable: stock `Loss`, fast deadline |

The three-row qualified manifest is
`.codex-hunt/dom_b2_q0_adjudication.jsonl`, SHA-256
`C3A5AA3002F32A9C3A53E9EDF6505914C68F30FE450960F0DCEE2B0F1AB91BCA`.
The complete classification log has SHA-256
`F7096101F42018385B225325DAE14BA16C3E534A9D5601AB0AB218B4DBEB60BD`.

### 6.2 Exhaustive hitter `F_3`

The batch ran for 2700.18 s and 90,294,987 nodes.  The smallest completable
subset is all eight split hitters plus the H hitter of audit parent 1: 9/12
exact `F_3` values, all attacker `Unknown` (defender rank 1).  The second H
hitter exhausted the remaining wall; the final two rows received only the
immediate expired-deadline check.

| Parent | First / role | Exact `F_3` | Nodes | Wall s |
|---|---|---|---:|---:|
| `32f44c499244b611:9` | `(-2,1)` / split | `Unknown` (1) | 255,761 | 7.720922 |
| `32f44c499244b611:9` | `(4,1)` / split | `Unknown` (1) | 255,787 | 7.809392 |
| `19b085e7aa9f6215:9` | `(-1,0)` / split | `Unknown` (1) | 256,994 | 7.750645 |
| `19b085e7aa9f6215:9` | `(5,0)` / split | `Unknown` (1) | 256,994 | 7.870011 |
| `498a61ae0b5cf4ef:9` | `(-2,2)` / split | `Unknown` (1) | 257,130 | 7.799901 |
| `498a61ae0b5cf4ef:9` | `(4,-4)` / split | `Unknown` (1) | 257,118 | 7.749364 |
| `fd688f189544bf72:9` | `(-2,0)` / split | `Unknown` (1) | 256,994 | 7.844294 |
| `fd688f189544bf72:9` | `(4,0)` / split | `Unknown` (1) | 256,994 | 7.757757 |
| `32f44c499244b611:9` | `(2,1)` / H | `Unknown` (1) | 49,852,265 | 1485.892702 |
| `19b085e7aa9f6215:9` | `(3,0)` / H | `INCOMPLETE` | 38,388,948 | 1151.573559 |
| `498a61ae0b5cf4ef:9` | `(2,-2)` / H | `INCOMPLETE` (deadline already expired) | 1 | 0.000077 |
| `fd688f189544bf72:9` | `(2,0)` / H | `INCOMPLETE` (deadline already expired) | 1 | 0.000081 |

Audit parent 1 is the only completed first-action comparison: its two split
hitters and its H representative are equal at rank 1.  Parents 2--4 do not
adjudicate a first-stone direction because their H aggregate is incomplete.
The result JSONL SHA-256 is
`CD273BC6CC0B767B863FA7A1F0AD49F0E90666E6F42D65BC5611410D33BD882B`;
the log SHA-256 is
`BC0147752698B367BCB12D3FACCD3A91D59DCCC10DDA24EB3B2A89AF44771232`.

### 6.3 d=4 quiet/frontier comparisons

The batch ran for 2700.05 s and 74,068,629 nodes.  The smallest completable
subset is one child: audit parent 1's split pair is exact attacker `Unknown`
(rank 1).  Its first H-containing comparator did not finish in the remaining
wall, so no d=4 directional comparison completed.

| Parent | Ordered pair / class | d=4 result | Nodes | Wall s |
|---|---|---|---:|---:|
| `32f44c499244b611:9` | `(-2,1);(4,1)` / split | `Unknown` (1) | 62,601,245 | 2285.247044 |
| `32f44c499244b611:9` | `(2,1);(-2,1)` / H-containing | `INCOMPLETE` | 11,467,375 | 414.722415 |
| `32f44c499244b611:9` | `(2,1);(4,1)` / H-containing | `INCOMPLETE` (expired) | 1 | 0.000082 |
| `19b085e7aa9f6215:9` | `(-1,0);(5,0)` / split | `INCOMPLETE` (expired) | 1 | 0.000079 |
| `19b085e7aa9f6215:9` | `(3,0);(-1,0)` / H-containing | `INCOMPLETE` (expired) | 1 | 0.000068 |
| `498a61ae0b5cf4ef:9` | `(-2,2);(4,-4)` / split | `INCOMPLETE` (expired) | 1 | 0.000075 |
| `498a61ae0b5cf4ef:9` | `(2,-2);(-2,2)` / H-containing | `INCOMPLETE` (expired) | 1 | 0.000069 |
| `fd688f189544bf72:9` | `(-2,0);(4,0)` / split | `INCOMPLETE` (expired) | 1 | 0.000074 |
| `fd688f189544bf72:9` | `(2,0);(-2,0)` / H-containing | `INCOMPLETE` (expired) | 1 | 0.000067 |
| `d7e1b56c925b7f32:19` | `(-1,0);(-2,3)` / H-containing | `INCOMPLETE` (expired) | 1 | 0.000132 |
| `d7e1b56c925b7f32:19` | `(-1,0);(-1,2)` / H-containing | `INCOMPLETE` (expired) | 1 | 0.000119 |

The result JSONL SHA-256 is
`CE60D033C3FFC4915C151224314776EB5471C9C8F5CDC9202CB293E286676BC1`;
the log SHA-256 is
`06BE3FA9E5C7671249A97630CAE0CC026F80E086AF4FB022D2AA74D81C20477E`.

### 6.4 Binding binary verdict

**PROOF-ROUND-READY; not DEAD.**  There is no reversal in any completed
repaired comparison.  In particular, the one complete `F_3` parent has
split/H equality, and no completed d=4 comparison supplies a reversed
inequality.  Therefore there is no witness to freeze.  This verdict follows
DOM-B2R2's binding zero-reversal rule even though Q0, three H aggregates, and
ten d=4 children remain failed closed after their full extended budgets; it
does not relabel any incomplete row as evidence.

The proof-round conjecture is precisely:

> For every reachable nonterminal defender-`FirstStone` parent `P` with
> budget `b=2`, `!own_win_now(P)`, nonempty complete attacker count-4/5 family
> `T`, and `mhs(P)=1`, let `H(P)` be the legal one-cell transversal. For every
> finite stopped horizon `d` and every legal first action `c` outside `H(P)`,
> exhaustive second-stone minimax satisfies
> `F_d(c) <= max_{h in H(P)} F_d(h)` in defender order.

Any consumer must branch on the **computed** pair `(budget=2,mhs=1)` and
retain all H-first actions.  It must not collapse coverers.  The proven b=1
prune remains separately fenced at `(budget=1,mhs=1)`, and the shipped K2 arm
remains the distinct `(budget=2,mhs=2)` path.  No production code changed.

### 6.5 Adjudication regeneration commands

After the existing pre-build command in Section 5, run these commands
serially.  Q0 exits nonzero when fewer than 16 rows qualify; that is the
intended failed-closed assertion.

```powershell
$freeGiB = Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
if ($freeGiB -le 9) { throw "Need >9 GiB free RAM; found $freeGiB" }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_DOM_B2_DEADLINE_MS='2700000'
$env:TSS_DOM_B2_Q0_PER_BUCKET='4'
$env:TSS_DOM_B2_Q0_NODE_CAP='1000000'
$env:TSS_REFERENCE_FAST_TT_BYTES='536870912'
$env:TSS_REFERENCE_FAST_D6='1'
$env:TSS_DOM_B2_Q0_MANIFEST="$PWD/.codex-hunt/dom_b2_q0_adjudication.jsonl"
cargo test -p hexfield_eq --release dom_hunt_b2_q0 -- --ignored --nocapture --test-threads=1
```

```powershell
$freeGiB = Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
if ($freeGiB -le 9) { throw "Need >9 GiB free RAM; found $freeGiB" }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_DOM_B2_DEADLINE_MS='2700000'
$env:TSS_REFERENCE_FAST_TT_BYTES='536870912'
$env:TSS_REFERENCE_FAST_D6='1'
$env:TSS_DOM_B2_F3_RESULTS="$PWD/.codex-hunt/dom_b2_f3_adjudication.jsonl"
cargo test -p hexfield_eq --release dom_hunt_b2_adjudicate_f3 -- --ignored --nocapture --test-threads=1
```

```powershell
$freeGiB = Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
if ($freeGiB -le 9) { throw "Need >9 GiB free RAM; found $freeGiB" }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_DOM_B2_DEADLINE_MS='2700000'
$env:TSS_REFERENCE_FAST_TT_BYTES='536870912'
$env:TSS_REFERENCE_FAST_D6='1'
$env:TSS_DOM_B2_D4_RESULTS="$PWD/.codex-hunt/dom_b2_d4_adjudication.jsonl"
cargo test -p hexfield_eq --release dom_hunt_b2_adjudicate_d4 -- --ignored --nocapture --test-threads=1
```
