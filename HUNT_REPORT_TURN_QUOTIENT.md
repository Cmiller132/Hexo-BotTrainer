# NQ4 machine hunt — search-space quotients beyond position transposition

Status: **COMPLETE — gated run PASS, no anomalies.** The single release run
finished in 106.48 s (119.6 s including the clean-target build), verified every
returned certificate, and reported `TQ_DONE result=PASS anomalies=0`.

## Scope and method

Engine: `hunt/turn-quotient` at `2430fc47` (C1-unified solver). The harness is
`packages/hexfield_eq/rust/src/tss_turn_quotient_hunt.rs`, registered test-only
from `lib.rs`. Solver counters are also `#[cfg(test)]` and require
`TSS_TURN_QUOTIENT_TELEMETRY`; production builds contain none of the telemetry.

Deterministic roots:

- all 19 forcing-corpus positions, cold at 10,000 and 100,000 nodes;
- `double_fork_compact`, C1 round-3 consume profile, cap 100,000 and absolute
  semantic horizon 45;
- 100 human-corpus FirstStone roots at placement >=20, sampled from the local
  6,902-game corpus by the established Fisher–Yates stream with seed
  `0x9E3779B97F4A7C15`, cap 10,000.

The gate independently verifies every returned certificate, rejects a WIN on
any forcing-corpus NO row, stops on every WIN/LOSS flip across rungs, and first
runs one cold forcing row telemetry-off vs telemetry-on with exact status, node,
and TT-hit identity assertions.

## Definitions (fixed before seeing results)

**Baseline.** `nodes` is the solver's expansion counter. Both indexed TT entries
and retained PN arena entries are reported because a full wide index can retain
unindexed frontier nodes. `tt_hits` is the engine counter, not a reconstruction.

**D6.** The shadow key decodes the exact wide position key, transforms stones
and a pending `SecondStone.first` under all 12 D6 images using the solver/cert
coordinate action, sorts, re-encodes, and selects the lexicographic minimum.
Duplicates are counted only when canonical equality joins distinct raw keys.
Expanded-node denominator is unique raw expanded positions, so staged reopening
of the same raw cutoff cannot masquerade as a D6 saving. Timings include all 12
images and exact re-encoding.

**Horizon.** The production wide TT is already keyed by position alone; it does
not create `(position, clock)` entries. To answer the requested counterfactual,
the harness snapshots the retained frontier after every staged-deepening rung
and replays those observations as if the TT were clock-keyed. An exact-clock
miss is monotone-settleable only when an earlier sound WIN used no more clock,
or an earlier sound refutation used at least as much clock. A `DepthCutoff` is
never a refutation. Branch verdicts are recomputed bottom-up from genuine child
verdicts, rather than trusting `dn=0`, because staged cutoffs temporarily carry
`dn=0` and are reopened by the engine.

**Commutation.** At each unique expanded position with two completed preceding
two-stone turns, the pair is INTERACTING if any of these hold (categories may
overlap):

1. a placement from each turn lies in a common length-six window;
2. the later turn's ordered placements would lack radius-eight legality support
   after removing both turns (earlier placements of that later turn may support
   later placements);
3. either turn occupies a cell in the opponent's pre-turn count>=3 window
   (later-turn stones are subtracted when reconstructing the earlier view).

The independent fraction is an *upper bound* on removable interiors, not a
sound pruning rate. Consecutive turns belong to opposite players, so the game
quantifiers are `exists attacker; forall defender`, not two freely permutable
loops. Endpoint equality alone cannot justify canonical turn order.

## Measurements

### Headline

| lever / exposed class | forcing 10k | forcing 100k | double_fork_compact | 100 human roots | incremental saving if built |
|---|---:|---:|---:|---:|---:|
| D6 duplicate TT entries | 0 / 267,457 | 0 / 1,283,238 | 0 / 258 | 0 / 259,824 | **0%** |
| D6 duplicate expanded states | 0 / 100,052 | 0 / 419,151 | 0 / 357 | 0 / 89,179 | **0%** |
| clock-only entries avoided by current position key | 771,047 / 1,034,795 (74.5%) | 5,363,378 / 6,640,486 (80.8%) | 0 (narrow, one clock) | 376,805 / 604,978 (62.3%) | **already realized** |
| monotone-settleable clock misses | 5,451 (0.527 pp) | 71,514 (1.077 pp) | 0 | 1,587 (0.262 pp) | only for a hypothetical clock-keyed/persistent cache |
| independent consecutive-turn interiors | <=162 / 100,051 (0.162%) | <=162 / 419,150 (0.039%) | <=10 / 185 (5.405%) | <=138 / 89,179 (0.155%) | upper bound; not yet sound pruning |
| **never-expanded retained wide entries** | **167,405 / 267,457 (62.6%)** | **864,087 / 1,283,238 (67.3%)** | n/a (direct-mapped narrow TT) | **170,645 / 259,824 (65.7%)** | **largest new TT/work lever** |

The requested quotients do not contain a large unbuilt win. D6 is exactly zero
on these per-root searches. The horizon quotient is large but the engine already
implements the strongest representation — one retained entry per position,
reopened in place. Consecutive-turn independence is tiny outside the compact
quiet witness and is adversarially unsafe without a new theorem. The hunt's
largest implementable finding is instead a representation quotient: keep
unselected generated children as edge thunks rather than eagerly allocating and
indexing a PN arena entry.

### 1. Baseline telemetry

| cohort | roots | verdicts W/L/U | nodes | indexed TT entries | TT hits | entries / unique expanded |
|---|---:|---:|---:|---:|---:|---:|
| forcing 10k | 19 | 8 / 2 / 9 | 100,300 | 267,457 | 30,667 | 2.673 |
| forcing 100k | 19 | 12 / 2 / 5 | 419,445 | 1,283,238 | 129,266 | 3.062 |
| double_fork_compact | 1 | 1 / 0 / 0 | 409 | 258 | 51 | narrow TT; not comparable |
| 100 human roots, 10k | 100 | 24 / 3 / 73 | 89,544 | 259,824 | 81,336 | 2.914 |

The forcing 10k→100k ladder has no WIN/LOSS flip. Four additional expected WIN
rows close at 100k; the five forcing NO controls never return WIN. The two LOSS
rows at both rungs are immediate certified losses. `double_fork_compact` is the
expected verified WIN in 409 nodes.

The required behavior tripwire is exact:

```text
TQ_IDENTITY id=0hz3hty status=WIN nodes=2412 tt_hits=2263 result=PASS
```

### 2. Horizon quotient and staged semantics

| cohort | shadow `(position,clock)` entries | distinct positions observed | positions at >1 clock | clock-only excess | monotone hits / clock misses |
|---|---:|---:|---:|---:|---:|
| forcing 10k | 1,034,795 | 263,748 | 138,195 (52.4%) | 771,047 (74.5%) | 5,451 / 1,034,795 (0.527%) |
| forcing 100k | 6,640,486 | 1,277,108 | 572,084 (44.8%) | 5,363,378 (80.8%) | 71,514 / 6,640,486 (1.077%) |
| double_fork_compact | 357 | 357 | 0 | 0 | 0 |
| human 100 | 604,978 | 228,173 | 144,048 (63.1%) | 376,805 (62.3%) | 1,587 / 604,978 (0.262%) |

Interpretation matters: these are **shadow entries in a counterfactual
clock-keyed TT**, not duplicates in the production TT. Production stores one
position entry and mutates/reopens it over stages, so its actual count of
clock-only duplicate entries is zero. This existing quotient avoids 62–81% of
the entries that a naive clock-keyed design would retain.

The replay visits globally increasing stage clocks. Therefore every prospective
monotone hit observed here is the positive direction: a sound proof already
obtained with less remaining clock settles a later query with more clock. A
refutation discovered at a smaller stage cannot settle a larger-stage query;
the reverse transfer is valid in principle but this chronological replay offers
no such query. Exact-clock hits are zero because each retained-stage snapshot
advances the clock.

The sound transfers are exactly:

- `WIN within h` transfers to any `h' >= h`;
- a **complete restricted-search refutation** with allowance `h'` transfers to
  `h <= h'`;
- typed terminal facts transfer only when their exact resolution label remains
  within the queried semantic horizon;
- imported zone proofs additionally require the existing `resolution_t`,
  `zone_build_t`, rebase, and verifier-preflight checks.

The following transfer nowhere: UNKNOWN; node/TT/certificate-cap exit;
unexpanded prior; partial branch; or staged `DepthCutoff`. A cutoff sits beyond
the current stage (`entry.depth > stage_depth`) and is excluded before it enters
the replay query stream. This is why no cutoff-derived `dn=0` appears as a
refutation, and why the output's in-domain `staged_cutoffs_excluded` counter is
zero. The engine's `reopen_depth_cutoffs` plus bottom-up refresh remains
mandatory: `dn=0` by itself is not a disproof.

### 3. D6 at the search TT

| cohort | TT duplicate fraction | expanded duplicate fraction | canonicalization µs / generated key | amortized µs / unique expansion |
|---|---:|---:|---:|---:|
| forcing 10k | 0% | 0% | 11.879 | 31.754 |
| forcing 100k | 0% | 0% | 15.033 | 46.024 |
| double_fork_compact | 0% | 0% | 6.498 | 7.426 |
| human 100 | 0% | 0% | 29.282 | 85.312 |

There is no prospective state saving to buy. The cost rises on the human sample
because those positions have longer histories/more stones to transform and sort.
Installing 12-image canonicalization on the search hot path would add material
work for zero hits on every measured cohort. The cert layer's D6 support is still
valuable for proof transport; this result is specifically about within-root
search-TT duplication.

### 4. Consecutive-turn commutation

| cohort | eligible interiors | independent | shared-window interaction | legality coupling | count>=3 threat/response interaction |
|---|---:|---:|---:|---:|---:|
| forcing 10k | 100,051 | 162 (0.162%) | 56.543% | 0% | 99.760% |
| forcing 100k | 419,150 | 162 (0.039%) | 49.101% | 0% | 99.943% |
| double_fork_compact | 185 | 10 (5.405%) | 56.757% | 0% | 93.514% |
| human 100 | 89,179 | 138 (0.155%) | 31.145% | 0% | 99.809% |

Interaction columns overlap. Radius-eight legality coupling never fires in this
sample; threat/response coupling removes essentially every candidate. Counting
one removable interior for every independent pair gives the displayed upper
bound. It is deliberately not called an achieved duplicate count: the alternate
ordering belongs to the other player at the wrong game turn and may not be a
reachable search path.

A canonical order at an attacker OR node is **not sound from these measurements**.
The proof round would owe a strategy-preserving diamond: for every defender
reply, swapping the local subproblems must preserve legal ownership/turn phase,
terminal timing, forcing-generator membership, hitting sets, and a bijection of
all subsequent defender responses, with the `exists attacker; forall defender`
quantifiers in the same order. Equal endpoint boards prove none of that.

### 5. Additional wasted class: eager unexpanded frontier entries

The wide engine indexes defender children during generation. Consequently it
retains 2.7–3.1 entries per unique expanded position, and 63–67% of indexed
entries are never expanded before proof or cap termination. This is not a game
state equivalence, but it is a directly implementable **frontier representation
quotient**: an unselected pending child can remain `(move, result, prior, exact
key thunk)` on its parent edge and acquire an arena/TT entry only when selected.

This does not promise a 63–67% node reduction — those entries are not counted as
expansions today. It does size a large reduction in key construction, hash-map
insertion, retained arena records, and TT pressure. It also targets the dominant
measured class while preserving the current position quotient.

## Recommendation and risk-adjusted ranking

| rank | build candidate | measured prospective saving | implementation / proof risk | decision |
|---:|---|---:|---|---|
| 1 | **lazy pending-child / frontier-admission quotient** | 62.6–67.3% of wide retained entries were never expanded | medium engineering; low game-soundness risk with refinement lemma | **BUILD FIRST** |
| 2 | monotone horizon-aware persistent proof lookup | +0.262–1.077 pp on counterfactual clock misses; current within-solve quotient already saves 62–81% | high semantic risk around staged and zone clocks | retain as a later proof-cache optimization |
| 3 | canonical order for independent consecutive turns | <=0.039–0.162% on broad cohorts; 5.405% on one compact witness | very high: adversarial quantifier theorem | do not implement before proof |
| 4 | D6 search-TT canonicalization | 0% state saving, positive 6.5–29.3 µs/key cost | medium | do not build |

**Recommendation: build the lazy pending-child/frontier-admission quotient
first.** It has by far the largest unbuilt measured TT/work leverage and avoids
the unsound adversarial reorder.

The required theorem is a **Lazy-Frontier Refinement Lemma**: replacing an
unselected pending child entry by a thunk that stores the same move/result/prior
and exact future position key preserves `child_numbers`, selection order, and
all parent PN/DN recomputations; on first selection, realization produces the
identical state/key and links to any existing transposition before expansion.
Thus eager and lazy frontiers have the same reachable PN fixed points and
materializable certificates. A cap-aware corollary must state honestly which
telemetry/traversal differences TT admission timing may cause; proof validity
must remain invariant even if capped UNKNOWN timing changes.

For completeness, the other theorem obligations remain:

- **D6 TT:** prove D6 covariance of the full search state (including pending
  first stone, player/phase, ordering-independent threat/legality stores, PN
  priors, and certificate remapping). This is the smallest conventional
  quotient; the cert layer already supplies the coordinate action.
- **Horizon:** WIN transfer is only `WIN within h => WIN within h'` for `h' >= h`.
  A complete restricted-search refutation transfers from a larger allowance to
  a smaller one. UNKNOWN, node-cap exits, unexpanded nodes, and staged
  `DepthCutoff`/its inherited `dn=0` transfer nowhere. Zone fragments add a
  build-horizon direction check and must retain the engine's existing rebase and
  preflight rules.
- **Commutation:** prove a strategy-preserving diamond/bisimulation, not merely
  equal endpoints: legal sets, forcing generator membership, threat/hitting
  analyses, terminal timing, and every defender response must correspond while
  preserving the alternating existential/universal quantifiers. Without that
  theorem, an OR-side canonical order can delete the only winning response to a
  defender choice.

## Gate evidence

- release build: PASS (one warning for an unused harness import; removed
  mechanically afterward, with no behavioral code change and no second Cargo
  invocation);
- test: `1 passed; 0 failed`, 113 filtered;
- campaign time: 106.48 s; clean-target build: 12.60 s;
- node identity: PASS (`0hz3hty`, 2,412 nodes and 2,263 hits both ways);
- certificate verification: PASS for every returned WIN/LOSS;
- WIN/LOSS anomalies: 0;
- forcing NO→WIN anomalies: 0.

## Regeneration

The one permitted command (after the required >9 GB free-RAM check) is:

```powershell
Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:CARGO_TARGET_DIR='.target-hunt'
cargo test --release -p hexfield_eq turn_quotient_campaign -- --ignored --test-threads=1 --nocapture 2>&1 `
  | Tee-Object -FilePath .codex-hunt/turn-quotient-cargo.log
```

No commits are made by this hunt.
