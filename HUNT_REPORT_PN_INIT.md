# NQ6 machine hunt — interior census pruning and PN/DN initialization

Status: **COMPLETE — gated campaign PASS, zero soundness findings.** The proof
contract and measurement definitions below were frozen before the campaign
results were examined. The single release test finished in 50.24 s after a
13.12 s clean-target build.

## Soundness boundary

The only executable gate measured here is Contract 8.1/8.2 from the reviewed
`dtw-census-two-gap/round-1` proof at
`ffdd414ad5197444eef44af4f28da376a5d95507`:

- attacker is the current player at a nonterminal `FirstStone` or
  `SecondStone` node;
- census `c` is the maximum attacker count over defender-free windows from a
  complete `WindowStore::entries()` scan, with `ac > 0`, `dc == 0`, and zero
  fallback;
- the phase-specific placement formula and ply table are used exactly;
- coordinates pass the proof's checked-wide conservative safety guard; and
- a requested WIN arm is dismissed only under the strict test
  `LB_plies > h_rem`.

No LOSS/opponent-refutation arm is gated. `SecondStone c=3` receives no extra
gap: its reachable ply-5 counterexample remains binding. Threat-only and local
window indexes are not used.

Every actual gated node is retained in the shadow trace with its full sorted
board identity. A final `pn=0`/positive certificate below that node is a
**SOUNDNESS FINDING** and stops the campaign.

## Frozen measurement design

The broad solves use a requested relative WIN horizon of 16. This makes
interior `h_rem` run from 16 to zero: the proved gate is evaluated at
`h_rem <= 8`, while conjectural phase-specific `(h,c)` screens cover horizons
9–16. The cohorts follow NQ4 exactly otherwise:

- all 19 forcing roots, cold at node caps 10,000 and 100,000;
- `double_fork_compact` under its C1 round-3 consume profile, cap 100,000 and
  absolute semantic horizon 45;
- 100 human-corpus `FirstStone` roots at placement >=20, Fisher–Yates sampled
  with seed `0x9E3779B97F4A7C15`, cap 10,000.

`subtree-expansion fraction` is the union of actual expansion events whose
first-expansion ancestry passes through an actual gated node, divided by all
recorded attempt expansions. It includes the gated expansion itself and avoids
double-counting nested gates. This is a deterministic trace counterfactual,
not a live gated rerun; transpositions reachable through another parent may
reduce the live saving.

The PN candidates are shadow-only claimant-window statistics computed for
fresh nodes:

1. number of live claimant windows at count >=4;
2. number at count >=3; and
3. deterministic greedy packing of count-4 windows with pairwise-disjoint
   two-cell gaps.

Higher counts receive lower PN rank. Correlation uses unique wide-PN entries
whose final trace label is proven or refuted. The node-count replay fixes those
observed outcomes and subtree costs, then changes only OR-child order. It is an
outcome-labelled replay, not a live df-pn run: changed ordering can change TT
reuse, thresholds, generated frontier, and cap exits.

## Results

### Headline

All percentages are prospective deterministic trace replays, not achieved live
speedups.

| lever / cohort | forcing 10k | forcing 100k | double_fork_compact | 100 human roots | measured cost / risk |
|---|---:|---:|---:|---:|---|
| **proved interior gate, gated eligible nodes** | 20,119 / 22,601 (**89.0%**) | 76,392 / 79,282 (**96.4%**) | 0 / 221 (0%) | 9,769 / 12,193 (**80.1%**) | full `entries()` scan mean **0.80 / 0.69 / 0.51 / 1.26 µs** |
| **proved gate, union trace-subtree expansions** | 73,842 / 89,405 (**82.6%**) | 285,181 / 324,163 (**88.0%**) | 0 / 408 (0%) | 41,894 / 78,970 (**53.1%**) | proof risk low; integration must remain WIN-only |
| best uncontradicted stronger target, FS `(h=9,c<=2)` | +92 (**0.103%**) | +92 (**0.028%**) | 0 | 0 | same census scan; new proof required |
| PN seed `live_ge3`, outcome-labelled solved-root replay | 1,093→355 (**67.5%**) | 1,093→355 (**67.5%**) | n/a (narrow path) | 652→408 (**37.4%**) | replay divergence high; global rho is slightly negative |
| PN seeds `live_ge4` / disjoint two-gap | 0% / 1.3% replay saving | 0% / 1.3% | n/a | 0% / 0% | empirically worthless here |

The gate's `subtree_saved` includes each gated expansion and all later
first-ancestry descendants, with nested gates counted once. Descendant-only
counts are therefore 53,723 at forcing 10k, 208,789 at forcing 100k, and 32,125
on human roots. This does **not** assert that a live gated solver will save the
same number: another transposition parent can still reach work that the trace
first reached below a gated node, and changed PN/DN values change selection.

### 1. Interior gate

| cohort | roots W/U | attempt expansions | eligible WIN-arm nodes at `h_rem<=8` | gated | subtree union | census scan mean / median / p95 µs |
|---|---:|---:|---:|---:|---:|---:|
| forcing 10k | 3 / 16 | 89,405 | 22,601 | 20,119 | 73,842 | 0.799 / 0.700 / 1.300 |
| forcing 100k | 3 / 16 | 324,163 | 79,282 | 76,392 | 285,181 | 0.685 / 0.500 / 1.600 |
| double_fork_compact | 1 / 0 | 408 | 221 | 0 | 0 | 0.510 / 0.500 / 0.600 |
| human 100, 10k | 23 / 77 | 78,970 | 12,193 | 9,769 | 41,894 | 1.264 / 1.100 / 3.500 |

The high interior rate is not merely the leaf `h=8,c<=2` predicate. The C1
atomic-turn frontier visits FirstStone horizons 0, 4, 8, 12, 16 (plus a few
partial-turn 1/5/9/13 nodes). At `h=0,1,4,5`, the exact phase formula also
gates many `c=3` nodes. At exact `h=8`, the expected `c<=2` split is visible:

| cohort | exact-h=8 FS nodes | exact-h=8 gated (`c<=2`) | exact-h=8 `c=3` not gated |
|---|---:|---:|---:|
| forcing 10k | 4,120 | 1,638 | 2,482 |
| forcing 100k | 5,052 | 2,162 | 2,890 |
| human 100 | 3,326 | 902 | 2,424 |

`double_fork_compact` is the useful negative control. Its C1 round-3 consume
solve runs through narrow compatibility and proves WIN in 409 solver nodes, but
none of its 221 eligible census evaluations passes the proved gate.

#### Soundness cross-check

Every returned root certificate was independently accepted by `TssVerifier`.
The trace then checked every actual gate against its entry's final positive
label. There were zero gated `pn=0`/positive entries. On solved roots this
covered 197 gate events at each forcing rung and 82 on the human cohort;
`double_fork_compact` had no gates. No forcing NO row returned WIN, and a
WIN-only solve never returned LOSS.

The on/off tripwire was exact:

```text
PNI_IDENTITY id=0hz3hty status=UNKNOWN nodes=9302 tt_hits=2872
             expansions=9301 result=PASS
```

### 2. Actual slack and stronger-bound counterfactuals

The recorded pair is `(Contract-8.1 LB_plies, h_rem)` at each claimant/WIN-arm
expansion. Representative FirstStone rows show where the proved bound stops:

| cohort / exact `h_rem` | census histogram `[c0..c5]` | `LB>h` | `LB=h` | `LB<h` |
|---|---:|---:|---:|---:|
| forcing 10k / 8 | `[0,0,1638,2482,0,0]` | 1,638 | 0 | 2,482 |
| forcing 10k / 9 | `[0,0,28,59,0,0]` | 0 | 28 | 59 |
| forcing 10k / 12 | `[0,0,155,401,0,0]` | 0 | 0 | 556 |
| forcing 100k / 8 | `[0,0,2162,2890,0,0]` | 2,162 | 0 | 2,890 |
| human / 8 | `[0,2,900,2424,0,0]` | 902 | 0 | 2,424 |
| human / 12 | `[0,3,273,742,0,0]` | 0 | 0 | 1,018 |

The hypothetical `(h,cmax)` screen means: suppose a new phase-specific theorem
certified every actual census `c<=cmax` through horizon `h`. The saving below is
incremental beyond the already-proved gate. A “positive collision” is a node
that the screen would dismiss but the baseline PN trace labels positive; it is
not allowed to graduate to a proof target without freezing and independently
verifying those subpositions.

| proposed FirstStone theorem | forcing 10k | forcing 100k | compact | human | disposition |
|---|---:|---:|---:|---:|---|
| `(9,c<=2)` | +92, 0 collisions | +92, 0 | 0 | 0 | only nonzero broad target; **tiny** |
| `(12,c<=1)` | 0, 0 | 0, 0 | 0 | +3, 0 | empirically worthless |
| `(16,c<=1)` | 0, 0 | 0, 0 | 0 | +4, 0 | empirically worthless; very ambitious theorem |
| `(9,c<=3)` | +11,026, **23** collisions | +34,072, **23** | +306, **136** | +19,187, **45** | reject; compact is a verified WIN control |
| `(12,c<=2)` | +1,269, **2** | +1,269, **2** | 0 | +1,288, **12** | reject from proof queue |

All `cmax>=3` rows inherit the same or more collisions. Horizons 13–16 at
`cmax=2` also collide. The only pair worth even a sharply time-boxed proof
attack is **FirstStone `(h=9,c<=2)`**, and its ceiling is only 92 expansions on
the forcing cohort (0.028–0.103%) and zero on the human sample. On value alone,
it should not displace implementation of the proved gate.

SecondStone coverage is insufficient for a positive proof recommendation: the
wide atomic-turn search produced only one claimant SecondStone row at `h=16`
(`c=3`), while the compact root is SecondStone `h=9,c=4`. The phase-specific
formula was applied correctly, but no `c<=2` SecondStone counterfactual fired.
The frozen reachable `SecondStone c=3` ply-5 counterexample from the proof
record remains decisive; this hunt does not reopen it.

### 3. PN/DN initialization shadow

The correlations below are Spearman rho between the raw statistic and
`proven=1` on unique wide-PN entries with a final proven/refuted label. Positive
rho would support “more windows => more likely proof.” All observed values are
slightly negative:

| cohort | classified / unique wide nodes | rho count>=4 | rho count>=3 | rho disjoint two-gap |
|---|---:|---:|---:|---:|
| forcing 10k | 17,943 / 89,277 | -0.0286 | -0.0106 | -0.0362 |
| forcing 100k | 53,720 / 324,035 | -0.0170 | -0.0093 | -0.0225 |
| human 100 | 20,718 / 78,747 | -0.0601 | -0.0351 | -0.0793 |

Despite that, count>=3 improves the outcome-labelled OR-order replay on the 3
forcing and 16 non-immediate human solved roots: 1,093→355 nodes and 652→408
nodes respectively. Count>=4 is exactly neutral; disjoint two-gap saves 14/1,093
forcing replay nodes and zero human nodes. Seven additional human WINs close at
the solver's pre-attempt immediate-root seam and therefore have no PN replay.
`double_fork_compact` uses narrow compatibility, so PN seeding is correctly
reported n/a rather than attributed a fabricated wide-PN order.

This conflict—good solved-root replay but negative population correlation—is
why count>=3 is **not a build-ready performance claim**. The replay fixes final
outcomes and observed subtree costs. A live seed changes df-pn thresholds,
interleaving, transposition reuse, staged reopenings, and which nodes exist
before the cap. The next admissible step is a test-only live on/off order trial
with the same identity/soundness gates, not a production seed change. Counts
>=3 and >=4 can piggyback on the mandated gate census only inside its evaluated
horizon; seeding every fresh node would require scans outside that domain whose
incremental cost was not isolated here. The greedy disjoint-gap packing also
requires extra empty-cell work and has no measured value.

## Risk-adjusted ranking and recommendation

| rank | lever | prospective saving | proof / implementation risk | decision |
|---:|---|---:|---|---|
| 1 | **proved interior gate at `h_rem<=8`** | 53.1–88.0% trace-subtree coverage on broad cohorts | low theorem risk; medium integration risk around WIN-only artifact/PN semantics | **BUILD FIRST** |
| 2 | count>=3 PN initialization | 37–68% on a small outcome-labelled solved-root replay; weak negative global rho | low game-soundness risk if ordering-only, high performance-divergence risk | live test-only A/B after gate; do not ship from replay |
| 3 | stronger census proof | best viable pair adds 0–0.103%; larger pairs collide with positives | high proof risk for negligible uncontradicted value | do not schedule a broad proof round |

**Recommendation: build the proved interior gate first.** Integrate it only at
claimant/current-player WIN nodes, preserve the strict requested-horizon result,
and run a live gated identity/soundness campaign because the trace saving is an
upper estimate under changed PN values. Do not gate universal defender arms or
turn a bounded no-WIN result into global LOSS. After that build, a small
test-only live `live_ge3` order A/B is the only follow-up with enough replay
signal to justify measurement. The stronger-bound proof round is dominated.

## Gate evidence

- reviewed proof read in full before implementation;
- release campaign: `1 passed; 0 failed`, 113 filtered;
- campaign 50.24 s; clean-target build 13.12 s;
- free RAM immediately before Cargo: 12.404 GB;
- on/off status, node, and TT-hit identity: PASS;
- every returned certificate: verifier ACCEPTED;
- forcing NO→WIN anomalies: 0;
- WIN-vs-LOSS anomalies: 0;
- actual-gate soundness findings: 0.

## Regeneration

Run from the worktree root in PowerShell. Recheck free RAM before every Cargo
invocation and wait/recheck below 9 GB:

```powershell
Get-CimInstance Win32_OperatingSystem | % {
    $_.FreePhysicalMemory/1MB
}
$env:CARGO_TARGET_DIR='.target-hunt'
cargo test --release -p hexfield_eq pn_init_campaign -- `
    --ignored --test-threads=1 --nocapture

rustfmt --edition 2021 --check `
    packages/hexfield_eq/rust/src/tss_solver.rs `
    packages/hexfield_eq/rust/src/tss_pn_init_hunt.rs
```

The recorded raw output is `.codex-hunt/pn-init-cargo.log`. No commits were
made.
