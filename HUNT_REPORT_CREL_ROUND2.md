# C-REL round 2: strict-discharge warm-template-cache shadow hunt

Date: 2026-07-18  
Branch / HEAD: `hunt/cert-support` / `450c41e4e53b582b379456eb3efcafa270e42bba`  
Scope: Stages 1-3 only; shadow-only; no production hard values  
Binding design: `DESIGN_C_REL.md` at `408dc5b6`  
Retained comparator: `HUNT_REPORT_CERT_SUPPORT.md` at `3cd224fe`

## Verdict

| Stage | Verdict | Stop-criterion result |
|---|---|---|
| 1. Consolidated shadow reproduction | **PASS** | No hard result was emitted without strict acceptance (`0` hard-mint calls). The full-certificate path produced `648` strict-accepted cross-root candidates beyond exact root equality, so the no-cross-root-acceptance stop criterion did not fire. |
| 2. Interface recorder and confusion matrix | **PASS** | Projection classes produced `381` cross-root matches. Matching preserved the O12 per-target acceptance indicator on all `350` unconditionally hittable targets and saved `6.000819 s` after charging `E+L+I`; neither Stage-2 stop criterion fired. |
| 3. Six fixed negative-control fixtures | **PASS** | H1-H6 each met its named condition. No case returned or installed hard evidence without strict acceptance. |

**Stage-4 economics are justified as the next experiment.** Stages 1-3 close
green, cross-root strict acceptance exists, and the finite interface has useful
selectivity. This is not an economics verdict: matched `S0/SR`, fixed-total-RAM
accounting, process RSS, and the source-clustered confidence interval remain
unmeasured and belong only to Stage 4.

## Safety and implementation boundary

All implementation is in the existing `cfg(test)`-only
`packages/hexfield_eq/rust/src/cert_support_hunt.rs` module.
`tss_verify.rs` is untouched. There are no production-path changes and no call
to `hard_value_from_verified`.

The test recorder:

- admits only a source certificate accepted by the unchanged `TssVerifier`;
- replays every reachable leaf occurrence and converts event clocks with
  checked arithmetic, rejecting saturation, underflow, disagreement, and
  ambiguous shared occurrences;
- records the exact v1 root projection, node-specific source-rederived zone
  hints, and per-node WF witnesses;
- cross-checks its rederived zone multiset against the verifier module's
  existing test-only zone rederivation helper;
- materializes a fresh complete `RootBinding::from_state(target)`, applies only
  D6 coordinate mapping and checked relative-clock translation, and then calls
  the unchanged strict verifier;
- evaluates `HintMatch` and bypasses it in a parallel shadow arm for Stage 2;
- uses SHA-256 of the canonical v1 test payload for candidate order (the
  standard `abc` digest vector is asserted before the campaign); and
- keeps negative probes under full payload, complete root binding, D6 action,
  status, absolute horizon, and materializer-contract identity.

The exact cohort happened to contain no zoned Universal declaration, so its 48
recorded interfaces have zero `zone_hints`. The zone extraction and
correspondence guard are implemented and exercised structurally, but this run
does not provide positive zoned-certificate coverage.

## Frozen acquisition

The complete retained acquisition manifest was used without replacement:

| Source class | Frozen roots | Strict certificates admitted | Outcome details |
|---|---:|---:|---|
| Official forcing plus `double_fork_compact` | 15 | 12 WIN | `0l4291i_live`, `lz60mfb`, and `double_fork_compact` remained unavailable at the frozen 10k/100k ladder. |
| Deterministic human roots | 200 | 34 WIN | 166 remained `UNKNOWN` at 30k, 64 MiB TT, root-plus-50 horizon. |
| Hand Loss fixtures | 2 | 2 LOSS | Both the `FirstStone` and `SecondStone` forms strict-accepted. |
| **Total** | **217** | **48 templates** | No strict-accepted source failed relative admission. |

Of the 48 admitted templates, 46 were `FirstStone` eligible. The two explicit
non-`FirstStone` skips were `hayes_20260712_placement31` and
`forced_loss_secondstone`. Thus the retained-comparable WIN denominator remains
exactly 45 roots x four trials = 180 targets per K; the one transferable hand
Loss fixture adds four targets per K without changing that comparator.

## Stage 1: consolidated shadow reproduction

Each K unit added one balanced two-stone turn by each color (+4 placements),
returned to the same player and `FirstStone` phase, and used only legal,
nonterminal cells outside the extracted v1 projection. Every unchanged and
rebound candidate went to the unchanged strict verifier.

| K | Retained-comparable attempts | Unchanged strict accepted | Rebound strict accepted | Conditional rate | Retained round-1 rate | Hand Loss accepted | Total rebound accepted | M total | V total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 180 | 0 | 169 | 93.89% | 93.89% | 4/4 | 173/184 | 4.104 ms | 431.380 ms |
| 2 | 180 | 0 | 173 | 96.11% | 96.11% | 4/4 | 177/184 | 3.879 ms | 503.135 ms |
| 4 | 180 | 0 | 150 | 83.33% | 83.33% | 4/4 | 154/184 | 3.366 ms | 448.267 ms |
| 8 | 180 | 0 | 140 | 77.78% | 77.78% | 4/4 | 144/184 | 3.585 ms | 403.130 ms |

The retained-comparable counts reproduce round 1 exactly: `169/180`,
`173/180`, `150/180`, and `140/180`. Every conditional rate lies in the
retained exact interval `[140/180, 173/180]`, displayed as 77.78-96.11%.
Total materialization time was `14.933 ms`; total strict-verification time was
`1.785912 s`.

The unchanged-strict negative control accepted `0/720`. All 648 rebound
acceptances were cross-root because every target had additional placements.
The harness had no hard-result constructor, and the explicit
`hard_without_strict` counter remained zero.

**Stage-1 verdict: PASS.** The stop criterion "any hard result emitted without
strict acceptance" did not fire. The stop criterion "no cross-root strict
acceptance beyond exact root equality" did not fire.

## Stage 2: interface recorder and confusion matrix

### Interface size

The 46 eligible `FirstStone` canonical v1 test payloads total `561,419` bytes.
The two admitted `SecondStone` templates were recorded during acquisition but,
as required by the frozen `m` protocol, were not admitted to the Stage-2 library.

| Quantity per artifact | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| Serialized bytes | 177 | 2,350 | 28,362 | 91,206 |
| Projection cells | 7 | 27 | 50 | 70 |
| WF witnesses | 2 | 80 | 998 | 3,402 |

These are actual serialized bytes from the declared fixed-tag,
little-endian test codec, not the retained round-1 body-cell proxy. Full bytes
remain the cache identity; SHA-256 only selects and orders them.

### Candidate population and confusion matrix

Stage 2 used all K=1/K=2 targets: 360 WIN targets and 8 hand-Loss targets.
Each WIN target bypass-tested 45 eligible bodies x 12 D6 actions = 540 triples;
each Loss target tested 1 x 12 = 12. Total: `194,496`
body-target-D6 triples.

| | Strict accepted in bypass arm | Strict rejected / not materializable | Total |
|---|---:|---:|---:|
| `HintMatch` | 358 | 23 | 381 |
| not `HintMatch` | 8 | 194,107 | 194,115 |
| **Total** | **366** | **194,130** | **194,496** |

All 366 strict acceptances and all 381 projection matches were cross-root;
exact-root acceptance was zero by construction. Hint precision among probed
matches was `358/381 = 93.963%`.

The eight candidate-level false negatives were alternate acceptable bodies on
the eight mutations of human root `ec9d9ae7e7d0ab42@99`. Each of those targets
also had one matched strict-accepted body. Consequently:

```text
unconditionally hittable targets = 350
matched hittable targets         = 350
filtered hittable targets        =   0
```

O12's per-query `A_j` therefore remained one on every hittable target. This is
why the eight alternate-body false negatives are reported but do not trigger
the design's economic stop condition.

Candidate buckets were small: 355 targets had one match and 13 had two. Every
accepted target's first accepted rank was one, so fanouts 1, 2, 4, 8, 16, and
32 all hit the same `350/368 = 95.109%` targets.

### Phase timers and O12 interface subtotal

| Phase | Time |
|---|---:|
| Extraction/serialization/index build `E` | 165.221 ms |
| Lookup/order `L` | 11.273 ms |
| Interface matching `I` | 10.208 ms |
| Bypassed materialization `M` | 1.587723 s |
| Bypassed unchanged strict verification `V` | 5.565704 s |

For the interface decision, failed matches would avoid `6.187521 s` of measured
`M+V`. Charging all measured interface phases gives:

```text
saved filtered-probe M+V = 6.1875212 s
E + L + I                = 0.1867018 s
Stage-2 interface net    = 6.0008194 s
lost target solve saving = 0 (A_j preserved on all 350 hit targets)
```

This is only the O12 interface/selectivity component. The bypass arm
intentionally pays every candidate's `M+V`; it is not a production timing arm,
and it does not supply `S0`, `SR`, fixed-total-RAM effects, or a confidence
interval.

**Stage-2 verdict: PASS.** Projection classes create cross-root matches beyond
exact keys. Under the measured O12 interface equation, matching preserves every
acceptable target and saves more strict-probe work than `E+L+I` costs.

## Stage 3: six fixed negative-control fixtures

Only the six cases named in design section 7.2 ran.

- `H1_NQ2_REMOTE` — **PASS**: replayed the exact 36-placement root with
  `SecondStone.first=(6,0)` and `r=(6,-6)`; regenerated the horizon-66 proof at
  the 10k cap (`4,959` solve nodes); the source and all 12 D6 images strict-accepted.
- `H2_NQ3_FAR5` — **PASS**: used the retained `0hz3hty` body and exact additions
  `[(-8,0),(-16,0),(6,-1),(7,-1),(-24,0),(-32,0),(8,-1),(9,-1),
  (-40,0),(-48,0),(10,-1),(-56,0)]`; unchanged and rebound candidates both
  strict-rejected.
- `H3_CLOCK_SATURATION` — **PASS**: base `u32::MAX-1`, logical delta two, and
  saturated stored value `u32::MAX` rejected as `saturated_event_encoding`.
- `H4_NEGATIVE_HORIZON_KEY` — **PASS**: the forced mismatch at absolute horizon
  109 created an exact-key negative entry but did not suppress horizon 110,
  which materialized and strict-accepted.
- `H5_STALE_DELIVERY` — **PASS**: exact-source `0hz3hty` materialization
  strict-accepted; after lexicographically first legal nonterminal
  outside-projection move `(-8,0)`, complete-binding mismatch returned
  `UNKNOWN` and did not install evidence.
- `H6_FORCED_MISS_ISOLATION` — **PASS**: a contract-ID mismatch before the
  retained `xsnfyll` 10k query left the post-miss cold run identical to direct
  cold execution in root snapshot, status (`Win`), certificate, and recorded
  solver-visible state (`82` nodes, `148` TT entries, `14,688` peak TT bytes).

**Stage-3 verdict: PASS.** No fatal contract stop condition fired.

## Execution gates and raw evidence

Every Cargo invocation used:

```text
CARGO_TARGET_DIR=.target-hunt
--target x86_64-pc-windows-msvc
--release
--test-threads=1
```

Before every invocation, both `Memory\\Available Bytes >= 10 GiB` and
`Win32_OperatingSystem.FreePhysicalMemory >= 5 GiB` were checked. Final launch
gates were:

- Stages 1-2: an initial attempt correctly aborted at 7.548 GiB available /
  7.626 GiB free physical; the retry launched at 14.864 / 14.898 GiB.
- Stage 3: 15.660 GiB available / 15.672 GiB free physical.

Final invocation runtimes were 98.551 s for Stages 1-2 and 0.765 s for Stage 3,
both below ten minutes. The build log retains an initial compile-only type error
and the corrected successful build; no experiment ran during the failed build.

Authoritative raw logs:

| File | Bytes | SHA-256 |
|---|---:|---|
| `CREL_BUILD_RAW.log` | 5,789 | `B0359F218A6B9BEDDF2FED3972F808FEEA0C91C9ECE532945671403E47150FCE` |
| `CREL_STAGE12_RAW.log` | 415,051 | `2498992F94851E63A7BC1CEDC674CCE6FB8E65357A38BC7682B9BBD1C3FC1975` |
| `CREL_STAGE3_RAW.log` | 2,345 | `4EE8C440A3834E6B7D8CACD29A0ECD725F2229E87FDA7DECCC3E25D436C18A73` |

Superseded incremental Stage-1/2 decision-rule/accounting logs and the pre-SHA
Stage-3 log are retained separately with `PRELIM` / `PRE_SHA` names. They are
not the source of any reported verdict or timer.

## Final boundary

The round supports only the additive strict-discharge template-cache project.
Every reported acceptance was minted conceptually by the unchanged strict
verifier over a complete target root; the shadow harness emitted no production
hard value. Stages 4-6 were not run and no production gate candidate was built.
