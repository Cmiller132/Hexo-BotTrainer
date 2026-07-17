# R-CR1 cap-ladder resume hunt

Date: 2026-07-17

Worktree: `tss-vcf-width`

Starting HEAD: `6ef67cfe49dfe4f016cab866d267ea07ff58d1ef`

## Verdict

**PROMOTE — 29.10% full-profile test-wall improvement (29.71% summed-solve-wall improvement).**

The cfg(test)-only resumable `WidePnSearch` session reduced the complete official
1 GiB lazy+gate test wall from the retained **495.940 s** baseline to
**351.620 s**, and summed measured solve wall from **495.592 s** to
**348.329 s**. The 5% gate is cleared by 24.10 percentage points.

Recommended next default: **harness-only**. The measured win is specifically on
the official cap ladder, and this round has deliberately not created a
production lifetime/API commitment. An exposed in-process handle is plausible
for repeated exact trainer queries, but should be a separate API/soundness
proposal. Serialization remains out of scope.

## Implementation and soundness

`CapResumeSession` and `advance_to_node_cap` exist only under `cfg(test)`. One
session owns one `WidePnSearch`, including its arena, position index, lazy
deferred-position map, proof/disproof numbers, commitment state carried by the
arena, and a persistent staged-depth cursor. The continuation driver does not
call `run()` again; it retains both `stage_depth` and whether that stage's
cutoffs have already been reopened.

The immutable binding contains:

- exact `RootBinding`, derived claimant, and `SolveGoal`;
- semantic horizon, width options, zone/search options, and final structural
  depth;
- TT enabled mode, caller TT byte cap, and hash mask;
- shared-fragment, lazy-frontier, lazy-key-validation, interior-census-gate,
  K-reply, and quotient-telemetry flags.

Only a strictly larger total node cap is accepted. A mismatch or non-monotone
cap permanently invalidates the session. The session marks itself invalid
before entering search/materialization, so a caught panic cannot reuse partial
state. The unit test confirms mismatch/non-monotone invalidation and the
subsequent `Discarded` result.

Unfinished cap rungs return `Unknown`; their PN/DN values remain test telemetry
and never cross the finder/verifier boundary as a game fact. A hard result is
built with the ordinary `WidePnSearch::materialize`, certificate compaction,
and zone rebase path, then checked by the unchanged `TssVerifier`. The corpus
harness verifies it again. Immediate-root results, which have no unfinished
wide frontier to retain, use the unchanged fresh pre-search path.

## Identity milestones

Notation: `pn/dn status; fresh expansions -> resumed cumulative expansions`.
`I` is the engine's `PN_INFINITY = 1,000,000,000`. All listed identities passed
and every listed hard certificate passed the strict verifier.

| Root | 10k | 100k | 1M | 20M |
|---|---|---|---|---|
| `0l4291i_live` | `34/471 U; 9,999 -> 9,999` | `34/732 U; 99,999 -> 99,999` | `34/734 U; 999,999 -> 999,999` | `0/I WIN; 1,879,611 -> 1,713,725` |
| `94gnnol` | `34/79 U; 9,999 -> 9,999` | `34/29 U; 99,999 -> 99,999` | `34/22 U; 999,999 -> 999,999` | **not run; see limitation below** |
| `lz60mfb` | `553/3 U; 9,999 -> 9,999` | `319/3 U; 99,999 -> 99,999` | `0/I WIN; 109,895 -> 109,907` | `0/I WIN; 109,895 -> 109,907` |
| `mvp2lvc` | `1,493/112 U; 9,999 -> 9,999` | `I/0 U; 17,956 -> 17,956` | `I/0 U; 17,956 -> 17,956` | `I/0 U; 17,956 -> 17,956` |
| `hayes_20260712_turn16` | `34/475 U; 9,999 -> 9,999` | `0/I WIN; 11,663 -> 11,454` | `0/I WIN; 11,663 -> 11,454` | `0/I WIN; 11,663 -> 11,454` |
| `hayes_20260712_placement31` | `34/475 U; 9,999 -> 9,999` | `0/I WIN; 11,663 -> 11,454` | `0/I WIN; 11,663 -> 11,454` | `0/I WIN; 11,663 -> 11,454` |
| `xsnfyll` (simple closer) | `0/I WIN; 81 -> 81` | `0/I WIN; 81 -> 81` | `0/I WIN; 81 -> 81` | `0/I WIN; 81 -> 81` |

Pause unwinding can legally alter a tie after a cap. This occurred only at hard
closure in the measured set: `0l` used 165,886 fewer expansions, both Hayes
rows used 209 fewer, and `lz60mfb` used 12 more. PN/DN/status remained
identical. Certificate sizes consequently differed for `0l` (18,871 fresh vs
17,716 resumed nodes) and Hayes (1,621/1,620 vs 1,606/1,605); all variants
strict-verified. `lz60mfb` retained 15,900 certificate nodes on both paths.

The campaign completed 27 of the literal 28 requested triples. The omitted
`94gnnol` 20M rung is outside the official NO-row ladder (which stops at 1M).
At 1M, one fresh plus one resumed path already cost 240.51 s. If the 20M rung
does not close early, the two required 20M paths can exceed 80 minutes, which
cannot fit the binding campaign rule of at most 45 minutes per gate-class
command. I did not weaken the cap, start a command that could not honor the
bound, or claim this unmeasured triple. Its final official-cap comparison is
complete: both paths are `34/22 UNKNOWN` at 999,999 expansions.

## Full official 1 GiB lazy+gate profile

The passing run started with **11.10 GiB free RAM**, used one Cargo process,
the MSVC target, `.target-codex`, one test thread, 1 GiB TT, lazy frontier on,
interior census gate on, and shared fragments/K-reply off. All expectation
flags were asserted. The inert unbounded-horizon census gate recorded zero
evaluations, as in the baseline analysis.

| Metric | Retained fresh-ladder baseline | Resumed ladder | Delta |
|---|---:|---:|---:|
| Test wall | 495.940 s | 351.620 s | **-144.320 s (-29.10%)** |
| Summed solve wall | 495.592 s | 348.329 s | **-147.263 s (-29.71%)** |
| Actual nodes | 4,507,362 | 2,937,397 | -1,569,965 (-34.84%) |
| Actual expansions | 4,507,328 | 2,937,378 | -1,569,950 (-34.83%) |
| Pair generation | 216.228 s | 151.597 s | -64.631 s (-29.89%) |
| Defender enumeration | 175.837 s | 115.495 s | -60.342 s (-34.32%) |
| Prior/regen | 43.036 s | 30.819 s | -12.217 s (-28.39%) |
| Expansion inclusive | 397.325 s | 270.547 s | -126.778 s (-31.91%) |
| Full stage refresh | 23.557 s | 28.304 s | **+4.747 s (+20.15%)** |
| Direct insertion | 3.283 s | 2.034 s | -1.249 s (-38.04%) |
| Peak bytes | 549,161,606 | 499,985,973 | -49,175,633 (-8.95%) |

The resumed expansion total is also **171,556 (5.52%) below** the baseline's
3,108,934 final-attempt-only expansions. This is pause-order scheduling, not
discarded-rung recovery, and should not be assumed for other corpora.

Re-entry counts by root were: `0l4291i_live=3`, `94gnnol=2`, `l9mxn59=2`,
`lz60mfb=2`, `mvp2lvc=2`, `zrugh2x=1`, `strongloss_a_prefix6=1`, and each
Hayes row `=1`; total **15**. All other searched roots closed on their first
rung. `8is963b` and `dy3dg99` were immediate pre-search LOSS results and used
the explicit no-frontier fallback.

### Measured overhead

Continuation adds one bottom-up refresh at each pause/re-entry boundary. That
is visible as the only aggregate timer regression: +4.747 s. Re-materializing
and re-verifying already-hard results is also visible on later no-op rungs:
about 0.7 ms for `xsnfyll`, about 40 ms per Hayes re-entry, about 0.03 ms for
exhausted `l9mxn59`, and about 10 ms for exhausted `mvp2lvc`. These costs are
already included in the 29.10% end-to-end result.

## Default-off and production identity

All implementation state and the campaign module are `cfg(test)`-gated. A
non-test release build succeeded, proving the production artifact compiles
without the session types/fields/driver.

With `TSS_CAP_RESUME` unset and `TSS_CORPUS_EXPECT_CAP_RESUME=0`, the fast
lazy+gate subset reproduced retained counts exactly:

- `mvp2lvc`: UNKNOWN at 10k/100k, 9,999/17,956 expansions, peaks
  1,948,056/3,928,520 bytes;
- `xsnfyll`: WIN at 10k, 81 expansions, peak 7,156 bytes.

Thus the feature-unused search decisions are unchanged, while production does
not contain the test API at all.

## Retained raw logs

- `CAP_RESUME_IDENTITY_XSN_RAW.log`
- `CAP_RESUME_IDENTITY_HAYES_RAW.log`
- `CAP_RESUME_IDENTITY_LZ_MVP_RAW.log`
- `CAP_RESUME_IDENTITY_0L_RAW.log`
- `CAP_RESUME_IDENTITY_94_1M_RAW.log`
- `CAP_RESUME_FULL_PROFILE_RAW_2.log` (passing official profile)
- `CAP_RESUME_FULL_PROFILE_RAW.log` (retained first attempt; clean unsupported
  immediate-root stop after `0l`)
- `CAP_RESUME_FALLBACK_RAW.log`
- `CAP_RESUME_DEFAULT_OFF_RAW.log`
- `CAP_RESUME_UNIT_RAW.log`
- `CAP_RESUME_UNIT_FINAL_RAW.log`
- `CAP_RESUME_PRODUCTION_BUILD_RAW.log`
- `CAP_RESUME_BUILD_RAW.log`

## Regeneration

Every Cargo command must first check that no Cargo process exists and that free
RAM is above 9 GiB; use above 11 GiB immediately before the full profile.

```powershell
$ErrorActionPreference = 'Stop'
if (Get-Process cargo -ErrorAction SilentlyContinue) {
    throw 'another cargo process is running'
}
$os = Get-CimInstance Win32_OperatingSystem
$free = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
if ($free -le 11) { throw "free RAM $free GiB is not above 11 GiB" }
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }

$env:CARGO_TARGET_DIR = '.target-codex'
$env:TSS_BACKWALK_TT_BYTES = '1073741824'
$env:TSS_LAZY_FRONTIER = '1'
$env:TSS_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CAP_RESUME = '1'
$env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS = '0'
$env:TSS_CORPUS_EXPECT_LAZY_FRONTIER = '1'
$env:TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CORPUS_EXPECT_K_REPLY_CONSUME = '0'
$env:TSS_CORPUS_EXPECT_CAP_RESUME = '1'

cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_corpus_check -- `
    --ignored --test-threads=1 --nocapture
```

Milestone campaign (optionally set `TSS_CAP_RESUME_ID`,
`TSS_CAP_RESUME_MAX_CAP`, and `TSS_CAP_RESUME_TT_BYTES`):

```powershell
$env:CARGO_TARGET_DIR = '.target-codex'
$env:TSS_LAZY_FRONTIER = '1'
$env:TSS_INTERIOR_CENSUS_GATE = '1'
$env:TSS_CAP_RESUME_TT_BYTES = '1073741824'
cargo test --release --target x86_64-pc-windows-msvc `
    -p hexfield_eq tss_cap_resume_campaign -- `
    --ignored --test-threads=1 --nocapture
```

Production cfg check:

```powershell
$env:CARGO_TARGET_DIR = '.target-codex'
cargo build --release --target x86_64-pc-windows-msvc -p hexfield_eq
```
