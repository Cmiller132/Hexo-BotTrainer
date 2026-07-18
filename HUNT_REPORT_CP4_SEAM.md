# R-CP4 — final UNKNOWN provenance seam

## Verdict

**DONE.** CP-2.2's total internal stop taxonomy and default-off observation
seam are implemented. The final serial release battery passes, the frozen
official-profile identity smoke is exact before/after, and no public or sealed
negative result was added.

- Input commit: `2825a675f52abf7ace8ca24f281a662eb8fd49c5`
- Branch/worktree: `hunt/completeness`
- Commit created: **none** (prohibited by the task)
- `tss_verify.rs`: **untouched** (`git diff --exit-code --
  packages/hexfield_eq/rust/src/tss_verify.rs` returned 0)

## Implementation

`run_until` now returns the total CP-2.2 `RunUntilExit` enum. Its two cap
guards return `NodeCap { expansions, cap }`; the root-number exits are
distinct; selected intermediate cutoffs return `SelectedCutoff`; final-stage
cutoff nonprogress and stall are distinct. There is no fallthrough `None`.

`run` now returns the exact ten-variant CP-2.2 `SearchStop`. After every
`run_until` exit it performs the existing mandatory bottom-up refresh and then
uses provenance-aware precedence:

1. refreshed `pn == 0` is the preliminary `RootProven` stop;
2. refreshed `dn == 0` can become `RootRefutedCandidate` only through
   `try_emit_no_tss_v1` returning a complete artifact;
3. a tagged selected cutoff remains an intermediate `StageEvent`, even though
   `DepthCutoff` shares the numeric `dn == 0` value;
4. the cap and each incomplete tagged exit retain their own provenance;
5. a genuine `RootDnZero` for which artifact emission fails becomes
   `ExhaustionArtifactFailed`, never a refutation candidate.

CP4 does not implement the v1 negative emitter or checker. The emitter gate is
therefore deliberately fail-closed in production: it currently returns a
closed `NoEmitFailure` reason for every arena shape. This makes
`RootRefutedCandidate` production-unreachable until a later emitter supplies
the required artifact; the variant and its mapping are still covered with a
synthetic typed emitter-output fixture.

The existing positive materialization chain was mechanically expressed as a
`Result<TssCertificate, WinMaterializationFailure>` so the observation seam
distinguishes root tag/build failure, compact limit, rebase failure, and strict
positive verification failure. The resulting production certificate remains
the same `Option<TssCertificate>` and follows the same operations in the same
order.

## Default-off seam

The narrowest seam is `cfg(test)` storage:

- `AttemptResult.search_stop: Option<SearchStop>` and
  `AttemptResult.stage_events: Vec<StageEvent>` exist only in test builds;
- `TssSolver.last_search_stops` and `TssSolver.last_stage_events`, plus their
  private accessors, exist only in test builds;
- precondition and immediate-root stops are recorded directly in the same
  test-only solver observation buffers because those paths do not create an
  `AttemptResult`;
- narrow-compat attempts use `search_stop: None`; the CP-2.2 seam instruments
  the wide `run`/`run_until` path named by the specification.

Production builds contain none of the buffers or accessors. `ProofStatus`,
`DeepResult`, public return values, positive certificate types, hard-value
minting, and solve-goal selection are unchanged. A clean production release
build is retained in `CP4_PRODUCTION_BUILD_RAW.log`:

```text
Finished `release` profile [optimized] target(s) in 0.09s
```

The old focused selected-cutoff assertion was left textually unchanged. A
test-only `PartialEq<Option<usize>>` compatibility implementation compares only
`RunUntilExit::SelectedCutoff`; production code never converts the new total
result back to `Option`.

## Variant fixtures

The closed variant set has ten `SearchStop` variants. Frozen engine positions
are used where they are fast and stable; otherwise the test uses an explicit
synthetic construction of the tagged value, as CP-2.2 permits.

| Variant | Forcing fixture | Observed |
|---|---|---|
| `RootProven` | frozen `xsnfyll` corpus root, wide official profile, cap 10k | `RootProven { final_stage: 14 }`; public `Win` |
| `RootRefutedCandidate` | synthetic typed emitter-output artifact with three structural boundaries | `RefutationCandidatePendingVerification`; public remains `Unknown` because no checker acceptance exists |
| `NodeCap` | frozen `xsnfyll`, public cap 2 = one root examination plus one wide expansion | `NodeCap { stage: 0, expansions: 1, cap: 1 }`; public `Unknown` |
| `CutoffNoProgress` | synthetic selected cutoff revisited at stage 6/depth 7 without progress | exact `CutoffNoProgress`; `UnknownIncomplete` |
| `NonAdvancingCutoff` | synthetic stage 8 cutoff whose encountered depth is also 8 | exact `NonAdvancingCutoff`; `UnknownIncomplete` |
| `Stalled` | synthetic no-work-progress stop at stage 2 | exact `Stalled`; `UnknownIncomplete` |
| `ExhaustionArtifactFailed` | synthetic `dn == 0` arena containing an eligible cutoff | `EligibleDepthCutoff`; `UnknownIncomplete` |
| `MaterializationFailed` | synthetic positive compaction-limit failure | `CompactLimit`; `UnknownIncomplete` |
| `PreconditionRejected` | frozen `xsnfyll` with zero public node cap | actual `ZeroNodeCap`; public `Unknown` |
| `InvariantViolation` | missing-root negative-control construction | `MissingRootEntry`; `UnknownIncomplete` |

The only `StageEvent` kind is exercised by an actual staged solve:

| Stage event | Forcing fixture | Observed |
|---|---|---|
| `SelectedCutoff { from_stage, encountered_depth }` | frozen `xsnfyll` at 10k | seven strictly advancing intermediate events; terminal stop remained `RootProven { final_stage: 14 }` |

Final focused fixture output (`CP4_FIXTURES_RAW.log`):

```text
CP4_VARIANT_COVERAGE search_stops=10 mapping=PASS sealed_negative=UNREACHABLE
CP4_STAGE_EVENT kind=SelectedCutoff count=7 terminal=RootProven final_stage=14
test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 144 filtered out
```

## Mapping and sealed-result boundary

The mapping test covers every closed variant:

- `RootProven` -> positive candidate;
- `RootRefutedCandidate` -> candidate pending independent verification;
- `NodeCap` -> `UNKNOWN(Capped)` (and therefore public `ProofStatus::Unknown`);
- every other non-proof stop -> `UNKNOWN(Incomplete)`.

There is no sealed-negative case in the mapping type. In the current engine,
the pending candidate also projects to public `ProofStatus::Unknown`; no
`NO_CONTRACT_WIN`, `Loss`, sealed no-result, or other hard negative is minted
from any `SearchStop`.

## Official-profile identity smoke

Fixture: frozen corpus root `xsnfyll`, wide pair-complete search, cap 10,000,
2 GiB TT (`2147483648`), semantic horizon `u32::MAX`, all optional flags off,
one serial release test thread.

The certificate fingerprint is FNV-1a over the complete deterministic `Debug`
representation of the certificate. It is report/test instrumentation only and
does not define a production serialization format.

| Measurement | Before | After | Verdict |
|---|---:|---:|---|
| status | `Win` | `Win` | identical |
| certificate nodes | 38 | 38 | identical |
| certificate fingerprint | `baf5bf3c1107025e` | `baf5bf3c1107025e` | identical |
| public nodes | 82 | 82 | identical |
| expansions | 81 | 81 | identical |
| TT entries | 148 | 148 | identical |

Quoted from `CP4_IDENTITY_CERT_BEFORE_RAW.log` and
`CP4_IDENTITY_CERT_AFTER_RAW.log`:

```text
CP4_IDENTITY status=Win cert_nodes=38 cert_fingerprint=baf5bf3c1107025e nodes=82 expansions=81 tt_entries=148
```

The documented corpus harness independently reproduced the structural row in
`CP4_IDENTITY_BEFORE_RAW.log` and `CP4_IDENTITY_AFTER_RAW.log`:

```text
CORPUS_MODE shared_fragments=off lazy_frontier=off interior_gate=off k_reply_consume=off cap_resume=off live_ge3_seed=off closure_counters=off threshold_counters=off threshold_delta=off incr_enum_counters=off incr_defender=off tt_bytes_cap=2147483648
CORPUS id=xsnfyll cap=10000 status=WIN expect=WIN nodes=82 expansions=81 tt_entries=148 tt_hits=1 tt_bytes_cap=2147483648 peak_tt_bytes=14688 stage_refreshes=15 ...
CORPUS_DONE failures=0 shared_fragments=off lazy_frontier=off interior_gate=off k_reply_consume=off cap_resume=off ...
```

Identity verdict: **PASS, exact on status/certificate/nodes/TT entries**.

## Battery

Final command (after the required host-wide RAM and foreign-`cargo.exe`
checks):

```text
cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq -- --test-threads=1
```

Final retained output (`CP4_FULL_TSS_RAW.log`):

```text
test result: ok. 111 passed; 0 failed; 38 ignored; 0 measured; 0 filtered out; finished in 3.10s

Doc-tests hexfield_eq
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

No existing test expectation was changed. Battery verdict: **PASS**.

Every cargo launch was preceded by both required memory readings and a
foreign-cargo check. All launches had at least 10 GiB available bytes and 5
GiB free physical memory; no foreign `cargo.exe` was observed. All retained
raw logs were written as UTF-8.

## Files touched

- `packages/hexfield_eq/rust/src/tss_solver.rs`
- `HUNT_REPORT_CP4_SEAM.md`
- evidence logs: `CP4_FIXTURES_RAW.log`, `CP4_FULL_TSS_RAW.log`,
  `CP4_IDENTITY_BEFORE_RAW.log`, `CP4_IDENTITY_AFTER_RAW.log`,
  `CP4_IDENTITY_CERT_BEFORE_RAW.log`, `CP4_IDENTITY_CERT_AFTER_RAW.log`, and
  `CP4_PRODUCTION_BUILD_RAW.log`

`packages/hexfield_eq/rust/src/tss_verify.rs` was not modified.

## CP-O28 status and open residue

**CP-O28 disposition: eased by the implemented fail-closed stop-provenance
seam, but not discharged; negative artifact emission/checking and the sealed
mint remain absent.**

Open residue is intentionally outside CP4: implement the exact v1 negative
emitter, independently checked `checkNo`, Rust/Lean/executed-checker
correspondence, and only then allow checker acceptance to cross the sealed
result boundary. Until that work lands, every structural exhaustion attempt
fails closed and no hard negative exists.
