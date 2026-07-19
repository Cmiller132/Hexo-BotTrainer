# R-CREL-6 Phase 3C: DOM-B2 depth-4 reopen check

**LIVE FINAL-CANDIDATE REPORT - PRIMARY MEASUREMENT IS NOT SEALED.**

Current disposition: **NO VERDICT.** This document is structured for the final
handoff, but its final three-lane measurement tables, certificate hashes, Loss
field, verdict, final hostile review, and phase manifest are intentionally
pending. No PASS, scoped KILL, final NULL, or hard abort is assigned here.

The only measurement snapshot reported below is immutable checkpoint 1: 20
credited PRIMARY invocations and 1,140 credited roots. Cases 0--2 were exact
`Unknown`; case 3 was credited only through index 186. Later bytes in the
moving queue, later closed shards, and current process state are deliberately
not copied into this report. They can become authoritative only through a new
seal or the complete final certificate path.

## Question, identity, and scope

- Input HEAD: `5f5da82a04d14f645fdbf08ea96937a428182cde`.
- Historical experiment commit: `af6f777c53b9a1f91bff0186fb1e05d99e2a5cce`.
- Candidate: K=1, budget-2 spare-stone domination, depth-4 evidence gate.
- Exact opt-in: `TSS_DOM_B2_D4_REOPEN=1`; otherwise the harness is inert.
- Current code identity: `DOM_B2_D4_PRIMARY_V3`.
- Machinery scope: `cfg(test)`, ignored exact-reference tests, and read-only
  PowerShell evidence tools. It is default-off.
- Cargo profile: `.target-hunt`, release, `x86_64-pc-windows-msvc`, serial
  `--test-threads=1`.
- This phase authorizes neither a production rule nor a verifier change.

The 11 rows are values below selected completed defender turns `O4(u,v)`.
They do not enumerate every legal second placement for a given defender first
placement and therefore are not exhaustive `F4(u)` values. A favorable result
can only reopen the preregistered K=1, budget-2 proof round. It cannot prove or
deploy the conjecture. An adverse mapped result kills only the fixed
completed-child claim unless a separate exhaustive `F4` witness is produced.

## Frozen scientific protocol and inherited state

The binding preregistration is `DOM_B2_D4_PREREG_RAW.log`, 7,618 bytes,
SHA256
`9A76F6F9E24F551E69A8A513FC33031D5EB5F350B129B99354C6FD81959B1981`.
`DOM_B2_D4_PREREG_SHA_RAW.log` sealed it before the first DOM-B2 Cargo launch.
The architecture record is `DOM_B2_D4_ARCH_RAW.log`, 3,868 bytes, SHA256
`19BC53DC9E20B9B6AB831492CBA83372CC0A691C053410531B34C92EE6913D94`.

The historical copy, `DOM_B2_D4_HISTORICAL_RAW.log`, is 3,617 bytes with
SHA256
`CE60D033C3FFC4915C151224314776EB5471C9C8F5CDC9202CB293E286676BC1`.
It records why the candidate entered this queue as
`OPEN-COMPUTATION-LIMITED`:

- all 11 selected completed turns were exact attacker `Unknown` at depth 3;
- the historical depth-4 batch completed only case 0, attacker `Unknown`,
  after 62,601,245 nodes and 2,285.247044 seconds;
- case 1 stopped incomplete and cases 2--10 inherited an expired shared
  deadline, so no complete mapped depth-4 direction existed; and
- inherited Q0 covered only 3/16 Loss slots, so Loss was never prequalified.

Historical rows provide context and the case-0 reproduction check. They do not
receive current V3 matrix credit.

### Binding scientific bars

Under defender preference `attacker Loss > Unknown > Win`, PASS /
REOPEN-EVIDENCE requires all of the following:

1. All 11 PRIMARY case aggregates are exact, non-Incomplete, and backed by a
   gapless frozen-root partition.
2. Case 0 reproduces the historical attacker `Unknown`, or a discrepancy is
   independently resolved by both mandatory replicas.
3. All four mapped relations hold:
   `case0 <= max(case1,case2)`, `case3 <= case4`, `case5 <= case6`, and
   `case7 <= case8`.
4. Cases 9 and 10 agree for the lifted equality control.
5. PRIMARY, D6_OFF, and SECOND_TT have exact per-root status agreement; both
   replicas reproduce every comparison, the control, the history check, and
   every non-`Unknown` primary case.
6. Every structural, source, binary, runner, chronology, chain, and verifier
   fence passes.

Scoped KILL requires a replicated mapped reversal or replicated case-9/10
control mismatch. It does not establish an exhaustive `F4` counterexample.
NULL / REMAINS PARKED applies when no scoped KILL exists and productive
bounded work remains incomplete, a lane/gap/conflict remains, a favorable
fence cannot be met, or a primary attacker Loss cannot be mechanically
qualified. A replay/census mismatch, D6/TT exact-status disagreement,
fabricated stopped value, source/binary/provenance failure, production-solver
change attributable to this phase, or verifier diff is a hard abort.

The frozen final adjudicator strengthens the favorable Loss boundary without
relaxing these bars: its v1 Loss capability is deliberately
`UNSUPPORTED_V1`. No self-attested Loss document can unlock PASS. After all
complete evidence fences pass, a replicated scoped reversal/control witness
takes precedence and remains scoped KILL even if a participating case is
Loss; otherwise any primary Loss is fail-closed NULL / REMAINS PARKED. A
future favorable Loss would require a separately preregistered mechanical
stock/fast protocol and a new adjudicator version.

## Enumeration architecture fixed before completeness

Completeness was defined before reading current statuses:

1. Replay each frozen parent; assert nonterminal defender FirstStone,
   threat-analysis budget 2, `own_win_now=false`,
   `min_hitting_set=Some(1)`, and a nonempty attacker count-4/5 family.
2. Assert each ordered defender pair is legal, singleton-nonterminal,
   completed-turn-nonterminal, covers every attacker threat window, and has
   its frozen SPLIT/H_CONTAINING class. The child is attacker FirstStone.
3. Enumerate the child's complete legal attacker first-placement set with the
   engine and independently with `tss_reference_fast::full_legal_moves`.
   Sort by raw `(q,r)`, require equality, then freeze exact coordinates,
   width, and fingerprint before solving any status.
4. Partition root indices into disjoint half-open shards. An immediate
   attacker win is exact; otherwise call the bounded exact reference for the
   remaining three plies. Record configuration, census identity, status,
   nodes, TT fields, and wall time.
5. Aggregate by the exact maximizing recurrence: any `Win` gives `Win`;
   otherwise missing/Incomplete poisons the case; otherwise any `Unknown`
   gives `Unknown`; otherwise all roots are `Loss`.
6. A closed exact prefix `[start,next_start)` may be retained. A stopped root
   and its suffix receive no status and may be recomputed. Reject gaps, extra
   indices, coordinate/fingerprint mismatch, conflicting complete rows, or
   malformed stop semantics.
7. If one root cannot complete in a bounded invocation, divide its complete
   sorted next-action universe and apply the same recurrence. Never relabel a
   stopped subtree `Unknown`.

`DOM_B2_D4_CENSUS_RAW.log` is the authoritative universe: 292,756 bytes,
SHA256
`436E9F6C4A93CDB611EEF6495A01F510615174A2897271CA1092A0E7422DD7BE`.
It contains 11 universe rows, 3,648 indexed root rows, and a PASS footer.

| Case | Parent | Class | Ordered pair | Roots | Fingerprint |
|---:|---|---|---|---:|---|
| 0 | `32f44c499244b611:9` | SPLIT | `(-2,1);(4,1)` | 329 | `827EEB0FCB78C698` |
| 1 | `32f44c499244b611:9` | H_CONTAINING | `(2,1);(-2,1)` | 312 | `7C4092D562D3E619` |
| 2 | `32f44c499244b611:9` | H_CONTAINING | `(2,1);(4,1)` | 312 | `319A510062631E51` |
| 3 | `19b085e7aa9f6215:9` | SPLIT | `(-1,0);(5,0)` | 330 | `27C689B6D4D0DC33` |
| 4 | `19b085e7aa9f6215:9` | H_CONTAINING | `(3,0);(-1,0)` | 313 | `09E733EF333378BA` |
| 5 | `498a61ae0b5cf4ef:9` | SPLIT | `(-2,2);(4,-4)` | 330 | `530A55C8F49F0911` |
| 6 | `498a61ae0b5cf4ef:9` | H_CONTAINING | `(2,-2);(-2,2)` | 313 | `324FB3CEA1CDCA7E` |
| 7 | `fd688f189544bf72:9` | SPLIT | `(-2,0);(4,0)` | 330 | `CCE0D5A475F109B6` |
| 8 | `fd688f189544bf72:9` | H_CONTAINING | `(2,0);(-2,0)` | 313 | `F9FC2F2E9BB41D72` |
| 9 | `d7e1b56c925b7f32:19` | H_CONTAINING | `(-1,0);(-2,3)` | 383 | `FF353A8E0556E088` |
| 10 | `d7e1b56c925b7f32:19` | H_CONTAINING | `(-1,0);(-1,2)` | 383 | `B57A000A7F5C6800` |

The widths sum to 3,648 roots per lane. Final completeness is exactly 3,648
roots and 11 cases in each of PRIMARY, D6_OFF, and SECOND_TT: 10,944
collision-free lane-roots total.

## Frozen implementation and evidence stack

The following exact identities are the final-candidate support fence:

| Role | Path | Bytes | SHA256 |
|---|---|---:|---|
| V3 source snapshot | `DOM_B2_D4_PRIMARY_V3_CODE_SNAPSHOT_RAW.log` | 3,399 | `1D4FBB37638668D0F2ED1972D27CDDD833826721A2EFD5500BFDC09DCF81B746` |
| V3 source-snapshot seal | `DOM_B2_D4_PRIMARY_V3_CODE_SNAPSHOT_SHA_RAW.log` | 396 | `38BB0351A021971A4F4CD67F20217BACC7C98A1E14E8E8A59F84CDC78A0527A1` |
| Test harness | `packages/hexfield_eq/rust/src/tss_dom_b2_d4_reopen.rs` | 17,116 | `23A8CC4BC6B62AC9A93A1A7692C659FC2BA9CBA91E032FE51F35D3924226F7D1` |
| Bounded exact reference | `packages/hexfield_eq/rust/src/tss_reference_fast.rs` | 23,392 | `6B31AEF4176A19541B17A4D5D14BAF3D2E4083DD3C55193FC44F3E46D9195E40` |
| Bound test binary | `.target-hunt/x86_64-pc-windows-msvc/release/deps/hexfield_eq-de26e3778420c4c2.exe` | 3,290,112 | `56B8FA5563D5CDE397133B8328DEB3B79D072E2577573C4C0A94619AA4750A14` |
| Strict verifier | `packages/hexfield_eq/rust/src/tss_verify.rs` | 78,741 | `9990D38618DA2204351E328CA0143BE2AEF98BB3001E4A0462CF346B707F2CE8` |
| PRIMARY runner | `scripts/dom_b2_d4_run_queue.ps1` | 49,762 | `25C6AB359B4BCB3F2207F426BF958A7DB588E1B5BFE7FC0D67FF4AC3EC385D8E` |
| Replica runner | `scripts/dom_b2_d4_run_replica_queue.ps1` | 56,516 | `B9E61444E1555E648575336B8B6E8CD406CB0FB24C406557106F639E9B1953C8` |
| Replica preregistration | `DOM_B2_D4_REPLICA_PREREG_RAW.log` | 11,620 | `2911856F2EB77355E9EE7B027EA487C46B706625AB7DB59BE4804B552B54F23A` |
| Replica static evidence | `DOM_B2_D4_REPLICA_RUNNER_STATIC_RAW.log` | 3,986 | `2D16DFDBE807FCF824640AEB465A8B94C5B1A0C157D52CF9CC327ABDD1B3210C` |
| Replica hostile review | `DOM_B2_D4_REPLICA_HOSTILE_AUDIT_RAW.log` | 4,229 | `18A0130D4136431EEDD4BB99C33C9A49CCF3768AE2450C13C88AF6F88F261F91` |
| Multi-lane analyzer | `scripts/dom_b2_d4_aggregate.ps1` | 77,182 | `862BCF23125EA63C66334C8DCAD192FA8A7528267A8954BB8C42D5AE4BD8BED5` |
| Analyzer static evidence | `DOM_B2_D4_ANALYZER_STATIC_TEST_RAW.log` | 6,641 | `C1853193844BA45AE35C8E7126711B703A294AF7FDF67AB53485A6F8DBCE95E5` |
| Chain auditor | `scripts/dom_b2_d4_chain_audit.ps1` | 116,000 | `E8991F499D90FEF81B9CCF492CC17047ED24863159F3F83279A7634A9DE383C8` |
| Chain contract | `DOM_B2_D4_CHAIN_AUDIT_PREREG_RAW.log` | 10,321 | `24F87552069BAE910EC87656CD7EF7A40BD40D504A10A984E7472D2361B42A34` |
| Chain static evidence | `DOM_B2_D4_CHAIN_AUDIT_STATIC_RAW.log` | 4,715 | `85AF8E3F55C02FD27D2D6A5649A9A04915E1ED5CF104D07E87B165293A21D9C7` |
| Chain hostile review | `DOM_B2_D4_CHAIN_AUDIT_HOSTILE_REVIEW_RAW.log` | 11,835 | `71BBA9B14D03107A728F16A41B459FACF2741C7D3DAFFC341265C6AF6FA8F8E6` |
| Final adjudicator | `scripts/dom_b2_d4_final_adjudicate.ps1` | 87,704 | `3C6E84CBC387745BB40085B2B1A711CBB5774F2C3A58EEE9E0DCF3DFBAAC9924` |
| Adjudicator contract | `DOM_B2_D4_FINAL_ADJUDICATOR_PREREG_RAW.log` | 14,456 | `4B95B0F1D17A3811BD57FBB110A3DA2A9FC748AE1C59DA4EEEC24D469C74B3BA` |
| Adjudicator static evidence | `DOM_B2_D4_FINAL_ADJUDICATOR_STATIC_RAW.log` | 6,542 | `5D48AD09CF6F3399B3602E0301BD4125325261447A9CA79FC5AB5C3D6D647D92` |
| Adjudicator hostile review | `DOM_B2_D4_FINAL_ADJUDICATOR_HOSTILE_RAW.log` | 12,436 | `8E42BF845B4F0E5BFC15AF02047C932509AC9809828D2755A167B7A6DD52FB98` |

The source snapshot binds 17 `CODE_FILE` entries plus the exact binary,
schema smoke, and strict verifier. Every credited V3 launch rechecks the
snapshot and configuration before starting, then rechecks source and binary
before a META sidecar may be created. The bounded reference returns
`Option<ProofStatus>`; incomplete nondecisive work propagates as `None`, is
never TT-cached as a status, and does not alter the unbounded test API.

Premeasurement checks were preserved separately from matrix credit. The
gated release build passed in 12.19 seconds (`DOM_B2_D4_BUILD_RAW.log`), the
harness status-label check passed 1/1 (`DOM_B2_D4_HARNESS_UNIT_RAW.log`), and
the bounded/unbounded recurrence check passed 1/1 in 1.56 seconds
(`DOM_B2_D4_REFERENCE_RAW.log`). An earlier broad filter selected ignored
tests and ran none; `DOM_B2_D4_REFERENCE_FILTER_RAW.log` marks it
non-authoritative. Four low-memory gate attempts launched no Cargo process.
The V3 schema smoke is support identity only, not a matrix root. None of these
checks contributes to the 3,648-root PRIMARY denominator.

## V3 PRIMARY runner and credit rules

PRIMARY is fixed to D6 on, 536,870,912 TT bytes, a 480,000 ms internal search
bound, and a 540,000 ms process watchdog. Before every Cargo invocation the
runner requires:

- available memory at least 10,737,418,240 bytes;
- free physical memory at least 5,368,709,120 bytes;
- zero pre-existing `cargo.exe` processes;
- exact source snapshot, census, binary, compiler/configuration, runner hash,
  and `DOM_B2_D4_PRIMARY_V3`; and
- serial release execution for the frozen Windows target.

The runner uses create-new raw, stdout, stderr, Cargo-exit, META, and journal
identities. It writes and flushes each result row. A raw can contribute only a
closed exact prefix bound by a matching journal RESULT, exact META, exact
Cargo-exit record, source/binary fences, resource gate, and runner identity.
Footerless work, stopped rows, console files, and present artifacts without an
exact credit chain receive zero roots. One logical Cargo invocation may appear
as a rustup proxy plus its toolchain `cargo.exe`; the prelaunch rule concerns
zero pre-existing Cargo processes and host-wide invocation serialization.

The frozen final provenance family begins at
`DOM_B2_D4_QUEUE_V3_RUN03_RAW.log`. Earlier queue families are retained only
for chronology. No result from a pilot, smoke, schema draft, PRIMARY RUN01 or
RUN02, V3 RUN01 or RUN02, pre-V3 raw, or raw lacking exact V3 META+RESULT
provenance is allowed into completeness or a verdict.

## Immutable checkpoint 1 - the only live measurement authority here

The complete partial artifact set is:

- `HUNT_REPORT_DOM_B2_D4_CHECKPOINT1.md`, 13,471 bytes, SHA256
  `C82ACA66DE4DAA57352BF0654832372354D4AF79327AEB59D47B07FEE59E508E`;
- `DOM_B2_D4_CHECKPOINT1_HASHES_RAW.log`, 12,684 bytes, SHA256
  `DF8A147CB6A7F1FCFBBDA47B595A8BDE30005037CD456E7E78AA04F1B5BAB4F4`;
- `DOM_B2_D4_CHAIN_AUDIT_PARTIAL_RAW.log`, 11,469 bytes, SHA256
  `536175702A92D617A8AB0666FDEF60457334D66836314B4A63B708252CFFAF29`;
  and
- exactly the first 59,673 bytes of the moving RUN03 journal, SHA256
  `D71A641D9B5542DCDD7102DC539251D822CB27DC3280138F590387E67EE92FDE`.

The checkpoint manifest contains 101 verified whole-file entries plus the
explicit journal-prefix entry. It covers all 20 credited raw/META/Cargo-exit
triples. The partial auditor reported grammar PASS, state machine PASS,
chronology PASS, race check PASS, no aborts, and one open RUN receiving zero
credit.

| Sealed row | Credited roots | Exact checkpoint status |
|---|---:|---|
| PRIMARY case 0 | 329/329 | `Unknown`; historical case-0 status reproduced |
| PRIMARY case 1 | 312/312 | `Unknown` |
| PRIMARY case 2 | 312/312 | `Unknown` |
| PRIMARY case 3 | 187/330 | indices 0--186 `Unknown`; aggregate Incomplete |
| PRIMARY cases 4--10 | 0/2,365 | not reached at the seal |
| D6_OFF | 0/3,648 | no replica artifact existed at the protocol/checkpoint seal |
| SECOND_TT | 0/3,648 | no replica artifact existed at the protocol/checkpoint seal |

Thus 1,140/3,648 PRIMARY roots (31.25%) and 1,140/10,944 planned lane-roots
(10.4167%) were credited. All 1,140 credited statuses were `Unknown`. Seven
closed raw files also contained one terminal `INCOMPLETE` attempt row after
their credited prefix; those seven rows received no status credit. The
unmatched case-3/start-187 RUN was classified `ACTIVE_UNCREDITED`; its open raw
was not inspected. Cases 0--2 make the first primary relation true at the
checkpoint, but the other relations, equality control, both replicas, and
final provenance remain absent. This is **NO VERDICT**, not a provisional
PASS or a final NULL.

## Legacy and untrusted chronology - zero verdict credit

Earlier artifacts remain on disk so execution history is not erased:

- resource gates below 10 GiB recorded no launch;
- a pilot and streaming smoke checked harness and durability behavior;
- an early wrapper attempt failed before admissible measurement credit;
- pre-V3 and draft-schema raws exercised resume and provenance handling;
- non-V3 PRIMARY RUN01/RUN02 and V3 RUN01/RUN02 all preceded the frozen V3
  RUN03 evidence family; and
- analyzer fixtures intentionally include untrusted V3 files to prove they
  remain excluded.

These materials may diagnose plumbing but are not alternate measurement
authority. In particular, the former RUN02/88-root snapshot and any pre-V3
pilot count are superseded and receive zero completeness or verdict credit.
The analyzer's exclusion regression observed 47 internally complete rows in
two untrusted V3 fixtures and still assigned PRIMARY credit 0, as intended.

## Mandatory replica lanes

`DOM_B2_D4_REPLICA_PREREG_RAW.log` froze both lanes before any replica data,
META, or journal existed. The order and configuration are immutable:

| Order | Lane | D6 | TT bytes | Required roots/cases |
|---:|---|---|---:|---:|
| 1 | D6_OFF | false | 536,870,912 | 3,648 / 11 |
| 2 | SECOND_TT | true | 268,435,456 | 3,648 / 11 |

Both lanes are mandatory regardless of the first replica's outcomes. They use
lane-tagged create-new artifacts, a lane lock plus a shared replica-global
lock, the same 480,000/540,000 ms limits, the same RAM/Cargo gates, exact V3
source/census/binary identity, and the same complete recurrence. PRIMARY must
be terminal before D6_OFF starts; D6_OFF must be terminal before SECOND_TT.
The runner requires no PRIMARY lock and zero Cargo processes both at the
ordinary gate and immediately before launch.

Replica completeness is separate by configuration. Each lane must enumerate
all 3,648 roots and 11 cases, then agree with PRIMARY at every root. Missing,
Incomplete, conflicting, wrong-lane, wrong-runner, wrong-config, or unbound
artifacts receive no lane credit. Exact status disagreement is a hard abort,
not sampling noise.

The frozen operational residual is explicit: the older PRIMARY runner does
not acquire the newer replica-global lock. Therefore no PRIMARY queue may be
started or restarted after replica execution begins. The replica runner
checks the PRIMARY lock and Cargo state twice, but a process could
theoretically appear in the final instruction interval. Any observed overlap
is a hard-constraint violation and the affected evidence cannot support a
verdict. The replica runner's preregistered hash must also be checked
externally immediately before launch.

## Scientific analyzer and its deliberate external boundary

The analyzer is read-only and admits exactly three approved mappings:

- PRIMARY: D6 true, 512 MiB TT, runner SHA `25C6...85D8E`;
- D6_OFF: D6 false, 512 MiB TT, runner SHA `B9E6...953C8`; and
- SECOND_TT: D6 true, 256 MiB TT, runner SHA `B9E6...953C8`.

Missing, extra, duplicate, or substitute mappings fail. The analyzer parses
only canonical raw/META pairs, keeps legacy and untrusted V3 accounting
separate, aggregates all 11 cases independently per lane, checks exact
coverage and recurrence, compares all 3,648 root statuses across lanes,
recomputes the four comparisons and equality/history controls, and emits the
non-`Unknown` and Loss lists. Same-lane conflict or cross-lane exact-status
disagreement is a hard validation error.

Static verification has parser errors 0 and a PASS built-in self-test. A
synthetic 11-case lane, mapping substitution, filename/config mismatch,
partial replica, root agreement, inequality, equality, and zero-credit legacy
surfaces were exercised. A prior apparent census mutation was refuted: a
second shard path crossed a native PowerShell `-File` boundary and rebound as
`CensusPath`; the reported hash was that shard's hash, while the real census
remained exactly `436E...D7BE`. No census mutation occurred.

The analyzer intentionally does **not** validate runner journals or the
Cargo-exit chain. It emits `DOM_B2_D4_CHAIN_AUDIT_FENCE` with
`EXTERNAL_MECHANICAL_AUDIT_REQUIRED` and `satisfied=false`. Consequently its
terminal scientific certificate remains pending for final PASS/KILL until an
independent complete chain certificate is joined by the adjudicator. There is
no parameter that lets a caller self-attest this fence.

## Provenance-chain auditor

The read-only chain auditor has parser errors 0, no write or Cargo commands,
and 39/39 self-tests PASS. It independently discovers frozen-family journals
and META sidecars while requiring the caller to list every journal explicitly.
For each credited PRIMARY chain it requires:

`GATE -> SOURCE_FENCE -> RUN -> SOURCE_FENCE_POST -> BINARY_FENCE -> RESULT`

Replica credit adds an exact `PRELAUNCH_GATE`. Every chain binds lane, case,
start/completed prefix, runner, all live source entries, census, code ID,
compiler/configuration, wrapper, binary, raw/META/Cargo-exit hashes and bytes,
and elapsed/journal wall below 600 seconds. Open RUNs are uninspected and
uncredited. Orphan META/RESULT, malformed grammar, overlap, reversal of lane
order, post-DONE activity, source drift, or journal mutation fails closed.

A final certificate must reconstruct all 3,648 roots in all 11 cases for each
lane, with zero open RUNs and aborts, three `COMPLETE` lane summaries, terminal
DONE, and chronology/final-fence/race-check/result all PASS. Historical OS
memory and process readings cannot be reconstructed after the fact; they
remain attestations in the hash-bound runner journal. The final caller must
hash the auditor before and after capture and keep any operational wrapper
separate from the pure certificate whose DONE line is final.

Checkpoint 1 is a valid PARTIAL chain certificate only for its sealed prefix.
It is not a final completeness certificate.

## Final adjudicator and fail-closed decision boundary

The read-only adjudicator was frozen before replica measurement. Static
verification found parser errors 0, no writes, no redirections, no Cargo
invocations, 28/28 self-tests PASS, and 29/29 frozen support snapshots PASS.
It accepts explicit analysis and pure chain-certificate paths plus their
caller-supplied SHA256 values, reads them as strict UTF-8, and rehashes inputs
after parsing.

It independently verifies the analyzer schema and science, verifies final
chain schema and completeness, then requires a one-to-one cross-certificate
join for every credited raw on filename, lane, case, start, shard result, raw
bytes/hash, derived META filename, and META hash. Individually favorable
certificates over different shard sets cannot be combined.

The complete-evidence decision order is:

1. `SCOPED_KILL` for a recomputed replicated mapped reversal or case-9/10
   mismatch after every structural/provenance/replica/join fence passes;
2. `NULL_REMAINS_PARKED` when there is no scoped KILL but any primary case is
   attacker Loss, because favorable Loss is unsupported in v1;
3. `PASS_REOPEN_EVIDENCE` only with no reversal/control mismatch, no Loss, and
   every complete fence satisfied; or
4. `HARD_ABORT` for explicit analyzer failure/status disagreement, malformed
   complete certificates, split evidence sets, source/support/hash/race drift,
   or another binding evidence-fence failure.

The adjudicator is complete-chain-only. Missing final analysis, a PARTIAL
chain, or a failed/non-emitted chain certificate yields `NO_ADJUDICATION`, not
an automatic NULL or hard abort. The report must apply the original partial
NULL/hard-abort bars to separately sealed partial evidence. An explicit
analyzer ABORT/VALIDATION_ERROR is checked before this boundary so a real
status disagreement cannot be hidden by a partial chain.

An optional Loss document is allowed only when Loss was actually triggered,
is recorded with `loss_document_authoritative=false`, and cannot improve the
v1 disposition. Supplying it when no Loss exists is rejected.

## Recorded hostile self-review and outcomes

The exact chain and adjudicator candidates were repeatedly challenged before
freeze. Earlier failed candidates and captures were rejected rather than
treated as authority.

| Refutation attempt | Recorded outcome |
|---|---|
| Let open, unknown, or abort journal activity survive final mode | Earlier chain candidate refuted; final mode now requires exact grammar, zero open RUNs/aborts, and terminal fences. Negative tests PASS. |
| Combine locally valid invocations with overlapping time or reversed lanes | Earlier chain candidate refuted; global non-overlap, run ordering, lane ordering, and post-DONE fences added. Tests PASS. |
| Credit a changing active raw or wrong fixed shard end | Earlier chain candidate refuted; unmatched RUN is uninspected/zero-credit and exact frozen end is checked. Tests PASS. |
| Preserve a snapshot manifest while changing a live source/verifier | Earlier chain candidate refuted; all live entries are hashed at start/end and traversal is rejected. Tests PASS. |
| Hide a >=10-minute aborted invocation or late runner exception | Earlier chain candidate refuted; completed and aborted wall bounds plus late-exception association added. Tests PASS. |
| Mishandle a PASS-gate/operator-stop race, equal event ticks, DONE counts, or stop case | Earlier chain candidates refuted; exact reason grammar, journal-index tie break, footer counts, and case-sensitive fields added. Tests PASS. |
| Hand-transcribe a credited META hash | Refuted the first partial capture: one hex digit was missing. The old partial/static/prereg hashes are rejected; corrected partial `536175...AF29` has 20/20 exact credit-line matches. |
| Let prose overclaim what the chain script verifies | Refuted twice; six semantic overclaims and one residual-scope overclaim were corrected. Final contract `24F875...2A34` passed review. |
| Collide replica and PRIMARY names or run replica lanes together | Refuted by lane-only names, create-new paths, lane/global locks, and exact mapping. The cross-runner final-instruction race remains disclosed and operationally gated. |
| Manufacture a census mismatch via multi-path PowerShell invocation | Refuted as argument rebinding; the apparent hash belonged to the second shard, and the census stayed unchanged. |
| Combine analyzer certificate A with unrelated chain certificate B | Earlier adjudicator draft refuted; exact one-to-one raw/META join added. Split-brain test PASS-rejected. |
| Use a self-attested Loss note to obtain favorable reopen | Earlier adjudicator draft refuted; v1 Loss capability frozen unsupported. Tests show the note cannot improve disposition. |
| Force NULL before an independently replicated scoped KILL involving Loss | Earlier adjudicator draft refuted; scoped reversal/control decision now precedes favorable-only Loss fence. Tests PASS. |
| Append fatal content after chain DONE or forge a completed prefix | Earlier adjudicator draft refuted; DONE must be final and prefixes are independently reconstructed. Tests PASS-rejected. |
| Mask explicit analyzer disagreement with a PARTIAL chain | Earlier adjudicator draft refuted; explicit analyzer failure is parsed before completion preflight. Test confirms hard-failure precedence. |
| Forge summaries, lane cases, comparisons, runner approvals, or scope | Exact field counts and independent recomputation reject all tested variants; exhaustive-`F4` promotion test PASS-rejected. |
| Cause support or input mutation during adjudication | Start/end hashes cover 11 bound support artifacts, 17 live code files, the adjudicator, and explicit certificate inputs. Independent 29/29 support check PASS. |
| Find hidden writes or Cargo execution in the adjudicator | AST census: 0 writes, 0 redirections, 0 Cargo; self-tests confirm read-only behavior. |

Surviving residuals are explicit: certificate tools do not rerun the solver;
historical resource state remains journal-attested; a failed/non-emitted final
chain produces no adjudication; favorable Loss is unsupported; and fixed-pair
scope never becomes exhaustive `F4`.

## Final scientific measurement tables - pending

**Do not fill these tables from an open raw or the current moving journal.**
Populate them only from a sealed final analyzer certificate whose credited set
has an exact one-to-one final chain certificate.

| Case | Roots | PRIMARY aggregate / W-U-L / nodes / wall | D6_OFF aggregate / W-U-L / nodes / wall | SECOND_TT aggregate / W-U-L / nodes / wall | Root agreement |
|---:|---:|---|---|---|---|
| 0 | 329 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| 1 | 312 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| 2 | 312 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| 3 | 330 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| 4 | 313 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| 5 | 330 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| 6 | 313 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| 7 | 330 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| 8 | 313 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| 9 | 383 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| 10 | 383 | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| **Lane total** | **3,648** | **[PENDING 3,648/3,648]** | **[PENDING 3,648/3,648]** | **[PENDING 3,648/3,648]** | **[PENDING 3,648 matches each]** |

| Frozen decision row | PRIMARY | D6_OFF | SECOND_TT | Final replicated result |
|---|---|---|---|---|
| `0 <= max(1,2)` | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| `3 <= 4` | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| `5 <= 6` | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| `7 <= 8` | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| `9 == 10` | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| Historical case-0 reproduction | **[PENDING]** | **[PENDING if needed]** | **[PENDING if needed]** | **[PENDING]** |

Final analyzer certificate path/SHA256: **[PENDING]**

Final pure chain certificate path/SHA256: **[PENDING]**

Cross-certificate join: **[PENDING]**

Non-`Unknown` replica fences: **[PENDING]**

Loss list: **[PENDING]**

Optional Loss documentation path/SHA256: **[PENDING / OMIT IF NO LOSS]**

Final adjudicator output path/SHA256: **[PENDING]**

Binding disposition: **[PENDING - NO VERDICT]**

Final hostile recheck against the actual certificates: **[PENDING]**

Final phase manifest: **[PENDING]**

## Completion and cold-gater checklist

1. Freeze PRIMARY only after its runner is terminal, its lock is absent, and
   no Cargo process remains. Do not infer completion from current live raws.
2. Run D6_OFF first with exact replica runner SHA `B9E6...953C8`; preserve all
   raw/stdout/stderr/exit/META/journal files. Do not start or restart PRIMARY.
3. After D6_OFF is terminal and all replica locks are absent, run mandatory
   SECOND_TT with the same frozen runner. Keep every invocation below 10
   minutes and apply the RAM/no-foreign-Cargo gate before every launch.
4. Invoke the exact analyzer SHA `862B...BED5` with explicit canonical shard
   paths and the exact three runner mappings. Seal the complete output without
   editing it.
5. Invoke chain auditor SHA `E899...83C8` with every frozen-family journal
   explicitly listed and final mode enabled. Preserve a pure certificate whose
   DONE is its final nonempty line; keep command/exit/pre-post hash wrapper
   evidence separate.
6. Invoke adjudicator SHA `3C6E...9924` with exact expected hashes for the
   analyzer and pure chain certificates. Supply Loss documentation only if
   Loss was triggered, understanding it cannot improve v1.
7. Recheck the strict verifier SHA
   `9990D38618DA2204351E328CA0143BE2AEF98BB3001E4A0462CF346B707F2CE8`,
   all frozen support hashes, source snapshot, census, and binary.
8. Hostilely re-evaluate the actual final headline, split-brain join, lane
   coverage, chronology, decision order, Loss behavior, and fixed-pair scope.
9. Replace every pending field above, assign exactly one binding disposition,
   then seal this report, all final certificates/wrappers/raws/tooling, and the
   unchanged verifier in a new final SHA256 manifest. Verify it entry by entry.

Cold-gater authority at this live handoff is checkpoint 1, not the moving
queue. `DOM_B2_D4_CHAIN_AUDIT_PARTIAL_RAW.log` is the authoritative partial
certificate; `DOM_B2_D4_CHECKPOINT1_HASHES_RAW.log` gates its full supporting
set and records the moving journal only as the exact 59,673-byte prefix. For a
final disposition, the final analyzer certificate, pure final chain
certificate, and final adjudicator output must agree exactly. Until then,
Phase 3C remains **NO VERDICT / PRIMARY CONTINUING**.
