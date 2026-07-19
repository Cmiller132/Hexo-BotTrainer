# R-CREL-6 Phase 3C: DOM-B2 depth-4 checkpoint 1

Disposition: **NO VERDICT — PRIMARY CONTINUING AT THE SEALED POINT.**

This is a complete, gateable artifact set for one immutable partial snapshot.
It is not the final DOM-B2 phase report and does not replace
`HUNT_REPORT_DOM_B2_D4.md`. Its only measurement authority is the provenance
chain snapshot captured at `2026-07-18T21:02:08.6823436Z`: 20 closed primary
invocations, 1,140 credited roots, cases 0--2 complete, and case 3 credited
through root index 186. Any queue progress after the first 59,673 bytes of
`DOM_B2_D4_QUEUE_V3_RUN03_RAW.log` is deliberately outside this checkpoint.

## Frozen identity and scope

- Input HEAD: `5f5da82a04d14f645fdbf08ea96937a428182cde`.
- Candidate: K=1, budget-2 spare-stone domination, depth-4 evidence gate.
- Opt-in: `TSS_DOM_B2_D4_REOPEN=1`; the machinery is `cfg(test)`, ignored,
  and default-off.
- Code identity: `DOM_B2_D4_PRIMARY_V3`.
- Primary configuration: D6 on, 512 MiB TT, 480,000 ms internal bound.
- Planned replicas: D6 off with a 512 MiB TT, then D6 on with a 256 MiB TT.
- Cargo profile: `.target-hunt`, release, `x86_64-pc-windows-msvc`, serial
  `--test-threads=1`.
- Every invocation requires at least 10 GiB available memory, at least 5 GiB
  free physical memory, and zero pre-existing host-wide `cargo.exe` processes.
  Credited and aborted invocation wall time must be strictly below 600 seconds.
- The strict verifier is unchanged at SHA256
  `9990D38618DA2204351E328CA0143BE2AEF98BB3001E4A0462CF346B707F2CE8`.

The binding scientific preregistration is `DOM_B2_D4_PREREG_RAW.log`, 7,618
bytes, SHA256
`9A76F6F9E24F551E69A8A513FC33031D5EB5F350B129B99354C6FD81959B1981`.
It was sealed before the first DOM-B2 Cargo launch by
`DOM_B2_D4_PREREG_SHA_RAW.log`. The later provenance-auditor contract does not
claim to retroactively preregister those scientific bars.

This selected-child matrix is narrower than an exhaustive first-action
statement. It evaluates 11 fixed completed defender turns. It does not
enumerate every legal defender second placement and therefore is not an
exhaustive `F4(first)` comparison. Even a future PASS only reopens a separately
fenced proof round; it is not a proof or deployment decision.

## Enumeration architecture fixed before completeness

Each of the 11 parents and ordered defender pairs is replayed and checked for
the frozen turn state, threat properties, legality, nonterminal state,
coverage, and SPLIT/H_CONTAINING classification. For the resulting attacker
FirstStone child, the engine legal-move enumeration and an independent
reference enumeration must match exactly after sorting by raw `(q,r)`.

The frozen census then partitions each complete legal root universe into
disjoint half-open shards. An immediate win is exact; every other root uses
the bounded exact reference for the remaining three plies. A closed prefix
`[start,next_start)` may be credited, but a stopped root and every later suffix
receive no value and may be recomputed. Case aggregation is the exact
maximizing recurrence: any `Win` yields `Win`; otherwise any missing or
Incomplete root poisons the case; otherwise any `Unknown` yields `Unknown`;
otherwise all roots are `Loss`. Gaps, extra indices, coordinate/fingerprint
mismatches, conflicts, or fabricated stopped values fail closed.

The authoritative census is `DOM_B2_D4_CENSUS_RAW.log`, 292,756 bytes, SHA256
`436E9F6C4A93CDB611EEF6495A01F510615174A2897271CA1092A0E7422DD7BE`.
Its frozen widths are:

| Case | Class | Roots | Fingerprint |
|---:|---|---:|---|
| 0 | SPLIT | 329 | `827EEB0FCB78C698` |
| 1 | H_CONTAINING | 312 | `7C4092D562D3E619` |
| 2 | H_CONTAINING | 312 | `319A510062631E51` |
| 3 | SPLIT | 330 | `27C689B6D4D0DC33` |
| 4 | H_CONTAINING | 313 | `09E733EF333378BA` |
| 5 | SPLIT | 330 | `530A55C8F49F0911` |
| 6 | H_CONTAINING | 313 | `324FB3CEA1CDCA7E` |
| 7 | SPLIT | 330 | `CCE0D5A475F109B6` |
| 8 | H_CONTAINING | 313 | `F9FC2F2E9BB41D72` |
| 9 | H_CONTAINING | 383 | `FF353A8E0556E088` |
| 10 | H_CONTAINING | 383 | `B57A000A7F5C6800` |

The widths sum to exactly 3,648 roots per lane. Final three-lane completeness
therefore requires 10,944 collision-free lane-roots, 3,648 and 11 complete
cases in each of PRIMARY, D6_OFF, and SECOND_TT.

## Binding verdict bars

Under defender preference `attacker Loss > Unknown > Win`, PASS /
REOPEN-EVIDENCE requires all of the following:

1. All 11 case aggregates are exact, non-Incomplete, and gapless.
2. Case 0 reproduces the historical attacker `Unknown`, or a discrepancy is
   independently resolved D6-off and at the second TT size.
3. `case0 <= max(case1,case2)`, `case3 <= case4`, `case5 <= case6`, and
   `case7 <= case8`.
4. Cases 9 and 10 agree for the lifted equality control.
5. Every compared inequality and every non-`Unknown` status is reproduced in
   both replica configurations. An attacker `Loss` additionally needs the
   frozen stock/fast qualification because inherited Q0 covered only 3/16.
6. No production/verifier edit or reference-consistency failure occurs.

KILL for this scoped matrix requires a replicated mapped reversal or a
replicated case-9/case-10 disagreement. It does not establish a universal
`F4` counterexample. NULL / remains parked applies if productive bounded work
still leaves a root or subtree incomplete, a gap/conflict survives, Loss
qualification fails, or the exact evidence cannot meet the PASS fences. A
replay/census mismatch, D6/TT exact-status disagreement, assigned stopped
value, production-solver change attributable to this phase, or verifier diff
is a hard abort.

## Sealed measurement result

The authoritative partial is
`DOM_B2_D4_CHAIN_AUDIT_PARTIAL_RAW.log`, 11,469 bytes, SHA256
`536175702A92D617A8AB0666FDEF60457334D66836314B4A63B708252CFFAF29`.
It was generated read-only from exactly the first 59,673 bytes of the moving
primary journal. That prefix has SHA256
`D71A641D9B5542DCDD7102DC539251D822CB27DC3280138F590387E67EE92FDE`.
The checkpoint manifest records it as a `PREFIX`, not as a whole-file hash.

The auditor credited exactly 20 closed provenance chains and their 20 raw,
20 META, and 20 Cargo-exit files. It credited 1,140 of 3,648 primary roots
(31.25%) and 1,140 of the planned 10,944 three-lane roots (10.4167%). Every
credited root status is attacker `UNKNOWN`.

| Lane/case | Credited roots | Sealed state |
|---|---:|---|
| PRIMARY case 0 | 329/329 | exact `Unknown`; historical case-0 status reproduced |
| PRIMARY case 1 | 312/312 | exact `Unknown` |
| PRIMARY case 2 | 312/312 | exact `Unknown` |
| PRIMARY case 3 | 187/330 | indices 0--186 `Unknown`; aggregate remains Incomplete |
| PRIMARY cases 4--10 | 0/2,365 | not reached at this seal |
| D6_OFF | 0/3,648 | no replica artifact existed at seal |
| SECOND_TT | 0/3,648 | no replica artifact existed at seal |

The next primary invocation, case 3 start 187, appeared as one unmatched RUN.
The auditor classified it `ACTIVE_UNCREDITED`, did not open or inspect its
footerless raw, and assigned it zero credit. Seven credited raw files contain
a final `INCOMPLETE` attempt row after their closed exact prefix; those seven
rows are likewise outside the 1,140 credited roots and were subsequently
eligible for recomputation. Older attempts and later live progress do not
change this sealed accounting.

This evidence already reproduces the historical case-0 `Unknown` and makes
the first mapped relation true in the primary lane because cases 0, 1, and 2
are all exact `Unknown`. That is only a partial observation: the remaining
primary relations, equality control, and both mandatory replica lanes are
absent. It cannot trigger PASS, KILL, or final NULL.

## Provenance-chain gate

The frozen read-only auditor is
`scripts/dom_b2_d4_chain_audit.ps1`, 116,000 bytes, SHA256
`E8991F499D90FEF81B9CCF492CC17047ED24863159F3F83279A7634A9DE383C8`.
Its pre-replica contract is
`DOM_B2_D4_CHAIN_AUDIT_PREREG_RAW.log`, 10,321 bytes, SHA256
`24F87552069BAE910EC87656CD7EF7A40BD40D504A10A984E7472D2361B42A34`.
At that freeze there were zero D6_OFF or SECOND_TT data files, META files, or
journals. The static capture is
`DOM_B2_D4_CHAIN_AUDIT_STATIC_RAW.log`, 4,715 bytes, SHA256
`85AF8E3F55C02FD27D2D6A5649A9A04915E1ED5CF104D07E87B165293A21D9C7`:
parser errors 0, write commands 0, Cargo commands 0, and 39/39 self-tests PASS.

For every credited primary invocation the chain is `GATE -> SOURCE_FENCE ->
RUN -> SOURCE_FENCE_POST -> BINARY_FENCE -> RESULT`. It binds the primary
runner, 20-entry live-source snapshot, census, code ID, compiler and search
configuration, child wrapper, binary, output paths, and exact raw/META/exit
bytes and hashes. The partial reported journal grammar PASS, state machine
PASS, chronology PASS, race check PASS, no aborts, and one open run with zero
credit. Historical OS resource readings cannot be reconstructed later; the
hash-bound journal and runner attest those readings. This limitation is
explicit and remains open.

## Hostile self-review

`DOM_B2_D4_CHAIN_AUDIT_HOSTILE_REVIEW_RAW.log`, 11,835 bytes, SHA256
`71BBA9B14D03107A728F16A41B459FACF2741C7D3DAFFC341265C6AF6FA8F8E6`,
records attempts to refute the checkpoint's provenance headline. Earlier
candidates were rejected and revised rather than silently retained:

| Refutation attempt | Outcome at final frozen candidate |
|---|---|
| Open/unknown/abort activity admitted by final mode | Earlier candidate refuted; final mode now fails closed; focused tests PASS. |
| Locally valid chains with overlapping or reversed global chronology | Earlier candidate refuted; global non-overlap, run ordering, lane ordering, and post-DONE fences added; tests PASS. |
| Credit from a changing active raw or wrong frozen shard end | Earlier candidate refuted; unmatched RUN is uninspected and zero-credit; range tests PASS. |
| Unchanged manifest while a live source/verifier changes | Earlier candidate refuted; all 20 live entries are rehashed at start/end and traversal is rejected; PASS. |
| Long aborted invocation or late runner exception escaping association | Earlier candidate refuted; every completed/aborted wall is bounded and late exceptions associate with the RUN; tests PASS. |
| PASS gate followed by a stop-file race | Earlier candidate refuted; only the exact allowed post-gate stop reason is accepted; negative tests PASS. |
| Equal-tick CASE_DONE/DONE, forged count, or invalid post-fence result | Earlier candidate refuted; journal-index tie break and exact footer/result truth tables added; tests PASS. |
| SECOND_TT starting before D6_OFF is terminal | Earlier candidate refuted; partial and final phase-order rules added; tests PASS. |
| Mixed-case stop reason | Earlier candidate refuted; canonical case-sensitive grammar required; test PASS. |
| Hand transcription of one META hash | Refuted the first capture: one hex digit was missing. The old partial/static/prereg hashes are rejected. The corrected partial has 20/20 exact credit-line matches and is the sole authority here. |
| Contract prose broader than the script | Refuted twice: six overclaims and one residual scope overclaim were corrected. Final contract hash `24F875...2A34` passed exact semantic review. |

The surviving residuals are sharp: a partial snapshot cannot bound future
queue completion; historical memory/process truth remains runner-journal
attestation; unbound raw/stdout/stderr/exit files stay outside credit; and the
scientific status still requires the separately frozen aggregate and verdict
bars. The review found no remaining blocker to using this exact snapshot as a
truthful partial checkpoint.

## Legacy and untrusted material

Only exact `DOM_B2_D4_PRIMARY_V3` chains from PRIMARY RUN03 are in this
checkpoint. PRIMARY RUN01/RUN02, all earlier schema/code identities, pilots,
smokes, recovery experiments, and any raw without an exact META+RESULT chain
receive zero completeness and verdict credit. Their retention on disk is not
evidence. The active case-3/start-187 raw also receives zero credit regardless
of any later contents. The 1,140-root authority cannot be increased by reading
the current moving queue; a later checkpoint or final report needs a new
explicit seal.

## What a final phase artifact must still prove

1. PRIMARY finishes all 3,648 roots and 11 cases with an exact terminal DONE.
2. Only after PRIMARY is terminal, D6_OFF completes its independent 3,648-root
   lane; only after D6_OFF is terminal, SECOND_TT completes its lane.
3. All three lanes are gapless, collision-free, mutually status-consistent,
   serial in time, and have no abort, open RUN, post-DONE activity, or source
   mutation.
4. The final chain auditor is externally hash-checked before and after capture,
   is passed every exact frozen-family journal explicitly, and ends with three
   `COMPLETE` lane summaries plus chronology, final fence, and race check PASS.
5. The separately frozen scientific aggregate reproduces all 11 rows in all
   lanes, evaluates every inequality/equality bar, and qualifies any Loss.
6. A final hostile review attempts to refute completeness and the scientific
   disposition. The authoritative final report, raws, scripts, source/binary
   bindings, and verifier are sealed in a new SHA256 manifest.

Until those conditions hold, the only truthful disposition is **NO VERDICT**.
`DOM_B2_D4_CHECKPOINT1_HASHES_RAW.log` is the gate for this partial artifact
set; its journal entry is explicitly prefix-scoped and the manifest does not
self-hash.
