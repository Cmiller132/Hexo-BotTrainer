# R-MAXSOLVE opening-atlas report

Date: 2026-07-19 (America/New_York)

## Result

The campaign certified 587 formerly UNKNOWN atlas rows:

| | Before | Added | After |
|---|---:|---:|---:|
| WIN | 2,532 | 68 | 2,600 |
| LOSS | 6 | 519 | 525 |
| Certified | 2,538 | 587 | 3,125 |
| UNKNOWN | 45,364 | -587 | 44,777 |
| Total | 47,902 | 0 | 47,902 |

Of the 587 upgrades, 577 are at source plies 0--11 and 10 are deeper. The
first-11 UNKNOWN population is therefore 45,314 -> 44,737; the deeper UNKNOWN
population is 50 -> 40.

The decisive raw is `OPENING_ATLAS_MAXSOLVE_RAW.txt`. It contains exactly 68
WIN and 519 LOSS `ATLAS_ROW` records plus a
`canonical_verifier_rejects=0` completion marker. Every row was emitted only
after the canonical certificate was accepted by the normative `TssVerifier`.
Every new WIN also has a nonempty principal variation that replays to a
literal claimant six (`win_line_terminal=1`).

## Positive yield

| Method | Roots certified | WIN | LOSS | Accepted-result nodes (sum / median / max) | Accepted-result solve time |
|---|---:|---:|---:|---:|---:|
| Narrow unbounded LOSS, ply 11 | 309 | 0 | 309 | 345,252 / 564 / 13,478 | 31.513 s |
| One-placement lift, ply-11 LOSS -> ply-10 WIN | 15 | 15 | 0 | 25,306 / 2,023 / 2,721 | 2.094 s |
| Complete-turn lift, ply-11 LOSS -> ply-9 WIN | 15 | 15 | 0 | 25,326 / 2,023 / 2,721 | 2.078 s |
| Narrow unbounded LOSS, plies 7--10 | 200 | 0 | 200 | 202,160 / 734.5 / 6,019 | 24.695 s |
| One/complete-turn lift, ply-9 LOSS -> ply-8/7 WIN | 38 | 38 | 0 | 79,207 / 1,967.5 / 6,019 | 6.114 s |
| Narrow unbounded LOSS, ply 12+ | 9 | 0 | 9 | 825 / 6 / 781 | 0.115 s |
| Quiet-width unbounded LOSS escalation, ply 12+ | 1 | 0 | 1 | 110,341 / 110,341 / 110,341 | 27.018 s |
| **Total** | **587** | **68** | **519** | | |

The 68 WINs are structural consequences of decisive same-claimant children,
not unverified value propagation. For every lift, the child was solved again,
one or two legal claimant `Choice` nodes were prepended, the full parent
certificate was rebuilt, and `TssVerifier` accepted that parent certificate.

Upgrade depth breakdown:

- LOSS: ply 9 = 96, ply 10 = 104, ply 11 = 309.
- Deeper LOSS: plies 37, 38, 41, 42, 49, 51, 52, and 54 = one each; ply 50 = two.
- WIN: ply 7 = 19, ply 8 = 19, ply 9 = 15, ply 10 = 15.

## Proof reuse and retrograde closure

The initial corpus-edge lifts found all 68 WINs above. A later exact-state
transposition census generalized this beyond recorded game prefixes:

- It indexed all 3,125 decisive positions, enumerated all 12 D6 images (37,233
  distinct transformed positions after orbit collapse), and compared them
  against all 44,777 residual UNKNOWN roots.
- It enumerated both legal one-placement predecessors and complete
  two-placement claimant-turn predecessors, including the exact
  `SecondStone { first }` root binding.
- It found zero additional residual parents.
- As a positive control, allowing already-upgraded parents produced exactly
  68 parents / 142 routing lines. The harness re-solved them and reproduced
  all 68 expected WIN IDs, with no missing or extra ID; every reconstructed
  certificate passed `TssVerifier` and every WIN line was terminal.

Thus the structural fixed point is dry: the generalized D6/transposition join
recovers exactly the already-found lifts and no more.

## Zero-yield escalation evidence

### Complete first-11 quiet/ranked passes

The quiet-turn profile was run to natural convergence on every residual
first-11 root in both directions:

- WIN ply 11: 7,089 roots at 2k; 60 residual at 10k; 2 residual at 100k.
  The last two converged at 12,913 and 10,005 nodes. Yield: 0.
- WIN ply 10: 7,723 at 2k; 14 residual at 10k; the last root converged at
  12,061. Yield: 0.
- WIN plies 7--9: 19,740 at 2k; 53 residual at 10k; 7 residual at 100k;
  final maximum 21,375. Yield: 0.
- WIN plies 0--6: all 10,185 roots converged below 2k (maximum 10). Yield: 0.
- LOSS plies 0--11: 44,737 roots at 2k; 19 residual at 10k; the last root
  converged at 12,189. Yield: 0.

The ranked-zone-only profile produced the same statuses, node counts, and
expansion counts on the corresponding complete cohorts. Yield: 0.

The wider `round3_consume` LOSS pass over all 7,089 residual ply-11 roots used
exactly 70,890,000 visits and 31,663.1 aggregate solver-seconds at the 10k
rung. Yield: 0. This agrees with the prior eight-hour ply-9 widening squeeze
described in the task context.

### Deeper positions

- Narrow unbounded LOSS proved 9/50 immediately; quiet-width LOSS later proved
  one more at 110,341 nodes. Forty deep UNKNOWNs remain.
- Quiet-width WIN: 39/41 residual roots converged below 10k. The two hard roots
  saturated 100k, 1M, and then 8M visits each with no proof. The 8M runs took
  1,061.355 s and 1,312.316 s. Their 1 GiB TTs admitted 3,005,390 and
  3,026,909 entries and rejected 4,994,552 and 4,973,021 further admissions.
- A 20M attempt crossed the practical memory cliff (about 18 GiB private for
  one process at roughly 8M progress). It was stopped without emitting a row;
  the controlled sequential 8M runs above are the retained ceiling evidence.
- Ranked deep WIN/LOSS matched quiet at the tested rungs. Yield: 0.
- `round3_consume` deep WIN completed 37/40 roots at 2k with zero result; three
  roots were so slow that they did not reach 2k within about ten minutes and
  were stopped without output. The same expensive width had already produced
  no ply-9 gain in the prior squeeze.
- `round3_consume` deep LOSS completed all 40 roots at 2k, 10k, 100k, and the
  final 1M rung. The 100k rung used 4,000,000 visits and 1,102.8 solver-seconds.
  The final rung used exactly 34,742,181 visits (some roots converged below the
  cap), 12,710.6 solver-seconds, and at most 766.5 s for one root. Yield: 0 at
  every rung.

### Default-off levers

- Shared-fragment seed, ply 10: 1,809/1,824 decisive seeds reproduced, 64,086
  stored fragments; target searches made 154,399 lookups with zero hits and
  zero imports. Yield: 0.
- Shared-fragment seed, ply 11: 2,997/3,072 seeds reproduced, 77,685 stored
  fragments; target searches made 442,285 lookups and 12 hits across three
  roots, but zero imports and zero verdicts.
- Deep shared-fragment seed: 53/53 seeds reproduced. The WIN targets made
  207,149 probes and the LOSS targets 1,132 probes, both with zero hits/imports
  and zero verdicts.
- `TSS_K_REPLY_CONSUME=1` on the deep wide-LOSS cohort produced the same
  statuses and node counts as the control at 10k. Yield: 0.
- `TSS_LAZY_FRONTIER=1` was used for the controlled high-node deep runs to
  stay within the memory envelope.
- The interior census gate is inert at `semantic_horizon=u32::MAX` by design.
  `TSS_INCR_DEFENDER` has no runtime reader in this build. The upgrade register
  already marks broad horizon laddering, support hashing, and D6 search-TT
  folding as economically dead, so they were not used to mint values.

## Convergence assessment

The last decisive result was the single deep quiet-width LOSS. After it, the
following independent full or targeted rounds all yielded zero: ranked deep
WIN, ranked deep LOSS, round3 deep LOSS at 2k/10k/100k/1M, K-reply deep LOSS,
seeded deep WIN, seeded deep LOSS, exact all-D6 transposition propagation, and
both 8M quiet deep-WIN probes. The first-11 quiet searches had already
converged naturally, and the full ply-11 wide-LOSS pass plus the prior ply-9
widening campaign were dry.

Further brute force is possible only by spending far more memory/time on the
same saturated trees (for example 20M on the two hard WINs, or multi-million
wide search over thousands of early-ply roots). The observed TT admission
cliffs, the failed 20M memory attempt, and repeated zero-yield rungs make that
an unfavorable continuation rather than a new sound method. On the sound
methods and implemented theorem-backed levers available in this worktree,
this is the measured practical ceiling.

Campaign wall time was approximately 4 hours 33 minutes from the first shard
start (14:17:27) through final validation work, excluding the prior eight-hour
squeeze supplied in the task context.

## Merge and safety validation

- `atlas-web/data/atlas.json` remains exactly 47,902 rows in the same order.
- The changed-ID set is exactly the 587 IDs in
  `OPENING_ATLAS_MAXSOLVE_RAW.txt`; all 47,315 other rows compare equal to the
  frozen pre-run atlas.
- All 2,538 pre-existing decisive rows compare byte-for-byte under the
  builder's compact JSON serialization. Their verdict, claimant, root, and
  win line are unchanged.
- Every changed row was UNKNOWN/certified=0 before and is now a strict
  WIN/LOSS/certified=1 with identical ID, source, source prefix, placements,
  side, phase, orbit, and moves.
- The builder's additive merge is byte-idempotent: a second build changed none
  of the 11 generated atlas/frequency artifact hashes.
- `atlas-web/index.html` carries the regenerated atlas cache-bust token.
- `node atlas-web/selfcheck.mjs` passes every check, including all 47,902
  canonical-ID round trips.
- `frequencies.json`, `frequencies-web.json`, its gzip, and its JSONP wrapper
  are byte-identical to the pre-run copies. Their SHA-256 values are,
  respectively:
  - `907216399E3CB727997654515C40495AE93ACF99CADA714BCD0B51BFBBDFAC10`
  - `EDEC8EA979ABB2186CC3F9F4C16A4DE88FE01FB75C7B80A07477C34CBD7C3A81`
  - `58473254B35EE55EA0FBECE9159124BFDAD4228360C95125F7CCB76EB5C2EFE0`
  - `04F0E5936FB2C4BF252591919B55817F021685A588B3633E769673645D0B021E`
- `tss_verify.rs` has an empty git diff and SHA-256
  `9990D38618DA2204351E328CA0143BE2AEF98BB3001E4A0462CF346B707F2CE8`.
- The modified ignored atlas harness compiled under
  `x86_64-pc-windows-msvc`; its D6 census test passed, and the complete
  68-parent transposition reproduction test passed.
- All Rust changes remain in the existing `cfg(test)` atlas module and all new
  behavior is ignored-test/env gated and default-off. No commit was created.

The temporary comparison snapshot `.maxsolve-baseline-20260719/` remains
untracked because the execution sandbox rejected both recursive and exact-file
deletion attempts. It is not read by the builder and contains only the five
pre-merge atlas/frequency copies used for the frozen-row and hash checks.

One subtlety: the atlas records canonical-root verifier acceptance as the
normative gate. The 12-image D6 remap loop is diagnostic and often accepts
only a subset of remapped certificate images (for many lifts the stored mask
is `0x081`). This campaign did not reinterpret those diagnostics as canonical
rejects and did not weaken the verifier.
