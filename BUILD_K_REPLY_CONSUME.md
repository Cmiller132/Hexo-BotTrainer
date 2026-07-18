# G2R8 build memo — Q8 `K_reply` consumption

Date: 2026-07-17 (America/New_York)

Base/HEAD before edits: `b8b67bf5e190dbc9ffcdcb898696f9ae1ce4728a`
(`shadow(g2r7): Q8 K_reply kernel validated across 220,160 fallback fires —
zero counterexamples, zero verdict disagreements`). No commit is permitted.

## Binding five-clause contract (verbatim)

The following is copied verbatim from the hostile review's signed production
pre-filter contract (`../hunt-quiet-locality/REVIEW_QUIET_LOCALITY.md`, “Q8
production pre-filter sign-off”, clauses 1–5). The proof basis is
`../hunt-quiet-locality/PROOF_QUIET_LOCALITY.md`, §4.2 and §5.

1. `P` is nonterminal, `P.current_player()==A`, and
   `P.phase()==SecondStone { first }`. The stored coordinate must be part of the
   state/root binding. Do not apply this rule at Opening or FirstStone.
2. Recompute from the current board, after the stored first placement:
   `D=A.other()` and
   `T_D(P)={W: active_player(W)==Some(D) and count_D(W) in {4,5}}`.
   Define `E_P(W)` explicitly as the current empty cells of `W`; "urgent" means
   `T_D(P)` is nonempty.
3. Compute `Win1_A(P)` from the same full legal set and retain every placement
   whose application immediately returns terminal winner `A`.
4. Compute
   `BlockAll_D(P)={c in Legal(P): for every W in T_D(P), c in E_P(W)}` and retain
   `Win1_A(P) union BlockAll_D(P)`.
5. If `T_D(P)` is empty, return all of `Legal(P)`. Do not substitute a locality
   tier for this vacuous case.

## Production hook and exact trigger

The consumption seam is the full-legal Consume fallback in
`NarrowCompatSearch::prove_choice`
(`packages/hexfield_eq/rust/src/tss_solver.rs:3754`):
only after the ordinary threat-creating candidates exhaust does it call
`state.write_legal_moves`. Q8 is derived from that unfiltered full legal
snapshot, before the existing `PairContext` quotient filter and deterministic
sort. It therefore replaces only the fallback Choice enumeration licensed by
the proof and does not widen the theorem to the ordinary frontier or to a
defender Universal node.

The solve reads `TSS_K_REPLY_CONSUME` once (`tss_solver.rs:261`). Consumption is enabled only when
its value is exactly `1`, the active width profile already consumes the quiet
fallback, and `k_reply_kernel` reports urgent. The kernel itself checks the
nonterminal state, claimant identity, and exact `SecondStone { first }` phase;
then it consumes the current post-first active-window family
(`tss_solver.rs:3351-3417`). The hot path uses the engine's incrementally
maintained `live_threats` set, which is specified as an exact mirror of
`entries().filter_map(threat_player)`
(`packages/hexo_engine/rust/src/tactics.rs:349-353`) and is exposed by the
documented exact-mirror iterator (`tactics.rs:415-420`). It has a randomized
create/block/undo mirror regression (`tactics.rs:840`).
The G2R8 trigger regression independently rederives the result from a full
`entries()` scan (`tss_k_reply_shadow.rs:498`). A pure claimant count-five
window is the exact non-mutating equivalent of applying each member of the same
full legal set to compute `Win1_A`: on a nonterminal position, a legal
placement wins immediately exactly when it fills such a length-six window.
Because every candidate is legal/empty, membership in every defender window
is membership in every `E_P(W)`, giving the required intersection.

The node's existing `ThreatAnalysis` is recomputed immediately before Choice
dispatch. At a claimant Choice node, its opponent family is exactly `T_D(P)`,
so `eligible && analysis.opp_threat_count > 0` is the exact urgency precheck
(`tss_solver.rs:3763`). Only urgent nodes construct the retained vector. The
nonurgent retained view borrows the already-written `Legal(P)` rather than
copying it.

When any eligibility condition fails or `T_D(P)` is empty, the candidate list
is unchanged. Flag-off does not compute Q8 and preserves the previous fallback
enumeration and ordering.

## Promotion and gates

The G2R7 `k_reply_kernel` is promoted from `#[cfg(test)]` to production-local
code. Its telemetry records and frozen/measurement harness remain test-only.
The production search receives one immutable per-solve boolean and consumes
the kernel only at the seam above. Promotion is default-off and requires:

1. focused frozen-witness and default-off regression tests;
2. paired flag-off/flag-on verdict identity over the forcing-19 10k/100k
   recipe, `double_fork_compact`, and the fixed-seed 200 human roots;
3. verifier acceptance of every emitted certificate, with certificate byte
   equality reported but not required;
4. no WIN for any forcing-NO row;
5. the official one-process 2 GiB all-19 gate with the flag on; and
6. paired node-expansion/wall measurements plus urgent-node retention counts.

Any flag-on/off verdict difference is a STOP condition: freeze the change and
write up the counterexample rather than promoting it.

## Build and gate results

### Green evidence

- Frozen required-remote witness and trigger matrix: PASS. Opening,
  FirstStone, wrong claimant, and nonurgent SecondStone cases match a full
  `WindowStore::entries()` scan; the urgent witness remains exactly `538 -> 1`.
- Default-off release suite: **97 passed, 0 failed, 25 ignored**.
- `double_fork_compact` consume profile (10k, horizon 45): flag-off/flag-on
  verdict **WIN/WIN**, both certificates verifier-accepted and structurally
  identical; expansions **409 -> 395** (-3.423%); paired wall
  **34.696 -> 33.546 ms** (-3.314%); the one urgent fallback was exactly
  **478 -> 1**, and consumed/shadow sets agreed.
- Fixed-seed human-200 campaign: all **200/200 verdicts identical**; all
  **13/13 paired hard certificates** verifier-accepted and structurally
  identical. Aggregate expansions were **1,870,025 -> 1,870,025** (0%). Paired
  wall was **81,049.072 -> 84,912.289 ms** (+4.766%). Flag-on traversal saw
  14,575 fallback fires and 11,158 urgent consumptions; by phase band:

  | band | roots | expansions off -> on | wall off -> on | urgent/fires | median full -> K |
  |---|---:|---:|---:|---:|---:|
  | ply <= 12 | 67 | 670,000 -> 670,000 | 26,234.305 -> 26,408.390 ms (+0.664%) | 1,424/3,008 | 1,423 -> 2 |
  | ply 13-40 | 67 | 630,007 -> 630,007 | 22,724.493 -> 23,871.343 ms (+5.047%) | 5,515/6,509 | 1,085 -> 2 |
  | ply > 40 | 66 | 570,018 -> 570,018 | 32,090.274 -> 34,632.556 ms (+7.922%) | 4,219/5,058 | 1,192 -> 2 |

- Official flag-on gate, one process with
  `TSS_BACKWALK_TT_BYTES=2147483648`: **PASS,
  `CORPUS_DONE failures=0`**, test time 438.03s. All 14 WIN rows closed and all
  five NO rows stayed non-WIN. The official harness selects
  `WidthOptions::vcf_pair_complete`, so this is the required regression gate;
  it does not itself enter the round-3 Consume fallback.

### Incomplete/negative evidence

The forcing-19 consumption campaign is **not complete**. Twelve of 19 rows
completed at the G2R7 10k/100k recipe with zero verdict differences, two
verifier-accepted/identical hard-certificate pairs, and no completed NO row
returning WIN. Seven 100k rows remain without completed paired evidence:
`0l4291i_live`, `94gnnol`, `lz60mfb`, `mvp2lvc`, `zrugh2x`,
`hayes_20260712_turn16`, and `hayes_20260712_placement31`. The first five hit
the non-official 10-minute command bound (including an OFF-only attempt for
`0l4291i_live` and `94gnnol`); the two Hayes rows were not started after the
repeated bound failures. Exact orphan test PIDs were stopped after each bound;
no partial result is counted.

Across the 12 completed forcing rows, expansions were
**280,002 -> 280,002** (0%), while paired wall was approximately
**217.526 -> 611.860 s** (+181.28%). Flag-on traversal recorded 8,858 fallback
fires and 432 urgent consumptions. Two severe trajectory outliers contradict
the expected modest-global story even though local retention is tiny:

- `0hz3hty`: UNKNOWN/UNKNOWN, 10k -> 10k expansions,
  **70.245 -> 291.420 s** (+314.86%), 3 urgent consumptions.
- `strongloss_a_prefix6`: UNKNOWN/UNKNOWN, 100k -> 100k expansions,
  **0.987 -> 173.567 s** (+17,487.49%), 400 urgent consumptions; median local
  retention was approximately **889 -> 2**.

The wall regressions are not trigger-scan or shadow overhead: `0hz3hty` was
repeated with shadow disabled and remained **70.942 -> 291.084 s** (+310.31%).
Exact pruning changes which much-more-expensive descendants consume the fixed
node budget.

## Disposition

The implementation is present and remains **default OFF**, but G2R8 is **not
promotion-green**. Verdict identity is green on every completed pair (including
all human roots), the official regression gate is green, and local branching
collapse is proven/measured. However, the mandatory forcing-19 identity
campaign is incomplete under its command bound and the completed forcing
measurements contain major wall-time regressions. Do not enable
`TSS_K_REPLY_CONSUME=1` as a production default from this evidence.
