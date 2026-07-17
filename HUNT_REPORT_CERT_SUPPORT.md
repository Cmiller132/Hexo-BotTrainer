# NQ3 machine hunt — certificate support locality

Date: 2026-07-16  
Branch/base: `hunt/cert-support` / `2430fc47`  
Harness: `packages/hexfield_eq/rust/src/cert_support_hunt.rs`  
Verdict: **(c) refuted for today's strict certificate format.** The proof body is often local after an explicitly non-production rebind/clock translation, but the accepted certificate is deliberately bound to the complete position.

## Headline

The current strict verifier has no bounded, root-fixed cell support. Its first check is:

```text
cert.root == RootBinding::from_state(state)
```

`RootBinding::from_state` enumerates every occupied cell and owner and binds current player, phase, placement count, and terminal result. Equality with that complete sparse occupancy also asserts that there is no extra stone anywhere else on the engine's conceptually unlimited board. At shared certificate DAG nodes, `ReplayKey` again stores the complete replay position. Therefore a root that differs by even one legal stone outside any proposed finite set is rejected before the proof body is read.

Consequences measured here:

- Strict unchanged-certificate transfer: **0/180 at every K in {1,2,4,8}**.
- Exact rejection diagnosis for every strict rejection: **the full `RootBinding` equality failed first**.
- Strict support-hash reuse versus today's full-position key: **0 additional collisions, 1.000× reuse multiplier**.
- No soundness finding. The verifier rejected the targeted far-defender count-5 construction, both unchanged and after the shadow rebind/clock shift.
- Shadow only: after replacing the full root binding and translating absolute clocks, the unchanged proof body accepted **169/180 (93.9%)**, **173/180 (96.1%)**, **150/180 (83.3%)**, and **140/180 (77.8%)** for K=1,2,4,8. This is promising engineering evidence, not a theorem and not behavior of the current certificate.

## Definitions

The campaign definitions are:

- **READ SUPPORT of a certificate: the set of board cells that any verifier check touches — window contents at every node's position delta, zone cells, quiet/defender edges, budget/role/exposure witnesses, D6 image cells.**
- **FRAME CELLS: cells in the support that are EMPTY at the root (the certificate asserts their emptiness at use time).**
- **SUPPORT RADIUS: max hex distance from the certificate's root-active region (stones placed by the cert's line + root stones it reads).**

For a frame lemma, READ SUPPORT must be a fixed dependency set for the certificate: any two roots agreeing on it must give the same verification result. A data-dependent iteration over the current root's occupied list is not a finite support, because adding a new cell changes the list and the result. Under that required interpretation, today's strict support is **domain-wide/unbounded**, its frame is every otherwise-empty board cell, and its support radius is not a bounded local radius. All root stones are inside strict support, so the fraction of root stones outside strict support is exactly zero.

The report also gives a finite **body footprint**. This is not READ SUPPORT. It is the union of certificate move/edge/commutation coordinates, named six-cell windows, and verifier-rederived zone cells. It approximates the payload a future relative/local certificate might hash after a proof removes all global obligations.

## Verifier-code audit: contribution by check

The audit followed `tss_verify.rs` and the shared `threats_shared.rs` analysis called by it.

| verifier check | cell contribution / dependency |
|---|---|
| Root binding (`verify_certificate`, `RootBinding::from_state`) | Every root occupied coordinate and owner; equality also asserts absence of every additional coordinate. It separately binds player, phase, absolute placements, and terminal result. This alone refutes bounded support. |
| Arena/metadata preflight | Certificate structure, deadlines, node cores, and the full bound root-stone vector; no smaller board binding is derived. |
| Shared-DAG replay memo (`ReplayKey`) | Complete occupied position and owners at every shared node, plus allowed commuted replies, player, phase, absolute placements, and terminal result. |
| Common node role checks | Current player, phase, terminal flag, and absolute placement clock. |
| `with_move` / `apply_with_delta` | The move cell's emptiness and legal-store membership; placement updates the 18 length-six windows containing the cell and tests whether any becomes a six. |
| Attacker placement WF | Scans all current claimant stones and all root stones, asking whether any is within distance 8 of the proposed move. Distant additions can make WF more permissive. |
| `OrCompletion` | Move cell, all six named witness cells after the move, role/terminal/absolute completion clock, and WF anchors. |
| `Win` leaf | All six named witness cells, claimant/opponent counts, empty completion cells, remaining-turn budget, absolute resolution clock, and WF anchors. |
| `Loss` leaf | Every named witness's six cells and empties; global live-threat analysis for `own_win_now`, defender budget, and hitting-set family; absolute resolution clock and WF anchors. |
| Universal precheck | Global live-threat analysis, every explicit edge cell, allowed commuted cells, and their legality. |
| Instant dispatch | All live claimant threat-window cells used to rederive the extendable hitting kernel. The optional debug oracle additionally enumerates every legal omitted move and runs lambda-1 after it; production strict verification did not enable that debug arm. |
| Full universal node | Exact equality between represented replies and the complete legal-move store. Any remote legal-store growth invalidates the old edge family. |
| Commutation | Both ordered pair cells, their edge identities, single-move legality/phase, two-move legality, and pair outcomes in both orders. |
| Zone summary | Replays every descendant edge/move; protects move cells and named-window empties; recomputes local defender budget. |
| Zone uniform exposure | Complete legal store, protected cells, root/replay occupied membership for pending cells, seed-band distance checks, and **every touched window entry** when collecting remote defender exposures. For local budget ≥6 it takes the complete legal set. |
| D6 | Verification itself does not search D6 images. `d6_remap_certificate` transforms every root coordinate, move/edge/commutation coordinate, and witness window before strict verification; those image cells then enter the same checks above. |

The finder in `tss_solver.rs` constructs/rebases these witnesses and zones, but certificate acceptance is controlled by the independent paths above. `tss_core.rs::hard_value_from_verified` exposes a hard value only after this exact verifier accepts.

## Measurement 1: official forcing WIN rows and `double_fork_compact`

Method: production pair-complete WIN solve, fresh solver at 10k then 100k nodes, 64 MiB TT, official forcing roots, unlimited semantic horizon. The official file has 19 total rows, of which 14 are labeled WIN. Twelve of 14 produced a certificate by 100k. The two remaining official WIN rows and `double_fork_compact` are honestly marked unsolved; no certificate statistics are invented.

Strict columns are identical for every certificate: support/frame/radius = **unbounded**, root stones outside = **0%**. The finite columns below are the non-authoritative body footprint.

| row | population | rung | nodes | TT hits | cert nodes | body cells | body frame | body radius | root stones outside body |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0hz3hty | 21 | 10k | 2,412 | 2,263 | 291 | 28 | 25 | 2 | 85.71% |
| 0l4291i_live | 63 | >100k, unsolved | — | — | — | — | — | — | — |
| acly7kb | 93 | 10k | 75 | 0 | 55 | 23 | 19 | 1 | 95.70% |
| g2xx6wl | 139 | 10k | 4,107 | 1,948 | 287 | 34 | 31 | 2 | 97.84% |
| hu01jk4 | 149 | 10k | 380 | 0 | 523 | 36 | 32 | 1 | 97.32% |
| jh7yo7y | 35 | 10k | 2,119 | 311 | 757 | 38 | 36 | 1 | 94.29% |
| jnzzmcm | 67 | 10k | 9,798 | 1,331 | 2,307 | 50 | 42 | 2 | 88.06% |
| lz60mfb | 41 | >100k, unsolved | — | — | — | — | — | — | — |
| xsnfyll | 13 | 10k | 82 | 1 | 38 | 21 | 19 | 2 | 84.62% |
| zrugh2x | 45 | 100k | 41,734 | 11,895 | 1,866 | 68 | 63 | 1 | 88.89% |
| strongloss_a_prefix6 | 9 | 100k | 16,126 | 9,895 | 725 | 54 | 52 | 2 | 77.78% |
| strongloss_b_prefix8 | 11 | 10k | 1,099 | 287 | 70 | 27 | 24 | 2 | 72.73% |
| hayes_20260712_turn16 | 31 | 100k | 11,664 | 3,016 | 1,621 | 46 | 45 | 2 | 96.77% |
| hayes_20260712_placement31 | 32 | 100k | 11,664 | 3,016 | 1,620 | 45 | 44 | 2 | 96.88% |
| double_fork_compact | 36 | >100k, unsolved | — | — | — | — | — | — | — |

For the 12 solved official rows, body support median/p90/max is **38/54/68** cells. That compactness is real evidence for redesign potential, but it cannot be substituted for strict READ SUPPORT.

## Measurement 2: 200 deterministic human-corpus roots

Recipe was ported from the leaf-width hunt:

- Source `E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl`.
- Every nonterminal `FirstStone` node from decisive games is eligible.
- Three placement bands: ≤12, 13–40, >40.
- Per-band deterministic Fisher–Yates with XorShift and master seed `0x9E3779B97F4A7C15`; quotas 67/67/66, total **200**.
- Pair-complete `SolveGoal::Win`, 30k nodes, 64 MiB TT, absolute horizon `placements + 50`, fresh solver per root.

Results: **34/200 WIN certificates**; the remaining 166 were UNKNOWN screens, not negative verdicts.

| quantity over the 34 WIN certificates | median | p90 | max |
|---|---:|---:|---:|
| strict READ SUPPORT | unbounded | unbounded | unbounded |
| body-footprint cells | 22 | 42 | 53 |
| root population | 31 | 81 | 149 |

The body footprint stays small while population grows, which explains the attractive prospective multiplier. The current strict format nevertheless reads/binds the complete position.

## Measurement 3: transfer

For each transferable solved certificate (45 `FirstStone` roots; the one `SecondStone` root was skipped), four deterministic trials per K were generated. One K unit is a full two-stone turn by each color: +4 placements, equal color counts, and return to the identical current player and `FirstStone` phase. Every added cell was legal, nonterminal, and outside the finite body footprint.

| K | trials | unchanged strict accepted | strict rate | shifted/rebound shadow accepted | shadow rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 180 | 0 | 0.00% | 169 | 93.89% |
| 2 | 180 | 0 | 0.00% | 173 | 96.11% |
| 4 | 180 | 0 | 0.00% | 150 | 83.33% |
| 8 | 180 | 0 | 0.00% | 140 | 77.78% |

Every unchanged strict rejection has the same exact first cause: the mutated complete occupancy and absolute placement count differ from `cert.root`. No later check runs. This is not a mysterious outside-support read; it is the explicit global binding that refutes the claimed finite support.

The shadow column clones the certificate, replaces `root`, and adds the placement delta to every leaf/completion clock, semantic horizon, and zone build horizon. It is deliberately not an unchanged certificate. Its failures name the proof-round obstacles: exact full-legal reply families, global live-threat analysis, zone exposure/legal-store growth, possible premature outcomes, and shared-state/global WF contracts. Because the strict experiment already rejects at root equality, the harness does not mislabel one of these later conditions as the strict rejection cause.

Frozen random-failure examples are printed as `TRANSFER_EXEMPLAR` records by the harness. For example `g2xx6wl`, K=8, trial 0 is rejected even after shifted rebinding, demonstrating that removing only administrative binding/clocks is insufficient.

## Measurement 4: adversarial far threat

Target: `0hz3hty`, whose certificate resolves 18 placements after the root, so it is not an immediate same-turn win. Outside the body footprint, the harness added balanced filler turns and built a remote defender count-5 window:

```text
window = [(6,-1),(7,-1),(8,-1),(9,-1),(10,-1),(11,-1)]
added  = [(-8,0),(-16,0),(6,-1),(7,-1),(-24,0),(-32,0),
          (8,-1),(9,-1),(-40,0),(-48,0),(10,-1),(-56,0)]
```

The unchanged certificate was rejected by root binding. More importantly, the shifted/rebound shadow certificate was also rejected. Thus the later verifier paths did notice the remote urgent defender formation; there is **no soundness finding**. This is a named proof boundary: any future local lemma must show how global opponent-threat/goal checks are represented in support, not merely preserve the claimant's local windows.

## Measurement 5: prospective TT win at forcing rungs

Coarse method: use solver expanded nodes as an upper-bound proxy for position-keyed entries and certificate arena nodes as the fragment population. Across the 12 solved official rows there were **101,260 expanded nodes**, **33,963 existing TT hits**, and **10,160 certificate nodes/fragments**.

For the current strict contract, support equivalence is full-position equivalence. Therefore support hashing produces exactly the same equivalence classes as today's full-position keys:

| proxy population | today full-position key | current strict-support key | additional reusable collisions |
|---|---:|---:|---:|
| expanded-node/TT-entry upper bound | 101,260 | 101,260 | 0 |
| certificate fragments | 10,160 | 10,160 | 0 |

Prospective multiplier under the current verifier: **1.000×**. The much smaller body footprints quantify why a redesigned relative certificate could be valuable, but no collision multiplier was fabricated without a defined local equivalence relation and a proof that later global checks are preserved.

## Sharpened NQ3 conjecture

The requested unconditional statement,

> a strict-verifier-accepted certificate remains valid at any root agreeing with the original on support(C)

is **refuted for every bounded `support(C)` under the current certificate format**.

A theorem-shaped replacement for the proof round is:

> Let `C_rel` be a relative certificate obtained from a strict certificate by (1) replacing complete `RootBinding` and shared `ReplayKey` occupancy equality with a proved support projection, (2) expressing all completion/resolution/zone clocks relative to the root, and (3) recording enough support to make move legality, complete universal reply obligations, terminal/no-new-six facts, live threat and hitting-set families, attacker-WF anchors, zone seed/exposure sets, commutation outcomes, player/phase, and D6 mapping invariant. Then `C_rel` remains valid at any nonterminal root with the same player and phase that agrees with the original on that support and satisfies the recorded legal-store boundary/no-new-threat conditions.

Named obstacles for that proof:

1. **Full binding/complement absence** — must be removed or projected; it is the present refutation.
2. **Absolute clocks** — placement count, leaf resolution, semantic horizon, and zone build horizon need a proved relative translation.
3. **Legal-store growth** — a remote stone can add legal moves; full Universal nodes currently require exact equality, while zones use the global legal set.
4. **Remote threats and goals** — `analyze`, dispatch, loss leaves, and zone exposure depend on live windows away from the claimant line.
5. **No new six / premature terminal** — replayed certificate moves and added stones must not create an earlier terminal outcome.
6. **WF anchors** — all claimant/root stones can witness the distance-8 placement rule.
7. **Shared DAG identity and commutation** — projected keys must preserve occurrence obligations and pair outcomes.
8. **D6 image closure** — support projection must commute with certificate remapping.

Exit verdict: **(c) refuted for current strict certificates**. There is a credible conditional redesign conjecture, supported by 78–96% shifted/rebound random transfer and compact body footprints, but it is not yet (a), and the shadow failures rule out calling it an unconditional (b).

## Reproduction and gates

All cargo runs used `CARGO_TARGET_DIR=.target-hunt`, one process, and `--test-threads=1`. Launch free RAM was 12.59 GiB for the corrected full campaign and 12.75 GiB for the adversarial test. The full campaign took 61.8 s after compilation, well below 10 minutes. No 2 GiB profile was used. Transfer trials use distinct seeds `SEED ^ rotate(root_ply) ^ rotate(K) ^ trial*0xD1B54A32D192ED03`; an earlier pilot with pairwise-colliding trial seeds was discarded and is not reported.

PowerShell:

```powershell
$ram = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
if ($ram -lt 9) { throw "wait: free RAM below 9 GiB" }
$env:CARGO_TARGET_DIR = '.target-hunt'
cargo test -p hexfield_eq --release cert_support_campaign -- --ignored --nocapture --test-threads=1

$ram = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
if ($ram -lt 9) { throw "wait: free RAM below 9 GiB" }
$env:CARGO_TARGET_DIR = '.target-hunt'
cargo test -p hexfield_eq --release cert_support_far_threat_adversarial -- --ignored --nocapture --test-threads=1

$ram = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
if ($ram -lt 9) { throw "wait: free RAM below 9 GiB" }
$env:CARGO_TARGET_DIR = '.target-hunt'
cargo test -p hexfield_eq strict_root_binding_is_a_global_obligation -- --test-threads=1
```

Optional downscoping controls: `TSS_CERT_SUPPORT_FORCING_CAP`, `TSS_CERT_SUPPORT_HUMAN_N`, `TSS_CERT_SUPPORT_HUMAN_CAP`, `TSS_CERT_SUPPORT_TT_BYTES`, and `TSS_CERT_SUPPORT_HUMAN`.
