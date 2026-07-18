# Group 2, round 7 — K_reply quiet-fallback shadow

Date: 2026-07-16 (America/New_York)

Base confirmed before edits: `2430fc4771027c23f73d7a43a4de187574db557c`.
No git commit is permitted or will be attempted.

## Checkpoints

### C0 — binding scope and proof read

- Initial free physical RAM: **13.02 GiB**. Builds and solves will remain
  serialized, use `CARGO_TARGET_DIR=.target-codex`, target
  `x86_64-pc-windows-msvc`, and `--test-threads=1`.
- Read the required inputs in order: the NQ2 proof, round-3 and round-6
  progress records, then the sibling hunt's frozen-witness test body.
- Production fidelity: the full-legal Consume fallback is unconditional after
  the ordinary frontier fails. “Quiet” is a post-placement property only for
  a nonterminal SecondStone completion.
- Q8 shadow contract: at a nonterminal attacker SecondStone position `P`, scan
  all active defender windows and define urgency by at least one defender
  count-4/count-5 window. `K_reply(P)` is exactly attacker immediate wins union
  legal cells belonging to every such defender window. No threat-only index
  may substitute for the full scan.
- The canonical frozen replay has 36 placements, claimant Player 0 at
  `SecondStone { first: (6,0) }`, and unique winning completion `(6,-6)`.
  Q8 must report urgent and the singleton kernel `{(6,-6)}`.

Checkpoint boundary: no source file has been edited. Next locate the post-C1
quiet-fallback seam and existing default-off shadow telemetry patterns.

### C1 — shadow and harness implementation drafted

- Located the post-C1 seam in `NarrowCompatSearch::prove_choice`: the Consume
  fallback begins only after the ordinary frontier has exhausted.
- Added test-only (`cfg(test)`) telemetry behind `TSS_K_REPLY_SHADOW`. Release
  library builds contain no telemetry record, sink, env lookup, or fallback
  branch.
- At each instrumented fallback the shadow snapshots the full engine legal
  universe before pair deduplication or sorting. Q8 eligibility is exactly a
  nonterminal claimant `SecondStone` node. Urgent defender windows are derived
  by scanning every `WindowStore` entry for defender count 4/5 and claimant
  count zero. Kernel membership is exact engine win-now union membership in
  every such defender window.
- Search generation/order is untouched. A record is marked WIN only after the
  existing recursive proof and certificate-node allocation succeed. A Q8 miss
  snapshots the exact `RootBinding` and the measurement harness panics with
  `Q8_COUNTEREXAMPLE` immediately.
- Added a default regression for the 36-placement frozen witness, an official
  corpus-row on/off status/node/TT/certificate identity gate, and separate
  forcing-19, `double_fork_compact`, and fixed-seed 200-root human-corpus
  measurement helpers. The human sampler carries the leaf-width hunt's three
  phase bands, per-band Fisher-Yates/XorShift derivation, and master seed
  `0x9E3779B97F4A7C15`; quotas 67/67/66 total exactly 200.

Checkpoint boundary: implementation is drafted but not yet compiled. Next run
targeted rustfmt/check and the focused fixture after the mandatory RAM guard.

### C2 — fixture and exercised identity green

- `tss_round7_k_reply_frozen_witness`: **PASS 1/0**. Exact replay has 538
  legal completions; Q8 reports urgent and `K_reply={(6,-6)}`; the other 537
  completions all leave `(6,-6)` as Player 1's immediate six.
- Replaced the initial measurement-only win-now implementation (clone/apply
  every legal cell) with its exact full-window equivalent: attacker-pure
  count-five windows identify `Win1_A`. This preserves Q8 and avoids hundreds
  of shadow-only state clones per urgent node.
- `tss_round7_k_reply_identity`: **PASS 1/0** on the exercised
  `double_fork_compact` corpus row. Telemetry OFF and ON produced identical
  status, **2,409 nodes**, TT hits, and byte-equal certificate; ON recorded one
  fallback fire. The identity harness uses an open horizon for sensitivity;
  the official consume witness remains separately bound to horizon 45 and its
  expected 409 nodes.
- The first broader identity-helper draft was stopped by its 120 s command
  bound while searching official rows for a non-vacuous fire. It emitted no
  disagreement and was replaced by the deterministic exercised corpus row.

Checkpoint boundary: core shadow fidelity and behavior identity are green.
Next run the focused `double_fork_compact` telemetry measurement, then the
forcing-19 and human sweeps.

### C3 — focused measurement green; forcing harness bounded

- `double_fork_compact` telemetry: **PASS**, WIN/409 and independently
  verified. One fallback fire, one urgent node, `|quiet|=478`, `|K_reply|=1`
  (0.002092 retention), one proved urgent WIN edge, one hit: **100%**.
- An initial all-19 implementation unnecessarily retried every UNKNOWN row at
  100k after its 10k observation. It was stopped and replaced with one
  documented rung per row: first closing rung when 10k/100k, and a 100k clamp
  for official 1M/20M or NO ladders. This avoids duplicate measurement work.
- The first stopped wrapper left a Rust test child alive; the exact two Cargo
  parents and `hexfield_eq` child were identified by command line and stopped
  before the next build. No other process was touched.
- The corrected aggregate remained stdout-buffered for an hour, so it was
  safely stopped with no miss/failure signal but no usable aggregate. The
  harness now supports `TSS_R7_CORPUS_ID`, emits unbuffered row start/done
  markers, and prints the exact `(quiet,kernel)->count` histogram so bounded
  row chunks can be recombined without percentile loss.

Checkpoint boundary: no Q8 miss has occurred, but the forcing measurement is
not yet complete. Next compile the chunkable harness and run the official rows
in auditable serialized groups.

### C4 — required measurement sweep complete; no Q8 miss

All solves were serialized with 256 MiB TT and a fail-fast free-RAM guard.
The forcing rows used exactly one documented rung each: ten rows at 10k and
nine rows at 100k (official higher rungs clamped to 100k). Exact histograms
from four bounded chunks were recombined with the harness percentile rule.

| Class | Fires | Urgent | Urgent % | median `quiet→K` | p90 `quiet→K` | median / p90 retention | proved urgent WIN hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| official forcing-19 | 216,668 | 160 | 0.073846% | 940→2 | 3,914→2 | 0.212993% / 0.224972% | 0/0 (N/A) |
| `double_fork_compact` | 1 | 1 | 100% | 478→1 | 478→1 | 0.209205% / 0.209205% | **1/1 (100%)** |
| human 200 | 3,491 | 177 | 5.07018% | 971→2 | 1,385→2 | 0.1996% / 0.3824% | 0/0 (N/A) |

Human phase-band detail (fixed leaf-width sampler, seed
`0x9E3779B97F4A7C15`, quotas 67/67/66, cap 10k, horizon `ply+50`):

| Band | Fires | Urgent | Urgent % | median `quiet→K` | p90 `quiet→K` |
|---|---:|---:|---:|---:|---:|
| ply <=12 | 1,595 | 21 | 1.3166% | 597→2 | 1,383→2 |
| ply 13–40 | 1,028 | 80 | 7.7821% | 971→2 | 1,188→2 |
| ply >40 | 868 | 76 | 8.7558% | 1,028→2 | 1,459→2 |

- Human root statuses were 12 WIN / 1 LOSS / 187 UNKNOWN. None of the hard
  roots proved through an urgent fallback edge.
- The official capped Consume attempts likewise had no proved urgent fallback
  WIN edge. Therefore their hit-rate denominator is empty, not a claimed
  empirical hit. The one nonempty denominator is the headline witness: 1/1.
- Aggregate across all three classes: **220,160 fires, 338 urgent**, one proved
  urgent fallback WIN, one Q8 hit, **100%**. No `Q8_COUNTEREXAMPLE`, WIN-vs-LOSS
  disagreement, or official-NO false WIN occurred.

Checkpoint boundary: measurement is complete and green. Next run only the
mandatory telemetry-OFF exit gates and final format/diff audit.

### C5 — telemetry-OFF exit gates and final audit green

- Final default release suite: **96 passed / 0 failed / 22 ignored**. The count
  adds the standing frozen-witness regression plus four explicit round-7
  ignored helpers to the prior 95/18 baseline.
- Final round-5 compatibility identity: **PASS 1/0** in 0.11 s.
- Final telemetry-OFF `double_fork_compact` consume witness: **WIN / 409 nodes
  / 51 TT hits / 67,177,998 peak TT bytes / 37 ms / verifier accepted**.
- The separate exercised telemetry identity remained green: ON/OFF status,
  node count, TT hits, and certificate were identical on
  `double_fork_compact`; ON recorded one fallback fire.
- A final gate attempt correctly refused to start at 9.47 GiB free RAM. After
  the shared host recovered to 12.94 GiB, every final command passed a fresh
  >10 GiB guard.
- `git diff --check` passes. `rustfmt --edition 2021 --check` passes on the new
  round-7 module and the two small harness-visibility files. The intended
  solver/new-module code was formatted with Rust 2021.
- Running rustfmt on `lib.rs` initially recursed through its modules and caused
  formatting-only churn. Initial clean status plus the saved exact churn patch
  allowed restoration of all nine unrelated files; the final tracked surface
  is only `lib.rs`, `tss_solver.rs`, `tss_corpus.rs`, and
  `tss_spare_corpus.rs`, plus new `tss_k_reply_shadow.rs`.
- HEAD remains `2430fc4771027c23f73d7a43a4de187574db557c`. No commit was
  attempted. All pre-existing untracked prompt/PID/native/target/round-5
  artifacts remain untouched.

## Final disposition

**Complete and green.** Q8 was implemented exactly in shadow-only test code,
the frozen singleton regression is standing, 220,160 fallback fires were
measured with 338 urgent nodes, and the only proved urgent fallback WIN edge
hit `K_reply` (1/1). Telemetry OFF and ON preserve search identity; production
generation, ordering, TT behavior, and semantics are unchanged.

## Regeneration log

Commands and results will be appended at each checkpoint boundary.
