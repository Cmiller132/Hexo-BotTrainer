# Group 2, round 1 — spare-turn / lambda-2 corpus

Date: 2026-07-16 (America/New_York)

Base: `ac3f455f` (`round-9b` certificate-grade `WidePnSearch`)

## Checkpoints

### C0 — source audit started

- Read `docs/TSS_VCF_WIDTH_BRIEF.md`, then the complete §5 of
  `E:\hexo-bot\docs\paper\RZOP_SOLVER_OPTIMIZATION.md`, then §7 of the
  sibling review plan, followed by the forcing-corpus loader/data,
  `DEEP_WIN_MOVES`, and `tss_reference.rs`, in the requested order.
- Confirmed the round boundary: measurement/test code and corpus data only;
  no normative engine changes.
- Confirmed the structural gap in `WidePnSearch::expand`: defender nodes are
  refuted unless `implicit_dispatch`, whose premise is
  `min_hitting_set == b`. The quiet/spare corpus must therefore contain an
  oracle WIN whose reference strategy reaches `k < b` and whose winning OR
  edge is absent from the current forcing-only universe.
- Toolchain: Windows `cargo 1.95.0`, `rustc 1.95.0`. All builds and solves in
  this round use `CARGO_TARGET_DIR=.target-codex` and are serialized.
- The worktree already contained untracked `.codex-round5/`,
  `.target-codex-iso/`, `.target-codex/`, and the current prompt/log under
  `.codex-group2/`. They are treated as pre-existing/user-owned; scratch made
  by this round will be separately named and removed.

Checkpoint boundary: no commit attempted (the operational contract forbids
git commits in this worktree). Continue with oracle harness and candidate
mining.

## Mining log

All finder/engine probes below used a fresh solver, the normative
`vcf_pair_complete` constructor for the forcing baseline, and the 2 GiB TT
ceiling. No oracle and engine solve overlapped.

| Candidate/source | Result | Disposition |
|---|---|---|
| `DEEP_WIN_MOVES` (`tss_bench.rs:169`) | default WIN/4 nodes; wide WIN/2 nodes; derived horizon 6 | Rejected. The historical “lambda-2” label means only that the *root position* has no lambda-1 verdict. Its winning pair creates two disjoint count-4 families and is continuous-forcing, so it cannot witness U12. |
| `deep_universal_fixture` | default WIN/3; wide WIN/2 | Rejected: direct `(4,4)` forcing shortcut. |
| single `(4,4)` blocker variants | wide WIN/2 | Rejected: other direct count-3 extensions remain. |
| `strongloss_a` history prefixes 7/5 | default UNKNOWN (314/5); wide UNKNOWN (2/10) | No finder proof; retained only as negative mining evidence. |
| `strongloss_b` history prefixes 9/7 | default UNKNOWN (314/5); wide UNKNOWN (2/8) | No finder proof. |
| `xsnfyll` history prefix 11 | default UNKNOWN/1,785; wide UNKNOWN/2 | No finder proof. |
| `deep_urgent_spare` | post-root `k=1<B=2`; default UNKNOWN/662; wide UNKNOWN/2 | Structurally valid, but its remote pads make the independent legal frontier 554 cells. Superseded by compact construction. |
| `compact_urgent_spare` | post-root `(3,0)` has `k=1<B=2`; default UNKNOWN/1,075; wide UNKNOWN/2 | Rejected as a WIN: the shared junction is a defender refutation. Retained as a finite-horizon NO candidate. |
| `double_fork_compact` | default finder WIN/2,884 at absolute horizon 45; normative wide UNKNOWN/2 at 1M; post-root `(4,0)` has `k=1<B=2` | Active oracle candidate. A complete 479/479 spare-reply structural sweep found no counterexample, but that sweep is diagnostic only; exhaustive reference result pending. |

The seed mismatch is not a WIN-vs-LOSS disagreement: both engines return WIN
and the certificate horizon is six. It is a terminology/acceptance mismatch,
so the entry is excluded rather than weakening the required wide-nonWIN
witness.

### C1 — first true spare boundary isolated

`compact_urgent_spare` is a legal 28-placement replay at an attacker
`SecondStone` root. P1 has a gapped five whose only hole is `(3,0)`, so P0's
only viable completion both blocks that counter-win and makes P0's capped
five. At the resulting defender `FirstStone`, threat analysis is
`b=2, k=1, opp_threat_count=2, own_win_now=false`; the three remote P0
count-three routes support the intended fork after the hit-plus-spare reply.
The wide forcing gate rejects the root completion and exhausts at two nodes.

Checkpoint boundary: no commit attempted (forbidden by the operational
contract). Independent `tss_reference` depth-9 run launched serialized; no
other cargo/solve lane is active.

### C2 - oracle correction and isolated native run

The C1 `compact_urgent_spare` construction was subsequently refuted at its
shared junction; it is not a WIN candidate. The active construction is now
`double_fork_compact`, a legal 36-placement P0 `SecondStone` root. Its intended
completion `(4,0)` leaves the defender at `b=2, k=1`: `(5,0)` is mandatory and
the other defender placement is a genuine spare. The historical default
finder proves WIN in 2,884 nodes at absolute horizon 45, while normative
`vcf_pair_complete` returns UNKNOWN in two nodes at a 1M cap. The representative
line has two `k<B` Universal placement nodes, one nonforcing completed attacker
turn, and maximum spare nesting one.

Those results are mining evidence, not oracle ground truth. Generic reference
attempts that were manually stopped after five to ten CPU-minutes are recorded
as timeouts, never as verdicts. One native (`target-cpu=native`, one codegen
unit) `tss_reference` depth-9 run was launched alone at 2026-07-16 13:10:08
local time. No other build or solve overlaps it. Its output will be copied
verbatim below before any WIN_PENDING row is frozen.

Checkpoint boundary: no commit attempted (forbidden); oracle pending and no
reference WIN/NO claim promoted.

## Ground-truth table

### C3 - stock-reference feasibility stop

The active native `double_fork_compact` depth-9 reference process was stopped
after 1,202.47 CPU-seconds / 20.06 wall-minutes with no verdict. This is a
timeout, not a reference result. A subsequent source-level lower-bound audit
showed why waiting longer is not a credible regeneration command:

- The smaller `double_fork_dense` root has 403 legal cells after its mandatory
  quiet completion, with one mandatory hit and 402 possible spare cells.
- Both legal orders of hit/spare reach the next attacker turn, for at least
  `2 * 402 = 804` branches.
- A genuine `k<B` root blocks every pre-existing count-4-or-better attacker
  window, so the attacker cannot terminate on the immediately following pair.
  A second defender turn is unavoidable.
- At that turn at least 399 of the original legal cells remain, producing at
  least `399 * 398` ordered defender pairs per first spare branch.
- Therefore the unmodified `tss_reference` recurrence must visit at least
  `804 * 399 * 398 = 127,676,808` nonterminal attacker nodes before it can
  return WIN. At roughly 39 occupied cells, its intentionally independent
  `legal_moves` rebuild attempts more than `10^12` BTree insertions at this
  forced layer alone.

This is structural for the requested witness, not a bad candidate or compiler
setting: `tss_reference.rs` deliberately has no TT, threat pruning, turn
commutation, or checkpointing. A proposed endpoint-preload acceleration was
also rejected correctly: it made eleven live post-root threat windows,
`min_hitting_set=None` at `B=2`, and both default and wide solvers proved the
position as a continuous-forcing WIN in 3/2 nodes. It is not lambda-2.

Two soundness controls were checked with the stock reference:

| Candidate | Reference horizon | Reference result | Nodes | Disposition |
|---|---:|---|---:|---|
| `compact_urgent_spare` | 2 plies | UNKNOWN | 1,601 | Honest finite-horizon NO control. |
| `strongloss_a_prefix6` back-off at 7 stones | 2 plies | UNKNOWN | 129,455 | Honest finite-horizon NO control. |
| `spare_tempo_prefix` | 2 plies | WIN | 2 | Rejected as NO; default and wide independently return immediate WIN. No disagreement. |

No WIN row has been frozen: doing so without an actual exhaustive reference
WIN would falsify the acceptance witness. The remaining decision is external
to the round's stated diff surface: either permit a test-only exact reference
accelerator (which would no longer be the unmodified oracle), or budget a very
long stock-reference run and accept that three independent regenerations are
not cheap.

Environmental note: HEAD is `dba6111d`, a documentation-only gate commit
directly atop requested engine commit `ac3f455f`; `git diff ac3f455f..HEAD`
contains only `.codex-round9b-gate/GATE.md`. Normative engine sources are
therefore the requested bytes.

## Wide-engine baseline (2 GiB TT)

_Pending._

## Diagnosis memo

_Pending._

## Regeneration commands

_Pending._

## Final verification and diff surface

_Pending._
