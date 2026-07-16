# Round-9b official all-19 gate — PASS at ac3f455f

- Date: 2026-07-16, 15:03Z. Runner: orchestrator (single process, serial).
- Commit: `ac3f455f759232ec1e1cfc210cddc90a4558ca06` (round-9b, normative engine).
- Command (from `packages/hexfield_eq/rust`):
  `TSS_BACKWALK_TT_BYTES=2147483648 cargo test --release tss_corpus_check -- --ignored --nocapture`
  (the documented 2 GiB test profile; the bare default is 512 MiB and is NOT
  the official profile — an initial bare-default run was killed and discarded).
- Result: `CORPUS_DONE failures=0`, test wall 436.8 s (round-9 gate: 1871 s).
- 14/14 WIN certified on the ladder; 5/5 NO entries non-WIN. Zero false wins.
- Full log: `final-matrix-19-9b.log` (per-rung status/nodes/wall + GEN_PROFILE).

Highlights vs the round-9 official gate (4daf1961):

| Entry | round-9 | round-9b (this run) |
|---|---|---|
| 0l4291i_live | WIN @20M, 1,831,556 n, 794.3 s | WIN @20M, 1,879,612 n, 177.7 s |
| lz60mfb | WIN @1M, 109,460 n, 32.1 s | WIN @1M, 109,896 n, 8.7 s |
| jnzzmcm | WIN @100k, 14,317 n | WIN @10k, 9,798 n (rung improved) |
| gate total | 1871 s | 436.8 s (4.3x) |

Node counts drifted modestly (0l 1.83M → 1.88M), matching the round-9b
progress note; wall dominates. No entry closed at a worse rung. This
discharges the "official all-19 gate re-run at the 9b tip" pending item:
round-9b `ac3f455f` is now the gate-verified normative engine.
