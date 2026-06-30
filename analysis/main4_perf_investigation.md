# hexfield_main_4 self-play throughput investigation

**Date:** 2026-06-22
**Question:** Why does LIVE `hexfield_main_4` self-play run ~10 pos/s when the known-good `hexfield_main_3` ran ~15-20 pos/s (~1.5-2x slower per move)? Is the GPU fully fed, or is main_4 host-bound?

**Verdict (one line):** The slowdown is ~100% GPU **co-tenancy contention** with a healthy `main_2`. pcr-192, the KataGo divergence flags, batch-feeding, and game length are all **measured null**. The GPU is fully fed *in aggregate* across co-runners but each single self-play stream gets ~half the NN-eval ceiling and shows host gaps.

---

## 1. WHY main_4 is slower — quantitative decomposition

The decisive metric is **NN-eval/s** = `flushed_states / elapsed_seconds` (the GPU forward feed rate). The entire gap lives here:

| Metric | main_4 (ep40-47, contended) | main_3 (ep43-54 healthy, near-solo) | Ratio |
|---|---|---|---|
| moves/s | 9.97 | 14.5 | **1.45x** |
| NN-eval/s (flushed_states/s) | 1979 | 2793 | **1.41x** |
| states/move (NN-evals per decision) | 199.1 | 196.2 | 1.01x (**equal**) |
| mean_flush_states (batch size) | ~227 | ~256 | 0.89x (main_3 LARGER) |
| game length (mean plies) | ~105 | ~154 | main_4 SHORTER |

**Identity check:** `moves/s gap (1.45) ≈ NN/s gap (1.41) / visits-per-move ratio (1.01)`. Since per-move work is identical, 100% of the slowdown is lower NN-eval feed rate.

### % attribution per candidate cause

| Cause | Attribution | Evidence |
|---|---|---|
| **GPU contention (co-tenancy)** | **~100%** | NN/s gap (1.41x) == moves/s gap (1.45x); the whole gap is feed rate. Confirmed by 3 natural experiments + same-run proof (below). |
| pcr_fast_visits 192→128 | **~0%** | states/move identical (199.1 vs 196.2). The +17% nominal fast visits are clawed back by early_stop (~1.5-1.6M visits saved/epoch) + cache hits (flushed/nominal=0.67). The bench (search() has no PCR split) couldn't exercise it directly, but live diagnostics settle it: zero marginal evals/move. |
| Divergence-flag host overhead | **~0%** | `on_move_seconds` (ALL divergence logic runs here) is <0.7% of elapsed in BOTH runs. **Isolation bench ran twice:** production (flags ON) is **1.13x FASTER** than parity (flags OFF) — divergences are net-positive (early_stop/lcb on in production). Live divergence-revert boundary (ep40 all-ON → ep41-47 reverted) = **0% throughput change**. |
| Batch feeding / assembly | **~0%** | `no_progress_flushes == flush_count` (100%) every epoch in BOTH runs — flushes are select-exhaustion-driven, queue NEVER reaches flush_target=1024. main_4 batches are NOT smaller than main_3's (227 vs 256, main_3 larger). The gap is fewer flush *cycles*/s (8.7 vs 12.0, =1.38x slower/cycle) because each forward gets a contended slice — a forward-time limit, not assembly. |
| Game length | **~0%** | At matched mean ply (±6, 47 paired epochs) main_3 NN/s is still 1.60x higher. main_4 plays SHORTER games yet is slower — favors main_4, ruling length out. |

### Three corroborating natural experiments

1. **main_2 downtime (the real cause):** main_3's "fast" reference epochs (ep47-54) completed 2026-06-19 06:47–12:34, *inside* main_2's hang (last epoch ep46 @ 06-17 20:40, next not until 06-20 02:27); main_1 dead since 06-16. So **main_3 ran near-solo on the GPU.** The prompt's "main_3 ran 3-way and was still faster" premise is FALSE on the timestamps. main_4 runs continuously co-tenant with a *healthy* main_2.
2. **Same-run proof on the unchanged co-runner:** main_2 (identical net+code) ran ~3150-3300 NN/s **solo** (06-17 ep3-45, peak ep40 = 3419 NN/s, 21.6 mv/s) but dropped to ~1700-2050 NN/s the moment it ran alongside main_4 (06-20/22) — the SAME ~1.7-1.8x drop main_4 shows. Pure contention, not a main_4 defect.
3. **Aggregate ceiling:** main_4 1986 + main_2 1761 = **~3747 NN/s ≈ the single-tenant ceiling (~2800-3400 NN/s)**. The GPU's forward throughput is the shared bottleneck; each co-runner gets ~half.

### Why the serve forward is identical between runs

soft_policy / cell_q / opp_policy heads are **train-only** (model.py:609-613, explicitly excluded from `forward_policy_value` serve path). All serve perf work (FlexAttention serve, build_attn_bias rewrite, deferred-decode, f16 H2D) predates main_3 and is in BOTH. main_4's policy is actually *sharper* (root entropy ~1.4 vs ~1.9-2.0), so support/batch sizes are not larger. No heavier-forward explanation exists.

---

## 2. IS THE GPU FULLY FED?

**In AGGREGATE: YES. For a SINGLE stream: NO (host-gapped).**

- **Aggregate:** With both runs (and any 3rd stream) active, GPU util pins to 93-100% at 180-205W, mem 11.9/12GB. Sum of co-runners' NN/s ≈ the solo ceiling. The GPU *can* and *does* saturate.
- **Single stream:** When only main_4 self-play feeds, nvidia-smi shows avg ~75-86% util with repeated dips to 42-59% (host-gap signature), 142-180W. The FlexAttention serve forward is light enough that one stream's per-decision host loop (select/expand/H2D/backup) cannot keep the GPU pinned. So each run is **its-own-share-bound**: partly host-gapped on its own loop, but the dominant limiter is that it only gets a ~half slice of an aggregate-saturated GPU.

main_4 is **co-tenant-bound, not divergence-host-bound and not batch-assembly-starved.**

---

## 3. RANKED perf fixes

| # | Fix | Type | Expected recovery | Risk | Changes learning? |
|---|---|---|---|---|---|
| **1** | **De-overlap main_4 & main_2 self-play** (pause/serialize main_2, or stagger so only one does inference at a time) | ops/scheduling | **~1.4-1.8x → main_4 to ~14-17 pos/s** (full gap) | LOW | No (throughput only). But main_2 loses its share — net cluster throughput unchanged; this is a *prioritization* choice, not a free win. |
| **2** | Raise batch occupancy: `active_games` 96→128/160 and/or `virtual_batch_size` 4→6 | config-only | ~few % (secondary host-gap close when less contended; mean_flush 227 vs 384 max) | LOW-MED | No. **Validate against the `active_games == games_per_epoch` cohort-refill regression in memory.** Relative win only; test under matched contention. |
| **3** | Revert `pcr_fast_visits` 192→128 | config-only | **≤5-13%, likely ~0** (states/move already identical) | LOW | **YES — affects search depth on fast moves / data quality.** Do this for data-quality reasons if any, NOT for speed. |
| **4** | Disable inert divergences: `pruned_dynamic_cpuct` (no-op at c_scale=0), `nucleus_f64` (moot at widening.mass=0.95<1.0) | config-only | **~0%** | LOW | Minimal (inert by construction). Harmless cleanup, zero pos/s. |
| **5** | "Reduce divergence host cost in code" | code | **~0%** (on_move <0.7%; parity is SLOWER) | — | **Do NOT pursue.** Measured null; turning divergences off makes search slower (early_stop disabled). |

### Key cautions
- **Fix #1 is the only lever that recovers the full gap**, but it trades main_2's throughput for main_4's — it does not increase total GPU work, it reprioritizes. If both runs matter equally, the current ~50/50 split is already the aggregate-optimal state and there is no slowdown to "fix" — main_4 at ~10 pos/s under co-tenancy is expected and healthy.
- **Fix #3 changes learning behavior** (fewer fast-move visits) — keep pcr-192 unless a play-quality reason justifies reverting.
- Do **not** chase pcr-192 or divergence-host-overhead as causes; both are measured null across the isolation bench, the revert-boundary natural experiment, and the on_move_seconds budget.

---

## Caveats
- Quantitative evidence is from per-epoch diagnostics (`metadata.result.selfplay.scheduler` in `epoch_0000NN.json`) measured across DIFFERENT real contention environments, plus live nvidia-smi/pos-s samples. The solo-vs-contended comparison is a **natural experiment** (file mtimes), not a controlled main_4-alone run (that requires pausing main_2).
- The isolation bench (`analysis/improve_probes/perf_bench.py`, ckpt epoch_000046.pt) contends with the live run; only the production-vs-parity RATIO (1.128, 1.126 across two runs) is load-bearing, and it is stable.
- `HexfieldMctsSession.search` has no fast/full PCR split, so pcr-192-vs-128 was settled via live diagnostics (states/move identical), not the bench.
