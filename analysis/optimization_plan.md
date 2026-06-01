# Optimization & implementation plan — dense_cnn Model 1

**Date:** 2026-05-29 · **Branch:** `review/scratch64-optimization-stage` (planning only; the live run
stays stopped at `epoch_000022.pt`).
**Goal this plan is organized around:** make **512 MCTS sims feasible alongside a 96-channel × 6-block
model** (current production: **128 sims, 64ch × 4block**), keeping epoch time and per-move latency
acceptable. The policy-diffuseness investigation
([`policy_diffuseness_investigation.md`](policy_diffuseness_investigation.md)) names raising sims
(≥400) **and** rebalancing the policy head + growing the trunk as the *co-primary, coupled* fixes for
Goal #4 — so the compute target is not optional, it is the whole point. This plan is the perf budget
that makes that target affordable.

> **Source tags:** **[M]** measured (MCTS microbench, `mcts_microbench_summary.json`); **[P]** measured
> in profiling ([`performance_profiling.md`](performance_profiling.md)); **[Est]** reasoned estimate.
> Every speedup below is labelled. Inputs: `performance_profiling.md` §9 (P0–P8),
> `mcts_code_review.md` §3–§5 (A1–A7, §4.2 tree parallelism, M1–M4),
> `mcts_microbench_summary.json`, `configs/dense_cnn_model1_scratch_64.toml`.

---

## 0. The two compute axes, quantified up front

**Axis 1 — sims 128 → 512 = 4×.** Profiling/review both establish that within self-play the
**evaluator, MCTS tree, and encode buckets scale ~linearly in sims**, while **Python orchestration does
not** (it is per *searched position*, not per *sim*). [P][M] Microbench confirms clean linearity through
400 (128/256/400 = 1.0/2.0/3.2×, every column). [M] So 4× sims ≈ 4× on the sim-scaling buckets, 1× on
orchestration.

**Axis 2 — model 64ch×4block → 96ch×6block.** Conv-trunk FLOPs ≈ `blocks × 2 × 9 × C² × H × W`. Ratio:

```
(6 blocks × 96²) / (4 blocks × 64²) = (6 × 9216) / (4 × 4096) = 55296 / 16384 = 3.375×
```

So **per-eval forward and the training fwd+bwd each get ~3.4× heavier** [Est, from FLOP scaling +
profiling's "step is trunk-compute-bound, head is ~2%/speed-neutral" finding [P]]. Trunk params rise
451K → ~1.5M (still tiny; the FC heads dominate param count but not compute). **Activation memory rises
~2.25×** (1.5× channels × 1.5× blocks) — a VRAM risk at bs256 on the 12 GB card (see Risk R1).

Model size scales **only the forward/backward compute**: evaluator-forward, training step, and the
eval-path forward. It does **not** touch tree CPU, encode, evaluator marshaling, orchestration, shuffle,
or the NPZ decompress tax — those are model-independent. [P]

**Combined worst-case raw blow-up** on the heaviest single component (evaluator forward) =
`4 (sims) × 3.4 (model) = 13.6×`. The job of this plan is to absorb most of that via overlap + CPU cuts
so the epoch grows ~1.7×, not ~3.4×.

---

## 1. Scoped work list

Each item: **what changes · files · effort · dependencies · risk · expected speedup (tagged)**.
Effort is rough calendar effort for one engineer familiar with the code.

### Group A — Quick wins (free / near-free, do first)

#### P0 — NPZ load-once in the training data path  *(MANDATORY)*
- **What:** `_batch_from_npz` re-indexes `data[KEY][start:stop]` per 256-row batch, and
  `NpzFile.__getitem__` **re-decompresses the whole array every access** (~31× compressed; `inputNCHW`
  is 694 MB uncompressed). Load each array once per shard into RAM, then slice in RAM. Apply in training
  **and** validation.
- **Files:** `trainer.py` (`_batch_from_npz`, `train_passes`, `_run_validation`).
- **Effort:** ~½ day (~5 lines + a small refactor of the per-batch loop). **Dependencies:** none.
- **Risk:** Very low — correctness-neutral; data is identical, only decode timing changes. RAM: one
  decompressed shard (~694 MB) resident at a time — bounded, well within budget.
- **Speedup:** **−~95 s/epoch [M]** (in-situ 714 → 503 ms/step; data-prep 20–23× faster). Removes the
  decompress tax entirely → training drops toward its GPU floor. This *grows in absolute value* with the
  bigger model? No — it's model-independent, so at 96×6 it stays ~95 s (a smaller *fraction*, same
  seconds).

#### PB — `cudnn.benchmark=False` for the evaluator (or bucket batch shapes)  *(MANDATORY)*
- **What:** `cudnn.benchmark=True` (`inference.py:77`) re-autotunes (~925 ms) on every never-seen batch
  shape; the evaluator sees ~900 distinct shapes → ~830 s of autotune on every cold/relaunch epoch.
  Set `benchmark=False` (0 ms penalty, identical steady speed [M]) **or** pad leaf batches to a few
  fixed buckets (= A7, the better long-term form that keeps autotune *and* kills the thrash).
- **Files:** `inference.py:77` (+ optionally `mcts.rs` batch bucketing for A7).
- **Effort:** ~5 min (`False`) / ~½ day (bucketing). **Dependencies:** none; A7 composes with A5.
- **Risk:** Very low. **Speedup:** **−~830 s on every cold/relaunch epoch [M].** Under the
  supervisor's relaunch pattern this is enormous and recurring. Also note: the microbench's *first*
  attempt **hung >10 min** in autotune thrash with `benchmark=True` — at 96×6 with more shape variety
  this risk is worse, so PB/A7 becomes **a correctness/liveness fix, not just speed**, for the target
  config.

#### P3 — uncompressed shuffle scratch  *(DEFERRED — see reason)*
- **What:** phase-1 scratch parts (`replay.py:753`, `np.savez_compressed`) are re-read a few lines later
  (`_load_part_arrays`, line 766) then deleted with `scratch_root`. Nothing outside the function reads
  them, so compressing them is pure waste (shuffle is **zlib-compress-bound**, ~538 MB/s [P]). Switch to
  `compresslevel=1`/uncompressed.
- **Files:** `replay.py:753`. **Effort:** ~10 min. **Dependencies:** none.
- **Speedup:** **−~40 s/epoch [P, modeled]** (116 → ~76 s shuffle).
- **DEFERRED — reason:** per the user's instruction this pass does **not** implement P3; it only marks
  it. A **TODO(P3, deferred)** comment was added at `replay.py:753` pointing here. It is trivial and
  low-risk, but it changes the byte layout of in-flight shuffle artifacts and is orthogonal to the
  sims/model goal, so it is scheduled as a standalone cleanup, not bundled into the structural work.
  *(Note: the final shuffled output at line 776 must stay compressed — training reads it. Only the
  transient `part_*.npz` are the P3 target.)*

#### P8 — right-size the eval cache  *(OPTIONAL, do with M1)*
- **What:** lower `mcts_session_cache_max_states` 262144 → 131072; the cache fills ~190k/262k within 24
  moves [M] and inserts == unique states (no recompute on eviction within an epoch), so this is RAM-only.
- **Files:** config. **Effort:** ~2 min. **Risk:** very low. **Speedup:** throughput-neutral, **−~340 MB
  RAM [P]**. **Correction (audit):** this is **host RAM**, not VRAM — the eval cache is an
  `Rc<RefCell<HashMap>>` of `Arc<RustEvaluation>` priors on the CPU, so lowering it does **nothing** for
  the 12 GB VRAM/activation risk (R1). Its real value is host-RAM headroom to **re-widen the replay
  window** (cut 600k→300k for RAM at epoch 16 — see config note), where more historical diversity
  helps learning. The VRAM risk is handled separately and automatically (R1). Folds into M1
  structurally.

### Group B — Self-play structural changes (the lever that makes high sims affordable)

#### A1 / P5 — pipeline select↔eval (double-buffered leaf batches)  *(MANDATORY for the goal)*
- **What:** today `run_searches_to_targets` is "barrier every pass": rayon selects a leaf batch, then
  one Python `evaluate_*` forward runs under the GIL while rayon sits idle, then backup. GPU duty ~29%
  [P]. Restructure into a **producer/consumer**: S rayon search workers push leaf-eval requests into a
  bounded queue; a dedicated eval consumer drains them into fixed-ish NN batches, calls Torch (releases
  GIL during forward + H2D/D2H), routes results back for backup. Virtual loss (already implemented) is
  the sync primitive; per-tree ownership (shard roots across workers) means **no global tree lock**.
- **Files:** `mcts.rs` (`run_searches_to_targets`, budget loop), `mcts_eval.rs` (queue/consumer),
  `inference.py` boundary unchanged. **Effort:** ~1–1.5 weeks (the moderate refactor the review scopes).
- **Dependencies:** benefits from Q3 fix (don't abort a root's batch on a pending collision — else
  batches under-fill) and from A5 (bigger virtual batch). **Risk:** Medium (Rust concurrency), but no
  new locking model.
- **Speedup:** raises GPU duty ~29% → 60–80% [Est]. **This is the change that converts self-play from
  `sum(GPU + CPU)` to `max(GPU, CPU)`.** At the target config it is what keeps self-play GPU-bound at the
  forward floor instead of paying forward + tree + encode + marshal serially (see §4). Directly motivated
  by the measured **root-saturation at ~8 roots on the serial eval stage** [M] — beyond ~8 roots every
  extra root just queues behind the one serial forward.

#### A2 — skip redundant per-ply `state_hash` when descending known children  *(MANDATORY-ish, cheap)*
- **What:** `select_pending_leaf` recomputes a full `state_hash` every ply even when `edge.child` is
  `Some` and the hash is never used (`mcts_tree.rs:438`). Only hash on the new-leaf / lookup branch;
  carry the stored `node.state_hash` when descending. (= code-review Q1.)
- **Files:** `mcts_tree.rs:438`. **Effort:** ~½ day. **Dependencies:** none. **Risk:** Low.
- **Speedup:** cuts a chunk of the ~86 s/epoch tree-CPU bucket; **grows with depth × sims** [Est], so it
  is worth proportionally more at 512 sims. Bundle with A1.

#### A3 — store leaf/parent state on the node (kill O(depth) replay-from-root)  *(MANDATORY for 512 sims)*
- **What:** every playout clones `root_state` and replays *every* edge (`apply_placement` + full state
  hash per ply, `mcts_tree.rs:401–470`). Store the state on the node (or on nodes above a visit
  threshold) so a playout applies **one** move from a cached state — O(depth) → O(1).
- **Files:** `mcts_tree.rs` (`RustNode`, `select_pending_leaf`). **Effort:** ~3–4 days.
  **Dependencies:** memory trade (more per-node bytes) — pairs with M1/M3 cache right-sizing.
  **Risk:** Medium (memory; correctness of cached-state invariant — virtual-loss/transposition tests
  must stay green).
- **Speedup:** this is **the dominant tree-CPU cost as sims rise** [Est]: trees deepen at 512 sims, so
  O(depth) replay × 4× sims is the steepest-growing CPU term. Without A3, tree CPU could grow worse than
  linearly in sims. Pairs with A2 to keep the (post-A1) CPU side under the GPU-forward floor.

#### A4 — keep legal-prior gather + value decode on-GPU, transfer once  *(RECOMMENDED)*
- **What:** `inference.py:188–232` does `scatter_reduce_`/`exp`/normalize then `.cpu()` per call, plus
  `frombuffer`/`reshape`. Fuse the gather/softmax/decode on-device and do a single D2H. (= A4; the
  evaluator is 92–95% forward but the ~6% marshal × 4× sims = ~192 s at the target, worth trimming.)
- **Files:** `inference.py:188–232`, `mcts_eval.rs` parse side. **Effort:** ~2–3 days.
  **Dependencies:** none (composes with A1). **Risk:** Medium (numerics — keep the validation boundary).
- **Speedup:** the ~48 s/epoch marshal bucket → ~½ [P/Est]; ×4 at 512 sims so ~192 → ~96 s. Helps the
  post-A1 CPU side.

#### A5 — bigger NN batches per forward (`virtual_batch_size` ≫ 4)  *(RECOMMENDED, compounds with A1)*
- **What:** `mcts_virtual_batch_candidates = [4]` (config) → each pass submits only ~4 leaves. Raise it
  so each Torch call is fatter; merge across more roots before calling. Needs Q3 (don't abort batches on
  collision) to actually fill.
- **Files:** `[model.config.performance]` calibration candidates, `mcts.rs` budget loop. **Effort:**
  ~1 day (+ recalibrate). **Dependencies:** A1, Q3. **Risk:** Low–Med.
- **Speedup:** fewer/fatter forwards → better GPU efficiency + amortized Python overhead [Est]. Matters
  more for 96×6 (bigger kernels prefer bigger batches) and is the natural batch-shape source for A7.

#### P4 — trim Python game orchestration  *(OPTIONAL; sims-independent so lower priority at high sims)*
- **What:** orchestration is the largest self-play bucket at 128 sims (12.2 ms/pos, 129 s/epoch [P]):
  batch the 256 per-game `.npz` writes, trim per-position bookkeeping in `selfplay.py`/sample finalize.
- **Files:** `selfplay.py`, sample finalize. **Effort:** ~2–3 days. **Risk:** Medium.
- **Speedup:** **−30–60 s/epoch [Est].** *Priority note:* orchestration does **not** scale with sims, so
  at 512 sims it is a shrinking fraction (129 of ~1100 s self-play). Worth doing but it is not on the
  critical path for the sims goal — schedule after A1/A3.

#### P6 / A7 — pad the small-batch evaluator tail to its sweet spot + stabilize shapes  *(RECOMMENDED)*
- **What:** per-state forward is U-shaped (best ~37 µs @64, plateau 55 µs ≥256 [P]); as games finish,
  many forwards run at small batch (≤32) at 2–2.5× cost. Pad/bucket leaf batches toward ~64–256 fixed
  sizes. This **also** fixes the cuDNN autotune thrash (= A7 / PB long-term form).
- **Files:** `mcts.rs` batching, `inference.py`. **Effort:** ~2 days. **Dependencies:** composes with
  A1/A5/PB. **Risk:** Medium.
- **Speedup:** **−15–30 s/epoch [Est]** at current sims; bucketing's bigger payoff is letting
  `cudnn.benchmark=True` converge for the 96×6 model (more shape variety) without the >10-min thrash
  [M]. Treat A7 as the **production form of PB**.

### Group C — Single-game / eval-path scaling (tree parallelism)

#### §4.2 — tree parallelism (virtual loss + relaxed atomics) for one root  *(MANDATORY for eval/play latency)*
- **What:** the single-game path (`player.decide`, SealBot eval + real play) sends **one root**
  (`player.py:75`) and runs **fully serial**: 128 sims = **184 ms/move**, 400 = **576 ms** [M] —
  already 3.7×/11× over a 50 ms budget at the *current* model. Add T worker threads (8–16) that descend
  the **one** shared tree under virtual loss and batch leaves to one forward. Synchronize with
  **relaxed atomics** on `visits`/`value_sum`/`pending` (KataGo approach; virtual loss makes occasional
  races benign) or sharded/striped locks; lay out `RustNode`/`RustEdge` to avoid false sharing. A
  near-zero-risk first step is **root parallelism for eval** (run N independent trees for the same
  position, sum visit counts) — uses idle cores with no new sync, scales worse per core but ships fast.
- **Files:** `mcts_tree.rs` (atomic stat fields, shared-tree descent), `mcts.rs`, `player.py`.
  **Effort:** ~1.5–2 weeks (the harder structural item). **Dependencies:** reuses A1's virtual-loss/leaf
  machinery — **do A1 first**. **Risk:** High (shared-tree concurrency + correctness of virtual-loss
  cancellation, §7 of the review).
- **Speedup:** turns the serial single-move into ~T-way parallel; at ~6–8× effective it brings
  per-epoch SealBot eval (currently 173 s, serial) back near baseline even at 4× sims + bigger model,
  and is the **only** path to a deployable per-move latency. See §4 for why epoch-eval is feasible but
  the *50 ms deploy budget* at 512 sims + 96×6 still needs an extra lever.

### Group D — Memory (enabling, not speed)

#### M1 / M2 / M3 — right-size, shard, and (optionally) f16 the eval cache  *(M1 OPTIONAL, M2 with A1)*
- **What:** the eval cache is the dominant MCTS RAM pool (~1–3 GB of priors at the 262k cap [P/M]).
  **M1** lower the cap / scope it per batch-of-games (= P8). **M2** shard by hash (16 shards, per-shard
  lock) so it stays shared across A1's pipelined workers without serializing every lookup — **M2 is a
  dependency of A1 at scale.** **M3** quantize priors to f16 (8 → 4 B/entry) if RAM dominates at high
  sims.
- **Files:** `mcts_eval.rs` (cache), config. **Effort:** M1 ~2 min; M2 ~3–4 days; M3 ~1 day.
  **Risk:** M1 very low, M2 medium, M3 low (f16 ample for softmax priors). **Speedup:** RAM-only;
  M2 removes a contention bottleneck that would otherwise cap A1.

#### A6 / M4 — re-root without deep-copying the subtree  *(OPTIONAL)*
- **What:** `advance_root` deep-clones the retained subtree + rebuilds the transposition map every move
  (`clone_subtree_nodes`, `mcts_tree.rs:634–664`) — O(subtree) copy per move that grows with sims.
  Re-root in place / generational arena with lazy compaction.
- **Files:** `mcts_tree.rs:advance_root`. **Effort:** ~3–4 days. **Risk:** Medium.
- **Speedup:** removes an O(subtree) copy + rehash per move; **scales with sims** [Est] — modest now,
  larger at 512 sims. Schedule after A3 (shares the node-state work).

### Group E — Code quality (do alongside the structural work, not separately)
Q1(=A2), Q2 (dead guard / `debug_assert`), Q3 (don't abort batch on collision — **needed for A5**),
Q4 (dead binding), Q5 (stale `Rc`→`Arc` comments — **mis-describes the threading contract A1/§4.2
depend on**), Q6 (two prior orderings), Q7 (dup `random_unit`, consider `rand_distr`), Q8 (decompose
`search()`), Q9. **Effort:** ~1–2 days total. **Risk:** Low. **Why now:** Q3 and Q5 are *prerequisites
for correct concurrency work* — fold them into the A1/§4.2 PRs.

### Group F — Model-side change that ships WITH the new config  *(MANDATORY)*

#### P7 — fully-conv policy head  *(MANDATORY — lands together with 96×6 + 512 sims)*
- **What:** replace the FC `PolicyHead`/`opp_policy_head` (`Conv(64→2)+Linear(3362→1681)`) with a
  fully-convolutional head (e.g. `3×3 Conv→ReLU→1×1 Conv→1 logit/cell`). **Speed-neutral** [P]
  (measured 472↔473 ms training, +0.26 ms/batch inference) — so it changes **none** of §4's feasibility
  seconds. It is the co-primary **quality** fix from the diffuseness investigation, promoted to
  **mandatory by user decision (2026-05-29).**
- **Why MANDATORY and why tied to the model/sims change:** the diffuseness report (§7, §8) finds the
  policy pathway and the search budget are *co-primary and coupled* — they bootstrap each other.
  **"Scaling to a bigger model while leaving 128 sims in place would likely waste the extra capacity —
  the bigger policy head would still be trained on narrowly-explored, prior-echoing targets."** The
  symmetric trap applies to the head: a 96×6 trunk feeding the *current* FC head (top-1 ≈ 0.4, top-32
  captures only ~70% of mass) still can't represent a sharp policy, so the added trunk capacity trains
  against a diffuse prior-echo. **Therefore the new head MUST land in the same fresh run as the 96×6
  trunk + 512 sims — not bolted on later.** All three are one model redefinition.
- **Files:** `architecture.py` (`PolicyHead`, `opp_policy_head`); no replay/D6/schema change — the
  training target is already spatial (`policyTargetsNCHW (N,1,41,41)`), which a fully-conv head outputs
  directly. **Effort:** ~2–3 days (architecture + retrain wiring). **Dependencies:** none in code, but
  **must be in place before the 512-sim/96×6 run starts** (it changes checkpoint shape → fresh run,
  `epoch_000022.pt` will not load). **Risk:** Medium (retrain; correctness of the new head's target
  alignment under D6 — keep the augmentation tests green).
- **Bonus (not load-bearing for perf):** P7 also *removes* the ~11.3 M params sitting in the two giant
  FC heads (the trunk is only 451 K → ~1.5 M at 96×6), so the param-bloat note in §0 disappears with it.

---

## 2. Dependency-ordered sequence

```
PHASE 1 — Quick wins (days; ship independently, before any structural work)
  P0  (training load-once)            ──┐ free, measured, unblocks training floor
  PB  (cudnn.benchmark=False)         ──┤ free, recurring 830 s/relaunch; also liveness for 96×6
  P8  (cache cap → 131072)            ──┘ host-RAM headroom (replay-window re-widen; NOT VRAM)
  [P3 marked deferred; standalone cleanup any time]
        │
        ▼
PHASE 2 — Self-play pipeline + tree-CPU cuts (the core enabler, ~2–3 weeks)
  Q3, Q5  (concurrency prerequisites) ──► A1 / P5  (pipeline select↔eval)   ◄── M2 (shard cache)
  A2  (skip redundant hashing)        ──┐
  A3  (state-on-node, kill replay)    ──┤ cut the CPU side so it stays under the GPU-forward floor
  A4  (on-GPU gather, transfer once)  ──┤
  A5  (bigger virtual batch)          ──┘ (needs Q3)
  A7/P6 (bucket/pad batch shapes)     ──► production form of PB; stabilizes 96×6 autotune
        │
        ▼
PHASE 3 — Single-game scaling (the eval/deploy latency, ~2 weeks)
  §4.2  tree parallelism (virtual loss + relaxed atomics)   [depends on A1 machinery]
  (optional first step: root-parallel eval — zero new sync)
        │
        ▼
═══ MODEL-CHANGE GATE — flip to the target config as ONE fresh run ═══════════════
  P7  (fully-conv policy head)  ──┐
  channels 64 → 96               ──┤ all three land TOGETHER (changes checkpoint shape →
  blocks   4  → 6                ──┤ fresh run; epoch_000022.pt will NOT load), AFTER the
  search_visits 128 → 512        ──┘ mandatory perf items above so epochs are ~33 min not ~65.
  Rationale (diffuseness §7): the bigger trunk + higher sims + rebalanced head are coupled;
  shipping any subset wastes the others (capacity trains on prior-echoing targets).
        │
        ▼
PHASE 4 — Optional / opportunistic
  P4  (trim orchestration)  ·  A6/M4 (re-root w/o deep copy)  ·  M3 (f16 priors)  ·  P3  ·  Q-cleanups
```

**Why this order:**
1. **Phase 1 is free and unblocks measurement.** P0 drops training to its GPU floor so the floor is
   visible before the model grows; PB removes the autotune tax that would otherwise *dominate and even
   hang* the 96×6 cold epochs; P8 buys host-RAM headroom (for re-widening the replay window — **not**
   VRAM; the VRAM risk is handled by R1's auto-fallback). None depend on anything.
2. **A1 is the keystone.** Everything in self-play hinges on converting `sum → max(GPU, CPU)`. The
   measured root-saturation at ~8 roots [M] proves the serial eval stage — not selection — is the cap,
   so pipelining is the highest-ROI structural change and must precede the CPU-cut tuning (A2/A3/A4)
   that balances the two halves. Q3/Q5 are folded in because correct concurrency needs them.
3. **A2/A3/A4/A5 follow A1** because their value is only realized once the GPU and CPU halves overlap —
   you cut the CPU side down to the GPU floor. A3 specifically must land before 512 sims goes live (its
   cost grows steepest in sims).
4. **§4.2 (tree parallelism) reuses A1's machinery**, so it comes after. It is the eval/deploy-latency
   fix, structurally separate from self-play throughput.
5. **Phase 4 items are sims-scaling but sub-dominant** (orchestration is sims-independent; A6 is modest)
   — real wins, lower priority than the critical path.

---

## 3. Expected speedup — per item and cumulative

### 3a. On the *current* workload (128 sims, 64×4) — validating the levers
Baseline steady epoch **1143 s** [P]: training 479, self-play 373, eval 173, shuffle 116.

| Item | Phase touched | Δ (this item) | Tag |
|---|---|---:|---|
| P0 | training | −95 s | [M] |
| PB | (cold/relaunch only) | −830 s on those epochs | [M] |
| A1 + A2/A3 + A4 + P4 | self-play | 373 → ~250 s (−123 s) | [Est from buckets] |
| §4.2 tree-parallel eval | eval | 173 → ~60 s (−113 s) | [Est] |
| P3 (deferred) | shuffle | −40 s | [P modeled] |

**Cumulative steady epoch (current workload), all applied except deferred P3:**
`479−95 + 250 + 60 + 116 = 384 + 250 + 60 + 116 ≈ 810 s` ⇒ **~1.4× steady speedup (1143 → ~810 s)**,
**plus −830 s on every cold/relaunch epoch [M]** (the single biggest recurring win given the supervisor
relaunch pattern). This is the headline cumulative number for today's config.

### 3b. On the *target* workload (512 sims, 96×6) — the feasibility number
See §4 for the full math. Summary: optimizations take the **naive ~65 min** target epoch down to
**~33 min** — absorbing ~half of a ~3.4× raw epoch increase, leaving the epoch at **~1.7× today's
19 min**.

---

## 4. Feasibility math for 512 sims + 96ch×6block

All per-phase seconds are built from the **measured** epoch-21 budget [P] scaled by the §0 multipliers.
Scaling rules (from §0): sim-scaling buckets ×4; forward/training compute ×3.4 (model); orchestration,
tree, encode, marshal, shuffle, decompress all model-independent; orchestration also sim-independent.

### 4.1 Self-play
Measured epoch-21 split [P]: orchestration 129, evaluator 108 (forward ~60 + marshal ~48), tree 86,
encode 50  → 373 s.

| Bucket | ×sims | ×model | 512 sims, 96×6 (serial) | After opts |
|---|---:|---:|---:|---|
| evaluator **forward** | 4× | 3.4× | 60·4·3.4 = **816 s** | 816 s (GPU floor) |
| evaluator **marshal** | 4× | — | 48·4 = 192 s | ~96 s (A4) |
| MCTS **tree** CPU | 4× | — | 86·4 = 344 s | ~180 s (A2+A3) |
| **encode** | 4× | — | 50·4 = 200 s | 200 s |
| **orchestration** | — | — | 129 s | ~90 s (P4) |
| **Serial total** | | | **1681 s (~28 min)** | — |
| **Pipelined (A1)** | | | — | **max(GPU 816, CPU 566) ≈ ~850 s** |

CPU side after opts = marshal 96 + tree 180 + encode 200 + orch 90 = **566 s** < GPU **816 s**, so with
A1 overlap **self-play is GPU-forward-bound at ~850 s** [Est]. Pipelining alone takes 1681 → ~850 s
(~2×); without A1 the 4×-sims + bigger-model serial sum (1681 s) is the wall.

### 4.2 Training (does not scale with sims; scales 3.4× with model)
Epoch-21 training 479 s [P] = GPU floor ~182 + decompress bug ~95 + cold-shard disk I/O ~202.

| Component | 96×6 | After opts |
|---|---:|---|
| GPU floor (391 steps × 0.465 s) | 182·3.4 = **619 s** | 619 s (irreducible w/o torch.compile/TensorRT) |
| NPZ decompress bug | 95 s | **0** (P0) |
| cold-shard disk I/O | ~202 s | ~150 s (partial overlap; P1-class prefetch optional) |
| **Total** | **~916 s** | **~770 s** |

VRAM caveat (R1): 96×6 at bs256 ≈ 2.25× activations may exceed 12 GB → drop to bs128 (steps double,
per-step halves; floor ~unchanged, mild small-batch inefficiency). **Audit note: this fallback is
automatic.** `calibrate=true` and `performance.py` benchmarks every candidate batch (training,
inference, **and** self-play) inside a `try/except RuntimeError → _is_oom` guard (`performance.py:194,
241`): an OOM candidate is marked `"status":"oom"` and dropped, so calibration selects the largest
*non-OOM* batch on the 12 GB card without a manual config edit. The same guard covers the self-play
`inference_batch_candidates` (incl. 1024) at 96×6. So R1 is a known, self-mitigating risk — verify the
selected batch in the calibration log on the first 96×6 epoch, but no pre-emptive config change is
required.

### 4.3 SealBot eval (single-game path)
Currently 173 s serial [P]. At 512 sims + 96×6 the serial single-game path scales ~4× (sims) and ~1.5–2×
(model, blended over small-batch forwards) → **~1100–1250 s serial** [Est] — unacceptable.
**§4.2 tree parallelism** at ~6–8× effective brings this to **~170–250 s** [Est], roughly baseline.
Tree parallelism is therefore **mandatory** for the eval phase, not just deployment.

### 4.4 Shuffle
Model- and sim-independent (scales with rows shuffled) → **~116 s** unchanged (−40 with deferred P3).

### 4.5 Epoch totals

| Scenario | Training | Self-play | Eval | Shuffle | **Epoch** | vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline** (128 sims, 64×4) [P] | 479 | 373 | 173 | 116 | **1143 s (19 min)** | 1.0× |
| **Target, NAIVE** (512, 96×6, no opts) | 916 | 1681 | ~1200 | 116 | **~3913 s (~65 min)** | ~3.4× |
| **Target, OPTIMIZED** (full stack) | 770 | 850 | ~230 | 116 | **~1966 s (~33 min)** | **~1.7×** |

**The optimization stack absorbs ~32 of the ~46 added minutes**, holding the target epoch at ~1.7× the
current 19 min despite a 4×-sims + 3.4×-model (up to 13.6× on the heaviest component) workload.

### 4.6 Critical path and where we'd still fall short
- **Critical path after opts = raw 96×6 GPU compute.** Self-play (816 s forward) + training (619 s
  fwd+bwd) = **~1435 s of the ~1966 s epoch is the 96×6 trunk on the GPU.** Both phases are
  GPU-compute-bound; CPU work hides under the forward (A1) and the data tax is gone (P0).
- **Only remaining lever to go below ~33 min:** make the forward/backward itself cheaper —
  **FP16/TensorRT** inference engine (self-play + eval) and **`torch.compile`** for training. TensorRT
  on the eval forward at ~1.5–2× would cut self-play ~850 → ~500–550 s and the epoch toward ~25 min.
  This is *unmeasured* [Est] and is the additional lever to close the gap if 33 min is too slow.
- **Where we genuinely fall short — the 50 ms deploy budget.** Even with §4.2 tree parallelism at ~8×
  on 32 threads, a single 512-sim move on 96×6 ≈ `184 ms (128/64×4) × 4 (sims) × ~2.5 (model, blended)
  ≈ ~1.9 s serial ÷ 8 ≈ ~240 ms/move` [Est] — still **~5× over 50 ms**. So **512 sims + 96×6 is
  feasible for *training* (epoch ~33 min) but NOT directly deployable at 50 ms/move.** Closing *that*
  needs one of: (a) fewer sims at deploy time (decouple train-sims from play-sims — standard
  AlphaZero/KataGo practice), (b) a distilled/smaller deploy net, (c) TensorRT FP16 on the eval forward
  (~2×) **plus** higher thread count, or (d) accept a slower-than-50 ms dense player and measure
  strength at its natural latency. **Recommendation: decouple training-sims (512) from deployment-sims**
  — the diffuseness fix needs high sims *during training* to generate good targets; real-time play can
  run fewer sims on the trained net.

---

## 5. Recommendation — mandatory vs optional

**Mandatory to hit 512 sims + 96×6 at an acceptable training epoch (~33 min):**
- **P0** (training would otherwise carry the +95 s decompress tax on top of a 3.4× floor).
- **PB / A7** (without it the 96×6 cold/relaunch epochs pay ~830 s of autotune and risk the >10-min
  hang the microbench hit; with more shape variety at 96×6 this is a *liveness* requirement).
- **A1 / P5** (the keystone: without pipelining, self-play is the serial 1681 s sum, not 850 s — this
  single change is the difference between a ~50 min and a ~33 min epoch).
- **A3** (+ **A2**) (kills the O(depth) replay-from-root whose cost grows steepest in sims; required to
  keep the post-A1 CPU side under the GPU floor at 512 sims).
- **§4.2 tree parallelism** (eval phase is ~1200 s serial otherwise; also the only route to deployable
  latency).
- **M2** (cache sharding) — a *dependency* of A1 at scale (shared cache must not serialize the workers;
  the cache is currently `Rc<RefCell>`, not even `Send`, so A1's worker threads cannot share it as-is).
- **Q3, Q5** — concurrency prerequisites folded into the A1/§4.2 PRs.
- **P7** (fully-conv policy head) — **promoted to mandatory by user decision (2026-05-29).** Speed-neutral
  (it changes none of the feasibility seconds), but per the diffuseness report it must land **in the same
  fresh run** as the 96×6 trunk + 512 sims — otherwise the added trunk/head capacity trains on
  narrowly-explored, prior-echoing targets and the compute spend is wasted. It is the model half of the
  coupled (sims ⊕ head ⊕ trunk) fix this whole plan exists to make affordable.

**Strongly recommended (large, lower-risk wins; do unless time-boxed out):**
- **A4** (cuts marshal 192 → ~96 s at 512 sims), **A5** (batch efficiency, compounds with A1),
  **A7/P6** (batch-shape stability for the bigger model), **P8/M1** (RAM/VRAM headroom).

**Optional / opportunistic (real but sub-dominant or sims-independent):**
- **P4** (orchestration is sims-independent → shrinking fraction at 512 sims), **A6/M4** (re-root
  without deep copy — modest), **M3** (f16 priors — only if RAM dominates), **P3** (−40 s shuffle,
  deferred), **Q-cleanups** (do alongside).

**Bottom line.** With the mandatory set (P0, PB/A7, A1, A2/A3, §4.2 tree-parallel, M2 + Q3/Q5, **and
P7**), **512 sims + 96×6 is feasible for training at ~33 min/epoch (~1.7× today)**, GPU-compute-bound on
the new trunk, with a rebalanced policy head that can actually learn from the higher-sim targets. The one place it still falls short is the **50 ms real-time deploy budget**, which is best closed
by **decoupling deployment sims from training sims** (and/or TensorRT FP16) rather than by any item in
this plan.
