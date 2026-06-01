# Code review — dense_cnn Model 1 MCTS (Rust core + Python orchestration)

**Date:** 2026-05-29 **Branch:** `review/scratch64-optimization-stage` (read-only; no
production code changed) **Target:** RTX 4070 Ti + Ryzen 9 7950X (16C / **32T**, confirmed)
**Scope:** `packages/hexo_models/dense_cnn/rust/src/{mcts.rs, mcts_tree.rs, mcts_eval.rs,
encoding.rs, state.rs, constants.rs}` and the Python boundary
`python/hexo_models/dense_cnn/{mcts.py, inference.py, selfplay.py, player.py}`.

> **Evidence tags.** **[M]** = measured by me in this session (read-only microbenchmark
> [`analysis/mcts_microbench.py`](mcts_microbench.py), random-init 64ch/4block net on the free
> GPU, cuDNN autotune disabled — see §8 for why). **[P]** = measured in the prior profiling pass
> ([`analysis/performance_profiling.md`](performance_profiling.md)). **[D]** = design judgment
> from reading the code, not separately benchmarked. Random weights are fine for **mechanics and
> throughput**; they say nothing about playing strength.

---

## 0. TL;DR

1. **The parallelism model is *root parallelism*, not *tree parallelism*.** `rayon`
   parallelizes leaf selection across the independent per-game search trees in one `search()`
   call (`mcts.rs:306` `par_iter_mut`). For **self-play this is the right shape** — 256 games map
   cleanly onto 32 threads. But the **single-game path (`player.decide`, used by SealBot eval and
   real play) sends one root** (`player.py:75`), so it runs **fully serial**: **[M]** a single
   128-sim move is **184 ms** (vs **57 ms/root** once ≥8 roots run concurrently — a 3.3× gap), and
   400 sims is **576 ms/move** — ~11× a 50 ms budget. That serial single-game path is the design's
   biggest structural gap for "raise sims under a time budget." **[M/D]**
2. **Search cost is clean-linear in sims** — 128→256→400 visits scaled 1.0×→2.0×→3.2× in wall,
   eval, and node count, with no super-linear blowup. So raising sims is *predictable* but *not
   free*: ~linear more GPU evals + CPU tree work. **[M]**
3. **Within a search the loop is serial select→eval→backup with no GPU/CPU overlap.** The Python
   evaluator call holds the GIL while `rayon` sits idle, and `rayon` selects while the GPU sits
   idle (`mcts.rs:run_searches_to_targets`). Prior profiling put GPU duty at **~29%** during
   self-play. **[P]** The single highest-leverage performance change is to **pipeline** these two
   stages. **[D]**
4. **Memory is already well-controlled on the hot path** (shared `Arc<RustEvaluation>` priors for
   interior nodes — the recent fix; lazy edge materialization; nucleus widening cap). **[M]** the
   *live tree* is cheap and bounded by subtree reuse (node count flat ~275/root over a game;
   materialized edges only **~0.35 MB across 32 trees**). The dominant consumers are (a) the
   **epoch-lifetime eval cache**, which fills toward its `262144` cap within ~24 moves (**[M]**
   257→190k entries), and (b) a surprisingly large **staged-prior pool** (**[M]** ~24→87 MB across
   32 trees, growing through the game) — ~200× the materialized-edge bytes (§5.2, flagged for a
   closer look). **[M/P]**
5. **State is re-derived from the root clone on every playout** (`select_pending_leaf` clones
   `root_state` and replays every edge with `apply_placement` + a full `state_hash` per ply,
   `mcts_tree.rs:401-470`). This is O(depth) engine work × sims and grows as sims/visits deepen
   trees. It is a prime candidate for the measured **~86 s/epoch "MCTS tree CPU"** bucket. **[P/D]**
6. **`advance_root` deep-clones the retained subtree every move** (`mcts_tree.rs:634-664`,
   `clone_subtree_nodes`) and rebuilds the transposition map — O(subtree) copy per move that scales
   with sims. **[D]**

The code is **clean, well-documented, carefully validated, and correct as far as I can tell** —
this review is mostly about *scaling headroom*, not bug-fixing. Concrete issues are in §2; the
prioritized performance list is §3; the multithreading proposal tailored to the 7950X is §4;
memory in §5; what to benchmark next in §6.

---

## 1. Current-design assessment

### 1.1 Component map
| Layer | File | Responsibility |
|---|---|---|
| Py session wrapper | `mcts.py` | `BatchedMctsSession.run`; decode byte-backed policies into `SearchResult` |
| Py evaluator | `inference.py` | `evaluate_model1_payload`: bytes→tensor, forward, legal-prior softmax-gather, →bytes |
| Native session | `mcts.rs` | PyO3 boundary; per-game `RustSearch` map; rayon batch loop; action selection; result payloads |
| Tree mechanics | `mcts_tree.rs` | `RustSearch`, `RustNode`, `RustEdge`, PUCT, nucleus widening, virtual loss, backup, subtree promote, Dirichlet |
| Eval adapter + cache | `mcts_eval.rs` | state hashing, dedup/coalesce, encode→payload, parse/validate evaluator bytes, bounded cache |
| Encoding | `encoding.rs` | 13-plane crop tensor build |
| State intake | `state.rs` | clone live `HexoState` across the engine C-ABI capsule |

### 1.2 What the design gets right
- **Separation of concerns is excellent.** Python owns Torch and config; Rust owns the tree, state
  cloning, payload marshaling, and validation. The boundary (`evaluate_model1_payload`) is a
  strict, length-checked byte ABI. This is genuinely KataGo-like and easy to reason about.
- **Lazy/nucleus-widened edges.** Nodes stage candidates and materialize an edge only when PUCT
  selects it, capped by a once-computed top-p nucleus count (`select_or_materialize_edge`,
  `nucleus_count_values`). Over a 400–1400 action space this is the correct way to bound branching
  and memory — far better than allocating an edge per legal move.
- **Shared-prior interior nodes.** `NodePriors::Shared(Arc<RustEvaluation>)` means a transposition
  or deep node holds an 8-byte `Arc` instead of copying a ~1400-entry prior vector
  (`mcts_tree.rs:101-112`, `shared_from_cache`). This is the single most important memory decision
  and it is already in place.
- **Transposition table.** `node_table: HashMap<StateHash, usize>` dedups states across the whole
  tree (and across roots at the eval boundary), so identical positions evaluate once.
- **Validation discipline.** `finalize_model_priors`, `read_value`, `require_exact_bytes`,
  `validate_*` reject non-finite, duplicate, zero-mass, and wrong-length payloads. This is the kind
  of defensive boundary that prevents a corrupt batch from silently poisoning training.
- **Virtual loss + batched leaves** exist (`apply_virtual_visit`, `mark_pending`,
  `backup_virtual`), so a single tree can gather several leaves per NN call.

### 1.3 The structural limits (the heart of this review)
- **Root parallelism only.** `run_searches_to_targets` does `searches.par_iter_mut()` — threads
  map to *games*, not to playouts within a game. Within one `RustSearch`, the `for _ in 0..budget`
  leaf-gather loop (`mcts.rs:319`) is **serial**. Consequences:
  - Self-play with 256 games: great 32-thread utilization on the *selection* half. **[M]** confirms
    per-root cost amortizes hard as roots grow (see §3/§6).
  - Single-game eval/play (`player.decide`, 1 root): **no parallelism at all**, and the GPU sees a
    `virtual_batch_size`(=4)-sized batch per pass — the worst case for both CPU and GPU efficiency.
- **No select/eval overlap.** Each outer `while` iteration: (a) rayon selects a leaf batch across
  roots, (b) one `evaluate_model1_state_refs_cached` call runs the Python forward under the GIL,
  (c) backup. (a) and (b) never overlap. GPU ~29% duty. **[P]**
- **State re-derivation per playout.** Selection replays from a fresh root clone every time
  (`mcts_tree.rs:401`). Cost is O(depth) engine applies + O(depth) full-state hashes per simulation.
- **Per-move subtree deep-copy** in `advance_root`.

---

## 2. Code-quality issues and fixes

Ordered roughly by value. None are correctness bugs I can prove; several are robustness/clarity.

**Q1 — `select_pending_leaf` recomputes a full `state_hash` every ply even when the child is
already known (`mcts_tree.rs:438`).** `current_hash = state_hash(&state)` runs unconditionally
after each `apply_placement`, but when `edge.child` is `Some` the running hash is never used (we
just descend). Hashing the whole `HexoState` per ply × depth × sims is real work.
*Fix:* only compute `current_hash` on the branch where `edge.child` is `None` (the lookup /
new-leaf cases). Carry `self.nodes[child_id].state_hash` when descending into a known child.

**Q2 — Apparent dead/defensive guard (`mcts_tree.rs:428`).** After `select_or_materialize_edge`
returns an index, the code re-checks `if edge.pending > 0 && edge.child.is_none() { return None }`.
But the candidate scan inside `select_or_materialize_edge` already `continue`s over exactly those
edges (`mcts_tree.rs:482-484`), and freshly materialized edges have `pending == 0`. So the returned
edge can never satisfy this condition. Either it is unreachable (delete it, or convert to a
`debug_assert!`), or there is an intended path that is not obvious — worth a comment either way.

**Q3 — Early-out on a pending collision wastes a whole selection (`mcts_tree.rs:428` / the budget
loop `mcts.rs:319-355`).** When selection hits a pending-but-unevaluated edge it returns `None`,
which `break`s the per-root budget loop. Virtual loss already de-weights in-flight paths, so this
guard is belt-and-suspenders, but it means one collision ends a root's contribution to the current
batch instead of trying the next-best sibling. With `virtual_batch_size=4` this is minor; if you
raise the virtual batch (recommended for bigger NN batches, §4) it will under-fill batches.
*Fix:* on collision, continue selecting siblings (KataGo keeps selecting under virtual loss) rather
than aborting the root's batch.

**Q4 — `advance_root` leaves a dead binding: `let _ = edge_index;` (`mcts_tree.rs:662`).** The
`edge_index` from the `find` is never used. Drop it from the destructure. Minor, but it reads like
a leftover.

**Q5 — Stale comments referencing `Rc` where the type is now `Arc`.** `mcts_tree.rs:145-147`
("reference cache priors by `Rc`"), `mcts_eval.rs:342` ("Wrap … in an `Rc` once"). The migration
to `Arc` (for `Send`) is done; the comments weren't updated and will mislead the next reader about
the threading contract.

**Q6 — Two prior orderings for the same concept.** `NodePriors::Owned` is sorted **ascending**
(pop from back), `NodePriors::Shared` is sorted **descending** (index from front), and
`ensure_root_owned` reverses to convert. It's documented (`mcts_tree.rs:101-112`) and tested, but
it's a real cognitive-load / future-bug surface. Consider a single canonical (descending) order
with a small cursor index for the owned case too, so `peek/materialize/remaining` share one mental
model.

**Q7 — `random_unit` is duplicated** verbatim in `mcts.rs:726` and `mcts_tree.rs:965`, and the
hand-rolled LCG/`DirichletSampler` reimplement what a vetted crate (`rand`, `rand_distr::Gamma`)
does. The hand-rolled Marsaglia gamma is correct-looking but is exactly the kind of code you do not
want to own/validate yourself in an RL pipeline. *Fix:* factor `random_unit` into one place;
consider `rand_distr` for Dirichlet (it also gives you a properly seeded, well-tested generator for
the action-selection sampling, which currently derives a single `u64`→unit float per call —
adequate but minimal).

**Q8 — `search()` is a ~200-line method** (`mcts.rs:83-286`) doing validation, reuse, root eval,
the run loop, diagnostics, action selection, payload build, and subtree promotion. It's readable
but would test better decomposed (e.g. `resolve_or_build_searches`, `finish_and_promote`). Several
helpers already exist; the orchestration body is the outlier.

**Q9 — `validate_bounded_f32` is defined and used only for the noise fraction**; `c_puct`,
`temperature`, etc. have bespoke checks. Minor consistency nit — fine to leave.

**Q10 — Unsafe `slice::from_raw_parts` zero-copy views** (`mcts.rs:439-464`, `mcts_eval.rs:191-195`,
`encoding.rs:50`). These are sound (the backing `Vec`/`Vec<f32>` outlives the `PyBytes::new` copy,
lengths are exact) and well-commented. No change needed, but they are the riskiest lines in the
crate if the surrounding code is ever refactored — keep the "buffer outlives the view" invariant
loudly commented.

---

## 3. Prioritized performance improvements

> Impact estimates marked **[Est]** are reasoned from §8 measurements + the prior profiling; they
> are not separately benchmarked end-to-end. Self-play phase split (authoritative, **[P]**):
> orchestration 129 s, MCTS tree CPU 86 s, encode 50 s, evaluator (forward+marshal) 108 s
> (~60 s raw forward + ~48 s Python marshal), GPU ~29% duty.

| # | Change | Where | Expected impact | Risk |
|---|---|---|---|---|
| **A1** | **Pipeline select↔eval** (double-buffer leaf batches; while the GPU evaluates batch *N*, rayon selects batch *N+1* for other roots under virtual loss) | `mcts.rs:run_searches_to_targets` | Raises GPU duty from ~29% toward 60–80%; the ~48 s marshal + ~60 s forward increasingly hide under the 86 s tree work. **[Est]** big self-play win | Med (concurrency) |
| **A2** | **Avoid per-ply full `state_hash`** when descending known children; carry stored `node.state_hash` (Q1) | `mcts_tree.rs:438` | Cuts a chunk of the 86 s tree-CPU bucket; grows with depth/sims | Low |
| **A3** | **Store the leaf/parent state on the node** (or only on nodes above a visit threshold) so a playout applies *one* move from a cached state instead of replaying from root | `mcts_tree.rs` `RustNode`, `select_pending_leaf` | Turns O(depth) per playout into O(1) applies; the dominant tree-CPU cost as sims rise. **[Est]** | Med (memory trade, §5) |
| **A4** | **Keep the legal-prior gather + value decode on-GPU and transfer once** (currently `scatter_reduce_`/`exp`/normalize then `.cpu()` per call, plus `frombuffer`/`reshape`) | `inference.py:188-232` | The ~48 s/epoch "marshal" bucket; in-situ eval is ~1.8× raw forward. **[P]** | Med |
| **A5** | **Larger NN batches per forward**: raise `virtual_batch_size` (currently **4**) so each pass submits more leaves, and/or merge across more roots before calling Torch | `selfplay`/calibration + `mcts.rs` budget loop | Fewer, fatter forwards → better GPU efficiency & amortized Python overhead; compounds with A1. Needs Q3 fix to fill batches | Low–Med |
| **A6** | **Subtree promotion without a deep copy** — keep the arena and re-root (e.g. mark new root, compact lazily / generationally) instead of `clone_subtree_nodes` every move | `mcts_tree.rs:advance_root` | Removes an O(subtree) copy + rehash per move; scales with sims | Med |
| **A7** | **Cap the cuDNN-autotune thrash** from variable batch shapes (see §8 / finding F-AUTOTUNE): pad leaf batches to a few fixed bucket sizes, or accept the steady-state and document it | `inference.py` | Removes a pathological warm-up cost on cold processes and after shape changes; also stabilizes per-pass latency | Low |

**Why not "shrink the policy head" for speed:** the prior profiling already refuted this — the FC
head is ~2% of the forward. It is a *quality* lever (policy diffuseness), not a speed lever, and
will not make more sims affordable. Agreed and re-affirmed here. **[P]**

### 3.1 Cost of raising sims (measured) **[M]**
Search cost is **linear in visits** through 400 (with no nasty constant blow-up). For 128
independent roots, autotune-off, random net (see §8 for the regime caveat):

| visits | wall (s) | eval (s) | encode (s) | tree+orch (s) | nodes (Σ128 roots) |
|---:|---:|---:|---:|---:|---:|
| 128 | 3.92 | 2.45 | 0.22 | 1.25 | 17 247 |
| 256 | 7.87 | 4.90 | 0.43 | 2.53 | 34 195 |
| 400 | 12.50 | 7.68 | 0.76 | 4.06 | 53 281 |
| 800 | 44.04 | **34.0** | 1.53 | 8.51 | 106 865 |

Ratios vs 128 through 400: 256→2.0×, 400→3.2× across every column — clean linearity, node count
tracks visits 1:1. **At 800 the *tree-CPU* stayed linear (2.1× of 400) but *eval* blew up
super-linearly (4.4×).** That blow-up is a **regime artifact**, not tree behavior: with autotune
**off** the per-pass NN batches grow at high sims and cuDNN's default algorithm is poor on the
larger shapes (exactly the F-AUTOTUNE problem, §8). In production (autotuned, bucketed batches) the
eval term should track the ~linear forward cost. **Takeaway:** the *search mechanics* scale linearly
in sims; raising self-play sims to 400 ≈ 3.2× search cost (consistent with the profiling's ~2.4–3.1×
estimate). Whether that is affordable depends entirely on A1/A4/A5 and on fixing batch-shape churn.

### 3.2 Single-root (eval / real-play) latency — **the budget problem, measured [M]**
The `player.decide` path searches **one root**, so it gets the serial penalty in full:

| visits | wall/move (1 root) |
|---:|---:|
| 128 | **184 ms** |
| 256 | 372 ms |
| 400 | 576 ms |
| 800 | 1163 ms |

At the **current 128 sims a single move already takes ~184 ms single-threaded** — ~3.7× over a 50 ms
SealBot move budget — and it is linear in sims (400 → 576 ms). You cannot "just raise sims" on this
path without either dropping sims or adding **within-tree parallelism (§4.2)**. This is the clearest
single piece of evidence in the review.

### 3.3 Root parallelism amortizes — then saturates on the serial eval stage **[M]**
Per-root wall at fixed 128 visits, scaling the number of concurrent roots:

| roots | wall/root (ms) |
|---:|---:|
| 1 | 185 |
| 8 | 56 |
| 32 | 59 |
| 128 | 57 |
| 256 | 57 |

Two things: (1) going 1→8 roots cuts per-root cost **3.3×** (185→56 ms) — root parallelism works and
the 7950X is being used. (2) From **8 to 256 roots the per-root cost is *flat* (~57 ms)** and total
wall scales linearly with root count. That plateau is the **serial GPU-eval stage**: once ~8 roots'
worth of leaves keep selection busy, every additional root just queues behind the one serial
`evaluate_*` call. This is direct, quantified motivation for **A1 (pipeline select↔eval)** — the
thing capping throughput beyond a handful of roots is the un-overlapped eval, not selection or
locking.

---

## 4. Multithreading proposal for the 7950X (16C/32T)

The right design is **different for the two call patterns**, and the codebase should serve both.

### 4.1 Self-play (many roots) — *keep root parallelism, add a pipeline*
Root parallelism is already correct and **[M]** confirms it amortizes well. The win is **overlap**,
mirroring KataGo's asynchronous design:

- **Producer/consumer split.** Run *S* CPU search workers (rayon) that walk trees and push
  evaluation requests into a shared, bounded **leaf queue**; run a dedicated **eval consumer** that
  drains the queue into fixed-ish NN batches, calls Torch, and routes results back for backup.
  Because the Python forward and the H2D/D2H copies release the GIL, the consumer's GPU time
  overlaps the producers' selection time. Target GPU duty 60–80% (from 29%).
- **Virtual loss is the synchronization primitive** (already implemented). Workers apply virtual
  loss on selection so other workers/iterations avoid the same leaf; backup removes it. This is what
  lets many in-flight leaves coexist without locking the whole tree.
- **Per-tree locking.** Each `RustSearch` is independent, so a **per-tree lock** (or sharding
  roots across workers so each tree is touched by one worker at a time) avoids contention entirely
  for the self-play case — no global tree lock needed. The eval cache is the only shared structure;
  keep it behind a short critical section (or shard it by hash, §5).
- **Batch shape stability** (A7): bucket leaf batches to a few sizes so cuDNN autotuning converges
  and per-batch latency is predictable.

This is a moderate refactor of `run_searches_to_targets` from "barrier every pass" to "queue +
consumer," reusing all the existing virtual-loss/pending machinery.

### 4.2 Single-game strong search (eval / real play) — *add tree parallelism*
This is the missing capability. To raise sims under a 50 ms move budget against SealBot, one tree
must use many cores:

- **Tree parallelism with virtual loss + leaf batching.** Spawn *T* worker threads (e.g. 8–16) that
  each descend the **single** shared tree, apply virtual loss, and contribute leaves to a batch;
  one NN forward evaluates the batch; backup is applied. This is exactly the self-play machinery
  pointed at one root instead of many.
- **Synchronization.** For a single shared tree you need either (a) **fine-grained per-node atomics**
  on `visits`/`value_sum`/`pending` (the KataGo approach: relaxed atomics, virtual loss makes the
  occasional race benign), or (b) **sharded locks** (lock striping keyed by node id) to cut
  contention versus one big `Mutex`. Atomics on the hot `RustEdge`/`RustNode` stat fields are the
  scalable choice; lay the struct out to avoid false sharing (pad/separate the frequently-written
  counters, or keep per-thread partial sums merged at backup).
- **Interaction with batched GPU eval.** Same producer/consumer as §4.1: *T* descending threads
  feed one eval consumer. The batch size is naturally `min(in-flight leaves, max_batch)`.
- **Root parallelism as a cheap complement.** For eval specifically, you could also run several
  *independent* trees for the same position and combine their root visit counts (root parallelism)
  — trivial to add (just call the existing batched path with N copies of the root and sum visit
  policies) and it needs no new synchronization. It scales worse than tree parallelism per core but
  is a near-zero-risk first step to use idle cores in eval.

### 4.3 Concrete recommendation
1. **First**, implement the **pipeline/overlap (A1)** in the existing root-parallel self-play loop —
   highest ROI, no new locking (per-tree ownership), and it directly attacks the 29% GPU duty.
2. **Then**, generalize the virtual-loss leaf machinery so it can run **T threads on one tree**
   (§4.2) with relaxed atomics on edge/node stats, giving the eval/play path real scaling.
3. Size worker counts to leave headroom: with 32 threads, ~24 search workers + Torch's own intra-op
   threads + the eval consumer is a reasonable split; benchmark the knee (§6).

References this draws on: KataGo's playout-parallel search with virtual loss and a central NN eval
batcher (`cpp/search`), the original AlphaGo/AlphaZero "virtual loss + parallel MCTS" scheme, and
the standard leaf/root/tree-parallel taxonomy (Chaslot et al., "Parallel Monte-Carlo Tree Search").
The "batch many leaves into one NN forward" pattern is also the core of the *Watch the Unobserved*
/ batched-PUCT line of work and of Leela/KataGo's GPU batching.

---

## 5. Memory efficiency

### 5.1 Current footprint **[D]**
- `RustEdge` ≈ 40 B (`action_id u32`, `action` 2×i16, `prior f32`, `visits u32`, `value_sum f32`,
  `pending u32`, `child Option<usize>` = 16 B because `usize` has no niche). Materialized lazily and
  capped by nucleus widening (`widening_max_children=32`), so per node ≤ ~32 edges = ~1.3 KB worst
  case, usually far fewer.
- `RustNode` ≈ ~80 B + its `edges` Vec. Interior nodes hold an 8-byte `Arc` to the shared prior
  vector (no per-node prior copy) — the key win. Only **root** nodes own a prior `Vec`
  (`NodePriors::Owned`), and there is one root per game.
- Per `search()`, node count ≈ Σ visits across roots (**[M]**: 17 247 nodes for 128 roots × 128
  visits ≈ 135 nodes/root ≈ ~1 node/visit). At 400 visits that's ~400 nodes/root; trees stay small.
- **The live tree is cheap and bounded across a game [M].** Over a 24-move timeline (32 roots, 256
  visits), node count stayed flat (~275/root) and **materialized `active_edge_bytes` held at
  ~0.33–0.36 MB total** — subtree reuse + nucleus widening keep the actual tree tiny.
- **But the *staged* prior pool is ~200× the active edges and grows through the game [M].** The
  diagnostics' `hidden_prior_bytes` (owned, not-yet-materialized candidate lists) rose **24 MB → 87
  MB** across 32 trees from move 0 → 24, while active edges stayed ~0.35 MB. I did **not** fully
  reconcile the per-root magnitude (it exceeds a naive "one owned root × legal-count" estimate), so
  I flag it rather than assert a mechanism: **the staged/owned priors — the one place priors are
  still *copied* per-tree rather than shared — appear to be the bulk of per-tree memory and grow
  with depth.** Worth a focused look; if confirmed, sharing them (extend the `Arc` treatment to the
  root) or representing root temperature/Dirichlet as a **sparse overlay** on the shared prior
  instead of a full owned copy would reclaim most of it. **(F-STAGED-PRIORS)**

### 5.2 The real memory consumer: the evaluation cache
`RustEvaluationCache` is created **once per epoch** in the session (`selfplay.py:77`) and lives for
all 256 games, bounded by `mcts_session_cache_max_states = 262144` (`MODEL1_EVAL_CACHE_MAX_STATES`
default is **1,048,576**). Each entry is an `Arc<RustEvaluation>` whose `priors: Vec<(PackedCoord,
f32)>` is up to the in-crop legal count (hundreds to ~1400 late-game), i.e. ~1–11 KB each. At the
262k cap and a realistic average that is **roughly 1–3 GB** of priors, and it is the largest MCTS
memory pool by far. The prior profiling noted lowering the cap to 131072 reclaims ~340 MB at
throughput parity (cache inserts == unique states, so eviction never forces recompute within an
epoch). Recommendations:

- **M1 — Right-size the cache to the working set.** It does not need a full epoch of history; a few
  generations of recent games suffice for transposition reuse. Lower the cap (P6 in profiling) or
  scope it per-batch-of-games. This is the cheapest large RAM reclaim. **[P]** **[M]** the cache
  reaches ~190k/262k entries within 24 moves of a 32-game batch — it genuinely fills the cap, so the
  cap directly sets the steady-state pool size.
- **M2 — Shard the cache by hash** (e.g. 16 shards each behind its own lock) so it can stay shared
  across the parallel/pipelined workers of §4 without serializing every lookup. Also reduces
  contention on the single `RefCell`. **[D]**
- **M3 — Smaller prior storage.** Priors are stored as `(PackedCoord=u32, f32)` = 8 B/entry. If the
  cache dominates RAM at higher sims, consider quantizing priors to `f16` (4 B/entry, halves the
  pool) — the values are already softmax outputs read back into PUCT, so f16 precision is ample.
  **[D]**
- **M4 — Subtree reuse vs. copy (A6).** `advance_root`'s `clone_subtree_nodes` allocates a fresh
  `Vec<RustNode>` + `HashMap` every move. Re-rooting in place (or a generational arena with lazy
  compaction) avoids the transient doubling of the retained subtree each move. **[D]**

### 5.3 Scaling with sims and a 1000+ action space
- Trees scale ~1 node/visit (**[M]**), edges capped by widening — so **raising sims grows tree RAM
  linearly and modestly** (400 visits → ~400 nodes/root × ~80 B + edges ≈ tens of KB/root; 256
  roots ≈ low tens of MB). Trees are *not* the memory risk.
- The action space hits memory **only** through (a) the per-node prior vector — already shared via
  `Arc`, paid once per unique state in the cache — and (b) the root's owned prior `Vec`, one per
  game. Both are bounded. The representation choice here is sound for 1000+ actions.

---

## 6. What to validate with benchmarks (next)

1. ~~**Root-count amortization**~~ **DONE [M]** (§3.3): per-root 185 ms (1) → ~57 ms (≥8), flat to
   256. Saturates at ~8 roots on the serial eval stage. → motivates A1.
2. ~~**Single-root latency vs sims**~~ **DONE [M]** (§3.2): 128 sims = 184 ms/move, 400 = 576 ms —
   single-game high-sims does **not** fit 50 ms today. → motivates §4.2 tree parallelism.
3. **A1 pipeline prototype A/B:** GPU duty and self-play pos/s before/after overlapping select↔eval
   on a fixed games×visits workload. The single most important number to justify the refactor.
4. **A3 state-cache A/B:** tree-CPU seconds with replay-from-root vs store-state-on-node, at visits
   ∈ {128, 400, 800} (deeper trees magnify the win).
5. **Cache cap sweep (M1):** RAM + pos/s at cap ∈ {64k, 131k, 262k} to pick the knee.
6. **Tree-parallel single-game prototype (§4.2):** sims achievable in 50 ms with T ∈ {1, 4, 8, 16}
   worker threads + virtual loss, vs today's serial path.

---

## 7. Correctness notes (no bug found, but verify if you refactor)
- **Virtual loss cancels exactly:** `apply_virtual_visit` subtracts `virtual_loss` uniformly along
  the path; `backup_virtual` adds `value + virtual_loss`. The penalty is perspective-independent and
  is removed on backup, so final node values are unaffected — only in-flight selection is steered.
  Correct, but any change to the perspective handling (`node.player == leaf_player`) must preserve
  this cancellation.
- **Transposition DAG + path backup:** because `node_table` makes the structure a DAG, a node's
  `visits` can exceed the sum of any single parent edge's visits (it's reached by multiple paths).
  PUCT uses `edge.visits` for the exploration term and `node.visits` for the parent sqrt — this is
  the conventional choice and looks consistent, but it's the kind of thing to keep in mind if you
  add value-averaging changes.
- **`advance_root` flattens cross-subtree transpositions** (it only follows `edge.child` and resets
  others to `None`). Dropped transposition links just cause re-evaluation next move — correct, mildly
  wasteful; fine.

---

## 8. Method, honesty, and the autotune caveat

- **What I ran:** [`analysis/mcts_microbench.py`](mcts_microbench.py) — random-init 64ch/4block
  `Model1Network` → production `DenseCNNInference` → real native `BatchedMctsSession`, driving
  fresh `hexo_engine` games exactly as self-play does, on the free GPU. **No run state, config,
  checkpoint, or supervisor was touched.**
- **F-AUTOTUNE (a finding, not just a caveat):** my first run with the **production**
  `cudnn.benchmark=True` hung in cuDNN autotuning for >10 min and never cleared warm-up, because the
  native search sends a **different batch size almost every pass** (number of uncached unique leaves
  varies), and `benchmark=True` re-autotunes on each novel shape. I disabled it for the bench to
  measure steady mechanics. **This affects production too:** cold processes and any shape change pay
  an autotune tax (consistent with the profiling's "cold first-epoch 4.5× evaluator" and the ~11
  pos/s cold number). Padding leaf batches to a few fixed bucket sizes (A7) would let `benchmark=True`
  converge instead of thrash.
- **Regime caveat for §3.1 numbers:** they were taken with autotune **off** and with modest batch
  sizes (≤ ~512), so the *eval fraction is inflated* relative to a warm, autotuned, large-batch
  production run (where the profiling measured the forward at ~55 µs/state and the CPU buckets
  dominate). The **shapes/ratios I rely on — linearity in sims, ~1 node/visit, eval-as-a-serial-
  blocking-stage — are robust to that**; the absolute eval/CPU split is not, and I defer to the
  prior profiling's in-situ split for production proportions.
- **Strength:** random weights ⇒ I am measuring *throughput and mechanics only*, never playing
  strength.
- **Reproduce:** `python analysis/mcts_microbench.py` → [`analysis/mcts_microbench_summary.json`](mcts_microbench_summary.json)
  (raw rows for the sims sweep, 1-root sweep, root scaling, and the 24-move memory timeline). The
  three I lean on hardest — sims linearity (§3.1), the 1-root 184 ms latency (§3.2), and the root
  saturation at ~8 (§3.3) — are robust to the autotune-off regime; the absolute eval/CPU *split* is
  not, and there I defer to the prior in-situ profiling.

---

## 9. Summary of recommendations (priority order)
1. **A1 — pipeline select↔eval** in self-play (biggest perf win; no new locking). §4.1
2. **§4.2 — tree parallelism w/ virtual loss + relaxed atomics** for the single-game eval/play path
   (the missing capability for "raise sims under a time budget"). §4.2
3. **A3/A2 — kill per-playout state re-derivation** (store state on nodes; skip redundant hashing).
   §3
4. **A4/A5 — cut evaluator Python marshaling; bigger NN batches** (`virtual_batch_size` ≫ 4). §3
5. **M1/M2/M3 — right-size + shard + (optionally) f16 the eval cache** (dominant RAM pool). §5
6. **A6/M4 — re-root without deep-copying the subtree.** §3/§5
7. **A7 — stabilize batch shapes** so cuDNN autotune converges. §3/§8
8. **Q1–Q9 — code-quality cleanups** (dead guard, stale `Rc` comments, dup `random_unit`, decompose
   `search()`). §2

*All numbers tagged **[M]** are from this session's read-only microbenchmark; **[P]** from the
prior profiling pass; **[D]** are design judgments from reading the code. Production code on `main`
was not modified.*
