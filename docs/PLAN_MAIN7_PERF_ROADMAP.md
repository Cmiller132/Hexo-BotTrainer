# main_7 performance roadmap — c=192, deeper, more attention (2026-07-03)

What I would actually do, in order, given the measured state of the system
after the 2026-07-02/03 speedup rounds (`docs/analysis/MAIN6_SERVE_SPEEDUP_2026-07-02.md`).
Everything here is grounded in tonight's profiles, not guesses; projections
are labeled as projections.

## 0. Where the system is (baseline for all numbers)

- Live main_6 (c=128, 16 convs + 3 attn, heads 4x32): serve device cost
  ~0.076 ms/state; live self-play ~22 pos/s (round-2), was 13.75 before any
  of this work.
- Forward split at live shapes: hex-conv kernel 32% (@ ~62 TFLOPS — done),
  attention 30% (@ ~10 TFLOPS — the one inefficient component left),
  pointwise/LayerNorm 21%, other GEMMs 17%.
- The host (Python serve dispatch) currently PACES the GPU (~80% busy).
  Consequence: GPU-side savings alone do NOT raise c=128 throughput; they
  only pay combined with a host fix — or with a bigger model that makes the
  GPU the wall again. **c=192 does exactly that**, which reshuffles the
  priority list below.

---

## Tier 1 — do regardless (cheap, near-zero risk)

1. **Soak round 1+2** for a few epochs; confirm the epoch-75 SealBot
   multistage eval is clean (serve_half's 4.7e-3 value drift + vbs=48 are
   the two accepted-risk items it guards).
2. **Commit the dev repo** (model.py, inference.py, trainer.py,
   _triton_conv.py, search.rs timing, benches, docs) — everything is
   currently uncommitted and the worktree was hand-synced.
3. **Eval-phase budget check**: the SealBot arena inherits the fast serve;
   verify its 637s dropped proportionally at epoch 75. If not, its
   games_budget/eval_visits are the cheapest wall-clock lever left
   (eval is ~6% of wall amortized).

## Tier 2 — the c=128 serve endgame (SKIP if main_7 starts soon)

Only worth doing if the run stays at c=128 for months. Both lose most of
their value at c=192 because the GPU becomes the wall again and Python
dispatch hides under it.

4. **Rust serve loop + CUDA-graph replay** (+15-25% at c=128, ~zero at
   c=192): steady-state pack→H2D→graph-launch→D2H entirely in Rust, Python
   only for first-sight shape capture. `_GraphCache` (built, parity-exact,
   flag-off) is the prerequisite half of this. 1-2 weeks careful work,
   elevated correctness risk (silent-wrong-eval class).
5. **Bespoke attention kernel for d=32** — superseded by the main_7 head
   reshape below; do not build a d=32 kernel if main_7 moves to d=64.

---

## Tier 3 — main_7 (c=192, deeper, more attention), designed for speed

### 3.1 Architecture recommendation

| knob | main_6 | main_7 proposal | why |
|---|---|---|---|
| channels | 128 | **192** | requested |
| heads | 4 x 32 | **3 x 64** | THE free speed win: every fast attention kernel (flex/FA2) is tuned for head_dim 64; d=32 runs at ~10 TFLOPS, d=64 runs 2.5-3x faster at identical FLOPs. Zero quality cost expected; per-head bias table just becomes 3 columns. |
| trunk | CCC A CCC A CC A (8 conv blocks, 3 attn) | **CC A CC A CC A CC A CC A** (10 conv blocks, 5 attn) | "moderately deeper + a few more attention blocks": +25% convs, +67% attention, and attention appears earlier/more evenly (first A after 4 convs instead of 6). |
| MLP ratio | 2 | 2 | keep — MLP is cheap and not the bottleneck |
| radius / features / tokens / bins | — | unchanged | serve stack + featurizer carry over untouched |

Params ~3x (still tiny, ~8M); VRAM is a non-issue on 12 GB.

### 3.2 What carries over from the 2026-07 speedup work (free)

- `HEXFIELD_TRITON_CONV` — the fused gather+GEMM conv kernel is
  channel-generic (%16); at c=192 its GEMMs are FATTER (K=1344) and run at
  equal-or-better efficiency.
- `HEXFIELD_FLEX_PAIR` — pair index is geometry-only, unchanged.
- `HEXFIELD_SERVE_HALF` — carries over; set main_7's serve parity gate with
  fp16 serving in mind from epoch 0 instead of inheriting the 3e-3 gate.
- `HEXFIELD_RUST_PACK`, `HEXFIELD_COPY_STREAM`, malloc tunables,
  `HEXFIELD_TRAIN_COMPILE`, vbs=48 — all model-agnostic.
- All parity/throughput harnesses (`_hexfield_serve_ref.py`,
  `_hexfield_selfplay_throughput.py` with phase timing, `_main6_*probe*`).

### 3.3 New kernel work worth doing for main_7 (ranked)

1. **Bespoke fused attention kernel @ d=64** (1-2 sessions, low risk):
   FA2-style online-softmax Triton kernel specialized to 3 heads x 64 with
   the uint8 pair-table bias fused into the score loop and fully-padded key
   tiles skipped. With 5 attention blocks at c=192, attention is the
   largest forward component; stock flex @ d=64 is decent (~25-30 TFLOPS),
   a bespoke kernel targets 40+. Worth ~10-15% of the main_7 forward.
2. **LayerNorm/mask/ReLU fusion into the conv epilogue** (1 session,
   low risk): each ConvBlock re-reads/re-writes the activation for
   LN -> ReLU -> mask after the conv kernel already had it in registers.
   Fusing the epilogue saves one full activation round-trip per conv
   (~20+ convs). Worth ~5-8%.
3. **fp8 (e4m3) conv GEMMs** (1-2 sessions, MEDIUM numerics risk): Ada
   tensor cores do fp8 at 2x fp16; the conv kernel's tl.dot can take fp8
   inputs with per-channel weight scales. Conv is the second-largest
   component at c=192; worth ~10-15% total. Gate it with a fresh main_7
   parity tolerance and the arena eval, same playbook as serve_half.
4. NOT worth it at main_7: Rust serve loop (GPU-bound again), windowed
   attention (radius-16 neighborhoods are bigger than the whole support at
   current board sizes — the bias table's own far-row structure was
   checked and locality does not pay), CUDA graphs (same reason as Rust
   loop).

### 3.4 Speed projection — "could it be comparable to the current one?"

Scaling from tonight's measured forward split (FLOPs x efficiency, then
utilization):

- Raw FLOPs: convs ~3.1x (2.25x width² x 1.25x depth), attention ~3.5x
  (2.25x width² on projections/MLP, 1.5x width on scores, 1.67x blocks).
  Blended ~3.2x.
- Efficiency recovered: d=64 heads (~2.5x on the attention kernel), fatter
  GEMMs everywhere, GPU-bound balance (utilization ~80% -> ~95% with NO
  host work — the dispatch hides under the bigger forward).

| configuration | serve cost vs today's main_6 | est. live pos/s |
|---|---|---|
| main_6 today (c=128, round-2) | 1.0x | ~22 |
| main_6 one week ago (pre-speedup) | — | 13.75 |
| main_7 naive (c=192-deep, d=32 heads, no new kernels) | ~2.6x slower | ~8-9 |
| main_7 + d=64 heads (architecture only) | ~1.9x slower | ~11-12 |
| main_7 + d=64 + bespoke attn kernel + LN fusion | ~1.55x slower | **~14-15** |
| main_7 + all of the above + fp8 convs | ~1.35x slower | **~16-17** |

**Answer: yes, effectively comparable.** A 3x-FLOPs main_7 built this way
serves FASTER than main_6 did before this week's work (13.75), at ~70-75%
of today's optimized throughput. Two additional effects close the rest of
the gap in practice:
- a stronger net typically needs fewer visits for the same move quality —
  budget-recalibrating `search_visits`/`gumbel_m` for main_7 (the
  established calibration playbook) plausibly recovers 10-30% of
  decisions/s at equal strength;
- epochs are worth more per position (better targets), so wall-clock per
  Elo is the honest metric, not pos/s.

### 3.5 Bring-up plan (established patterns, nothing novel)

1. Constants: `HEXFIELD_CHANNELS=192`; heads 4->3 and the new trunk order
   are code changes in model.py (BIAS table columns follow ATTENTION_HEADS
   automatically).
2. Warm-start via BC prefit on recent main_6 samples (the main_4-from-main_3
   playbook; prefit stays on its eager path — the TRAIN_FLEX perf note
   still applies).
3. Before launch, re-run the three calibration/parity gates at c=192:
   serve parity battery (set the fresh tolerance), forward profile
   (`_hexfield_main6_profile.py` with the new constants — decides whether
   kernels behave at c=192 shapes), and eval-side gumbel_m budget
   calibration (in-tree from the 2026-07-01 work).
4. Launch config: carry vbs=48, all round-1/2 env flags, PAIR_BUDGET
   probably needs a small DOWNWARD retune for training (c=192 activations
   in the flex backward — the 8e7 experiment showed where that cliff is).

### Effort summary for the main_7 speed package

| item | effort | risk | payoff |
|---|---|---|---|
| d=64 heads (arch) | ~0 (design choice) | none | ~1.4x serve vs naive |
| bespoke d=64 attention kernel | 1-2 sessions | low (parity-gated) | ~1.12x |
| LN/epilogue fusion | 1 session | low | ~1.06x |
| fp8 convs | 1-2 sessions | medium (numerics) | ~1.13x |
| visit-budget recalibration | 1 session | quality-neutral by design | 1.1-1.3x decisions/s |

Total: roughly a week of kernel work spread alongside the run, most of it
reusing this week's harnesses, for a 3x-capacity model that trains at
~main_6's historical speed or better.
