# hexfield-v2 ("Hexfield-Plus") — best-theoretical-model synthesis

Status: **PROPOSED — graft roadmap, not a rewrite.** Synthesized 2026-06-13 from a 3-stage
multi-agent design competition for "the best theoretical ML model for Hexo": **21 complete model
designs** (one per ML paradigm/lens) → **21 adversarial validations** (1:1, scored) → **18
dimension synthesizers + 2 grand integrators + 1 adversarial merger**. The merger chose the
buildability-led blueprint as the base and imported four corrections from the strength-led one.

> **Headline result.** No exotic paradigm cleanly beats the trusted hexfield backbone. Every
> "exciting" architecture (GNN, MuZero, Mamba/SSM, slot-attention, DEQ, diffusion, full G-CNN,
> point-cloud, threat-hypergraph) either re-proposed an **already-built-and-shelved** in-house
> design, or pitched a throughput win against a **misdiagnosed** cost while adding new cost — a
> net regression risk on the load-bearing metric. The best theoretical model for Hexo is the
> **hexfield backbone kept verbatim, plus a small ranked set of additive, individually-gated,
> game-native grafts**, most of which *promote hexfield's own parked §11 levers* under the
> competition's convergent mandate. Ship them one at a time off the live baseline.

The competition's full per-design ranking, critiques, and the two integrated blueprints are in the
run artifacts. This doc is the merged, decision-ready conclusion.

---

## REVIEW VERDICT (2026-06-13, 30-agent code-grounded review)

A follow-up 30-agent review read the **actual repo code** (not this doc's claims) and corroborated
the run diagnostics end-to-end. Net finding: **as packaged, these grafts are NOT yet a proven
improvement over the live run.** Only three *near-free correctness fixes* are near-certain wins —
and none is a headline architectural graft. Everything that touches play strength genuinely
requires a runtime A/B to know, and several grafts are redundant with the engine verdict / existing
planes / augmentation, or target a crop-era disease the architecture deletes by construction.

**Corrections to this doc (verified against code/diagnostics):**
- **Worst-calibrated epoch is ep1 (ece 0.340), not ep2 (0.299).** The min-ECE argument survives
  (it would pick ep0 at 0.108 over the live ep2 warm-start), but the "ep2 = worst" label is wrong.
- **`w_value` is not a real knob.** Value weight is `VALUE_WEIGHT=1.0` hardcoded (`losses.py:27`)
  with no TOML/CLI path; "down-weight value during BC" is an *unbuilt code change*, not a flip, and
  is confounded — park it (it is NOT a clear win).
- **The throughput floor is uncorroborated in-repo.** No committed hexfield artifact backs "5.82
  pos/s @512v ≈ 1.2× dense, above the 0.8× floor"; the only perf logs are a stale hexgt-lineage
  256v profile, and the throughput script compares against `restnet ~9.7 pos/s @256v` (0.8× =
  7.8) — at which 5.82 is *below* floor. Every systems verdict (B/D cost, X, P, Cap) hangs on this.
- **The "NaN/Inf grad-norm storm" is a logging artifact**, not instability: `prefit.py:139` appends
  `float(grad_norm)` unconditionally on AMP-skip steps while `trainer.py:126` is already guarded;
  `grad_norm_p95` is a healthy 4.0–4.2 and `amp_scale` climbs. So "harden GradScaler" is cosmetic;
  the value collapse is label-correlation overfit, not a NaN pathology.
- **`K-lengthknees` is not "config-only"** — hexfield has no policy-surprise row-weighting layer to
  attach the knees to; promotion requires porting machinery. **`P-packeranchor` is ~90% already
  shipped** and redundant with the live `torch.compile(dynamic=True)`. Both were adversarially
  overturned from `gate_ab` → `park`.

**Reconciled per-graft verdict (3 independent arbiters, unanimous tiers):**

| Tier | Grafts | Note |
|---|---|---|
| **IMPLEMENT NOW** (near-certain) | STEP0 **min-ECE checkpoint selection** (switch RL warm-start ep2→ep0); STEP0 **`prefit.py:139` isfinite one-liner**; STEP0 **port the 500-step LR warmup into the RL trainer** (drop the bundled head-LR re-warm + value-only pass) | correctness/curriculum only; the min-ECE switch is the single highest-value item. **Not** the headline grafts |
| **GATE A/B** (improvement unknown; isolate; measure premises first) | **M-movesleft** (first reconcile to spec: demote default-ON→OFF behind the L0 heal-gate); **at most ONE** of {A1-maxpool, A2-tokens, B-axisconv} in isolation (B needs its own M8 throughput re-measure of the un-budgeted on-axis index); **X-flexattn** (measure vs the *already-live* torch.compile/fused-SDPA baseline, not eager) | all `low` actual-improvement-likelihood; deltas likely below the current ~2000-row noise floor |
| **PARK / DROP** | D-gradedplanes (park — owner already parked; breaks byte-exact ABI), E-antilength (park — value-target-family distortion vs a deleted disease; myopia risk), K-lengthknees (park), P-packeranchor (park), C-gradedaux (drop — tautological/redundant with D), F-tiedtap (park/drop — dead gate, probe falling), Bias-D6 (park/drop — dead gate), Cap-contingency (park — demote the GatedResBlock arm below plain-block) | redundant, premature, mis-specified, or fighting a crop-era disease the support set removes |

**Gate before spending ANY A/B slot:** (1) commit a real hexfield median-N throughput artifact at
production 512v vs a 512v dense reference; (2) measure whether the in-crop lengthening residual even
reproduces on the unbounded board. **Dominant hazard:** attribution collapse — the roadmap touches
the fragile BC→RL value head from four directions and the axis representation from five, on a signal
within ~1.3σ of 50%, which would make the project's own M3 `value_ece≤0.08` steering gate (itself
possibly unreachable — best ever 0.108 at ep0) unattributable. **Serialize and isolate.**

---

## 0. The one organizing principle (resolves every cross-dimension conflict)

The engine — `packages/hexo_models/rust/src/threats_shared.rs` — **already computes the exact
phase-aware min-hitting-set / forced-win / forced-loss verdict at every leaf** (`analyze()` →
`ThreatAnalysis`, with the correct *disjoint windows = hitting set 2 / shared cell = 1*
distinction, capped at the per-node budget B, and a `has_threats()` short-circuit), and the search
consumes it three ways in Rust: tactical-cell injection at expansion, a **hard leaf-value override**
(proven leaves are backed up directly and *never enqueued to the GPU* — a throughput feature), and
a root move-guard.

Therefore **the net's job is NOT tactical defense.** The net learns the **prior**, the **value**,
and the **strategic/representation signal in the quiet regime** where `verdict()` returns `None`.
This single fact kills a whole class of proposals (learned hitting-set / threat-count /
forced-loss / standing-win heads all re-learn an exact oracle that is already in the loop) and
points every surviving graft at the same target: *sharpen the quiet-regime representation at
near-zero throughput cost, never touch the binding 12 GB batch constraint, change one thing at a
time.*

Two more facts gate everything:
- **Throughput at the N≈600–900 median is the war that decides the run.** A model 2× too slow at
  the median loses the self-play budget even if it wins at the rare 3k marathon tail. hexfield's
  own honest projection (0.6–1.2× dense) already **overlaps** the M8 ≥0.8× floor, so the design is
  near the cliff *before* any graft.
- **The materialized O(S²) attention bias is widely misdiagnosed.** It is *not* hexfield's "#1
  cost" in the sense the exotic designs assumed. The irreducible quadratic attention term is ~**33%
  of serve MACs at N≈900 and ~45% at N≈1500** (arithmetic from the spec's `576·S²` model, M8 to
  confirm); what FlexAttention removes is only the *separate ~10–15% bias-materialization
  wall-clock tax* (the `(B,4,S,S)` transient + histogram backward). Never conflate the two. Any
  design pitching "kill O(S²) → bigger batch → more evals/s" is attacking compute that the
  whole-board disjoint-threat routing genuinely needs.

---

## 1. Competition ranking (21 designs)

| Rank | Design (lens) | Score | Verdict |
|---|---|---|---|
| 1 | **AxisField** — dilated axis-line conv trunk, D6-tied (sparse hex-conv refined) | 5 | revise |
| 2 | WindowNet — bipartite threat-hypergraph (window-as-hyperedge) | 5 | revise |
| 3 | AxialPerceiver — Hex-RoPE set transformer | 5 | revise |
| 4 | LatentHex — threat-grounded MuZero-as-representation | 5 | revise |
| 5 | TriScan — window-native Mamba/SSM | 5 | revise |
| 6 | MotifRetrieval — memory-keyed tactical evaluator | 5 | revise |
| 7 | AxisFlow — hex relative-position attention, maximized | 5 | revise |
| 8 | FoveaNet — strategy/tactics dual-network | 5 | revise |
| 9 | WindowNet (clean-slate game theory) | 5 | revise |
| 10 | ThreatGraph bipartite GNN (hypergraph dup) | 4 | reject |
| 11 | HexSteer — D6-equivariant G-CNN | 4 | revise |
| 12 | SymHexo — neuro-symbolic | 4 | reject |
| 13 | HexMoE — phase/threat-routed mixture-of-experts | 4 | revise |
| 14 | HexPyramid — hierarchical multiscale | 4 | revise |
| 15 | AxialPointFormer — point-cloud / continuous-coordinate | 4 | reject |
| 16 | HitNet — auxiliary-rich representation-first | 4 | revise |
| 17 | HexoDiffuser — generative / iterative-decoding policy | 4 | reject |
| 18 | SlotHive — object-centric slot-attention | 4 | revise |
| 19 | ThreatGraph — heterogeneous window-bipartite GNN | 3 | reject |
| 20 | HexEquilibrium — DEQ recurrent "think longer" | 3 | reject |

(21 lenses; scores varied ±1 across two full runs — top viability 5–6/10, most 4–5. The value of
the competition is the *convergent ideas*, not the winner: the best ideas recurred across many
independent designs, and the worst flaws recurred too.)

---

## 2. MUST KEEP — the trusted substrate (do not touch)

1. **Engine-true SUPPORT SET** on the native hex lattice (`stones ∪ engine-legal ∪ 1-ring halo`,
   no crop, variable-N), carried as flat CSR in `[legal | stones | halo]` with the **legal-prefix
   property** so policy is a positional prefix softmax and coverage loss is structurally
   impossible. The single load-bearing correctness substrate; must not be cropped, pooled,
   latent-compressed, or graph-of-windows-ified.
2. **Engine verdict wiring** at all three Rust sites (tactical-cell injection, hitting-set
   leaf-value hard override, root move-guard), with proven leaves never enqueued to the GPU. The
   net must NOT re-derive any tactical verdict; **no learned hitting-set/threat/forced-loss head
   crosses the ABI.**
3. **The 15 trusted input planes** with their exact thresholds and `placements≥7` gate — *retained
   as a strict subset*; graded channels are *added*, never replacing the binary planes.
4. **Trunk shape `C C C A C C A C A` at C=96** (HexNodeConv 7-tap gather+single-GEMM; full dense
   self-attention over `[8 tokens ; cells]`; one shared 237-row relative-position bias; LayerNorm
   everywhere; zero-init residual closure). **Width is LOCKED at 96** — capacity contingency is
   *depth* (7th/8th block, dense_cnn's proven 96×8) or dense_cnn's `GatedResBlock`, **never width**
   (conv term scales C²·N and dominates at the median).
5. **65-bin distributional value head, segment-softmax policy over the legal prefix, opp_policy
   train-only @ 0.25**, and the **variable-N exactness discipline** (mean-over-rows with
   step-global denominators → micro-bucket accumulation bit-exactly equals a monolithic batch
   under LN). Masked-BN stays a contingency only.
6. **The kernel-dispatch traps** that took three separate discoveries: fp32 `_BiasGather`
   histogram backward, finite −3.0e4 pad-key mask, stride(-1)==1 contiguous `attn_mask` layout
   (keeps the fused fp16 SDPA kernel firing), and the dual sdpa/materialized impl as oracle.
7. **D6 by training-time augmentation** as the shipped default (exact, zero activation cost;
   verified working — `probe_d6_kl` 0.093→0.064 and falling). Full 12× G-CNN is a documented dead
   end (4× activations shrink the binding batch; transposition sharing dead at 1.2% hit rate).
8. **The shipped serve packer** (`plan_groups`: QUANT_NODES=64, WASTE_FRACTION=0.18,
   PAIR_CEILING=3.8e7) + dedup/cache path, and the byte-identical ABI.
9. **The four search divergences as-built** (LCB greedy, early-stop overtake pruning, visit-scaled
   c_puct as a lesion candidate, moves-left utility mechanism) and the exploration block verbatim
   (KataGo total-alpha Dirichlet `per-move = total/num_legal` — the only branching-invariant choice
   for the 271→1500+ legal-count swing).

---

## 3. The grafts (ranked, additive, each byte-equivalent when off)

Every graft is gated by an oracle/property test at its milestone + an M3 BC A/B + a matched-visits
arena A/B + an M8 throughput re-measure + a **pre-committed one-line reversion**. None widens C,
adds a custom kernel, or introduces ragged batching that defeats the static-shape packer.

| ID | Graft | What | Why (game fit) | Serve cost | Risk |
|---|---|---|---|---|---|
| **A** | **Masked MAX-pool into value** + **threat-seeded per-axis tokens** | value/aux input = `concat(T0,T1,mean,max)=384`; init 3 of 8 summary tokens as Q/R/QR aggregators pooled from existing hot/win-now cells (≥1 free slot, empty-set→learned-constant fallback) | sudden-death value hinges on the single worst window; mean-pool over 600–1500 cells washes it out. Per-axis tokens are D6-equivariant under the axis 3-cycle | ~0 (one masked reduction; tokens reuse existing planes) | low — **mandatory fp32 −inf pad-fill before max** + pad-inertness unit test |
| **B** | **Axis-line convolution** (×2) | two depthwise 11-tap on-axis strip convs (one per Q/R/QR), then C×C recombine; zero-init residual; rot60-tied filters | a 6-window spans 5 steps; an isotropic 7-tap needs ~5 stacked layers to see it, an on-axis stencil reads it in O(1) depth; only a per-axis line op represents the same-line (defensible) vs cross-axis (fork) asymmetry that *is* the game | ~25k·N ≈ 3% of the linear term | **needs its own composed on-axis index** (`support.py` nbr is radius-1/6-dir only — dilation would read off-support; strip clipped to support, off-support taps = pad-zero) parity-pinned vs engine window membership |
| **D** | **Graded per-axis line-potential INPUT planes** (F=15→21) | planes 15–20 = max single-colour window fill-count per axis per side, scattered to empty cells, count/5 ∈ {0.2..1.0}; **superset** (binary hot/win-now planes retained) | axis identity + count grade (3 = one turn from threat, 4 = threat, 5 = win-in-1) is the race-tempo distinction the rules turn on; generalizes the §12.7 standing-win-plane decision | +6 fp16 wire channels (+12 B/node, ~26%), +576 stem params; scan already in hot path | **named largest representational risk** — ship LAST, superset form, one-line reversion to F=15 |
| **E** | **Anti-lengthening package** (RL phase only) | per-move value discount γ<1 (~0.997–0.999/decision) at target construction + sudden-death focal value weighting (up-weight flip rows) | sharpens near-terminal value, de-incentivizes dawdling | train-only | **secondary** fix; γ changes the value-target family → introduce at RL boundary with value-head LR ramp; gate on M9 length band |
| **C** | **Train-only graded-potential aux head** (PARKED) | per-cell `Linear(C→6)` predicting the per-axis fill, weight 0.05 | shapes the trunk to encode axis-resolved fill without touching the input contract | 0 (train-only) | **likely redundant** with F=21 inputs → parked; enable only if B+D show the trunk failing to encode it; grad-cos kill-switch |
| **F** | **Axis-permutation tied-tap HexNodeConv** (PARKED) | reparametrize the (7,C,C) taps from an orbit-tied free set; identical FLOPs/activations | activation-free partial D6 equivariance | 0 | reduces directional free capacity; only if the D6 probe shows augmentation NOT converging late (unlikely — probe is fine). Try the D6-orbit-tied **bias** table first |

---

## 4. ADOPT NOW (the ladder, in order)

**STEP 0 — zero-cost curriculum + transition hardening (fixes the one *measured* failure; ships
before any architecture change).** The real BC run (`runs/hexfield_bc_1`) showed policy igniting
cleanly (top-1 0.387→0.398→0.405) but **value calibration collapsing** (held-out `value_ece`
0.108→0.339→0.299 while train value-CE fell) — classic overfit on a value-poor signal (one
bootstrap label shared by 60–200 correlated positions, no draws).
- Down-weight value to `w_value ≈ 0.25–0.5` during BC (policy is the BC product; MCTS + the exact
  engine verdict carry tactical value; RL relocates the scalar).
- **Select the BC→RL handoff checkpoint by MIN held-out `value_ece` AND top-1, never by epoch
  index** (the run currently warm-starts RL from epoch-2, the *worst*-calibrated checkpoint).
- Make `nan_trips` a counted, gated metric; harden GradScaler against the verified NaN/Inf
  grad-norm + `clip_fraction≈1.0` storm.
- Anchor M3 top-1 against a **dense_cnn(model1)** BC number on the same corpus/split (model1 is the
  best learner; restnet is the struggling reference).

**BC→RL transition build-out (the genuinely under-built link).** Verified: `initialize_from` does
not load optimizer state, `trainer.py` has no warmup (`config.warmup_steps=0` is unread; warmup
lives only in `prefit.py`), so RL begins with converged weights + a cold AdamW + full 1e-3 LR on
step 1 — a large unconditioned update at the most fragile moment.
- **Port** the 500-step linear LR warmup into the RL trainer, gated on *optimizer-state-not-loaded*
  (not `global_step==0`, so genuine resumes aren't re-warmed).
- First-500-RL-step clip-fraction tripwire; re-warm value/STV/moves-left head LRs (2–3× lower for
  ~1k steps); ramp `w_value` back to 1.0 only after a short value-only warm pass on the first
  self-play shards.

**STEP A — graft A** (max-pool + threat-seeded tokens). Cheapest, architecture-agnostic, zero
serve-MAC.

**Promote the dormant C1 replay length-decay knees** to live (config-only) as the *first*
anti-lengthening step, before any value-target-family change.

**Ratify the 64-quantum waste-aware packer** as canonical and add the small **closed anchor list**
(`{…,384,512,768,1024,1536,2048,3072,+ceil tail}`, ≤8 shapes) as torch.compile/CUDA-graph keys
(the only not-yet-present piece).

**Demote the moves-left search utility to default-OFF** behind the L0 heal-gate + nightly
control-flip probe + per-epoch auto-disable (defensible conservatism — but see open question #1).

---

## 5. GATED UPGRADES (the rest of the ladder)

- **STEP B** — axis-line conv. Gate: M0 on-axis-index parity vs engine window membership + M1
  oracle vs a stacked-isotropic reference + **M8 ≥0.8× floor (its O(N) cost is the first real
  throughput test)** + M3 BC A/B. Run an initial lesion arm fed the *binary* planes to attribute
  the primitive independently of graded inputs.
- **STEP D** — graded INPUT planes F=15→21 (LAST). Re-pin the byte-exact ABI golden +
  `NUM_FEATURES` + abi version guard in lockstep. Switch STEP B to consume the graded channels;
  compare vs its binary-plane lesion arm. Gate: M3 BC A/B (top-1 within 2 pts, `value_ece≤0.08` on
  the min-ECE checkpoint) — this gate must **steer** (value down-weight + ECE selection), not
  merely trip.
- **STEP E** — γ<1 + focal at the RL phase (M9). Gate: `_hexfield_band_check` game-length ratio in
  [0.5, 2.0] + value-CE-by-length-quartile not regressing; **hold C1 length-decay FIXED in the γ
  arm** (relax, never stack three shortening pressures).
- **FlexAttention** (perf fast-follow, parallel track): inline `score_mod` bias to delete the
  materialized transient + enable jagged batching — **explicitly NOT to change the O(S²) math**.
  Gate: oracle-equivalence + ≥10% measured win on the actual WSL/Triton stack; sdpa/materialized
  dual path remains the always-correct default so v1 ships unblocked. `torch.compile(dynamic=True,
  ≤8 specializations)` on the frozen inference clone, OFF until M8.
- **STEP C / STEP F** — parked contingencies (see table).
- **Capacity contingency** (if M3/M9 underlearn *after* B+D): +7th/8th conv block, then
  GatedResBlock, each A/B'd one-at-a-time; masked-BN (conv-blocks-only) only on a 2-of-3
  quantitative trigger. **Width frozen at C=96 throughout.**

---

## 6. REJECTED (and why) — the convergent failure mode

- **Window-node bipartite GNN / threat-hypergraph (ThreatGraph, WindowNet, SymHexo).** This exact
  architecture was **built and removed in-house** (hexgnn RETIRED `NODE_TYPE_WINDOW`); that lineage
  is the documented struggling learner. The "active windows are sparse" premise is false (active
  windows ≈ 50–60% of support cells; window-window mean degree ~35), and variable-degree scatter
  defeats the static-shape packer. The min-hitting-set head re-learns an exact engine oracle.
- **Throughput-by-killing-O(S²) (TriScan/Mamba, AxisFlow, MotifRetrieval, SlotHive, HexMoE).**
  Attacks a ~10–15% bias-*materialization* tax (already FlexAttention-fixable) while adding SSD
  scans / multi-axis re-sorts / retrieval / experts that raise the *dominant* term → net regression
  at the median N. Custom kernels assumed don't exist on this WSL/Triton/no-TRT stack.
- **Full 12× G-CNN (HexSteer) / MuZero (LatentHex) / DEQ (HexEquilibrium) / diffusion
  (HexoDiffuser) / Perceiver bottleneck.** 4× activations shrink the binding batch; learned
  dynamics earn nothing when the engine is cheap and exact; adaptive depth defeats the static-shape
  packer; an 8-token bottleneck regresses long-range disjoint-threat wiring; RoPE over 3
  non-orthogonal axes doesn't commute with D6. MCTS already expands FirstStone→SecondStone as two
  plies, so an autoregressive 2nd-stone head is redundant.
- **HexPyramid's equivariance claim is mathematically false** (floor(q/3) is an anisotropic
  partition — breaks D6 on ~64% of cells, making it *less* symmetric than augmentation).
- **Big-bang synthesis** (shipping all grafts at once): AxisField's own rank-1 worst flaw — 4
  correlated bets, no clean A/B, un-bisectable regression, and the stacked O(N) cost almost
  certainly busts the M8 floor.

---

## 7. Biggest unvalidated bets (open questions)

1. **moves-left utility default-OFF** — the spec (`STAGE3_MOVES_LEFT_FEASIBILITY`) argues the only
   blocker was a flood-damaged legacy head and hexfield's head trains on *clean* targets, i.e. the
   blocker may be absent → default-ON could be the spec-supported choice. Resolve by measuring the
   L0 heal-gate metrics through the BC checkpoint before committing to OFF.
2. **γ<1 vs genuine long forced wins** — the ~+40 dec/game *in-crop* lengthening residual
   (`MAIN4_RECOMMENDATION.md:34`, the part crop-deletion does *not* cure) is unproven on hexfield's
   unbounded board, and γ<1 risks making value myopic about real long forced wins. The single
   biggest unvalidated bet; M9 length-band + value-CE-by-quartile is the only arbiter.
3. **Will stacked O(N) grafts clear the ≥0.8× median-N floor**, given the 0.6–1.2× baseline already
   overlaps it *before* any graft? Answer by M8 measurement of *each graft in isolation* against
   the live baseline; FlexAttention + half-B contingency must be **pre-built, not speculative**.
4. **Is the BC value collapse fully explained** by label correlation, or is the NaN/Inf grad-norm
   storm corrupting the value head specifically? The fix treats both; if a third cause exists, the
   M3 `value_ece≤0.08` gate may be unreachable even on the min-ECE checkpoint.
5. **Does F=21 destabilize BC beyond what the superset form protects against** — can the net
   actually *ignore* the graded channels to recover F=15, or does the wider stem change the
   optimization trajectory enough that the A/B isn't one-variable? Verify the stem fan-in is the
   only delta and the binary planes are bit-identical.
6. **Actual measured build cost of the per-batch on-axis index** in the featurize↔forward overlap
   pipeline at N≈900 — asserted to amortize like `nbr`/`gather_idx`, but unmeasured; if it doesn't
   overlap cleanly it's a hidden O(N) serve tax on top of the AxisLineConv compute.

---

## 8. Provenance

3-stage workflow (`hexo-best-model-design`): 21 designs → 21 adversarial validations (pipelined) →
synthesis. The synthesis stage was rate-limited twice (a synchronized ~0.5M-token burst from 20
large-prompt agents tripped a server TPM throttle); it was recovered with a throttle-safe
stage-3-only workflow (`hexo-synth-stage3`: compact ~5k-token digest, 18 dimension synthesizers in
3 waves of 6 with barriers, 2 grand integrators, 1 adversarial merger). Total ≈ 11M agent tokens.
The merger selected the buildability-led blueprint as base and imported the honest MAC accounting,
the lengthening-cure scoping, the aux-head demotion, and the M8-floor owner-sign-off framing from
the strength-led blueprint. Full per-design critiques and both blueprints are in the run artifacts.
