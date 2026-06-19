# hexfield_main_4 — Final Config Review & Recommendation

Synthesizer pass over 53 per-knob critiques + targeted CPU simulations.
Date: 2026-06-19. Reviewer lens: **Hexo, not Go** — sudden-death/no-draws,
[-1,1] no-score Q (measured mean|q|~0.18, 35% of children |q|<0.1),
unbounded/growing branching (337-777 legal), 2 stones/turn, and main_3's
**measured pathology = conversion-failure / defensive-lock** (mean length
41->157 ply, 40-100 truncations/epoch, root_value_mean~0, opening collapse to
the (-4,0) orbit). The governing principle below: **favor decisiveness and
conversion, avoid lengthening non-converting games, keep tactical safety, do
not import Go constants unexamined.**

Config audited from `claude/hexfield-main4:configs/hexfield_main_4.toml` and the
backing Rust/Python on the same branch.

---

## TL;DR — what to change vs the planned main_4

| # | Knob | from -> to | Confidence |
|---|------|-----------|------------|
| 1 | **expand_backend** | `"rust"` -> **`"serial"`** (until replay_expand.rs projects value_mask) | **HIGH** |
| 2 | **root_policy_temperature** | `1.1` -> **`1.05`** | HIGH |
| 3 | **root_policy_temperature_early** | `1.25` -> **`1.07`** (i.e. flat / ramp ~off) | MED |
| 4 | **temperature_halflife_plies** | `60.0` -> **`30.0`** | MED |
| 5 | **root_dirichlet_noise_fraction** | `0.25` -> **`0.15`** | MED |
| 6 | **forced_playout_k** | `2.0` -> **`1.0`** | MED |
| 7 | **moves_left_weight** | `0.1` -> **`0.2`** | MED |
| 8 | **soft_policy_weight** | `8.0` -> **`4.0`** + soften power to `^0.5`, support-only target | MED |
| 9 | **pcr_fast_visits** | `128` -> **`192`** (secondary; only if throughput allows) | MED-LOW |

**c_puct: KEEP 1.5** (do NOT drop to KataGo's 1.1). See decision section below.

Everything else in the planned main_4 is **kept as-is** (the six KataGo-faithful
search divergences, FPU mass-scaling intent, lcb_z, c_scale=0, c_base, root_fpu=0,
TSS, the replay/cadence cluster, value/policy weights, lr, batch, clip, wd).

---

## The c_puct 1.1 vs 1.5 decision (explicit, from simulation)

The toml ships c_puct=1.5 "pending an explicit exploration-vs-faithfulness
decision." **Decision: keep 1.5.**

I ran a PUCT root sim under the *actual* main_4 regime (`c_scale=0` so c is a
constant; `root_fpu_reduction=0`; `forced_playout_k=2`; shaped Dirichlet
alpha=10.83 frac=0.25; root_policy_temp 1.1-1.25; compressed Q calibrated to
mean|q|~0.18). Results (`/tmp/main4_sims/cpuct_decision.py`):

```
Scenario                c=1.1                  c=1.5
block rank3 (mid)       found=0.87 eff= 9.7    found=0.93 eff= 9.7
block rank8 (late)      found=0.87 eff=10.4    found=0.88 eff=10.4
win   rank8 (mid)       found=0.60 eff=11.2    found=0.66 eff=11.1
win   rank16 (mid,hard) found=0.47 eff=11.2    found=0.58 eff=11.4   <-- +11pp
opening breadth         eff=11.6 top1=0.577    eff=12.0 top1=0.573
mid breadth             eff=12.3 top1=0.561    eff=12.5 top1=0.556
late breadth            eff=12.6 top1=0.556    eff=13.1 top1=0.539
```

Two load-bearing facts:

1. **c=1.5 is uniformly >= c=1.1 on tactical conversion/block reliability**, and
   the gap *widens* exactly where Hexo's conversion failure lives — underrated,
   barely-in-set winning lines (rank16 win: 0.58 vs 0.47). With a compressed/weak
   value head the search is prior/U-dominated, so the U-term (scaled by c) is what
   re-deepens onto the correct-but-underrated move. A *higher* c helps conversion
   here, the opposite of the Go intuition.
2. **Breadth is essentially identical** between 1.1 and 1.5 (eff moves and top1
   share differ by <0.5 / <0.02). Forced playouts + the binding nucleus set
   breadth, not c — so 1.5 does **not** over-explore. This reproduces the repo's
   own grid ("c_puct dead lever within noise") on the *breadth* axis while showing
   a small but consistent *conversion* edge for 1.5.

**Interaction that seals it:** main_4 simultaneously zeroes `root_fpu_reduction`
(0.2->0) and `c_scale` (0.45->0), both of which *remove* exploration/selection
pressure. Dropping c_puct to 1.1 on top would be a third simultaneous reduction
in the U-channel that the prior/U-dominated, weak-Q search actually relies on to
convert. Keep 1.5 to partially offset the root_fpu/c_scale zeroing. KataGo's 1.1
is a Go self-play constant tuned with a wide score-utility Q (~1.4 radius) that
re-sharpens visits for free; Hexo has no such force.

---

## Changes (from -> to + why)

### 1. expand_backend: "rust" -> "serial"  [HIGH — correctness bug]
**This is the single most important change.** The toml keeps `expand_backend="rust"`
with the comment "the soft target is derived in collate, so the Rust kernel needs
no change." That reasoning is correct for the *soft-policy* target but irrelevant
to the real defect: **truncated-game value masking.**

Verified on-branch:
- `expand_backends.py:333-343` — when the Rust result has no `value_mask` key it
  falls back to `np.ones(r)`, i.e. **every Rust row is treated as a completed
  game.** The code's own TODO says replay_expand.rs must read `outcome_valid` and
  emit `value_mask` "BEFORE expand_backend is flipped to rust on a buffer that
  contains truncated rows."
- `replay_expand.rs` — grep for value_mask/outcome_valid returns **only
  `stvalue_mask`**; there is no top-level `value_mask` output and no
  `outcome_valid` read.

main_4 *inherits truncated-game training* (policy/opp_policy trained on truncated
rows, outcome heads meant to be masked). So under `rust`, the value/stvalue/cell_q
heads on truncated games are silently **un-masked** and fed a fabricated/bootstrap
outcome. Truncated games are the longest (157+ ply) and most numerous in the
lock pathology (40-100/epoch); a sim of the row mix shows **32% (40 trunc) to 63%
(100 trunc) of value-head rows mislabeled.** In sudden-death Hexo where targets
should be clean +/-1, this either pins the value head at ~0 (reinforcing the very
defensive lock main_4 must break) or injects pure label noise on the head whose
calibration drives conversion. This also corroborates the standing MEMORY note
("do NOT flip expand_backend to rust until replay_expand.rs projects
outcome_valid"). **Set `serial` (or `pool`) now; restore `rust` only after a new
truncated-fixture parity test passes.**

### 2. root_policy_temperature: 1.1 -> 1.05  [HIGH]
The planned 1.1 *raises* flattening from main_3's 1.07. Wrong direction. The
prior is the dominant search signal (compressed Q can't re-concentrate visits),
the pathology is conversion-lock not under-exploration, and the sibling Hexo line
(dense_cnn_restnet) grid-searched this on the same game and moved 1.1 -> 1.05,
concluding "every flattening increment costs monotonically." 1.1 also pushes more
mass past the widening cap (worse cap-bind at ply7-20) and dilutes the few
converting moves late. Keep at or below 1.07; 1.05 matches the Hexo-validated
sibling and is the safe choice for a fresh run.

### 3. root_policy_temperature_early: 1.25 -> 1.07 (ramp effectively off)  [MED]
1.25 is a Go constant. Sim: vs an already-saturated opening (Dirichlet already
gives ~70 distinct first stones, matching measured support), 1.25-early adds ~1
distinct first move while raising risky-tail (rank>=30) opening picks 16.8%->24.2%.
In sudden-death Hexo a single unsound opening stone that seeds a completable
threat is far costlier than in scored/draw Go. The opening collapse is a *learned*
value/policy collapse — renormalizing the prior to the 1/1.25 power is
ranking-invariant and will not un-collapse it. Spend the opening-diversity budget
on Dirichlet (already present), not prior-flattening. If a ramp is kept at all,
cap early at <=1.10 with a short halflife; otherwise set early=steady=1.07 (which
makes `root_policy_temperature_halflife=19` a harmless no-op — leave it).

### 4. temperature_halflife_plies: 60 -> 30  [MED]
This is the chosen-move sampling halflife. At ply40 (conversion-critical) hl60
gives T=0.63 and only a 0.43 chance of playing the search's top (converting)
move, with a 0.32 chance of a rank>=3 move — i.e. it keeps mid/endgame play too
random exactly where decisiveness is needed, manufacturing both blunders and
longer non-converting games (length sim rises monotonically with halflife:
45->50->57 ply at hl19/30/60). Crucially, raising halflife buys ~zero opening
diversity (opening Neff is governed by Dirichlet + root temp, flat across hl). The
config's stated reason for 30->60 ("greedy endgame collapse") misdiagnoses the
pathology, which is *too-random non-conversion*, not over-greediness. Revert to 30
(floor still 0.15, reached ~ply82). Pairs with the moves-left decisiveness lever.

### 5. root_dirichlet_noise_fraction: 0.25 -> 0.15  [MED]
Exploration is not the bottleneck. With total_alpha=10.83 over Nlegal 337-777 the
per-move alpha is 0.014-0.032, so the draw is extremely spiky (~88% of mass on
~18 random cells); eps=0.25 hands those ~18 junk cells ~28% (143) of the 512-visit
budget, starving the tactical depth that converts. Discovery sims show high eps
never materially finds deep underrated wins in this Q-compressed/large-Nlegal
regime, it only dilutes. `dirichlet_shaped=true` (kept) already halves junk mass,
which makes a *lower* fraction safe and complementary. Trim to 0.15; raise back
only after value sign-acc improves. (Also reconciles the config-family
discrepancy — the dense main_4 toml already carries 0.20.)

### 6. forced_playout_k: 2.0 -> 1.0  [MED]
KataGo's k=2 was tuned for a ~361 board with a sharp score-utility value. At
Hexo's flat opening prior (top1~0.12 over 337-777) the forced floors
n_forced=floor(sqrt(k*p*N)) sum to ~48-60% of the 512 budget and are applied as a
root override, spreading visits across all ~96 widened children down to prior
~0.001 — *spraying*, not tail-tasting. PV depth drops ~half, and the compressed Q
can't claw it back. That directly starves opening conversion in the exact ply
range the run collapsed. Lowering to 1.0 roughly halves the override fraction
while preserving PV depth; the decorrelation goal is already served by widening +
Dirichlet. (Export-side prune limits *target* damage but cannot refund the
shallower in-search reads that produce the blunders.)

### 7. moves_left_weight: 0.1 -> 0.2  [MED]
The moves-left head feeds the in-search moves_left_utility — the **one lever
explicitly built to attack the conversion-failure/defensive-lock pathology** —
and an auto heal-gate disables that utility when the head is unhealthy. So a
starved head means the decisiveness lever silently stays OFF (opportunity cost,
not a search risk: the gate protects search). Hexo makes the head harder than Go
(target 0..209 decisions, bimodal with a marathon tail). Head calibration rises
monotonically with weight; 0.2 costs only ~3.5% of the value head's trunk
gradient share. Escalate to 0.3 if the head_audit still fails after ~3 epochs; do
not exceed 0.3 (value/policy precision is the costliest thing to erode in
sudden-death Hexo). Keep ml_two_sided=false and ml_final_pick_band=0.08.

### 8. soft_policy_weight: 8.0 -> 4.0, power ^0.25 -> ^0.5, target over visited support only  [MED]
`(p+1e-7)^0.25` at weight 8.0 is a savage flattener and is board-size-coupled:
with Hexo's 337-777 legal cells the unvisited-legal tail absorbs 40-72% of soft
target mass, and on a near-forced defensive block (top1=0.85) it collapses the
must-play target to 0.10 (literal) or 0.44 (support-only). Teaching the policy
head to put <half its mass on the unique 2-stone hitting move is exactly the kind
of blunder the conversion pathology is made of, and Hexo (no draws) cannot absorb
it the way Go can. hexfield *also already runs* KataGo's policy-surprise reweight
at max_weight=8.0 — a second softening of the same head — so 8.0 here
double-flattens. If the soft head is kept this run: weight 4.0, power ^0.5,
compute the soft target over the **visited support only** (no eps-leak onto the
full legal set). Lower-risk alternative: do not introduce the soft head in main_4
at all (the surprise reweight already regularizes the policy head).

### 9. pcr_fast_visits: 128 -> 192  [MED-LOW, throughput-gated]
67% of moves are fast/greedy and shape the training data. Immediate blocks are
safe at 128 (protected by TSS, not visit count — verified), but 2-3-ply
conversions that 1-ply TSS can't prove ride the compressed value head: late-game
fast moves blunder 10.8% at 128 vs 0.3% at 512. 192 buys deeper conversion
resolution on the majority of moves. **But** deep-game forward is GPU-FLOP-bound
and this is +50% on 67% of moves — adopt only if pos/s headroom exists, and prefer
spending the budget by lifting `pcr_full_proportion` 0.33->0.40 instead (full
moves also add noise + forced-playout exploration). Lowest priority of the set.

---

## Keep as-is (verified good for Hexo)

- **search_visits=512** — past the tactical knee; raising it cannot touch the
  widening/Q-compression mechanism behind conversion failure and costs pos/s.
- **pcr_full_proportion=0.33** — pure row-density dial; the recorded-row *mix* is
  p-invariant so it can't fix or worsen the pathology. (May rise to 0.40 only as
  the throughput-friendly way to buy fast-move quality, see #9.)
- **c_puct=1.5** — see decision above.
- **c_scale=0.0, c_base=500, visit_scaled_c_puct=true(no-op)** — self-play-faithful;
  the log ramp is anti-decisive at production budgets, correctly disabled.
- **root_fpu_reduction=0.0 (root), fpu_reduction=0.2 (interior)** — zeroing root
  FPU surfaces noise-boosted underrated good moves with negligible sudden-death
  risk (Q self-corrects losers in a few rollouts). Keep interior 0.2 (do NOT zero
  interior — that weakens tactical pruning). **Interior FPU should be mass-scaled
  (`fpu_max*sqrt(policyMassVisited)`) per the new_child_fpu/KataGo form if
  implementable; if only flat is available, 0.2 is acceptable but ~0.12 is gentler
  for Hexo's compressed Q.** root_fpu_zero_under_noise=false — set explicitly
  (defuses the dual-default footgun).
- **lcb_z=1.6** — sits on a wide safe plateau; matched to Hexo's compressed Q;
  raising it risks preferring drawish high-visit moves over decisive ones.
- **root_dirichlet_total_alpha=10.83** — spiky-is-right for growing branching;
  raising toward Go's per-move 0.03 would over-broaden. (Trim the *fraction*
  instead, #5.)
- **dirichlet_shaped=true** — concentrates noise on plausible moves; strongly
  Hexo-favorable and makes the lower noise fraction safe. Verify Rust search
  parity before the live flip.
- **forced-playout / widening parity knobs** (widening_policy_mass=0.95,
  widening_max_children=96, widening_min_children=2, lazy_widening=true) — lazy
  widening makes FPU+PUCT the real gate; the 96 cap never binds at runtime
  (~13-17 children open). Keep min_children>=2 (the 2-stone defense needs >=2
  candidates).
- **nucleus_f64=true, clean_root_prior_cache=true, pruned_dynamic_cpuct=true,
  new_child_fpu=true** — genuine correctness fixes, all Hexo-benign-to-helpful.
  new_child_fpu (fpu+U baseline) removes a sign(parent_value) bias that is
  uniquely harmful in Hexo's near-0 lock band.
- **tss_enabled=true** — exact 1-ply hitting-set proof (complete because a turn
  places exactly 2 stones); the load-bearing sudden-death safety net and the
  reason 128 fast visits are safe. Never disable while Q is weak.
- **max_game_plies=256** — the knee: just above MOVES_LEFT_CAP=209 (so moves_left
  targets don't saturate), preserves ~94% of would-be-late converts vs lowering,
  and does not refuel the lock the way raising would. Fix marathons via ml-utility
  + drop_truncated_rows, not the cap.
- **ml_two_sided=false** — true would pay the *defender* to find the longest
  surviving line: a direct subsidy for the lock. Keep false.
- **ml_final_pick_band=0.08** — gated decisiveness tie-break among value-tied
  won-position moves; matched to compressed Q; 0.05 too inert, 0.12+ pulls in
  genuinely distinct moves.
- **policy_weight=1.0, value_weight=1.0, opp_policy_weight=0.25,
  short_term_value_weight=0.1, q_head_weight=0.1** — value head is calibrated, not
  under-confident; conversion is a resolution/selection problem (fixed by
  moves-left), not a value-loss-weight problem. Up-weighting value steals trunk
  gradient from the prior/U-dominated policy and, by sharpening, raises
  trap-miss regret in sudden death. cell_q/stvalue are train-only, support-
  invariant, correctly minor.
- **policy_surprise_max_weight=8.0 / uniform_fraction=0.5** — per-row CE reweight
  that concentrates gradient on the tactical rows MCTS corrected; train-time only,
  orthogonal to the lock, well-matched to Hexo.
- **learning_rate=3e-4** — at the climb/steady-state knee for a *fresh* run; the
  plateau micro-sweep ("lower is better") measured steady-state, not fresh climb;
  +/-1 targets make higher lr noisier, so do not raise (and definitely don't go
  below 3e-4).
- **train_samples_per_epoch=48000, batch_rows=32, games_per_epoch=256,
  passes_per_epoch=1** — reuse stays in the healthy 3-6.7x band and *falls* under
  the lock pathology (self-correcting). batch_rows=32 is a floor given Hexo's
  ~14x value-residual variance; only move it up, jointly with lr.
- **weight_decay=1e-4, adaptive_clip cluster** — wd is a negligible tether
  (~0.8% over the run); raising it would compress value amplitude (the wrong
  direction). Adaptive clip auto-adapts to the trunc-train grad mix.
- **replay reuse governor (8.0/500000) and window cluster** — never bind; window
  taper tracks (does not over-weight) fresh long-game share.

---

## Flagged interactions

1. **c_puct x c_scale x root_fpu (the "no extra exploration sources" stack).**
   main_4 zeroes c_scale and root_fpu and ships shaped Dirichlet. Each removes
   U-channel pressure. Keep c_puct=1.5 to offset; do NOT also drop to 1.1 or the
   weak-Q search loses its main conversion driver. Sim confirms 1.5 adds
   conversion with ~no breadth cost.
2. **root_policy_temperature x root_dirichlet x widening cap.** Temp flattens
   *before* noise is mixed; both add opening breadth and both push mass past the
   cap. We cut temp (1.1->1.05), drop early (1.25->1.07), and trim noise fraction
   (0.25->0.15) together — do not cut all three so hard that opening exploration
   collapses; dirichlet_shaped at total_alpha 10.83 still carries ~70-stone
   opening diversity.
3. **temperature_halflife x moves_left_weight x sudden death.** Both target
   conversion/decisiveness. Lower halflife (60->30) makes mid/endgame committal;
   raising ml_weight (0.1->0.2) sharpens the head that drives ml-utility. Tune
   together and attribute carefully — do not also stack a C1 length-decay change
   in the same epoch.
4. **max_game_plies x buffer-window x conversion-failure.** Longer games raise
   rows/epoch and *lower* reuse, so the buffer self-corrects; the cap should bound
   marathons, decisiveness knobs should shorten them. Don't conflate.
5. **soft_policy_weight x policy_surprise x value_weight.** Two softeners on the
   same policy head at 8.0 each = double-flatten; cutting soft to 4.0/^0.5 (or
   omitting it) keeps the surprise reweight as the primary regularizer and avoids
   teaching near-uniform-over-legal targets that feed the opening drift.
6. **expand_backend x truncated training x temperature_halflife.** Lowering
   halflife produces *fewer* truncated games, shrinking the rust value-mask
   contamination — but does not fix it. Must set expand_backend=serial regardless.

---

## Top risks

1. **(If not fixed) expand_backend=rust silently un-masks 32-63% of value rows on
   truncated games** — actively reinforces the defensive lock the run exists to
   break. Highest-stakes item; treat #1 as a launch blocker.
2. **Over-cutting opening exploration.** Stacking temp 1.05 + early-off + noise
   0.15 + forced_k 1.0 simultaneously could under-explore the opening for a fresh
   run. Mitigation: dirichlet_shaped at alpha 10.83 retains broad opening
   diversity; if opening monoculture appears, restore noise to 0.20 before
   touching temperature.
3. **soft_policy head (even at 4.0) teaching near-tie defensive-block targets too
   flat** in sudden death. Mitigation: support-only target + ^0.5; or omit the
   head this run.
4. **Under-powered eval** can't resolve sub-250-Elo deltas (16 games/champion
   edge); a real decisiveness gain from these changes may be invisible or a
   regression rubber-stamped. Mitigation (out of strict knob scope but
   recommended): concentrate the multi-stage budget on the single champion edge
   and/or raise games_budget; rely on the fixed SealBot anchor for fine progress.
5. **moves_left head fails the heal-gate** even at weight 0.2 -> ml-utility stays
   off and the lock persists. Mitigation: escalate to 0.3, not beyond.
6. **Changing many knobs at once** muddies attribution. Suggested ordering if
   staged: (a) expand_backend serial [blocker], (b) temp/halflife/dirichlet
   decisiveness bundle, (c) moves_left_weight, (d) soft_policy, (e) forced_k,
   (f) pcr_fast_visits.

---

## Recommended final hexfield_main_4 knob set (selfplay + training)

```
# --- search / selection ---
search_visits              = 512        # keep
pcr_full_proportion        = 0.33       # keep (0.40 only as throughput-friendly fast-quality buy)
pcr_fast_visits            = 192        # CHANGED 128->192 (throughput-gated; else keep 128)
c_puct                     = 1.5        # KEEP (decision: not 1.1)
c_scale                    = 0.0        # keep
c_base                     = 500.0      # keep
visit_scaled_c_puct        = true       # keep (no-op at c_scale=0)
lcb_z                      = 1.6        # keep
fpu_reduction              = 0.2        # keep interior (mass-scale if available; ~0.12 if must stay flat)
root_fpu_reduction         = 0.0        # keep
root_fpu_zero_under_noise  = false      # keep (set explicitly)
forced_playout_k           = 1.0        # CHANGED 2.0->1.0
widening_policy_mass       = 0.95       # keep (parity)
widening_max_children      = 96         # keep (parity; lazy gate is real control)
widening_min_children      = 2          # keep
lazy_widening              = true       # keep
new_child_fpu              = true       # keep
nucleus_f64                = true       # keep
clean_root_prior_cache     = true       # keep
dirichlet_shaped           = true       # keep
pruned_dynamic_cpuct       = true       # keep
tss_enabled                = true       # keep

# --- root exploration ---
root_dirichlet_total_alpha   = 10.83    # keep
root_dirichlet_noise_fraction= 0.15     # CHANGED 0.25->0.15
root_policy_temperature      = 1.05     # CHANGED 1.1->1.05
root_policy_temperature_early= 1.07     # CHANGED 1.25->1.07 (ramp ~off)
root_policy_temperature_halflife = 19.0 # keep (inert once early==steady)

# --- move-selection temperature ---
temperature                = 1.0        # keep (opening diversity; cool via halflife not base)
temperature_floor          = 0.15       # keep
temperature_halflife_plies = 30.0       # CHANGED 60->30

# --- policy_init (opening probe) ---
policy_init_fraction       = 0.25       # keep
policy_init_avg_plies      = 4.0        # keep
policy_init_max_plies      = 8          # keep
policy_init_temperature    = 1.4        # keep (optional defensive 1.25)

# --- decisiveness / moves-left ---
ml_two_sided               = false      # keep
ml_final_pick_band         = 0.08       # keep
moves_left_weight          = 0.2        # CHANGED 0.1->0.2 (escalate to 0.3 if head_audit fails)

# --- length / cap ---
max_game_plies             = 256        # keep

# --- throughput cluster ---
active_games               = 96         # keep (<games_per_epoch)
active_root_limit          = 192        # keep
virtual_batch_size         = 4          # keep
flush_target               = 1024       # keep
# cache_max_states          = 262144    # keep

# --- training loss weights ---
policy_weight              = 1.0        # keep
value_weight               = 1.0        # keep
opp_policy_weight          = 0.25       # keep
short_term_value_weight    = 0.1        # keep
q_head_weight              = 0.1        # keep
soft_policy_weight         = 4.0        # CHANGED 8.0->4.0 (+ power ^0.5, support-only); or omit head
policy_surprise_max_weight = 8.0        # keep
policy_surprise_uniform_fraction = 0.5  # keep

# --- optimizer / cadence ---
learning_rate              = 3e-4       # keep
weight_decay               = 1e-4       # keep
batch_rows                 = 32         # keep
train_samples_per_epoch    = 48000      # keep
games_per_epoch            = 256        # keep
passes_per_epoch           = 1          # keep
adaptive_clip              = true       # keep (clip_c 1.75 / ema 0.99 / warmup 50 / grad_clip 5.0)
max_train_bucket_per_new_data = 8.0     # keep
max_train_bucket_size      = 500000     # keep
# shuffle_keep_target_rows=300000, exponent=0.65, scale=20000, min_rows=20000, expand_per_row=0.4  # keep

# --- INFRA (correctness) ---
expand_backend             = "serial"   # CHANGED "rust"->"serial" until replay_expand.rs projects value_mask
```
