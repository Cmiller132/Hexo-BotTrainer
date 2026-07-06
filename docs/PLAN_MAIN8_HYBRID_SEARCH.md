# PLAN: hexfield_main_8 — hybrid search (PUCT Full / Gumbel Fast)

Status: DESIGN SPEC (2026-07-06). Not yet implemented.
Owner decision points are marked **[DECIDE]**; everything else is a recommendation
with rationale.

## 0. One-paragraph summary

main_8 keeps the main_7 architecture, training stack, and PCR budget split, and
swaps the search regime **per move class**: Full moves (33% of plies, 1024
visits, the only rows that export policy targets) run **classic PUCT with the
full KataGo exploration kit** — root Dirichlet noise, forced playouts with
target pruning, visit-scaled c_puct, progressive (lazy) widening, LCB play
selection; Fast moves (67% of plies, 192 visits, value-only rows) run the
**Gumbel root + sequential halving**, which is the provably-good regime at tiny
budgets and replaces the pure-greedy LCB pick with bounded, principled
stochasticity. This inverts today's arrangement (search.rs gates Gumbel to
`MoveClass::Full`; Fast already falls through to a — currently neutered — PUCT
root), so the core change is flipping one gate and giving each class its own
`Divergences`, not building a new search.

## 1. Why change, and why this hybrid

### What main_7's all-Gumbel regime taught us

1. **The π' export target is structurally too sharp.** At 1024 visits the
   σ-gain is ~278× (ΔQ = 0.01 → 16× probability ratio); published Gumbel-AZ
   practice is ~80×. We had to bolt on an export-only hack
   (`gumbel_target_c_scale = 0.35` → ~97×) to break the ep30–35 plateau. It
   worked (+52.5 ± 48 vs ep30 at ep40), but it is a hand-tuned constant with
   no principled basis, and it decouples the trained target from the search
   that produced it. PUCT's pruned-visit-count target has *natural* sharpness
   scaling and no such constant.
2. **Root coverage is capped by m.** Gumbel top-m=32 sees 10–20% of legal
   moves mid-game at radius 4. Draw temperature τ=1.5 (widen the candidate
   draw) measured **inert** (ep50 vs ep40 = −3.5 ± 39). Raising m needs visits
   ∝ m (m=64 wants ≥1536 visits) — unaffordable. PUCT + shaped Dirichlet noise
   + forced playouts explores the *whole* legal set with no candidate wall.
3. **Sequential halving imposes a play-selection ceiling.** SH forces finalist
   visit ratios (228:196 at 1024/m=32), which caps the winner-pick rate at
   ~0.73 forever — a permanent, non-annealing noise floor on Full-move play.
4. **Fast moves are pure exploitation.** 67% of plies are strict T=0 LCB
   picks. The one attempt to soften that (`pcr_fast_temperature = 0.1`) was
   convicted of poisoning value labels via P0 outcome-skew (+7pp P0 share,
   −45 Elo). Gumbel at the Fast root adds *symmetric, regret-bounded*
   stochasticity through the candidate draw instead of through outcome-flipping
   temperature — main_7 ran exactly this noise source on Full moves for 78
   epochs with P0 share healthy (0.48–0.51 post-fix), so the mechanism is
   field-tested at scale.

### Why not "just switch back to PUCT" everywhere

Gumbel genuinely is the stronger algorithm at low simulation counts (its
regret guarantee is the whole point of the paper) — 192 visits over ~600 legal
moves is squarely that regime. Throwing it away on Fast moves buys nothing.
The hybrid puts each algorithm where its assumptions hold:

| | Full (1024v, exports π target) | Fast (192v, value-only) |
|---|---|---|
| Needs | broad exploration, unbiased trainable target | cheap good move, unbiased value trajectory |
| Best fit | PUCT + noise + forced playouts + pruned target | Gumbel top-m + SH (low-budget optimal) |

### Honest accounting: the main_5/main_6 precedent

The house verdict of 2026-07-04 was "fix Gumbel, don't switch" partly because
the PUCT era plateaued while the Gumbel era climbed. Two things blunt that
precedent:

- **We never ran full-kit PUCT on the modern stack.** main_5's PUCT was
  neutered: `c_scale = 0.0` (visit-scaled c_puct compiled in but *disabled* —
  static c_puct 1.5 the whole run), `forced_playout_k = 1.0` (KataGo uses 2.0),
  flat 45-ply temperature halflife, c=128-era net, pre-telemetry, pre-prefit
  pipeline, and PUCT on *all* moves including the 67% Fast plies.
- The Gumbel era's climb came only after the export-target hack; the stock
  algorithm plateaued too. Neither algorithm's stock config survived contact —
  the question is which *fixed* variant has more headroom, and that's an
  empirical question this run answers with kill-gates (§7).

## 2. Search design

### 2.1 Full moves — PUCT, full kit

At 1024 visits, root and interior selection are classic PUCT. All levers below
already exist in `tree.rs`/`search.rs`; nothing new to build for this half.

| Lever | main_5 value | **main_8 value** | Rationale |
|---|---|---|---|
| `c_puct` | 1.5 | **1.5** | House-calibrated base. |
| `visit_scaled_c_puct` / `c_scale` / `c_base` | true / **0.0 (off!)** / 500 | true / **0.45** / 500 | KataGo-style log growth of exploration with root visits. main_5 shipped this dead; the 0.45 default was never actually exercised. This is the single biggest "PUCT we never ran" lever. |
| `root_dirichlet_noise_fraction` | 0.25 | **0.25** | Root exploration restored (main_7: 0.0). |
| `root_dirichlet_total_alpha` / `dirichlet_shaped` | 10.83 / true | **10.83 / true** | Shaped alpha proportional to clean policy — proven house config. |
| `forced_playout_k` | 1.0 | **2.0** | KataGo paper value; forces `sqrt(k·P(a)·N)` visits per noised child so noise actually reaches the tree. |
| `pruned_dynamic_cpuct` | true | **true** | Target pruning subtracts forced visits using the dynamic c_for — the exported target is de-noised. |
| `lazy_widening` / `new_child_fpu` | false / true | **true / true** | This is the codebase's progressive widening: children materialize one at a time when the new-child score `(FPU + U)` beats every expanded edge — expansion driven by evidence, not a frozen nucleus count. `new_child_fpu` is documented as its intended partner (tree.rs). |
| `widening_policy_mass` / max / min | 0.95 / 96 / 2 | **0.95 / 96 / 2** | Nucleus cap unchanged; lazy widening changes *when* children open, not *which*. |
| `fpu_reduction` / `root_fpu_reduction` | 0.2 / 0.2 | **0.2 / 0.2** | Unchanged. `root_fpu_zero_under_noise` stays false (house precedent); flag it for the ep20–30 recalibration if noised-child uptake looks weak in telemetry. |
| play selection | T-schedule → LCB | **T-schedule → LCB** (`lcb_z = 1.6`) | Keep main_7's schedule: T=1.0, floor 0.15, halflife 10 plies (main_5's 45-ply halflife predates the current data mix). No SH ceiling anymore: post-floor play is a real LCB argmax, not a 73/27 coin. |
| exported target | — | **pruned visit counts** (`policy_target = "visit"`) | Trainer default; `batching.py` falls back to the visit `policy` row automatically. Surprise-reweight KL is now computed on the same histogram that is trained — closing the known mismatch for free. |

`root_policy_temperature` (1.1 / early 1.15 / halflife 19) becomes live again
under a PUCT root (in main_7 it only fed the surprise baseline) — keep main_5
values.

### 2.2 Fast moves — Gumbel root + SH at 192 visits

| Lever | **main_8 value** | Rationale |
|---|---|---|
| `gumbel_root_enabled` (Fast class only) | **true** | The gate inversion (§3). |
| `gumbel_sequential_halving` | **true** | SH at 192 visits; `init_gumbel_root` already auto-calibrates m down to the budget (tested: `init_gumbel_root_calibrates_m_to_budget`), so `gumbel_m = 32` acts as a cap and the effective m at 192v will be small (~8–16). |
| `gumbel_c_visit` / `gumbel_c_scale` | **50.0 / 1.0** | σ mult at Fast budget ≈ (50+96)·1.0 ≈ 146·q — sane. These constants only rank SH candidates now; they no longer touch any trained target. |
| `gumbel_draw_temperature` | **1.0** | τ=1.5 measured inert at full budget; start stock, one fewer delta. |
| `gumbel_nonroot_select` | **true (Fast searches only)** | Interior selection consistent with the root regime per class (§3). |
| `gumbel_play_prune` | **true** | Keep the serve-cost win. |
| play selection | **SH winner (T=0)** | `pcr_fast_temperature = 0.0` stays locked (P0-skew conviction, 2026-07-04). The Gumbel draw supplies the trajectory diversity that T=0.1 was reaching for, without touching outcome labels asymmetrically. |
| `gumbel_target_enabled` / `gumbel_target_c_scale` / `gumbel_target_min_visits` | **removed** | Fast rows are value-only — there is no π' export anywhere in main_8. The 0.35 hack and its keys die with it. |

### 2.3 Eval / arena profile

Eval and multistage matches (`eval_visits = 128`, `full_search_visits = 512`)
run the **Full/PUCT profile without root noise** (noise is a selfplay-only
input via `noise_for`; eval paths already pass none) with LCB +
`ml_final_pick_band` re-engaged. Per-checkpoint search profiles in the
showcase/dashboard need a main_8 entry (PUCT profile — same shape as the
legacy main4/main5 entries that already exist there).

## 3. Implementation: per-class divergences

Today one `Divergences` struct is built from config and applied to every
search; the *only* class-conditional bit is the hardcoded gate
(`search.rs:1044`):

```rust
if divergences.gumbel_root && matches!(move_class, MoveClass::Full) { init_gumbel_root(...) } else { clear_gumbel_root() }
```

### Required change (Rust, ~1–2 days incl. tests)

1. **Two divergence views.** Python builds `full` and `fast` override maps
   (`build_divergence_overrides` grows a per-class layer); Rust resolves them
   into `divergences_full` / `divergences_fast` and the driver applies
   `divergences_for(move_class)` at the same sites that already call
   `visits_for` / `forced_k_for` / `noise_for` / `root_fpu_for` (the
   `ContinuousMovePolicy` per-class dispatch pattern is established — this
   extends it, it doesn't invent it).
2. **Replace the hardcoded class gate** with `divergences_for(class).gumbel_root`
   — config now expresses Full→PUCT, Fast→Gumbel (or any other assignment,
   which also gives us a free A/B lever).
3. **`KNOWN_DIVERGENCE_KEYS`**: add every new key *in the same commit* as the
   parser change. (The 2026-07-04 outage: parser accepted keys the whitelist
   rejected → 3 crashes → breaker HALT. There is a pyo3 regression test for
   this now — extend it.)
4. **Session/eval path** (`search.rs:688`, non-continuous): takes the Full
   profile; no per-ply classes there.

### Interactions audited (design-time)

- **Tree reuse across classes.** A game's tree alternates PUCT-rooted and
  Gumbel-rooted moves. This is *already the live behavior in main_7* (Fast
  reuse calls `clear_gumbel_root()` and runs the PUCT root; Full reuse rebuilds
  the candidate set) — main_8 inverts which class does which. Node stats
  (visits, value sums, stored logits) are shared and regime-agnostic.
- **Interior-rule flip on shared subtrees.** `gumbel_nonroot_select` becoming
  per-class means the same interior node can be selected by the Gumbel rule on
  ply k and PUCT on ply k+1. Both rules are stateless functions of node stats,
  so this is sound — but it needs a regression test (reuse a tree across a
  class boundary, assert no panic / sane visit distribution).
- **Logit availability.** `request_logits` is derived from the gumbel flags
  (`search.rs:660`); with per-class divergences it must OR across both views.
  Full searches keep `export_root_prior_logits = true` (surprise baseline +
  dashboard).
- **Widening/FPU divergence fields** consumed at interior nodes must come from
  the *owning search's* current class view — set alongside
  `set_divergences(...)` per move, same as today.

### Test plan

- Golden bit-equivalence: main_8 config with `fast = gumbel off` ≡ pure-PUCT
  run; with `full = gumbel on, fast = gumbel on` ≡ main_7 behavior (both
  classes reproduce existing goldens).
- Cross-class reuse test (above).
- Whitelist regression test extended with the per-class keys.
- Full-move export: pruned-visit target excludes forced playouts
  (`pruned_dynamic_cpuct` path) — assert against a hand-built tree.
- Fast rows remain value-only, `pcr_full=false` in payload, no gumbel π'
  fields set.

## 4. What carries over unchanged

- **Architecture**: c=192, 3×64 heads, `CCACCACCACCACCA` (8.13M) — deliberately
  unchanged so the run isolates ONE variable (search regime). Same
  `HEXFIELD_*` env in a new `hexfield-supervisor-8.service`; same Triton
  kernels (`TRITON_ATTN`, `TRITON_CONV_LN`).
- **PCR budget**: 1024 / 0.33 / 192 at launch. The tier-2 case (0.45/768) was
  data-density-driven and its binding constraint (SH cliff below ~640 visits)
  was *Gumbel-specific* — under PUCT-Full the visit floor is softer, so the
  ep20–30 budget calibration should re-run the sweep including 0.45/768 and
  0.40/896.
- **Training config**: batch 256, LR 5e-4 launch (house fresh-run precedent;
  decay to 4e-4 at the mid-run plateau per the main_6 playbook), all aux-head
  weights, surprise reweighting, shuffle/window params, reuse cap 8.0 — all
  main_7 values. Only `policy_target` changes: `"gumbel"` → `"visit"`.
- **Warm start**: BC prefit, main_7 pattern (`scripts/_main7_prefit_*.sh` →
  `_main8_prefit_*.sh`). **Data hygiene: source prefit samples from clean
  main_7 epochs only — ep55–67 and post-recovery (ep79+); EXCLUDE the seeded
  era ep69–78** (those rows carry the adversarial-seed contamination that cost
  −70 Elo). Prefit on gumbel-π'-era rows is fine for initialization (BC target
  soft-ness doesn't constrain the RL phase).
- **Multistage eval + SealBot** unchanged; anchors:
  `main5_ep105`, `main6_ep73`, **`main7_best` (best clean checkpoint at launch
  time — ep67 today, later if the post-seeding grind passes it)**, `ep5`,
  `ep30` (self).

## 5. Other improvements bundled into main_8 (ranked)

1. **Fix the train-bucket governor rollback freeze before launch**
   (chip `task_8351add4`). It has bitten twice (ep36/37 silent no-train after
   the plateau rollback; blocks the pending ep67 rollback decision on main_7).
   Any long run WILL roll back at some point; the governor must key on a
   watermark that survives quarantines. This is a pre-launch requirement, not
   a nice-to-have.
2. **Powered paired gates as the primary strength read, on a fixed cadence.**
   The multistage tripwire misled us three times (ep35 "−188" was 4×
   exaggerated; ep55 "+191" was ~7× exaggerated; seeding damage invisible in
   losses). Bake the scratchpad `h2h_match.py` flow in as
   `scripts/_h2h_gate.sh`: every 10 epochs, 200 paired games, epN vs epN−10 at
   the eval profile, result appended to `diagnostics/h2h_gates.jsonl` and
   surfaced on the dashboard. Multistage verdicts get demoted to tripwire-only
   (they trigger a powered match; they never decide anything).
3. **P0-share tripwire.** The `pcr_fast_temperature` skew took 13 epochs to
   notice. Supervisor warning when selfplay P0 win share leaves [0.44, 0.56]
   for 3 consecutive epochs. One-liner against existing telemetry.
4. **fp8 serve A/B during the prefit window** (GPU is otherwise busy only with
   BC). ~+15% serve throughput if the 4.5e-2 value deviation costs <5 Elo at
   the arena; the prefit window is the free slot to measure it.
5. **`policy_init_fraction` 0.25 → 0.35.** Standing recommendation from the
   explore/exploit review; opening diversity channel independent of the search
   regime; cheap and reversible.
6. **Resume-path segment diagnostics** (generalization of chip
   `task_ef8b8fcd`): the blunder-seed resume no-op exposed that mid-epoch
   resume segments silently drop feature paths and telemetry keys. Audit the
   resume path for class-conditional features (per-class divergences must
   apply on resume segments too — add a test).
7. **Config hygiene**: delete now-dead keys from the main_8 TOML —
   `gumbel_target_*`, `gumbel_draw_temperature`, `pcr_fast_temperature`
   (locked 0.0 = default), `blunder_seed_*`. Keep the file describing only
   live levers (main_7's lesson: dead levers with non-default config defaults
   caused two near-misses).

### Considered and deferred (with pointers)

- **MCGS / graph search** — analyzed 2026-07-02, rejected; revisit only after
  measuring duplicate-state fraction.
- **Blunder seeding v2** — only behind an offline A/B, with seeds selected by
  outcome-vs-deep-search disagreement (not self-surprise-max) and
  down-weighted value labels on seeded rows.
- **m=64 / visit scaling** — data-limited regime; rejected 2026-07-05.
- **fp8 training** — separate risk class from fp8 serve; not this run.

## 6. Config sketch (selfplay section)

```toml
[model.config.selfplay]
search_visits = 1024
pcr_full_proportion = 0.33
pcr_fast_visits = 192
active_games = 192
virtual_batch_size = 48
flush_target = 1024
active_root_limit = 192

# ---- Full-move PUCT (exports policy targets) ----
c_puct = 1.5
visit_scaled_c_puct = true
c_scale = 0.45                      # LIVE for the first time (main_5 shipped 0.0)
c_base = 500.0
root_dirichlet_noise_fraction = 0.25
root_dirichlet_total_alpha = 10.83
dirichlet_shaped = true
forced_playout_k = 2.0              # main_5 was 1.0; KataGo value
pruned_dynamic_cpuct = true
fpu_reduction = 0.2
root_fpu_reduction = 0.2
lazy_widening = true                # progressive widening: evidence-driven child materialization
new_child_fpu = true
widening_policy_mass = 0.95
widening_max_children = 96
widening_min_children = 2
lcb_z = 1.6
nucleus_f64 = true
clean_root_prior_cache = true
temperature = 1.0
temperature_floor = 0.15
temperature_halflife_plies = 10.0
root_policy_temperature = 1.1
root_policy_temperature_early = 1.15
root_policy_temperature_halflife = 19.0
policy_init_fraction = 0.35         # up from 0.25 (explore/exploit rec #2)
policy_init_avg_plies = 4.0
policy_init_max_plies = 8
policy_init_temperature = 1.4
ml_two_sided = false
ml_final_pick_band = 0.08
export_root_prior_logits = true

# ---- Fast-move Gumbel (value-only rows; new per-class syntax, name TBD in impl) ----
fast_gumbel_root_enabled = true
fast_gumbel_sequential_halving = true
fast_gumbel_nonroot_select = true
fast_gumbel_c_visit = 50.0
fast_gumbel_c_scale = 1.0
fast_gumbel_m = 32                  # cap; auto-calibrates to ~8-16 at 192 visits
fast_gumbel_play_prune = true

max_game_plies = 256
tss_enabled = true
cache_max_states = 262144
```

```toml
[model.config.training]
# identical to main_7 EXCEPT:
policy_target = "visit"             # pruned visit counts; "gumbel" retired
```

## 7. Rollout plan and kill-gates

1. **Build + test** the per-class divergence refactor on a branch; golden
   parity suites green (§3).
2. **Prefit**: BC on clean main_7 samples (§4); launch-gate parity dvalue
   ≤ 3e-3 (house gate). fp8 A/B runs in this window (§5.4).
3. **Bring-up epoch 1–3**: watch pos/s (PUCT-Full explores wider than m=32 —
   expect a modest serve-throughput drop from lower cache locality; if >20%
   pos/s regression, profile before touching search params), P0 share,
   entropy by phase, forced-playout uptake (new telemetry counter worth
   adding: mean forced visits/root).
4. **ep10 gate**: powered 200-game match, ep10 vs prefit-ckpt. Want clearly
   positive (main_7 equivalent stretch was strongly positive).
5. **ep30 gate**: ep30 vs ep20 slope ≥ ~+2 Elo/epoch (main_7's post-softening
   grind was 2.4/epoch avg) AND cross-run read: main_8 ep30 vs main_7 ep67
   anchor for absolute position.
6. **Kill/pivot criterion [DECIDE the threshold]**: if by ep40 the powered
   slope is < +1 Elo/epoch over two consecutive gates, or main_8 tracks
   > 100 Elo behind main_7's same-epoch trajectory, stop and decide: retune
   PUCT-Full (first knobs: `c_scale`, noise fraction, `forced_playout_k`) or
   fall back to the main_7 regime — the per-class gate makes the fallback a
   config change, not a rebuild.
7. Deploy discipline (standing lessons): stop supervisor → rebuild
   (`scripts/_rebuild_hexfield.sh`) → test → commit → start; `git -C` the main
   tree for verification; remember `supervisor_halted.flag` on failed starts.

## 8. Risks

| Risk | Exposure | Mitigation |
|---|---|---|
| PUCT-Full underperforms fixed-Gumbel (precedent risk) | The main verdict of 2026-07-04 favored fixing Gumbel | Kill-gates at ep10/30/40; fallback is config-only (§7.6); the c_scale=0 discovery materially weakens the precedent |
| Visit-count targets noisier than π' at 1024v | Policy CE plateaus earlier | Forced-playout pruning de-noises; KataGo trains at comparable budgets; surprise-KL is now measured on the true target so we'll see it |
| Cross-class tree interactions | Panic or subtle stat corruption | Reuse regression tests (§3); the inverse arrangement has 78 epochs of production soak |
| Throughput drop (wider Full roots) | Lower pos/s, slower wall-clock | Measure at bring-up; `widening_max_children=96` caps fan-out; fp8 serve (if A/B passes) buys back ~15% |
| Per-class refactor bugs on resume path | Silent no-op like the seeding bug | Resume-segment test explicitly asserts per-class divergences apply (§5.6) |
| Governor freeze on any future rollback | Repeat of ep36/37 | Pre-launch fix is requirement #1 (§5.1) |

## 9. Decisions (resolved 2026-07-06)

1. **main_7: PARKED at its best clean checkpoint.** Supervisor stopped
   2026-07-06 (grinded to ep91 before stop). It does not resume — GPU is fully
   free for the main_8 prefit + fp8 A/B. main_7's run ends here.
2. **Rollback: GRIND-FORWARD wins — use ep78+.** No rollback, no governor
   surgery for main_7. ep79–91 are all post-seeding-recovery clean. The
   `main7_best` anchor = the strongest post-recovery checkpoint (working
   default ep91; confirmed by the ep91-vs-ep67 match run 2026-07-06). Prefit
   data window = clean ep55–67 + ep79–91, EXCLUDING seeded ep69–78.
3. **Kill-gate: LOOSER / more patience — no pivot before ep50.** PUCT-Full's
   target dynamics differ from Gumbel; give it runway. Replace §7.6: pivot only
   if by **ep50** the powered slope is < +1 Elo/epoch over two consecutive
   gates OR main_8 tracks > 100 Elo behind main_7's same-epoch trajectory. No
   early ep20 hard gate. (The ep10/ep30 gates in §7 remain *observational* —
   they inform, they don't trigger a pivot.)
4. **Build: refactor approved pending user's understanding of the change**
   (this section's breakdown was requested; §3 is the implementation).
