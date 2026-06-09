# Playout Cap Randomization (PCR) — KataGo-faithful mapping for hexgt

_Owner directive (2026-06-04): "up the sims to 1024 and implement PCR exactly how
katago does it... let's do 50% PCR." This doc pins down what KataGo actually does,
then maps each piece onto our pipeline, recording every choice + justified deviation
BEFORE the code. It is the spec the implementation in `selfplay.py` follows._

## 1. What KataGo actually does (Wu 2020, "Accelerating Self-Play Learning in Go", §PCR + codebase semantics)

KataGo's Playout Cap Randomization attacks a tension in AlphaZero self-play: the
**value** target wants many cheap games (play lots, learn the outcome z), while the
**policy** target wants a few deep searches (a sharp visit distribution). PCR
decouples them per move:

1. **Per MOVE** (independently, not per game), with probability **p** do a **FULL**
   search to a large visit cap **N**. With probability **1 − p** do a **FAST**
   search to a small cap **n**.
2. **Only FULL-search positions are recorded as training rows** (both the policy
   target = the full visit distribution, AND that row's value target). FAST-search
   positions are played but produce **no training row at all**. The game outcome z
   still flows into the recorded full rows.
3. **The exploration machinery runs on FULL searches ONLY**: Dirichlet root noise,
   forced playouts + policy-target pruning, and temperature-based move selection.
   **FAST searches run clean** — no root noise, no forced playouts, and are played
   **greedily** (sharper/no added exploration).
4. KataGo's published main-run numbers: **N = 600, n = 100, p = 0.25** (ratio
   n/N = 1/6). Tree reuse across moves is standard and unchanged by PCR.

Net effect: ~p of positions get an expensive, high-quality policy target; every
position (full + fast) advances the game so the value/outcome signal is dense and
cheap; average cost/move = p·N + (1−p)·n, far below N·(every-move).

## 2. Our pipeline, per move (what exists today)

`selfplay.py::run_selfplay_games` runs a game-driven loop: keep `active_games` in
flight; each round, search **one move for every playable game** in a single batched
`HexgtMctsSession.run` call (all games' leaves coalesced into shared GPU forwards),
then apply the selected move to each. Every searched position today is:
- searched at `selfplay.search_visits` (currently 512),
- recorded via `sample_from_state` → appended to the game's `pending` list as
  `(player, sample, root_value)`,
- at game end, `finalize_game_samples(pending, …)` assigns value / opp-policy /
  short-term-value (STV) targets and writes a compact `.npz` shard (one per game).

The native `HexgtMctsSession.run` already takes **per-call** `visits`,
`root_dirichlet_total_alpha` / `…_noise_fraction` (None ⇒ no noise),
`forced_playout_k` (0 ⇒ no forced playouts / no policy-target pruning of the exported
target), and **per-root `move_temperatures`** (0.0 ⇒ greedy argmax). The tactical
machinery — Phase-4 tactical injection at expansion, Phase-5 hitting-set leaf-value
override, and the 1-ply tactical move guard on the played/exported move — is
structurally part of expansion/selection/result in `mcts.rs` + `mcts_tree.rs` and is
**independent of noise/forced-playouts/temperature**. So it already applies to any
search regardless of caps.

## 3. The mapping (decisions, with justification)

| KataGo piece | Our implementation |
|---|---|
| Per-move full/fast coin, prob p | Per `(game, move)` deterministic coin `_pcr_is_full(base_seed, epoch, game_key, move_index, p)` (splitmix64 hash → [0,1) < p). Reproducible + decorrelated across games/moves/epochs. |
| Full cap N | `selfplay.search_visits` = **1024** (the owner's "up the sims to 1024"). No separate knob — full search == the configured self-play visits. |
| Fast cap n | `selfplay.pcr_fast_visits` = **170** (see §3.1). |
| Proportion p | `selfplay.pcr_full_proportion` = **0.50** (owner's "50% PCR"; KataGo used 0.25 — see §3.2). |
| Full = noise + forced playouts + temperature schedule | Full subset → `run(..., visits=1024, root_dirichlet_*=configured, forced_playout_k=configured(2.0), move_temperatures=schedule(ply))`. Identical to today's single call. |
| Fast = clean + greedy | Fast subset → `run(..., visits=170, root_dirichlet_*=None, forced_playout_k=0.0, move_temperatures=0.0)`. No noise, no forced playouts, greedy argmax. |
| Record full only | Each searched position tagged `pcr_full` in (in-memory) metadata; at write time keep only `pcr_full` rows. Fast rows are dropped (never written), so policy AND value rows halve, exactly as KataGo records only full searches. |
| Tactical safety on both | Unchanged — injection / hitting-set override / move guard apply to both subsets automatically (see §3.4). |

Implementation: split each round's `playable` games into `full_games` / `fast_games`
by the coin, then issue **two** batched `run` calls (one per subset) with the params
above. Each subset still coalesces all its games' leaves into shared forwards, so the
throughput property is preserved (two good-sized batches per round instead of one).
Subtree reuse is per-game-key and unaffected; `set_additional_visits` adds the call's
cap on top of the reused tree, so a game alternating full/fast caps across moves is
handled correctly.

### 3.1 Why n = 170

KataGo's n/N ratio is exactly **100/600 = 1/6**. Applied to N = 1024 that is
1024/6 ≈ 170.7 → **n = 170**. This is the faithful ratio, not an ad-hoc pick. It is
exposed as `--pcr-fast-visits` so the owner can retune. (Owner suggested 150–250 and
used n≈200 in the throughput sketch; 170 is the exact-ratio choice within that band.)

### 3.2 Deviation: p = 0.50 (KataGo used 0.25)

Owner's explicit choice ("let's do 50% PCR"). Consequence vs KataGo's 0.25: **twice**
the recorded-row fraction (50% vs 25% of moves), stronger average play (half the moves
are full 1024-visit searches), at higher compute/move. This is a deliberate, owner-set
deviation, documented as such.

### 3.3 STV / opp-policy chain — the critical interaction (Option A: dense chain)

Our value head at λ=0 (current) uses **value = z** (pure hard outcome), so the value
target is independent of which positions are recorded — fully KataGo-faithful. But two
hexgt-specific **aux** heads read the *future trajectory* in `pending`:
- **STV** (short-term-value, horizons 4/12/24 @ aux weight 0.10): EMA of future
  per-move `root_value`.
- **opp-policy** (aux weight 0.25): the next opponent decision's visit distribution.

If `pending` held only full moves, those chains would skip ~50% of plies, **doubling
the effective horizon** (4/12/24 measured in recorded moves, not game plies) and
mis-aligning the opp-policy "next move."

**Decision — Option A (dense chain, write full only):** append **every** searched
position (full AND fast) to `pending`, so STV/opp-policy are computed over the dense
per-ply trajectory; then **filter to `pcr_full` rows at write time**. Justification:
STV/opp are EMA/lookahead quantities whose horizons are defined in **game plies**;
keeping the dense trajectory preserves that semantics exactly, and the EMA smoothing
plus modest aux weights (0.10 / 0.25) absorb the extra noise of fast-search
`root_value`s far better than the alternative of silently doubling the horizon. This
is the owner's leaned option ("keep recording root_value for all moves for the STV
chain since it's an EMA"). Cost: `sample_from_state` is built for fast moves too (then
dropped) — negligible (facts-building is ~1% of self-play wall; the bottleneck is the
NN forward). `pending` size is therefore **unchanged** vs today (all moves), so
`finalize_game_samples` cost is unchanged; only a write-time filter is added.

### 3.4 Tactical machinery — applies to BOTH search types (verified)

- **Phase-4 tactical injection** (`mcts_tree.rs`: `add_node_from_eval` /
  `RustSearch::new` via `threats::tactical_cells`) is part of node expansion → both.
- **Phase-5 hitting-set leaf override** (`mcts.rs::select_leaf_batch` via
  `threats::analyze(..).verdict()`) is part of leaf selection → both.
- **1-ply tactical move guard** (`tactical_guard_weights`, applied in both
  `select_search_action` and the result payload) masks proven-losing / forces
  proven-winning moves on the *played* move → both, **including greedy fast moves**.

So safety does not regress on fast moves. No Rust change is required; PCR is
implemented entirely in Python (lowest risk — no native rebuild, cannot disturb the
running process's loaded `.so`).

### 3.5 policy-surprise interaction

`materialize_policy_surprise_rows` (KataGo frequency weighting via row duplication on
KL(visits‖prior)) already operates on `to_write` — i.e. **recorded rows only**. After
the PCR filter, `to_write` = full rows, whose visit distribution is the high-quality
1024-visit search. So policy-surprise weights exactly the right rows; no change.

## 4. Throughput + buffer math (to verify live)

- **Cost/move**: 0.5·1024 + 0.5·170 = 512 + 85 = **597 visits/move** avg, vs current
  512 → **+16.6%** search work/move. Expect self-play pos/s (moves/s) to drop ~10–15%
  (NN-forward-bound, visit-count-proportional), minus some batching efficiency from two
  forward streams/round. **Measure actual pos/s after engagement and report the delta.**
- **Recorded rows/epoch**: games still play to terminal, so total moves played/epoch is
  ~unchanged, but only ~50% are recorded → **raw_samples roughly HALVES** (~15–20k →
  ~8–10k). The trade: each recorded row is a 1024-visit target (vs 512 today).
- **Buffer/window**: replay-pool-cap 500k with recency decay 0.9 and a growing window.
  Fewer fresh rows/epoch ⇒ the window reaches **further back** in epochs to fill the
  cap, and the train pass (512 steps × batch 64 = **32,768 samples/pass**) draws from a
  pool where each fresh full row is over-sampled ~2× more than before (32,768 / ~9k vs
  32,768 / ~17k). Net: recent high-quality rows get more reuse; pool position-diversity
  at a given epoch count ~halves but each position carries a sharper target.
  **Recommendation:** do **not** change `games_per_epoch` initially — observe 2–3 epochs
  and the epoch-33/36 SealBot evals first; if train policy-loss drops while SealBot
  strength stalls (over-reuse/overfitting), raise `games_per_epoch` (256→~400–512) to
  restore fresh-row volume, or lower `train-steps-per-epoch`. Reported, owner decides.

## 5. Watch items after engagement

- Q2 opening diversity (`uniq_open`, `m2H`): greedy fast moves could sharpen play;
  full-move noise + temperature should keep the opening diverse. Watch for collapse.
- pos/s delta (above), GPU health, VRAM envelope.
- raw_samples ≈ half; pool fill / window span in the train log.
- SealBot evals at epochs 33/36 are the **strength ground truth** (λ=0, policy-surprise,
  temperature curve all still active and reported in the startup log).
