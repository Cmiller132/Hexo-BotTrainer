# HEXGT exploration diagnostics — is the self-play actually exploring?

**Question (user):** the model "plays sensibly" and doesn't visibly try new moves —
could a sharp BC prior + under-powered Dirichlet/temperature mean self-play just
REPLAYS the BC policy (no improvement signal), explaining the 45→40→25 regression?

**Verdict: NO. Exploration is sufficient-to-high on every measure. Under-exploration
is NOT the cause of the regression.** The BC prior is *diffuse* (not sharp), the
Dirichlet noise meaningfully reshapes the search, temperature keeps selection broad,
and the generated games are diverse. The likely regression drivers are elsewhere
(value-target recalibration / lr / data volume — the latter now being addressed by
the 256-game/65536-sample richer run), not a lack of exploration.

Method: read-only on `runs/hexgt_rl_main` self-play (150 games / ~18k positions +
9 example-trace files), plus a CPU Dirichlet decomposition on the BC seed (45
positions). Same C1 exploration config as the live run. By phase: opening<15 /
mid 15–60 / end≥60 plies.

## 1. Concentration (mean) — visits are SPREAD; the prior is even more diffuse

| phase | legal | VISIT top1 / top5 / effN | SELECT top1 / effN | PRIOR effN / H |
|---|---|---|---|---|
| opening | 303 | 0.43 / 0.71 / **10.3** | 0.47 / **8.8** | **15.7** / 2.75 |
| mid | 477 | 0.39 / 0.66 / **12.7** | 0.73 / 3.4 | **21.6** / 3.07 |
| end | 713 | 0.36 / 0.64 / **13.5** | 0.77 / 2.5 | **30.1** / 3.41 |

The post-search VISIT distribution (the policy target) is spread over **10–13
effective moves** — the top move only gets 36–43%. The PRIOR is *more* diffuse
still (effN 16–30), so MCTS *sharpens* a diffuse prior rather than a sharp one
collapsing. This is the opposite of the "sharp prior" worry.

## 2. Realized exploration — P(played move ≠ visit-argmax)

| phase | expected (from selection dist) | ACTUAL (reconstructed played moves) |
|---|---|---|
| opening | 0.53 | **0.58** |
| mid | 0.27 | **0.36** |
| end | 0.23 | **0.35** |

**35–58% of played moves are NON-argmax** — sampling genuinely picks a non-top
move a third to over half the time. This is literal exploration, and it is far
from zero outside the opening. Temperature IS buying exploration.

## 3. Diversity across games

- **First move:** 1 unique / 150 games — but Hexo's first move is *forced* (1
  candidate), so this is expected, not a collapse.
- **Opening 6-ply lines: 118 unique / 150 games (79% unique)**, top line repeats
  only 5×. Openings branch widely.
- **Transpositions: 5.5%** of positions are revisits (17.5k unique of 18.3k).
  Games explore distinct positions; no line-collapse.

## 4. Dirichlet decomposition — the noise is STRONG, not weak

CPU forward of the BC seed → raw prior, then mixed with the exact MCTS noise
(α_i = 6.6/cands, eps=0.25), averaged over 32 draws:

| phase | cands | α_i | prior effN → noised effN | **KL(noised‖prior)** | **P(top move changes)** |
|---|---|---|---|---|---|
| opening | 123 | 0.054 | 38.9 → 40.9 | **0.61** | **40%** |
| mid | 210 | 0.032 | 41.2 → 44.0 | **0.84** | **20%** |
| end | 427 | 0.016 | 129.7 → 111.0 | **0.68** | **44%** |

KL of 0.6–0.8 and a **20–44% top-move-change rate** mean the noise is doing real
work — at the BC seed it flips the most-likely root move a fifth to nearly half
the time. This is the opposite of "too weak." (If anything it's on the assertive
side.) Note the BC-seed prior here (effN 39–130) is more diffuse than the
later-epoch trace priors (effN 16–30): the model *sharpens* its prior as RL
proceeds, so re-check this on the live run after several epochs.

## 5. Temperature tracking (from traces)

The visit entropy stays high (~1.7–2.9 nats) and top-visit-fraction low
(~0.12–0.53) across plies 2–90 — the search keeps visits spread, and selection
samples broadly through the opening and midgame. The schedule (T 1.0→0.2 by ply
30) does sharpen selection in mid/end (SELECT effN 8.8→2.5), but never to greedy.
Ply 0 reads top=1.0 only because the first move is forced.

## Bottom line + recommendation

- **Exploration: about right, leaning HIGH.** All five lenses agree — diffuse
  prior, spread visit targets, 35–58% non-argmax plays, 79% unique openings,
  strong noise (20–44% top-move flips). The model *is* trying different moves; the
  reason it "looks sensible" is that (a) EVAL games are deterministic/greedy (no
  noise) — that's what the user is watching — and (b) the BC prior makes the
  *candidate* moves reasonable, so exploration = trying different *good-looking*
  moves, not visibly weird ones.
- **Under-exploration does NOT explain the regression.** The self-play data is
  diverse and exploratory; the targets are not BC-replays.
- **Do NOT increase exploration.** If anything, the 35–58% non-argmax rate is high
  enough that it may *add noise* to the value targets (outcomes partly decided by
  exploratory moves) — a reason to look at lr / value-recalibration / data volume
  instead. The richer-data run (256 games, 65536 samples/epoch) is the right
  lever to test next; if the regression persists there, the next knobs to try are
  **lower lr** (the policy may be drifting faster than it improves) or a **slightly
  sharper temperature** (cleaner targets), NOT more noise.

This is analysis only — no config was changed.
