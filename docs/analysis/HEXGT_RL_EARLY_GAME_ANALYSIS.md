# hexgt RL — learning review + "early-ending strategy" hypothesis (epochs 0-2)

Read-only analysis of `runs/hexgt_rl_main2` (512-visit RL, BC/epoch-8 seed, C1
exploration). 3 RL epochs complete (0,1,2 = dashboard 1,2,3), epoch 3 in progress.
Sources: `rl_train.log` per-epoch Q-metrics + train loss, `eval/*_eval.json`
(head-to-head), and the per-game `.npz` shards (`scripts/_game_analysis.py`).
NOTE: shards hold sampled (nonterminal) positions only, so winner is read from the
final value target and the terminal/winning move itself is not in the data — the
must-block-cell micro-analysis needs full `.hxr` records (see fix recommendation).

## 1) Is it learning? YES — improving, not stalling.

| metric | ep0 | ep1 | ep2 | trend |
|---|---|---|---|---|
| train total loss | 3.658 | 3.492 | 3.424 | ↓ steady |
| policy loss | 1.992 | 1.906 | 1.882 | ↓ |
| value loss | 0.825 | 0.752 | 0.716 | ↓ (value calibrating) |
| opp-policy loss | 3.361 | 3.335 | 3.304 | ↓ |
| prior entropy (Q3) | 2.64 | 2.55 | 2.51 | ↓ (policy sharpening) |

Head-to-head @512 visits (eval every 3 epochs, so only the ep0 point exists yet;
next at ep3): **baseline 30% / 5% → ep0 45% / 7.5%** vs dense_cnn e24 / SealBot.
→ Loss decreasing on every head, prior sharpening, strength up. Clearly learning.
(No fixed-holdout loss is computed by this run; train-loss + eval are the signals.)

## 2) Recent-games analysis (256 games/epoch)

| metric | ep0 | ep1 | ep2 |
|---|---|---|---|
| length median / mean | 50 / 66.7 | 48 / 63.2 | 50 / 60.1 |
| length min / max | 16 / 239 | 16 / 239 | 22 / 226 |
| % short (<40) | 35.5 | 40.6 | **31.6** |
| % very short (<30) | 14.1 | 16.4 | 11.7 |
| decisive | 98.0 | 99.6 | **100.0** |
| **first-mover (p0) win %** | 46.1 | 47.3 | **47.7** |
| distinct 3-move openings | 69 | 68 | **51** |
| top-3 opening share | 28.5% | 18.4% | 30.1% |
| visit entropy / top-1 visit | 1.59 / .52 | 1.25 / .60 | 1.35 / .58 |
| effective moves (search) | 9.1 | 6.5 | 7.3 |

**Length is BIMODAL** (ep2 histogram, buckets of 20): `20-39: 81`, `40-59: 76`,
`60-79: 45`, `80-99: 24`, `100-119: 17`, … `140-159: 8` — a large short cluster
(~30-40% under 40 moves) plus a long decisive tail to ~220. Median ~50, well below
the older ablations' ~100-160. **But it is stable, not collapsing** (med 50→48→50;
%short 35→41→**32**, i.e. the short cluster is NOT growing — ep2 has the fewest).

**Winner balance is ~50/50** (p0 46-48%, so the second mover wins slightly more).
This is the key result: there is **no one-sided first-mover advantage** — both
sides convert the short decisive games. An *unanswered* exploit would skew the
winner hard toward the side that plays it; it does not.

**Opening concentration is moderate and narrowing** (distinct 3-move openings
69→51; summary uniq_open 77%→64%, move-2 entropy 1.92→1.67). Top recurring 3-move
lines at ep2: `(0,0),(-2,0),(0,1)` ×33 (13%), `(0,0),(0,3),(1,-1)` ×24, … — a
handful recur, but the top-3 still cover only ~30% and ~50 distinct lines remain.
(Move 0 is always center `(0,0)`.) So it is concentrating, but not onto a single
killer line.

## 3) Is it learning the counter / stuck or transient?

Leans **transient self-play sharpening, not a stuck exploit**:
- Strength ↑ (30→45% vs dense), loss ↓, prior sharpening — healthy RL.
- Short-game fraction NOT growing (ep2 < ep1 < — i.e. 32% < 41%); length stable.
- Winner ~50/50 → defenders ARE converting; no unanswered tactic dominating.
- Search stays exploratory enough (visit entropy ~1.3, top-1 ~0.58, ~7 effective
  moves, ~30% "forced") — it is not collapsing onto one move.

Only 3 epochs exist, so this is an early read; the **ep3 eval (next, dashboard
epoch 4)** is the decisive check on whether strength keeps climbing.

## Verdict
- **Learning: YES** — improving on loss + head-to-head over the first 3 epochs.
- **Early-ending-strategy hypothesis: PARTIALLY supported, NOT a single unanswered
  exploit.** Games are short (med ~50, down from ~100-160) and bimodal with a
  persistent ~35% short-decisive cluster, and openings are narrowing — consistent
  with sharp, tactical, decisive play. But the ~50/50 winner balance and the lack
  of a dominant opening line contradict the "one side found a tactic the other
  can't block" framing. Best description: **the seed plays sharp decisive tactical
  games that often resolve early, and both sides convert** — likely the epoch-8
  seed's style + 512-visit sharper search, not a degenerate broken line.
- **On track, not stuck** (so far): short-game share is flat-to-down and strength
  is rising. Watch the ep3/ep6 evals + whether median length recovers as the prior
  keeps sharpening.

## Limitation / follow-up
The precise "is the defender failing to play the must-block T1 cell, and is it
starting to put visit mass there over epochs" test needs the **terminal/winning
move + threat geometry**, which the `.npz` shards omit. This is the same gap behind
the truncated replays — writing full `.hxr` game records in `selfplay.py` (low
effort; `game["actions"]` already holds the full sequence incl. the winning move)
would enable exact winning-line + blocking-move analysis next.
