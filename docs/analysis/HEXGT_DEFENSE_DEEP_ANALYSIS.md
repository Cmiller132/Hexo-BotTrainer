# hexgt defensive-blindness / plateau — deep analysis

**Framing question:** AlphaZero learns strong defense from *pure* self-play — why isn't
hexgt's self-play teaching defense, and how do we fix it purely inside hexgt (no
mixed/external games)?

**Status:** training STOPPED (HALT flag set, supervisor down, GPU freed) for this
analysis. Checkpoints preserved (latest = epoch 42; analysis run on epoch-39 net).
Read-only on the run; no training code/config changed.

---

## HEADLINE CONCLUSION

**Root mechanism — confidently MIS-calibrated value head on off-self-play-distribution
positions.** hexgt's self-play is diverse in *openings* but collapses to a narrow,
sharp *midgame*; it never covers the attack/defense structures that real opponents
(dense_cnn, SealBot) create. On those out-of-distribution positions the value head is
not just overconfident but **anti-calibrated** (predicts +0.6…+1.0 at positions it
actually LOSES; predicts losing at positions it wins). MCTS trusts that wrong value, so
it **never plays the defensive block — even though the block is always in the candidate
set, and even at 4096 visits**. More search makes it *worse* (drives confidence +0.61→
+1.00 at a losing position). It is a **self-reinforcing blind spot entirely inside the
self-play loop**: overconfident value → sharp/narrow midgame → no defensive coverage →
value never corrected → games shorten (median 59→37) → plateau (~54%, flat).

**Top plan items (all pure self-play, no external games):**
1. **Recalibrate the value head** (root fix): soft/temperature-scaled value targets
   (label-smoothing ε≈0.05–0.10 or z×0.9) ± a calibration term, and lean on the STV
   heads. Kills the +1.0 over/anti-confidence → search stops trusting a wrong value →
   explores → self-play covers defenses → value self-corrects.
2. **Restore MIDGAME exploration** (coverage fix): keep temperature high longer
   (`temperature_decay_moves` 30→~60–80, `temperature_floor` 0.1→~0.25), raise
   `forced_playout_k` (2→3–4), add KataGo playout-cap randomization. Makes self-play
   actually *play* defenses → generate defended games → value learns "block = good."
3. **Sims/capacity are NOT the bottleneck** (proven below) — secondary; revisit only
   after 1+2.

---

## METHOD
- Net: `hexgt_rl_epoch000039.pt` (the net that played the analyzed eval games).
- Data: eval `.hxr` records vs dense_cnn e24 and SealBot best-50ms, epochs 36+39 (40
  games each); per-epoch self-play metrics + eval win-rates from `rl_train.log`.
- Tools: candidate set via `rust_bridge.candidate_ids`; search via
  `HexgtPlayer`/`mcts_session.run` at 512–4096 visits (GPU); value via
  `forward_policy_value` + `decode_binned_value` (reliability + swing).

---

## FINDINGS

### A) The plateau is real (not noise)
Eval-vs-dense slope **+0.24%/epoch** over e0–39 (flat); last-8 mean **53.8%**, range
30–68, σ 10.7 (≈ the 40-game ±8% noise band; no climb past the early ~62–68% peaks).
Loss side: **policy creeping up** (2.1→2.3), **value flat** (~0.58–0.62).
**Self-play game length FALLING: median 59 (e0–7) → 37 (e30–42)**, 100% decisive.

### B) Exploration is NOT collapsing at the opening (rules out the obvious cause)
Opening uniqueness **99–100%** (e8+), move-2 entropy **rising 1.9→3.0**, prior-entropy
2.5→2.8. So the plateau is *not* from replaying a few openings. The concern is the
**shortening, sharpening midgame** (root exploration works; midgame exploration doesn't
— temperature has decayed to ~greedy by the time the game is actually decided).

### C) Loss forensics
- **Color:** worse as **first player — P0 45% vs P1 60%** (under-converts initiative).
- **Length:** wins are **long grinds (med 167)**; losses **shorter (dense med 53 /
  SealBot med 31)**; vs SealBot **101/180 losses fall in 21–40 moves**.
- **Value-swing localization:** dense losses flip late (frac 0.82, 7/8 in the >60%
  bucket); SealBot losses flip at frac 0.97 — value just before the swing **med +0.80,
  8/8 games the net was confidently "winning" then lost.**
- **Motif:** no early blunder; the opponent **completes a winning line the value head
  never flagged** — a defensive-recognition failure.

### #1 Candidate-set audit (was the TOP-priority hypothesis) — REFUTED
At the last blockable swing position, was the defensive block in the candidate set?
**Block IN candidates: 28/28 (14/14 dense, 14/14 SealBot); MISSING: 0/28.** Candidate
sizes med ~210–246. **It is NOT a candidate-generation bug** — the block is always
available; search/value simply doesn't choose it. (The active-window rule guarantees an
opponent threat's empty cells are candidates.)

### #2 Search-depth test — DECISIVE: more search does NOT find the defense
At the crit (block) positions, re-searched with the epoch-39 net, greedy:

| game | block cells | v512 | v1024 | v2048 | v4096 |
|---|---|---|---|---|---|
| dense-0001 | (11,-14)(12,-14) | +0.61 | +0.80 | +0.90 | +0.95 |
| dense-0004 | (12,-2)(13,-2) | +0.59 | +0.80 | +0.90 | +0.95 |
| sb-0000 | (1,1)(2,1) | +0.61 | +0.80 | +0.90 | +0.95 |
| sb-0005 | (-23,16)(-21,14) | +0.60 | +0.80 | +0.90 | +0.95 |
| … (10/10) | | | | | |

**Every position: value rises toward +1.0 as visits increase, and the block is NEVER
chosen (`no-blk` at all depths).** The net is *confidently wrong*; MCTS amplifies the
error (it backs up the net's positive leaf values). → Not a search-strength problem;
not candidates (#1); the threat was blockable (≥2 empties, not yet an open-four). **The
NET cannot represent/recognize the defense.**

### #4 Self-play exploration config (the "why narrow midgame")
Live self-play: `search_visits=512`, `c_puct=1.5`, Dirichlet `alpha_sum=6.6 eps=0.25`
(root only), `temperature=1.0 → final 0.2 over temperature_decay_moves=30, floor 0.1`,
`forced_playout_k=2.0`, widening 0.95/96/2. Games are ~37 moves → by the time the game
is decided (midgame), **temperature is already ~0.2 (near-greedy)** and there is **no
root noise past the opening**. Root Dirichlet diversifies *openings* (hence B) but the
midgame is played sharply *on the overconfident value* → narrow midgame coverage → the
attack/defense structures real opponents use never get generated or learned.

### #5 Value calibration — anti-calibrated off-distribution (reliability diagram)
Per-position, predicted value vs actual outcome (28 eval games):

| net prediction | n | mean ACTUAL |
|---|---|---|
| +.6 … +1 (win) | 221 | **−0.10** |
| +.2 … +.6 | 892 | **−0.24** |
| −.2 … +.2 | 1095 | −0.01 |
| −.6 … −.2 | 675 | +0.26 |
| −1 … −.6 (loss) | 245 | +0.31 |

**Confident predictions are inverted** on real-opponent positions. The net is roughly
calibrated only in the uncertain middle. (It fits its *training* value targets fine —
loss ~0.58 — so this is a DISTRIBUTION/calibration failure, not raw capacity.)

### #6 First-player under-conversion (45% P0 vs 60% P1)
Same root cause: as the first/attacking player the net **over-values its own attack**
(reliability table: confident-positive → actually loses) and can't convert (wins need
167-move grinds); as second player it plays more reactively and does better. A facet of
the value miscalibration, not a separate opening bug.

---

## CAUSAL CHAIN (entirely inside the self-play loop)
1. Value head is **overconfident** (decisive ±1 self-play outcomes → extreme targets).
2. Overconfident value → MCTS plays the **midgame sharply/greedily** (temp→0.2 by
   move 30, no midgame noise) → **narrow midgame** trajectories.
3. Narrow midgame → self-play **never generates** the attack/defense structures strong
   opponents use → those positions are **out-of-distribution**.
4. On OOD positions the value is **confidently wrong / anti-calibrated** (#2, #5) → MCTS
   trusts it → **never plays the (available) block** (#1) → loses.
5. Because self-play doesn't play those defenses, the value is **never corrected** on
   them → loop persists; as the net sharpens, games **shorten (59→37)**, coverage
   shrinks further → **plateau**.

This is exactly why "pure self-play teaches defense in AlphaZero" isn't happening here:
AlphaZero keeps the self-play distribution broad (sufficient exploration + the value
staying honest), so the value learns defense from its own games. hexgt's loop collapsed
to a narrow, overconfident regime — so its self-play distribution is too narrow to cover
(and thus learn) defense.

---

## REMEDIATION PLAN (ranked; pure hexgt self-play, NO external/mixed games)

### 1. Recalibrate the value head — ROOT FIX (highest leverage)
- **Soft value targets:** scale the ±1 outcome (e.g. `z*0.9`) or label-smooth the
  65-bin target by ε≈0.05–0.10. Code: `dense_cnn/samples.py::_winner_value` (target
  source) or the bin construction in `hexgt/losses.py::scalar_to_binned_target` /
  `binned_value_loss`; wire a `value_label_smoothing` config knob.
- Optionally a small **value-confidence penalty** (entropy regularizer on the value
  softmax) in `hexgt_loss`.
- **Lean on the STV heads** already added (short-horizon value is less extreme; they
  shape the trunk toward calibrated near-term value).
- *Expected effect:* value can't sit at ±1 → MCTS stops over-trusting it → explores →
  self-play covers defenses → value self-corrects.
- *Validate:* reliability diagram becomes monotonic; |val| drops below 1.0; the crit
  positions' value falls toward 0/negative (re-run #2/#5).

### 2. Restore MIDGAME exploration — COVERAGE FIX
- `temperature_decay_moves` **30 → 60–80** and `temperature_floor` **0.1 → ~0.25** so
  the *midgame* (where Hexo is decided) is still sampled, not greedy.
- `forced_playout_k` **2 → 3–4** (forces visits to low-prior moves → discovers the
  block).
- Consider **root Dirichlet beyond the opening** (or KataGo **playout-cap
  randomization**: a fraction of moves get full search, the rest reduced — cheaply
  generates diverse, defended games + honest value targets). Code: `_rl_train.py`
  EXTRA_ARGS + `hexgt/selfplay.py` temperature/noise schedule.
- *Expected effect:* self-play actually *plays* defenses and longer games → defended
  outcomes train the value.
- *Validate:* midgame visit/policy entropy up; self-play game length stops shrinking /
  rises; OOD calibration improves; eval win-rate climbs past the ~54% plateau.

### 3. Sims / search strength — SECONDARY (only after 1)
Proven (#2) that more sims alone make it worse while the value is wrong. After
recalibration, raising self-play `search_visits` (512→768/1024) would then help search
find defenses. Don't raise sims first.

### 4. Capacity — LAST
The net is confidently *wrong*, not *uncertain* → distribution/calibration, not raw
capacity. Only revisit (token_dim/layers up) if, after 1+2, the search-depth test shows
the net still can't represent defenses it has now seen in-distribution.

---

## ADDENDUM — quantified side-to-move OPTIMISM bias (both players think they're winning)
Same identical board, value evaluated from each side's perspective (FirstStone
positions, owner-swap exact; 250 positions each):

| set | mean(vA+vB) | median | BOTH predict win | sum>+0.5 | sum<−0.5 |
|---|---|---|---|---|---|
| self-play (in-dist) | **+0.82** | +0.80 | **51%** | 196/250 | **0/250** |
| eval (OOD) | +0.60 | +0.52 | 38% | 135/250 | 0/250 |

A zero-sum-consistent value would sum to ~0; instead it sums to **+0.8** and is **never
strongly negative** — a pure, systematic **optimism bias of ~+0.4 per side**: whoever is
to move is told they're ~0.4 ahead even on a balanced board, so **both players think
they're winning** (51% of identical boards). This is the mechanistic source of the
defensive blindness: each side's value says "I'm ahead, press the attack" → neither
prioritizes defense → attacker wins fast → short games; and it's exactly why the
first/initiative player over-presses and under-converts (#6). (Within-game adjacent
cross-player ply pairs are both-positive only 6–8% — lower because those span a real
board change; the same-board test isolates the pure bias.) This optimism + the OOD
extrapolation together produce the anti-calibrated confident-wrong values (#2/#5).

**Binning is IDENTICAL to dense_cnn** (confirmed): both 65 bins, `linspace(-1,1,65)`,
`position=(v+1)*((65-1)/2)` adjacent-bin soft target, cross-entropy loss.
hexgt: `losses.py:27-94` (`scalar_to_binned_target`/`binned_value_loss`), value head
`architecture.py:221-223` reading the SIDE hub token (`_graph_readout`
`architecture.py:256-265`). dense_cnn: `constants.py:12` (VALUE_BINS=65),
`losses.py:33/80`, value head `ValueBinnedHead` `architecture.py:143-159` (1×1 conv →
flatten over all 1681 board cells → MLP). **The bin structure + loss are the same; the
only difference is the READOUT** (hexgt: single SIDE token; dense_cnn: whole-board pool).
This makes value soft-label calibration (plan item 1) directly portable from dense_cnn.

## NOT the cause (ruled out with data)
- **Candidate generation** — block is always a candidate (#1, 0/28 missing).
- **Opening-exploration collapse** — opening diversity 99–100%, rising (B).
- **Pure search depth** — 4096 visits can't defend; makes it worse (#2).
- **Raw capacity (primary)** — value fits training targets (loss 0.58); failure is OOD
  miscalibration, not under-capacity (#5).
