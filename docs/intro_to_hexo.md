# Introduction to Hexo (the game)

Audience: a developer landing in this repo cold. Everything in sections 1-6 is
derived directly from the rules engine at `packages/hexo_engine/rust/src/`
(`state.rs`, `rules.rs`, `legal.rs`, `tactics.rs`, `coord.rs`), which is the
single source of truth for game rules. Model/training behavior (sections 7-8)
comes from the analysis docs under `docs/analysis/` and is labeled as such.
Where a doc and the engine disagree, the engine wins; discrepancies are flagged
in section 10.

## 1. What Hexo is

Hexo is a two-player placement game -- essentially **Connect6 played on an
unbounded hexagonal grid**. Players alternately place stones (Player 0 first),
and the first player to own **six stones in a contiguous straight line** wins
immediately. There are no captures, no territory, and **no draws**
(`state.rs:64`: "Hexo has no normal draw under the current rules").

Key properties at a glance:

| Property | Value | Source |
|---|---|---|
| Board | unbounded sparse hex grid | `coord.rs:3`, `board.rs` (AHashMap storage) |
| Coordinates | axial `(q, r)`, each `i16`; third cube axis `s = -q - r` | `coord.rs` |
| Opening | Player 0 must place exactly one stone at the origin `(0, 0)` | `rules.rs:17-23`, `state.rs:49` |
| Normal turn | two single-stone placements by the same player | `state.rs:46-56` (TurnPhase) |
| Legality | any empty cell within hex-distance **8** of any existing stone | `legal.rs:11` (`LEGAL_RADIUS = 8`) |
| Win | a fully-owned 6-cell line window; checked after **every single placement** | `tactics.rs:14,206-208`, `state.rs:304-310` |
| Draws | none in the engine | `state.rs:64` |

Note on "board size": the **game board is unbounded**. The `BOARD_SIZE = 41`
you will see in `packages/dense_cnn_restnet/python/dense_cnn_restnet/constants.py`
(and `packages/hexo_models/dense_cnn/.../constants.py`) is the **model's input
crop** -- a radius-20 hex disk (41 = 2*20+1) around the stone centroid. It is a
neural-network featurization choice, not a game rule. Confusing the two caused
a real training collapse (section 7.3).

## 2. Board geometry and coordinates

- Cells are addressed by axial coordinates `HexCoord { q: i16, r: i16 }`
  (`coord.rs:11`). Distance is the standard cube-coordinate hex distance
  (`coord.rs:77-82`).
- Straight lines exist along exactly **three axes** (`tactics.rs:23-30`):
  `Q = (1, 0)`, `R = (0, 1)`, `QR = (1, -1)`. A hex grid has 6 directions but
  only 3 unique line axes (vs 4 on the square grid of Gomoku/Connect6).
- Every cell coordinate has a stable packed **action ID**:
  `(q + 32768) << 16 | (r + 32768)` (`legal.rs:24-28`). Integer ordering of IDs
  equals deterministic `(q, r)` ordering; these IDs are persisted in training
  shards, `.hxr` records, and the frontend. The Python mirror lives in
  `packages/hexo_engine/python/hexo_engine/types.py` (`pack_coord_id`).

## 3. Turn structure (verified in `state.rs` / `rules.rs`)

Turns are represented **autoregressively**: the engine only ever applies one
stone at a time, and a phase machine tracks where the current player is inside
its turn (`TurnPhase`, `state.rs:46-56`):

| Phase | Who | Legal placements | Then |
|---|---|---|---|
| `Opening` | Player 0 | only `(0, 0)` (`rules.rs:17-23`) | Player 1 enters `FirstStone` |
| `FirstStone` | current player | any empty cell within distance 8 of any stone | same player enters `SecondStone` |
| `SecondStone { first }` | same player | as above, but **not** the cell just played (`rules.rs:25-29`, `MoveError::ReusedFirstStone`) | control passes; opponent enters `FirstStone` |

So the placement sequence of a game is:

```
ply 0: P0 forced at (0,0)          (one-stone opening turn)
ply 1: P1 FirstStone   } P1's turn
ply 2: P1 SecondStone  }
ply 3: P0 FirstStone   } P0's turn
ply 4: P0 SecondStone  }
...
```

This is exactly the Connect6 "1 then 2-2-2-..." scheme. The phase transition
logic is `state.rs:312-330`; the explicit comment at `state.rs:313-316` confirms
the opening is "a special one-stone turn by Player 0".

Legality detail: the radius-8 neighborhood is taken around **any stone of
either color** (union of disks; see the test oracle
`recompute_non_opening_legal_ids`, `state.rs:559-569`). After the opening, the
legal set is the radius-8 disk around the origin: 216 cells (217-cell disk
minus the occupied origin).

## 4. Win condition and "sudden death"

- **Win = six in a line.** The engine tracks every 6-cell straight-line window
  (`WINDOW_LEN = 6`, `tactics.rs:14`). A placement touches exactly 18 windows
  (3 axes x 6 offsets, `tactics.rs:17`). A window is a win for a player when
  all six of its cells are that player's stones (`is_win_for`,
  `tactics.rs:206-208`). Six **or more** in a row wins -- there is no overline
  rule, because any 7-in-a-row contains a fully-owned 6-window.
- **Resolution is immediate and per-placement.** A win is checked after every
  single stone (`state.rs:304-310`). The moment a placement completes a
  6-window, the game is terminal: the winner is recorded, no further legal
  moves exist (`legal_move_count` returns 0, `state.rs:204-213`), and -- per
  the header comment at `state.rs:9-10` -- **"If the first stone of a two-stone
  turn wins, the second stone is never played."**
- **There is no separate "sudden death" mechanic and no simultaneous-threat
  rule.** A repo-wide grep for "sudden" returns nothing. Because stones are
  placed strictly one at a time and the win check runs after each, two players
  can never complete winning lines simultaneously; whoever physically completes
  six first wins, full stop. "Both players have unstoppable threats" resolves
  purely by move order: the player whose placement lands first wins. There is
  no tiebreak, no priority rule, and nothing to resolve.

## 5. Threats (engine-level tactics vocabulary)

The engine maintains incremental window masks (`WindowStore`, `tactics.rs`)
that the model packages' Threat-Space Search (TSS) builds on:

- **Active window**: a 6-window containing stones of exactly one player
  (`is_active`, `tactics.rs:184-186`). A window with both colors is dead for
  winning purposes.
- **Threat**: an active window with **>= 4** stones of one color
  (`threat_player`, `tactics.rs:189-192`). With two placements per turn, a
  4-of-6 single-color window can be completed in one turn, hence the
  threshold. The defender's per-turn placement budget (1 stone in the opening
  reply, otherwise 2) drives the hitting-set logic in
  `packages/hexo_models/rust/src/threats_shared.rs`.

## 6. Draws and truncation

- **The engine has no draw.** `GameOutcome` (`state.rs:66-71`) always has a
  winner. The unbounded board can never fill up. A game that has not produced
  six-in-a-row simply continues.
- **Truncation is a training/runner artifact, not a rule.** The match runner
  aborts a game at `spec.max_actions`
  (`packages/hexo_runner/python/hexo_runner/loop.py:87-94`, stage
  `runner.max_actions`), and self-play configs default to `max_actions = 1024`
  (`packages/dense_cnn_restnet/python/dense_cnn_restnet/config.py:169`). A
  truncated game has winner `None`; under the original sample finalization,
  every row got value label z = 0 -- "provably wrong" labels, per the comment
  at `config.py:122-128`. The live main_4 run therefore sets
  `drop_truncated_rows = true` (C2 in `docs/analysis/MAIN4_RECOMMENDATION.md`)
  so truncated games write no training rows at all.

## 7. Strategy notes -- what the trained models actually learned

These are empirical observations from the ResTNet self-play lineage, not rules.

### 7.1 Learned opening shape (from `docs/analysis/RESTNET_OPENING_DIVERSITY.md`)

By epoch ~30 of run main1, a strongly role-asymmetric opening had emerged
through RL (it was not present after the human-corpus prefit):

- **Player 1** (first free turn): FirstStone on the radius-8 legal frontier
  (distance 7-8 from origin) in ~79% of games, then SecondStone back adjacent
  to the origin (~81%). The "one near + one at the edge" signature rose from
  ~8% (ep1) to ~86% (ep29).
- **Player 0** (replying): the opposite -- both stones clamp tightly around the
  origin (within distance 3 in 94-100% of games). Its FirstStone collapsed to a
  single D6-canonical cell at ep29 (0.00 bits of canonical entropy).
- Diversity survives in **direction** (openings rotate freely around the
  origin; ~22 D6-canonical first cells) but is stereotyped in **radius/shape**.
- The raw policy prior (not search) carries the pattern: ~96% of player 1's
  FirstStone prior mass sits on the edge ring; MCTS slightly softens it. The
  doc's verdict: a learned, value-driven strategy that was still drifting, not
  a featurization bug -- and it did not hurt strength (peak SealBot eval at the
  time of analysis).

### 7.2 First-mover asymmetry

Player 0's "advantage" is one forced stone at the origin; Player 1 then gets
the first free two-stone turn. The roles see genuinely different positions, and
the models learned opposite spatial styles for them (above). Practical
repo-side consequence: position statistics conditioned on FirstStone vs
SecondStone phase have systematic parity effects -- an owner-swap value-head
probe was misread as "optimism" until the FirstStone-parity confound was
identified (see the project memory note on value-head optimism).

### 7.3 The frozen-win zugzwang (a cautionary curiosity)

The root cause of the main_3 run collapse (`docs/analysis/MAIN4_RECOMMENDATION.md`)
is a perfect illustration of game-vs-model boundary confusion: the **engine**
allows play within radius 8 of any stone on an unbounded board, but the
**model** only sees/considers a radius-20 crop around the stone centroid. In
long games the stone blob outgrows the crop, and a standing immediate win whose
completion cell lies just outside the rim becomes simultaneously unplayable,
unblockable, and invisible -- **for both players**. Games froze for hundreds of
plies (median 509 vs 183 healthy) waiting for centroid drift (~0.02 cells/ply)
to re-admit a win cell -- effectively a coin flip. 47/47 audited cases were
engine-verified standing wins left unplayed. The fix (main_4, C3) is an
engine-truth side-channel: `packages/dense_cnn_restnet/python/dense_cnn_restnet/win_tracker.py`
incrementally tracks standing 6-window wins, and self-play plays the winning
stone (clone-verified against the engine) instead of the search move when all
wins are out-of-crop. Moral: the engine never had this bug; the model's view of
the game did.

### 7.4 Game length

Healthy self-play games run roughly 95-135 decisions (the main_4 gate band);
the adaptive temperature schedule uses a persisted EMA of mean decisions/game
(seeded at 115 for main_4).

## 8. Comparisons to related games

| Game | What transfers to Hexo | What does not |
|---|---|---|
| **Connect6** | Almost everything: the 1-then-2-2-2 placement scheme (designed to fix Gomoku's first-mover advantage), 6-in-a-row goal, threat-counting logic (a turn answers at most 2 threats), no draw concern in practice | Square grid has **4** line axes; Hexo's hex grid has **3** (`tactics.rs:23`). Connect6 is played on a bounded 19x19; Hexo is unbounded with a radius-8 placement locality rule and a forced-origin opening |
| **Gomoku / Renju** | Line-completion intuition, threat sequences (fours/threes generalize to Hexo's >= 4-of-6 windows) | One stone per turn; 5-in-a-row; overline/forbidden-move rules (Renju) have no Hexo equivalent |
| **Hex** | Only the grid. Despite the name, Hexo is NOT the connection game Hex | Hex's goal is connecting opposite board edges on a bounded rhombus; no line-of-N condition; Hex provably has no draws by topology, Hexo simply never fills its infinite board |
| **Go** | Nothing rule-wise. What transfers is the **training method**: this repo is an AlphaZero/KataGo-style pipeline (PUCT MCTS + policy/value net, Dirichlet root noise, playout-cap randomization, D6 symmetry augmentation mirroring Go's 8-fold square symmetry) | Captures, territory, ko, komi, passing -- none exist in Hexo |

## 9. Glossary

| Term | Meaning here |
|---|---|
| **ply / placement** | One single stone placed. The engine is fully autoregressive: a "move" in engine terms is always one stone (`Placement`, `state.rs:59-62`). |
| **turn** | One logical turn: 1 placement for the opening, 2 placements otherwise (`MoveRecord`, `state.rs:86-93`). |
| **decision** | One model search decision = one ply chosen by MCTS in self-play. The forced opening stone is applied but is not a free choice (legal set has size 1). "dec/game" in run telemetry counts decisions per game (`total_decisions` in `selfplay.py`). |
| **window** | A specific 6-cell straight-line segment on one of the 3 axes; the unit of win/threat detection (`WindowKey`, `tactics.rs:56`). |
| **threat** | An active window (single-color) with >= 4 stones (`tactics.rs:189-192`). |
| **visits** | MCTS simulation count for one decision (e.g. `search_visits = 512`); the per-decision search budget. |
| **PCR** | Playout Cap Randomization (KataGo, Wu 2020): each decision is independently a "full" search (recorded for policy training, with noise/forced playouts) with probability `pcr_full_proportion`, else a cheap "fast" search that only plays a move (`config.py:223-233`). |
| **SealBot** | An external C++ minimax baseline bot (separate checkout at `E:\SealBot`), used as the fixed evaluation opponent via the subprocess adapter `packages/hexo_runner/python/hexo_runner/adapters/sealbot.py`. |
| **.hxr record** | The binary game-record format (magic `HEXOREC1`): header + per-game action-ID sequences, winner, abort metadata. Rust codec in `packages/hexo_utils/rust/src/records.rs`, consumed via `hexo_runner.records`. Every self-play/eval/match game is persisted as `.hxr`. |
| **action ID** | Packed `u32` cell coordinate, `(q+2^15)<<16 | (r+2^15)` (`legal.rs:24-28`); the stable move encoding in records, shards, and the frontend. |
| **crop** | The model-side radius-20 (41x41) input disk around the stone centroid. A featurization construct, not a rule. |
| **frozen win** | A standing engine-verified win whose completion cell lies outside the model crop; see 7.3. |
| **D6** | The order-12 symmetry group of the hex grid about the origin (6 rotations x reflection), used for training-data augmentation and for canonicalizing opening statistics. |

Active-vs-legacy note: the engine (`packages/hexo_engine`) is active and shared
by everything. The active model lineage is `packages/dense_cnn_restnet`
(run main_4); `packages/hexo_models/dense_cnn` (Python side),
`packages/hexo_models/hexgt`, and `packages/hexgnn` are legacy/parked lineages,
though the dense_cnn **Rust** accelerator remains the active native engine
bridge for restnet.

## 10. Engine-vs-docs discrepancies found while writing this

1. **"Sudden death" does not exist.** No engine code, doc, or UI string
   mentions a sudden-death or simultaneous-threat rule (repo grep for "sudden"
   is empty). If you encounter the term elsewhere, the engine reality is
   simply: win checked after every single placement, first completion wins,
   ties impossible by construction (`state.rs:304-310`).
2. **`new_game(seed=..., scenario=...)` silently discards both arguments**
   (`pybridge.rs`, `let _ = seed; let _ = scenario;`). Callers such as
   `hexo_runner` and the frontend pass a seed through, implying engine-side
   randomness/reproducibility that does not exist -- the game itself is fully
   deterministic given the placement sequence. Misleading API shape, not a
   rules bug.
3. **"Board size" terminology in model docs** (BOARD_SIZE = 41, "radius-20
   board") refers to the model crop only; the engine board is unbounded with
   `i16` coordinates. Several analysis docs say "the board" when they mean
   "the crop"; the engine wins -- legal play extends radius 8 beyond any stone,
   indefinitely.
4. **GameSpec.scenario is vestigial**: the runner raises if it is non-None
   (`loop.py:57-58`), so no alternative starting positions exist despite the
   field's presence.
