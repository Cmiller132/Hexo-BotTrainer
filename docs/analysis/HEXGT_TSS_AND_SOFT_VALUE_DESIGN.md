# hexgt: Threat-Space Search & Softer Value Targets — Deep Design Analysis

Design / analysis only. **No code, config, or model files are changed by this
document, and nothing here runs training or touches the live dense_cnn run.**
This is a deeper dive on two of the recommendations already ranked in the
companion doc
[`HEXGT_ARCH_DESIGN_EXPLORATION.md`](HEXGT_ARCH_DESIGN_EXPLORATION.md) — namely
**Threat-Space Search (TSS)** (which §5/§6.3 of that doc gestured at as "hot
tokens" / a deferred search-side option) and **soft value targets** (its Rank
0a). Read that doc first; this one assumes its findings as background and does
not repeat the full architecture diagnosis.

Every hexgt-side claim is cited to a file path. Most are in the hexgt worktree
`E:\Hexo-BotTrainer-hexgt` (read-only for this pass; a separate session is
editing it). A few are in the dense_cnn main tree `E:\Hexo-BotTrainer`. Paper
claims are cited to the source, and the **provenance of each source (full text
fetched vs. reasoned-about) is listed in the final section** so the reader knows
which formulas are quoted verbatim and which are reconstructed from the
well-documented method.

> ## HEADLINE (corrected threat model)
>
> **Threat model (authoritative).** Hexo/Connect6 place **two stones per move**,
> so **any active length-6 window with ≥4 of one player's stones is an immediate
> winning threat**: a count-4 has two empty gaps the owner fills with one
> two-stone move; a count-5 has one gap. **A single four is forcing**, count-4
> and count-5 are roughly equal danger, and an open four/five usually costs the
> defender *both* placements to block. **Any unanswered ≥4 window loses** — which
> makes the defensive problem *more* acute than an earlier draft of this doc
> claimed (that draft wrongly treated only the *double*-four as forcing). The
> engine agrees: its threat predicate is `count(player) >= 4`
> (`tactics.rs:188-203`).
>
> **Recommended TSS form.** A depth-1 threat probe whose candidate set is **the
> empty cells inside any active ≥4 window** (offense = complete; defense = cover).
> Land it in three pieces: (1) a **depth-1 leaf override** in MCTS that returns a
> HARD WIN if the side to move can complete a ≥4 window this move, else a HARD
> LOSS if it cannot cover all of the opponent's ≥4-window gaps with its two
> placements — the *hard ground-truth fix* for "value +0.8 right before a forced
> loss"; (2) **complete-now / must-answer "hot token" CANDIDATE features**
> (including the currently-missing **count-4** completion flag); (3) graded
> forcing/open-ends severity features. Defer the full multi-ply VCF/VCDT solver
> (root-only, depth-bounded if ever).
>
> **Where it ranks.** Soft value targets (PART 2) stay **Rank 0a — do first**.
> The TSS hot-token features sit **just below Rank 1** (folded into the readout
> pool); the **depth-1 leaf override** is a new, sharper member of that family —
> the search-side *guarantee* the feature/readout fixes only make *likely*. The
> deep in-tree solver remains the deferred search-side option.

---

## 0. The failure this is meant to fix (recap, grounded)

The forensic pass diagnosed hexgt's core weakness as **defensive value
miscalibration**, with two measured symptoms:

1. **Over-confident loss.** The value head predicts ≈ **+0.8 (winning)** in the
   plies right before losing, in 8/8 lost games.
2. **Side-to-move optimism bias of +0.82.** For the *same* board, `v(A) + v(B)`
   sums to ≈ +0.82 instead of ≈ 0. A perfectly calibrated zero-sum value would
   sum to 0; hexgt's value is **anti-calibrated** — both sides think they are
   ahead.

Self-play games are **short, 100% decisive, and getting shorter**
(`selfplay.py` logs `game_lengths`, `forced_decisions`, opening diversity).
External/other-bot games are **rejected** by the user; every fix must live
inside **pure self-play**.

The current value-target machinery (grounded):

- **Main value target is the hard game outcome.**
  `dense_cnn/.../samples.py::finalize_game_samples` sets
  `value=_winner_value(winner, player)` → `+1 / −1 / 0`
  (`samples.py:198`, `samples.py:318-321`). hexgt reuses this finalize verbatim
  (`hexgt/.../selfplay.py:462` calls `finalize_game_samples`).
- **Value head is a 65-bin distributional head**, identical binning to
  dense_cnn: `VALUE_BINS = 65` (`hexgt/.../constants.py:16`), and the binning
  math is **reused verbatim** (`hexgt/.../losses.py:1-23` docstring;
  `scalar_to_binned_target` at `losses.py:40-60` maps any scalar in `[−1,1]`
  to a soft two-bin target; `binned_value_loss` at `losses.py:63-94`).
- **The only bootstrapped value signal today is the auxiliary STV heads.**
  `_short_term_value_targets` (`samples.py:337-369`) builds, per horizon `h`, an
  **EMA of future MCTS root values** with decay `λ = h/(h+1)`,
  perspective-corrected (`samples.py:354-356`). They are weighted small
  (`short_term_value_weight = 0.25` default, `config.py:83`; the live run has
  used 0.10) and read from the SIDE hub like the main value.
- **`root_value` is captured per decision and is in hand at finalize time.**
  `selfplay.py:424` appends `(player, sample, search.root_value)` to
  `game["pending"]`, and `finalize_game_samples` receives exactly that triple
  (`samples.py:168-185`). It is currently consumed *only* by STV
  (`_root_value` is discarded by the main-value loop, `samples.py:185`).
- **Replay is recency-weighted** with geometric decay `0.9^(current−epoch)`
  (`_rl_train.py:121-122` `epoch_recency_weight`, default
  `--replay-recency-decay 0.9` at `_rl_train.py:175`), cap-bounded
  (`select_window_epochs`, `_rl_train.py:94`).

Both halves of this doc attack the value miscalibration: **Part 2 (soft value
targets)** recalibrates the *label* the value head learns; **Part 1 (TSS)**
gives the search/representation an *exact tactical* signal so the net is never
the sole arbiter of "am I about to lose to a forced threat."

---

# PART 1 — Threat-Space Search (TSS) for Connect6 / Hexo

> **Correction note (this revision).** An earlier draft of this section
> imported the *Gomoku* threat hierarchy wholesale and asserted that "in
> Connect6/Hexo a single four is NOT forcing — the defender blocks with one of
> two stones." **That is wrong for this game and was the load-bearing error in
> the old Part 1.** Because Hexo/Connect6 place **two stones per move**, a
> **count-4 window is an *immediate* winning threat**: it has two empty gaps,
> and the mover fills *both* gaps with their two placements next turn to make
> six. A count-5 (one gap) wins with a single placement. So **a single four is
> forcing**, count-4 and count-5 are roughly equal danger, and the old
> "double-threat is the unit of analysis" framing understated the defensive
> problem. The corrected model below — and the engine's own
> `count >= 4 ⇒ threat` predicate — is the authority; the Connect6 papers are
> reconciled *to* it, not the reverse. The reason the change is auditable: it
> makes the defensive burden *strictly worse* than the old draft claimed (ANY
> unanswered ≥4 window loses, not just an unanswered double-four), which is
> exactly the direction that matches hexgt's measured "+0.8 right before a
> forced loss."

## 1.1 What a "threat" is (corrected model — and it matches the engine)

In the *k*-in-a-row family (Gomoku/Renju, Connect6, six-in-a-row), a **threat**
is a partial line that is close enough to completion that the opponent must
respond. The Connect6 literature's working definition (Wu & Kang,
*Dependency-Based Search for Connect6*; restated in the RZOP / TSS line) is:

> *A "threat" is the number of connections of five or four stones which can
> become checkmate only with one more move.*

**The decisive detail for Hexo/Connect6 is the two-stones-per-move rule**
(`TurnPhase::FirstStone` then `SecondStone`, `candidates.rs:414-418`;
`F_SIDE_IS_SECOND`, `constants.py:115-116`). Under it, "one more *move*" is **one
more two-stone move**, so a window that is **two stones short of six is still
one move from a win.** Concretely, on a length-6 window:

- **Count-5 (a "five"): 1 empty gap.** One placement fills it → six. An
  immediate winning threat.
- **Count-4 (a "four"): 2 empty gaps.** The mover fills **both gaps with the two
  placements of a single move** → six. **This is also an immediate winning
  threat.** Count-4 and count-5 are therefore *roughly equal danger* — both lose
  for the defender if left unanswered.
- **A threat = any active window with `count(player) >= 4`** (count-4 or
  count-5). There is no "non-forcing four" in this game.
- **Blocking a four or five usually costs the defender two placements**, because
  an open-ended four/five belongs to several overlapping 6-windows and filling a
  single gap does not neutralize all of its completions. So the defender's
  whole move can be consumed answering *one* ≥4 window.
- A **winning threat sequence** is a chain of forcing moves (each creating or
  extending a ≥4 window the opponent must answer) that ends in a position the
  defender's two stones cannot cover.

**Hexo maps onto this exactly, and the engine already encodes the corrected
definition.** Hexo's win is a **length-6 line** (engine `WINDOW_LEN = 6`,
`tactics.rs:14`) along one of **three hex axes** Q / R / QR (`tactics.rs::Axis`,
`tactics.rs:22-53`). The engine exposes the entire threat vocabulary on each
length-6 window: `count(player)` (`tactics.rs:134`), `empty_mask` /
`empty_cells` (`tactics.rs:144-156`), `is_win_for` (count == 6,
`tactics.rs:206-208`), `active_player` (one-color window, `tactics.rs:172-181`),
and — **crucially and in agreement with the corrected model** — a ready-made
threat predicate: **`is_threat` / `threat_player` / `is_threat_for` = an active
window with `count(player) >= 4`** (`tactics.rs:188-203`), plus window-pair
relations `intersects` / `touches` (`tactics.rs:210-222`). The store even
maintains live threat iterators `threat_entries(player)` and `threats()`
(`tactics.rs:386-395`). So the engine's own `threat` is precisely
"any ≥4 active window," matching the corrected model — **not** the
double-four-only notion the old draft used. **A TSS solver for Hexo needs almost
no new geometry** — the threat windows are already enumerated incrementally
(18 windows per placement, `tactics.rs:16-17`) and surfaced as WINDOW tokens in
the graph builder (`candidates.rs::window_tokens`, `candidates.rs:203-227`,
count-3/4/5).

## 1.2 Threat-Space Search, VCF, VCDT — the real method

**Threat-Space Search (TSS).** Instead of searching all legal moves, search
**only the forcing moves** — moves that create a threat the opponent *must*
answer. Because the opponent's replies to a threat are nearly forced (block the
threat), the branching factor collapses and the tree explores deep forcing
lines cheaply. TSS proves a **forced win** by finding a sequence of threats the
opponent cannot all parry.

**VCF (Victory by Continuous Fours).** The Gomoku/Renju form: a chain of
**fours** (each a one-move-from-win threat), where every opponent reply is the
forced block, ending in a position the defender cannot answer. In Hexo the
analog is **a chain of ≥4 windows**: each forcing move makes a fresh ≥4 window
(or extends one to count-5), the defender must spend their move neutralizing it,
and the attacker keeps the initiative until the defender's two stones can no
longer cover all live completions. **Unlike Gomoku, a single four already
forces here** (it is win-in-one-move), so the "continuous" chain is built from
*single* ≥4 threats, not only from doubles.

**Dependency-Based Search (DBS).** Wu & Kang's refinement (the
`Dependency-Based Search for Connect6` paper). The insight: threats in
**different regions of the board are independent**, and a brute force
threat-sequence search re-explores their interleavings combinatorially. DBS
splits the search into:

- a **dependency stage**, which builds the **dependency tree** of threats that
  *causally depend on each other* (a threat whose creation uses a stone or cell
  another threat produced — i.e. threats that share/extend the same line), and
- a **combination stage**, which **combines independent threat sub-trees** to
  find a winning conjunction (two independent threats that the opponent cannot
  answer in one turn).

This composes threats efficiently: solve each dependent chain once, then look
for **combinations of independent chains** that overwhelm the defender. (DBS
generalizes Allis's threat-space search + the "conflict / combination" idea to
Connect6's two-stone rule.)

**VCDT (Victory by Continuous Double-Threat-or-more).** The Connect6
forcing-search method. Wu et al. studied a winning strategy of **continuous
double-threat-or-more moves**:

> *Wu et al. showed a winning strategy called victory by continuous
> double-threat-or-more moves (VCDT), similar to victory by continuous four
> (VCF) … in Connect6, moves are classified as single-threat, double-threat,
> triple-threat, or non-threat; one player clearly wins by a
> triple-threat-or-more move.*

**How VCDT reconciles to the corrected model.** VCDT's "double/triple-threat"
classification is about **what *guarantees* an unstoppable continuation given
optimal defense**, not about what counts as a threat at all. A *single* ≥4
window already forces a response (it is win-in-one-move under the two-stone
rule), so it constrains the opponent — but the defender's *own* two stones can
usually parry one single threat and still develop. A move that creates **two
independent ≥4 windows at once** forces the defender to spend *both* stones, and
a **triple-threat-or-more** cannot be covered by two stones at all and wins
outright. So in VCDT terms: single ≥4 threats are the *forcing currency that
keeps initiative*; double/triple threats are the *terminating conjunctions that
end the chain*. This is the crux for the next subsection — but note the chain is
made of single forcing threats, and the win condition is the unanswerable
conjunction, **not** (as the old draft wrongly had it) that single fours fail to
force.

## 1.3 What two-stones-per-move actually does (the Hexo crux — corrected)

In **Gomoku** (one stone per move) a **four** is one cell short of a win, so it
is forcing; a **three** is two short and not forcing.

In **Connect6 / Hexo** the mover places **two stones per move** (Hexo:
`TurnPhase::FirstStone` then `SecondStone`, `candidates.rs:414-418`;
`F_SIDE_IS_SECOND`, `constants.py:115-116`). The two-stone rule does **not** make
single fours harmless — it does the opposite: it makes the *count-4* window a
**win-in-one-move** threat, because one move now completes *two* cells.
Consequences:

- **A single ≥4 window IS forcing.** A count-4 window has two empty gaps; the
  owner fills *both* with their next move and wins. A count-5 wins with one of
  the two placements. Either way, **an unanswered ≥4 window loses for the
  defender on the very next move.** The defender cannot ignore it. (This is the
  exact correction to the prior draft, which wrongly said a single four "is NOT
  forcing.")
- **Answering a ≥4 window typically costs the defender both stones.** An
  open-ended four/five lies on multiple overlapping 6-windows; filling one gap
  leaves another completion live, so the defender often must place **both**
  stones to neutralize a *single* threat — and then has nothing left to develop.
  So a single forcing threat already seizes the initiative.
- **A double threat (two independent ≥4 windows) forces both defender stones**
  and, if their gaps cannot all be covered by two placements, **wins outright.**
  A **triple-threat-or-more** can never be covered by two stones and is a win,
  modulo the defender having a strictly faster win of their own. This is the
  VCDT terminating condition.
- **Because single fours already force, the defensive problem is MORE acute,
  not less.** The old draft's "you need a double threat to force" understated
  the danger. The true statement: **ANY unanswered ≥4 window is a loss.** The
  defender must, every move, neutralize *every* opponent ≥4 window AND avoid
  walking into one — often spending their whole two-stone move just to survive.
- **Defender symmetry.** Both sides get two stones, so attacker and defender
  analysis are the same predicate read from each side: the attacker chains ≥4
  threats toward an unanswerable conjunction; the defender must answer **every**
  live opponent ≥4 window with their two stones and not already be lost. **This
  is exactly hexgt's blind spot:** the value head must recognize "the opponent
  has a ≥4 window (or a conjunction of them) I cannot fully parry with my two
  stones" — a multi-window relationship that §1.4 Gap C of the companion doc
  shows a 3-hop GNN + single-SIDE-hub readout under-propagates.

**Design implication.** A TSS for Hexo reasons about **≥4 windows across the 3
axes** as the forcing unit. The atom of *offense* is *a placement-pair that
creates or completes a ≥4 window* (completing a count-4 needs the pair — both
gaps; completing a count-5 needs one). The atom of *defense* is *can the
defender's 2 placements cover the empty gaps of every live opponent ≥4 window?*
— a small **set-cover over the threats' empty cells**, where two threats are
independent when their gaps differ (use `WindowKey::intersects`/`touches`,
`tactics.rs:210-222`). All the enumeration primitives already exist in
`tactics.rs` (`count >= 4` threat windows via `threat_player`/`is_threat_for`,
their `empty_cells`, the pairwise relations) and are surfaced as window tokens
with their `empty_cells` in `candidates.rs:203-227`.

## 1.4 How hexgt would do TSS — three options, evaluated

All three reuse the **existing window walk** — there is no new board scan. The
windows are already enumerated by the engine (`tactics.rs`) and materialized as
WINDOW tokens (`candidates.rs::window_tokens`, `candidates.rs:203-227`) with
`count`, `empty_count`, and the actual `empty_cells`/`stone_cells` per window.
The candidate↔window edge loop in the featurizer (`features.rs:131-182`) already
visits exactly the (candidate, window, owner, count) tuples a threat detector
needs — and a CANDIDATE node is joined to a WINDOW node by an
`EDGE_CANDIDATE_WINDOW` edge **precisely when that candidate is an empty cell of
that window** (`candidates.rs:354-358`). So "the empty cells inside an active ≥4
window" are *already* an explicit, cheap relation in the graph.

### The TSS candidate set (corrected)

**TSS candidate set = the empty cells inside ANY active count-4 or count-5
window** (i.e. the `empty_cells` of every window token with `count >= 4`). Those
are exactly the cells that (offense) complete a threat or (defense) block one,
so TSS expands only these forcing/defensive cells instead of all legal moves.
Two facts make this nearly free in hexgt:

1. The candidate builder **already seeds candidates from active-window empties**
   (`candidate_cells` walks `entry.empty_cells()` for active windows,
   `candidates.rs:162`), so the threat cells are present as candidate nodes.
2. The `EDGE_CANDIDATE_WINDOW` join (`candidates.rs:354-358`) lets a detector
   recover, for each candidate, the set of ≥4 windows it sits in — the offense
   side — and for each ≥4 opponent window, the empty cells that must be covered
   — the defense side — by one pass over those edges.

Restricting/augmenting expansion to this set whenever threats are present
guarantees defensive lines are always searched (the search can never "miss" the
move that answers a ≥4 window).

### Shared data structures (cheap, grounded)

```
Threat       = { owner, axis, window_key, count∈{4,5}, gaps: SmallVec<HexCoord> }
                 // gaps = the window's empty cells (from WindowEntry::empty_cells)
                 //   count==5 → 1 gap (one placement completes the six)
                 //   count==4 → 2 gaps (the two placements of ONE move complete it)
                 // BOTH counts are immediate winning threats under the 2-stone rule.
ThreatMap    = per owner: Vec<Threat>, plus an index cell→[threat ids] (which
                 ≥4 windows a given empty cell belongs to — the "must-cover"
                 pivot for both completion and blocking).
```

Building `ThreatMap` is **O(active windows)** with the fields already on each
WindowToken (`count`, `empty_cells`). Cell/move classifications under the
corrected model:

- **own-win-now (move):** the side to move can WIN THIS MOVE if it owns a
  `count==5` window (fill its 1 gap with one of the two placements) **or** a
  `count==4` window (fill BOTH gaps with the move's two placements). The latter
  is the case the engine/feature layer does **not** currently flag (see the code
  note below) and the new detector adds.
- **opp-win-next (must-answer):** the opponent owns a ≥4 window. The defender's
  move must cover its gap(s) (1 gap for count-5, 2 for count-4) or lose next
  move. Whether the defender's **two** placements can cover the gaps of **all**
  live opponent ≥4 windows is a small set-cover; if not, the position is a
  **forced loss for the side to move.**
- **double/triple threat:** an owner has ≥2 *independent* ≥4 windows (distinct
  gap sets; confirm independence with `WindowKey::intersects`/`touches`,
  `tactics.rs:210-222`). Two stones cover at most a bounded gap set, so a
  triple-threat-or-more is unanswerable.

**Code note (correction to the old draft).** The old draft equated the existing
`F_CAND_COMPLETE_OPP`/`F_CAND_COMPLETE_OWN` features with "win-now / must-block."
In the actual code those flags are set **only when `cnt == 5`**
(`features.rs:150-152` for own, `features.rs:162-163` for opp;
`constants.py:83-84` document them as "completes a **count-5** window"). Under
the corrected model a **count-4 window is also win-now** (with two placements),
and there is currently **no feature flagging the count-4 completion** — the v2
slots `F_CAND_OWN_WIN4`/`F_CAND_OPP_WIN4` (`features.rs:146,158`;
`constants.py:108,111`) merely *count* count-4 windows through a cell, they do
not mark "completable this move." This gap is precisely what Option (b)/(c)
below should fill, and it is one reason the value head is blind to count-4
losses.

### Option (a) — TSS/VCF tactical solver inside MCTS (root and/or nodes)

A bounded forcing-move solver that, at a node, reads the `ThreatMap` for both
sides and **overrides the net** when a depth-1 proof exists:

- **HARD WIN (+1) — checked FIRST, since the side to move moves first:** the
  side to move owns any ≥4 window it can complete this move — a `count==5` window
  (one placement fills the gap) **or** a `count==4` window (the move's two
  placements fill both gaps). It simply plays the completing move and wins before
  the opponent replies. (No double-threat is required — a single ≥4 window the
  mover owns is already a win on its own move.) Set node value **+1**. This
  dominates any opponent threat, because the mover acts first.
- **HARD LOSS (−1) — only when there is NO own win:** the side to move cannot
  win this move, AND the opponent has one or more ≥4 windows whose gaps the side
  to move **cannot all cover with its two placements** (a tiny set-cover over the
  opponent's threat gaps; trivially true if the opponent has a
  triple-threat-or-more, or two independent fours/fives whose gaps exceed two
  cells). Then the opponent wins on its next move regardless — set node value
  **−1**. This is the **direct cure for "value ≈ +0.8 right before losing."**
- **MUST-ANSWER (no override, mark only):** the opponent has ≥4 window(s) the
  side to move *can* still cover with its two placements (e.g. a single count-5
  whose one gap is coverable, or a single count-4 whose two gaps the move can
  fill). Not auto-losing, but the solver marks the covering cell(s) so PUCT
  prioritizes the forced defense and the search does not wander.

Note the asymmetry vs. the old draft: a HARD WIN needs only a *single* own ≥4
window (because a four is win-in-one-move here), and a HARD LOSS triggers
whenever the opponent's ≥4 threats *out-run* the defender's two stones — which,
because single fours force, happens far more often than the old "unstoppable
double/triple only" rule admitted.

**Where it hooks (grounded).** The MCTS leaf/backup is in `mcts.rs`:
`select_leaf_batch` (`mcts.rs:511-571`) classifies each selected leaf as
terminal / existing-node / needs-eval. For an engine-terminal leaf it computes
`terminal_value(outcome, leaf_player)` (imported from `mcts_tree`, `mcts.rs:33`)
and backs it up via `search.backup_virtual(&selected.path, leaf_player,
leaf_value, virtual_loss)` (`mcts.rs:538-541`); otherwise it pushes the leaf for
a network eval (`mcts.rs:545-555`). Crucially, the full leaf `state` is in hand
at this point — line 539 already calls `selected.state.current_player()`, so
`selected.state.board().windows()` (the `WindowStore`, with `threat_entries` /
`threats` and per-window `count`/`empty_cells`) is directly available to compute
the depth-1 threat check. **A TSS override is "a one-ply-lookahead terminal":**
in `select_leaf_batch`, *before* the needs-eval branch, run the depth-1 threat
probe on `selected.state`; if it proves a HARD WIN/LOSS, back up ±1 via the
existing `backup_virtual` path (`mcts.rs:541`) and **skip the net eval entirely**
(do not push to `leaves`). That is the minimal, surgical hook — it reuses the
terminal-leaf code path and the cache stays correct (a proven node is as good as
terminal). The "side to move can complete a ≥4 window now" check is computable
from exactly what the leaf already exposes (the window store), so no extra state
plumbing is needed.

**Compute cost.** The *depth-1* probe is cheap — one pass over the active
windows to read each side's ≥4 windows and run the two-stone set-cover, O(active
windows) per leaf, far cheaper than a GNN forward. A *deep* VCF/VCDT proof (the
full forcing-sequence search that chains threats over multiple plies) is the
expensive classic algorithm and would dominate leaf cost; it should be
**root-only and depth-bounded** if used at all. Recommendation: **depth-1
(immediate win/loss) at every leaf, optional depth-bounded forcing-move TSS at
the root only.**

**Pure-self-play interaction (important caveat).** A hard value override
**changes the training target distribution**: leaves the solver proves never
get a *network* value, and proven-loss nodes inject −1 where the net said +0.8.
That is the point — but it interacts with two existing mechanisms:
1. **Forced playouts (`forced_playout_k`, default 0.0 in config but used at 2.0
   in the RL run, `_rl_train.py`/`config.py:137`).** Forced playouts already
   inflate visits on under-explored moves; a hard must-block/forced-win override
   on top of that **over-determines** the visit distribution, which is exported
   as the **policy training target** (`pruned_visit_policy`, `mcts.rs:875-915`).
   Stacking both risks teaching a near-one-hot policy on tactical positions.
   Mitigation: keep the override on **value/selection** but let the **exported
   policy** stay the (forced-playout-pruned) visit policy — i.e. do **not** also
   hard-collapse the policy target to the proven move; let visits concentrate
   naturally. The existing KataGo policy-target pruning (`mcts.rs:865-915`)
   already removes the *forced* tail, so the proven move's visits remain a strong
   but soft target.
2. **Decisiveness / shortening.** Games are already 100% decisive and getting
   shorter; an aggressive solver could shorten further (it ends games the moment
   a forced win appears). That is *correct* play but reduces positional
   diversity. Pair with the opening-temperature / diversity levers already in
   self-play (`selfplay.py` opening logging) and judge by H2H, not game length.

**D6-safety.** Threats are pure board geometry; D6 maps windows→windows,
owners→owners, threat-gaps→images bijectively (the same argument
`constants.py:103-106` makes for the v2 window-count features and §5.5 of the
companion doc makes for hot cells). A ±1 override is a **scalar**, D6-invariant.
**Safe** — the solver does not touch the D6-invariant feature/graph
construction.

### Option (b) — TSS as a candidate/feature generator ("hot tokens")

Run the **depth-1 ≥4-threat detector** and emit, on CANDIDATE nodes, the
**complete-now-own / must-answer-opp** flags in the **reserved slots `[30:32)`**
(`constants.py:117`). The point of these two slots is to fix the count-4 gap
identified above: a candidate gets **complete-now-own = 1** when it sits in an
own ≥4 window the side to move can finish this move (count-5: that cell is the
gap; count-4: this cell is one of the two gaps the move fills), and
**must-answer-opp = 1** when it sits in (or covers the gaps of) an opponent ≥4
window — covering **both** count-4 and count-5 opponent threats, where the
existing `F_CAND_COMPLETE_OPP` only fires on count-5 (`features.rs:162-163`).
This is exactly the companion doc's §5 "hot tokens (i)" + its §6.3
reconciliation that hot tokens are "the neural-network-side analog of TSS/VCF."
It feeds the net richer tactical inputs without changing the search.

**Where it hooks.** `features.rs` candidate↔window loop (`features.rs:131-182`),
writing two new columns; `constants.rs`/`constants.py` slot names + a
`NEW_FEATURE_SLOTS_V3` tuple + `FEATURE_SCHEMA_VERSION` 2→3; the byte-parity
test (`tests/test_hexgt_feature_buffer.py`, named in `features.rs:11`).

**Cost.** Cheap (rides the existing loop). **D6-safe** (D6-invariant flags).
**Drop-in onto the live checkpoint with NO cold start** via the proven zero-init
layer-expansion (`architecture.py::zero_init_expanded_feature_columns`; the v2
slots used the same path, `constants.py:123-129`).

**Limitation.** As §5.2(i) of the companion doc notes, an opponent-hot
*candidate* feature still has to propagate candidate→window→SIDE to reach the
**value** head — the thin path. So (b) alone strengthens the **policy's**
must-block reflex but does **not** structurally guarantee the value head sees
the threat. That guarantee needs the readout fix (Rank 1 / §5.2-iii) or the
hard value override (a).

### Option (c) — lightweight VCF/VCDT check to set graded threat features

A middle form: run the depth-1 detector and set the **graded "forcing" / open-
ends features** the companion doc's Rank 2 already proposes (window `open_ends ∈
{0,1,2}`, a `forcing` flag, candidate `n_opp_winning_replies`), *plus* the
binary hot flags of (b). Under the corrected model the `forcing` flag fires for
**any ≥4 window** (count-4 and count-5 alike, since both are win-in-one-move),
and `n_opp_winning_replies` counts the opponent ≥4 windows whose gaps the
defender's two stones would fail to cover — i.e. graded "how close to a forced
loss." Same hooks as (b) (`features.rs` + constants + parity test), same
D6-safety and drop-in story. It is (b) with continuous severity added, which the
companion doc Rank 2 argues helps mid-game calibration.

### Recommendation for Part 1

**Start with (b)+(c) as one feature pass, and add the (a) override only at the
LEAF level for depth-1 forced-win/forced-loss, root-only for deeper VCDT.**
Rationale:

- (b)+(c) is **cheap, D6-safe, live-graftable with no cold start**, reuses the
  existing window walk, and is the lowest-risk way to hand the net the tactical
  signal it lacks. It is the same intervention the companion doc already ranked
  just-below-Rank-1.
- The **depth-1 leaf override (a)** is the *only* option that gives a **hard
  guarantee** against "value +0.8 right before a forced loss," because it
  replaces the net's value with a proof. It is a small, surgical hook in
  `select_leaf_batch` reusing the terminal-value path. Keep it **depth-1** to
  avoid leaf-cost blowup, and **do not hard-collapse the policy target** (only
  the value/selection) to avoid distorting the KL/visit policy under forced
  playouts.
- **Defer the full deep VCF/VCDT in-tree solver.** It is the classic strength
  multiplier but is the heaviest change (Rust MCTS, depth/breadth bookkeeping,
  set-cover over winning cells for the two-stone defender), it most disturbs the
  visit/policy targets, and its ceiling is only worth chasing if (a)-depth-1 +
  (b)/(c) + the value fixes don't move the defensive-calibration metric. Make it
  **root-only and depth-bounded** if pursued.

---

# PART 2 — Softer value targets, in depth

## 2.1 The paper: soft-Z, A0C, A0GB (Willemsen, Baier, Kaisers, NCAA 2022)

*"Value targets in off-policy AlphaZero: a new greedy backup"* (DOI
10.1007/s00521-021-05928-5). **Full text fetched and read** (PDF extracted
locally). It defines a **family of value targets** parameterized by *how far you
bootstrap in two orthogonal directions* — the **real self-play game** (`n_real`
steps under the AlphaZero behavioural policy) and the **simulated MCTS tree**
(`n_sim` steps under the greedy MCTS policy) — plus a **backup-width** choice
(back up the network estimate `v̂_NN` vs the subtree MCTS estimate `v̂_MCTS`).
The unified target (paper Eq. 8) is:

```
y_target(s) = v̂_MCTS( π_MCTS,greedy^{n_sim}(s_root) , s_root )
   where     s_root = π_AlphaZero^{n_real}(s)
```

The four named members (exact, as quoted from the paper):

- **AlphaZero (original):**
  `y_AlphaZero(s) = v̂_MCTS(s_terminal, s_terminal) = r(s_terminal)` —
  i.e. the **final game outcome** reached by following the (exploratory)
  AlphaZero policy to the end. `n_real = ∞`, no simulated bootstrap. *Biased by
  the exploratory moves played for the rest of the game*; high variance (it is a
  Monte-Carlo return).
- **soft-Z:** `y_soft-Z(s) = v̂_MCTS(s, s)` — the **MCTS root value at the
  position itself** (`n_real = 0`, `n_sim = 0`; back up the whole-subtree
  estimate). This is the *search's own value estimate at s*. **Note: in the
  paper, soft-Z is purely the MCTS value — NOT a convex blend with z.** The
  convex-blend form is a practical variant (below). soft-Z has the **highest
  bias but the lowest variance** of the three alternatives.
- **A0C** (Moerland): `y_A0C(s) = v̂_MCTS(π_MCTS,greedy^1(s), s)` — take **one
  greedy step** in the tree, back up that child's MCTS value. Bootstraps one
  more level than soft-Z; **bias and variance both in between** soft-Z and A0GB.
- **A0GB** (the paper's contribution): `y_A0GB(s) = v̂_MCTS(π_MCTS,greedy^{K−1}(s),
  s) = v̂_NN(π_MCTS,greedy^{K−1}(s), s)` — follow the **greedy MCTS policy all the
  way to a leaf/terminal** and use that leaf's value (which equals the network
  value `v̂_NN` because a leaf has a single visit). It removes *all* exploration
  from the target, so the target policy is greedy (closest to the optimal
  greedy policy). **Lowest bias, highest variance.**

**Results.** All three alternatives (**soft-Z, A0C, A0GB**) achieve **better
performance and faster training** than the original AlphaZero hard-outcome target
on **Connect-Four** and **Breakthrough (6×6)** (and A0GB uniquely converges to
the optimal policy in a tabular Tic-Tac-Toe where the hard target fails). The
paper's mechanistic argument is precisely the one that bites hexgt: the **final
outcome is a high-variance, off-policy-noisy label** — the rest of the game was
played by a different (older, exploration-noisy) policy, so `z` is only loosely
about *this* position — whereas the **MCTS value at the position is a
lower-variance, on-position estimate** that already integrated the search's
findings.

## 2.2 Tie to KataGo and to the +0.82 optimism bias

**KataGo (Wu 2019, arXiv 1902.10565)** is the project's design baseline. Its
value handling, relevant here: a relatively **high value-loss weight** (≈1.5×
the policy in the headline config) and a battery of **auxiliary value-flavored
targets** (short-term value, ownership, score-distribution) that regularize the
trunk and force *localized* value learning. hexgt already imports the
short-term-value idea (`STV`, `samples.py:337-369`) — which is **literally soft-Z
applied to auxiliary heads at horizons (4,12,24)**: an EMA of future
`root_value`. So the project has *already validated, on its own pipeline*, that
training a value head on the search's `root_value` is stable and learns
(memory: "BC-from-96x8 validated, loss drops"; STV weight tuned and trained).

**Why this attacks the +0.82 bias mechanically.** The hard `z` label is **±1**
for *every* position in a won/lost game, regardless of how close that position
actually was. Two failure modes follow:

1. **Magnitude saturation.** The value head is only ever shown **±1** (and 0 for
   draws). It never sees the *graded* magnitudes of a position that is "+0.3
   ahead" or "−0.6 behind." Pushed to the ±1 extremes by every label, its softmax
   over the 65 bins collapses toward the end bins → **systematically
   over-confident**, exactly the "+0.8 right before losing." soft-Z replaces ±1
   with the search's `root_value ∈ (−1,1)`, which for a near-lost position is
   *already* a non-saturated, more-honest number — so the head is **trained on
   the full magnitude range** and stops pinning at the extremes.
2. **Anti-calibration / non-zero-sum sum.** The +0.82 `v(A)+v(B)` sum means the
   value is **not respecting the zero-sum structure** — both sides are taught
   they're winning because *both* eventually-won-or-lost-game labels are
   over-confident in their own favor relative to the true position. The MCTS
   `root_value` is, by construction, an **estimate of the same scalar from the
   side-to-move's perspective**, and it is the search's *integrated* estimate
   (it saw the refutation lines). Blending the label toward it pulls **both**
   sides' targets toward the search's more-calibrated, closer-to-zero-sum
   estimate, **shrinking the sum toward 0.** It does not *guarantee* zero-sum
   (the net still must learn the symmetry), but it stops *teaching* the
   anti-calibration that the hard label injects.

In short: **soft-Z softens the over-confident ±1 label toward the search's own
more-calibrated estimate AND exposes the value head to non-terminal magnitudes**
— the two mechanisms that the +0.82 / +0.8 symptoms call for.

## 2.3 Concrete hexgt implementation

### Where the soft target is constructed (grounded)

**One place: the main-value assignment in
`dense_cnn/.../samples.py::finalize_game_samples`** (`samples.py:198`,
`value=_winner_value(winner, player)`). This is the file hexgt's self-play calls
(`hexgt/.../selfplay.py:462`). Everything needed is already in scope:

- `winner` and `player` (hard outcome via `_winner_value`, `samples.py:318-321`).
- the per-decision `root_value`, currently bound as `_root_value` and **discarded
  by the main-value loop** (`samples.py:185`) — it is the MCTS root estimate from
  *that* position, already what STV consumes (`samples.py:351-356`).

**Perspective.** `root_value` (`search.root_value`, `mcts.rs:677`
`result.set_item("root_value", root.value())`) is **from the side to move at
that decision** — the same convention STV relies on
(`samples.py:354-356` flips future values to the decision's perspective; *this*
position's own `root_value` is already in the decision player's perspective). So
no sign flip is needed for the position's own value: `_winner_value(winner,
player)` and `root_value` are both in `player`'s perspective. (Confirm with a
unit test asserting `decode_binned_value(forward(s))` and the stored
`root_value` share sign on a known-winning position before relying on this.)

### The exact blending formula (recommended: convex soft-Z)

The paper's pure soft-Z is `y = root_value`. For a first, low-risk hexgt change
that keeps the hard outcome as an anchor (and is robustly defined even when
`root_value` is itself early-RL-noisy), use the **convex blend** the companion
doc's Rank 0a proposed:

```
value_target(decision) = (1 − λ) · _winner_value(winner, player)
                        +     λ  · root_value_in_player_perspective
```

- `λ = 0` → today's hard outcome.
- `λ = 1` → the paper's pure soft-Z.
- **Recommended start: λ = 0.5.** Justification: (1) the paper found soft-Z (the
  λ→1 extreme) already strictly better than hard-z, so a substantial λ is
  warranted; (2) hexgt's *own* STV heads already train on `root_value` and are
  stable, so `root_value` is a trustworthy signal here; (3) but the very disease
  is that early-RL `root_value` is *itself* miscalibrated, so keeping a 0.5
  anchor on the **ground-truth outcome** prevents a soft target from
  re-teaching its own error. **Anneal λ from ~0.3 → ~0.7 as calibration
  improves** (measured by the §2.5 metrics) if 0.5 helps but plateaus. This is
  a config scalar; no schema or head change. (For a draw/truncation,
  `_winner_value` is 0 and `root_value` carries the search's estimate, which is
  *more* informative than a hard 0 — a free side-benefit on the
  `max_actions_draw` rows, `samples.py:193-194`.)

**The 65-bin mapping is automatic and requires no new code.** The blended scalar
is still in `[−1,1]`, so it flows through the **existing**
`scalar_to_binned_target` (`losses.py:40-60`), which splits it across the two
adjacent bins (`position = (v+1)·(64/2)`, floor/ceil, linear weights) — the
**same binning dense_cnn uses, confirmed verbatim** (`losses.py:1-8` docstring;
`VALUE_BINS = 65`, `constants.py:16`). `binned_value_loss` (`losses.py:63-94`)
is unchanged. So **the soft scalar maps to bins via the existing scalar→bin
code; no new value-head plumbing.**

### Greedy-backup (A0GB) variant — deferred

A0GB would set `value_target = v̂_NN(greedy-leaf)`, i.e. follow `π_MCTS,greedy`
down the *already-built* search tree to a leaf and use that node's value. hexgt
**has the tree** at decision time, but it is **discarded** after the move
(`mcts.rs` advances/promotes the root subtree; the per-decision value exported
to Python is only the scalar `root_value`, `mcts.rs:677`). Exposing a greedy-leaf
value would require the Rust search to walk `π_greedy` to a leaf and export that
scalar — a small but real Rust change. **Defer A0GB**: convex soft-Z captures
most of the benefit with **zero Rust change** (it reuses the already-exported
`root_value`), and the paper shows soft-Z alone already beats hard-z. Revisit
A0GB only if soft-Z's residual bias (it is the highest-bias member) caps the
gain.

### Interaction with the STV heads

soft-Z (main value) and STV (aux heads) are **the same idea on different heads**
and compose cleanly:

- STV trains horizons (4,12,24) on an **EMA of future** `root_value`
  (`samples.py:337-369`); soft-Z gives the **main** head a **horizon-0**
  bootstrap (this position's own `root_value`). No conflict; soft-Z is
  effectively "STV horizon 0 folded into the main value."
- Keep STV's small weight (`short_term_value_weight`, 0.10–0.25,
  `config.py:83`); do **not** also blend the STV *targets* with the hard outcome
  — they are already pure bootstraps and deliberately weighted low.
- One watch-item: if both the main value (now partly bootstrapped) and STV
  bootstrap from the *same possibly-miscalibrated* `root_value`, an early-RL
  feedback loop is conceivable. The 0.5 hard-outcome anchor on the **main** head
  is the circuit-breaker; STV stays small. Monitor the §2.5 metrics for
  divergence.

### Interaction with 0.9/epoch recency weighting

Independent and complementary. `epoch_recency_weight` (`_rl_train.py:121-122`)
weights **which epochs** are sampled; soft-Z changes **what scalar each row's
value target is**. They compose without conflict — recent rows (heavily sampled)
will carry the recalibrated target, so the value head re-learns calibration
fastest on the freshest self-play, which is what we want. No change to the
replay window code (`build_replay_window`, `_rl_train.py:362`) is needed.

## 2.4 Why soft-Z is mechanically the right first value fix (summary)

1. **Softens the saturating ±1 label** → trains the 65-bin head on graded
   magnitudes → directly counters "+0.8 right before losing."
2. **Pulls both sides' targets toward the search's integrated estimate** →
   shrinks the +0.82 `v(A)+v(B)` sum toward zero-sum.
3. **Validated** (paper: faster + stronger on Connect-Four / Breakthrough) and
   **already proven on this exact pipeline** via the STV heads.
4. **Trivial, D6-safe, fully drop-in:** a few lines in one function + a config
   scalar; the scalar→bin mapping and loss are unchanged; weights load
   identically onto the live checkpoint with **no cold start** (it changes only
   the *target a future epoch trains toward*).

## 2.5 Validation plan (the metrics this is for)

Run all three before/after on held-out self-play; judge by these, **not** train
loss (a documented measurement-artifact pitfall in this project — the dense_cnn
"rising loss" was an artifact; judge by eval/calibration):

1. **Same-board `v(A)+v(B)` sum (primary).** Re-measure the optimism bias on a
   fixed probe set of boards, evaluating the value from *both* sides. soft-Z
   should move the sum from **≈ +0.82 toward 0**. This is the most direct
   readout of the fix.
2. **Value-head calibration on defensive / opponent-hot positions.** Slice
   held-out self-play to positions where the opponent has a four/five-threat
   (label cheaply with the Part-1 detector). Measure value CE/Brier vs realized
   outcome and the **reliability curve** (predicted vs realized win-rate by
   confidence bucket). soft-Z should reduce over-confidence specifically on this
   slice.
3. **The 8/8-lost-game value trace.** Re-run the forensic probe on the known
   lost games: the value prediction in the final plies should **drop toward the
   loss** instead of pinning near +0.8. The traces are already logged
   (`selfplay.py:425-440` records per-move `root_value`; `_write_game_record`
   persists the game).
4. **H2H** vs dense_cnn e24 via the existing `run_head_to_head` (player.py /
   evaluation.py) as the integrative judge.
5. **λ ablation.** Sweep λ ∈ {0, 0.3, 0.5, 0.7} for a few RL epochs from the
   live checkpoint (no cold start) and pick by metrics (1)–(4).

---

# Synthesis — recommendation & validation order

**Headline.** Two complementary value-defense fixes, both inside pure self-play:

- **soft value target = the first value-calibration fix** (companion doc's Rank
  0a). Convex soft-Z: `value = (1−λ)·z + λ·root_value`, **start λ = 0.5**
  (anneal 0.3→0.7), constructed in `samples.py::finalize_game_samples`, mapped
  to the existing 65-bin head via the unchanged `scalar_to_binned_target`. It is
  the **highest-leverage-to-risk single change in either doc**: a few lines + a
  config scalar, D6-safe, drop-in onto the live checkpoint with no cold start,
  externally validated and already proven on this pipeline via STV. **Do it
  first.**
- **TSS for hexgt = the cheap detector + a depth-1 leaf override, full solver
  deferred.** The forcing unit is **any active ≥4 window** (single fours force
  here — two placements complete a count-4), and the TSS candidate set is **the
  empty cells inside active ≥4 windows**. Land **(b)/(c)**: the depth-1
  ≥4-threat detector emitting **complete-now-own / must-answer-opp** (including
  the currently-missing count-4 completion flag) + graded-forcing CANDIDATE
  features in the reserved slots `[30:32)` (rides the proven zero-init
  feature-expansion, no cold start, D6-safe, reuses the existing window walk in
  `features.rs`). Then add the **depth-1 forced-win/forced-loss value override at
  the MCTS leaf** (`select_leaf_batch`, reusing the terminal-value backup path):
  HARD WIN if the side to move can complete a ≥4 window this move, else HARD LOSS
  if its two placements cannot cover every opponent ≥4-window gap — the only
  thing that *hard-guarantees* the value head can't be "+0.8 before a forced
  loss." Keep the override on value/selection only (don't hard-collapse the
  policy target) to avoid distorting the visit/KL policy under forced playouts.
  **Defer** the full deep in-tree VCF/VCDT solver (root-only, depth-bounded if
  ever) — highest ceiling, highest cost, most target distortion; pursue only if
  the cheaper fixes don't move the defensive-calibration metric.

**How these rank among the existing fixes** (companion doc): soft-Z stays
**Rank 0a (try first)**. The TSS detector features are the **§5 "hot tokens"
(i)** intervention with TSS vocabulary — just below Rank 1, folded into it via
the readout pool (iii-a). The depth-1 leaf override is a **new, sharper member**
of the same family: it is the *search-side* guarantee that the feature/readout
fixes only make *likely*. The deep in-tree solver is the companion doc's §6.3
**deferred search-side option**, unchanged in rank.

**Validation order (cheapest-to-validate first):**

1. Establish the **`v(A)+v(B)` sum** and **opponent-hot calibration** metrics
   (§2.5) as the standing defensive-calibration harness.
2. **soft-Z λ-sweep** from the live checkpoint (no cold start, fastest A/B).
   Confirm the sum moves toward 0 and the 8/8-trace stops pinning at +0.8.
3. **TSS detector features (b)/(c)** — zero-init graft, re-measure the same
   metrics + H2H.
4. **Depth-1 leaf override (a)** — re-measure; specifically confirm proven-loss
   nodes no longer carry +0.8 net values, and watch game-length/diversity and
   the policy-target sharpness (don't let forced playouts + override over-determine
   the policy).
5. Only if defense still lags: **deep root-bounded VCDT** and/or the readout/
   ownership fixes from the companion doc.

**Honest validated-vs-speculative ledger.**

- **Validated externally:** soft-Z/A0C/A0GB beat hard-z (Connect-Four,
  Breakthrough; A0GB optimal in tabular) — paper full-text read. TSS/VCF/VCDT as
  the domain-standard way to handle forcing threats — confirmed from multiple
  Connect6 sources. The **corrected** two-stone crux — that a single ≥4 window is
  already a win-in-one-move (count-4 completed by the two placements of one move)
  so any unanswered ≥4 window forces — is confirmed by the engine's own threat
  predicate `count(player) >= 4` (`tactics.rs:188-203`) and supersedes the
  earlier draft's incorrect "only a double-four forces" reading of the Connect6
  papers.
- **Validated on this pipeline:** training a value head on `root_value` is
  stable and learns (the STV heads already do it).
- **Grounded in code (this pass):** every hexgt hook cited above — the hard
  value target (`samples.py:198`), `root_value` in hand at finalize
  (`selfplay.py:424`, `samples.py:185`), the 65-bin head & verbatim binning
  (`losses.py`, `constants.py:16`), the leaf classification/backup path
  (`mcts.rs:511-571`, leaf has full `state`/`WindowStore` access at lines
  538-541), forced playouts & policy-target pruning (`pruned_visit_policy`
  `mcts.rs:875`+), the window/threat primitives — the `count >= 4` threat
  predicate (`tactics.rs:188-203`), `empty_cells`/`count` (`tactics.rs:134-156`),
  window-pair `intersects`/`touches` (`tactics.rs:210-222`), window tokens with
  `empty_cells` (`candidates.rs:203-227`), the candidate↔window-empty-cell edge
  (`candidates.rs:354-358`), the candidate↔window feature loop with the
  count-5-only completion flags (`features.rs:131-182`, esp. 150-163) — the
  reserved feature slots & zero-init graft (`constants.py:117-129`).
- **Speculative (hypotheses to validate, not yet measured):** that soft-Z
  *will* move the +0.82 sum toward 0 and fix the 8/8 trace by the quantitative
  amounts hoped; that λ=0.5 is the right start (sweep it); that the depth-1
  override improves H2H net of the diversity cost; that a deep VCDT solver's
  ceiling is worth its cost in this pure-self-play regime. The *mechanism*
  arguments (saturation, anti-calibration, zero-sum shrinkage, blindness to
  unanswered ≥4 windows) are reasoned and code-grounded; the *magnitudes* are
  not yet measured.

---

## Sources — fetched vs. reasoned-about

- **FULL TEXT READ:** Willemsen, Baier, Kaisers, *"Value targets in off-policy
  AlphaZero: a new greedy backup,"* Neural Computing and Applications 34 (2022)
  1801–1814, DOI 10.1007/s00521-021-05928-5 — PDF fetched from the TU/e
  repository and text-extracted; all formulas in §2.1 are quoted from it
  (Eqs. for soft-Z/A0C/A0GB and the unified Eq. 8, the bias-variance ordering,
  and the Connect-Four/Breakthrough result).
- **PARTIAL (abstract / search-snippet level, full text paywalled — reasoned
  about the well-documented method, flagged):**
  - Wu & Kang, *"Dependency-Based Search for Connect6"* (Springer LNCS;
    Semantic Scholar 5e4ad49c…) — ResearchGate/Springer full text returned
    403/401. The dependency-stage/combination-stage split, the threat
    definition, and VCDT are reconstructed from the abstract + the
    Connect6/RZOP literature snippets and the chessprogramming wiki.
  - Chessprogramming wiki *Connect6* page — fetched (rules + a list of
    techniques: VCF, MCTS/UCT, PNS+RZOP); did **not** contain the detailed TSS
    mechanics, so those were sourced from the search snippets and reasoned.
  - Princeton thesis *"Playing Connect6 With Threat Space Search And Temporal
    Difference Learning"* (dataspace 88435/dsp016t053k07c) — repository returned
    HTTP 401; **not read**; referenced by title only.
  - *"Deep learning approaches to the game of Connect6"* (ScienceDirect
    S1875952124000752) — HTTP 403; **not read**; referenced by title only.
  - Web-search snippets (IEEE/ResearchGate titles + abstracts) supplied the
    precise threat definition ("connections of five or four stones … one more
    move"), the single/double/triple-threat classification, the "triple-threat
    wins" claim, and VCDT/VCDTS naming. These corroborate the reasoned TSS/VCDT
    account but the underlying full papers were not all retrievable.
- **CODE (read this pass, hexgt worktree `E:\Hexo-BotTrainer-hexgt`, read-only):**
  `tactics.rs`, `candidates.rs`, `features.rs`, `mcts.rs`, `constants.py`,
  `losses.py`, `selfplay.py`, `dense_cnn/.../samples.py`, `_rl_train.py`,
  plus the prior `HEXGT_ARCH_DESIGN_EXPLORATION.md` for grounding.
- **KataGo (Wu 2019, arXiv 1902.10565):** referenced from prior knowledge /
  the companion doc's §6.2 (value-loss weight, auxiliary ownership/score/STV
  targets); not re-fetched this pass.
