# hexgt: Threat-Space Search & Softer Value Targets — Finalized Design

Design / analysis only. **No code, config, or model files are changed by this
document, and nothing here runs training or touches the live dense_cnn run.**
This is the *finalized* plan for two of the recommendations ranked in the
companion doc
[`HEXGT_ARCH_DESIGN_EXPLORATION.md`](HEXGT_ARCH_DESIGN_EXPLORATION.md) —
**Threat-Space Search (TSS)** and **soft value targets**. It supersedes the
earlier draft of this file: every mechanical claim below has been **empirically
verified against the real `hexo_engine`** (harness:
[`scripts/_tss_verify.py`](../../scripts/_tss_verify.py),
[`scripts/_tss_verify_c.py`](../../scripts/_tss_verify_c.py),
[`scripts/_tss_verify2.py`](../../scripts/_tss_verify2.py)) and against the hexgt
MCTS source. The independent verification that drove this revision is recorded in
[`HEXGT_TSS_VERIFICATION.md`](HEXGT_TSS_VERIFICATION.md); the per-claim evidence is
summarized in the **Verification log** at the end of this doc.

Engine source lives in the hexgt worktree `E:\Hexo-BotTrainer-hexgt`. **Note:** the
window/threat primitives are in the shared **`hexo_engine`** crate
(`packages/hexo_engine/rust/src/tactics.rs`), *not* the hexgt model crate; the
graph/feature/MCTS code is in `packages/hexo_models/hexgt/rust/src/`.

> ## HEADLINE (finalized)
>
> **Threat model (authoritative, engine-confirmed).** Hexo/Connect6 place **two
> stones per move**, so at the **full-turn** level any active length-6 window with
> ≥4 of one player's stones is an immediate winning threat: a count-4 has two empty
> gaps the owner fills with one two-stone move, a count-5 has one. The engine's
> threat predicate is exactly `active && count(player) >= 4`
> (`tactics.rs:189-203`), and the win is checked **after each single placement**
> (`state.rs:304`). Verified: count-4 wins in one two-stone move, count-5 in one
> placement (TESTS A/B).
>
> **Two corrections that reshape the search design:**
>
> 1. **Per-node ≠ per-turn (phase-awareness).** The hexgt MCTS expands **one stone
>    per node** (`TurnPhase::{FirstStone, SecondStone}`, one placement each). A
>    count-4 is "win-now" **only when two placements remain in the turn**
>    (FirstStone); at a **SecondStone** node only one placement is left, so a
>    count-4 becomes a count-5 and the turn passes — *not* a win (TEST G). Every
>    "win-now / forced" statement must therefore carry a **placements-remaining**
>    qualifier.
>
> 2. **Defense is a hitting-set, not "fill all gaps."** One defender stone in
>    **either** empty cell of an opponent threat window makes that 6-cell window
>    two-coloured, so the opponent can no longer complete **that** window (TEST C:
>    one stone kills a simple four). The forced-loss test is therefore the
>    **minimum number of stones that hits ≥1 empty of every active opponent ≥4
>    window** (a transversal), compared against the *placements remaining*. The raw
>    count of threat windows is **not** the metric: an open four is **3** threat
>    windows yet a **2-cell** hitting set defends it (TEST H), and a single shared
>    gap can kill **four** windows with **one** stone (TEST I). "Triple-threat-or-
>    more is always unstoppable" is **false** in general.
>
> **Recommended TSS form — a four-part integration, in priority order:**
>
> - **(a) Tactical-candidate injection at expansion (the load-bearing search fix).**
>   hexgt's MCTS materializes children lazily and caps them by a static
>   policy-nucleus widening limit (`widening_max_children = 96` in the live config,
>   `mcts_tree.rs:485-544,762-826`). A **low-prior** urgent block (or own win) ranked
>   beyond the cap is **never expanded**, so a leaf value override alone cannot save
>   it — the move is never searched. Fix: at any node where a ≥4 threat exists,
>   **force-inject the tactical cells** (own winning completions + the empties of
>   every opponent ≥4 window) as **guaranteed-expanded children**, regardless of
>   prior/widening.
> - **(b) Phase-aware, hitting-set leaf value override** for **true 1-ply forced
>   positions only**: HARD WIN if the side to move can complete a window with the
>   placements it has *this node*; else HARD LOSS if the **minimum hitting set** over
>   the opponent's ≥4 windows exceeds the side-to-move's *remaining placements*.
> - **(c) Threat / "hot-token" CANDIDATE features** (including the currently-missing
>   count-4 signals), as a learned prior — distinct from the count-5 "this single
>   placement wins" flag.
> - **(d) Defer the deep multi-ply VCF/VCDT solver** (root-only, depth-bounded if
>   ever).
>
> **Value side (PART 2).** **soft-Z** value targets and the **global-pooled value
> readout** are **complementary, not alternatives**: soft-Z *recalibrates* the
> label (fixes optimism/saturation) but does **not** make the value head attend to
> the whole board; the global-pooled readout is what structurally lets the value
> integrate the whole board. Both are needed; the TSS threat features/override add
> the explicit danger signal. soft-Z stays **Rank 0a (do first)**; the global-pooled
> readout is its **mandatory structural partner** (companion-doc Rank 1).

---

## 0. The failure this is meant to fix (recap, grounded)

The forensic pass diagnosed hexgt's core weakness as **defensive value
miscalibration**, with two measured symptoms:

1. **Over-confident loss.** The value head predicts ≈ **+0.8 (winning)** in the
   plies right before losing, in 8/8 lost games.
2. **Side-to-move optimism bias of +0.82.** For the *same* board, `v(A) + v(B)`
   sums to ≈ +0.82 instead of ≈ 0 — the value is **anti-calibrated** (both sides
   think they are ahead).

Self-play games are **short, 100% decisive, and getting shorter**
(`selfplay.py` logs `game_lengths`, `forced_decisions`, opening diversity).
External/other-bot games are **rejected**; every fix must live inside **pure
self-play**.

The current value-target machinery (grounded):

- **Main value target is the hard game outcome.**
  `dense_cnn/.../samples.py::finalize_game_samples` sets
  `value=_winner_value(winner, player)` → `+1 / −1 / 0`
  (`samples.py:198`, `samples.py:318-321`). hexgt reuses this finalize verbatim
  (`hexgt/.../selfplay.py:462`).
- **Value head is a 65-bin distributional head**, identical binning to dense_cnn:
  `VALUE_BINS = 65` (`constants.py:16`); `scalar_to_binned_target`
  (`losses.py:40-60`) maps any scalar in `[−1,1]` to a soft two-bin target;
  `binned_value_loss` (`losses.py:63-94`).
- **The only bootstrapped value signal today is the auxiliary STV heads.**
  `_short_term_value_targets` (`samples.py:337-369`) builds, per horizon `h`, an
  **EMA of future MCTS root values** with decay `λ = h/(h+1)`, perspective-
  corrected (`samples.py:354-356`); weighted small (`short_term_value_weight`,
  0.10–0.25, `config.py:83`).
- **`root_value` is captured per decision and in hand at finalize.**
  `selfplay.py:424` appends `(player, sample, search.root_value)`;
  `finalize_game_samples` receives that triple (`samples.py:168-185`) but the
  main-value loop **discards** it (binds `_root_value`, `samples.py:185`).
- **Replay is recency-weighted** `0.9^(current−epoch)` (`_rl_train.py:121-122`).

The two roots are **distinct** and need **distinct fixes** (this is the crux the
final plan makes explicit):

| Root cause | Symptom | The fix that addresses it |
|---|---|---|
| Saturating ±1 label / off-policy MC noise | over-confidence, +0.8-before-loss, +0.82 sum | **soft-Z** value target (PART 2) — *recalibration* |
| Value head reads a single SIDE hub; under-propagates multi-window danger | structurally blind to "opponent has an unanswerable ≥4 conjunction" | **global-pooled value readout** (companion Rank 1) — *whole-board integration* |
| Net is the sole arbiter of immediate tactics | misses/mis-evaluates 1-ply forced win/loss | **TSS injection + override + features** (PART 1) — *exact tactical signal* |

These compose; none substitutes for another.

---

# PART 1 — Threat-Space Search (TSS) for Connect6 / Hexo

## 1.1 What a "threat" is (engine-confirmed)

A **threat** is an active (single-colour) length-6 window with `count(player) ≥ 4`.
Under the two-stones-per-move rule, at the **full-turn** level both count-4 (two
gaps, filled by the two placements) and count-5 (one gap) are **immediate**
winning threats; there is no "non-forcing four." The engine encodes exactly this:
`threat_player` = `active_player()?` then `count(player) >= 4`
(`tactics.rs:189-192`); `is_threat`/`is_threat_for` (196-203); `empty_cells`
(154-156); `is_win_for` = count == 6 (206-208); window-pair `intersects`/`touches`
(210-222); live iterators `threat_entries(player)` / `threats()` (386-395). A
placement touches 18 windows (`tactics.rs:16-17`). All threat geometry the search
needs already exists.

> **An active threat window =** one player's stones with **none** of the other's.
> Its **empties** (`WindowEntry::empty_cells`) are the cells that (offense) complete
> it or (defense) neutralize it. Placing **one** opponent stone in **any** empty of
> a window makes it two-coloured → `active_player()` returns `None` → it can never
> be completed (the engine's `blocked_windows_are_not_threats` test, tactics.rs).

## 1.2 Per-node vs per-turn: the phase qualifier (verified)

The hexgt MCTS searches **single placements**, not whole turns. The engine's turn
is a phase machine (`state.rs:312-329`): `Opening` (P0 places one stone), then
alternating two-stone turns `FirstStone → SecondStone → (pass control)`. Each
placement is a separate node/edge in the tree; the candidate set is single cells
(`candidates.rs`), and a leaf in `select_leaf_batch` can be at **either** phase.

Consequently **"a count-4 is win-now" is only true with two placements remaining**:

- **FirstStone node (2 placements left this turn):** the mover can complete a
  count-5 (1 gap) *or* a count-4 (both gaps) before control passes → **win-now**.
- **SecondStone node (1 placement left):** the mover can complete a **count-5**
  (1 gap) only. A **count-4 is NOT win-now** — filling one gap yields count-5 and
  **control passes** (TEST G: `terminal=False`, turn → opponent).

Define, per node, `B = placements_remaining_in_turn(state.phase())` (2 at
FirstStone, 1 at SecondStone; Opening is a one-stone P0 turn, `B = 1`). Every
"forced"/"win-now"/"cover" statement below is parameterized by `B`.

## 1.3 Defense is a hitting-set problem (verified)

The defender (side to move) survives the opponent's immediate threats iff it can,
with its `B` placements, place **at least one stone in the empties of every active
opponent ≥4 window**. That is a **minimum hitting set (transversal)** over the
family `{ empty_cells(W) : W active, opponent-owned, count(W) ≥ 4 }`, and the
position is a **1-ply forced loss** iff (the mover has no own win this node and)
**`min_hitting_set > B`**.

Why the earlier "set-cover / fill all gaps / a count-4 costs the defender both
placements" framing was wrong, with evidence:

- **One stone neutralizes one window** (TEST C): a simple count-4
  `O O O O . .` is killed by a defender stone in *either* empty. `min_hitting_set = 1`,
  not 2.
- **Window count ≠ hitting-set size** (TEST H): an *open four* (4 consecutive) is
  **3** overlapping count-4 windows, yet a **2-cell** hitting set (the two ends)
  defends all three. "Open four costs 2" is true — but because it is *three windows
  with no common empty*, which falls straight out of the hitting set; it does **not**
  generalize to "every count-4 costs 2."
- **"Triple-threat-or-more is unstoppable" is false in general** (TEST I): when
  windows **share** empties, ≤2 stones — even **one** stone — can hit many. A cell
  on the intersection of several windows' empties kills all of them at once
  (verified: one stone killing **four** shared-gap windows). A "triple threat" is
  unstoppable only when the three are **independent** (disjoint empties, hitting set
  3 > `B`).

So the offense/defense atoms are:

- **own-win-this-node:** an own count-5 window (1 gap) for any `B ≥ 1`; *plus*, only
  when `B = 2` (FirstStone), an own count-4 window (the two placements fill both
  gaps).
- **opp-must-answer / forced-loss:** opponent ≥4 windows; compute the minimum
  hitting set over their empties; **loss iff `min_hitting_set > B`** and no own win.
- **independence** is read from the empties directly (shared empty ⇒ co-hittable);
  `WindowKey::intersects`/`touches` (`tactics.rs:210-222`) are available but the
  empties already settle hittability.

The hitting set is tiny (each window has ≤2 empties; threats are few), so an exact
solver is trivial: for `B = 1`, a hitting set of size 1 exists iff all opponent ≥4
windows share a common empty cell; for `B = 2`, brute-force pairs over the (small)
union of empties. This is `O(active windows + |empties|²)` per node — far cheaper
than a GNN forward.

## 1.4 How hexgt does TSS — the finalized four-part integration

All parts reuse the **existing window walk** (no new board scan). Windows are
enumerated by the engine and materialized as WINDOW tokens
(`candidates.rs::window_tokens`, `candidates.rs:203-227`) with `count`,
`empty_count`, and the actual `empty_cells`/`stone_cells`. A CANDIDATE node is
joined to a WINDOW node by `EDGE_CANDIDATE_WINDOW` **exactly when that candidate is
an empty cell of that window** (`candidates.rs:354-358`). The candidate builder
**already seeds candidates from active-window empties**
(`candidate_cells`, `candidates.rs:160-166`), so **every tactical cell (own-win and
opponent-block) is already a legal candidate** — the tactical set is a *subset* of
the existing candidates, never a new/illegal move.

### (a) Tactical-candidate injection at expansion — the load-bearing fix

**Why the leaf override is not enough (verified).** hexgt's PUCT tree materializes
children **lazily** and caps them with a **static policy-nucleus widening limit**:

- `select_or_materialize_edge` (`mcts_tree.rs:485-544`) only materializes the next
  staged candidate while `edges.len() < max_eligible_children`; once the cap is
  reached, **the candidate set is permanently closed** and PUCT only chooses among
  already-materialized edges.
- `max_eligible_children = nucleus_count(...)` is computed **once at node
  construction** (`owned_root_from_evaluation` 781-826; `shared_from_cache`
  762-779; `nucleus_count_values` 839-863) — the smallest top-prior prefix covering
  `widening_policy_mass` (0.95), clamped to `[widening_min_children=2,
  widening_max_children]`. The **live RL config sets `widening_max_children = 96`**
  (`configs/hexgt_model2.toml:84-87`).
- Candidates are materialized **strictly highest-prior-first**
  (`materialize_next_candidate` indexes `priors[edges.len()]`).

With a **diffuse GNN policy** over a large radius-3 candidate set, the nucleus
cutoff routinely hits the 96 cap, so **only the top-96 priors by rank are ever
materialized**. An urgent defensive block (or even an own winning completion) with a
**low prior ranked beyond the cap is never expanded** — so the search never reaches
the leaf where the override would fire. *The override fixes the value at a tactical
leaf; injection guarantees that leaf exists.* Both are required.

**The fix.** At node construction, when the node's engine state has any active ≥4
threat (either color), compute the **tactical set** `T(state)` and **force every
cell in `T` to be a materialized child**, bypassing the nucleus cap:

```
T(state) =  { empties of own  ≥4 windows completable with B placements }   // own wins
          ∪ { empties of every opponent ≥4 window }                        // all blocks
   where B = placements_remaining_in_turn(state.phase())
   (own count-4 empties are included only when B == 2; own count-5 always;
    opponent empties always — they are the hitting-set search space)
```

- **Where it hooks.** `owned_root_from_evaluation` and `shared_from_cache`
  (`mcts_tree.rs:762-826`) already receive `&RustHexoState`, so
  `state.board().windows()` (the `WindowStore`) is in hand — **no extra plumbing**.
  After the prior list is built, intersect `T` with the candidate/prior list (it is
  always a subset), then:
  1. **Eagerly materialize an edge for each cell in `T`** (pull them ahead of nucleus
     widening), and
  2. set `max_eligible_children = max(nucleus_count(...), |T|)` so the cap can never
     exclude them.
- **Representation note.** Interior nodes use `NodePriors::Shared` and materialize
  strictly by descending-prior index (`priors[edges.len()]`); injecting an
  out-of-order low-prior edge breaks that invariant. So a node with `T ≠ ∅` should
  either (i) switch to an **`Owned`** prior list with the `T` cells materialized
  first, or (ii) carry a small **separate forced-edge list** materialized eagerly,
  with the widening index tracked independently of `edges.len()`. Threats are
  **rare** (only when a ≥4 window exists), so the `Owned`-copy cost is negligible and
  the common (`T = ∅`) path is **unchanged**.
- **Root specifically.** Injecting `T` at the root (built by
  `owned_root_from_evaluation`) is what directly cures the "miss the block at the
  root" failure: the urgent defensive move is guaranteed to be a root child and to be
  searched.

This is pure board geometry → **D6-safe** (D6 maps windows→windows, empties→images
bijectively; the injected set is the image set). It does **not** alter feature/graph
construction.

### (b) Phase-aware, hitting-set leaf value override — 1-ply forced positions only

A depth-1 probe at the MCTS leaf that **overrides the net** when a one-node proof
exists. Hooks into `select_leaf_batch` (`mcts.rs:511-571`), which already classifies
each selected leaf and **backs up terminal / existing-node values without a network
eval** (terminal → `terminal_value(outcome, leaf_player)` backed up via
`backup_virtual`, `mcts.rs:538-541`; existing node → `node.value()`, 542-544; else
push for eval, 545-555). The full leaf `state` is in hand (`mcts.rs:539`), so the
probe reads `state.board().windows()` directly and, on a proof, backs up ±1 via the
existing terminal path and **skips the net eval** (a proven node is as good as
terminal; the cache stays correct).

With `B = placements_remaining_in_turn(state.phase())`:

- **HARD WIN (+1) — checked first** (the side to move moves first): the side to move
  owns a window completable with `B` placements — an own **count-5** (any `B ≥ 1`),
  or an own **count-4 only when `B = 2`**. (At a SecondStone node a count-4 is *not* a
  win — TEST G; do not return +1 for it.)
- **HARD LOSS (−1) — only when there is no own win:** compute the **minimum hitting
  set** over the opponent's active ≥4 windows' empties; if `min_hitting_set > B`, the
  opponent wins next regardless → back up **−1**. (Trivially a loss on a true
  *independent* triple-threat; **not** a loss when the windows are co-hittable —
  TESTS H/I.)
- **MUST-ANSWER (no override):** opponent ≥4 windows with `min_hitting_set ≤ B`. Not
  auto-lost; injection (a) has already guaranteed the covering cells are children, so
  PUCT will search the defense. Leave the value to the net/search.

**Scope honesty.** This is a **1-ply** guarantee — it is exact for positions that
are a win/loss on the very next node, and it is the direct, *hard* cure for "value
+0.8 at a position that is actually 1-ply lost." It does **not** catch multi-ply
forced losses (those are (d)); as the search deepens, leaves nearer the terminal get
caught.

### (c) Threat / hot-token CANDIDATE features (learned prior)

Emit, on CANDIDATE nodes, threat flags into the reserved slots `[30:32)`
(`constants.py:117`), bumping `FEATURE_SCHEMA_VERSION` 2→3 and the byte-parity test
(`tests/test_hexgt_feature_buffer.py`, named in `features.rs:11`). This is the
neural-network analog of TSS and strengthens the policy's reflexes (which improves
the priors injection (a) relies on), but it is a **signal, not a guarantee**.

**Fix the count-4 gap with correct semantics.** Today the completion flags fire
**only at `cnt == 5`** (`features.rs:150-152` own, `162-163` opp; documented at
`constants.py:83-84`); `F_CAND_OWN_WIN4`/`F_CAND_OPP_WIN4` (`features.rs:146,158`;
`constants.py:108,111`) merely *count* count-4 windows. Add features — but keep the
two meanings **distinct**, because they are not the same event:

- `complete_now` (single-placement win) = sits in an own/opp **count-5** gap → "this
  one stone wins" (unchanged meaning).
- `pair_completes` (two-placement win) = sits in an own/opp **count-4** window's
  empties → "one of the two placements that *together* win." This is **phase-relevant**
  (only a win-now at FirstStone) and must not be folded into `complete_now`, or the
  net is taught that a single count-4 placement wins.
- graded severity (Option c): window `open_ends ∈ {0,1,2}`, a `forcing` flag (fires
  for **any** ≥4 window), and `n_opp_unanswerable` = number of opponent ≥4 windows
  whose **hitting set exceeds the defender's `B`** (graded "how close to a forced
  loss," using the §1.3 metric — *not* a raw window count).

Same hooks (`features.rs:131-182` + constants + parity test), **D6-safe**, and a
**drop-in onto the live checkpoint with no cold start** via the proven zero-init
layer-expansion (`architecture.zero_init_expanded_feature_columns`; the v2 slots used
the same path, `constants.py:123-129`).

### (d) Deep VCF/VCDT solver — deferred

The classic multi-ply forcing-move search (chain ≥4 threats / VCDT
double-threat-or-more until the defender's stones run out). It is the real strength
multiplier but the heaviest change (in-tree depth/breadth bookkeeping, hitting-set
defense at every ply, termination proofs), and it most disturbs the visit/policy
targets. **Root-only and depth-bounded** if ever, and only if (a)+(b)+(c) plus the
value fixes don't move the defensive-calibration metric. The standard cautions apply
and are real (verified against the Connect6 TSS literature): a forcing-only search
can miss wins needing a quiet setup move, and must account for the defender's own
counter-threats — which is exactly why the restricted candidate set in (a)/(b) is
sound only for the **1-ply** claim, not as a general solver.

### Pure-self-play interaction (caveats)

1. **Policy-target distortion.** Injection (a) adds children and the override (b)
   changes backed-up values; both feed the exported **policy training target**
   (`pruned_visit_policy`, `mcts.rs:865-915`). Combined with **forced playouts**
   (`forced_playout_k = 2.0` in the RL run, `config.py:137`), stacking can
   over-determine the visit distribution. **Mitigation:** keep injection/override on
   **selection + value** only; **do not** hard-collapse the exported policy to the
   proven/injected move — let visits concentrate naturally and let the existing
   KataGo policy-target pruning (`mcts.rs:865-915`) remove the forced/injected tail.
   Injected-but-bad moves (a block when a faster win exists) get few visits and are
   pruned.
2. **Decisiveness / shortening.** Games are already 100% decisive and shortening; a
   sharp tactical layer can shorten further. That is *correct* play but reduces
   diversity — pair with the opening-temperature / diversity levers
   (`selfplay.py` opening logging) and **judge by H2H, not game length**.

### D6-safety (all of PART 1)

Threats are pure board geometry; D6 maps windows→windows, owners→owners,
empties→images bijectively (the argument `constants.py:103-106` makes for v2
window-count features). The injected tactical set is the image set; a ±1 override is a
D6-invariant scalar; the new candidate features are D6-invariant flags. **Safe.**

---

# PART 2 — Softer value targets, in depth

## 2.1 The paper: soft-Z, A0C, A0GB (Willemsen, Baier, Kaisers, NCAA 2022)

*"Value targets in off-policy AlphaZero: a new greedy backup"* (DOI
10.1007/s00521-021-05928-5). It defines a **family of value targets** parameterized
by how far you bootstrap along the **real self-play game** (`n_real`) and the
**simulated MCTS tree** (`n_sim`), plus a backup-width choice. The four named
members:

- **AlphaZero (original):** the **final game outcome** (`n_real = ∞`). High variance
  (a Monte-Carlo return biased by later exploratory moves).
- **soft-Z:** `y = v̂_MCTS(s, s)` — the **MCTS root value at the position itself**
  (`n_real = 0, n_sim = 0`). The search's own integrated estimate at `s`. **Highest
  bias, lowest variance** of the alternatives. (In the paper soft-Z is purely the
  MCTS value — *not* a convex blend with `z`; the blend below is a practical anchor.)
- **A0C** (Moerland): one greedy tree step then back up that child's MCTS value;
  bias/variance between soft-Z and A0GB.
- **A0GB** (the paper's contribution): follow the greedy MCTS policy to a
  leaf/terminal and use that leaf's value (= `v̂_NN` at a single-visit leaf). **Lowest
  bias, highest variance.**

**Results (verified via the paper/abstract).** soft-Z, A0C, **and** A0GB achieve
**better performance and faster training** than the hard-outcome target on
**Connect-Four** and **Breakthrough (6×6)**; A0GB uniquely reaches the optimal policy
in a tabular Tic-Tac-Toe where the hard target fails. Mechanism: the final outcome is
a high-variance, off-policy-noisy label, whereas the MCTS value at the position is a
lower-variance, on-position estimate that already integrated the search.

## 2.2 Why this attacks the +0.82 optimism bias — and what it does NOT fix

**soft-Z recalibrates the label.** The hard `z` label is **±1** for *every* position
in a won/lost game regardless of how close it was. Two failure modes follow, both of
which soft-Z addresses:

1. **Magnitude saturation.** The head only ever sees ±1, so its 65-bin softmax
   collapses toward the end bins → systematic over-confidence ("+0.8 right before
   losing"). soft-Z trains it on the search's `root_value ∈ (−1,1)`, exposing the full
   magnitude range.
2. **Anti-calibration / non-zero-sum sum.** Both sides are taught they are winning;
   `v(A)+v(B) ≈ +0.82`. The MCTS `root_value` is the search's integrated, closer-to-
   zero-sum estimate; blending toward it shrinks the sum toward 0.

> **CLARIFICATION (explicit — this is a complementarity, not a substitution).**
> **soft-Z only *recalibrates / softens* the value target. It does NOT make the value
> head attend to the whole board.** A recalibrated label still flows through the
> *same* readout, which today reads from a **single SIDE hub**; a 3-hop GNN +
> single-hub readout structurally **under-propagates** multi-window danger ("the
> opponent has a ≥4 conjunction I cannot parry"). Making the value *structurally*
> integrate the whole board is the job of the **global-pooled value readout**
> (companion-doc Rank 1: replace/augment the single-hub readout with mean/max pooling
> over all nodes). The two are **orthogonal and both required**:
>
> | Change | Root it fixes | What it canNOT do alone |
> |---|---|---|
> | **soft-Z target** | label saturation, +0.82 optimism, over-confidence | give the head whole-board receptive field |
> | **Global-pooled readout** | structural under-propagation of multi-window danger | de-saturate the label / fix optimism |
> | **TSS features + override** | explicit 1-ply danger signal & guarantee | replace either of the above for general positions |
>
> Ship **soft-Z and the global-pooled readout together**; the TSS layer adds the
> exact tactical signal on top. STV already validates, on this pipeline, that
> training a value head on `root_value` is stable and learns.

## 2.3 Concrete hexgt implementation (soft-Z)

**One place:** the main-value assignment in
`dense_cnn/.../samples.py::finalize_game_samples` (`samples.py:198`). Everything is
already in scope: `winner`/`player` (hard outcome), and the per-decision `root_value`
currently bound as `_root_value` and **discarded** (`samples.py:185`) — the same
signal STV consumes.

**Perspective (verified).** `root_value` (`search.root_value`, `mcts.rs:677`
`root.value()`) is **from the side to move at that decision**, the same convention STV
relies on; STV sign-flips only **future** values to the decision's perspective
(`samples.py:354-356`), so *this* position's own `root_value` needs **no flip** —
both `_winner_value(winner, player)` and `root_value` are already in `player`'s
perspective. (Confirm once with a unit test asserting `decode_binned_value(forward(s))`
and the stored `root_value` share sign on a known-winning position.)

**Blend (recommended: convex soft-Z).**

```
value_target(decision) = (1 − λ) · _winner_value(winner, player)
                       +     λ  · root_value_in_player_perspective
```

- `λ = 0` → today's hard outcome; `λ = 1` → the paper's pure soft-Z.
- **Start λ = 0.5**, anneal ~0.3 → ~0.7 as calibration improves (§2.4). Rationale:
  the paper found soft-Z already strictly better than hard-z, so a substantial λ is
  warranted; hexgt's own STV trains on `root_value` stably; but early-RL `root_value`
  is itself miscalibrated, so the 0.5 hard-outcome anchor prevents a soft target from
  re-teaching its own error. A config scalar; **no schema/head change.** For
  draw/truncation rows (`samples.py:193-194`) `_winner_value = 0` and `root_value`
  carries the estimate — a free improvement.

**The 65-bin mapping is automatic (verified).** The blended scalar stays in `[−1,1]`
(convex combination of two values in `[−1,1]`), so it flows through the **existing**
`scalar_to_binned_target`: `position = (v+1)·((VALUE_BINS−1)/2) = (v+1)·32`, with
floor/ceil linear two-bin weights (`losses.py:50-59`); `binned_value_loss` unchanged.
**No new value-head plumbing.**

**A0GB variant — deferred.** A0GB needs the Rust search to walk `π_greedy` to a leaf
and export that scalar (the tree is discarded after the move; only `root_value` is
exported, `mcts.rs:677`). soft-Z captures most of the benefit with **zero Rust
change**; revisit A0GB only if soft-Z's residual (highest-bias) caps the gain.

**Interactions.** STV (aux, horizons 4/12/24) and soft-Z (main, horizon-0) are the
same idea on different heads and compose; keep STV's small weight and do **not**
blend STV targets with the hard outcome. Watch for an early-RL feedback loop if both
bootstrap from the same miscalibrated `root_value` — the 0.5 hard anchor on the main
head is the circuit-breaker; monitor §2.4. Recency weighting
(`epoch_recency_weight`) is orthogonal (it weights *which epochs*), and composes —
fresh self-play carries the recalibrated target.

## 2.4 Validation plan

Judge by these, **not** train loss (a documented artifact pitfall here):

1. **Same-board `v(A)+v(B)` sum (primary).** Re-measure optimism on a fixed probe
   set from both sides; soft-Z + global-pooled readout should move it ≈ +0.82 → 0.
2. **Calibration on opponent-hot slices.** Slice held-out self-play to positions
   where the opponent has a ≥4 threat (label cheaply with the Part-1 detector);
   measure CE/Brier and the reliability curve.
3. **The 8/8-lost-game value trace.** Re-run the forensic probe; final-ply value
   should drop toward the loss instead of pinning near +0.8 (traces logged,
   `selfplay.py:425-440`).
4. **H2H** vs dense_cnn e24 via `run_head_to_head` (player.py / evaluation.py).
5. **λ ablation** ∈ {0, 0.3, 0.5, 0.7} for a few RL epochs from the live checkpoint
   (no cold start).

---

# Synthesis — recommendation & validation order

**Two complementary value-defense tracks, both inside pure self-play.**

**Value track (PART 2 + companion Rank 1) — do first, together:**
- **soft-Z** main-value target `value = (1−λ)·z + λ·root_value`, **start λ = 0.5**
  (anneal 0.3→0.7), in `samples.py::finalize_game_samples`, mapped via the unchanged
  65-bin `scalar_to_binned_target`. Recalibration; few lines + a config scalar;
  D6-safe; drop-in, no cold start; externally validated and STV-proven here.
- **Global-pooled value readout** (companion Rank 1): the *structural* partner that
  gives the value head whole-board receptive field. **soft-Z and the readout fix are
  complementary, not alternatives** — neither alone fixes both the saturated label
  and the structural blindness.

**Search/representation track (PART 1):**
- **(a) Tactical-candidate injection at expansion** — the load-bearing fix; without
  it a low-prior urgent block is never expanded under the `widening_max_children = 96`
  nucleus cap, and the override never gets a leaf to correct.
- **(b) Phase-aware, hitting-set leaf value override** — the 1-ply *hard* guarantee
  (HARD WIN with `B`-placement completions; HARD LOSS iff `min_hitting_set > B`).
- **(c) Threat / hot-token candidate features** — learned prior; adds the missing
  count-4 signals with phase-correct semantics; strengthens the priors (a) depends on.
- **(d) Deep VCF/VCDT** — deferred (root-only, depth-bounded if ever).

**Ranking among existing fixes (companion doc):** soft-Z stays **Rank 0a**, with the
**global-pooled readout as its mandatory Rank-1 partner**. TSS injection+override is a
new search-side guarantee just below Rank 1; the deep in-tree solver is the deferred
§6.3 option.

**Validation order (cheapest first):**
1. Establish the **`v(A)+v(B)` sum** and **opponent-hot calibration** harness (§2.4).
2. **soft-Z λ-sweep + global-pooled readout** from the live checkpoint (fastest A/B);
   confirm the sum moves toward 0 and the 8/8 trace stops pinning at +0.8.
3. **Tactical injection (a) + features (c)** — re-measure metrics + H2H; verify
   urgent root blocks are now expanded (instrument materialized-edge counts on threat
   nodes).
4. **Leaf override (b)** — confirm proven-loss leaves no longer carry +0.8 net values;
   watch game-length/diversity and policy-target sharpness (don't let forced playouts
   + injection/override over-determine the policy).
5. Only if defense still lags: **deep root-bounded VCDT**.

---

## Verification log (this revision)

Each owner refinement was re-checked empirically against `hexo_engine` and/or the
hexgt MCTS source. Harness scripts noted; tests labeled A–I.

| Claim | Verdict | Evidence |
|---|---|---|
| **(1) Phase-awareness** — a count-4 is win-now only with two placements left; at a SecondStone node it is not | **Confirmed** | TEST G (`_tss_verify2.py`): at FirstStone the mover owns a count-4; after wasting the first stone (→ SecondStone) the lone remaining placement fills one gap → count-5, `terminal=False`, control passes to the opponent. Phase machine: `state.rs:312-329`; win-after-each-placement: `state.rs:304` |
| **(2a) One stone neutralizes one window (hitting-set, not fill-all-gaps)** | **Confirmed** | TEST C (`_tss_verify_c.py`/`_tss_verify2.py`): opponent simple count-4 `O O O O . .`; one defender stone in either empty → window two-coloured → no ≥4 threat remains (`min_hitting_set = 1`) |
| **(2b) "Triple-threat-or-more is always unstoppable" is false** | **Refuted** | TEST H: open four = **3** count-4 windows, min hitting set **2** → defended by 2 stones. TEST I: cell X is the shared empty of **4** count-4 windows → **one** stone kills all four. Hitting-set min, not window count, decides |
| **(3a) Leaf hook backs up terminal/existing values without a neural eval** | **Confirmed (code)** | `select_leaf_batch` `mcts.rs:511-571`: terminal → `terminal_value` via `backup_virtual` (538-541); existing node → `node.value()` (542-544); else push for eval (545-555); full leaf `state` in hand (539) |
| **(3b) Children are materialized lazily and capped by static nucleus widening — a low-prior urgent move can be starved** | **Confirmed (code)** | `select_or_materialize_edge` `mcts_tree.rs:485-544` only widens while `edges.len() < max_eligible_children`, then closes the set; cap computed once at construction (`nucleus_count_values` 839-863; node build 762-826); materialize strictly highest-prior-first (`materialize_next_candidate`); live cap `widening_max_children = 96` (`configs/hexgt_model2.toml:84-87`). ⇒ injection (a) is required |
| **(value) soft-Z recalibrates only; whole-board attention needs the global-pooled readout** | **Confirmed (design)** | soft-Z changes only the *scalar target* (`samples.py:198/185`) routed through the *unchanged* single-SIDE-hub readout; structural under-propagation is a readout-topology issue → needs the companion Rank-1 global pooling. Documented as complementary in §2.2 |

**Also re-confirmed from the first pass:** threat predicate `active && count ≥ 4`
(`tactics.rs:189-203`); count-4 wins in one two-stone move / count-5 in one placement
(TESTS A/B); open four needs 2 blocks / two independent open fours = genuine forced
loss (TESTS E/F); completion flags fire only at `cnt == 5` (`features.rs:150-152,
162-163`); 65-bin mapping `(v+1)·32` verbatim (`losses.py:50`); PART 2 grounding
(root_value triples received, discarded in the main-value loop, STV EMA `λ=h/(h+1)`).

---

## Sources — fetched vs. reasoned-about

- **FULL TEXT / abstract verified:** Willemsen, Baier, Kaisers, *"Value targets in
  off-policy AlphaZero: a new greedy backup,"* Neural Computing and Applications 34
  (2022) 1801–1814, DOI 10.1007/s00521-021-05928-5 (soft-Z/A0C/A0GB definitions,
  bias-variance ordering, Connect-Four/Breakthrough results).
- **Connect6 TSS/VCDT literature (search-snippet / abstract level):** Wu & Kang,
  *Dependency-Based Search for Connect6*; RZOP / VCDTS / ITSS lines (IEEE/Springer/
  ResearchGate). Corroborate the TSS/VCDT framing and the single/double/triple-threat
  classification; the **defensive hitting-set** correction in §1.3 is grounded in the
  **engine** (one stone two-colours a window), not these papers.
- **CODE (read, hexgt worktree `E:\Hexo-BotTrainer-hexgt`):** `hexo_engine`
  `tactics.rs`, `state.rs`; hexgt `candidates.rs`, `features.rs`, `mcts.rs`,
  `mcts_tree.rs`, `constants.py`, `losses.py`, `config.py`, `selfplay.py`,
  `configs/hexgt_model2.toml`; dense_cnn `samples.py`, `_rl_train.py`.
- **Empirical harness (this repo):** `scripts/_tss_verify.py`,
  `scripts/_tss_verify_c.py`, `scripts/_tss_verify2.py` (run under the WSL CUDA venv;
  only `hexo_engine` needed).
- **KataGo (Wu 2019, arXiv 1902.10565):** referenced from the companion doc's §6.2
  (value-loss weight, auxiliary targets); not re-fetched.
