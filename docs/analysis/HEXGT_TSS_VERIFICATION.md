# hexgt TSS & Soft-Value Design — Independent Verification Report

**Subject:** [`HEXGT_TSS_AND_SOFT_VALUE_DESIGN.md`](HEXGT_TSS_AND_SOFT_VALUE_DESIGN.md)
(committed `bbbb082`, branch `bench/inference-backends-wsl`).
**Method:** adversarial review — every mechanical claim re-derived and, where the
engine can settle it, **tested empirically against the real `hexo_engine`** (not
re-reasoned). Reproduction harness: [`scripts/_tss_verify.py`](../../scripts/_tss_verify.py)
and [`scripts/_tss_verify_c.py`](../../scripts/_tss_verify_c.py) (run under the WSL
CUDA venv; only `hexo_engine` is needed). Engine source read in the hexgt worktree
`E:\Hexo-BotTrainer-hexgt` (read-only). No training run was touched.

---

## HEADLINE

**The plan is _mostly_ well-founded, with one unsound flagship piece.**

- **PART 2 (soft value targets) is sound** — the soft-Z formula, the 65-bin
  mapping, the code hooks, and the paper characterization all check out. Ship it
  as written (modulo the one perspective unit-test the doc already asks for).
- **PART 1's corrected threat model is correct** — `count ≥ 4 ⇒ threat`, a single
  four forces, count-4 wins in one two-stone move, count-5 in one placement: all
  **empirically confirmed**. The feature-generator options (b)/(c) and the
  "count-4 completion is currently unflagged" code note are accurate and useful.
- **PART 1's flagship "depth-1 leaf value override" (Option a) is algorithmically
  unsound _as specified_** and must not be implemented verbatim. It is pitched as
  the "hard ground-truth fix" / "hard guarantee," but its loss test uses the wrong
  combinatorial primitive and is blind to the engine's two-phase move structure.

**Issue count:** **2 High, 2 Medium, 2 Low/caveat.** Everything else verified
(≈30 individual code/rule claims confirmed; citations are accurate).

**Most important problem (HIGH-1):** the override declares a **HARD LOSS (−1)**
whenever the defender's two placements "cannot cover **all** the opponent's
≥4-window **gaps**" — a *set-cover over the union of gaps*. That is the wrong
model. Neutralizing a threat window needs only **one** stone in its empty set
(it breaks the window's one-colour status), so the correct test is a **size-≤2
hitting set** over the active ≥4 windows' empty-cell sets. The doc's own example —
"two independent fours/fives whose gaps exceed two cells ⇒ loss" — is **false**:
two independent simple fours have 4 gap cells yet are **fully defended by 2
stones**. Implemented verbatim, the override would stamp confidently-wrong **−1**
labels onto **defensible** positions — injecting exactly the kind of value
miscalibration the doc set out to cure.

---

## Verdict table

| # | Claim in the doc | Verdict | Evidence | Sev |
|---|---|---|---|---|
| 1 | Engine threat predicate is `count(player) >= 4` on an active (one-colour) window (`tactics.rs:188-203`) | **Confirmed** | `threat_player`: `active_player()?` then `count(player) >= 4` (tactics.rs:189-192); `is_threat`/`is_threat_for` 196-203 | — |
| 2 | A **count-4** window is winnable in **one two-stone move** | **Confirmed** | TEST A: stones at offsets {0,1,4,5}, gaps {2,3}; mover plays both gaps → six (`reason='six_in_line'`, winner P0). First gap → count-5 (not terminal), second → win | — |
| 3 | A **count-5** window wins with **one** placement | **Confirmed** | TEST B: one stone in the single gap → terminal win after the *first* stone (engine checks win after each placement, `state.rs:304`) | — |
| 4 | Blocking an **open four** costs the defender **two** placements | **Confirmed** | TEST E: open four (4 consecutive) = **3** overlapping count-4 windows; a single end-block leaves a live count-4/5 on the far side | — |
| 5 | An unstoppable conjunction (two independent **open** fours / triple-threat) is a genuine forced loss | **Confirmed** | TEST F: 2 independent open fours = 6 threat windows needing ≥4 distinct blocks; P1's best 2 stones still lose — P0 completes the other four | — |
| 6 | Override **HARD LOSS** = "defender's 2 placements cannot **cover all** opponent ≥4-window **gaps**" (set-cover); "two independent fours whose gaps exceed two cells ⇒ loss" | **Refuted** | TEST D: two independent **simple** fours (gaps {(2,0),(3,0)} and {(2,3),(3,3)}, 4 cells) → defender plays (2,0),(2,3) → **both windows dead**, P0 has **no** ≥4 window, cannot win next move. Position is **defensible**; override would falsely return −1 | **High** |
| 7 | "the defender's move must cover its gap(s) (… **2 for count-4**) or lose" — a count-4 needs **both** gaps covered to defend | **Refuted** | TEST C: opponent owns a simple four; defender plays **one** stone in **one** gap → window contaminated (not one-colour) → **no** ≥4 threat remains. One stone, not two | **High** |
| 8 | HARD WIN: "side to move owns a **count-4** window ⇒ plays both gaps ⇒ wins **this move**" (assumes 2 placements available) | **Needs-fix** | TEST G: at a **SecondStone** leaf the mover has **one** placement left; filling one count-4 gap → count-5, **turn passes** (terminal=False). count-4 is win-now only at **FirstStone** | **High** |
| 9 | The override is a "hard guarantee" against "value +0.8 right before a forced loss" | **Overstated** | Depth-1 only catches a loss **exactly one opponent move away**; multi-ply forced losses (VCF/VCDT chains) still get the net's value at the relevant leaf. Doc itself defers the deep solver | **Med** |
| 10 | Option (b): a single binary `complete-now-own` flag covers both count-5 and count-4 completion | **Needs-fix** | Semantics differ: count-5 gap = "this placement wins"; count-4 gap = "one of **two** placements needed." Folding both into one flag (plus the phase issue) mislabels "win-now" | **Med** |
| 11 | Restricting TSS expansion to "empty cells of ≥4 windows" guarantees no missed defence | **Incomplete** | True for *blocking* defences (those cells are candidates). Misses *non-blocking* refutations (counter-threats / quiet moves) — classic forcing-search incompleteness. Harmless at depth-1; matters only for a deeper solver (deferred) | **Low** |
| 12 | `F_CAND_COMPLETE_{OWN,OPP}` fire only at `cnt == 5`; count-4 completion is unflagged; `*_WIN4` slots only count | **Confirmed** | features.rs:150-152 (own, `if cnt==5`), 162-163 (opp); WIN4 at 146/158 just `+= 1.0`; constants.py:83-84,108,111 | — |
| 13 | MCTS leaf hook exists with full state/window access; `root_value`, pruned policy as cited | **Confirmed** | `select_leaf_batch` mcts.rs:511; terminal/`backup_virtual` 538-541; `selected.state.current_player()` 539 (so `state.board().windows()` reachable); `root_value` 677; `pruned_visit_policy` 875-915 | — |
| 14 | Soft-Z 65-bin mapping is verbatim: `position=(v+1)·(64/2)`, two-bin split | **Confirmed** | losses.py:50 `position=(flat+1.0)*((VALUE_BINS-1)/2.0)`; floor/ceil weights 51-59; `VALUE_BINS=65` constants.py:16 | — |
| 15 | PART 2 grounding: `finalize_game_samples` gets `(player,sample,root_value)`; main loop discards `_root_value` (uses `_winner_value`); STV is an EMA of future root_value with `λ=h/(h+1)`, perspective-corrected | **Confirmed** | samples.py:168-185 (`_root_value` discarded), 198 `_winner_value`, 318-321, 337-369 (STV; future values sign-flipped, own not). Convex blend `(1−λ)z+λ·root_value` stays in [−1,1], in-perspective | — |
| 16 | soft-Z / A0C / A0GB beat hard-z (Connect-Four, Breakthrough); A0GB = greedy-to-leaf | **Confirmed** | Paper + abstract (Willemsen/Baier/Kaisers); matches doc §2.1 | — |
| 17 | Citations point at the right files/lines | **Confirmed (1 nit)** | All line numbers accurate. Nit: `tactics.rs` lives in **`hexo_engine`** (`packages/hexo_engine/rust/src/tactics.rs`), not the hexgt crate, though the doc lists it under the "hexgt worktree" | **Low** |

---

## The core defect, precisely (HIGH-1 & HIGH-2)

The doc models the defender's task as a **set cover**: "cover the empty gaps of
every live opponent ≥4 window" with two placements (§1.3, §1.4, Option a). The
correct model is a **hitting set (transversal)**:

> A threat window `W` (active, `count ≥ 4`) is neutralised iff the defender places
> **≥1** stone in `W`'s empty cells — that single stone makes `W` two-coloured, so
> `active_player()` returns `None` and `W` can never be completed.
> The position is a depth-1 forced loss for the side to move iff it **cannot win
> this move** AND there is **no set of ≤B placements that hits every active ≥4
> opponent window's empty set**, where the placement budget `B` is **2 at a
> FirstStone leaf and 1 at a SecondStone leaf.**

Why the distinction is not academic:

- **Set-cover over-fires (HIGH-1).** "Fill all gaps" charges 2 placements per
  count-4 window and 1 per count-5, then compares the union to 2. So a *five + an
  independent four* (3 gap cells) or *two independent simple fours* (4 gap cells)
  read as losses. Hitting-set charges **1 per window** → both are defended in 2
  moves. TEST C/D confirm empirically. The "open four needs 2 blocks" fact the doc
  leans on is real (TEST E) but it is a property of the open four being **three
  overlapping windows with no common empty** — it falls straight out of hitting-set
  (TEST F), and does **not** generalise to "every count-4 costs 2."

- **Phase-blindness over-claims wins and mis-budgets defence (HIGH-2).** The
  Rust MCTS expands **single placements**; nodes alternate through
  `TurnPhase::{FirstStone, SecondStone}` (one stone each, `state.rs:312-329`), and
  a leaf in `select_leaf_batch` can be either phase. At SecondStone the mover has
  **one** stone left: a count-4 is **not** win-now (TEST G), and the defensive
  budget is **1**. The doc's "two placements per move" never maps to the leaf.

Both are fixable without new machinery — the engine already exposes
`threat_entries(player)`, per-window `empty_cells`, and the phase. The override
should: (1) read `state.phase()` for the budget `B`; (2) HARD WIN only if an own
window is completable with `B` stones (count-5 always; count-4 only when `B==2`);
(3) HARD LOSS only if no own win AND no ≤`B` hitting set over opponent ≥4 windows;
(4) be advertised as a **1-ply** guarantee, not a general "never +0.8 before a
loss" cure.

---

## Prioritised fixes

1. **(HIGH) Reformulate the override's loss test as a ≤B hitting set**, not a
   set-cover over gap-unions. Delete/replace the "two independent fours ⇒ loss"
   example. Add the TEST C/D positions as regression fixtures so a future
   implementation can't silently regress to set-cover.
2. **(HIGH) Make the override phase-aware** (`B = 2` FirstStone, `1` SecondStone).
   Restrict HARD WIN by phase (count-4 ⇒ win-now only at FirstStone).
3. **(MED) Downgrade the "hard guarantee" language** to a depth-1 (one-opponent-
   move) guarantee; keep the deep VCF/VCDT solver as the thing that would actually
   cover multi-ply forced losses.
4. **(MED) Split the count-4 vs count-5 completion features** (e.g. keep
   `complete-now` = single-placement win = count-5, and add a distinct
   `pair-completes-own` for count-4) so "win-now" stays unambiguous.
5. **(LOW) State the candidate-restriction caveat** — a ≥4-window-empties-only
   expansion misses non-blocking refutations; only relevant if/when the deep solver
   is built.
6. **(LOW) Fix the `tactics.rs` location note** (it is in `hexo_engine`).

## What is solid and should proceed

- **Soft value targets (PART 2) in full** — formula, binning, hooks, perspective
  reasoning, and external/internal validation all check out. Highest leverage,
  lowest risk; the one open item (sign of `root_value` vs `_winner_value` on a
  known-winning board) is already called out by the doc as a unit test.
- **The corrected threat model** (count ≥ 4 forces; single four wins in one
  two-stone move) — matches the engine exactly.
- **Feature options (b)/(c)** as a *signal* to the net (not a hard label),
  including adding the missing count-4 completion flag — accurate and low-risk,
  subject to fix #4 on flag semantics.

---

## Reproduction

```
wsl  ~/hexo-wsl-cuda-venv/bin/activate
python scripts/_tss_verify.py     # TESTS A,B,D,E,F  (B/D refute; A/E/F confirm)
python scripts/_tss_verify_c.py   # TEST C           (single stone kills a simple four)
```
TEST G (SecondStone count-4 ≠ win-now) is the inline snippet in this review's
session; it places an unrelated first stone then the lone gap stone and observes
`terminal=False`, control passing to PLAYER_1.

All positions are built by **legal alternating self-play** through the real
engine (opening = one P0 stone, then two-stone turns), so owner/phase parity is
exactly what the engine enforces; window counts/empties are read from the live
`WindowStore` masks.
