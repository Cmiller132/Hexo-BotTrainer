# Machine hunt report — GAP-RAW (Erdős–Selfridge potential layer)

**Open problem** (`docs/PROOF_TSS_DEFENDER_ZONES.md` §12 item 7; foundations in
`docs/proof_parts/ES_POTENTIAL.md`, `ES_GLOBAL_BOUNDARY.md`): does **every**
nonterminal Defender-`FirstStone` position with `Φ < 1` admit **some** (possibly
non-greedy) forever-blocking defender strategy? Dynamic touched-window greedy is
already refuted as the universal witness (`ES_GLOBAL_BOUNDARY` Theorem 1). Either
some non-greedy defense always exists (theorem), or some `Φ < 1` position is an
attacker win (a refutation, a major result).

**Potential.** `λ = √3`; `Φ(P) = Σ` over attacker-touched alive windows
(`cnt_A ≥ 1`, `cnt_D = 0`) of `λ^{−#empties}`. All-empty windows excluded.

---

## VERDICT

**EVIDENCE-FOR (GAP-RAW is likely a theorem), with the exhaustive-refutation arm
INCONCLUSIVE by a hard depth wall. No refutation candidate was found.**

- **No refutation.** Across the whole battery of `Φ < 1` Defender-`FirstStone`
  positions, the sound Maker-Breaker minimax found **no** attacker forced win at
  any horizon it could exhaust (horizon 4 = 2 attacker placements, complete;
  every position `Unknown`, `refutation=false`).
- **The depth wall is structural, not a budget artifact.** `ES_GLOBAL_BOUNDARY`
  Theorem 2 *proves* the first **five** attacker placements safe from `Φ < 1`.
  So a refutation, if one exists, cannot appear before the **6th** attacker
  placement — the known greedy refutation completes at the **7th** (ply 15 from
  the defender root). Exhaustive minimax over ~230-wide branching cannot reach
  ply ≥ 11–15. The interesting regime is provably out of exhaustive range on one
  machine; a refutation cannot be manufactured at the depths that are reachable.
- **The canonical hard case has an explicit non-greedy survivor.** On the ES
  greedy-refutation core (`Φ = 0.8340`), where dynamic touched-window greedy
  *loses* to a fixed attacker line (reproduced exactly, attacker wins at ply
  15), a **non-greedy** defender (fixed-cohort / target-lock) **foils** that
  same line (at ply 6). GAP-RAW's "the escape exists here" holds for the very
  position that breaks greedy.
- **Key structural finding — the greedy dilemma.** *Neither* greedy is a
  universal witness. Each defends one attack family and loses to the other:

  | defender \ attack | ES cohort-target line | fresh birth 6-line |
  |---|---|---|
  | dynamic touched-window greedy | **LOSES** (ply 15) | foils (ply 10) |
  | fixed-cohort / target-lock     | foils (ply 6) | **LOSES** (ply 12) |

  A GAP-RAW witness, if it exists, must be **adaptive** — simultaneously
  target-locking the initial cohort *and* answering births. This is direct
  input to a future proof: the theorem cannot be a single fixed greedy rule.

### The three hardest positions

| rank | position | `Φ` | frontier | what happened |
|---|---|---:|---|---|
| 1 | `es_core` `A={(0,0)}, D={(1,0)}` | **0.833950** | 1 att, 1 def | The canonical greedy-refutation core. Dynamic greedy **loses** (attacker completes `W={(-5,0)..(0,0)}` at ply 15). A non-greedy target-lock survivor foils it (ply 6). MB horizon-4 exhaustive: **no refutation**. |
| 2 | `blocker_2_0` `A={(0,0)}, D={(2,0)}` | **0.898100** | 1 att, 1 def | Near-threshold single-blocker (14 count-1 windows). MB horizon-4 exhaustive (303,600 nodes): `Unknown`, **no refutation**. |
| 3 | `blocker_3_0` `A={(0,0)}, D={(3,0)}` | **0.962250** | 1 att, 1 def | Closest to the threshold in the battery (15 count-1 windows; `15² = 225 < 243`). MB horizon-4 exhaustive (322,351 nodes): `Unknown`, **no refutation**. |

---

## Method and semantics

**Harness:** `packages/hexfield_eq/rust/src/gap_raw_hunt.rs` (test-gated; never
touches production paths). Independent of the engine's window/threat store: `Φ`
and the game primitives are reconstructed from occupancy so they *differential-
test* against `hexo_engine` and the trusted reference solver.

**Role convention** (proven from `state.rs` turn machine + `ES_GLOBAL_BOUNDARY`
Prop 1): the engine opener **`Player0` is the DEFENDER**; **`Player1` is the
ATTACKER**. A Defender-`FirstStone` position is `Player0` to move in
`TurnPhase::FirstStone`; the cycle from there is `D1 D2 A1 A2 …`. `Φ` sums over
`Player1`(attacker)-alive windows.

**Ground truth for "attacker forces a win."** Two solvers, deliberately:

1. **Blanket Maker-Breaker minimax** (`mb_search`) — the game GAP-RAW is stated
   over: the Attacker is the only player who can win (by completing a length-6
   window); the Defender is a pure Breaker (its own lines are ignored). Returns
   `AttackerWin` **only** when the attacker forces a completed window within the
   ply horizon against **every** defender reply; a node-cap abort yields
   `Unknown`, never a false `AttackerWin`. **This is the primary, sound hunt
   tool.** An `AttackerWin` from a `Φ < 1` root = a refutation.
2. **Maker-Maker reference** (`crate::tss_reference::solve`) — the *actual*
   engine game, where the Defender can also win by making six. Used as an
   independent cross-check. A reference `Loss` (defender-root forced to lose)
   would be, a fortiori, a blanket refutation; but a reference *survival* can
   hide a blanket refutation (defender escaped only by racing to its own six),
   so it is the conservative check, not the primary tool.

Why the distinction matters: `ES_POTENTIAL` §1 keeps the blanket game
(Defender-only-blocks) precisely because "ignoring [defender wins] is
conservative." GAP-RAW is the blanket claim, so the Maker-Breaker solver is the
faithful and more sensitive instrument; the reference solver is the trusted
tie-down.

---

## Φ validation (exact, no floating point in any decision)

`Φ` is stored as the exact count profile `(n₁,…,n₅)` and compared by the
integer identity `27·Φ = A + B√3` with `A = 3n₂+9n₄`, `B = n₁+3n₃+9n₅`, so
`Φ < 1 ⇔ A < 27 ∧ 3B² < (27−A)²` (equivalent to §10 Cor. 2's
`b≤8 ∧ a²<3(9−b)²`). Validated:

| check | expectation | result |
|---|---|---|
| ES core `A={(0,0)}, D={(1,0)}` | profile `(13,0,0,0,0)`, `13√3/27 = 0.833950 < 1` (`169 < 243`) | **exact match** (`phi_core_matches_doc`) |
| lone attacker, no blocker | 18 count-1 windows, `2/√3 = 1.1547 ≥ 1` | **match** (`phi_single_attacker_no_defender`) |
| three separated count-4 | `Φ = 3·(1/3) = 1.0`, **not** `< 1` (Prop 2 strictness) | **match** (`phi_three_count4_equals_one`) |
| two count-5 windows | `2/√3 = 1.1547 ≥ 1` (⇒ `Φ<1` allows ≤ 1 count-5) | **match** (`phi_two_count5_exceeds_one`) |
| single count-5 window | `1/√3 = 0.5774 < 1` | **match** (`phi_single_count5_window`) |
| **danger machinery** — core `D0.1` | `27·d_max = 5√3`, maximizer set `{(-1,1),(0,-1),(0,1),(1,-1)}` | **exact reproduction** of `ES_GLOBAL_BOUNDARY` Theorem 1's table (`dynamic_greedy_core_first_danger_matches_doc`) |

Primitive differential tests vs. the engine: blanket legal-move enumeration ==
`tss_reference::legal_moves` on random reachable occupancies
(`blanket_legal_moves_match_reference`); blanket attacker-six detector ==
engine terminal on a `Player1` win (`blanket_six_matches_engine_on_player1_win`).

All 13 validation tests pass:
`CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release gap_raw_hunt -- --test-threads=1`
→ `13 passed; 0 failed`.

---

## Evidence table

All rows from commit `dba6111d`, regenerable with the report command below.
`MB h4` = sound Maker-Breaker full minimax at horizon 4 (2 attacker placements),
node cap 3,000,000. Every row completed (`nodes < cap`) and returned `Unknown`
(no forced attacker win) — i.e. **no refutation**.

| position | `A` | `B` | `Φ` | profile `(n₁..n₅)` | att/def stones | root legal | MB h2 | MB h4 (nodes) | refutation |
|---|--:|--:|--:|---|---|--:|---|---|---|
| `es_core` | 0 | 13 | 0.833950 | (13,0,0,0,0) | 1 / 1 | 232 | Unknown | Unknown (285,427) | **none** |
| `es_core_translated` | 0 | 13 | 0.833950 | (13,0,0,0,0) | 1 / 1 | 232 | Unknown | Unknown (285,427) | **none** |
| `es_core_reflected` (q,r)→(r,q) | 0 | 13 | 0.833950 | (13,0,0,0,0) | 1 / 1 | 232 | Unknown | Unknown (285,259) | **none** |
| `blocker_2_0` | 0 | 14 | 0.898100 | (14,0,0,0,0) | 1 / 1 | 249 | Unknown | Unknown (303,600) | **none** |
| `blocker_3_0` | 0 | 15 | 0.962250 | (15,0,0,0,0) | 1 / 1 | 266 | Unknown | Unknown (322,351) | **none** |
| `blocker_1_-1` | 0 | 13 | 0.833950 | (13,0,0,0,0) | 1 / 1 | 232 | Unknown | Unknown (285,259) | **none** |
| `blocker_2_-2` | 0 | 14 | 0.898100 | (14,0,0,0,0) | 1 / 1 | 249 | Unknown | Unknown (303,320) | **none** |
| `enum2_-2_-2__-2_0` | 0 | 14 | 0.898100 | (14,0,0,0,0) | 1 / 2 | 282 | Unknown | Unknown (340,125) | **none** |

Notable defender resources observed: in every case the defender node in the MB
search **early-exits on its first surviving reply**, i.e. a surviving defender
move is found immediately at every reachable depth — the positions are visibly
"loose" for the defender at shallow horizons, consistent with Theorem 2.

### Engine cross-check (Section 5)

A minimal *reachable* Defender-`FirstStone` position (`Player0` opens `(0,0)`;
`Player1` plays `(-1,0),(-2,0)`; 3 stones) has `Φ = 1.714862` — **not** `< 1`.
This is itself an observation: a reachable `Φ < 1` Defender-`FirstStone`
position is hard to produce in a *short* legal game because the attacker is
under-blocked early (`Φ` starts well above 1). GAP-RAW is stated over general
blanket positions, which the harness handles directly. On this reachable board
the two independent solvers **agree**: Maker-Maker reference `solve(state, 2)`
→ `Unknown` (defender not lost, 82,685 nodes); blanket Maker-Breaker at
horizon 2 and 4 → `Unknown`, `refutation=false` (302,563 nodes at h4). No
attacker forced win by either semantics at the shallow horizon.

---

## The greedy dilemma (Section 2 — the substantive result)

On the canonical core `A={(0,0)}, D={(1,0)}` (`Φ = 0.8340`):

```
defender=dynamic_greedy  attack=es_cohort_target   → AttackerWon(ply 15)
defender=cohort_greedy   attack=es_cohort_target   → ScriptFoiled(ply 6)
defender=dynamic_greedy  attack=fresh_birth_line   → ScriptFoiled(ply 10)
defender=cohort_greedy   attack=fresh_birth_line   → AttackerWon(ply 12)
```

- `es_cohort_target` = the exact `ES_GLOBAL_BOUNDARY` Theorem 1 script
  `(2,-4),(2,2),(-5,0),(-4,0),(-3,0),(-2,0),(-1,0)`, completing the Q-line
  `W = {(-5,0)..(0,0)}`. This is one of the initial 13 alive windows (the only
  *alive* Q-window through `(0,0)`; the other five Q-windows die on `D=(1,0)`).
- `fresh_birth_line` = the attacker instead builds a brand-new 6-line
  `{(8,0)..(8,5)}` away from the cohort — a window *born* after the root.

**Why each greedy leaks.** Dynamic touched-window greedy enrolls the attacker's
newborn spray windows near `(2,±)` and chases them (high overlap ⇒ high danger),
abandoning the low-danger single window `W`; the attacker completes `W`
untouched. Fixed-cohort greedy commits to the initial 13 windows and never
scores births, so it pre-empts `W` in time (see the move trace below) but leaves
a fresh 6-line completely unguarded.

**Non-greedy defensive resource (what a proof must capture).** The winning
non-greedy move against the ES script is a **target-lock**: the cohort-greedy
defender's move sequence vs. the ES script is

```
GAPRAW_COHORT_DEFENSE_MOVES = [(-1,1), (0,-1), (-5,0), (0,1)]
```

The first two placements clear the high-overlap R/QR windows near `(0,0)`
(exactly the dynamic-greedy moves), but the **third placement `(-5,0)` locks the
Q-line target `W`** at ply 5 — *before* the attacker fills it (attacker reaches
`(-5,0)` only at ply 7). Dynamic greedy never makes this move because `W`'s
danger (`√3`, a single count-1 window) is dominated by the birth spray's danger.
The resource is: **do not let a low-danger but *completable* target starve while
chasing high-danger but well-covered clusters.** The dilemma table shows this
alone is insufficient — the same commitment blinds the defender to births — so
the GAP-RAW witness must be an *adaptive* combination: hold every completable
target of the initial cohort **and** re-enroll a birth line once it is itself a
genuine (non-redundant) threat. This matches why the named companion gaps
`GAP-GLOBAL-RENEWAL` / `GAP-AMORTIZED-ABANDONMENT` are open: the account must
discount abandoned births yet stay safe against their revival.

---

## Why exhaustive refutation cannot reach the regime (the depth wall)

- `Φ < 1` forbids a fast win: two count-5 windows already give `Φ = 1.1547 > 1`,
  so a `Φ < 1` root has **at most one** count-5 window — a single 5-threat that
  the defender (moving first, two stones) neutralizes immediately. No
  double-5 fork exists at `Φ < 1`.
- `ES_GLOBAL_BOUNDARY` Theorem 2 upgrades this to a *proven* **five-attacker-
  placement** safety certificate for `Φ < 1`.
- Therefore a refutation cannot surface before the **6th** attacker placement
  (ply ≥ 11 from the defender root); the known greedy refutation lands at the
  **7th** (ply 15).
- Sound exhaustive minimax must consider **all** defender replies (pre-emption
  is a real resource, so the defender move set cannot be soundly pruned for a
  refutation claim). With root branching ≈ 230–280 and growth as stones are
  added, exhaustion is feasible only to ~2 attacker placements (horizon 4,
  ≈ 3×10⁵ nodes here). Reaching ply ≥ 11 is ~230¹¹⁺ — out of range on one host.

So the exhaustive arm can only *confirm the already-proven shallow safety* (which
it does, cleanly), not decide GAP-RAW. Section 4's deeper `Unknown` rows
(`plies 8, 12`) are **budget-capped, not survival proofs** — indeed Section 2
exhibits the concrete cohort-greedy loss at ply 12; the capped exhaustive search
simply did not reach that specific line under 3,000,000 nodes.

---

## Observations for a future proof attempt

1. **No fixed greedy rule can be the witness.** The 2×2 dilemma is a clean,
   reproducible obstruction: target-lock defends cohort completions but not
   births; danger-greedy defends births but not starved cohort completions. The
   witness must interleave both, i.e. be genuinely adaptive.
2. **The right invariant is not `Φ` itself** (Cor. 2 already shows `Φ<1` is not
   renewable — one clean escape adds `4/√3 > 1`). A candidate witness should
   maintain a *two-part* account: a frozen fixed-family potential `Ψ_F` over the
   initial cohort (blocked forever by Theorem 1), **plus** a birth ledger that
   re-enrolls a newborn window only when it becomes a non-redundant threat and
   pays for it from same-turn maturity (the `GAP-AMORTIZED-ABANDONMENT` shape).
3. **The hard cell is the starved single target.** Empirically the position that
   breaks greedy is the lone alive collinear window left behind by the blocker
   (here `W`, the only alive Q-window through the attacker stone). A proof should
   treat "the unique alive window on an axis where all near copies are dead" as a
   named, always-serviced target.
4. **`Φ` near the threshold is dominated by count-1 mass.** Every battery member
   has profile `(n₁,0,0,0,0)` with `n₁ ∈ {13,14,15}` (`15` is the largest with
   `Φ<1`, since `16² = 256 > 243`). Count-2..5 profiles that stay `< 1` are much
   sparser (a single count-5 already spends `0.577`); the refutation-prone
   dense-threat shapes the task flags mostly sit at `Φ ≥ 1` and are outside the
   claim.

---

## Reproduction

Commit `dba6111d`; deterministic (no RNG on any scored path; fixed lexicographic
tie-breaks; fixed node caps). Set `CARGO_TARGET_DIR=.target-hunt`.

**Validation (13 tests, exact Φ + doc reproduction + engine differential):**
```
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  gap_raw_hunt -- --test-threads=1
```

**Full hunt report (all sections, machine-readable `GAPRAW_*` rows):**
```
CARGO_TARGET_DIR=.target-hunt cargo test -p hexfield_eq --lib --release \
  gap_raw_hunt::tests::gap_raw_hunt_report -- --ignored --nocapture --test-threads=1
```

Harness source (left in place): `packages/hexfield_eq/rust/src/gap_raw_hunt.rs`
(module registered test-only in `packages/hexfield_eq/rust/src/lib.rs`).
Node caps: MB scan 3,000,000; cohort probe 3,000,000. Runtime ≈ 4 s (report),
≈ 3 s (validation) on the shared host.
