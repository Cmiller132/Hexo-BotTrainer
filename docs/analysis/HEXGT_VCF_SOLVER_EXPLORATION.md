# hexgt VCF/VCDT forcing-solver — exploration, benchmark & recommendation (Phase 6)

Companion to [`HEXGT_TSS_AND_SOFT_VALUE_DESIGN.md`](HEXGT_TSS_AND_SOFT_VALUE_DESIGN.md)
PART 1 (d). The TSS plan defers the deep multi-ply forcing solver; this is the
data behind that decision, now grounded in a working prototype rather than prior.

**Recommendation: DEFER.** Keep the depth-1 hitting-set override (Phase 5) +
tactical injection (Phase 4) + threat features (Phase 3) as the tactical layer.
The deep solver is **cheap enough to run root-only with a hard cap** but is **not
sound enough to feed the value target**, and its marginal benefit over the
override is unmeasured. Revisit only if trained defense still lags.

## What was built

`packages/hexo_models/hexgt/rust/src/vcf.rs` — a depth-bounded AND-OR forcing
search restricted to threat moves (the only way to bound Connect6's 2-stone
branching), exposed via the `hexgt_vcf_solve(state, ply_budget, node_cap,
time_limit_ms)` PyO3 hook. It is **not** wired into the live MCTS. Harness:
`scripts/_vcf_bench.py`; sanity tests: `tests/test_hexgt_vcf.py`.

- OR node (attacker to move): win if it has a win-now this turn (the Phase-5
  `analyze`); else try each forcing placement (extends a window to >=4), win if
  any line wins.
- AND node (defender to move): refuted if the defender has its own win-now; win
  if the attacker's threats are unanswerable within the defender's placements
  (`min_hitting_set > B`); else the defender must try every neutralizing cell and
  the attacker wins only if **every** defense still loses.

## Benchmark (CPU, hexgt-build venv)

Representative random midgame positions; per-call cost of the root forcing search.

| Cap (time / nodes) | p50 | mean | p99 | max | hit-cap rate | forced-wins found |
|---|---|---|---|---|---|---|
| 100 ms / 2,000,000 | 5 µs | 466 µs | ~274 µs | 100 ms | rare | ~1.7 % |
| **5 ms / 50,000** | **8 µs** | **158 µs** | **5.0 ms** | 5.1 ms | **11 / 1000 (1.1 %)** | ~1 % |
| 2 ms / 20,000 | 8 µs | 119 µs | 2.05 ms | 2.1 ms | 18 / 1000 (1.8 %) | ~0.9 % |

For context, one MCTS move (visits=128) is orders of magnitude more expensive than
a single root VCF call (≈ 0.0 % of a move at the mean). The cost profile:

- **Median is ~free** (8 µs): most positions have no forcing moves, so the search
  returns immediately.
- **Heavy tail**: ~1 % of positions explode (many count-3 attacker windows ⇒ wide
  forcing branching) and run to the cap. A hard time/node cap bounds this cleanly
  (5 ms / 50k nodes ⇒ worst case 5 ms, p99 = the cap, only ~1 % of positions).
- **Root-only is affordable**; **per-node (every leaf) is not** — the tail × the
  thousands of leaves per search would dominate, and the depth-1 override already
  covers leaves at ~free cost.

## Correctness

Sanity (from the harness): finds the win-now on an own count-4 at FirstStone
(TEST G, 1 node); reports **no** forced win when the side to move has no attack of
its own (TEST D, ~121 nodes); returns instantly with no forcing moves at the
opening.

**The central caveat (why it stays out of the value signal):** the solver is
**forcing-restricted** — the defender move set is only threat-neutralizing cells,
so a defender **counter-threat / quiet refutation is never considered**. A `win`
from this prototype is therefore **not a sound proof**; it can over-claim. (It did
not false-positive on the cases tried, but the structural gap is real and is the
classic forcing-search incompleteness the design doc flags.) Wiring an unsound
±1 into the soft-Z / override value target would inject exactly the
miscalibration the plan set out to cure.

Meanwhile the **depth-1 override is exact** for 1-ply forced positions at ~free
cost, and deeper forced lines are caught **progressively as MCTS deepens** (leaves
nearer the terminal hit the override). So much of the multi-ply value is already
approximated by override + injection + search depth; the *marginal* tactical gain
of a deep solver on top is unmeasured.

## If it is ever adopted (bounded form)

- **Root-only**, hard-capped at **ply ≤ 8, node_cap ≤ 50k, time ≤ 5 ms**.
- As a **move-selection bias only** (prefer a found forcing win at the root),
  **never** as a value-target ±1 — until soundness is addressed (add defender
  counter-threat generation and prove termination), which is a much larger change.
- Gate adoption on a measured **H2H / defensive-calibration** improvement over the
  override-only configuration; do not adopt on cost grounds alone.

## Verdict

Defer. The Phase 3–5 tactical layer (features + injection + exact depth-1
override) is the right cost/correctness point for now; the deep solver is a
future, bounded, root-only, move-selection-only option pending evidence that
trained defense still needs it.
