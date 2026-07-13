# TSS v2 — main_3 Rung-Enablement Run-Book

Everything below is BUILT, tested, and default-off on branch
`claude/tss-v2-build`. Attaching to the live main_3 run = merge the branch
into the run branch → rebuild the extension → relaunch at an epoch boundary
(the launch script doubles as the relauncher) → flip ONE flag per rung.
Revert of any rung = flip the flag back + relaunch (checkpoints unaffected;
target-semantics rungs additionally note `target_regime` below).

## The flags

| Flag (SelfplayConfig / TOML `[model.config.selfplay]`) | What it does |
|---|---|
| `tss_interior_guard` (bool) | Lever 0: interior nodes at the fully-forced boundary (`k == B`) narrow to the hitting-cell universe. |
| `tss_policy_target_sharpen` (bool) | Lever 1: recorded visit/π′ targets get the guard math; shards carry `target_regime=1`. |
| `tss_solver_mode` (0/1/2/3) | Stage 4 ladder: off / shadow / +verified hard LOSS / +verified hard WIN at gated leaves. |
| `tss_solver_node_cap` (int, 2000) | Deterministic per-solve node cap. Size from shadow histograms. |
| `tss_solver_sample_16` (int, 16) | Leaf solve subsample (of 16). Lower it if shadow shows CPU drag. |
| `tss_solver_root_guard` (bool) | Rung 6: verified root solves; a certified WIN's move is always played (`action_selection="tss_deep_root_win"`); row proofs deep-upgrade. |
| `tss_solver_async` (bool) | Async rung (2026-07-13): gated leaves ENQUEUE to a background worker pool (identical solver→verifier→mint path); results drain into the memo and are consumed by the selection descent-stop on later visits. Verified `Done` entries persist across moves (binding re-checked at every consumption). Flag-ON self-play is NOT bit-reproducible (arrival timing); flag-off unchanged. Wired for continuous self-play AND the lockstep eval/arena loop. |
| `tss_solver_async_threads` (int, 8) | Pool worker threads (Rust clamps to 1–32). |
| `tss_solver_async_inline_16` (int, 0) | Hybrid inline tier under async: gated leaves with `(hash & 0xF)` below this solve inline (first-touch consumption, the pre-async path); the rest enqueue. Deploy shape: `sample_16=16` + `async=true` + `inline_16=4` keeps the proven 4/16 inline tier verbatim and adds pool coverage for the other 12/16 at ~zero critical-path cost. |

## Rung order (one per relaunch; watch ≥1–2 epochs + the next eval before the next rung)

0. **Merge only, all flags off.** This alone turns on the Stage-0 shadow
   telemetry (λ¹ class column, proof scalar, counters). Play and targets are
   bit-identical (golden-digest-proven). Start collecting the gate metrics.
1. **`tss_solver_mode=1` (shadow).** Adds deep solves + verification at gated
   leaves, consuming nothing (bit-identical, twin-run-proven). Collects
   `deep_*` histograms — the cost/UNKNOWN data that sizes `node_cap` and
   decides the vise kill-criterion.
2. **`tss_interior_guard=true`.** First behavior change (search-only).
3. **`tss_policy_target_sharpen=true`** — ONLY if the shadow gate says it
   matters: `tss.win_retained_mass_mean` clearly below ~1.0 (mass actually
   moves). If ≈1.0, skip: π′ is already proof-sharp.
4. **`tss_solver_mode=2`** (verified hard LOSS + eval elision). Before/after,
   deliberately probe avoided lines offline (false LOSS is the silent
   failure): re-run the solver harness + spot-solve avoided positions.
5. **`tss_solver_mode=3`** (verified hard WIN).
6. **`tss_solver_root_guard=true`** (serve too — the eval arena inherits it).
7. **Lever-2 train-read swap (rung 5 of the plan, NOT YET BUILT — build at
   this moment):** value target := `tss_proof` where nonzero at expand (both
   backends), proof-valid-under-truncation mask. Gate: the
   `tss.proof_disagreements` stream shows deep proofs actually disagreeing
   with outcomes often enough to matter. Both labels are already captured per
   row (`value` + `tss_proof`), so no data is being lost meanwhile.

## Watch on every rung (epoch JSON `hexfield.selfplay.epoch_*.json`, `tss` block)

- **`deep_verify_failed` — MUST BE 0.** Nonzero = a solver claim failed its
  certificate check (values degraded safely, but the solver has a bug): set
  `tss_solver_mode=0`, keep shadow data, investigate before re-enabling.
- `injection_fire_rate` — the closed-loop metric; should fall as the net
  internalizes tactics.
- `win_retained_mass_mean` / `sharpened_rows` — Lever-1 gate + effect.
- `forced_defense_fraction`, `prune_eligible/dropped` — Lever-0 scale.
- `deep_calls/win/loss/unknown/nodes` by epoch — solver cost + the vise check
  (UNKNOWN rate at production caps in the threat-dense endgame).
- `proof_rows` / `proof_disagreements` — the Lever-2 gate.
- Async rung: `async_dropped` ~0 (bounded queue 4096; sustained drops ⇒ more
  threads/wider queue — never a correctness issue, dropped leaves take the
  plain net eval); `async_stale` informational (late results still land memo
  entries and serve later moves); `deep_hard_backups` vs the pre-async
  epochs — the async tier consumes less per solve than inline first-touch
  (descent-stop needs a re-visit), which is why the deploy shape keeps the
  inline tier. Consumption regression below inline-only ⇒ raise `inline_16`.
- Standard health: pinned entropy metric (L0+L1 compound sharpening), pos/s,
  eval cadence h2h (pool/Strix/SealBot), earlyoom/VM memory.

## Mechanics reminders

- Build: `maturin develop --release -m packages/hexfield_eq/Cargo.toml` in the
  `hexfield-dev` venv FROM THE RUN TREE being deployed (editable install drops
  the `.so` into that tree). Never build from a different tree into the live
  one mid-epoch; relaunch picks it up.
- Tests: `cargo test -p hexfield_eq` (96) and
  `PYTHONPATH=packages/hexfield_eq/python:<main>/packages/hexo_utils/python
  pytest tests/test_hexfield_eq_tss_shadow.py` (18) must be green pre-merge.
- The Stage-0 golden digest (`tests/data/hexfield_eq_tss_stage0_golden.json`)
  pins flag-off bit-identity; regenerate ONLY for an intentional
  behavior-change baseline (two-build procedure in the test docstring).
- Solver memory: per-solve TT capped at 256 KiB, per-move memo ≤8192 entries
  (cleared every move inline; async retains verified `Done` entries for the
  life of the game's search object, still ≤8192) — no unbounded growth (host
  earlyoom discipline). Async adds: request queue ≤4096 state clones + one
  persistent solver TT per worker thread (byte-capped per solve).
