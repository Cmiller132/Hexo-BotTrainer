# Group 2, round 5 — C1 migration build (step 1 of 2)

Date: 2026-07-16 (America/New_York)

Base confirmed before edits: `961623a7` (round-4 consolidation). No git
commit is permitted or attempted. Existing untracked prompt, PID, executable,
target, and `.codex-round5/` artifacts are user-owned and remain untouched.

## Checkpoints

### C0 — production map and migration boundary confirmed

Read `.codex-group2/round4-progress.md` in full. The default live route remains:
`tss_solve_verified` (256 KiB) → `solve_goal` → `split_tt_cap` → `prove_for` →
`prove_for_at_depth` → legacy `SearchContext`, with the count>=3 claimant
generator. `TssSolverSlot`, the payload root guard, and async workers continue
to construct persistent `TssSolver::default()` instances.

Checkpoint boundary: no code edited; HEAD and tracked cleanliness confirmed.

### C1 — default-off `WidePnSearch` narrow compatibility mode built

- Added `WidthOptions::narrow_compat()` and the private
  `narrow_engine_migration` dispatcher bit. `WidthOptions::default()` leaves it
  false, and no production caller enables it.
- When the bit is true, `prove_for` enters
  `WidePnSearch::prove_narrow_compat`. The compatibility entry point preserves
  the historical recursive DFS expansion order, absolute ply/deadline checks,
  defender enumeration/ordering, certificate arena construction/compaction,
  and solve-local plus persistent shared-TT composition.
- Renamed the implementation state to `NarrowCompatSearch`; the default-off
  legacy route retains the explicit `SearchContext` type alias. This makes the
  round-6 deletion boundary reviewable without changing the default route.
- The migrated branch names the generalized count-threshold generator at
  threshold 3. The historical `threat_creating_moves` wrapper remains live in
  the default-off route and is not deleted this round.

Architectural constraint: the ordinary wide PN frontier cannot provide exact
legacy node identity. It persistently retains siblings, selects work by proof
and disproof numbers, stages depth, and counts frontier expansions; the narrow
DFS commits to deterministic recursive child order and counts recursive node
entries. Even when both find the same theorem and certificate, their expansion
sequence and cap boundary are not identical. The compatibility mode therefore
uses the wide engine dispatch seam while deliberately preserving the narrow
DFS scheduler. This is not a relaxation of identity; it is required to meet
`nodes ==` exactly.

Checkpoint boundary: compatibility mode and default-off dispatcher compile.

### C2 — exact identity harness green

Added ignored release harness `tss_round5_narrow_compat_identity`. It compares
the legacy and migration-on paths for:

- exact status, node count, TT hit count, peak TT bytes, structural certificate,
  and an explicit canonical binary encoding of every certificate field;
- exact solve-local TT terminal layout/signature and exact persistent shared-TT
  contents/accounting;
- cache-warm second-solve behavior, including a positive shared-fragment hit
  and reduced warm node count;
- default narrow deterministic, forced-defense, immediate-WIN, dual-LOSS,
  spare-tempo, deep-universal, goal-filter, and expired-horizon fixtures;
- forced full-key collisions, zone-enabled D6 images, and independent verifier
  plus dispatch-oracle acceptance for every hard claim;
- 512 fixed-seed legal positions spanning Opening, FirstStone, and SecondStone,
  both one-sided goals, node caps 0/1/32/64, zero/small TT, the production
  256 KiB TT cap, and finite semantic deadlines.

Every assertion includes an exact coordinate replay. A status mismatch is an
unconditional test failure; none occurred.

Result: **PASS: 1/0**, 0.10 s test time, observed free RAM 12.85 GiB.
The final-source rerun after adding the explicit binary encoder also passed
1/0 in 0.10 s with 12.84 GiB free.

```powershell
$free = Get-CimInstance Win32_OperatingSystem | ForEach-Object { $_.FreePhysicalMemory / 1MB }
if ($free -le 10) { throw 'Free RAM must exceed 10 GiB before identity gate' }
$env:CARGO_TARGET_DIR='.target-codex'
cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq tss_round5_narrow_compat_identity -- --ignored --test-threads=1 --nocapture
```

Checkpoint boundary: every round-5 identity dimension is green. No mismatch or
WIN-vs-LOSS finding exists.

### C3 — flag-off and Stage-0 gates green

- Final-source default release suite: **PASS: 95 passed / 0 failed / 18
  ignored**, 2.84 s test time. The sole new ignored test is the round-5 identity
  harness; the 95 default tests are unchanged.
- Stage-0 golden digest in the documented WSL CPython 3.12 `hexfield-dev`
  environment: **PASS: 1/0**. The final-source rerun took 4.21 s pytest time.
- `WidthOptions::default()` still leaves `narrow_engine_migration=false`.
  Grep/audit confirms no production caller constructs `narrow_compat()`.

Checkpoint boundary: flag-off Rust behavior and the outer trainer stream are
green.

### C4 — witness and official corpus gates green

- `double_fork_compact`: **WIN / 409 nodes / 51 TT hits / 67,177,998 peak TT
  bytes / 36 ms / strict verifier accepted** at the 10k rung.
- Official all-19 at `TSS_BACKWALK_TT_BYTES=2147483648`: **PASS,
  `CORPUS_DONE failures=0`**, all 14 WIN rows and all five NO non-WIN rows
  accepted, 439.66 s test time.
- There was no WIN-vs-LOSS disagreement at matched semantics.

Operational note: the documented WSL Stage-0 build and native Windows Cargo
share `.target-codex`. Immediately after Stage-0, an unqualified Windows build
found a Linux `hexo_engine` rlib and stopped before starting the witness. A
scoped `cargo clean -p` removed zero files. The native gates were therefore
rerun with `--target x86_64-pc-windows-msvc`, still wholly under the mandated
`.target-codex`; this isolates native artifacts without deleting unrelated
user/build state. One attempted rebuild was also stopped before compilation
when free RAM read 9.37 GiB; work resumed only after it recovered to 13.44 GiB.
Neither event was a solver execution or semantic mismatch.

Checkpoint boundary: all deep evidence gates are green.

## Required gates

| Gate | Result | Regeneration command |
|---|---|---|
| Round-5 identity | **PASS: 1/0** | command in C2 |
| Default release suite, flag off | **PASS: 95/0** | `$env:CARGO_TARGET_DIR='.target-codex'; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq -- --test-threads=1` |
| Stage-0 golden digest | **PASS: 1/0** | documented WSL command below |
| Official all-19, 2 GiB | **PASS: failures=0** | `$env:CARGO_TARGET_DIR='.target-codex'; $env:TSS_BACKWALK_TT_BYTES='2147483648'; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture` |
| `double_fork_compact` consume witness | **PASS: WIN/409, verified** | `$env:CARGO_TARGET_DIR='.target-codex'; $env:TSS_R3_CAP='10000'; $env:TSS_BACKWALK_TT_BYTES='536870912'; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq tss_round3_consume_witness -- --ignored --test-threads=1 --nocapture` |

Stage-0 documented environment:

```powershell
$free = Get-CimInstance Win32_OperatingSystem | ForEach-Object { $_.FreePhysicalMemory / 1MB }
if ($free -le 10) { throw 'Free RAM must exceed 10 GiB before Stage-0 build' }
wsl -e bash -lc 'set -euo pipefail; source ~/.cargo/env; source /root/.venvs/hexfield-dev/bin/activate; cd /mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/tss-vcf-width; export CARGO_TARGET_DIR=.target-codex; maturin develop --release -m packages/hexfield_eq/Cargo.toml; PYTHONPATH=packages/hexfield_eq/python:packages/hexo_runner/python:packages/hexo_models/python:/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python python -m pytest tests/test_hexfield_eq_tss_shadow.py::test_stage0_digest_matches_golden -q'
```

## Round-6 flip/deletion checklist (not performed this round)

1. Re-run the complete round-5 identity harness at the then-current tip.
2. Flip `TssSolver::default()`/`prove_for` narrow dispatch to the compatibility
   route in a separately reviewed change; do not alter caller caps or TT split.
3. Run the complete identity harness a second time after the flip.
4. Delete the legacy `SearchContext` alias/route and the historical
   `threat_creating_moves` count>=3 wrapper; retain the generalized threshold-3
   generator used by narrow compatibility mode.
5. Re-run identity after deletion, then the default release suite, Stage-0 WSL
   golden digest, verifier/mutation gates, `double_fork_compact` witness, and
   official serialized all-19 gate at the 2 GiB profile.
6. Audit `TssSolverSlot`, payload root guard, async persistent-worker and
   panic-reset construction sites; confirm no caller-specific dispatch or cap
   change accompanied the engine flip.

## Final audit

- `rustfmt --edition 2021` was run only on the edited Rust file;
  `rustfmt --edition 2021 --check packages/hexfield_eq/rust/src/tss_solver.rs`
  passes.
- `git diff --check` passes.
- Tracked implementation diff is confined to `tss_solver.rs`; this progress
  memo is new. Existing untracked user/build artifacts remain untouched.
- `rg` finds the migration constructor only at its definition and in the
  ignored identity harness. No production consumer enables it.
- The `SearchContext` alias and historical `threat_creating_moves` wrapper are
  both still present, as required for round 5. No default dispatcher flip or
  deletion occurred.
- Final HEAD remains `961623a7`; no commit was attempted.
- Final free physical RAM reading: 12.77 GiB.

Final disposition: **complete and green for step 1 of 2**. Round 6 remains the
separately gated second identity run, flip, legacy deletion, and final gate
sequence listed above.
