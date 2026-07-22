# horizon_native

Dependency-free compiled search kernel for the Horizon R3 h13/h14 interaction
endpoint.  It does not read the engine or the registry.  A Python driver builds
the exact finite `NextModel` and streams models to this process; one JSON result
line is flushed after every `END`, so a killed batch retains completed rows.

## Input protocol

Whitespace is insignificant.  IDs must not contain whitespace.  Blank lines
and lines beginning with `#` are ignored.

```text
HORIZON_NATIVE_V1
MODEL <id> <horizon:13|14> <phase:first|second> [timeout_ms]
CELL <q> <r> <anchored:0|1> <root_legal:0|1>
... one CELL per quotient-universe physical cell, in index order ...
TA <cell-index>...       # one target-anchored residual window
OA <cell-index>...       # one opponent-anchored residual window
NE <cell-index>...       # one retained root-empty/near window
PREF <cell-index> [cell-index]  # optional first-action ordering hint
PREF_CELL <cell-index>          # optional required-cell ordering block
END
... more MODEL blocks ...
```

Every `TA`, `OA`, and `NE` line has one through six distinct in-range cell
indices.  Near windows are automatically added to both players' families,
matching `build_next_model` in `horizon_r3.py`.  The executable reconstructs
the exact R3 first-action set from the anchored/root-legal flags and cell
coordinates.  A `second` model is accepted only at h13.

Output is newline-delimited JSON.  `status` is `win`, `negative`, `timeout`,
or `error`; only the first two are verdicts.  Witnesses are universe indices
for the first attacker action and can be mapped through the Python model's
`cells` tuple.

`PREF_CELL` is ordering-only: every A1 action containing the indicated cell is
moved ahead of actions that do not, with the exhaustive remainder preserved.
The driver uses it for verifier-accepted certificate root `Choice` cells; a
one-cell Choice is never treated as a complete pair or as endpoint proof.

D1 is treated specially for true radius-eight legality.  After A1 the kernel
enumerates physical active pairs, checks that one cell is currently legal and
the other is currently legal or within distance eight of it, and only then
deduplicates by incidence-class pair.  It also retains the projected EMPTY
action and one singleton per incidence class having a currently legal member.
All later active carriers are saturated by a root or newly placed stone and
use the faster node-local incidence quotient.

Live residual families are reduced to their inclusion-minimal antichain:
duplicates and strict supersets are exact terminal/blocking duplicates.  At a
universal defender-pair node, a defender residual of size at most two is an
immediate loss for A; otherwise an A family of one/two-cell residuals with no
two-cover is an immediate quantified win for A.  The A3 endpoint is evaluated
directly from its already-live antichain by subtracting each candidate pair,
without rebuilding all root windows.  These are exact state identities, not
certificate-based pruning.

When the immediate A family does have a two-cover, only defender pairs that
actually hit every such residual are generated: every non-cover pair leaves
an immediate A completion and is answered by definition.  Symmetrically, A2
returns immediately on its own one/two-cell residual, and when D has immediate
threats it generates only their exact cover actions (or fails immediately if
no two-cover exists).  This retains the full quantifiers while avoiding
construction of trivial successors.

At ordinary pair nodes, every pair contained in an own residual of size
`k+2`—where `k` is the mover's placement quota after the pair—is placed in a
tactical prefix.  One selected cell has no surviving effect there, but two
cross the exact quota threshold.  The ordinary exhaustive stream remains as
the fallback, so this synergy rule is ordering-only.

## Build and test (serialize Cargo host-wide)

```powershell
$env:CARGO_TARGET_DIR='.scratch/horizon_native/.target'
cargo test --manifest-path .scratch/horizon_native/Cargo.toml
cargo build --release --manifest-path .scratch/horizon_native/Cargo.toml
```

Run a batch with:

```powershell
.scratch\horizon_native\.target\release\horizon_native.exe `
  --timeout-ms 0 --max-cache 2000000 < models.txt > results.jsonl
```

`--timeout-ms 0` means no deadline.  A positive timeout is checked inside the
search and produces a `timeout` status, never a false verdict.  The search is
single-threaded.

The repository-aware driver supplies the protocol and flushes evidence after
each root:

```powershell
# All 155 depth-13/14 registry certificates, unbounded per root
python .scratch/horizon_native/driver.py registry --per-root-ms 0 `
  --out .scratch/horizon_r4_phase1_registry.jsonl `
  --summary .scratch/horizon_r4_phase1_registry.json

# Frozen cohorts: fresh h14, SecondStone h13; opening is reported unsupported
python .scratch/horizon_native/driver.py cohorts --per-root-ms 250 `
  --out .scratch/horizon_r4_phase1_cohorts.jsonl `
  --summary .scratch/horizon_r4_phase1_cohorts.json

# Three immediate wins plus one universal-D1 negative, checked against Python
python .scratch/horizon_native/driver.py synthetic --per-root-ms 5000 `
  --out .scratch/horizon_r4_phase1_synthetic.jsonl `
  --summary .scratch/horizon_r4_phase1_synthetic.json

# Retry only timeout rows from an earlier registry JSONL (ordering hints only)
python .scratch/horizon_native/driver.py registry --per-root-ms 10000 `
  --prior-timeouts .scratch/horizon_r4_phase1_registry.jsonl `
  --out .scratch/horizon_r4_phase1_registry_retry.jsonl `
  --summary .scratch/horizon_r4_phase1_registry_retry.json

# Or target one/more IDs directly with repeated --id
python .scratch/horizon_native/driver.py registry --per-root-ms 10000 `
  --id human_41e78c67c2ac8570_p20 --id sp_0_p51

# Resume a killed cohort shard without rewriting its durable rows; repeat the
# exclusion flag when several completed shards precede the new one.
python .scratch/horizon_native/driver.py cohorts --cohort selfplay_v1 `
  --per-root-ms 10 --max-cache 50000 `
  --exclude-ids-from .scratch/horizon_r4_phase1_cohorts_selfplay_1_rows.jsonl `
  --limit 800 --out .scratch/horizon_r4_phase1_cohorts_selfplay_2_rows.jsonl `
  --summary .scratch/horizon_r4_phase1_cohorts_selfplay_2.json
```
