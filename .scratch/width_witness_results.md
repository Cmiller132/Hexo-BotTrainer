# Width-witness evidence ledger

All claims below are measurements or artifact reads; no tracked engine file was edited.

## Direct baseline reproduction

Command (run from `../opening-atlas` against the three moves in
`.scratch/width_witness_moves.txt`):

```powershell
$env:CARGO_TARGET_DIR='<research-div>/.cargo-target'
$env:OPENING_ATLAS_MOVES_FILE='<research-div>/.scratch/width_witness_moves.txt'
$env:OPENING_ATLAS_WIDTH='vcf_pair_complete'
$env:OPENING_ATLAS_GOAL='win'
$env:OPENING_ATLAS_NODE_LADDER='100000'
$env:OPENING_ATLAS_TT_BYTES='268435456'
$env:OPENING_ATLAS_UNBOUNDED='1'
$env:OPENING_ATLAS_WALL_SECONDS='600'
cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq opening_atlas_pass1 -- --ignored --nocapture
```

Natural exhaustion (not cap binding):

| Root | status | nodes | expansions | TT entries | peak TT bytes |
|---|---:|---:|---:|---:|---:|
| `oa-773ca1a59e95f4e1` | UNKNOWN | 42 | 41 | 68 | 5,492 |
| `oa-0153903c5a863630` | UNKNOWN | 42 | 41 | 68 | 5,492 |
| `oa-6fda812864c6d19a` | UNKNOWN | 20 | 19 | 27 | 2,178 |

## Certified lifts and exact omitted turns

Atlas source: `../opening-atlas/atlas-web/data/atlas.json` and
`../opening-atlas/OPENING_ATLAS_MAXSOLVE_RAW.txt`.

| parent | root moves | omitted claimant turn/spare | decisive child | child verdict / solve nodes / cert nodes | parent solve nodes / cert nodes |
|---|---|---|---|---|---:|
| `oa-0153903c5a863630` | `0,0;-6,1;-7,2;0,1;0,2;-7,1;-8,2` | pair `(0,-1),(-1,2)` | `oa-5166cc20e3ecc7b7` | LOSS / 2,009 / 521 | 2,009 / 523 |
| `oa-773ca1a59e95f4e1` | `0,0;-8,0;-8,1;1,-1;2,-2;-8,2;-9,2` | pair `(3,-3),(3,-2)` | `oa-3fa9037cf3f1144b` | LOSS / 2,023 / 521 | 2,023 / 523 |
| `oa-6fda812864c6d19a` | `0,0;-4,8;4,-8;-1,0;1,0;1,-1;-1,1;-2,0` | SecondStone spare `(0,-2)` | `oa-9e524a9bf4fab453` (D6 image) | LOSS / 1,402 / 521 | 2,056 / 600 |

Every decisive child is `certified=1`, claimant P0, canonical `TssVerifier`
accepted (`d6_verified >= 2`), with a terminal 20-placement win line.  Every
parent is also `certified=1` with a terminal line.  The maxsolve harness
re-solved each child, prepended the legal same-claimant Choice move(s), rebuilt
the parent certificate, and strictly verified the parent.

Exact parent terminal lines:

- `oa-0153903c5a863630` (22):
  `0,-1;-1,2;0,-3;0,3;1,0;2,-1;-3,4;3,-2;1,2;2,2;-3,2;3,2;2,0;2,1;2,-3;2,3;3,0;4,-1;-2,0;-1,0;4,0;5,0`
- `oa-773ca1a59e95f4e1` (22):
  `3,-3;3,-2;-2,2;4,-4;1,-2;0,-2;-2,-2;4,-2;3,-4;3,-5;3,-7;3,-1;1,-3;2,-4;-2,0;4,-6;1,-4;-2,-4;-1,-4;1,-6;1,-5;1,0`
- `oa-6fda812864c6d19a` (21):
  `0,-2;-4,0;2,0;0,-1;0,-3;0,-5;0,1;1,-2;-2,1;3,-4;-3,2;-2,-2;-2,-1;-2,-4;-2,2;-1,-2;-5,2;-4,-2;-4,1;-3,-2;2,-2`

In each witness the already-forcing first stone/state has three claimant
count-4 windows, minimum hitting number 2, and zero defender win-now windows.
The omitted spare is not in the exact `vcf_pair_complete` second universe.  It
creates no immediate count-4 threat; it promotes count-1 live windows to
count-2 on two axes with multiplicities respectively `(5,5)`, `(5,5)`, and
`(4,4)`.

## Candidate-count cohort measurement

Reproduce with:

```powershell
python .scratch/measure_j2.py
```

`J2near` means an outside-normal-universe spare supported by at least four
claimant-live count-1 windows on each of at least two axes, evaluated after
the forcing first stone (or in the existing SecondStone state). Counts are an
exact geometry replay/candidate-count proxy on real rows, not solver runtime.

| cohort | rows / usable | forcing-eligible roots | roots with J2near | added candidates among eligible (mean / p50 / p90 / max) | `(current+added)/current` (mean / p50 / p90 / max) |
|---|---:|---:|---:|---:|---:|
| grinds | 248 / 248 | 0 | 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| puzzle_v3 | 468 / 463 | 21 | 15 | 4.43 / 2 / 14 / 20 | 1.191 / 1.067 / 1.519 / 2.053 |
| human_v1 | 2,720 / 2,701 | 100 | 38 | 1.22 / 0 / 4 / 13 | 1.039 / 1.000 / 1.057 / 2.250 |
| selfplay_v1 | 3,255 / 3,124 | 4 | 1 | 0.25 / 0 / 0 / 1 | 1.010 / 1.000 / 1.000 / 1.040 |

Exact witness root costs are `19 + 20 = 39` accepted forcing pairs for each
FirstStone root (2.053x), and `8 + 4 = 12` accepted second placements for the
SecondStone root (1.5x). Broad all-count1 widening would instead add
`110, 110, 56`; the looser two-axis J2 tier has puzzle mean/p90 multipliers
`1.519/2.074` and human `1.423/1.811` (human max 5.167).

## Scratch analyzers

- `.scratch/analyze_width_witnesses.py`: exact witness threats, hitting number,
  candidate membership, and axis-support threshold sweep.
- `.scratch/measure_j2.py`: real-cohort prevalence and candidate-count proxy.
- `.scratch/width_witness_moves.txt`: three exact roots for the Rust harness.
