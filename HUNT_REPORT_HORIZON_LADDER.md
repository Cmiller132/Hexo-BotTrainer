# NQ8 machine hunt — semantic-horizon ladder

Status: **STOP — finite-horizon `round3_consume` WIN certificate rejected by
the independent verifier. DO NOT BUILD.** The forcing cohorts completed before
the stop and were verdict-identical, but the compact cohort exposed a bounded
horizon certificate-contract bug. Per the campaign rule, no human roots were
run after that finding.

## Resolution-depth census (report first)

Resolution depth is `max(exact typed-leaf resolution clock) - root
placements_made`. Every value below came from a cold `semantic_horizon =
u32::MAX` WIN certificate accepted by `TssVerifier`.

| completed direct profile | verified WIN certificates | <=8 | <=12 | <=16 | <=24 | <=32 |
|---|---:|---:|---:|---:|---:|---:|
| forcing 10k | 8 | 0 (0%) | 0 (0%) | 2 (25.0%) | 6 (75.0%) | 8 (100%) |
| forcing 100k | 12 | 0 (0%) | 0 (0%) | 2 (16.7%) | 6 (50.0%) | 11 (91.7%) |
| compact direct | 1 | 0 (0%) | 1 (100%) | 1 (100%) | 1 (100%) | 1 (100%) |
| all measured certificate instances | 21 | 0 (0%) | 1 (4.8%) | 5 (23.8%) | 13 (61.9%) | 20 (95.2%) |

The 10k forcing WINs are a subset of the 100k WINs and have the same depths.
Using each root only once (the 100k forcing result plus compact), the measured
unique-root distribution is 0/13 <=8, 1/13 <=12, 3/13 <=16, 7/13 <=24, and
12/13 <=32. The 12 forcing-100k depths are
`14,14,18,18,22,22,26,26,26,29,30,34`; compact is depth 9.

The repository contains forcing/human root corpora and prior aggregate logs,
not serialized certificates from which exact leaf clocks can be recovered.
Accordingly, “existing corpus data” was covered by freshly solving the 19
forcing roots. The human-corpus census is missing because the mandated STOP
occurred first; the table is not presented as a full-population estimate.

## Ladder versus direct: completed forcing cohorts

Each unique `(root, relative horizon, rung cap)` was run cold with a fresh
solver and the census gate on. Identical rungs shared by multiple candidate
schedules were measured once and composed additively; they did not share a TT
or fragments. The one cold direct solve was also the unchanged gate-off
unbounded final-rung measurement. Thus node/expansion totals are exact
deterministic sums; wall totals are paired additive measurements rather than
repeated noisy executions of identical final rungs.

“Economic W/L/T” compares total ladder nodes with direct nodes per root; it is
not the proof status. All proof statuses in this table were identical to the
direct status.

| cohort | schedule | economic W/L/T | nodes direct -> ladder | expansions direct -> ladder | wall direct -> ladder |
|---|---|---:|---:|---:|---:|
| forcing 10k | 8,16,24,32,final | 2/17/0 | 100,300 -> 249,316 (+148.6%) | 100,281 -> 249,239 (+148.5%) | 9.917s -> 23.154s (+133.5%) |
| forcing 10k | 8,16,final | 1/18/0 | 100,300 -> 118,668 (+18.3%) | 100,281 -> 118,614 (+18.3%) | 9.917s -> 11.009s (+11.0%) |
| forcing 10k | 16,final | 3/16/0 | 100,300 -> 117,972 (+17.6%) | 100,281 -> 117,937 (+17.6%) | 9.917s -> 10.989s (+10.8%) |
| forcing 10k | 8,16,24,32 @ 1k,final | 1/18/0 | 100,300 -> 130,996 (+30.6%) | 100,281 -> 130,912 (+30.5%) | 9.917s -> 12.923s (+30.3%) |
| forcing 100k | 8,16,24,32,final | 3/16/0 | 419,445 -> 721,413 (+72.0%) | 419,426 -> 721,341 (+72.0%) | 40.943s -> 69.590s (+70.0%) |
| forcing 100k | 8,16,final | 1/18/0 | 419,445 -> 440,206 (+5.0%) | 419,426 -> 440,152 (+4.9%) | 40.943s -> 42.120s (+2.9%) |
| forcing 100k | 16,final | 3/16/0 | 419,445 -> 439,510 (+4.8%) | 419,426 -> 439,475 (+4.8%) | 40.943s -> 42.099s (+2.8%) |
| forcing 100k | 8,16,24,32 @ 1k,final | 1/18/0 | 419,445 -> 450,141 (+7.3%) | 419,426 -> 450,057 (+7.3%) | 40.943s -> 43.946s (+7.3%) |

The best tested schedule was the single `h16` probe, and it still lost 17.6%
of nodes at 10k and 4.8% at 100k. The `h8` rung never found a forcing WIN. The
full ladder found every direct forcing WIN by h32 (and no direct UNKNOWN), but
the failed intermediate work overwhelmed that benefit.

## Verdict identity and STOP finding

The completed forcing coverage was 38 direct rows and 152 schedule
comparisons:

- forcing 10k direct: 8 WIN / 11 UNKNOWN;
- forcing 100k direct: 12 WIN / 7 UNKNOWN;
- zero ladder/direct status differences across all four schedules;
- every produced forcing WIN certificate verifier-accepted; and
- no forcing-NO row returned WIN.

The next root triggered the required stop:

```text
root=double_fork_compact
root placements=36
direct unbounded: WIN, nodes=2409, expansions=2408,
                  verified certificate, relative resolution=9
h8 (absolute 44): UNKNOWN, nodes=482, expansions=481
h16 (absolute 52): solver returned WIN, nodes=2409, expansions=2408,
                   cert_nodes=1917, relative resolution=9,
                   VERIFIER REJECTED

R3_VERIFY_TRACE node=1915 failure=clock
  stored_d=8 derived_B=4 build_horizon=52 semantic_horizon=52
```

The failure is at a zoned `Universal` node reached at placement 37. Its zone
build clock equals the certificate clock, but its stored global remaining
defender allowance (`d=8`) disagrees with the verifier-reconstructed local
proof budget (`B=4`). The materializer's `rebase_zone_distances` currently
relabels zones from the caller's external semantic horizon, while the verifier
requires the exact local budget induced by the materialized proof DAG. An
extra-slack finite horizon can therefore return a certificate that the
independent checker rejects.

This result does **not** show that the compact game position is not a WIN: its
unbounded certificate verifies. It shows that the bounded WIN cannot be used
for monotone transfer, so the proposed ladder is not currently a
verifier-closed orchestration of the existing solver. No compact ladder
summary, human results, or global verdict-identity claim is valid after this
point.

## Waste anatomy and fragment-reuse bound

Failed-rung nodes below include only economically losing roots. “Win saving”
is the direct-minus-ladder node saving on economically winning roots, grouped
by the successful rung.

| cohort | schedule | failed-rung nodes on losing roots | winning roots / saving |
|---|---|---|---|
| forcing 10k | 8,16,24,32 | h8 576; h16 17,907; h24 58,620; h32 72,288 | h16: 1 / 296; h24: 1 / 64 |
| forcing 10k | 8,16 | h8 592; h16 18,087 | h16: 1 / 296 |
| forcing 10k | 16 | h16 18,087 | h16: 3 / 415 total |
| forcing 10k | capped full | h8 592; h16 6,651; h24 11,582; h32 12,182 | h16: 1 / 296 |
| forcing 100k | 8,16,24,32 | h8 560; h16 20,124; h24 144,510; h32 143,906 | h16/h24/h32: 3 / 7,576 total |
| forcing 100k | 8,16 | h8 592; h16 20,480 | h16: 1 / 296 |
| forcing 100k | 16 | h16 20,480 | h16: 3 / 415 total |
| forcing 100k | capped full | h8 592; h16 6,651; h24 11,582; h32 12,182 | h16: 1 / 296 |

The reusable object would be a verified, horizon-labelled positive fragment
or still-open position entry carried from a failed bounded rung into a later
rung/final solve (the U18/T10 persistence direction). The data do not support
building it for this ladder. Under the deliberately optimistic assumption
that all earlier-rung work overlaps later work perfectly, charging each root
only its largest rung reduces `h16` to 99,885 versus 100,300 nodes at 10k
(0.414% prospective saving) and 419,030 versus 419,445 at 100k (0.099%). The
full ladder's same optimistic bound is +0.673% at 10k but still **5.95% worse**
at 100k because some later bounded successes themselves cost more than direct.
Real fragment reuse cannot be assumed to attain perfect overlap, and its clock
and zone proof obligations are substantial.

## Recommendation

**DON'T BUILD. No consume-round spec is authorized from NQ8.** There are two
independent reasons:

1. correctness: a finite-horizon `round3_consume` WIN escaped the solver with
   a verifier-invalid zone budget, violating the ladder's required
   verifier-accepted transfer seam; and
2. economics: every tested schedule was net negative on both completed forcing
   cohorts. Even an unrealistically perfect cross-rung reuse model leaves only
   0.1–0.4% for the best schedule.

A future re-hunt must first add a focused verifier gate for compact at
`root + {8,12,16,24,32}` with gate both off and on, and must establish that
zone `d` is derived from the materialized proof's local budget (or otherwise
prove and verify the intended external-horizon label). Only after that gate
passes should the unchanged forcing/compact/human campaign be rerun. This
report does not authorize that production fix.

## Harness, logs, and regeneration

Only the existing test-only module was edited; production solver code is
unchanged:

- `packages/hexfield_eq/rust/src/tss_pn_init_hunt.rs`
- `.codex-hunt/horizon-ladder-cargo.log` — completed forcing rows and STOP
- `.codex-hunt/horizon-ladder-compact-repro.log` — focused verifier trace

Full campaign (expected to stop at the compact h16 verifier rejection on this
revision):

```powershell
Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:CARGO_TARGET_DIR='.target-hunt'
Remove-Item Env:TSS_INTERIOR_CENSUS_GATE -ErrorAction SilentlyContinue
Remove-Item Env:TSS_HORIZON_LADDER_ONLY_GROUP -ErrorAction SilentlyContinue
Remove-Item Env:TSS_HORIZON_LADDER_HUMAN_N -ErrorAction SilentlyContinue
cargo test --release -p hexfield_eq horizon_ladder_campaign -- `
    --ignored --test-threads=1 --nocapture 2>&1 | `
    Tee-Object -FilePath .codex-hunt/horizon-ladder-cargo.log
```

Focused reproduction:

```powershell
Get-CimInstance Win32_OperatingSystem | % { $_.FreePhysicalMemory/1MB }
$env:CARGO_TARGET_DIR='.target-hunt'
$env:TSS_HORIZON_LADDER_HUMAN_N='0'
$env:TSS_HORIZON_LADDER_ONLY_GROUP='double_fork_compact'
cargo test --release -p hexfield_eq horizon_ladder_campaign -- `
    --ignored --test-threads=1 --nocapture 2>&1 | `
    Tee-Object -FilePath .codex-hunt/horizon-ladder-compact-repro.log
```

Formatting check:

```powershell
rustfmt --edition 2021 --check `
    packages/hexfield_eq/rust/src/tss_pn_init_hunt.rs
```

The full partial run used 12.876 GB free RAM and stopped after 111.74 s of test
time. The focused reproduction used 9.908 GB free RAM and stopped after 1.54 s
of test time. Cargo processes were serialized. No commit was made.
