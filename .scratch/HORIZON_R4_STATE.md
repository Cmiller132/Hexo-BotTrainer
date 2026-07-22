# Horizon R4 successor state

Status: partial at the h17/h18 theory gate.  Branch `claude/deadline-ladder`,
starting and current tracked HEAD `43cbdffb77d412b8b6800a239c2af9a67006623c`.
No commit was made.  No engine, verifier, package, config, or Lean file was
edited.

## Exact stopping point

Phase 1 is closed for the inherited R3 h13/h14 normalized endpoint.  The
standalone dependency-free Rust kernel in `.scratch/horizon_native/` returns
WIN on all 155 required registry IDs in one canonical single-threaded pass
(149 fresh h14, 6 SecondStone h13), with no timeout, negative, or error.  The
155 IDs represent 154 exact ordered move histories.  The canonical evidence is
`.scratch/horizon_r4_phase1_registry_final{,_rows}.json[l]`, consolidated in
`.scratch/horizon_r4_phase1.json`.

The global exact ladder does **not** advance beyond h14.  The h17 static final
singleton-cover sublemma is closed for attacker sets of at most eight stones,
but the dynamic interaction/remote normalization needed to expose that
endpoint is unproved.  At h18 an exact local obstruction shows why the R3
single-reserved-pair proof cannot simply be reused.  Consequently no h17/h18
compiled verdicts were claimed and the 907 exact-depth registry IDs at that
rung were enumerated, not run.  Fresh `Win22 = Win23 = Win24` is proved at the
definition/schedule level, but it cannot transport an h22 result that has not
been established.

## The precise open theorem

Let the attacker have built the six-stone nonterminal precursor

`X6 = {(-2,0), (-1,2), (0,-1), (0,0), (0,1), (1,0)}`.

The exact local suffix is `forall D0, exists A_activation, forall D_cover,
exists A_final`.  Its activation-pair carrier has 24 cells and 48 dangerous
pairs.  Without preblocks, 40 activated residual families have cover number 3
and 8 have cover number 4.  A dominance-complete 17-cell defender carrier has
minimum precover number 3; one witness is `{(-4,0), (-3,4), (0,-3)}`.  All
cells are radius-8 self-chainable conditional on a legal seed.  Therefore
reserving only `D_cover` is insufficient.

What remains unproved is not this local calculation.  It is the coupling
statement:

> If earlier defender replies spend the three-stone local tax needed to
> precover every dangerous continuation of such an excursion, then those
> replies can be charged against the defender's obligations in the anchored
> interaction (or mirrored excursion) without weakening the normalized
> defender strategy.

Equivalently, prove an exact tempo/domination ledger for every radius-8 remote
excursion of up to ten attacker stones, or find a reachable counterexample in
which the defender must pay both the local tax and an incompatible anchored
tax.  The current artifacts establish neither reachability nor forceability of
`X6`; they establish the minimal local obstruction to the old proof schema.

At h21/h22 the weaker reserve-pair schema is already statically impossible:
the nonterminal nine-stone two-axis cross has four distinct singleton
completion cells, preserved by the ten-stone extension.  Any continuation
proof must budget more than one defender pair or use a stronger coupling
invariant.

## Next actions, in order

1. Formalize the interaction/excursion ledger as a finite state invariant.  A
   useful state records anchored residual antichains, each excursion component,
   placements spent to connect it under radius 8, defender preblocks in its
   dominance carrier, and which defensive placements also hit anchored
   residuals.  Search for the smallest state violating the proposed charge.
2. Exhaustively enumerate this ledger first for attacker budgets 7--10 using
   the exact `X6` 17-cell dominance carrier.  Quotient translations and line
   symmetries only after retaining component ownership and radius-8 connection
   cost.  A counterexample must include a legal chronological placement order,
   not merely a static set.
3. If the ledger theorem closes h18, generalize the native kernel's state key:
   the present `StateKey` stores at most four placed cells per player and is
   valid only for h13/h14.  Add phase-clock quotas, more pair layers, and
   per-layer true legality.  Do not just change the parser horizon check.
4. Implement the exact symbolic A3 family solver described in the completed
   `ladder_semantics` audit: symbolic defender `Fixed`/`Free` modes, incidence
   bitsets `I3/I4`, allocation-free two-cover tests, and a semantic residual-
   family cache.  Keep the current solver as a test oracle.  This is a runtime
   optimization, not a substitute for the remote theorem.
5. Validate h17/h18 against the exact 907-ID rung set (182 SecondStone d17,
   725 fresh d18; ID hash
   `68A6D26CAA96DA0D6C3DED50DBC2EEA6362362D7EF3A2817450E6405343A1A2D`).
   Only after 907/907 is closed should h21/h22 be attempted.
6. For h21/h22, solve the four-singleton-cross obstruction with the larger
   defender budget, then validate 1,049 IDs (301 d21, 748 d22; ID hash
   `C9BF9C64499B03E12EEA56692CE556456EDFBA34B0FE452A418321192865824F`).
   Fresh h22 then carries to h24, but SecondStone/opening h24 are new attacker
   placements and need separate endpoints.

## Correctness boundaries that must be retained

- R3 Python D1 enumerates radius-8-illegal fringe pairs.  All 155 h13/h14 IDs
  have fringe cells (11,973 total); the rank-67 `atlas_oa-c515...`
  counterexample is in `.scratch/horizon_r4_d1_legality.json`.  The native D1
  iterator checks physical pair chronology before incidence deduplication.
- R3 Python also adds two-cell A1 masks at every SecondStone h13 root despite
  the one-placement clock.  The six-root exhaustive audit is
  `.scratch/horizon_r4_python_boundary.json`; native matches the correct legal
  singleton counts, while Python adds 37,723 illegal pair masks in total.
- The four-case Python/native parity suite has three immediate WINs and one
  fresh h13 NEGATIVE.  The negative exhausts 29,027 A1 actions; every
  counterreply is a two-cell completion within distance five of a root stone,
  so the Python D1 legality bug is immaterial for that case.  The immediate
  SecondStone case does not validate Python's exhaustive SecondStone space.
- Registry uniqueness is by ID: 2,941 IDs but 2,788 exact ordered move
  histories.  Preserve both ID and move-history accounting.  At <=14 the
  counts are 278 IDs / 275 histories; the requested validation ladder remains
  ID-based.
- Timeouts are boundaries, never negatives.  The frozen cohort sweep completed
  13 WIN memberships and 6,649 timeouts with zero completed negatives/errors.
  Forty-eight opening self-play rows were unsupported at this endpoint.
- `.scratch/horizon_r4_phase2.json`'s 6,228-union latent-pivot census is limited
  to unions of two consecutive-cross precursors within relative radius 1.  It
  is not an exhaustive k<=12 remote theorem.

## Reproduction

Serialize Cargo host-wide and keep the trainer constraint:

```powershell
$env:CARGO_TARGET_DIR='.scratch/horizon_native/.target'
cargo test --manifest-path .scratch/horizon_native/Cargo.toml
cargo build --release --manifest-path .scratch/horizon_native/Cargo.toml
python .scratch/horizon_r4_registry.py --out .scratch/horizon_r4_registry.json
python .scratch/horizon_native/driver.py registry --per-root-ms 10000 --max-cache 500000 --out .scratch/horizon_r4_phase1_registry_final_rows.jsonl --summary .scratch/horizon_r4_phase1_registry_final.json
python .scratch/horizon_native/driver.py synthetic --per-root-ms 60000 --max-cache 500000 --out .scratch/horizon_r4_phase1_synthetic_rows.jsonl --summary .scratch/horizon_r4_phase1_synthetic.json
python .scratch/horizon_r4_d1_legality.py --out .scratch/horizon_r4_d1_legality.json
python .scratch/horizon_r4_cert_hints.py --output .scratch/horizon_r4_cert_hints.json
python .scratch/horizon_r4_python_boundary.py
python .scratch/horizon_r4_consolidate.py
python .scratch/horizon_r4_remote.py --out .scratch/horizon_r4_phase2.json
python .scratch/horizon_r4_ladder.py --registry .scratch/horizon_r4_registry.json --out .scratch/horizon_r4_phase3.json
python .scratch/horizon_r4_hashes.py
```

The cohort shards are already complete and should not be repeated.  Their
budgets, paths, and hashes are embedded in `.scratch/horizon_r4_phase1.json`.
The first 2,134-row human shard was launched with `--per-root-ms 10
--max-cache 50000` but its 900-second shell kill prevented the companion
summary from being written; its JSONL rows plus the disjoint 586-row resume
are the durable provenance.
