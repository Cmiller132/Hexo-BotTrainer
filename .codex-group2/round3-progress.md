# Group 2, round 3 — quiet turns and certified ranked defender zones

Date: 2026-07-16 (America/New_York)

Base: `1e082d40` (round-9b normative engine plus round-2 exact-oracle work)

## Checkpoints

### C0 — binding brief, proof, and scope audit

- Read `.codex-group2/round2-progress.md` in full before implementation, then
  skimmed `.codex-group2/prompt-r1.txt` and `round1-progress.md` for the witness
  columns and corrected lambda-order lore.
- Read the normative sibling proof definitions D9–D18, T3/T4, L9′, L11–L14,
  and the (Z2)/(Z4)/(Z5′) zone clauses. The implementation contract is:
  `Z_dir = Prot ∩ Legal`; `Z_seed` is the legal radius-`8(r-1)` band around
  protected nonlegal nonstone cells with live rank at least one; `Z_touch`
  contains every empty of a D-alive defender-touched window satisfying
  `cnt_D + E^D >= 6`; and `Z_virgin` contains legal cells within
  `8(E^D-6)` of an all-empty window when `E^D >= 6`. T4 permits the uniform
  admissible wrapper `r=B` and conservative B-clock completion guards.
- The document wins over the current code. The pre-round verifier still uses
  an obsolete `Z1`, treats full DAG cores as obligations, and forms a seed
  band with radius `8*d`; those are not the revised D10/D11/T4 definitions and
  must not be copied into the new contract.
- Confirmed `HEAD=1e082d40dce4dce6f8cb8fee9bd445b35e071ad5` and initial free
  physical RAM 13.18 GiB. No `AGENTS.md` exists in the worktree.
- Initial tracked worktree state was clean. Existing untracked prompt, PID,
  native executable, target, and `.codex-round5/` artifacts are user-owned and
  will be preserved. No commit will be attempted.

Checkpoint boundary: Step 1 only. Add shadow computation and instrumentation;
do not consume either new universe.

## Step 1 — SHADOW

### C1 — certificate-derived shadow implementation green

- Added two tri-state, default-OFF width flags:
  `quiet_turn_or_edges` and `ranked_unforced_defender_zone`. Their SHADOW
  values are passed through the wide profile but are not consulted by search;
  a focused identity check proved ordinary-wide and shadow-wide status, node
  count, and certificate equality on `double_fork_compact`.
- Shadow zone derivation starts from a completed certificate, not finder
  candidate hints. It reconstructs D10 live roles as designated attacker
  placements plus WIN/LOSS leaf-entry witness empties, computes D14 `B`
  bottom-up (`0`, LOSS `b`, OR child, AND `1+max`), and records the uniform
  four-part T3/T4 set at `k<b` AND nodes. Ranking is hitting-first then
  canonical-coordinate and changes ordering only.
- The pre-R1b seed wrapper remains the conservative shipped radius `8*B`
  through Steps 1–3. T4 explicitly permits larger admissible upper bounds;
  Step 4 owns the requested one-relay shrink to `8*(B-1)`.
- Focused witness/control gate passed 1/0 in 0.12 s. The historical finder
  returned its known verifier-accepted WIN in 2,884 nodes. Coverage:

| Position/node | k/b | B | T3 searched | Full legal | Ratio | Components `(dir,seed,touch,virgin)` | represented reply ranks | quiet fires / full quiet edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `double_fork_compact`, ply 37 | 1/2 | 4 | 62 | 478 | 0.129707 | 19, 0, 50, 0 | 1–62 | 1 / 478 |
| `double_fork_compact`, ply 38 | 0/1 | 3 | 18 | 479 | 0.037578 | 18, 0, 0, 0 | 1–18 | included above |
| `compact_urgent_spare` | — | — | 0 nodes | — | — | — | — | 0 |
| `strongloss_a_backoff_7` | — | — | 0 nodes | — | — | — | — | 0 |

### C2 — all-19 shadow coverage green

- Ran all 19 forcing rows in one serialized process at 1M nodes / 2 GiB TT.
  Result: 0 quiet fires and 0 `k<b` zone nodes across every row, independently
  confirming that the forcing corpus cannot accept this round's machinery.
- WIN rows with certificates at this cap: `0hz3hty` 2,412; `acly7kb` 75;
  `g2xx6wl` 4,107; `hu01jk4` 380; `jh7yo7y` 2,119; `jnzzmcm` 9,798;
  `lz60mfb` 109,896; `xsnfyll` 82; `zrugh2x` 41,734;
  `strongloss_a_prefix6` 16,126; `strongloss_b_prefix8` 1,099; and both Hayes
  rows 11,664. NO rows `8is963b` and `dy3dg99` returned LOSS in one node.
- UNKNOWN rows at this fixed shadow cap were `0l4291i_live` 1,000,000;
  `94gnnol` 1,000,000; `l9mxn59` 226; and `mvp2lvc` 17,957. UNKNOWN is only
  a coverage disposition here, not a changed corpus expectation.
- All-19 shadow test passed 1/0 in 223.17 s. No search result was consumed.

Checkpoint boundary: Step 1 is GREEN. Start Step 2 verifier work only; neither
new universe is consumable until all required certificate mutations reject.

## Step 2 — VERIFY

### C3 — independent replay contract and mutation suite green

- Replaced the obsolete verifier-side `Z1`/full-core shortcut for the new
  contract. The verifier now independently replays the certificate subtree,
  reconstructs D10 roles (every designated attacker move and leaf-entry
  witness empty), derives D14 `B`, and forms
  `Z_dir ∪ Z_seed ∪ Z_touch ∪ Z_virgin` plus D9's deterministic nonempty
  fallback. Finder candidate lists are never inputs.
- Zone certificates remain ordinary D9 AND nodes: every represented defender
  edge is engine-replayed, every mandatory re-derived zone cell must have an
  edge, heuristic supersets are allowed, and no reply-count cap exists.
  Zone nodes require `k<b`, exact stored local `B`, and an exact build-horizon
  binding. Quiet attacker choices are replayed under the existing (Z4)
  attacker/root distance-8 legality rule.
- Byte-for-byte cross-check passed on every shadow-recorded witness node for
  both the full historical certificate and the independently found zoned
  certificate.
- Positive zoned baseline: WIN in 492 finder nodes, 427 certificate nodes;
  strict independent verifier ACCEPTED.
- Mandatory mutation controls all REJECTED: omitted stable `Z_touch` zone
  cell; omitted second stable defender edge; dropped quiet completion edge;
  wrong local budget; wrong horizon; forged leaf; and required-zone edge
  replaced by an outside legal cell. An initial arbitrary-edge omission was
  accepted because it removed only a legal heuristic-superset edge; the test
  was correctly tightened to position-derived mandatory `Z_touch` cells.
- Complete verifier module passed 11/0. No consumption started before this
  checkpoint was green.

Checkpoint boundary: Step 2 is GREEN. The two flags may now enter CONSUME
mode; any finder/verifier disagreement stops the ladder.

## Step 3 — CONSUME

### C4 — headline witness closes at the first rung

- CONSUME keeps the forcing path first. Only after it is exhausted/refuted at
  an attacker OR node does it enumerate the complete legal completion set;
  at a turn start this is every canonical pair of turn-start legal cells plus
  every newly legalized second placement. This is an OR universe, not a cap.
- At `k<b` defender nodes the ranked zone is the complete searched set; rank
  changes order only. Same-turn commutation is deliberately disabled on zone
  nodes because it is a separate forced-dispatch contract; every zone reply is
  represented directly.
- The first 10k rung initially found WIN/409 but was verifier-rejected because
  a zone node carried parent commutation allowances. Consumption stayed red,
  the ladder stopped, and the mixed contract was removed. Re-run result:
  **WIN / 409 nodes / 51 TT hits / 67,177,998 peak TT bytes / 24 ms;
  strict verifier ACCEPTED**. This is the first closing rung, so 100k and 1M
  were not run.

Checkpoint boundary: headline witness is GREEN at 10k. Run the flags-off
official all-19 gate, unchanged spare NO controls, default suite, and diff/
narrow-identity audit before Step 3 can close.

### C5 — Step-3 no-regression matrix green

- Re-ran the mutation suite immediately before the long gate: positive cert
  ACCEPTED and all seven mutations REJECTED.
- Official all-19 single-process gate, flags OFF, 2 GiB TT profile:
  **PASS, `CORPUS_DONE failures=0`**, 445.41 s. `0l4291i_live` retained its
  established 20M closing rung (WIN at 1,879,612 nodes); all other
  expectations also held.
- Spare corpus semantics were not promoted or used as positive acceptance.
  Both honest NO controls stayed UNKNOWN/2 at 10k, 100k, and 1M.
- Default release suite: **95 passed / 0 failed / 15 ignored** in 2.82 s.
  The four additional ignored tests are round-3 shadow/verify/consume harnesses;
  the required default passed count remains 95/0.
- Narrow behavior is unchanged by construction: both new modes default OFF;
  the only attacker/unforced-AND branches are guarded by explicit CONSUME
  checks, and the full default suite's narrow certificate/determinism tests
  pass. No default constructor, candidate ordering, or narrow branch changed.

Checkpoint boundary: Step 3 is GREEN. Start the separately gated R1b shrink.

## Step 4 — R1b seed-radius shrink

### C6 — one-relay shrink and separate gate green

- Read hunt commits `53864998` and `1435680d` plus their R1b report and chain
  fixtures. The binding family attains `8(B-1)` exactly and is shed at
  `8(B-2)`; the pre-round `8B` wrapper therefore carried one removable relay.
- Added shared production `seed_band_radius(d) = 8*(d-1)` using saturating
  subtraction, hence zero for `d<=1`. It lives in `tss_core`, so finder and
  verifier share theorem arithmetic without the verifier importing finder
  candidates or implementation hints.
- Justification: L9′ bounds the first protected occupation by at most `B`
  defender placements and therefore at most `B-1` radius-eight links from the
  first legal seed: `8(B-1) <= 8(d-1)` for admissible `d>=B`.
- `hunt_r1b_chain_sharpness` passed: for B=2..5 the binding seed is retained at
  `8(B-1)` and excluded at `8(B-2)`. The updated production finder cross-check
  passed and explicitly distinguishes the new radius from the former `8B`.
- Mutation suite remained green. The pre-shrink verified historical, zoned,
  and headline certificates all regenerated and verified after the shrink;
  no rollback condition fired. Headline result remained WIN/409 at 10k with
  verifier acceptance.
- Post-shrink official all-19 single-process gate, 2 GiB profile:
  **PASS, `CORPUS_DONE failures=0`**, 442.48 s.
- Post-shrink default release suite: **95 passed / 0 failed / 17 ignored** in
  2.79 s. The two additional ignored rows are the R1b fixture/cross-check.

Checkpoint boundary: Step 4 is GREEN. All requested implementation and gate
work is complete; perform only final format/diff/status audit.

## Regeneration commands

```powershell
rustfmt --edition 2021 packages/hexfield_eq/rust/src/tss_solver.rs packages/hexfield_eq/rust/src/tss_spare_corpus.rs packages/hexfield_eq/rust/src/tss_corpus.rs
$env:CARGO_TARGET_DIR='.target-codex'
cargo test --release -p hexfield_eq tss_round3_shadow_spare_coverage -- --ignored --test-threads=1 --nocapture

$env:TSS_BACKWALK_TT_BYTES='2147483648'
$env:TSS_R3_SHADOW_CAP='1000000'
cargo test --release -p hexfield_eq tss_round3_shadow_forcing_coverage -- --ignored --test-threads=1 --nocapture

cargo test --release -p hexfield_eq tss_round3_verifier_mutations -- --ignored --test-threads=1 --nocapture
cargo test --release -p hexfield_eq 'tss_verify::tests' -- --test-threads=1 --nocapture

$env:TSS_R3_CAP='10000'
$env:TSS_BACKWALK_TT_BYTES='536870912'
cargo test --release -p hexfield_eq tss_round3_consume_witness -- --ignored --test-threads=1 --nocapture

$env:TSS_BACKWALK_TT_BYTES='2147483648'
cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture
foreach($cap in @('10000','100000','1000000')) {
  $env:TSS_SPARE_CORPUS_CAP=$cap
  cargo test --release -p hexfield_eq tss_spare_corpus_check -- --ignored --test-threads=1 --nocapture
}
Remove-Item Env:TSS_SPARE_CORPUS_CAP -ErrorAction SilentlyContinue
Remove-Item Env:TSS_BACKWALK_TT_BYTES -ErrorAction SilentlyContinue
cargo test --release -p hexfield_eq -- --test-threads=1

# Step 4 focused and full gates
cargo test --release -p hexfield_eq hunt_r1b_chain_sharpness -- --ignored --test-threads=1 --nocapture
cargo test --release -p hexfield_eq hunt_seed_band_matches_production -- --ignored --test-threads=1 --nocapture
cargo test --release -p hexfield_eq tss_round3_verifier_mutations -- --ignored --test-threads=1 --nocapture
$env:TSS_R3_CAP='10000'
$env:TSS_BACKWALK_TT_BYTES='536870912'
cargo test --release -p hexfield_eq tss_round3_consume_witness -- --ignored --test-threads=1 --nocapture

$env:TSS_BACKWALK_TT_BYTES='2147483648'
cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture
Remove-Item Env:TSS_BACKWALK_TT_BYTES -ErrorAction SilentlyContinue
cargo test --release -p hexfield_eq -- --test-threads=1
```

## Final verification and diff surface

### C7 — final audit

- Final headline smoke after formatting: WIN/409 at 10k, strict verifier
  ACCEPTED (25 ms measured wall in the harness).
- `git diff --check` passed.
- Tracked diff is limited to the five intended Rust files:
  `tss_core.rs`, `tss_solver.rs`, `tss_verify.rs`, and the two ignored coverage
  harness modules `tss_corpus.rs` / `tss_spare_corpus.rs`. This progress memo is
  new. Pre-existing untracked prompt/PID/native/target artifacts remain
  user-owned and untouched.
- Default/narrow is protected by OFF defaults and explicit CONSUME guards;
  no workspace-wide formatting was run. Only the five edited Rust files were
  formatted with `rustfmt --edition 2021`.
- `HEAD` remains `1e082d40dce4dce6f8cb8fee9bd445b35e071ad5`.
  No git commit was attempted, per the operational contract.
- Final free physical RAM was 13.31 GiB. Every build/solve was serialized and
  used `CARGO_TARGET_DIR=.target-codex`.

Final disposition: **complete and green**. `double_fork_compact` is now a
verifier-accepted wide WIN at the first (10k) rung; all shadow, mutation,
official-corpus, spare-control, default-suite, and post-R1b gates pass without
any WIN-vs-LOSS disagreement.
