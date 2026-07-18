# TSS gated-lever consolidation: merge resolution and tip verification

Date: 2026-07-17

## Inputs and final Git state

| Role | Ref | Object |
|---|---|---|
| Shared base | base | <code>2430fc4771027c23f73d7a43a4de187574db557c</code> |
| Receiving branch / unchanged HEAD | <code>claude/tss-vcf-width</code> | <code>c4b496ed5ad2f9843a4a33696d6bad5285f10a73</code> |
| Turn-quotient tip | <code>hunt/turn-quotient</code> | <code>5f836b709f8c1ca93f7bf093d3d2c4bdcc90c9d6</code> |
| PN-init tip | <code>hunt/pn-init</code> | <code>2454fa91383aca909a4b7d74d2bd04aed4bb57d4</code> |

The receiving HEAD was confirmed to be the G2R8 landing, with G2R7 as its
parent and the shared base as its grandparent. The tracked tree was clean at
entry. Pre-existing untracked <code>.codex-group2</code>,
<code>.codex-round5</code>, <code>.target-codex-iso</code>, and
<code>.target-codex</code> artifacts were left untouched.

No commit was created. The turn-quotient side was merged with:

~~~powershell
git merge --no-commit --no-ff 5f836b70
~~~

After resolving and staging that merge, the PN-init delta was generated
against the exact shared base and applied three-way into the same uncommitted
merge index:

~~~powershell
$patch = Join-Path $env:TEMP 'hexgt-pn-init-2454fa91.patch'
git diff --binary 2430fc47..2454fa91 --output=$patch
git apply --3way --index --whitespace=nowarn $patch
~~~

Thus HEAD remains <code>c4b496ed</code>, MERGE_HEAD is
<code>5f836b70</code>, and the PN-init content is present as staged manual
three-way application. The final tree has no unmerged paths.

## Conflict-resolution log

### Turn-quotient into G2R8

1. <code>lib.rs</code> test-module registration: retained
   <code>tss_k_reply_shadow</code> and added
   <code>tss_turn_quotient_hunt</code>, each independently guarded by
   <code>cfg(test)</code>.
2. <code>tss_solver.rs</code> solve-start test state: combined the G2R7
   narrow-signature and K-reply-shadow clears with NQ4's quotient-report clear
   in one test-only block. The production K-reply flag remains sampled once
   immediately afterward.
3. <code>tss_solver.rs</code> narrow-search state: retained
   <code>k_reply_consume</code> and the test-only shadow sink alongside the
   test-only NQ4 quotient telemetry.
4. <code>tss_solver.rs</code> narrow test constructor: initialized K-reply
   consume off, shadow absent, and NQ4 telemetry from its test environment.
5. <code>tss_solver.rs</code> shared narrow constructor: threaded K-reply
   consume/shadow and initialized NQ4 telemetry. No search ordering was
   changed.

These were additive telemetry/field conflicts. No semantic choice was needed.

### PN-init three-way application

The application reported content conflicts in <code>lib.rs</code>,
<code>tss_core.rs</code>, and <code>tss_solver.rs</code>.
<code>tss_corpus.rs</code> and <code>tss_spare_corpus.rs</code> applied
cleanly.

1. Solver imports: kept fragment-store <code>Arc</code>, test-only NQ4
   <code>RefCell</code>, and unconditional <code>Instant</code> required by
   production interior-gate timing.
2. Primal solve call: passed both immutable solve-local flags, K-reply then
   interior gate.
3. Dual solve call: used the identical flag ordering.
4. <code>prove_for</code> signature: retained both booleans.
5. Narrow dispatch: passed K-reply consume, the test-only shadow sink, and the
   interior-gate flag; the wide path receives only the interior flag because
   K-reply is a narrow quiet-fallback lever.
6. Wide <code>SolveStats</code>: retained fragment lookup/hit/store fields and
   PN-init expansion/TT/interior counters; default initialization covers the
   later-set fragment import count.
7. <code>merge_stats</code>: retained saturating fragment and interior
   accumulation, max TT/peak accounting, and current fragment-store residency.
8. <code>WidePnSearch</code> fields: retained the lifetime-parametric fragment
   store and lazy frontier, then added the interior flag and its three counters.
9. NQ6 wide helpers and generic implementation: retained PN-init record/finalize
   helpers, adapted their search type to <code>WidePnSearch&lt;'_&gt;</code>,
   then restored <code>impl&lt;'store&gt; WidePnSearch&lt;'store&gt;</code>.
10. <code>prove_narrow_compat</code> signature: carried K-reply consume,
    test-only shadow, then interior gate.
11. Its <code>NarrowCompatSearch::with_shared</code> call used the same order.
12. Removed the now-obsolete pre-certificate narrow stats literal; retained
    PN-init's complete post-certificate literal and added a default tail for
    fragment fields.
13. Wide constructor: initialized both lazy-frontier mode and interior
    flag/counters.
14. Wide expansion telemetry: ran NQ4 quotient observation and NQ6 expansion
    recording together under <code>cfg(test)</code>.
15. Narrow-search struct layout: placed K-reply, NQ4, and interior fields inside
    the single struct before defining <code>NarrowQuotientTelemetry</code>.
16. Narrow test constructor: initialized all three feature families.
17. Narrow shared-constructor signature: retained K-reply/shadow and interior
    arguments.
18. Narrow shared-constructor initializer: retained all corresponding fields
    and counters.
19. Narrow <code>prove</code>: retained PN-init's telemetry guard/closure and
    census dismissal, retained NQ4 expansion observation, and called G2R8's
    current <code>prove_choice(state, claimant, ply, &analysis, pair)</code>.
    This preserves the fresh analysis required by K-reply.
20. Focused shared-TT constructor call: supplied
    <code>false, None, false</code> for K-reply, shadow, and interior gate.

File-level resolutions:

- <code>lib.rs</code>: all three test modules survive:
  <code>tss_k_reply_shadow</code>, <code>tss_turn_quotient_hunt</code>, and
  <code>tss_pn_init_hunt</code>.
- <code>tss_core.rs</code>: all ten newly introduced stats fields survive:
  expansions, TT entries, five fragment fields, and three interior fields.
- <code>tss_corpus.rs</code>: retained the G2R8 test visibility, turn-quotient
  hard-verdict certificate assertions and strict verification, fragment
  telemetry, and PN-init stats printing. <code>CORPUS_MODE</code> and
  <code>CORPUS_DONE</code> now echo all four flags. New expectation variables
  <code>TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE</code> and
  <code>TSS_CORPUS_EXPECT_K_REPLY_CONSUME</code> complement the existing
  shared/lazy assertions.
- <code>tss_spare_corpus.rs</code>: retained G2R7's
  <code>pub(crate) mining_candidate</code> visibility and added the unchanged
  R-FIX1 regression.
- <code>rebase_zone_distances</code>: PN-init's R-FIX1 implementation replaces
  the old external-countdown stamping. It reconstructs exact D14 budgets in
  postorder and stamps the certificate build horizon. The separate
  shared-fragment DAG relabeler remains present and exact.

Incoming Markdown trailing whitespace in two shared-fragment reports was
removed so the final diff check is clean. Rustfmt also expanded one pre-existing
long Python registration call in the now-touched <code>lib.rs</code>.

### Semantic-overlap review

No unresolved semantic choice was found.

- The wide fragment lookup remains at its turn-quotient insertion point, before
  the interior census gate. A compatible entry is an already verified positive
  fragment; otherwise search continues to the census gate. No gate or campaign
  exposed a contradictory fragment-hit/census-dismissal state.
- In narrow search, the interior dismissal occurs before entering the claimant
  choice arm. When it does not dismiss, K-reply consumes the same freshly
  computed threat analysis and full legal set as G2R8.
- Lazy-frontier admission, shared-fragment materialization, interior-gate
  pruning, and K-reply consumption remain independently default off. R-FIX1 is
  the sole always-on behavior change.

## Verification command discipline

Every Cargo row below ran in a fresh PowerShell process. The literal ordinary
(9 GiB floor) prefix was:

~~~powershell
$ErrorActionPreference='Stop'
if (Get-Process cargo -ErrorAction SilentlyContinue) {
    throw 'another cargo process is already running'
}
$os = Get-CimInstance Win32_OperatingSystem
$free = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
Write-Host "RAM_PREFLIGHT free_gib=$free floor_gib=9"
if ($free -le 9) {
    throw "free RAM $free GiB is not above 9 GiB"
}
Get-ChildItem Env: |
    Where-Object Name -Like 'TSS_*' |
    ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR='.target-codex'
~~~

For every 2 GiB row the two literal floor values in that prefix were
<code>11</code> instead of <code>9</code>. The exact row command below followed
that prefix. In human-corpus rows, the <code>$human</code> assignment shown in
the table and its <code>Test-Path -LiteralPath</code> check occurred immediately
after the cargo-process check and before the RAM query; the remaining profile
assignments and exact Cargo command followed the prefix.

## Full merged-tip battery

| Gate | Exact profile variables and Cargo command | RAM at launch | Result at merged tip |
|---|---|---:|---|
| Default release suite | <code>cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq -- --test-threads=1</code> | 15.82 GiB; floor 9 | PASS: 104 passed, 0 failed, 32 ignored; tests 3.10 s |
| Official legacy, 2 GiB, all off | <code>$env:TSS_BACKWALK_TT_BYTES='2147483648'; $env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS='0'; $env:TSS_CORPUS_EXPECT_LAZY_FRONTIER='0'; $env:TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE='0'; $env:TSS_CORPUS_EXPECT_K_REPLY_CONSUME='0'; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture</code> | 15.79 GiB; floor 11 | PASS: mode all off; <code>CORPUS_DONE failures=0</code>; 447.33 s |
| Recommended, 1 GiB lazy | <code>$env:TSS_BACKWALK_TT_BYTES='1073741824'; $env:TSS_LAZY_FRONTIER='1'; $env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS='0'; $env:TSS_CORPUS_EXPECT_LAZY_FRONTIER='1'; $env:TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE='0'; $env:TSS_CORPUS_EXPECT_K_REPLY_CONSUME='0'; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture</code> | 15.78 GiB; floor 9 | PASS: exact mode echo; <code>CORPUS_DONE failures=0</code>; 495.94 s |
| Composition, 2 GiB fragments + lazy | <code>$env:TSS_BACKWALK_TT_BYTES='2147483648'; $env:TSS_SHARED_FRAGMENTS='1'; $env:TSS_LAZY_FRONTIER='1'; $env:TSS_CORPUS_EXPECT_SHARED_FRAGMENTS='1'; $env:TSS_CORPUS_EXPECT_LAZY_FRONTIER='1'; $env:TSS_CORPUS_EXPECT_INTERIOR_CENSUS_GATE='0'; $env:TSS_CORPUS_EXPECT_K_REPLY_CONSUME='0'; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq tss_corpus_check -- --ignored --test-threads=1 --nocapture</code> | 15.82 GiB; floor 11 | PASS: exact mode echo; <code>CORPUS_DONE failures=0</code>; max store 64 entries / 6,027,694 bytes; 490.08 s |
| R-FIX1 focused regression | <code>cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq bounded_horizon_compact_win_certificate_verifies -- --test-threads=1 --nocapture</code> | 16.04 GiB; floor 9 | PASS: 1 passed; bounded h16 zoned WIN strictly verifies; 0.23 s |
| G2R8 double-fork identity | <code>$env:TSS_R7_TT_BYTES='268435456'; $env:TSS_R3_CAP='10000'; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq --lib tss_round8_k_reply_double_fork_identity -- --ignored --test-threads=1 --nocapture</code> | 16.39 GiB; floor 9 | PASS: WIN/WIN, 409 to 395 nodes, certificate equal, urgent 478 to 1; 0.08 s |
| R-LF1 equivalence campaign | <code>$human='E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl'; $env:TSS_LAZY_FRONTIER_TT_BYTES='2147483648'; $env:TSS_LAZY_FRONTIER_VALIDATE_KEYS='1'; $env:TSS_TURN_QUOTIENT_HUMAN_CORPUS=$human; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq lazy_frontier_equivalence_campaign -- --ignored --test-threads=1 --nocapture</code> | 15.98 GiB; floor 11 | PASS: 59 roots; <code>LF_EQ_DONE result=PASS node_identity=exact certificate_bytes=exact</code>; 159.19 s |
| R-IG1 live, flag off | <code>$human='E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl'; $env:TSS_PN_INIT_TT_BYTES='536870912'; $env:TSS_PN_INIT_HUMAN_N='100'; $env:TSS_PN_INIT_HUMAN_CORPUS=$human; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq interior_gate_live_campaign -- --ignored --test-threads=1 --nocapture</code> | 15.96 GiB; floor 9 | PASS: frozen expansions 89,405 / 324,163 / 408 / 78,970; <code>IG_DONE ... PASS</code>; 48.77 s |
| R-IG1 live, flag on | <code>$human='E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl'; $env:TSS_PN_INIT_TT_BYTES='536870912'; $env:TSS_PN_INIT_HUMAN_N='100'; $env:TSS_PN_INIT_HUMAN_CORPUS=$human; $env:TSS_INTERIOR_CENSUS_GATE='1'; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq interior_gate_live_campaign -- --ignored --test-threads=1 --nocapture</code> | 16.44 GiB; floor 9 | PASS: verdict splits frozen; expansions 18,909 / 21,302 / 408 / 46,419; <code>IG_DONE ... PASS</code>; 9.23 s |
| NQ4 telemetry identity | <code>$human='E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl'; $env:TSS_TURN_QUOTIENT_TT_BYTES='536870912'; $env:TSS_TURN_QUOTIENT_HUMAN_CORPUS=$human; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq turn_quotient_campaign -- --ignored --test-threads=1 --nocapture</code> | 16.00 GiB; floor 9 | PASS: <code>TQ_IDENTITY id=0hz3hty status=WIN nodes=2412 tt_hits=2263 result=PASS</code>; <code>TQ_DONE ... PASS</code>; 106.49 s |
| NQ6 PNI identity | <code>$human='E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl'; $env:TSS_PN_INIT_TT_BYTES='536870912'; $env:TSS_PN_INIT_HUMAN_N='100'; $env:TSS_PN_INIT_HUMAN_CORPUS=$human; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq pn_init_campaign -- --ignored --test-threads=1 --nocapture</code> | 16.43 GiB; floor 9 | PASS: <code>PNI_IDENTITY id=0hz3hty status=UNKNOWN nodes=9302 tt_hits=2872 expansions=9301 result=PASS</code>; <code>PNI_DONE ... PASS</code>; 51.15 s |
| G2R9b eager single root | <code>$human='E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl'; $env:TSS_SHARED_FRAGMENT_TT_BYTES='536870912'; $env:TSS_SHARED_FRAGMENT_LAZY_MODE='off'; $env:TSS_SHARED_FRAGMENT_CASE_ID='human_014_g1531_p95'; $env:TSS_TURN_QUOTIENT_HUMAN_CORPUS=$human; cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq shared_fragment_soundness_and_warm_campaign -- --ignored --test-threads=1 --nocapture</code> | 15.98 GiB; floor 9 | PASS: cold UNKNOWN identity; warm UNKNOWN to verified WIN; 10,000 to 3,770 expansions; monotone contract PASS; 4.33 s |
| Formatting and patch hygiene | <code>rustfmt --edition 2021 --check --config skip_children=true packages/hexfield_eq/rust/src/lib.rs packages/hexfield_eq/rust/src/tss_core.rs packages/hexfield_eq/rust/src/tss_corpus.rs packages/hexfield_eq/rust/src/tss_solver.rs packages/hexfield_eq/rust/src/tss_spare_corpus.rs packages/hexfield_eq/rust/src/tss_turn_quotient_hunt.rs packages/hexfield_eq/rust/src/tss_pn_init_hunt.rs packages/hexfield_eq/rust/src/tss_k_reply_shadow.rs; git diff --check; git diff --cached --check</code> | n/a | PASS |

## Reproduction notes

All requested verdict, certificate, frozen-baseline, and monotone-contract
checks reproduced at the merged tip. There was no gate failure or verdict flip.

The exact wall times were naturally not byte-for-byte historical: legacy
447.33 s, recommended lazy 495.94 s, and fragments+lazy 490.08 s. The deep
identity values did reproduce:

- Legacy <code>0l4291i_live</code>: WIN at 1,879,612 nodes and
  1,729,265,069 peak TT bytes.
- 1 GiB lazy and 2 GiB fragments+lazy: WIN at 1,879,612 nodes, 37,076 TT
  hits, and 549,161,606 peak bytes.
- Official fragments+lazy creates a fresh solver per rung, so fragment
  lookups/hits/imports are zero while the resident store still reaches the
  expected 64 entries / 6,027,694 bytes.
- The filtered single-root G2R9b command intentionally reports
  <code>mutation=SKIP_FILTERED</code>; mutation coverage was not part of the
  requested single-root spot-check, while its strict verifier and monotone
  contract both passed.
- R-LF2 is report-only and therefore has no executable gate beyond the merged
  report evidence and the newly rerun 1 GiB lazy official gate.

## Tracked file inventory

Unique turn-quotient files retained:

- <code>.codex-hunt/prompt-g2r9.txt</code>
- <code>.codex-hunt/prompt-g2r9b.txt</code>
- <code>.codex-hunt/prompt-lf1.txt</code>
- <code>.codex-hunt/prompt-lf2.txt</code>
- <code>.codex-hunt/prompt-nq4-1.txt</code>
- <code>BUILD_SHARED_FRAGMENTS.md</code>
- <code>HUNT_REPORT_LAZY_FRONTIER.md</code>
- <code>HUNT_REPORT_LAZY_MEMORY_WALL.md</code>
- <code>HUNT_REPORT_SHARED_FRAGMENTS.md</code>
- <code>HUNT_REPORT_TURN_QUOTIENT.md</code>
- <code>PROOF_LAZY_FRONTIER.md</code>
- <code>packages/hexfield_eq/rust/src/tss_turn_quotient_hunt.rs</code>

Unique PN-init files retained:

- <code>.codex-hunt/prompt-fix1.txt</code>
- <code>.codex-hunt/prompt-ig1.txt</code>
- <code>.codex-hunt/prompt-nq6-1.txt</code>
- <code>.codex-hunt/prompt-nq8-1.txt</code>
- <code>BUILD_INTERIOR_GATE.md</code>
- <code>FIX_ZONE_CLOCK.md</code>
- <code>HUNT_REPORT_HORIZON_LADDER.md</code>
- <code>HUNT_REPORT_PN_INIT.md</code>
- <code>packages/hexfield_eq/rust/src/tss_pn_init_hunt.rs</code>

Composed files:

- <code>packages/hexfield_eq/rust/src/lib.rs</code>
- <code>packages/hexfield_eq/rust/src/tss_core.rs</code>
- <code>packages/hexfield_eq/rust/src/tss_corpus.rs</code>
- <code>packages/hexfield_eq/rust/src/tss_solver.rs</code>
- <code>packages/hexfield_eq/rust/src/tss_spare_corpus.rs</code>
- <code>MERGE_RESOLUTION.md</code>
