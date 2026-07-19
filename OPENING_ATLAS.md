# OPENING ATLAS — certified first pass

Status: first bounded pass on `atlas/opening` at
`ad606d0e5100dcef6f67cf5d8ede120e11ce5627`, using the normative round-9b
engine target `ac3f455f759232ec1e1cfc210cddc90a4558ca06`.

## 1. What a row means

The unit of this atlas is an exact, nonterminal engine position reached by a
legal opening prefix. A move list is retained as a replay witness, but position
identity is the full root binding: sorted occupied coordinates with owners,
side to move, exact phase (including the `SecondStone.first` witness),
placement count, and terminal fact.

Verdict vocabulary is deliberately strict:

- **CERTIFIED WIN**: the side to move has a winning strategy and the strict
  independent `TssVerifier` accepted the solver's certificate.
- **CERTIFIED LOSS**: the other side has a winning strategy and the strict
  verifier accepted the dual certificate.
- **UNKNOWN**: no certificate was completed inside this pass's bounds. It is
  not evidence of a draw, balance, or a contrary verdict.
- **CERTIFIED DRAW** is reserved in the schema but cannot be emitted by the
  current proof format. Hexo's engine has no draw outcome, and this pass makes
  no draw claim.

Each machine row printed by the harness has schema version 1 and these fields:

| Field | Meaning |
|---|---|
| `id` | `oa-` plus FNV-1a-64 of the canonical full-position key |
| `source`, `source_prefix` | Shallow census class, or human game hash and number of replayed placements |
| `moves` | One legal D6-canonical replay witness, as `q,r;q,r;...` |
| `placements`, `side`, `phase` | Exact root turn state |
| `orbit` | Number of distinct images among the 12 D6 transforms |
| `cap`, `nodes`, `expansions` | First node rung returning the printed result and actual work |
| `horizon` | Absolute semantic deadline supplied to the solver |
| `derived_horizon` | Maximum exact leaf resolution in this certificate |
| `status`, `claimant`, `certified` | Side-to-move status, winning player, and strict-verification flag |
| `cert_nodes`, `cert_edges`, `cert_commutations`, `cert_zones` | Certificate shape |
| `d6_verified`, `d6_mask` | Audit count and bitmask of transformed roots on which the mechanically remapped certificate was independently accepted; representative certification requires symmetry 0 because the certificate is minted for that exact root |
| `cert_fnv1a64_debug_v1` | Reproduction fingerprint of Rust's full certificate `Debug` form; useful for comparison, not a cryptographic commitment or a substitute for verification |

The supplied deadline was root placement count plus 12. `derived_horizon` is
the smallest deadline that admits that *specific returned certificate*; it is
not a claim that no different certificate could finish earlier. Similarly,
`cap` is the first rung in the 10k→100k ladder that found the certificate, not
an intrinsic proof-complexity lower bound.

## 2. D6 reduction

The implementation uses the same coordinate action exported by the verifier:

1. Symmetries 0–5 repeatedly rotate axial `(q,r)` by `(-r,q+r)`.
2. Symmetries 6–11 first reflect by `(q,-q-r)`, then apply rotations 0–5.
3. For each image, replay the transformed prefix through `hexo_engine` and
   form the exact root binding described above.
4. Choose the lexicographically least full binding; use transformed replay
   coordinates only as a deterministic tie-breaker.

Thus owner colors and side to move are never exchanged. At a completed turn,
irrelevant order inside an earlier two-stone turn does not enter the position
key; at `SecondStone`, the transformed `first` witness does enter it because it
is rule-relevant. For every hard row, the harness additionally remaps the
certificate itself and invokes the strict verifier on all 12 transformed
replays. This is an audit of the certificate-remapping utility, not the verdict
gate: the strict verdict is attached to the exact canonical representative,
while D6 position equivalence comes from the rule-preserving coordinate
action. A remap rejection is printed and never disguised as verifier
acceptance for that image.

## 3. Exact shallow census

The cargo-free enumerator `scripts/opening_atlas_census.py` exhausts the finite
radius-8 opening space through Player 1's first completed turn. It counts
positions, so Player 1's completed pair is unordered.

| Root after | Raw positions | D6 representatives | Orbit histogram |
|---|---:|---:|---|
| P0 origin, then P1 first placement | 216 | 24 | 12 orbits of size 6; 12 of size 12 |
| P0 origin, then P1 completed pair | 42,768 | 3,684 | 12 orbits of size 3; 222 of size 6; 3,450 of size 12 |

For the completed pair, legality is exact: the first P1 cell lies in the
radius-8 disk about the origin, and the other cell lies within radius 8 of the
origin or that first cell. Deduplication then forgets pair order and applies
D6. The orbit weights sum back to 42,768. The Rust harness independently pins
the representative totals in `opening_atlas_d6_census_constants`.

## 4. Certified results

The corpus slice is deterministic and independent of file order: from all
6,902 decisive games, sort by `game_hash`, take the first eight, then inspect
the final 12 nonterminal prefixes of each game from latest to earliest. The
selected source hashes and inclusive prefix ranges are:

| Game hash | Corpus winner | Prefix placements scanned |
|---|---:|---:|
| `00070cdd8fb87f42` | P1 (`-1`) | 67–78 |
| `001165e4e1d7e246` | P0 (`1`) | 9–20 |
| `001c0059e69f6973` | P0 (`1`) | 45–56 |
| `002f5360162bac9b` | P0 (`1`) | 45–56 |
| `0035f32035e5468b` | P1 (`-1`) | 47–58 |
| `00386e2d3c6f65fd` | P1 (`-1`) | 27–38 |
| `003c115aa968eb5a` | P0 (`1`) | 29–40 |
| `004759ff34cefdc2` | P1 (`-1`) | 35–46 |

These are 96 distinct D6-canonical positions. Corpus winners are metadata
only: they never enter certification, and a played continuation is never
treated as a strategy proof.

### Imported certified baseline

Before spending new compute, this pass imports the hard rows from the retained
round-9b-lineage all-19 gate `CLOSURE_COUNTER_FULL_OFF_RAW.log`. This is not a
label import: `tss_corpus_check` asserts that every hard result has a
certificate and calls the strict `TssVerifier` on it before printing the row.
The retained run ended `CORPUS_DONE failures=0` and `test result: ok`.

The run used a 1 GiB TT, the standard 10k→100k→1M→20M ladder, pair-complete
width, `semantic_horizon=u32::MAX`, lazy frontier on, and the interior census
gate on. The two census/telemetry flags do not mint verdicts; every row below
is certificate-backed. Coordinates are the exact replay roots in
`forcing_corpus_moves.txt`. They each seed one D6 orbit; the retained log did
not record 12-image certificate remapping, so that stronger field is marked
only by the new pass.

| Source position | Stones | Side / phase | Certified verdict | First rung | Nodes | Solve ms |
|---|---:|---|---|---:|---:|---:|
| `0hz3hty` | 21 | P1 / FirstStone | **WIN** | 10k | 2,412 | 150.6 |
| `0l4291i_live` | 63 | P0 / FirstStone | **WIN** | 20M | 1,879,612 | 231,016.4 |
| `8is963b` | 103 | P0 / FirstStone | **LOSS** | 10k | 1 | <0.1 |
| `acly7kb` | 93 | P1 / FirstStone | **WIN** | 10k | 75 | 10.8 |
| `dy3dg99` | 35 | P0 / FirstStone | **LOSS** | 10k | 1 | <0.1 |
| `g2xx6wl` | 139 | P0 / FirstStone | **WIN** | 10k | 4,107 | 709.6 |
| `hu01jk4` | 149 | P1 / FirstStone | **WIN** | 10k | 380 | 116.5 |
| `jh7yo7y` | 35 | P0 / FirstStone | **WIN** | 10k | 2,119 | 293.3 |
| `jnzzmcm` | 67 | P0 / FirstStone | **WIN** | 10k | 9,798 | 1,030.1 |
| `lz60mfb` | 41 | P1 / FirstStone | **WIN** | 1M | 109,896 | 11,248.8 |
| `xsnfyll` | 13 | P1 / FirstStone | **WIN** | 10k | 82 | 5.1 |
| `zrugh2x` | 45 | P1 / FirstStone | **WIN** | 100k | 41,734 | 4,789.4 |
| `strongloss_a_prefix6` | 9 | P1 / FirstStone | **WIN** | 100k | 16,126 | 1,248.6 |
| `strongloss_b_prefix8` | 11 | P0 / FirstStone | **WIN** | 10k | 1,099 | 74.7 |
| `hayes_20260712_turn16` | 31 | P0 / FirstStone | **WIN** | 100k | 11,664 | 1,530.6 |
| `hayes_20260712_placement31` | 32 | P0 / SecondStone | **WIN** | 100k | 11,664 | 1,492.3 |

This baseline contributes 14 certified wins and two certified losses. The
three other NO controls in the 19-row corpus remained UNKNOWN and are not
verdict rows. No draw was claimed.

### New bounded pass

The new pass attempted all 122 planned representatives and stopped with
`residual=0` inside that slice.

| Slice | Attempted | CERTIFIED WIN | CERTIFIED LOSS | UNKNOWN |
|---|---:|---:|---:|---:|
| Empty root + origin root + 24 first-reply D6 representatives | 26 | 0 | 0 | 26 |
| Eight human games × final 12 nonterminal prefixes | 96 | 37 | 6 | 53 |
| **New-pass total** | **122** | **37** | **6** | **79** |

Every one of the 43 hard rows closed at the first 10k rung; every UNKNOWN row
was retried at 100k. The successful pass expanded 8,678 nodes (8,800 solver
nodes), with 800.405 ms summed per-row solve time. The 26 quiet shallow roots
all returned UNKNOWN in two nodes each: that is a precise bounded result, not
a balance verdict.

The exact certified prefix set is below. `p:S-V(T,n)` means source prefix `p`,
side `S` to move, certified status `V`, derived certificate horizon `T`, and
solver nodes `n`.

| Human game | Certified prefixes |
|---|---|
| `00070cdd8fb87f42` | `67:P0-WIN(T77,n14)`; `73:P1-WIN(T83,n7)`; `77:P1-WIN(T79,n1)`; `78:P1-WIN(T79,n1)` |
| `001165e4e1d7e246` | `19:P0-WIN(T21,n1)`; `20:P0-WIN(T21,n1)` |
| `001c0059e69f6973` | `51:P0-WIN(T53,n1)`; `55:P0-WIN(T57,n1)`; `56:P0-WIN(T57,n1)` |
| `002f5360162bac9b` | `47:P0-WIN(T57,n148)`; `48:P0-WIN(T57,n6619)`; `51:P0-WIN(T57,n2)`; `52:P0-WIN(T57,n2)`; `53:P1-LOSS(T57,n1)`; `54:P1-LOSS(T57,n1)`; `55:P0-WIN(T57,n1)`; `56:P0-WIN(T57,n1)` |
| `0035f32035e5468b` | `49:P1-WIN(T59,n8)`; `50:P1-WIN(T59,n135)`; `53:P1-WIN(T59,n2)`; `54:P1-WIN(T59,n2)`; `55:P0-LOSS(T59,n1)`; `56:P0-LOSS(T59,n1)`; `57:P1-WIN(T59,n1)`; `58:P1-WIN(T59,n1)` |
| `00386e2d3c6f65fd` | `27:P0-WIN(T33,n2)`; `31:P0-WIN(T37,n2)`; `37:P1-WIN(T39,n1)`; `38:P1-WIN(T39,n1)` |
| `003c115aa968eb5a` | `35:P0-WIN(T41,n2)`; `36:P0-WIN(T41,n2)`; `37:P1-LOSS(T41,n1)`; `38:P1-LOSS(T41,n1)`; `39:P0-WIN(T41,n1)`; `40:P0-WIN(T41,n1)` |
| `004759ff34cefdc2` | `35:P0-WIN(T45,n23)`; `36:P0-WIN(T45,n8)`; `39:P0-WIN(T45,n2)`; `40:P0-WIN(T45,n2)`; `43:P0-WIN(T49,n2)`; `44:P0-WIN(T49,n2)`; `45:P1-WIN(T47,n1)`; `46:P1-WIN(T47,n1)` |

All 43 canonical-root certificates passed the strict verifier. The additional
D6-remap audit accepted all 12 images for 35 certificates. For eight
nontrivial WIN certificates it accepted six images and rejected six (48
printed audit rejections total). Those eight representative verdicts remain
strictly certified at symmetry 0; this pass does **not** claim strict
certificate acceptance for their rejected serialized images. This is a
precise follow-up seam in `d6_remap_certificate`, not an UNKNOWN downgrade of
the exact representative.

## 5. Sharp examples and observed thresholds

The imported baseline already contains two useful kinds of sharp example:

- `xsnfyll` is a compact 13-stone P1 win, certified in only 82 nodes at the
  10k rung. Its exact replay is
  `(0,0);(-1,0);(1,-2);(-2,0);(1,0);(0,-2);(1,-3);(0,-3);`
  `(2,-5);(2,-4);(1,-4);(3,-4);(3,-2)`.
- `8is963b` and `dy3dg99` are genuine dual results: both are P0-to-move
  **CERTIFIED LOSS** roots, resolved in one solver node. They are not merely
  NO/UNKNOWN controls anymore at this profile.
- The node-rung threshold is materially sharp across the baseline. Twelve
  wins close by 100k, `lz60mfb` first closes at 1M, and `0l4291i_live` first
  closes only at the 20M rung. These are discovery thresholds for this fixed
  ladder/profile, not lower-bound theorems.

The new pass found one exact adjacent verdict flip in the scanned corpus:

| Game | Before | Played placement | After |
|---|---|---|---|
| `004759ff34cefdc2` | prefix 44, P0 `SecondStone`: **CERTIFIED P0 WIN**, 2 nodes, derived T=49 | source ply 44: `(14,-3)` | prefix 45, P1 `FirstStone`: **CERTIFIED P1 WIN**, 1 node, derived T=47 |

Thus `(14,-3)` is a certificate-backed losing blunder in that exact opening:
the proven winner changes from P0 to P1 after one placement. This is a game
verdict flip, not merely a change from known to UNKNOWN.

The deepest new proof by node count is
`oa-558f79a590c31b6a` (game `002f5360162bac9b`, prefix 48): P0 to move at
`SecondStone`, **CERTIFIED WIN**, 6,619 nodes, 18 certificate nodes, derived
T=57. Its immediately preceding prefix 47 is also a P0 win but needs only 148
nodes; the large fixed-profile cost jump is a useful solver-sharp example,
not a proof-complexity lower bound.

## 6. Certificate provenance and reproduction

Normative inputs for this run:

| Input | Provenance |
|---|---|
| Worktree tip | `ad606d0e5100dcef6f67cf5d8ede120e11ce5627` |
| Normative engine gate | round-9b `ac3f455f759232ec1e1cfc210cddc90a4558ca06`; `.codex-round9b-gate/GATE.md` records 14/14 forcing WIN certificates, 5/5 NO non-WIN, 436.8 s |
| Human corpus | `hexo_human_corpus.jsonl`, SHA-256 `54fae7aebcef2a9d19d13c1946fae36c0565e21bc726c25e2e4e230cfb42a5b7` |
| `tss_solver.rs` | SHA-256 `29260ed9455d776a6d5427f4c25a09af12917b99c7b044340f58d319d645d244` |
| `tss_verify.rs` | SHA-256 `9990d38618da2204351e328ca0143be2aef98bb3001e4a0462cf346b707f2ce8` |
| `tss_core.rs` | SHA-256 `20586fd9874f8429eae405be184a2751d874d0da172c1397fa4a3ac8ddbe03f2` |
| Imported hard-row log | `CLOSURE_COUNTER_FULL_OFF_RAW.log`, commit `b7e9f36c62b2bbf185548e82f37d74b5e363c449`, SHA-256 `085bc5437737f4843bc674690dc084ce983154025ba33e2f00def188e5ac5280` |
| New pass log | `OPENING_ATLAS_PASS1_RAW.txt`, SHA-256 `f721071a1ca46df49edf70c3668cdc3299f885ce0717ebfd25fa2b37ff1446e4` |

From the worktree root, after satisfying the host-wide RAM/Cargo gate:

```powershell
Get-ChildItem Env: | Where-Object Name -Like 'TSS_*' | ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:CARGO_TARGET_DIR='.target-atlas'
$env:OPENING_ATLAS_CORPUS='E:\Hexo-BotTrainer-hexgt\data\hexo-bootstrap-corpus\hexo_human_corpus.jsonl'
$env:OPENING_ATLAS_GAME_COUNT='8'
$env:OPENING_ATLAS_BACKTRACK='12'
$env:OPENING_ATLAS_TT_BYTES='536870912'
$env:OPENING_ATLAS_RELATIVE_HORIZON='12'
$env:OPENING_ATLAS_WALL_SECONDS='1200'
cargo test --release --target x86_64-pc-windows-msvc -p hexfield_eq opening_atlas_pass1 -- --ignored --test-threads=1 --nocapture
```

The test is ignored and the module is compiled only under `cfg(test)`; it
cannot alter production behavior. `OPENING_ATLAS_CORPUS` is optional: omitting
it runs only the shallow roots. The complete solve log is
`OPENING_ATLAS_PASS1_RAW.txt`; filter `ATLAS_ROW` for schema rows and
`ATLAS_DONE` for the batch boundary. Reproduction requires strict verifier
acceptance; matching the FNV debug fingerprint is only a useful secondary
identity check.

The cargo-free census is:

```powershell
python scripts/opening_atlas_census.py
```

## 7. Precise boundary and residual

The catalog now contains **59 distinct certified representatives**: 51 WIN and
8 LOSS, with no overlap between the 16 imported and 43 newly solved roots. No
draw is certified.

The exact stopping boundary is:

1. **Systematic shallow tree.** Empty, forced origin, and all 24 D6 classes
   after P1's first placement were run at +12 through 100k: 26/26 UNKNOWN.
   The next complete layer has exactly 3,684 D6 representatives after P1's
   two-stone turn; **0/3,684 were solved**, so all 3,684 are the exact residual
   at placement depth 3. No systematic enumeration was attempted beyond that
   layer.
2. **Human final-12 slice.** Across all 6,902 games there are 82,824 raw
   source prefixes and exactly 82,792 D6-unique positions under the schema.
   This pass attempted 96 and left **82,696 D6 representatives unattempted**.
   Of the attempted 96, 43 are certified and 53 remain UNKNOWN at +12/100k.
3. **Forcing corpus.** All 19 retained roots were run by the imported gate:
   16 are certified here. `94gnnol`, `l9mxn59`, and `mvp2lvc` remain UNKNOWN;
   their NO labels are not verdicts.
4. **Resource/horizon boundary.** New hard rows are proven only for their
   exact roots and returned derived horizons. The 79 new UNKNOWN rows need a
   larger semantic horizon, more than 100k nodes, a wider search regime, or
   some combination; this pass does not guess which. No result is extrapolated
   to an unsearched child.
5. **D6 certificate audit.** Eight certified representative WINs have 48
   rejected mechanically remapped images (six each). Investigating that
   remapper/verifier asymmetry is explicit residual work; it does not weaken
   strict acceptance at the exact representative root.

This boundary is intentionally a proven partial atlas rather than a claim to
have adjudicated quiet early Hexo globally.

## 8. Compute accounting and landing note

The session began at 2026-07-18 19:28:50 EDT. The mandated foreign-Cargo gate
cleared at 21:03:35 EDT with 13.07 GiB available and 12.93 GiB free; no Cargo
command from this lane ran before that point.

New Cargo/solver work was short:

- compile + Rust D6 census test: 17.6 s wall (test body 0.08 s);
- first audit pilot: 0.51 s wall, stopped honestly on the unexpected D6-remap
  rejection and used only to correct the provenance schema;
- successful full pass: 10.09 s command wall, 0.908 s harness wall, 8,800
  solver nodes, 8,678 expansions, and 800.405 ms summed row solve time;
- default production-profile `hexfield_eq` build: 3.67 s;
- maximum single successful row: 421.087 ms. No position approached a long
  solve, and the 20-minute batch stop never fired.

Total new Cargo command wall was about 31.9 s. End-to-end task wall through
final verification was about 104 minutes, of which roughly 95 minutes was the
required host-wide Cargo wait.

What landed: this document; an ignored `cfg(test)` certificate harness;
the cargo-free exact census script; the retained raw new-pass log; and a
task-local target-directory ignore. Production behavior is unchanged and all
new solve code is default-off. Exact remaining work is the 3,684 completed
two-stone shallow representatives, 82,696 unattempted D6-unique final-12 human
prefixes, 53 attempted-but-UNKNOWN human roots, three imported UNKNOWN forcing
roots, and the eight-row D6 remap audit seam described above.
