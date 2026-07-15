# Round 6 progress

Build base: `199d078d8d2dc1717f710a36964bf1bc229c9d32`

## Checkpoint 1: clean baseline

- Windows toolchain: cargo 1.95.0, rustc 1.95.0.
- Target reused in place: `.target-codex/`; `CARGO_BUILD_JOBS=4`.
- Full command: `cargo test --release -p hexfield_eq`.
- Result: 100 passed, 0 failed, 3 intentionally ignored; doc tests green.
- 12-entry harness used only `TSS_CORPUS_ID` and `TSS_CORPUS_MAX_CAP=100000`;
  no engine behavior override was set.

| Entry | Banked rung | Status | Nodes | TT hits |
|---|---:|---|---:|---:|
| 0hz3hty | 10k | WIN | 3,132 | 3,788 |
| acly7kb | 10k | WIN | 75 | 0 |
| g2xx6wl | 10k | WIN | 5,610 | 4,470 |
| hu01jk4 | 10k | WIN | 380 | 0 |
| jh7yo7y | 10k | WIN | 1,681 | 194 |
| jnzzmcm | 10k | WIN | 8,032 | 1,634 |
| xsnfyll | 10k | WIN | 77 | 1 |
| zrugh2x | 100k | WIN | 51,733 | 13,830 |
| strongloss_a_prefix6 | 100k | WIN | 15,923 | 7,614 |
| strongloss_b_prefix8 | 10k | WIN | 682 | 151 |
| hayes_20260712_turn16 | 100k | WIN | 14,888 | 3,747 |
| hayes_20260712_placement31 | 100k | WIN | 14,888 | 3,747 |

Harness result: `CORPUS_DONE failures=0` (465.9 s wall time).

## lz60mfb re-bank

- Clean default-profile 100k ladder (all engine A/B hooks explicitly absent):
  - 10k: UNKNOWN, 10,000 nodes, 2,616 TT hits.
  - 100k: UNKNOWN, 100,000 nodes, 10,838 TT hits.
- The 100k rung is therefore insufficient on `199d078d`; proceed unchanged
  to the permitted 1M rung.
- 1M ladder result: **WIN at 125,020 nodes**, 12,300 TT hits.
- Banked rung: 1M. This improves the historical pre-unconditional-pair bank
  of 213,854 nodes, but remains 25,020 nodes above the 100k rung.

## 0l4291i_live backward-walk localization

The existing ignored helper was run unchanged at its hard-coded 10k cap.
Its final failing assertion is expected diagnostic behavior.

| Prefix | Goal result | Nodes | TT hits |
|---:|---|---:|---:|
| 32 | WIN | 1 | 0 |
| 28 | WIN | 2 | 0 |
| 24 | WIN | 29 | 0 |
| 20 | WIN | 68 | 6 |
| 16 | WIN | 72 | 6 |
| 12 | UNKNOWN | 10,000 | 1,919 |

Automatic placement probes across attacker turn 4 and defender turn 4:

| Prefix | Expected | Result | Nodes | TT hits |
|---:|---|---|---:|---:|
| 13 | WIN | UNKNOWN | 10,000 | 2,100 |
| 14 | LOSS | UNKNOWN | 10,000 | 2,104 |
| 15 | LOSS | LOSS | 525 | 43 |
| 16 | WIN | WIN | 72 | 6 |

Localization: the line is cheap again immediately after the reference first
defender placement (prefix 15), while the defender-at-FirstStone state at
prefix 14 explodes. The first missing search-shape mechanism is therefore
the prefix-14 defender universal / its interior ordering or frontier
resolution. It is not attacker width, the turn-forcing gate, or root pair
selection.

### Prefix-14 branch localization

- The state has two disjoint live count-4 windows, `tau=b=2`, and a four-cell
  K2 kernel. Pair canonicalization produces four distinct defender-pair
  obligations (the 2x2 transversal cross-product).
- At the default 10k trace, three obligations are already proven. The only
  unresolved pair is `(9,-2)+(12,-1)` (`pn=997,dn=663`). The fixture's
  `(9,-3)+(12,-1)` reply is already proven, so the actual blocker is an
  off-reference defense.
- Graph-PN A/B at prefix 14: UNKNOWN at 10k, 1,986 TT hits. This profile only
  changes Choice PN ordering, while prefix 14 is a Universal selected by DN;
  it does not address this root.
- Legacy certificate DFS: UNKNOWN at 10k and 100k (100k took 13.5 s). The
  existing DFS is therefore not itself a cheap frontier solver for this
  branch.
- Exact unresolved defender-pair child with interior-equivalent tier OFF:
  UNKNOWN at 10k, 1,962 TT hits.
- Root-tier A/B on that child is decisively worse: it spends the full 10k on
  the sole tier-0 pair (`pn` rises to 2,036) while the two tier-1 pairs remain
  unlinked, and still returns UNKNOWN. Interior tier persistence is rejected.

