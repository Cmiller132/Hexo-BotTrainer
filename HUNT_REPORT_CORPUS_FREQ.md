# HUNT_REPORT_CORPUS_FREQ — empirical grounding of the TSS phenomena in human play

**Question.** How often do the TSS campaign's theoretical phenomena actually occur
in real human Hexo games? This grounds the flagship paper and ranks which theory
fronts real play exercises most.

All numbers are reproducible from the in-crate measurement module
`packages/hexfield_eq/rust/src/tss_freq_hunt.rs` (two ignored `#[test]` helpers,
`freq_cheap` and `freq_vcf`), added on branch `hunt/corpus-freq`. Threat semantics
are exactly `crate::threats_shared` (the single source of TSS threat truth used by
every model lineage); the WIN oracle is exactly the corpus gate's
`WidthOptions::vcf_pair_complete()`; the ES potential Φ is a verbatim port of
`hunt/gap-raw`'s `gap_raw_hunt.rs`.

---

## 0. Corpus provenance

| field | value |
|---|---|
| dataset | HuggingFace `timmyburn/hexo-bootstrap-corpus` (MIT) |
| local file used | `E:\Hexo-BotTrainer-hexgt\data\hexo-bootstrap-corpus\hexo_human_corpus.jsonl` |
| sha256 | `54fae7aebcef2a9d19d13c1946fae36c0565e21bc726c25e2e4e230cfb42a5b7` |
| bytes | 3,696,030 |
| games | **6,902** (per `dataset_metadata.json` `n_games=6902`; 0 duplicates) |
| created_at | 2026-06-04T14:27:03Z |
| schema | one game/line: `game_hash`, `moves` `[[q,r]…]` (axial, opener always `(0,0)`), `winner` ±1 (engine convention: +1 = Player0/opener, −1 = Player1), `elo` `[p1,p2]` |
| source filter | rated, ≥20 moves, decisive by six-in-a-row (no draws) |

A byte-identical second copy exists at
`runs\dense_cnn_restnet_main1_prefit\hf_corpus\hexo_human_corpus.jsonl` (same sha256).

> **Discrepancy flagged.** The auto-memory note `hexo-human-corpus.md` records "8,698
> games" and sha256 `b2fe61eb…840c`. The actual published/local dataset is **6,902
> games**, sha256 `54fae7ae…a5b7`, confirmed by the dataset's own
> `dataset_metadata.json`, `README.md`, and `SCHEMA.md` (all three say 6,902). This
> report uses the 6,902-game local copy. The memory figure appears stale and should
> be corrected.

Replay sanity: all 6,902 games open at `(0,0)` (`opener_not_origin=0`); replaying
every game to its recorded terminal produced **431,495** nonterminal decision nodes
(one per single-stone placement), all legal.

Engine: worktree `hunt/corpus-freq` at `b4ec2e73` (round-9b certificate-grade
engine + Group-2 mining module). `CARGO_TARGET_DIR=.target-hunt`.

**Oracle soundness (validated).** Running the branch's own acceptance gate at my
10k node cap (`TSS_CORPUS_MAX_CAP=10000 … tss_corpus_check`) returns **zero** WINs
on the 5 `expect=NO` positions (they return LOSS/UNKNOWN) and proves 8/14 true-WIN
positions already at 10k. So every WIN my oracle reports is a real certified forced
win, and 10k is a **conservative** cap → the VCF-exists rates below are **lower
bounds**.

---

## 1. Defender-width denominators (capstone-relevant)

At every nonterminal node the side to move is the "defender" with respect to the
opponent's active count-≥4 windows. Budget `B = placements_remaining` (2 at
FirstStone = start of turn, 1 at SecondStone). Hitting number `k` = minimum cells
covering an empty of every opponent count-≥4 window (`crate::threats_shared`, exact,
capped at B; `k>B` ⇒ 1-ply forced loss). A **threatened defender node** faces ≥1
such window; **genuine** = also excludes nodes where the mover itself wins now.

| set | count | % of all nodes |
|---|---:|---:|
| all nonterminal decision nodes | 431,495 | 100% |
| threatened (opp count-≥4 window ≥1) | 85,502 | 19.8% |
| …of which mover-wins-now (excluded) | 6,516 | 1.5% |
| **genuine defender nodes** | **78,986** | **18.3%** |
| … at B=2 (FirstStone, start of turn) | 46,902 | — |
| … at B=1 (SecondStone) | 32,084 | — |

**Hitting number k vs budget B (genuine defender nodes):**

| regime | B=2 | B=1 | total | % of genuine def |
|---|---:|---:|---:|---:|
| **unforced** `k<B` (defender has a spare stone) | 20,307 (k=1) | 0 (impossible) | **20,307** | **25.7%** |
| forced-exact `k=B` (all stones pinned) | 22,357 (k=2) | 27,091 (k=1) | 49,448 | 62.6% |
| forced-loss `k>B` (no defense) | 4,238 | 4,993 | 9,231 | 11.7% |

### Headline
**25.7%** of threatened defender nodes are **unforced** (`k<B`) — the regime the
Group-2 zone machinery governs (defender not compelled to a unique reply → a wide
viable move-set). The other **74.3%** are the already-solved forcing case (62.6%
exact-forced + 11.7% forced-loss). Restricted to start-of-turn nodes only,
`k<B` rises to **43.3%** of B=2 defender nodes (20,307 / 46,902) — because slack
is only possible when 2 placements remain.

`|Legal|` is **not** a usable width denominator: at genuine defender nodes it is
min 272 / p50 503 / p90 839 / max 3,871 / mean 576 — always in the hundreds,
because the infinite board's legality is "radius-8 around any stone." The
tactically binding width is the hitting structure (`k` vs `B`), not `|Legal|`.

Regenerate:
```
CARGO_TARGET_DIR=.target-hunt \
TSS_FREQ_CORPUS=<path> \
cargo test --release -p hexfield_eq freq_cheap -- --ignored --nocapture --test-threads=1
```
(tags `FREQ_THREATENED`, `FREQ_DEF_B`, `FREQ_KB`, `FREQ_UNFORCED`, `FREQ_LEGAL`)

---

## 2. VCF incidence

Fixed-seed sample of mid/late attacker-to-move nodes: every FirstStone node with
`placements_made ≥ 20` is a candidate (**pool = 143,336**); a seeded Fisher-Yates
shuffle (`seed = 0x9E3779B97F4A7C15 = 11400714819323198485`) takes the first N.
Each is solved with `vcf_pair_complete`, node cap 10,000, TT cap 512 MiB, one solve
at a time. WIN = certified continuous-forcing win exists for the mover; UNKNOWN =
not provable at 10k (labelled honestly, never a NO); LOSS = mover is itself in a
forcing loss.

| N | WIN-exists | UNKNOWN | LOSS (mover losing) | **WIN rate** |
|---:|---:|---:|---:|---:|
| **2000** (primary) | 506 | 1,422 | 72 | **25.3% ± 1.9%** |
| 500 (consistency) | 134 | 349 | 17 | 26.8% ± 3.9% |

**Human accuracy.** When a certified win exists, did the human's actually-played
first stone keep the win (child still WIN under the same oracle)?
**325 / 506 = 64.2%** (n=2000; 67.2% at n=500). At ~1000–1100 median Elo, humans
find a winning first move about two-thirds of the time when one exists.

> Because 10k is a conservative cap (§0), 25.3% is a **lower bound** on the true
> forced-win incidence among mid/late attacker-to-move positions.

Regenerate:
```
CARGO_TARGET_DIR=.target-hunt TSS_FREQ_CORPUS=<path> TSS_FREQ_VCF_N=2000 \
cargo test --release -p hexfield_eq freq_vcf -- --ignored --nocapture --test-threads=1
```
(tags `FREQ_VCF_INCIDENCE`, `FREQ_VCF_HUMAN`; knobs `TSS_FREQ_VCF_SEED`,
`TSS_FREQ_VCF_CAP=10000`, `TSS_FREQ_TT_BYTES=536870912`, `TSS_FREQ_MIN_STONES=20`)

---

## 3. Sharp-phenomenon incidence

### 3a. Pileup (≥3 simultaneous count-≥4 attacker windows facing a defender turn)

| metric | value |
|---|---:|
| genuine defender nodes with opp count-≥4 windows ≥3 | 27,326 (**34.6%** of genuine def) |
| …at B=2 (start of turn) | 24,526 (**52.3%** of B=2 def) |
| games containing ≥1 pileup node | **6,355 / 6,902 = 92.1%** |

Overlapping length-6 windows inflate the raw count (a single open-4 already spans
~3 windows), so cross-tabbing pileup nodes by hitting number `k` separates cheap
clusters from genuine forks:

| pileup sub-class | count | % of pileup |
|---|---:|---:|
| single-hit `k=1` (one stone kills the cluster) | 215 | 0.8% |
| two-target `k=2` at B=2 (both stones needed) | 20,125 | 73.6% |
| forced-loss `k>B` | 6,986 | 25.6% |
| **hard (`k≥2` or loss)** | **27,111** | **99.2%** |

So the pileup pattern in real play is **overwhelmingly the hard case**: a ≥3-window
pileup consumes the defender's whole turn (or loses) 99.2% of the time. This is the
GAP-RAW dense-fork regime, and it is common (present in 92% of games).

Distribution of opponent window count at genuine defender nodes: 1→51.0%, 2→14.4%,
3→26.5%, 4→5.8%, ≥5→2.3%.

### 3b. λ² signature (winning line required a quiet / non-threat move)

Among sampled attacker-FirstStone nodes where the **mover eventually won the game**
(n=2000), the fraction with **no** VCF-forcing win at the node = the win demanded a
quiet, non-threatening (λ²⁺) preparation move, bucketed by plies-to-end:

| plies to game end | won nodes | VCF-forcing | quiet-required | **quiet %** |
|---|---:|---:|---:|---:|
| 1–6 | 196 | 179 | 17 | 8.7% |
| 7–12 | 79 | 42 | 37 | 46.8% |
| 13–20 | 148 | 64 | 84 | 56.8% |
| 21–40 | 258 | 52 | 206 | 79.8% |
| 41+ | 391 | 51 | 340 | **87.0%** |

A clean monotone gradient: near the end wins are forcing (VCF-provable); wins that
originate >12 plies out require quiet setup **~80–87%** of the time. The λ²
phenomenon is not exotic — it is the norm for wins that begin far from the end.
(n=500 gave the same shape: 12% → 29% → 51% → 81% → 83%.)

### 3c. Low-Φ defender positions (Φ < 1)

Exact ES potential Φ (surd `A + B√3`, port of `gap_raw_hunt.rs`; convention
attacker = Player1, defender = Player0) evaluated at every Defender(Player0)-
FirstStone node:

| set | nodes | Φ<1 | fraction |
|---|---:|---:|---:|
| all Player0-FirstStone nodes | 104,452 | 25 | **0.024%** |
| developed (≥6 attacker stones) | 90,648 | 5 | 0.006% |

The GAP-RAW `Φ<1` regime is **essentially absent** from real human play. Caveat: Φ
uses the asymmetric Maker-Breaker convention (attacker = second mover, defender =
opener) and counts only attacker-alive windows, ignoring the defender's own
offense; by any real Player0-to-move node the opponent is well developed, so Φ ≥ 1
almost always. The `Φ<1` boundary — central to the zone-soundness proof — describes
a measure-zero slice of actual games.

(All of §3 regenerate from `freq_cheap` / `freq_vcf` above; tags `FREQ_PILEUP`,
`FREQ_PILEUP_K`, `FREQ_THREATN`, `FREQ_L2`, `FREQ_PHI`.)

---

## 4. Opening families (atlas seed)

Each opening family is the P2 first-reply stone pair `{moves[1], moves[2]}`
canonicalized up to the 12 D6 symmetries (the opener at the origin is D6-fixed).
This is the natural top-level branch the certified atlas root-splits on. Coordinates
below are canonical offsets from the center opener `(0,0)`.

**262** distinct D6-canonical P2 replies (vs 863 raw pre-canonicalization). Top families:

| rank | P2 reply (canonical) | games | frac | cumulative |
|---:|---|---:|---:|---:|
| 1 | `{(-1,0), (0,1)}` | 1,337 | 19.4% | 19.4% |
| 2 | `{(-1,0), (-1,1)}` | 1,157 | 16.8% | 36.1% |
| 3 | `{(-2,0), (-2,2)}` | 523 | 7.6% | 43.7% |
| 4 | `{(-3,0), (-1,1)}` | 258 | 3.7% | 47.5% |
| 5 | `{(-2,1), (0,-1)}` | 175 | 2.5% | 50.0% |
| 6 | `{(-2,0), (0,2)}` | 169 | 2.4% | 52.5% |
| 7 | `{(-9,1), (-8,0)}` | 165 | 2.4% | 54.9% |
| 8 | `{(-2,1), (-1,0)}` | 149 | 2.2% | 57.0% |
| 9 | `{(-2,0), (0,1)}` | 146 | 2.1% | 59.2% |
| 10 | `{(-2,0), (-1,1)}` | 133 | 1.9% | 61.1% |

The distribution is strongly concentrated: **top 2 = 36% of games, top 5 = 50%,
top 10 = 61%** of the whole corpus. (Rank 7 `{(-9,1),(-8,0)}` is a legitimate
"far-frontier" opening style — P2's first stone at radius-8 then extends.)

Finer "first-two-turn" classes (P2 reply pair + P1's second-turn pair, labelled,
D6-canonical): **2,556** distinct classes; the single largest is
`P2 {(-1,0),(-1,1)}, P1 {(-1,-1),(0,-1)}` at 6.0% — i.e., each P2 family fans out
into many P1 continuations, so atlas ordering by the P2 family (above) is the
actionable ranking.

Regenerate: `freq_cheap` (tags `FREQ_FAM_COUNT`, `FREQ_FAM_P2`, `FREQ_FAM_TWO`).

---

## 5. Implications — which theory fronts real play exercises most

Ranked by how much of real play each governs:

1. **Forcing / hitting-set case dominates threatened defense (74%).** Of the 78,986
   genuine defender nodes, 62.6% are exact-forced (`k=B`) and 11.7% are forced-loss
   — the already-solved λ¹ regime. The engine's forcing machinery is the workhorse.

2. **Dense pileups are the common shape of a threatened turn.** 34.6% of threatened
   defender nodes (52% of start-of-turn ones) face ≥3 simultaneous windows, and
   99.2% of those genuinely need both defender stones (or lose). Present in 92% of
   games. The "sharp" GAP-RAW pileup is not a corner case — it is what a defended
   turn typically looks like.

3. **The Group-2 zone machinery (unforced, `k<B`) governs ~1/4 of threatened
   defense (25.7%; 43% of start-of-turn nodes).** Substantial and the right next
   front — but a minority of threatened decisions, and the capstone should frame it
   as "the ~quarter of defended turns where the defender has real choice," not the
   bulk.

4. **λ² (quiet-move) wins are the norm for non-terminal-range wins.** Only ~1/4 of
   mid/late attacker-to-move positions have a pure continuous-forcing win at 10k
   (25.3%, a lower bound), and 80–87% of wins that begin >12 plies from the end
   required a quiet setup move. Pure VCF explains endgames; the interesting wins are
   λ²⁺. This most justifies the deepening beyond continuous forcing.

5. **The `Φ<1` boundary regime is empirically vanishing (0.02%).** It is essential
   to the zone-soundness *proof* but describes almost none of real play; the paper
   should present it as a theoretical soundness boundary, not an empirically frequent
   situation.

6. **Human accuracy leaves clear headroom.** When a forced win exists, humans find a
   winning first move only 64% of the time — a concrete bar the trained agent should
   clear.

**Ranked opening-family list for the certified atlas** (solve in this order; D6-
canonical P2 replies, offsets from center; cumulative corpus coverage):
`{(-1,0),(0,1)}` (19.4%) → `{(-1,0),(-1,1)}` (36.1%) → `{(-2,0),(-2,2)}` (43.7%) →
`{(-3,0),(-1,1)}` (47.5%) → `{(-2,1),(0,-1)}` (50.0%) → `{(-2,0),(0,2)}` (52.5%) →
`{(-9,1),(-8,0)}` (54.9%) → `{(-2,1),(-1,0)}` (57.0%) → `{(-2,0),(0,1)}` (59.2%) →
`{(-2,0),(-1,1)}` (61.1%). Solving the top 5 families certifies half the corpus's
openings.

---

## 6. Caveats

- **VCF cap.** 10k node cap → WIN-exists (§2) and VCF-forcing (§3b) are lower
  bounds; UNKNOWN never means NO. Validated sound (no false WIN on the 5 NO gate
  positions at 10k).
- **Sampling.** §2/§3b use a 2,000-node fixed-seed sample of the 143,336-node pool
  (95% CI ≈ ±1.9% on the 25.3% headline; some λ² buckets are smaller). §1/§3a/§3c/§4
  are full-corpus (all 431,495 / 104,452 nodes; no sampling).
- **Window overlap.** Raw window counts over-count geometric threats (open-4 ≈ 3
  windows); §3a's `k`-cross-tab is the overlap-robust view. The `k`-vs-`B` measure
  (§1) is inherently overlap-robust (hitting sets account for shared cells).
- **Φ convention.** Asymmetric (attacker = Player1); Φ ignores the defender's own
  offense (§3c).
- **"Human found the win" proxy.** Defined as "the mover's actual first stone leaves
  a child that is still a certified WIN." Sound because the child is a strict
  subproblem of a proven-WIN parent (§2).
- Raw per-tag engine output saved at worktree root: `freq_cheap_output.txt`,
  `freq_vcf_output.txt` (n=500), `freq_vcf_2000_output.txt` (n=2000).

*Not git-committed — left for the orchestrator to gate and commit.*
