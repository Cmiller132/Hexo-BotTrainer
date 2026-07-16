# HUNT REPORT — Domination / Inferior-Cell Patterns (hunt/domination lane)

> **Status.** Empirical catalog (DATA, not proofs). Round discipline:
> computation → conjecture → (later round) proof. Every claim below is
> labeled SURVIVED-ALL-ATTACKS / REFUTED / UNDECIDED with its attack budget.
> Base: branch `hunt/domination` @ `5536b2bb` (round-3 wide engine).
> Harness: `packages/hexfield_eq/rust/src/tss_domination_hunt.rs`
> (test-gated, `#[ignore]`, registered `#[cfg(test)]` in lib.rs; no
> production code touched). Machine-readable data:
> `dom_hunt_records.jsonl` + `dom_hunt_counterexamples.jsonl` beside this
> file. All runs deterministic (fixed-seed xorshift; no RNG on any solved
> path).

## 0. Scope and adjudication model

**Node universe.** Defensive **SecondStone** nodes (defender to place, budget
b = 1, attacker holds ≥1 live count-4/5 window, defender has no own
win-now). This is where U11's single-cell sub-hitting dispatch lives and
where a single-cell reply can be adjudicated cleanly: the child is an
attacker-FirstStone node.

**Referee.** The round-3 wide engine (`WidthOptions::vcf_pair_complete`,
`SolveGoal::Win`) as a **sound attacker-WIN detector** at the child.
`ProofStatus::Loss` requires a dual certificate and is rarely minted, so
child values are three-ranked for the defender:
`att_cant_win (best) > att_unknown > att_win (worst; sound)`.
A pattern "keep x / delete y" is **REFUTED** when the kept child is proven
attacker-WIN while the deleted child resists at a much higher wide cap —
deleting y would then flip a real defense into a false attacker WIN at P.
Two-stage escalation: narrow scan cap → wide confirm cap; only firings
surviving both stages count as refutations (caps recorded per record).

**Replay conventions** copied verbatim from the validated leaf-width hunt
(branch hunt/leaf-width @ 8d97ac8d; replay validated 300/300 there;
re-validated 200/200 in this harness's selftest).

**Attack surface per candidate:** (a) corpus positions (real-game junctions
and threat clusters); (b) reachable random playouts; (c) the G3
defender-counterfork adversary explicitly enumerated at every forced node
(counter-threat cells = empties completing a defender count-≥3 alive
window); (d) directed re-verification of every flagged position at
escalating caps to 1,000,000 wide forcing nodes.

---

## 1. Catalog (ranked by fire rate × branching saved)

Numbers are from the MAIN SWEEP (N = 4,001 defensive SecondStone nodes,
seed 7766554433221100, §5.2); the independent pilot (N = 401, §5.1) agreed
on every rate.

| # | Pattern | Claim | Label | Fire rate (N=4,001 def-nodes) | Branching saved when fires |
|---|---------|-------|-------|-------------------------------|----------------------------|
| 1 | **DISPATCH-B1** (defender non-coverer dismissal, b=1) | at a forced node (mhs=1), every reply not hitting ALL attacker threat windows is dominated by any full-coverer | **SURVIVED-ALL-ATTACKS** (0/20,495 adversarial pairs; + proof sketch, §3.1) | 3,412/4,001 = **85.3%** of def-SecondStone nodes | ~all of legal-set minus the coverer frontier (hundreds of cells) — already the engine's U3 `implicit_dispatch`; this round adds the adversarial G3 validation it lacked |
| 2 | **DRQ dead+inert pass-equivalence** (feeds U24) | any two legal cells that are dead (all 18 windows two-coloured) AND frontier-inert are mutually outcome-equivalent; keep one class representative | **SURVIVED-ALL-ATTACKS** (0/288 adjudicated mismatches; + proof sketch, §3.2) | 5,133 pair-firings / 4,001 nodes | class size − 1 per node |
| 3 | **DBD-SUPERSET, guarded** (kill-superset block dominance) | if kill-set(x) ⊋ kill-set(y), new-support(x) ⊆ new-support(y), and neither x nor y touches a defender-alive window, then y is dominated by x | **SURVIVED within attack budget** (0 refuted of 1,329 firings / 117 guarded; UNDECIDED as a theorem candidate — see §3.3) | guarded firings 117/4,001 nodes (~2.9%) | 1 cell per firing (small) |
| 4 | **NAIVE COVERER-INTERCHANGE** | two cells both hitting every attacker threat window are interchangeable | **REFUTED** — 22 outcome-grade corpus counterexamples, **4 with BOTH verdicts proven**; canonical §2 | would fire at 3,013/3,412 = **88.3%** of forced nodes; referee-differs on 27/3,013 = 0.9% of pairs (each strong one a would-be false WIN) | (unsound) |
| 5 | **P2 as proven** (dead-spoke interchangeable hits) | (already PROVEN in DOMINATION.md) | fire-rate measurement only; **never violated** (0 P2-protected mismatches) | **2 firings / 4,001 nodes ≈ 0.05%** | ~0 in practice |
| 6 | **Attacker extension dominance** (class 3 of the brief) | extension toward open territory dominates extension toward dead territory | **UNDECIDED — not hunted this round** (defender-side hunting consumed the budget; the WIN-referee direction does support attacker-side adjudication — deleting attacker move y is refuted when kept x is not-WIN and y is WIN — so this is mechanically ready for a follow-up) | — | — |

**Headline shape of the boundary.** The proven layer (P2) has ≈zero corpus
fire rate (2/4,001 nodes) because its dead-spoke hypotheses almost never
hold in real positions; the naive generalization that WOULD fire (88% of
forced nodes) is refuted by 22 outcome-grade corpus counterexamples, four
of them with both verdicts proven. Meanwhile the two patterns that both
fire and survive (DISPATCH-B1, DRQ) are exactly the ones with short
first-principles proofs from the existing DOMINATION.md machinery. The
practical "dramatic search-space cut" at defender nodes is therefore NOT a
new exotic pattern — it is (a) hard-validating the dispatch frontier the
engine already uses, and (b) the dead-region quotient for U24.

---

## 2. Canonical counterexamples (exact coordinates + replay)

**REFUTES:** "same-window / equal-count hitting cells are interchangeable"
(and any coverer-collapse not gated by P2's hypotheses). The sweep found
**27** unique coverer-pair mismatches (all in `dom_hunt_records.jsonl`),
split by evidence strength:

- **4 doubly-proven** (attacker WIN after one block, attacker CANT-WIN
  after the other — both verdicts sound, zero cap caveat):
  `d7e1b56c925b7f32:20` (−2,3)/(−1,2); `1b73025a7265899c:38` (2,−5)/(2,−4);
  `41c4a1056d405fc7:90` (−1,−3)/(1,−3); `24d8dc7181b59042:40` (2,−1)/(4,−3).
- **18 strong** (proven attacker WIN after one block vs UNKNOWN at wide
  cap 250,000 after the other).
- **5 weak** (proven attacker-cant-win vs UNKNOWN — a proof-strength
  asymmetry, NOT a proven outcome difference; excluded from the
  refutation headline, kept in the data).

### 2.1 Canonical — both sides proven (game `d7e1b56c925b7f32`, prefix 20)

- Replay prefix = first **20** placements (standard convention: P0 opening
  at (0,0), alternating two-stone turns, `apply_placement` each (q,r) in
  order). Node: defender to place, SecondStone (b=1). Attacker holds
  exactly ONE threat window: axis **QR**, start (−3,4), count 4, empties
  **{(−2,3), (−1,2)}**. Both empties hit the single threat ⇒ both are
  minimum hitting cells; `mhs = 1`.
- **Block (−2,3): attacker forced WIN** — proven by the wide engine,
  stable at caps 60,000 / 250,000 / 1,000,000.
- **Block (−1,2): attacker CANT WIN** — sound λ¹ forced-loss for the
  attacker at the child: this block simultaneously completes a defender
  counter-fork whose threats the attacker cannot all answer with budget 2.
- Same window, same count, both minimum hitting cells — **opposite proven
  outcomes.** The mechanism is the G3 counterfork working in the
  *defender's* favour at one hitting cell but not the other: the
  intersection hypergraph through the two empties decides, exactly as
  DOMINATION.md §4's "why count profiles are insufficient" predicts — now
  with a real-game, doubly-proven witness.
- `p2_protected = false`: the non-shared spokes are not all dead, so the
  proven P2 correctly refuses to equate the two cells. Empirical proof
  that P2's dead-spoke condition is **load-bearing**, not bookkeeping.

```
CARGO_TARGET_DIR=.target-hunt TSS_DOM_VERIFY_HASH=d7e1b56c925b7f32 \
TSS_DOM_VERIFY_PREFIX=20 TSS_DOM_VERIFY_A="-2,3" TSS_DOM_VERIFY_B="-1,2" \
cargo test -p hexfield_eq --release dom_hunt_verify -- --ignored --nocapture --test-threads=1
```

### 2.2 Secondary — proven WIN vs 10^6-node resistance (game `060fe37eff4145c4`, prefix 56)

- Same node shape: one attacker threat window, axis **Q**, start (4,−1),
  count 4, empties **{(5,−1), (6,−1)}**, `mhs = 1`.
- **Block (5,−1): attacker forced WIN** (proven; stable at caps 4,000 /
  60,000 / 250,000 / 1,000,000). **Block (6,−1): attacker UNKNOWN at
  1,000,000 wide forcing nodes** — the defense holds every forcing line at
  that budget. The deeper cell blunts the attacker's follow-up window
  family; the shallow cell does not.
- Remaining doubly-proven witnesses (all `att_win` vs `att_cant_win` at
  cap 250,000): `1b73025a7265899c` prefix 38, (2,−5) vs (2,−4);
  `41c4a1056d405fc7` prefix 90, (1,−3) vs (−1,−3);
  `24d8dc7181b59042` prefix 40, (2,−1) vs (4,−3).

```
CARGO_TARGET_DIR=.target-hunt TSS_DOM_VERIFY_HASH=060fe37eff4145c4 \
TSS_DOM_VERIFY_PREFIX=56 TSS_DOM_VERIFY_A="5,-1" TSS_DOM_VERIFY_B="6,-1" \
cargo test -p hexfield_eq --release dom_hunt_verify -- --ignored --nocapture --test-threads=1
```

Machine-readable: `dom_hunt_counterexamples.jsonl` (canonical entries) and
`dom_hunt_records.jsonl` (all 27, trailing refutations section).

---

## 3. Top survivors — lemma statements ready for a hostile proof round

### 3.1 Lemma L-DISPATCH-B1 (b = 1 non-coverer dismissal)

**Hypotheses.** `P` nonterminal post-opening, defender `X` to place, phase
SecondStone (remaining budget 1). `T` = the set of live attacker windows
with count ≥ 4 (single-colour, attacker = X.other()). `T ≠ ∅` and `X` has
no own win-now. Let `c` be any legal reply leaving some `W ∈ T` with no
`X` stone.

**Claim.** `c` is n-outcome-dominated (Definition 5, DOMINATION.md) by
every legal full-coverer `a` (a reply hitting every window of `T`), for
every horizon n ≥ 3; hence non-coverers may be pruned at forced b=1 nodes.

**Proof shape.** `W` unhit stays alive-for-attacker with count ≥ 4
(permanence, Lemma 2). Its ≤ 2 remaining empties lie inside the 6-cell
window, pairwise distance ≤ 5 from attacker stones of `W`, hence legal.
After `X`'s completing stone the attacker moves FIRST with budget 2 and
fills the remaining empties of `W` — an attacker win within the turn,
before any defender counter-threat can fire. So `V_D^n(P+c) = −1`, the
minimum, and (5) of DOMINATION.md Lemma 3 gives domination by any reply.
(The only subtlety for the hostile round: `own_win_now = false` at `P`
must be shown to imply the completing stone cannot deliver an immediate
defender win from `c` either — or weaken the claim to "dominated unless
`P+c` is an immediate defender win", which the solver tests anyway.)

**Empirical.** **0 refutations over 20,495 adversarial pairs**
(full-coverer keep vs defender-counter-threat delete — the G3 mechanism)
across **3,412** forced nodes, scan cap 3,000 / confirm cap 250,000 (main
sweep, §5.2; pilot at 60,000 agreed with 0/1,929). This is the missing
adversarial validation for the engine's `implicit_dispatch` premise at b=1.

### 3.2 Lemma L-DRQ (dead + frontier-inert pass-equivalence; U24 seed)

**Hypotheses.** `P` nonterminal post-opening, mover `X` (either role).
`x ≠ y` legal empty cells, each **dead** (all 18 incident windows
two-coloured). (Frontier-inertness is then automatic by DOMINATION.md
Lemma 7.)

**Claim.** `x ⪯_n y` and `y ⪯_n x` for every n: the two replies are
mutually outcome-equivalent. Consequently the dead-cell class of `P` may
be collapsed to ONE representative macromove ("pass-into-dead-region"),
and this holds at every node type — the class-partition collapse U24 wants,
on the dead subclass.

**Proof shape.** Symmetrize P1: in P1's proof the discarded reply `b` is
dead+inert and the searched reply `a` needs only frontier-inertness. Here
BOTH cells are dead+inert, so the `a↔b` transposition argument runs in
both directions (every window through either cell is permanently dead —
mask channel inert; Lemma 7 twice — frontier channel inert; occupancy
channel handled by the same transposition as P1/P2). The bisimulation is
outcome-preserving, giving equivalence rather than one-way domination.

**Empirical.** **288 adjudicated dead+inert pairs (main sweep), 0 referee
mismatches**; 5,133 pair-firings observed over 4,001 nodes. Also directly
consistent with the executed MV-L7 spot checks (400 adversarial dead-cell
configs, 0 frontier violations) recorded in PROOF_TSS_DEFENDER_ZONES.md §11.

### 3.3 Conjecture C-DBD-SUPERSET (guarded kill-superset dominance) — weakest survivor

**Hypotheses.** Defensive node as above. Replies `x, y` with
kill-set(x) ⊋ kill-set(y) (windows of `T` hit), `new-support(x) ⊆
new-support(y)`, and neither `x` nor `y` incident to any defender-alive
window (both "pure blocks").

**Claim (conjectured).** `y ⪯_n x`.

**Status: SURVIVED within attack budget, but UNDECIDED as a lemma.** The
honest caveat: at b=1 forced nodes this conjecture is nearly subsumed by
L-DISPATCH-B1 (a strict-superset partial coverer is still a non-coverer
unless x covers everything), and the guarded firing rate is low
(117/4,001 nodes; 0 refuted of 1,329 total firings at scan cap).
The counterexample risk is exactly the canonical §2 mechanism — x's and
y's *attacker follow-up* spokes differ even when both are pure for the
defender. The guard set does not yet pin the attacker-side spokes, so a
proof attempt should EXPECT to need an additional "x's non-T attacker
spokes ⊆ y's" hypothesis. Recommended for the proof round only after
L-DISPATCH-B1 and L-DRQ.

---

## 4. What U11's sub-hitting dispatch algebra should actually assert

Given the data, the algebra splits into three legs with different fates:

1. **Dismissal leg (SOUND, provable).** At a b=1 forced defender node,
   prune every reply that is not a hitting cell of ALL live attacker
   count-≥4 windows. This is L-DISPATCH-B1 = the theorem form of the
   engine's `implicit_dispatch` premise at `min_hitting_set == b == 1`,
   now with 20,495-pair adversarial evidence including the G3 counterfork
   mechanism that killed naive r=2 trimming. U11 can cite this as the
   proven core of "sub-hitting dispatch".

2. **Interchange leg (UNSOUND without P2).** The algebra must NOT collapse
   distinct minimum hitting cells/sets to a canonical representative.
   22 outcome-grade corpus counterexamples (§2), including FOUR with BOTH
   verdicts proven: same-window count-4 hitting cells with opposite
   proven outcomes. The error rate matters: 0.9% of multi-coverer pairs
   referee-differ, and a wrong-way collapse mints a FALSE attacker WIN —
   at 88% of forced nodes having ≥2 coverers, an unguarded collapse would
   corrupt roughly 1 in 130 forced defender nodes. Any collapse must be
   gated by the PROVEN P2
   predicate (all non-shared spokes dead + successor-support equality) —
   whose measured corpus fire rate is ~0 — or by a genuinely new lemma
   with attacker-follow-up-spoke hypotheses (C-DBD-SUPERSET's missing
   guard). Nested-window "skip dominated hitting cells" enumeration is in
   this leg and is therefore NOT sound as sketched in the ledger.

3. **Spare-stone leg (b = 2, UNDECIDED by this referee).** With a spare
   placement the interesting dominations (which spare accompanies the hit;
   which of several 2-cell hitting sets) produce *successful quiet
   defenses* on both sides of the comparison — a WIN-only referee cannot
   separate them (both children go att_unknown). This needs the exact
   test-only oracle (`tss_reference_fast`) or a LOSS-complete engine as
   referee; deliberately left to the next round rather than reporting
   soft numbers here.

**Consequence for U24.** The macromove/class-partition collapse is real
but its sound core is the dead+inert class (L-DRQ), matching the plan's
own "capped by frontier-inertness, honest impact medium/low" note. The
measured dead+inert classes are non-trivial in mid/late-game corpus
positions (5,133 pair-firings / 4,001 nodes), so the collapse is worth
its small implementation cost, but it is a tail optimization, not the
dramatic cut. The dramatic cut at defender nodes remains the dispatch
frontier (leg 1), which the engine already exploits — now with the
adversarial validation and a proof-ready lemma statement it lacked.

---

## 5. Runs, tallies, and regeneration

### 5.1 Pilot scan (COMPLETE; independent agreement check for §5.2)

- Sample: 400 corpus + 1 random defensive SecondStone nodes (pre-filtered),
  seed `1441309534458279` (harness default), scan cap 3,000, confirm cap
  60,000, horizon slack 40, TT 256 MiB per solve.
- Tallies: def_nodes 401; forced_loss 71; P2 firings 0; DRQ pair-firings
  478 (40 adjudicated, 0 mismatches); DBD firings 149 (12 guarded, 0
  refuted); dispatch forced nodes 330, adversarial pairs 1,929, 0
  refutations; coverer-multi nodes 291, pairs 291, 1 mismatch (the §2.2
  counterexample; p2_protected mismatches 0). Wall 102.4 s.
- REGEN:

```
CARGO_TARGET_DIR=.target-hunt TSS_DOM_PER_SOURCE=400 TSS_DOM_RANDOM_POS=3000 \
TSS_DOM_SCAN_CAP=3000 TSS_DOM_CONFIRM_CAP=60000 \
cargo test -p hexfield_eq --release dom_hunt_scan -- --ignored --nocapture --test-threads=1
```

### 5.2 Main sweep (COMPLETE; the report's headline numbers)

- Sample: 4,000 corpus + 1 random defensive SecondStone nodes
  (pre-filtered by the def-node classifier), seed `7766554433221100`,
  scan cap 3,000, **confirm cap 250,000**, horizon slack 40, TT 256 MiB
  per solve. Wall 902.9 s, single process; free RAM never dropped below
  9.8 GB (house-rule monitor active throughout).
- Tallies: def_nodes **4,001**; forced_loss **589** (14.7%); P2 firings
  **2** (never violated); DRQ pair-firings **5,133** (288 adjudicated,
  **0 mismatches**); DBD firings **1,329** (117 guarded, **0 refuted**);
  dispatch forced nodes **3,412** (85.3%), adversarial pairs **20,495**,
  **0 refutations**; coverer-multi nodes **3,013** (88.3% of forced),
  pairs 3,013, **27 mismatches** (all recorded; graded 4 doubly-proven /
  18 strong / 5 weak, §2; 0 P2-protected). Records:
  `dom_hunt_records.jsonl` (89 rows: sampled consistent firings + all
  refutations in the trailing section).
- REGEN:

```
CARGO_TARGET_DIR=.target-hunt TSS_DOM_PER_SOURCE=4000 TSS_DOM_RANDOM_POS=500 \
TSS_DOM_SCAN_CAP=3000 TSS_DOM_CONFIRM_CAP=250000 TSS_DOM_SEED=7766554433221100 \
TSS_DOM_RECORDS=E:/Hexo-BotTrainer-hexgt/.claude/worktrees/hunt-domination/dom_hunt_records.jsonl \
cargo test -p hexfield_eq --release dom_hunt_scan -- --ignored --nocapture --test-threads=1
```

### 5.3 Selftest / verify runners

```
cargo test -p hexfield_eq --release dom_hunt_selftest -- --ignored --nocapture --test-threads=1
# single-position escalating-cap verify: see §2 regen block
```

All runners honor the RAM house rule (poll free RAM, sleep below 8 GB),
run single-process, `--test-threads=1`, `CARGO_TARGET_DIR=.target-hunt`.

---

## 6. Honest limitations

- The referee is WIN-only; "UNKNOWN at cap" is evidence, not proof, on the
  resisting side of a refutation. Counterexamples are therefore graded:
  4 doubly-proven (both verdicts sound), 18 strong (proven WIN vs 250k
  wide-node resistance; canonical ones re-verified to 10^6), and 5 weak
  proof-strength differences that are reported but NOT counted as
  outcome-grade refutations.
- Random-playout defensive SecondStone nodes are rare (~1/3000 playouts
  reach one under the classifier), so source B contributed 1 node of
  4,001; the catalog's frequency claims are corpus-weighted (human play),
  which is the intended deployment distribution for MCTS-leaf integration
  anyway. Directed constructions were replaced by the stronger
  per-position escalating-cap verifier (`dom_hunt_verify`) applied to
  every flagged counterexample.
- Attacker extension dominance (brief class 3) was not hunted; the
  referee direction supports it mechanically (see catalog row 6).
- b=2 (FirstStone spare-stone) domination is UNDECIDED here by referee
  limitation, not by attack failure — flagged for an oracle-refereed round.
