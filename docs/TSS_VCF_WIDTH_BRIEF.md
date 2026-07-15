# TSS VCF-Width Upgrade — Implementation Brief

## Goal

Add an **opt-in, default-off** wide attacker-move universe to the TSS deep
solver (`packages/hexfield_eq/rust/src/tss_solver.rs`) so that it proves
full-width connect-6 VCF wins — specifically, all 14 WIN positions of the
external forcing corpus at `packages/hexfield_eq/rust/corpus/
forcing_corpus_moves.txt`, with **`0l4291i_live` as the single highest-
priority position**.

## Definition of done (hard gate)

1. `cargo test --release -p hexfield_eq tss_corpus_check -- --ignored
   --nocapture` **passes**: every `expect=WIN` entry reaches
   `ProofStatus::Win` somewhere on the node-cap ladder (10k → 100k → 1M →
   20M), and **no `expect=NO` entry ever returns Win** (Loss/Unknown both
   fine). The acceptance test already exists at `rust/src/tss_corpus.rs` and
   currently fails to compile — it calls the API you must build:
   `TssSolver::set_width_options(WidthOptions::vcf_pair_complete())`.
   - Partial-success fallback (only if full 14/14 proves genuinely
     unreachable after real effort): `0l4291i_live` MUST prove Win. But aim
     for 14/14 — the reference pdspn driver solved every one of these, most
     in <2000 of its nodes, worst case 264s.
2. The **entire existing test suite stays green**:
   `cargo test --release -p hexfield_eq` (the ~58 unit tests, not counting
   ignored harnesses). Narrow-mode (default) behavior must be
   **byte-identical** — `WidthOptions::default()` = all-off = today's
   generator, bit-for-bit.
3. No production semantics change: nothing in `search.rs` / `tree.rs` /
   `tss_async.rs` call sites flips the new option on. Rust-level option
   only; **no Python/TOML plumbing in this task**.

## Where the width is lost today (root cause, verified empirically)

`threat_creating_moves` (tss_solver.rs ~line 1031) only emits empties of
claimant-owned windows with `count >= 3`. In connect-6 the attacking unit is
the PAIR (two placements per turn): pair-builds through count-2 windows and
threat+build tempo moves are structurally invisible. Verified on the corpus:
`hayes_20260712_turn16` exhausts its universe after **5 nodes** at every cap;
14/14 corpus WINs return Unknown; several stall at fixed node counts far
below cap (314, 511, 1787, 1868).

## Specification of the wide universe

When `WidthOptions::vcf_pair_complete` is active, at claimant (OR) plies the
candidate set becomes:

- empties of claimant-owned windows with **count >= 2** (was >= 3), at BOTH
  plies of the turn (FirstStone and SecondStone). Rationale: count-2 + the
  turn's two stones = count-4 (immediate threat); and the classic tempo
  pattern (ply 1 = forcing count-3→4 extension, ply 2 = quiet count-2→3
  build for the NEXT turn) requires the count-2 tier on the second ply too.
- everything the narrow generator already emits (win-now, count>=4
  handling, defender-threat blocks) is unchanged and stays first in
  ordering.
- If, after implementing count>=2 width, specific corpus WINs still
  exhaust-without-proof (watch for stall-below-cap in the test output), add
  an escalation tier behind the same option: empties of claimant count>=1
  windows within distance 3 of any stone (the r3 locality bound from the
  zones work: threat-creating moves are provably within dist 3). Escalate
  only on exhaustion, not by default — branching cost.

The DEFENDER (AND-node) side is already provably complete (hitting universe
+ full-legal fallback) — do not change it.

### Ordering (this decides whether 20M nodes is enough)

The count-2 tier multiplies branching; proof-number search survives via
ordering. Suggested priority within the widened candidate set:
1. moves completing/extending count>=4 (immediate),
2. count-3 extensions that create a new count-4 (forcing),
3. count-2 pair-starts ranked by resulting fork degree (number of distinct
   windows through the cell that would reach count>=3), then by proximity
   to existing own stones.
Also strongly consider enabling the already-implemented pair canonicalization
(`tss_pair_commutation` machinery / P3) semantics inside wide mode to dedupe
(a,b)/(b,a) turn transpositions — this roughly halves the pair space.

### API shape

- `WidthOptions` struct (Default = narrow), constructor
  `WidthOptions::vcf_pair_complete()`.
- `TssSolver::set_width_options(&mut self, opts: WidthOptions)` — follow the
  existing `set_zone_options` pattern INCLUDING dropping the persistent
  positive-fragment cache on option change (profile isolation — same
  rationale as the zone options: cached node-cost provenance must not leak
  across profiles).
- Solver internals thread the option to the OR-node generator. Verifier
  (`tss_verify.rs`) note: WIN certificates witness attacker moves
  explicitly, so a wider searched set should not weaken verification — if
  you find any verifier assumption tied to the narrow generator, fix the
  assumption gap explicitly and say so in the commit message rather than
  silently relaxing a check.

## Corpus reference data (what the original solvers did)

| id | expect | ref driver | ref notes |
|---|---|---|---|
| 0hz3hty | WIN | idtt 0.05s | dfpn 6k nodes |
| **0l4291i_live** | **WIN** | **pdspn 264s, 1058 nodes, 733 leaf solves** | idtt+dfpn both failed at 20M — the monster; PRIORITY |
| 8is963b | NO | all agree | trivially dead |
| 94gnnol | NO | pdspn 21s, 108 nodes | idtt+dfpn failed |
| acly7kb | WIN | idtt 7ms | depth 4 |
| dy3dg99 | NO | all agree | trivially dead |
| g2xx6wl | WIN | idtt 0.15s | depth 6 |
| hu01jk4 | WIN | idtt 18ms | depth 6 |
| jh7yo7y | WIN | idtt 0.11s | depth 6 |
| jnzzmcm | WIN | idtt 0.44s | depth 7 |
| l9mxn59 | NO | dfpn 1.4k nodes | |
| lz60mfb | WIN | idtt 1.2s | depth 13 (deepest) |
| mvp2lvc | NO | dfpn 15k nodes | |
| xsnfyll | WIN | idtt 0.7ms | depth 4 (easiest) |
| zrugh2x | WIN | idtt 1.0s | depth 8 |
| strongloss_a_prefix6 | WIN | idtt 31ms | +2 remote defender pad stones (parity fix; pads are >=8 away and inert — a WIN here transfers a fortiori to the unpadded original) |
| strongloss_b_prefix8 | WIN | idtt 9ms | same padding note |
| hayes_20260712_turn16 | WIN | idtt 0.28s | depth 7; currently dies at 5 nodes |
| hayes_20260712_placement31 | WIN | idtt 0.17s | mid-turn (1 placement left) |

"depth" = attacker turns in the reference forcing line. The corpus format is
documented at the top of `rust/src/tss_corpus.rs`.

## Environment / how to build and test

- Cargo >= 1.95 required (lockfile v4). On this machine that means WSL with
  `export PATH="$HOME/.cargo/bin:$PATH"` (system /usr/bin/cargo is 1.75 and
  FAILS). Run:
  `wsl -e bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && cd /mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/tss-vcf-width && CARGO_TARGET_DIR=/tmp/tss-vcf-target cargo test --release -p hexfield_eq tss_corpus_check -- --ignored --nocapture'`
  (plain `cargo` from Windows may also work if a Windows toolchain >= 1.95
  is present — WSL is the proven path).
- ALWAYS set `CARGO_TARGET_DIR=/tmp/tss-vcf-target` — never build into the
  tree.
- The full unit suite: same command with no test filter and without
  `--ignored`.

## Constraints

- Work only in this worktree (`.claude/worktrees/tss-vcf-width`), branch
  `claude/tss-vcf-width`. A LIVE training run executes from a sibling
  worktree — do not touch anything outside this tree.
- Default-off discipline: `WidthOptions::default()` must reproduce today's
  behavior exactly. Production call sites keep defaults.
- Do not modify the acceptance test's assertions or the corpus file to make
  the gate easier (fixing a genuine harness bug is fine — explain it in the
  commit message).
- Commit as you go with clear messages; small commits preferred.

## Out of scope

- Python/TOML exposure of the option, selfplay integration, root-guard
  rungs, zone (AND-side) changes, perf tuning of narrow mode.

---

## ADDENDUM (2026-07-14, after first implementation round)

The first round implemented width options and iterated on `xsnfyll` without
reaching WIN — first exhausting at 28k nodes, then (after widening) burning
full caps (1M nodes / minutes) with heavy TT traffic. Diagnosis from the
session lead; treat these as binding design guidance:

### 1. The blocker is NOT attacker width alone — it is missing FORCING discipline
Analysis of `xsnfyll`: its winning first turn extends TWO count-3 windows
((-1,-1) completes the (1,-1)-direction window holding (0,-2),(1,-3),(2,-4);
(1,-5) completes the (0,-1)-direction window holding (1,-4),(1,-3),(1,-2)) —
both already inside even the NARROW universe. Yet no proof is found at any
cap. The explosion is on the DEFENDER side: an AND node that faces no
immediate threat has no small hitting set and falls back to (near-)full-legal
(~hundreds of moves) — one such node per line is enough to kill the search.

**Required rule for vcf_pair_complete mode: turn-level forcing.** Any
attacker TURN (the pair of plies) that completes without creating at least
one new claimant count>=4 window — and without winning outright — is PRUNED
(the OR node treats it as unavailable; do not expand the defender reply).
Consequences, which are the point:
- every defender node inside the search faces a live threat => replies come
  from the small hitting universe, NEVER the full-legal fallback;
- count-2 stepping stones and quiet tempo builds remain legal as ONE ply of
  a turn whose OTHER ply creates the threat (threat+build), which is the
  pattern the corpus wins need;
- the search space returns to VCF shape (this is what the reference
  solver's "tight" width means), where a depth-4 win costs thousands of
  nodes, not millions.
Implement the discipline at turn granularity (evaluate after the second
ply / at defender entry), not by pre-filtering ply-1 candidates.

### 2. Debug ladder — use the reference lines (new fixture)
`rust/corpus/forcing_corpus_lines.txt` now contains the reference winning
LINE for each of the 14 WIN entries (alternating turns, attacker first, two
placements per turn except possibly the final winning stones; same coord
format as the moves file). Backward-walk protocol for a failing id:
apply the full line prefix of length k onto the corpus position, solve that
state at cap 10k. At k=full the state is at/next-to win-now (should prove in
~1 node). Decrease k turn-by-turn until the solve stops proving WIN — the
first failing k tells you exactly which mechanism is missing (generation at
that turn, defender reply set, leaf evaluation, or turn-forcing pruning).
Add a small ignored helper test for this; keep it in tss_corpus.rs.

### 3. Efficiency rules (binding)
- While ANY position fails, iterate at caps <=100k. Do NOT raise caps to
  chase a proof — when the discipline is right, xsnfyll-class positions
  prove in <=10k nodes (the reference pdspn proved most of this corpus in
  <2000 of its nodes). Cap-burning at depth 4 means the design is wrong,
  not the budget.
- Commit working checkpoints as you go (the previous round produced zero
  commits over ~3 hours — do not repeat that; commit after each coherent
  step even if the gate is not yet green).
- Only after ALL other WIN entries prove should you spend big caps on
  0l4291i_live (the monster; reference needed 264s).

### 4. Soundness reminder
The turn-forcing prune is a WIN-search restriction (it can only cause missed
wins, never false WINs) — soundness of WIN certificates is unchanged. Keep
the NO entries' requirement (never WIN) as-is; UNKNOWN via forcing-universe
exhaustion is the expected NO behavior and is cheap under the discipline.

---

## ADDENDUM 2 (2026-07-14, mid round 2 — new ground truth + optimization directives)

### 1. PROVEN: every corpus WIN is reachable under turn-forcing (checked externally)
The session lead replayed all 14 reference lines from
`rust/corpus/forcing_corpus_lines.txt` against their corpus positions with an
independent window model (3 axes, 6-windows). Result: **every attacker turn
of every line — including all 9 turns of `0l4291i_live` — either wins
outright or creates a new defender-free count>=4 window.** All 14 lines end
in a verified 6-in-a-row; zero parity/legality flags.

Consequences, binding:
- The turn-forcing prune is NEVER the reason an entry fails. If an entry
  exhausts without proof, the gap is (a) a missing width tier (some line
  placement not in the candidate set at that ply) or (b) ordering/caps.
  Backward-walk the reference line (ADDENDUM §2 protocol) to find which.
- `0l4291i_live` is pure-VCF. Do not design a quiet-move/strategy tier for
  it — it needs only width + ordering + budget under the forcing discipline.

### 1b. Width tier is SETTLED — count>=2 is sufficient, do NOT build escalation
Externally measured (same checker): all 181 attacker placements across the
14 reference lines sit inside the count>=2 defender-free-window universe at
the moment they are played — 128 are tier-0 (count>=3, the narrow universe),
53 need the count>=2 tier, **zero need count>=1/r3 escalation, zero need
full width**. Binding consequences:
- Do NOT implement the count>=1/dist-3 escalation tier from the original
  brief — it is provably unnecessary for this gate and the owner's
  single-path directive forbids dead machinery. Strike it.
- Any entry that exhausts without proof under count>=2 + turn-forcing has an
  ORDERING or BUDGET problem, never a width problem. Stop widening; fix
  selection.
- Ordering prior: tier-0 candidates (count>=3 forcing extensions) before
  tier-1 (count>=2 builds) — 71% of reference placements are tier-0.

### 2. Optimization directives (owner-approved, apply inside wide mode)
- **Kernel restriction at forced defender nodes**: when the minimum hitting
  set size equals the defender budget b, restrict defender replies to the
  extendable-hit kernel (cells whose block keeps a live defense possible).
  Under turn-forcing every defender node is forced, so this applies at every
  AND node. Sound (zones paper T6; scope mhs<=b only).
- **Sparse LOSS/exhaustion witnesses**: when concluding a branch is dead,
  witness with <=3 windows (defender budget 1) / <=5 (budget 2) rather than
  full enumeration (paper L13, round-7 tightened caps 3/5).
- **Fork-degree ordering** stays the top ordering signal for count-2
  pair-starts (measured ~100x on refutations in the zone experiments).
- **tau-informed pn/dn initialization** (proof-derived, soundness-free —
  pn/dn only steer expansion): initialize defender-node disproof numbers
  with the hitting-set size tau (the number of replies that must actually
  be refuted; the mhs machinery already computes it), and attacker-node
  proof numbers from fork degree, instead of uniform 1/1 or hand-tuned rank
  biases. This is the principled fix for selection-cost blowups on deep
  cases (hayes/lz60mfb class).
- **U9 ES-potential futility is STRUCK — do not implement.** The round-7
  proof campaign REFUTED the all-ties greedy ES argument (the raw
  existential claim remains open), so the futility cut has no proven basis.
  If any U9 scaffolding exists in the WIP, delete it.
- **Consolidate selection machinery under the principled form.** Any ad-hoc
  selection state accumulated during the corpus fight (scouting phases,
  commitment thresholds, bespoke tie-breakers) should be replaced by, or
  re-validated against, tau-informed pn/dn initialization + fork-degree
  ordering + kernel K_b. Keep a bespoke rule only if it beats the principled
  form on the full corpus in a direct A/B; otherwise delete it. The
  horizon-advance-on-top-ranked-cutoff staging is a keeper. After every
  engine change, re-verify ALL previously closed entries at their caps
  (no silent regressions).
- Pair canonicalization via ply-level PN nodes + TT meet (if that is the
  current representation) is an accepted implementation of P3 — keep the
  forcing gate exactly at turn completion (after the second stone).
- **Keep `WidthOptions` minimal**: one constructor, no sub-flag
  proliferation. Owner directive: after validation this design gets
  CONSOLIDATED into the single mainline TSS path (flags are build
  scaffolding, not final architecture) — do not add structural mode
  switches beyond narrow/vcf_pair_complete.

### 3. Out of scope (owner directive)
Zone-based defender sets at unforced nodes (ranked zone T4 / quiet-turn
allowance) are DEFERRED — proof work is still in flight. Do not implement
any AND-node dismissal machinery in this task.

---

## ADDENDUM 3 (2026-07-14, round-3 handoff — round-2 state + binding marching orders)

### A. Round-2 status (verified from its logs; its WIP is UNCOMMITTED in this tree)
- 12/14 WIN entries prove at <=100k under the round-2 engine, including
  hayes_turn16 (83,421 nodes) and hayes_20260712_placement31 (94,115).
  Remaining: lz60mfb and 0l4291i_live only.
- FIRST ACTIONS, before any new engine work: (1) cargo build; run the
  forcing regression test plus 2-3 solved-entry spot checks; repair anything
  left half-edited; (2) `git commit` the round-2 WIP as a checkpoint
  immediately. Commit working checkpoints frequently from then on (round 2
  committed nothing in 4.5 hours — do not repeat that).
- Round-2 engine capital to KEEP (all in the WIP): staged deepening that
  advances the horizon on a top-ranked-completion cutoff; defender-risk
  priority at partial-turn roots; urgent-block sequential treatment at pair
  roots; the width-sorter fix (count-2 second-ply moves completing a tight
  forcing turn join the top tier); compact varint exact TT keys + full
  512MiB wide budget + parentless deepest-first stage refresh (reaches a
  clean 1M cap).
- Known bugs round 2 identified but did NOT fix — fix these early:
  (a) table-full stall at ~92k nodes (search stalls instead of failing over
  when the table fills); (b) zero-cap semantics; (c) an unaffordable
  attacker child repeatedly stalling the frontier.

### B. Failure localizations (do not re-derive these)
- lz60mfb: everything from prefix 4 inward proves <=100k (prefix 4 =
  92,007). The root blocker is the FIRST defender universal: 4 hitting
  cells -> 2 nonterminal unordered reply pairs; each proves individually at
  ~78-92k; together they exhaust 100k. The gate ladder allows 1M — FIRST
  check whether lz already proves at the 1M rung with the PN frontier and
  bank that, THEN optimize toward 100k. Also verify the two replies share
  transposed continuations in the TT (the second should cost far less than
  solo if sharing works).
- 0l4291i_live: first failing checkpoint is prefix 12; at a clean 1M cap
  (memory verified not the limiter) it is still UNKNOWN at depth 34. This
  is a search-shape problem and it is the gate's priority entry.

### C. Marching orders (binding order)
1. Stabilize + commit (section A).
2. Implement ADDENDUM 2 section-2's principled toolkit BEFORE building any
   more bespoke ordering machinery, in this order: tau-informed pn/dn
   initialization; kernel restriction K_b at forced defender nodes (this
   should directly shrink the lz defender conjunction — and every AND
   node); L13 sparse witnesses (3/5 caps); fork-degree as the top OR-side
   ordering signal. U9 is STRUCK — delete any scaffolding if present.
3. Bespoke-heuristic audit: each round-2 special case (scouting phases,
   commitment thresholds, tie-breakers) must beat or complement the
   principled form in a full-corpus A/B, or be deleted. One system, few
   flags — WidthOptions stays minimal.
4. After EVERY engine change: re-verify all 12 closed entries at their
   proven caps. No silent regressions.
5. Then close lz60mfb and 0l4291i_live.
6. Gate unchanged: all 14 expect=WIN prove on the ladder, zero WIN on the
   5 expect=NO entries, full suite green, narrow/default mode
   byte-identical.
