# STRIX (hexo-solver) vs TSS ENGINE — comparison + attack plan

Date: 2026-07-21. Self-contained (written ahead of a context clear).
Companion: `docs/INVESTIGATION_PDSPN_IMPORTS.md` (the four pdspn import
candidates, owner-dispositioned, Codex-lane-ready).
Battery raws: `%TEMP%/hexo-strix-clone/strix_battery.csv`,
`/tmp/our_corpus_gate18.log` (regen recipes in the investigation doc).

## 1. What Tyto's solver is

`hexo-solver` crate in github.com/SootyOwl/hexo-strix (`hexo-rs/hexo-solver`;
NOT shipped in our vendored `hexo_rs` wheel, which is why it was invisible
here until 2026-07-21):

- **`forcing.rs` (~3.2k lines)** — production VCF solver: iterative
  deepening + persistent TT, "fully-forcing" (every attacker turn creates
  threats consuming the defender's whole turn). Outcomes Win{depth}/No/
  BudgetExceeded. `solve_wide` = experimental partner-cell widening
  (strict superset, ~1.5–1.7x cost).
- **`prover/`** — research CLI drivers over a shared kernel: `dfpn`
  (Nagai, 1+ε thresholds ε=0.25, 2-way set-associative TT with per-entry
  `work`), `pdspn` (Winands PDS-PN: df-pn level 1 + bounded 50k-node PN
  probe seeding each new frontier node, probe tree discarded), `pn`,
  `hybrid`, `race` portfolio. These ARE the "idtt/dfpn/pdspn" reference
  columns of our campaign; our forcing corpus descends from his race
  corpus (tss_corpus.rs header).
- **Integration = root-only tripwire.** Both his MCTS paths run
  `solve_wide` (depth cap 6 attacker turns, 2,000 nodes) at the ROOT
  before search; a Win skips the NN search and writes a one-hot/two-hot
  policy target. No leaf solves, no backups, no loss side, no in-tree
  consumption. (Note: his proven-root one-hot targets are the design our
  owner rejected — a cert designates ONE arbitrary winning move.)

## 2. Structural similarity (be honest with him and ourselves)

His kernel is built from the same forcedness insight as ours: hot cells
from ≥2-stone windows (≈ our C≥2), pair partners include defender-block
cells (block-and-threaten tempo turns), pair filter = post-move cover
number B ≥ 2 ("pin the defender's whole turn" ≈ our mhs==2), defender
node = all minimum covers at B==2 + defender-wins-first check + B≥3 ⇒
win. **Both engines search essentially the same forcing tree.** The
~2,000x branching reduction vs naive (~20k legal pair-turns → ~10 forcing
pairs) is the shared VCF idea, competently implemented on both sides.

## 3. Real differences

Ours over his:

1. **Proven width.** Our attacker universe is an exact characterization
   (S(P,a) = (T(P)\{a}) ∪ G₁(P,a); post-apply identity S_exact) with a
   written proven-vs-asserted ledger and concrete counterexample witnesses
   marking the completeness boundary (3 atlas wins the forcing class
   provably cannot express). His generator is heuristic ("experimental",
   no completeness argument). Our width is also wider (count-2 seeding +
   G₁ promotion + tempo blocks run as the default, proven sound).
2. **Proof-carrying + verified.** Every WIN/LOSS emits a replayable
   certificate; an independent frozen verifier replays each before ANY
   consumption incl. cache hits (vf=0 across tens of millions of solves);
   Lean kernel-checks "verifier accepts ⇒ game-theoretic attacker win"
   for covered cert classes + byte-level codec completeness. He has PV
   strings and honesty conventions (BudgetExceeded-never-fabricate,
   support sets) — good, but advisory, no verification boundary.
3. **Structural cache identity.** Our TT compares full canonical
   positions; his is keyed by 64-bit Zobrist alone — his own code comment
   records the historical collision bug that "silently poisoned every
   hash-keyed table, returning verdicts computed for unrelated positions"
   (now 2⁻⁶⁴-improbable; ours is impossible-by-construction).
4. **LOSS side.** Dual certificates (opponent-win exhausting OUR legal
   moves) + leftover-budget dual pass. He has no loss concept at all.
5. **Determinism.** Pure function of (state, caps); no clocks, no hash
   iteration order; golden-digest + 6,443-position identity battery
   discipline. His idtt samples wall clocks in research mode.
6. **Deployment depth.** Ours: every leaf + every root + interior guard,
   ~3M solves/epoch, verified hard backups, policy sharpening, loss
   stream (main_4). His: root tripwire that decided 5/19 of his own
   corpus at his deployed config.

His over ours (measured today, matched host):

1. **idtt easy-win latency**: 2–5x faster on sub-second wins (our wall
   includes fresh-per-rung ladder tax ~30% + in-wall cert emission +
   verification; but it's real).
2. **pdspn memory-constrained deep proofs**: 0l4291i WIN at 256 MB where
   our 512 MiB profile saturated (our official 2 GiB gate rerun on a
   quiet host is PENDING — the last unsettled row).
3. **Fast disproofs**: his No on 94gnnol (25 s) / mvp2lvc (1.5 s) vs our
   Unknown at 1M nodes — the width trade-off: our wider win-complete
   universe makes disproof-by-exhaustion proportionally harder.
4. **Two good tricks we lack**: PN² probe-seeding (worth ~4 orders of
   magnitude on 0l: dfpn 50M blind nodes FAIL vs pdspn 1,058 seeded
   nodes WIN) and 1+ε thresholds. Both written up as import candidates.

## 4. First direct matched-host battery (2026-07-21; both sides under live
   trainer load; walls load-inflated ~equally; 15/15 shared-id positions
   verified stone-identical)

WIN rows (13 available; 0l excluded pending 2 GiB):

| Position | Ours (verified cert) | idtt | dfpn | pdspn |
|---|---|---|---|---|
| 0hz3hty | 0.24s | 0.06s | 0.18s | 1.27s |
| acly7kb | 0.02s | 0.01s | 0.004s | 0.004s |
| g2xx6wl | 0.98s | 0.31s | 0.75s | 0.54s |
| hu01jk4 | 0.17s | 0.03s | 0.06s | 0.06s |
| jh7yo7y | 0.39s | 0.07s | 0.04s | 0.04s |
| jnzzmcm | 1.39s | 0.49s | 1.23s | 0.47s |
| xsnfyll | 0.005s | 0.001s | 0.001s | 0.005s |
| strongloss_b | 0.12s | 0.01s | 0.09s | 0.006s |
| strongloss_a | Σ3.0s | 0.05s | 0.34s | 0.01s |
| hayes_turn16 | Σ3.6s | 0.26s | 2.28s | 0.48s |
| hayes_pl31 | Σ3.7s | 0.26s | 2.53s | 0.66s |
| zrugh2x | Σ8.5s | 1.64s | 2.55s | 5.56s |
| lz60mfb | Σ30.3s | 2.45s | FAIL | 16.98s |

NO rows (5): 8is963b + dy3dg99 = **our LOSS proven ~0ms** (claim his
engines cannot express) vs his No; l9mxn59 our width-exhaust 25ms ≈ his
No; 94gnnol our Unknown vs pdspn No 25s (idtt/dfpn FAIL); mvp2lvc our
Unknown vs dfpn/pdspn No ~1.5s (idtt FAIL).

His deployed self-play config (wide/depth6/2k, root-only): **5/19**.
0l4291i: pdspn Win 305s @256MB (his crown until our 2 GiB quiet rerun).
Our gate: PASS, 13/13 wins certified, zero false claims, 269s total.

**Verdict:** as a deployed system — categorical advantage, ours. As a
single-position engine — peer fight: we win coverage + claim strength +
verification; idtt wins easy-win latency; pdspn wins memory-constrained
deep proofs and wide-class disproofs.

## 5. Attack plan (pre-registered suggestions for the solver-improvement
   workflow; Codex fresh usage)

### 5a. Harness reliability / accuracy-to-prod (do FIRST — cheap, compounds)

- **H1. Quiet-host reference rerun + archive.** Host is now quiet (main_4
  stopped after ep25 eval, owner-ordered). Rerun the full both-sides
  battery + the 0l official 2 GiB gate (`TSS_BACKWALK_TT_BYTES=
  2147483648`, needs ≥10 GB free — now available) and archive as the
  pinned reference row (their pdspn 0l should also be rerun quiet for
  fairness; historical quiet mark 264s).
- **H2. Cap-resume promotion.** The ~30% fresh-per-rung ladder tax is a
  methodology artifact vs their single-call drivers. Promote
  `CapResumeSession` (cfg(test), built, gated −29.10% official wall) into
  the official ladder, or report cumulative-vs-per-rung separately.
  Biggest single "make our walls honest" item.
- **H3. Disproof metrics.** The harness scores WIN/LOSS coverage but has
  no refutation-completeness metric — today's 94gnnol/mvp2lvc gap was
  invisible to our own instruments. Add: time-to-width-exhaustion,
  disproof coverage on NO-labeled sets, and certified-refutation status
  once R2 below exists.
- **H4. Contended-bench tier.** Live-load behavior (bail 76–85% at 300ms
  park) diverged badly from quiet bench (0%). We measured 12-way
  contention = 1.1–1.7x solve-wall inflation; add a bench tier that
  replicates production contention so park/queue decisions stop needing
  live epochs.
- **H5. Parallel batch sweep** (rayon, per-thread solvers) — queued cargo
  item; batch-order dependence already MEASURED ZERO on verdicts.

### 5b. Codex-ultra exploration — search-space narrowing with real teeth

Ranked by expected advantage over his engine:

- **U1. PN² probe-seeding + 1+ε + TT-replacement audit + Unknown
  summaries** — see INVESTIGATION_PDSPN_IMPORTS.md (owner-dispositioned;
  items 1–2 deep-solve-scoped, 3–4 investigate). The 4-OOM 0l datum is
  the prize marker. CAVEAT on 1+ε: read the R-TS1 "threshold null" round
  first; it may close by reading.
- **U2. Certified refutations (NEW capability, nobody has this).** Our
  disproofs (dn=0 = wide universe exhausted) currently produce NO
  artifact — "Unknown" to the trainer, nothing checkable to anyone else.
  Design a refutation certificate: AND over the exact attacker universe
  (the S_exact characterization IS the coverage argument) with per-branch
  refutations; verifier extension; Lean model on top. Turns his fastest
  axis (cheap "No") into our strongest (checkable "No"). Feeds the atlas
  (certified non-wins), the NO-side of the corpus, and kills the l9mxn59
  ambiguity class. This is the disproof-side twin of what we already did
  for wins.
- **U3. B1 removal candidates** (R-B1-EXACT's ranked list): stale
  defender-block-only second cells (clearest proof path — exact set
  identity already stated), weak-only G₁ (0.0245% usage, needs the
  pair-normalization lemma, counterexamples expected), G≥2 as
  ordering-view (generation savings). Proof-backed branching reduction he
  cannot match without a completeness argument.
- **U4. B2 tempo-budget pruning from k\*=3 EXACT** — the proven
  non-forcing tempo budget licenses sound attacker-line cuts. Never
  implemented; entirely proof-capital we already own.
- **U5. A4 reply equivalence/dominance** (substitution envelopes,
  domination patterns): refute one representative per interchangeable
  defender class — attacks the 40% unforced-defender-generation residue
  AND speeds disproofs (helps the 94gnnol class). Rides the same cert
  grammar as U2/G2 — design them into ONE contract extension, not three.
- **U6. Production-wall P7 sequels** (the actual production levers):
  second_candidates HashSet churn ~8%, first-candidate enumeration ~15%,
  D_FORCED_GEN ~20%, P6 cross-node generation memoization. Bit-identity
  gates as with P7 (`2c262e10`, 1.42x).
- **U7. Race portfolio for offline labeling** (steal his `race` shape):
  goal=win / goal=loss / narrow-disproof arms on threads, first
  definitive verdict wins — cheap wall win for the atlas/labeling path,
  zero soundness surface.

### 5c. Leveraging the Lean framework (the moat he cannot cross)

- **L1. Close the Rust correspondence** (CP-O14/O15/O19/O27): extend
  accept⇒won to the FH/T6 cert classes and tie it to the literal
  executed verifier bytes. End state: "our verifier's acceptance is a
  kernel-checked win theorem about the exact production binary" — no
  other game engine has this, and it converts every future verifier
  extension (U2/U5/G2) into checked territory.
- **L2. Formalize the attacker-universe exactness** (S_exact identity +
  the P3 pair-quotient, both recorded as unformalized): the width
  completeness claim becomes kernel truth, and U3's removals get proven
  against it instead of property-tested.
- **L3. Lean-first pruning doctrine** (already proven to work:
  DEADLINE_ES → census blocking → solver consumption unblocked): for
  every U2–U5 reduction, prove the licensing theorem in Lean BEFORE the
  solver consumes it. New pruning ships pre-proven, hostile review
  becomes confirmation rather than gatekeeping.
- **L4. FHW-T3-R core formalization** (C1) — owner-launched session per
  standing ruling; unblocks zone-family consumption claims if zones ever
  find an economic cohort.
- **L5. Certified-refutation semantics (with U2):** the Lean game model
  already states AttackerWinsBy/NoContractWin — a refutation cert's
  soundness statement ("verifier accepts refutation ⇒ no forcing-class
  win exists") is the natural next theorem family and makes U2
  publishable-grade.

### Sequencing suggestion

H1+H2 (one session, mostly mechanical) → U1 items 2,1 (the pdspn imports,
deep-scoped) → U2+U5 designed together as one cert-grammar extension
(ultra design round + hostile review) → U3/U4 proofs alongside → L1/L2 as
the parallel Lean track. U6 whenever a production-wall session is wanted;
U7 opportunistic.

## 6. Explaining it to Tyto (short form)

"Same search family as your dfpn — AND/OR proof-number search over the
forcing tree with threshold-controlled depth-first descent. Three
verdicts kept strictly apart: proven WIN (certificate), proven refutation
of the forcing class (disproof side completes), and Unknown (budget died
— never converted to a claim). LOSS is a fourth thing: a second solve of
the same position with the claimant flipped, proving YOUR forced win
against every one of MY legal moves, funded by the win pass's leftover
budget. What's different from yours: the attacker universe is an exact
proven characterization rather than a generator (wider, with the
completeness boundary itself proven via counterexample witnesses); every
verdict is a replayable certificate checked by an independent verifier
before anything consumes it; and the Lean work kernel-checks 'verifier
accepts ⇒ game-theoretic win' for the covered classes — because it runs
at every MCTS leaf inside a training loop (~3M solves/epoch), where one
false verdict silently poisons training. Your pdspn's frontier seeding
beat me on the hard disproofs at matched memory, and I'm stealing it."
