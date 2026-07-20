# R-CLOSURE-1 impossibility ledger

Audit date: 2026-07-20. Repository baseline: `claude/tss-vcf-width` at
`6ee9ecfb`. This ledger records only routes for which the cited artifact states
and supports a negative result. A route is not enlarged beyond its stated
quantifiers. `DEAD` means the named route must not be proposed again;
`REPAIRED-BY` means the old rule is superseded but a sound replacement exists.

For Git-only paper artifacts, the pointer is deliberately executable: from the
Hexo repository, run `git show <commit>:<path>`. References into `E:\tss-lean`
are to that read-only proof checkout.

## Ledger

### IL-01 — blanket per-node radius-two defender restriction

- **Status / exact dead statement:** **DEAD.** It is unsound to replace the
  legal defender set at every non-forced (`k < B`) node by cells within hex
  distance two of existing stones. The death is for the blanket restriction,
  not for radius two as a complete set for a single-window purpose.
- **Killing counterexamples:** G1 has a distance-three junction/pre-block that
  is the saving multi-window move; the capped B2-v2 root exhausts all 217
  radius-two defender turns and returns a false restricted-search WIN in 2,374
  uncapped nodes. G3 has a distance-three defender counterfork whose placement
  creates three disjoint count-four demands. Follow
  `git show 6dc08d7a:docs/PLAN_TSS_MOVESET_ZONES.md`, **Verdict in one line**
  (lines 23–34), **§7.1 G1**, **§7.3 G3**, and **§9 Experiments** (especially
  lines 382–405). The normative proof's **§9** records that the same harness
  produces false WINs for the old radius-two controls
  (`consolidate-main/docs/PROOF_TSS_DEFENDER_ZONES.md:1658–1669`).
- **Open boundary:** Radius two remains sufficient for the document's exact
  single-window purposes. Certificate-relative D21/FHW zones, radius three
  plus active-window terms, domination/equivalence classes, and other
  multi-window-aware restrictions are not killed.
- **Machine check / regression:** The source experiment was machine-checked
  against `hexo_engine`. The checked-in Rust test
  `tss_solver::tests::zone_adversary_geometry_scaffolds_match_python_reference`
  (`tss_solver.rs:10636–10745`) pins G1/G3 geometry only; it is not the full
  false-WIN differential. The full matched-horizon G1/G3 differential remains
  the required regression named by
  `consolidate-main/docs/PLAN_TSS_SOLVER_UPGRADES.md`, **U10** (`:514–541`). It
  should be promoted as
  a permanent ignored/release gate before any new defender trim ships.

### IL-02 — position-independent pairing defense for Hexo length six

- **Status / exact dead statement:** **DEAD.** No partial matching of lattice
  cells can cover every length-six Hexo window by placing a complete matched
  pair inside that window. Therefore the hoped-for global, position-independent
  reply-by-mate collapse is unavailable for the actual `k=6`, three-axis game.
- **Killing theorem:** T8's density contradiction at radius 20 forces more
  matched-cell incidences than a matching can supply. Follow
  `consolidate-main/docs/PROOF_TSS_DEFENDER_ZONES.md`, **§8, T8**
  (`:1511–1536`). The boundary is exact: **T8.2** (`:1550–1627`) constructs a
  periodic perfect matching for length seven with period-lattice index 12 and
  proves that index minimal among periodic constructions. Thus the `k=7`
  pairing **exists**; it does not rescue length six.
- **Open boundary:** Partial pairings on constrained/occupied regions,
  position-dependent or dynamic pairings, and non-matching domination schemes
  remain open. The proof explicitly says constrained-region partial pairings
  are not ruled out (`:1628–1630`).
- **Machine check / regression:** Kernel-checked. See
  `E:\tss-lean\TssZones\Pairing\NoSix.lean:637` (`T8_noPairing`) and
  `E:\tss-lean\TssZones\Pairing\Construction.lean:308,361`
  (`thresholdMatching_exactlyOne`, `T8_2_exists`). The claim/proof map is
  `E:\tss-lean\LEDGER.md`, rows **T8** and **T8.2**.

### IL-03 — unrestricted dynamic touched-window ES greedy defense

- **Status / exact dead statement:** **DEAD.** The policy “at each defender
  placement choose any maximum-danger cell for the current attacker-touched
  alive-window potential at `lambda=sqrt(3)`” is not a sufficient global
  defense from every `Phi < 1` position, for any tie-breaking rule.
- **Killing counterexample:** From `A={(0,0)}`, `D={(1,0)}`, defender at
  `FirstStone`, `Phi=13sqrt(3)/27<1`, the fixed attacker continuation in
  Theorem 1 wins on every exact greedy branch. Follow
  `E:\tss-lean\sources\ES_GLOBAL_BOUNDARY.md`, **§1** and **§2, Theorem 1
  (all-ties greedy refutation)** (`:25–140`). The consolidated statement and
  its boundaries are in `consolidate-main/docs/PROOF_TSS_DEFENDER_ZONES.md`,
  **§10, Strengthened global boundary** (`:1720–1740`).
- **Open boundary:** The raw existential claim—some non-greedy forever-blocking
  strategy from every such position—remains open. Fixed-family greedy under
  its theorem hypotheses and the finite-horizon ES certificates are not
  refuted.
- **Machine check / regression:** The proof source describes an exact-surd,
  exhaustive, no-cutoff checker `scripts/_es_global_check.py`
  (`ES_GLOBAL_BOUNDARY.md:49–54`), but that script is absent from the audited
  `E:\tss-lean` checkout. `E:\tss-lean\LEDGER.md`, rows **ES Global dynamic
  boundary** and **ES Global Theorem 1**, records the intended declaration
  names but the declarations are likewise not present in the current
  `TssZones/Potential` files. Therefore the finite counterexample is
  machine-checkable in design, but there is no runnable regression at this
  checkout; restore the exact-surd checker or land the stated Lean theorem
  before calling it continuously machine-checked.

### IL-04 — radius-nine proof by constant substitution

- **Status / exact dead statement:** **DEAD AS PROOF METHOD.** Replacing each
  radius-eight constant in the production zone derivation by nine and seeing
  the same formulas or favorable telemetry is not an independent proof of
  radius-nine robustness.
- **Killing argument:** The substituted formulas reuse the theorem whose
  legality/frontier assumptions changed; telemetry cannot establish universal
  branch coverage. Follow `DESIGN_GROUP2_NEXT.md`, **R-Z11 repair record**
  (`:28–29`) and **§6.3** (`:782–819`). The independent review confirms the
  disposition in `PROOF_TSS_ZONES_FHW_REVIEW2.md`, **§5** (`:630–633`).
- **Open boundary:** A fixed-horizon exhaustive `Legal_9` policy model check is
  explicitly open and specified; native radius-eight correctness and ordinary
  stress telemetry are unaffected.
- **Machine check / regression:** Not applicable to the invalid inference
  itself. The replacement should be the future exact ignored test
  `tss_group2_next::radius9_exhaustive_gate`, specified in
  `DESIGN_GROUP2_NEXT.md`, **§6.7** (`:1019–1076`); no such implementation is
  present in this worktree.

### IL-05 — H1152 population-prevalence inference

- **Status / exact dead statement:** **DEAD.** The deterministic H1152/H1152-B
  benchmark cannot estimate human-play population prevalence, and its rows
  cannot be used for a population-weighted aggregate. A PASS or a large
  accepted-node count does not repair the sampling design.
- **Killing argument:** H1152 is a fixed-key regression/materiality cohort, not
  a probability sample. Follow `DESIGN_GROUP2_NEXT.md`, **R-Z11 repair record**
  (`:26`), **§6.1** (the H1152-B cohort and prohibition), and **§7–§8**
  (`:1123`, `:1179–1182`). The hostile audit confirms this exact reading in
  `PROOF_TSS_ZONES_FHW_REVIEW2.md`, **§5** (`:630–631`).
- **Open boundary:** Fixed-key materiality distributions, regression identity,
  and a separately designed representative/random population sample remain
  valid or open. This death does not say FHW eligibility is rare or common.
- **Machine check / regression:** Statistical applicability is not a code
  theorem. The future manifest/harness must label H1152-B as fixed-key and
  reject any `population_prevalence` output field. Cohort membership and
  manifest identity are machine-checkable under `DESIGN_GROUP2_NEXT.md`,
  **§6.7** (`:1030–1066`).

### IL-06 — fast stealing conversion by importing the same-history S59 reserve

- **Status / exact dead statement:** **DEAD AT S65 SCOPE.** The named S59
  prepaid reserve cannot be spliced into the genuine S15/S49 fast history to
  turn the terminal misalignment into a real-F win.
- **Killing counterexample/theorem:** The exact query census has no `RES_1` at
  any fast checkpoint; importing prepayment changes the physical history.
  Follow `git show 88bca52d:STRATEGY_STEALING_ROUND9.md`, **§75.1–§75.3,
  S64/S64.1/S65**, and **§84.1**; independently follow
  `git show 88bca52d:STRATEGY_STEALING_REVIEW_ROUND9.md`, **Findings 2–4** and
  **Named fast-conversion obstruction verdicts**.
- **Open boundary:** Reserve strategies on other histories, asynchronous or
  outer carriers, recoding, the fast outcome, and `NL_F` remain open.
- **Machine check / regression:** Hand/exact-table proof; the review states an
  independent recomputation, not a program run. No regression test is present.
  A future checker should pin the S64 count/minimum table and reject a
  same-history S59 admission at all four `q_j` checkpoints.

### IL-07 — fast stealing conversion by two ordinary appends

- **Status / exact dead statement:** **DEAD AT S65 SCOPE.** The two ordinary
  real placements available in the tested terminal approach cannot supply the
  missing sixth real F stone before the terminal engine stops.
- **Killing theorem:** S65's physical count/cadence calculation. Follow
  `git show 88bca52d:STRATEGY_STEALING_ROUND9.md`, **§75.3–§75.5, S65**,
  **§78.3–§78.4**, and **§84.1**; then
  `git show 88bca52d:STRATEGY_STEALING_REVIEW_ROUND9.md`, **Finding 3** and the
  **Named fast-conversion obstruction verdicts** table.
- **Open boundary:** A non-one-for-one asynchronous/outer carrier is not
  covered; neither the fast outcome nor a general stealing theorem is decided.
- **Machine check / regression:** No runnable regression is recorded. A future
  finite trace checker should assert the real-F count after each of the two
  legal appends and the engine's immediate terminal stop.

### IL-08 — fast stealing conversion by section-53 paired-final-event closure

- **Status / exact dead statement:** **DEAD AT S65 SCOPE.** Treating the paired
  final event atomically pairs existing appends but does not create another
  physical stone and cannot continue a terminal engine, so it cannot repair
  S49.
- **Killing theorem:** Follow
  `git show 88bca52d:STRATEGY_STEALING_ROUND9.md`, **§75.3, S65**, **§81.1
  item 29**, and **§84.1**; then
  `git show 88bca52d:STRATEGY_STEALING_REVIEW_ROUND9.md`, **Finding 3** and
  **Named fast-conversion obstruction verdicts**.
- **Open boundary:** Other closure semantics backed by a genuinely different
  physical carrier are not ruled out. Fast outcome and `NL_F` remain open.
- **Machine check / regression:** No current test. A trace-level regression
  should make closure a zero-stone accounting event and forbid post-terminal
  continuation.

### IL-09 — fast stealing conversion by terminal-moment S63

- **Status / exact dead statement:** **DEAD AT S65 SCOPE.** S63 cannot be
  invoked solely at the reached F-role terminal microstep: it requires a
  common-live, mirror-clean, first-unsafe S-role event, which the trace does
  not provide.
- **Killing theorem:** Follow
  `git show 88bca52d:STRATEGY_STEALING_ROUND9.md`, **§75.3, S65**,
  **§78.2–§78.4**, and **§84.1**; then
  `git show 88bca52d:STRATEGY_STEALING_REVIEW_ROUND9.md`, **Finding 4** and the
  **Named fast-conversion obstruction verdicts** table.
- **Open boundary:** S63 remains valid on its exact S-role premise. Arbitrary
  lag, outer carriers, and other terminal-fidelity arguments remain open.
- **Machine check / regression:** No current test. A future role/phase trace
  checker should reject S63 when either common-liveness, mirror cleanliness,
  first-unsafe status, or S-role phase is absent.

### IL-10 — safe tempo intervention at `P_3^pl`

- **Status / exact dead statement:** **DEAD.** There is no legal defender action
  at the named `P_3^pl` plateau with returned one-successor risk below three;
  every action is unsafe. The earlier claim that the cap has exact risk two at
  `P_3` and hence `k*=4` is refuted.
- **Killing counterexample/theorem:** The legal stock-assisted response
  `b*=((9,-2),(10,-2))` refutes the proposed cap; the review's `H_3`-miss /
  untouched-fan dichotomy extends this to every action. Follow
  `git show f5349d3e:GAP_RAW_REVIEW_ROUND9.md`, **Finding 1** and **Per-theorem
  verdicts**, then `git show f5349d3e:GAP_RAW_PROOF_ROUND9.md`, **§86.1** and
  **§83**.
- **Open boundary:** Exact `B_1(P_3^pl)` above the proved lower bound, Q2
  strategy-independent reachability, and a perpetual earlier policy remain
  open. The local stop does not prove every root strategy reaches `P_3`.
- **Machine check / regression:** The paper/review explicitly records no
  program or machine enumeration in this round. A finite exact-board test for
  `b*` plus an exhaustive `H_3` dichotomy checker should be added if this
  result is consumed by pruning.

### IL-11 — safe tempo intervention at `P_4^pl`

- **Status / exact dead statement:** **DEAD.** Every legal defender action at
  the named `P_4^pl` plateau has a legal nonterminal response returning demand
  at least three; the cap has exact risk three.
- **Killing theorem:** Follow
  `git show f5349d3e:GAP_RAW_PROOF_ROUND9.md`, **§78, R9.1 and L15.1.1**,
  plus **§86.2/§86.4**. Then follow
  `git show f5349d3e:GAP_RAW_REVIEW_ROUND9.md`, **Finding 4** and its
  per-theorem table.
- **Open boundary:** Q2 forcing of arrival and policies acting before this
  plateau remain open. The forgotten inherited connector `g` repairs wording,
  not the stop theorem.
- **Machine check / regression:** Finite hand quotient, not a checked-in test.
  Preserve a future exhaustive action/response quotient fixture and the exact
  cap-risk-three witness.

### IL-12 — safe tempo intervention at `P_5^pl=P_stock`

- **Status / exact dead statement:** **DEAD.** Every action at the named stock
  plateau is unsafe; waiting through `P_5` cannot provide the desired safe
  intervention.
- **Killing theorem:** The inherited R7.2 result is carried as binding in
  `git show f5349d3e:GAP_RAW_PROOF_ROUND9.md`, **§83** and **§86.6**. The corrected
  combined ladder is `P_0,P_1,P_2` safe and `P_3,P_4,P_5` unsafe, so the exact
  transition is `k*=3`.
- **Open boundary:** The safe `P_0–P_2` policies and their exact risk-two
  results survive. General initialization/renewal and Q2 root forcing remain
  open.
- **Machine check / regression:** No current runnable regression is cited by
  the audited Round-9 artifacts. Any tempo-pruning implementation needs a
  fixed-state exhaustive `P_0–P_5` ladder test, with `P_2`'s `(5,-4)` incidence
  repair included.

## Superseded route record (not a global death)

### IL-R1 — original overlapping `kappa_cut`; repaired by `kappa_cut^*`

- **Status / exact disposition:** **REPAIRED-BY `kappa_cut^*`.** The original
  displayed `kappa_cut` was not a function on a non-FC edge into an all-empty
  D-alive target with `d in W` and `q<6`: its direct-incidence row said one and
  its `q<6` row said zero. The zero reading undercounted a reachable
  `1+5=6` continuation. This supersedes the old rule and withdraws original
  FHW-T3; it does **not** kill target-local danger cuts generally.
- **Counterexample and repair:** Follow `PROOF_TSS_ZONES_FHW.md`, **Repair
  record** (`:3–28`), **FHW-T3-R** (`:449–525`), **§2.2a** (`:591–642`), and
  **Review erratum/disposition** (`:1189–1215`). The independent reconstruction
  is `PROOF_TSS_ZONES_FHW_REVIEW2.md`, **§2 A0** (`:122–201`) and **§6**
  (`:643–658`). The repaired direct row charges one and rejects `1+5<6`.
- **Open boundary:** `kappa_cut^*` is proved only on the stated D22/RC/WC
  annotated class. Arbitrary mixed D17/D22 histories, mixed-history support
  reach, scalar-`B` debit, global logical minimality, and total-zone shrink are
  open (`PROOF_TSS_ZONES_FHW_REVIEW2.md:584–621,688–697`).
- **Machine check / regression:** The trace is finite and machine-checkable,
  but FHW-T3-R is not implemented here. The future verifier extension mandates
  the A0 rejection trace and all guard-boundary mutations in
  `DESIGN_VERIFIER_FHW_EXTENSION.md`, **§6, Mandatory R-Z10 rejection trace**
  (`:604–635`) and **§9.1** (`:748–751`). Required test name should include
  `fhw_rz10_a0_rejects_direct_one_plus_five`.

## Unverified claims appendix

None of the supplied seed claims had to be discarded after scope correction.
This does **not** mean every negative result has a currently runnable test:
IL-03 and IL-06–IL-12 explicitly record missing machine regressions, and IL-01
has only its geometry scaffold checked in rather than the full false-WIN gate.
