# R-T1-REV — hostile review of `DFPN_RETRAVERSAL_THEORY.md`

Reviewed at worktree HEAD `28a276e5e1715681c291d0a86b70cbbfc588297f`.
This was a document-only audit: no Cargo command and no Lean command was run.
The review treats all ten IDs marked **PROVEN** in the result ledger (T1, F1,
T2, T2b, C2, T3, T4, T5, T6, D1) as guilty until independently re-derived.
Counts below use the note's own definitions: expanding an unresolved arena
entry, including marking it `DepthCutoff`, is one expansion event; merely
reopening a cutoff is not.

## Findings

1. **MAJOR — the advertised “engine-faithful numerical core” is not faithful at the finite sentinel.**

   **Exact claim.** “**DEFINITION (model M).** ... Arithmetic is exact below a
   formal \(\infty\)” (`DFPN_RETRAVERSAL_THEORY.md:209-218`), under the heading
   “Engine-faithful numerical core”; and T2 is quantified “for every integer
   \(q\ge2,M\ge1\)” (`:33`, `:363`).

   **Counter-derivation.** The abstract formal-infinity model is coherent, but
   it is not the cited experimental engine at saturation. The engine sets
   `PN_INFINITY = 1_000_000_000` (`tss_solver.rs:1977`), clamps experimental
   inherited thresholds and additions to that value (`:3999-4005`,
   `:4033-4042`), and clamps branch sums to it (`:4959-4969`). It also uses the
   clamped current-plus-one as the purported progress floor (`:4159-4169`,
   `:4175-4203`), so at the sentinel the floor is not strict.

   T2 gives a concrete divergence. After expanding root \(R\), its DN is 2, so
   experimental `work` supplies \(D\) a DN threshold
   \(I-(2-1)=I-1\), where \(I=10^9\), not infinity. If \(M\ge I-1\), expanding
   \(D\) makes its summed DN at least \(I-1\), and the stopping test at
   `tss_solver.rs:4080-4085` returns before any \(x_i\) expands. Thus even the
   supported \(q=2\) engine trace is not \(E_q=M+3\) for all \(M\), although
   the ideal-model trace is.

   **Repair.** Rename the section “idealized numerical core below saturation,”
   add saturation explicitly to the engine exclusions, and state a separate
   finite-sentinel realization range. For T2, the simple sufficient range is
   \(2\le q<I\) and \(1\le M<I-1\); a literal current-engine prior also has
   \(q\le37\). This finding does not refute the formal-infinity theorem, but it
   does bar unrestricted model-to-engine transfer.

2. **NOTE — T1's path charge is valid, but only because progress certification assumes the hard part.**

   **Exact claim.** “In a progress-certified run of model M,
   \(V\le(d+1)E\). ... With exact selected-cutoff deepening,
   \(E\le2N-1\)” (`DFPN_RETRAVERSAL_THEORY.md:275-284`).

   **Independent confirmation.** Map each activation to the first expansion in
   its dynamic extent. Progress certification (`:231-236`) guarantees that the
   map exists. When event \(e\) occurs, every activation mapped to \(e\) is
   simultaneously live on the one recursion stack, which has at most \(d+1\)
   frames. Hence every event receives at most \(d+1\) charges and summing gives
   \(V\le(d+1)E\). Persistence gives at most one ordinary expansion per entry.
   With exact deepening, the root is never too deep, and every other entry is
   either expanded once normally or is expanded once into `DepthCutoff` and
   once after reopening. Therefore \(E\le1+2(N-1)=2N-1\).

   This is not an unconditional engine result. Actual `work` can return without
   an expansion because of an already-cut-off entry, a crossed threshold, node
   or soft cap, policy stall, or yielded Universal child
   (`tss_solver.rs:4047-4092`, `:4402-4438`); the finite-sentinel floor can also
   cease to be strict (Finding 1). The note candidly excludes the first group at
   `DFPN_RETRAVERSAL_THEORY.md:298-303`, so there is no hidden reduction to
   discharge.

   **Repair.** At `:291-294`, replace wording that can suggest three events
   (“expands once” plus cutoff plus reopen) with the either/or wording above,
   and say that a run terminates when the root solves so post-solution calls are
   outside \(V\).

3. **MINOR — F1's exact count survives small cases; “reopened frontier” is false at stage zero.**

   **Exact claim.** “The \(d+1\) stages have caps \(0,1,\ldots,d\). ...
   \(E=2d+1\) ... \(R=d(d+3)/2\)”
   (`DFPN_RETRAVERSAL_THEORY.md:314-329`).

   **Hand simulation.** For \(d=2\): cap 0 activates
   \([v_0,v_1]\), expands \(v_0\), then cuts off \(v_1\), so \((V,E)=(2,2)\).
   Cap 1 activates \([v_0,v_1,v_2]\), expands reopened \(v_1\), then cuts off
   \(v_2\), so \((3,2)\). Cap 2 activates \([v_0,v_1,v_2]\) and expands
   reopened \(v_2\) to the terminal, so \((3,1)\). Totals are
   \(N=3,E=5,V=8,R=5\), exactly the formulas. For \(d=1\), the two stages are
   \((2,2),(2,1)\), giving \(N=2,E=3,V=4,R=2\). In general,
   \(V=\sum_{k=0}^{d-1}(k+2)+(d+1)=(d^2+5d+2)/2\), and subtracting the
   \(d+1\) first activations gives \(R=d(d+3)/2\).

   The engine really increments the expansion count before marking a cutoff
   (`tss_solver.rs:5055-5080`) and advances to the exact selected cutoff depth
   (`:2085-2094`, `:3735-3774`). The only error is prose: at cap 0, \(v_0\) is
   fresh, not reopened.

   **Repair.** Change “expands the reopened frontier through \(v_k\)” at
   `DFPN_RETRAVERSAL_THEORY.md:315-317` to “expands the current frontier
   \(v_k\) (reopened when \(k>0\)).”

4. **NOTE — T2 is exact in model M; unrestricted transfer to the finite-sentinel engine fails.**

   **Exact claim.** “With \(\delta=1\) ... \(E_1=3\) including the root” and
   “With \(\delta=q\) ... \(E_q=M+3=N\)”
   (`DFPN_RETRAVERSAL_THEORY.md:381-403`).

   **Hand simulation.** Take \(q=2,M=2\), so \(N=5\). Under +1 the exact
   event sequence is `R, D, W`: \(D\)'s call threshold is 2; expanding \(D\)
   exposes two children of PN 2 and makes \(pn(D)=2\), so it returns before
   either child; \(W\) then proves. Thus \(E_1=3\). Under +2 the sequence is
   `R, D, x1, x2, W`: the threshold is 3, \(pn(D)=2\) while either \(x_i\)
   remains, and each false expansion changes its own DN to zero. Only after
   `x2` is \(D\) refuted. Thus \(E_2=5=M+3\), exactly two extra events. For
   \(q=3,M=2\), +1 is again `R,D,W`, while +3 is
   `R,D,x1,x2,W`; the same count confirms that the proof is not special to 2.

   Conjunctive DN subtraction is nonbinding only because model M supplies a
   formal-infinite opposing budget. The current code implements the same
   algebra below saturation (`tss_solver.rs:4147-4170`) but not for all
   \(q,M\), as Finding 1 demonstrates. The same-kind Choice-to-Choice edge is
   also an ideal-model device; the note expressly says so at
   `DFPN_RETRAVERSAL_THEORY.md:411-416`.

   **Repair.** Keep the ideal theorem, but append “not an engine theorem for
   unbounded \(q,M\); see the finite-sentinel range” to the ledger and T2.

5. **NOTE — T2b's all-unit-prior construction and node count are exact.**

   **Exact claim.** “For \(\delta=q\) ... the ladder ... needs exactly \(\ell\)
   expansions to reach PN \(q\). ... \(E_q=\ell+3,\quad
   N=\ell+q+4\)” (`DFPN_RETRAVERSAL_THEORY.md:433-447`).

   **Hand simulation.** Use the smallest case \(q=2,\ell=2\). The eight nodes
   are \(R,A,B,Y\), the two-node chain \(X\) (Choice) then \(Z\) (bottom
   Universal), and the two unexpanded children of \(Z\). Under +1, the exact
   sequence is `R,A,B`: expanding \(A\) backs it up from PN 1 to
   \(pn(X)+pn(Y)=2\), meeting its root threshold 2. Hence \(E_1=3\). Under
   +2, \(A\)'s threshold is 3. At \(A\), conjunctive subtraction gives selected
   \(X\) PN threshold \(3-(2-1)=2\), while its competitive DN threshold is 3.
   Expanding \(X\) reveals its unary child; expanding bottom Universal \(Z\)
   reveals two unit-prior children and backs \(X\) up to PN 2 without expanding
   either child. Then \(A\) reaches PN 3 and returns, followed by \(B\). The
   sequence `R,A,X,Z,B` gives \(E_2=5=\ell+3\), overhead 2, and
   \(N=8=\ell+q+4\). Unary alternation preserves the carried pair, so this
   extends directly to even \(\ell\); with fixed \(q=2\), the overhead is
   \(\ell=N-6=\Theta(N)\) at \(H=1\).

   **Repair.** None required. A short trace like the one above would make the
   otherwise terse ladder definition much easier to falsify and verify.

6. **MAJOR — the official off-versus-2 experiment changes sentinel clamping as well as the second-best increment.**

   **Exact claim.** “The selected delta value enters the scheduling equations
   only through the competitive second-best term”
   (`DFPN_RETRAVERSAL_THEORY.md:137-142`), and R-TS1 is “an exact local test of
   the literature's proposed remedy” (`:101-111`).

   **Counter-derivation.** The next sentence in the note mentions clamping, but
   the causal description never establishes that it was inert. The official +1
   raw says `TSS_CORPUS_EXPECT_THRESHOLD_DELTA=off` and
   `threshold_delta=off` (`THRESHOLD_COUNTER_FULL_RAW.log:11,19`). The delta-2
   raw sets both the expectation and `TSS_THRESHOLD_DELTA=2`
   (`THRESHOLD_DELTA2_FULL_RAW.log:11,15,20`). Any `Some(delta)`, unlike `None`,
   also clamps inherited PN/DN thresholds, second-best additions, and the unit
   progress floors to `PN_INFINITY` (`tss_solver.rs:3999-4005`, `:4033-4042`,
   `:4154-4203`). Therefore the official A/B is not literally a one-expression
   change if a live value reaches the sentinel.

   There is useful but incomplete mitigation: retained
   `THRESHOLD_FULL_D1_RAW.log:13,186-190` shows that `Some(1)` reproduced the
   off run's integer structural totals exactly (9,080,708 visits, 4,574,016
   revisits, 8,464,552 threshold returns, 8,056,474 reselections, and 6,188,156
   switches). That makes the clamp observationally inert in the retained +1
   structural counters. It does not prove that no delta-2 path reached a live
   saturated PN/DN, because the retained delta-2 aggregate has no sentinel-hit
   counter.

   **Repair.** Call the A/B exact only below the sentinel, cite the `Some(1)`
   structural equality, and require either an identically clamped control or a
   sentinel-hit assertion/counter before attributing every schedule difference
   solely to second-best widening. This does not change the empirical fact that
   the shipped +1 mode beat the tested delta-2 implementation.

7. **MINOR — the revisit attribution uses unexplained truncation rather than conventional rounding.**

   **Exact claim.** “**CITED (local measurement).** ... Proportional
   attribution by the visit/revisit ratio assigned 34.79 s, or 7.01% of wall”
   (`DFPN_RETRAVERSAL_THEORY.md:56-64`; source report at
   `HUNT_REPORT_THRESHOLD_SCALE.md:23-34`).

   **Recalculation.** `THRESHOLD_COUNTER_FULL_RAW.log:192` gives
   \(V=9{,}080{,}708\), \(R=4{,}574{,}016\), and exclusive descent time
   69,080.618 ms. It also exactly gives the cited 8,464,552 threshold-cross
   returns, 8,056,474 reselections, and 6,188,156 sibling switches. Thus

   \[
   R/V=0.5037069797,
   \qquad 69.080618(R/V)=34.796389\text{ s},
   \]

   and against the stated uncontaminated 495.94 s baseline this is
   7.0162498%. Standard nearest rounding is **34.80 s and 7.02%**, not 34.79 s
   and 7.01%. The recorded 13.93% descent share is correct because
   \(69.080618/495.94=13.92923\%\). The published figures are consistent with
   truncation, but neither the report nor the theory says that truncation rather
   than nearest rounding was intended.

   T6's later arithmetic is internally correct for its explicitly chosen
   \(f=0.0701\): \(f/(1-f)=0.0753844\) and \(0.05/f=0.713267\). Using the
   unrounded raw-derived share instead gives a 7.55% ceiling and still 71.3% to
   one decimal.

   **Repair.** Use 34.80 s / 7.02%, recompute the displayed ceiling as 7.55%,
   report “about 34.8 s (about 7.0%),” or label 34.79 / 7.01 explicitly as
   truncated values.

8. **NOTE — every hard-row ratio and absolute counter quoted in the theory checks out.**

   **Exact claim.** “Expansions rose from 1,879,611 to 6,054,588 (3.2212x)” and
   “revisits per expansion fell from ... 0.819 to ... 0.604, a 26.3% reduction
   ... [while] absolute revisits [rose] 2.374x and visits 2.840x”
   (`DFPN_RETRAVERSAL_THEORY.md:66-88`).

   **Recalculation.** The source rows are exactly
   `THRESHOLD_COUNTER_FULL_RAW.log:40,42` and
   `THRESHOLD_DELTA2_FULL_RAW.log:41,43`. They give:

   - expansions: \(6{,}054{,}588/1{,}879{,}611=3.221192\);
   - revisits/expansion: 0.81875665 versus 0.60350300;
   - relative intensity reduction: 26.2903%;
   - absolute revisit ratio: \(3{,}653{,}962/1{,}538{,}944=2.3743307\);
   - visit ratio: \(9{,}708{,}510/3{,}418{,}518=2.8399763\); and
   - peak indexed bytes: 549,161,606 versus 1,073,741,810.

   The official wall increase is also correct:
   \((927.59/499.85)-1=85.5737\%\), rounding to 85.6%. The expansion difference
   named later in the resume point is exactly 4,174,977. The counter-run
   overhead relative to 495.94 s is 0.7884%, consistent with the report's
   approximately 0.8%, and 549,161,606 bytes is 523.721 MiB, correctly rounded
   to 523.7 MiB. No repair is required.

9. **NOTE — 218 is the exact verdict recount; “228” is conclusively a report typo.**

   **Exact claim.** “The eight raw rows contain
   \(4(16)+(39+39+38+38)=218\) verdicts ... The ‘228’ ... is therefore an
   arithmetic typo” (`DFPN_RETRAVERSAL_THEORY.md:90-99`).

   **Recalculation.** `THRESHOLD_LEAF_AB_RAW.log:58-61` has four h8 rows of 16
   verdicts, totaling 64. Lines 62-65 have 39, 39, 38, and 38, totaling 154.
   Hence the total is exactly \(64+154=218\). Every row has `verified` equal to
   `verdicts`, every `contradictions` field is zero, and the test increments
   `verified` only after `TssVerifier.verify` succeeds
   (`tss_leaf_surface_hunt.rs:407-423`). No retained row supports the extra ten
   in `HUNT_REPORT_THRESHOLD_SCALE.md:62-63`. The expansion fields also match
   the theory exactly: every h8 row is 1,852; the four h16 rows are 6,649,
   6,726, 7,106, and 7,135.

   **Repair.** Change 228 to 218 in the hunt report. The theory's correction is
   confirmed as written.

10. **NOTE — C2 is a valid formal asymptotic, not an unbounded fixed-sentinel engine trace.**

    **Exact claim.** “Setting \(q=2\) fixes \(\delta=H=2\) and depth at two
    while \(M\) grows. ... no \(o(N)\) upper bound can be a function only of
    \(\delta,H,d\)” (`DFPN_RETRAVERSAL_THEORY.md:405-409`).

    **Independent confirmation and boundary.** For \(M=1\), \(N=4\): +1 has
    `R,D,W`, so \(E_1=3\); +2 has `R,D,x1,W`, so \(E_2=4\), gap
    \(1=N-3\). For \(M=2\), the traces in Finding 4 give \(E_1=3\) and
    \(E_2=5\), gap \(2=N-3\). In formal model M this continues for all \(M\),
    proving a linear lower bound and excluding every universal \(o(N)\) bound
    under only those fixed parameters.

    It does not continue indefinitely in the delta-2 engine. With
    \(I=\texttt{PN_INFINITY}\), \(D\)'s opposing threshold is \(I-1\); for
    \(M\ge I-1\), its saturated DN fires first (Finding 1). A fixed finite
    sentinel cannot preserve this trace along an \(N\to\infty\) family. The
    theorem remains sound
    because model M explicitly uses formal infinity, but phrases such as “a
    delta-2 catastrophe” must retain that scope.

    **Repair.** State “formal model M” in C2 and the negative boundary at
    `:799-804`; add that the exact delta-2 engine trace requires
    \(M\le\texttt{PN_INFINITY}-2\). A less awkward statement of the negative
    result is: “No universal bound \(g(N;\delta,H,d)=o(N)\) exists for model M.”

11. **MINOR — T3's count is exact after supplying an omitted opposing-budget/DN condition.**

    **Exact claim.** “With heuristic initialization ... all \(M\) false
    children must be expanded ... The cost is \(M+3=N\). Therefore ... ratio
    2 causes \(M+1=N-2\) extra expansions”
    (`DFPN_RETRAVERSAL_THEORY.md:468-488`).

    **Hand simulation.** For \(M=2\), the admissible run is `R,W`, since W and
    D tie at PN 1 and W wins the declared tie: \(E=2\). In the heuristic run W
    has PN 2, so the exact sequence is `R,D,x1,x2,W`. D is the unique PN-1
    child and receives PN threshold 3. Its first false child receives threshold
    2; after that child refutes, D remains PN 1. Its last child refutes under the
    remaining parent cap, then W proves. Thus \(E=5=N\), excess 3, exactly
    \(M+1=N-2\). For \(M=1\), the corresponding costs are 2 and 4, excess 2;
    there is no off-by-one.

    The bounded-branching recurrence also survives: \(n_0=1\), while
    \(F_1\) contains its Choice node, two unary wrappers, and two leaves, so
    \(n_1=5=3+2n_0=4\cdot2^1-3\). Induction gives
    \(n_k=4\cdot2^k-3\), and the PN-1 plateau forces every one of these nodes
    before W in the heuristic run.

    The definition specifies scalar PN priors but, unlike T2 at `:378-379`,
    does not explicitly give all DN priors as 1 and the root opposing budget as
    nonbinding. Those facts are used by “all \(M\) ... must be expanded”; an
    arbitrary low DN budget could end D's call early. They are the evident
    intended completion, not a hard new lemma. An explicit
    `TSS_THRESHOLD_DELTA=1` arm would also inherit the clamping caveat in
    Finding 6, whereas production/default +1 (`threshold_delta=None`) retains
    the open `u32::MAX` inherited threshold.

    **Repair.** Add “all DN priors are 1 and root thresholds are infinite” to
    \(S_M\). At `:799-804`, say “winner-prior overestimate ratio in model M,”
    rather than suggesting a theorem for every possible global heuristic-ratio
    definition.

12. **NOTE — T4's calibrated bound, matching cases, and paired-run obstruction all survive.**

    **Exact claim.** “\(S\le\sum_{i\ne w}(p_w+\delta-p_i)_+
    \le(b-1)(H+\delta-1)\)” and only the envelope increment, not actual paired
    work, is bounded by \((b-1)(\rho-1)P\)
    (`DFPN_RETRAVERSAL_THEORY.md:527-569`).

    **Independent derivation.** While w is unselected, a numerically selected
    distractor has score at most \(p_w\), and w itself ensures that its
    second-best sibling score is at most \(p_w\). Both the unit progress floor
    and competitive threshold are therefore at most \(p_w+\delta\). Under the
    stated monotone unit response, distractor i consumes at most
    \((p_w+\delta-p_i)_+\) expansions before refuting or becoming strictly
    worse than w. Summation proves (8), and \(p_w\le H,p_i\ge1\) proves its
    coarser envelope.

    Exact checks attack both off-by-one risks. With \(b=2,p_w=2,p_i=1,
    \delta=2\), the distractor runs \(1\to2\to3\to4\): three expansions,
    exactly \(p_w+\delta-p_i\). With \(\delta=1,b=3,H=2\), ordered scores
    A1, B1, W2 evolve A \(1\to2\) (one expansion), B \(1\to3\) (two), then A
    \(2\to3\) (one), totaling 4, exactly \((b-1)H\).

    The paired obstruction also recounts exactly. For \(P=2,\delta=2\), the
    two distractors use 2 and 3 expansions, so \(S(2)=5\). With heuristic score
    3, they use 2 and 4, then the first tie-winner uses 2 more, so
    \(S(3)=8\). The actual excess 3 is greater than
    \((b-1)(3-2)=2\), while the envelope calculation remains valid.

    **Repair.** None. The calibration and nonbinding-threshold hypotheses are
    load-bearing and are already stated clearly.

13. **MINOR — T5 is correct in formal model M; clamped engine returns require a different charge.**

    **Exact claim.** “At its end, score \(y\ge s+\delta\). ... positive
    variation ... at least \(y-x\ge\delta\). ... Therefore
    \(B_{\rm barrier}\le PV/\delta\)”
    (`DFPN_RETRAVERSAL_THEORY.md:580-605`).

    **Independent confirmation.** In model M, minimum selection gives
    \(x\le s\). A qualifying interval ends at \(y\ge s+\delta\), so its
    positive variation is at least its net increase and hence at least
    \(\delta\). Disjoint edge intervals can therefore be charged once each.
    Under monotonicity through U, total charge on edge i is at most
    \(U-p_i\), and the integer count gives
    \(\lfloor(U-p_i)/\delta\rfloor\).

    For a hand check, take three arms at score 1, \(\delta=2,U=3\). The first
    two arms each run \(1\to3\) and make a barrier return. During the final
    arm, the process reaches its exclusive stopping boundary U, so that
    activation is excluded from the barrier-return count. Thus \(B=2\), while
    (11) gives 3; the claimed near-match ratio is exactly 2/3.
    Independent one-return gadgets with a \(1\to3\) change have \(PV=2\) each
    and attain (10) exactly.

    The real delta-arm threshold is
    \(\min(s+\delta,\texttt{PN_INFINITY})\), not always \(s+\delta\)
    (`tss_solver.rs:3999-4005`, `:4148-4166`). Let
    \(x=s=I-1\) and \(\delta=2\). The engine's actual barrier is I, so one
    increase \(I-1\to I\) returns with positive variation 1. Calling this an
    engine competitive-barrier return makes \(1\le PV/2\) false. Reading the
    note literally avoids contradiction only by excluding this real return
    from its definition, because it never reaches the impossible literal
    \(s+\delta=I+1\). Production root/commitment policy can also select
    \(x>s\) (`tss_solver.rs:4095-4127`), defeating the same charge unless such
    calls are filtered.

    **Repair.** For direct engine use, require numerical minimum selection and
    \(s\le I-\delta\). More generally charge each return by its actual margin
    `effective_threshold - entry_score`; then the valid weighted statement is
    that the sum of those margins is at most PV. Do not describe (10) as a
    counter ceiling for every production threshold-cross return.

14. **MINOR — T6's identity is exact; only its rounded empirical substitution needs errata.**

    **Exact claim.** “\(W_\delta<W_1\iff
    \gamma(1-f)<\sigma f\iff f>\gamma/(\gamma+\sigma)\)” and a necessary
    condition is \(\gamma<f/(1-f)\)
    (`DFPN_RETRAVERSAL_THEORY.md:628-683`).

    **Independent derivation.** Direct subtraction gives
    \(W_\delta-W_1=\gamma C_{\rm other}-\sigma C_{\rm rev}
    =W_1[\gamma(1-f)-\sigma f]\). For \(\gamma+\sigma>0\), rearranging proves
    (13). Since \(\sigma\le1\), a win implies
    \(\gamma(1-f)<f\), hence (14). The separately handled
    \(\gamma=\sigma=0\) case is correct.

    Substituting the document's chosen \(f=0.0701\) gives
    \(0.0701/0.9299=0.07538445\) and
    \(0.05/0.0701=0.71326676\), so 7.54% and 71.3% are internally consistent.
    At 7.54% inflation and perfect revisit removal, the normalized new cost is
    \(0.9299(1.0754)=1.00001446\), confirming the strict “cannot win” wording.
    Finding 7 explains why the raw-derived attribution would conventionally be
    7.02%, shifting only the second decimal of the ceiling.

    **Repair.** Keep the theorem. Align the empirical substitution with the
    corrected rounding in Finding 7, or state explicitly that 0.0701 is a
    deliberately truncated report value.

15. **MINOR — D1's direct-child duplication is exact; its integer domain and counted work should be explicit.**

    **Exact claim.** “The saturated index creates and expands one private copy
    per parent edge, for \(kM\) child expansions. The exact amplification is
    \(kM-M=(k-1)M\)” (`DFPN_RETRAVERSAL_THEORY.md:689-709`).

    **Hand simulation and code check.** For \(k=2,M=2\), an unlimited index
    expands z1 and z2 under the first parent and the second parent hits them:
    two direct child-entry expansions. A saturated admission-only index creates
    z1/P1, z2/P1, z1/P2, z2/P2: four, for exact excess 2. For \(k=3,M=1\), the
    counts are one versus three, excess two. These equal \((k-1)M\).

    The engine's relevant data-structure behavior matches the abstraction. An
    indexed key hits (`tss_solver.rs:3606-3612`); otherwise a persistent arena
    entry is always pushed (`:3621-3637`) and only its index admission is
    conditional on the byte cap (`:3639-3651`). The private returned id is then
    linked on that parent edge (`:4272-4274`, `:4347-4349`). An earlier
    unindexed arena copy is not discoverable through `by_position`. The hard
    delta-2 raw is also consistent: 6,054,588 expansions, only 3,586,248 indexed
    entries, and peak index use 1,073,741,810 bytes—14 bytes below the cap
    (`THRESHOLD_DELTA2_FULL_RAW.log:41`). The minimum retained index charge is
    larger than 14 bytes (`tss_solver.rs:2857-2864`), so admission was saturated.
    As the theory correctly says, these aggregates do not count duplicate
    semantic keys.

    The lemma never explicitly says \(k,M\ge1\); \(k=0\) makes its “first
    parent” proof and formula interpretation nonsensical. It should also say
    that (15) counts the z entries' own first expansions (for example make each
    z terminal-on-expansion), not arbitrary descendant work that may differ
    after copying.

    **Repair.** Quantify positive integers \(k,M\) and define each z as one
    terminal-on-expansion entry, or rename the quantity “direct z-entry
    expansions.” The D1-to-T2 production composition is correctly left SKETCH.

16. **NOTE — the remaining status discipline passes hostile review.**

    **Exact claims.** The note says E1 is “consistent with counters, not
    identified causally” (`DFPN_RETRAVERSAL_THEORY.md:41`) and labels the
    T2/D1 composition SKETCH (`:717-725`).

    **Confirmation.** The composition explicitly leaves lazy admission and all
    production selection overrides unproved. The saturation discussion says
    “consistency, not causation” and disclaims evidence about duplicate semantic
    keys (`:727-734`). E1 is labeled CONJECTURE and admits that neither causal
    clause is isolated (`:736-742`). The promotion gate remains CONJECTURE and
    restricts its sufficiency (`:784-795`); the combined positive bound and DAG
    extension remain SKETCH (`:806-838`). The sole `CITED-FROM-MEMORY` passage
    (`:197-203`) is explicitly contextual and unused by the proofs. No PROVEN
    item secretly relies on it.

    **Repair.** None beyond the model/engine scope and small specification
    errata already identified above.

17. **NOTE — below saturation and outside disclosed policy overrides, Equation (1) matches the code.**

    **Exact claim.** “`WidePnSearch` uses the saturated standard recurrences”
    and a selected Choice child receives the second-best threshold plus
    conjunctive DN-budget subtraction and a current-plus-one floor
    (`DFPN_RETRAVERSAL_THEORY.md:113-145`).

    **Confirmation.** For Choice, the code computes the minimum other-child PN
    and adds the selected delta (`tss_solver.rs:4147-4157`), applies unit PN/DN
    floors (`:4159-4165`), and uses
    `dn_threshold - (parent_dn - child_dn)` with saturating subtraction
    (`:4166-4169`). Universal is the exact PN/DN dual when commitment is off:
    second-other DN at `:4183-4194`, unit floors at `:4175-4177` and
    `:4196-4199`, and conjunctive PN subtraction at `:4200-4203`. A call checks
    backed-up terminal values and both thresholds at `:4069-4085`.

    Recompute uses Choice `min(pn)`/saturated `sum(dn)` and Universal
    saturated `sum(pn)`/`min(dn)` exactly as stated
    (`tss_solver.rs:4943-4976`), with terminal/cutoff values at `:4950-4951`.
    The mean arm is the floor integer mean of immutable nonselected sibling PN
    priors at Choice or DN priors at Universal (`:3965-3995`). The note also
    correctly discloses root sequential/width-tier selection and Universal
    commitment (`DFPN_RETRAVERSAL_THEORY.md:155-159`; `tss_solver.rs:4095-4127`,
    `:4690-4913`) and correctly distinguishes visits from expansions and cutoff
    reopening (`tss_solver.rs:3870-3893`, `:3921-3939`, `:4053-4068`). The only
    numerical mismatch is the saturation/clamping boundary already isolated in
    Findings 1, 6, 10, and 13.

    **Repair.** None beyond explicitly restricting the “engine-faithful” label
    to this unsaturated, numerical-minimum-selection regime.

## Verdicts

| PROVEN item | Verdict | Hostile-review result |
|---|---|---|
| T1 | **CONFIRMED** | The stack charge and \(2N-1\) event bound are valid under the explicitly assumed progress certification. The assumption is not proved for production, but the note already says so. |
| F1 | **CONFIRMED-WITH-ERRATA** | Exact \(d=0,1,2,3\) traces and the closed forms check out; stage zero is not a “reopened” frontier. |
| T2 | **CONFIRMED** | Exact under the document's explicit formal-infinity model scope. The separate “engine-faithful” rhetoric needs the finite-sentinel caveat. |
| T2b | **CONFIRMED** | The \(q=2,\ell=2\) trace, thresholds, event count, node count, alternation, and \(H=1\) asymptotic all check out. |
| C2 | **CONFIRMED** | The \(N-3\) lower bound and no-\(o(N)\) conclusion are valid in formal model M; this trace does not persist unboundedly with a fixed finite engine sentinel. |
| T3 | **CONFIRMED-WITH-ERRATA** | The \(M+1\) excess and bounded-branching recurrence are exact after explicitly supplying unit DN priors and a nonbinding root opposing budget. |
| T4 | **CONFIRMED** | The calibrated sum bound, equality cases, envelope Lipschitz bound, and paired-schedule counterexample all recount exactly. |
| T5 | **CONFIRMED** | The stated formal variation charge is correct. Extrapolation to all engine threshold returns would need minimum selection and an unsaturated barrier, or a charge by actual clamped margin. |
| T6 | **CONFIRMED** | The accounting equivalence and necessary condition are exact. The separately cited 7.01% input has a truncation/rounding erratum. |
| D1 | **CONFIRMED-WITH-ERRATA** | The direct-child excess \((k-1)M\) and admission-only code mapping are exact; quantify \(k,M\ge1\) and define the counted z-entry work. |

E1 remains **CONJECTURE**, correctly classified. The T2/D1 engine composition
and later combined bound remain **SKETCH**, also correctly classified. There
are no UNRESOLVED theorem verdicts.

**Overall verdict: CONFIRMED-WITH-MAJOR-ERRATA.** None of the ten formal-model
PROVEN items is refuted or needs downgrade to SKETCH. The document must,
however, stop calling the core unconditionally engine-faithful: finite
`PN_INFINITY` clamping breaks unrestricted T2/C2 transfer and prevents an
unqualified production extrapolation of T5, while the official off-versus-2 A/B also changes that clamping
behavior without a retained delta-2 sentinel-hit control. The raw empirical
story otherwise survives: every hard-row ratio is correct, 218 is the exact
leaf verdict total, and “228” is indeed a typo. Correct the proportional
attribution's conventional rounding from 34.79 s / 7.01% to
34.80 s / 7.02% (or label the former as truncation).
