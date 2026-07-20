# R-Z10 Phase 3: Group-2 `lambda^2` next-round design

## R-Z11 repair record and per-bar disposition

R-Z10-REV upheld G2-Z1 sound-on-success and native `lambda^2` soundness on
the design class, but left the implementation bars at MAJOR REPAIR. R-Z11
repairs the paper definitions below at input HEAD `ad606d0e` on branch
`claude/tss-vcf-width`. No build or measurement was run, and no commit was
made. The older provenance block below is retained as R-Z10 authoring history;
its `UNLANDED` placeholder is not the current artifact identity.

The controlling zone authority for this disposition is the required
read-only file
`E:\Hexo-BotTrainer-hexgt\.claude\worktrees\consolidate-main\docs\PROOF_TSS_DEFENDER_ZONES.md`,
2011 lines, SHA-256
`39197460D068CE5442BA0AFFC687F1408DF3F28EEEB26C4DD7192B87A202064B`.
The local 899-line snapshot is not authoritative. Every refined certificate
record must carry this authority digest plus the digest of
`PROOF_TSS_ZONES_FHW.md`; a mismatch makes the lane `INELIGIBLE`.

| reviewed bar / defect | R-Z11 disposition | binding resolution |
|---|---|---|
| FHW-T3 selector | **BUILD-READY** | Use repaired FHW-T3-R and `kappa_cut^*`; an all-empty direct fill pays one and requires `1+q<6`. |
| finite matched FHW clock ratio | **BUILD-READY** | Use the canonical finite `I_FHW` index in section 6.5; old/new values share exactly the same keys. |
| matched net-zone index `J` | **BUILD-READY** | Use `J_zone` in section 6.5, bound to one frozen certificate, node, role set, summary, and child-plan digest; any unmatched key fails the comparison. |
| H1152 population prevalence | **DEAD** | H1152 is a deterministic regression/materiality benchmark only. No population prevalence or population-weighted aggregate may be inferred from it. |
| H1152 benchmark bars | **BUILD-READY** | Its existing canonical lexicographic membership is retained as `H1152-B`; results describe only those fixed keys. |
| radius-nine constant-substitution proof | **DEAD** | Replacing every visible 8 by 9 is stress telemetry, not an independent proof. |
| radius-nine robustness replacement | **SPEC-FOR-CARGO** | Run the exhaustive bounded policy model checker specified in sections 6.3 and 6.7; no radius-substituted zone theorem is used. |
| baseline/variant identities and native promotion measurements | **SPEC-FOR-CARGO** | The exact harness, profiles, horizons, repetitions, artifacts, and failure rules are frozen in section 6.7. |
| Exact/FHW/SR materiality and economics | **SPEC-FOR-CARGO** | Use `J_zone`/`I_FHW` plus the explicit node/wall/peak gates in section 6.5. |

`BUILD-READY` here means the paper contract is complete enough to implement;
it is not evidence that code exists. `SPEC-FOR-CARGO` means the decision is
empirical and remains DEFERRED-NEEDS-CARGO until the exact specified run
passes. `DEAD` means the old interpretation must not be revived by a PASS in
a related stress test.

> **Provenance.** Worktree
> `E:\Hexo-BotTrainer-hexgt\.claude\worktrees\tss-vcf-width`, branch
> `claude/tss-vcf-width`, input HEAD
> `7c2706c86a0362f8e9ddff35ddb1e3185fa0670c` (short `7c2706c8`). Written
> 2026-07-18, America/New_York. Original R-Z10 reading order: the checked-in zone corpus;
> `docs/PLAN_TSS_SOLVER_UPGRADES.md` §§I.2/I.5 and cited Group-2 records;
> `PROOF_TSS_ZONES_FHW.md`; then read-only engine/finder/verifier source. No
> Cargo command was used, no commit was made or is authorized, and every
> executable measurement below is marked **DEFERRED-NEEDS-CARGO**.
> **Landed-hash placeholder:** `UNLANDED (post-review folding
> owner/orchestrator action required)`.
>
> **Status convention.** **PROVEN-ON-CLASS** is a scoped PROVEN claim under
> its stated hypotheses. **DEFERRED-NEEDS-CARGO** is OPEN empirically: it is a
> preregistered build/run, not evidence that the bar will pass.

## 0. Outcome and claim ledger

The next structural round should migrate G2R3's quiet OR universe and
certificate-dependent unforced AND zones into the **native proof-number
engine**. At this HEAD, enabling both consume flags deliberately routes to
`WidePnSearch::prove_narrow_compat`; native PN still refutes every defender
node whose forcing-dispatch premise fails. Thus G2R3 supplies sound λ²
certificates, but not native PN scheduling of them.

| ID | Statement | Status |
|---|---|---|
| G2-Q1 | The complete quiet attacker-turn universe has a finite exact index, including newly legal second stones. | **PROVEN** |
| G2-Q2 | Safe turn-start pairs admit a canonical commutation quotient; dynamic and singleton-terminal cases do not. | **PROVEN** |
| G2-Z1 | Certificate-dependent unforced zones admit a finite inflationary closure over frozen child proofs. | **PROVEN sound-on-success** |
| G2-Z2a | Slack `k<b` pressure alone licenses a generic FHW window debit. | **PROVEN FALSE** by FHW-O2 |
| G2-Z2b | Phase-2 support reach remains sound through arbitrary D17/D22 mixed histories. | **OPEN** outside FR-T1's class |
| G2-PN1 | Hidden quiet width and open zone closure have sound non-selectable PN-debt algebra. | **PROVEN as a state-machine theorem** |
| G2-NATIVE | A native implementation satisfying the stated invariants materializes only verifier-valid λ² certificates. | **PROVEN-ON-DESIGN-CLASS; implementation DEFERRED-NEEDS-CARGO** |
| G2-MEASURE | The native design meets the preregistered corpus, mutation, and performance bars. | **DEFERRED-NEEDS-CARGO** |

## 1. Ground-truth seam at input HEAD

G2R3 (`bfd03ca9`, now an ancestor of the input) established two consuming
capabilities:

- complete quiet-turn fallback at attacker OR nodes after the forcing tranche
  is exhausted; and
- the T3/T4 four-part ranked zone at `k<b` defender AND nodes, with rank used
  only for ordering and never as a cap. Its production uniform fallback uses
  `seed_band_radius(d)=8(d-1)` for `d>=1` and zero for `d<=1`; every native
  comparison retains that exact G2R3 baseline unless a proved refined class
  is explicitly selected.

Its strict witness `double_fork_compact` changed the structural result from
wide `UNKNOWN/2` to verifier-accepted `WIN/409` at the 10k rung. The two
unforced nodes searched `62/478=0.129707` and `18/479=0.037578`; their
`dir/seed/touch/virgin` component counts were `19/0/50/0` and `18/0/0/0`
(overlap explains why component sums need not equal the union). The 19
forcing rows fired neither feature, so they are regression coverage, not
λ² measurement coverage.

The current source has a narrower implementation truth:

1. `tss_solver.rs:889-890` makes `uses_wide_pn=false` when both G2R3 consume
   flags are active; `:1061-1090` dispatches that combination to
   `WidePnSearch::prove_narrow_compat` (`:3727-3765`).
2. Native PN's defender expansion at `:5971-5985` accepts only
   `min_hitting_set==b`; every unforced defender node is immediately refuted.
3. The recursive compatibility path already contains the correct operational
   prototypes: full-legal quiet fallback at `:7703-7802`, and append-only
   zone closure at `:7804-7947` using `zone_certificate_extras`
   (`:9094-9155`).
4. Native PN's ordinary Choice/Universal recurrences are at `:5469-5502`;
   its Universal materializer currently emits `zone:None` at `:6884-6929`.
5. The independent verifier reconstructs roles/budgets and the uniform zone
   at `tss_verify.rs:1026-1149`, triggers its check at `:1216-1268`, and
   validates same-turn commutations at `:741-826`.

Line numbers name seams at the input HEAD and may drift after implementation.
The design target is not another zone formula alone. It is a native PN state
machine which cannot prove or refute a node while a required move universe is
still hidden.

## 2. Exact attacker-turn universe

### 2.1 Enumeration before completeness

For a nonterminal claimant position `P`, let `L(P)` be the engine's finite
legal set. At Opening or SecondStone, the raw complete-remainder language is

```text
{ [x] : x in L(P) },
```

with terminal/nonterminal outcome recorded by replay. At FirstStone, put

```text
L_0 = L(P),
I   = { x in L_0 : P+x is an immediate claimant win },
L_x = L(P+x), for x in L_0 \ I.
```

the finite raw full-turn language is exactly

```text
{ [x]   : x in I }
union
{ [x,y] : x in L_0 \ I, y in L_x }.
```

**G2-Q1 (raw quiet-turn completeness). PROVEN.** The displayed language is
exhaustive.

*Proof.* Engine cadence is FirstStone to SecondStone to the opponent's
FirstStone. A win is tested after each placement and terminates immediately.
Thus a nonwinning first placement has exactly one more placement, chosen from
the legal set after that first placement. These cases are disjoint and
exhaustive. Each set is finite because a finite occupied board has a finite
union of inclusive radius-eight legal halos; the empty-board Opening has the
engine's singleton origin move. ∎

For deduplication only after raw completeness is established, use
`Single(x,outcome)` for **every** `x in L(P)` at Opening or SecondStone; the
outcome tag records terminal/nonterminal replay. At FirstStone use four kinds:

1. `Win1(x)`: `[x]`, `x in I`;
2. `StartPair({x,y})`: one **unordered quotient key** for distinct
   `x,y in L_0\I`, with a separately replayed legal orientation;
3. `TerminalSecond(x,y)`: ordered, `x in L_0\I`, `y in I`;
4. `FrontierPair(x,y)`: ordered, `x in L_0\I`, `y in L_x\L_0`.

**G2-Q2 (canonical quotient). PROVEN.** `Single` plus the four FirstStone
kinds retain an existentially sufficient representative of every raw
game-semantic turn-outcome class. This is not equality of engine histories,
`last_turn` metadata, or terminal phase keys. A `TerminalSecond` is also
dominated by its earlier `Win1(y)` for existential proof search, but must not
be justified by commutation.

*Proof.* After a nonwinning placement, legality is monotone except that the
occupied cell disappears; every other turn-start legal cell remains legal.
If neither singleton wins, `[x,y]` and `[y,x]` have the same owner-labelled
board. If the pair is nonterminal, Hexo's next legal game state has the same
opponent, FirstStone cadence, and placement count; order-specific `last_turn`
metadata may differ, but it lies outside the native TSS search-semantic
signature and may be safely aliased there. If the pair jointly wins, both
legal orders end with the claimant win after the second placement, even though
their terminal phase metadata differs. One replayed orientation is therefore
sufficient for an existential Choice. If `y` was initially illegal,
the reverse order is not legal. If `y` wins as a singleton, `[y,x]` does not
exist because play stops after `y`. Those cases remain ordered. ∎

`StartPair` stays unordered through normalization. Materialization chooses and
replays one legal orientation; it does not assume order-specific certificate
identity. A D6 stabilizer can swap the endpoints, so no globally
equivariant literal orientation is promised. If a cache seam requires exact
ordered covariance, retain both orders for that stabilizer-ambiguous pair.

G2R3's current pair filter is also complete, though it may keep duplicates.
Let `T subseteq L_0` be its turn-start forcing candidates. It rejects a
second `y` after `x` only when `y in T` and `key(y)<key(x)`; the reverse starts
with enumerated `y` and retains `x`. A newly legal `y` is outside `L_0`, hence
outside `T`, and cannot be filtered. A singleton-winning `y` already provides
the dominating retained edge.

### 2.2 Dynamic-legality obstruction

Turn-start-only unordered pairs are incomplete. The full legal replay is:

```text
ply 0  Player0 Opening:    (0,0)
ply 1  Player1 FirstStone: x=(8,0)
ply 2  Player1 SecondStone:y=(16,0)
```

`x` is legal at axial distance exactly eight from `(0,0)`. Before `x`, `y`
is illegal at distance sixteen from `(0,0)`; after `x`, it is legal at
distance exactly eight from `x`. Radius eight is inclusive and legality is
color-blind. All coordinates are distinct, cadence is `P0;P1,P1`, and no
prefix can win because Player1 has at most two stones and Player0 one.

**G2-O1. PROVEN obstruction.** Any “complete” quiet generator restricted to
pairs of turn-start legal cells omits the legal turn `[(8,0),(16,0)]`.
Locality pruning is also unavailable: the previously frozen remote-block
witness has a unique winning completion at distance six outside every live
attacker window. The next round must enumerate full engine legality, using
the quotient only for proven commutations.

## 3. Zone validity at an unforced AND node

### 3.1 Non-negotiable certificate conditions

Let `N` be a non-opening defender node with remaining budget `b` and current
threat transversal number `k<b` (taking no live attacker threat as `k=0`). A
restricted Universal proof is valid only after all of the following hold:

1. a finite D9 certificate grammar with path-derived cadence, typed terminal
   leaves, exact resolution indices, and no defender-terminal proof edge;
2. an independently nonempty searched set `S(N)`;
3. one exact legal, nonterminal child proof for every `d in S(N)`;
4. D10's union of every reachable attacker-placement, leaf-witness, and
   checkpoint role over those exact child proofs;
5. D14's local resolution budget and every LOSS remainder/horizon check;
6. searched coverage of `Z_dir union Z_seed union Z_touch union Z_virgin`,
   with `(Z2)`, `(Z4)`, and the relevant legality-frontier proof;
7. final independent verifier reconstruction from the frozen children.

Rank is an ordering key only. It never excuses an omitted mandatory edge.
At a b=2 turn the two placements remain two cadence nodes; same-turn defender
commutation stays disabled on zone nodes, as in G2R3.

The new Phase-1/2 results refine condition 6 only on their proved classes:

- **FHW-T1 (PROVEN-ON-CLASS):** exact protected tight gates may use paired
  `f/Q^D` clocks with the off-kernel `b` floor.
- **FHW-T2 (PROVEN-ON-ANNOTATED-CLASS):** a tight gate may map kernel replies
  through a frontier-covered representative while retaining the exact
  branch-paired `F+H_W` interpretation.
- **FHW-T3-R (PROVEN-ON-ANNOTATED-CLASS):** finite role/window danger cuts
  give a separately named target-local cut clock using the disjoint
  `kappa_cut^*` decision tree. Every D-alive `d in W` pays one; an all-empty
  direct edge also requires `1+q<6`. It is not relabelled as global `F+H_W`.
  Both T2/T3-R keep full scalar `B`, roles, LOSS clauses, and escape horizons.
- **FR-T1 (PROVEN-ON-CLASS):** at ordinary global-zone nodes whose descendants
  use ordinary or protected exact-copy gates, the scalar seed band may be
  replaced by the finite backward support-reach set `SR`.

They are not interchangeable. The unforced parent itself is an ordinary
full-cost defender opportunity. FHW-O2 proves that slack pressure `k<b`
alone gives **no generic per-window debit**: with `k=1,b=2`, the defender can
play a quiet first placement and spend the last placement during the tight
gate's escape. Combining `SR` with arbitrary D17/D22 mixed substitutions is
OPEN. Therefore the build must attach a class tag and sufficient verifier
data to every refined node; otherwise it uses the landed uniform G2R3 zone.

### 3.2 Frozen-certificate closure

A zone depends on the certificates of the children it is meant to restrict.
That circularity has a finite constructive resolution.

Fix parent position `P`, finite `L=Legal(P)`, and an initial nonempty
`S_0 subseteq L`. For every `d in S_i`, freeze one exact successful child
proof `C_d`; do not later swap it for another proof. Let

```text
Sigma(S_i) = (
  1 + max_{d in S_i} B(C_d),
  union_{d in S_i} Prot(C_d),
  all per-role/per-window labels required by the selected zone class
).
```

Let `Zone(P,Sigma)` be the independently specified uniform, FHW, or `SR`
selector, required by construction to satisfy `Zone(P,Sigma) subseteq L`
(equivalently intersect its output with the fixed parent legal set). Iterate

```text
S_{i+1} = S_i union Zone(P,Sigma(S_i)),
```

proving and freezing every newly added child before recomputing the summary.

**G2-Z1 (finite inflationary closure). PROVEN sound-on-success.** Each strict
round adds a previously absent member of finite `L`, so the process terminates
after at most `|L\S_0|` additions. If every newly required child proves and a
successful fixed point `S_*` is reached,

```text
Zone(P,Sigma(S_*)) subseteq S_*.
```

Together with the frozen exact child proofs and the unchanged T3/T4 class
hypotheses, this licenses the final Universal certificate. A refuted child or
resource failure ends this proof candidate fail-closed; termination of set
inflation does not promise success.

*Proof.* Inflation and finiteness prove termination. At termination the set
difference is empty by definition. The final certificate uses exactly the
same child proofs which produced `Sigma(S_*)`, so the relevant zone theorem
applies without a finder/verifier summary mismatch. Raw `Zone` need not be
monotone: a deterministic fallback may disappear when a substantive
component becomes nonempty. Retaining old edges via union makes that
irrelevant. ∎

**Completeness caveat (PARTIAL).** Closure is sound when it succeeds, but is
not complete over alternative child certificates. One frozen proof can expose
a wider role union than another; a newly required child may then refute that
proof plan even though a different global selection would close. Searching
the product of alternative child proofs is plausibly exponential and remains
OPEN. Mutable re-selection during closure is forbidden because it can
oscillate and make materialization disagree with the summary.

### 3.3 Why one pass is insufficient

Adding a child can increase `B`, introduce a new attacker move or leaf empty,
or expose a checkpoint role. Any of those can add mandatory cells on the next
derivation. Thus “derive once from the first proved children” has no theorem.
The existing recursive `zone_certificate_extras` loop already embodies the
right fixed-point shape; the native round must preserve its append/freeze
semantics, not merely call its helper once.

## 4. Native PN state machine

Use the existing convention `Proven=(0,INF)`, `Refuted=(INF,0)`, with Choice

```text
pn = min child.pn,       dn = sum child.dn,
```

and Universal

```text
pn = sum child.pn,       dn = min child.dn.
```

Every displayed addition means

```text
inf_add(a,b) = min(PN_INFINITY, a.saturating_add(b));
inf_sum      = fold(inf_add),
```

and likewise for `dn`. The engine sentinel is below `u32::MAX`; plain integer
saturation is insufficient because `PN_INFINITY+1` must remain the sentinel.

The two missing-universe obligations can be expressed algebraically, but
must be control state rather than fake game moves.

### 4.1 Hidden quiet Choice debt

A hidden quiet tranche contributes the non-selectable debt `(INF,1)`:

```text
enum QuietState { NotApplicable, Hidden, Revealed }

recompute Choice:
    pn = min(concrete child pn), default INF
    dn = inf_sum(concrete child dn)
    if quiet_state == Hidden:
        dn = inf_add(dn, 1)
```

**G2-PN-OR (hidden-universe algebra). PROVEN.** A concrete forcing proof still
sets `pn=0`, while forcing-only exhaustion cannot set `dn=0`. Removing the
debt only after the exact quiet supplement is installed restores the ordinary
Choice recurrence.

Reveal is an event with priority over threshold return and ordinary child
selection:

```text
if quiet_state == Hidden:
    if some concrete child is genuinely proven:
        allow Choice proof
    else if all concrete forcing children are genuinely refuted,
            or final semantic depth has no genuinely selectable forcing child:
        actions = exact quotient from §2
        normalize against already represented complete-turn actions
        append every missing action in deterministic order
        only after complete enumeration set quiet_state = Revealed
        recompute
```

Reveal is sound at any time; the condition only preserves forcing-first
scheduling. A DepthCutoff at an intermediate stage deepens normally. At final
semantic depth it may trigger reveal because it is not selectable for further
progress, but it is never called a genuine refutation; a retained cutoff flag
still prevents an external hard-loss result. Resource failure leaves `Hidden`
intact and returns Unknown. An empty forcing vector reveals immediately; it
is not a refuted Choice node.

### 4.2 Open unforced-Universal debt

An open zone closure contributes non-selectable debt `(1,INF)`:

```text
enum ZoneClosure {
    None,
    Open { generation, frozen_plans },
    Closed { zone_info, frozen_plans }
}

recompute Universal:
    pn = inf_sum(concrete child pn)
    dn = min(concrete child dn), default INF
    if zone_closure is Open:
        pn = inf_add(pn, 1)
```

**G2-PN-AND (closure-debt algebra). PROVEN.** A concrete child refutation
still yields `dn=0`; even if all current children prove, `pn>=1` prevents a
premature Universal proof. After a successful fixed point, removing the debt
permits `pn=0`. In particular, an open node with no children has `dn=INF`,
not the current empty-Universal default zero.

Current `DepthCutoff` also has numeric pair `(INF,0)`. Therefore, while
`ZoneClosure::Open`, numeric `dn=0` is **not** a hard refutation unless
`child_is_genuinely_refuted` identifies a concrete refuting child. The closure
gate/genuine-refutation check must precede every hard-number exit, including
the root/result checks in `run_until` and `work`, not only df-pn threshold
crossing and child selection. An Open node whose only children are cutoffs
deepens or returns Unknown; it never enters `Refuted`.

The closure event also precedes threshold return and selection:

```text
if zone_closure is Open:
    if any concrete child is genuinely refuted:
        refute this restricted claimant proof candidate
    else if any concrete child is DepthCutoff/resource-unknown:
        deepen if possible, otherwise retain Open and return Unknown
    else if every concrete child has pn == 0:
        freeze the exact materialization plan for every child
        sigma   = derive exact parent summary from those plans
        missing = Zone(parent_state, sigma) - explicit_moves

        if missing is empty:
            store exact zone_info and sigma binding
            atomically mark Closed
        else:
            append every missing legal move in deterministic order
            increment generation; remain Open
        recompute
```

Required invariants:

1. append only; never remove/reorder old children or silently swap a frozen
   proof plan;
2. each added edge uses the exact parent position and per-placement cadence;
3. a defender-terminal edge refutes the claimant's Universal candidate;
4. summary/resource failure keeps the debt and returns Unknown;
5. closing and storing its summary are atomic—there is no transient debt-free
   state before all missing children exist;
6. materialization requires `Closed`, emits `zone:Some(...)`,
   `implicit_dispatch:false`, and no same-turn commutations, then submits the
   whole certificate to the unchanged strict verifier;
7. a TT or shared-fragment record may advertise a proven unforced node only
   with the closed-zone binding and frozen materialization plan. Hidden/open
   debt is search state, not a cacheable legal child.
8. every `pn/dn` hard-verdict site consults closure state and genuine child
   status before interpreting the numeric sentinel.

### 4.3 Native end-to-end theorem

**G2-NATIVE (native λ² soundness). PROVEN-ON-DESIGN-CLASS.** Assume:

1. §2's exact quiet enumeration and terminal handling;
2. ordinary PN child correctness;
3. non-selectable debts and event priority as above;
4. immutable child proof plans during §3's inflationary closure;
5. one of the explicitly valid zone classes in §3.1; and
6. fail-closed resources plus final independent verification.

Then every native-PN `pn=0` materializes a valid certificate, and no
unforced Universal can prove before its certificate-dependent zone is closed.

*Proof.* At a Choice, hidden width cannot contribute a selectable proof, but
its positive `dn` prevents an incomplete disproof; a concrete existential
proof is sound without revealing unused alternatives. At an unforced
Universal, the positive `pn` debt prevents proof until G2-Z1 reaches a fixed
point, while a concrete refuting defender reply remains decisive. Once debts
are removed, the children are exactly an ordinary complete Choice universe or
a T3-licensed searched Universal set. Induct on materialized children and
apply the selected zone theorem at each closed Universal. The strict verifier
reconstructs every binding. ∎

This proves the proposed state-machine contract, not its implementation or
progress rate. Both remain **DEFERRED-NEEDS-CARGO**.

## 5. Build sequence — all DEFERRED-NEEDS-CARGO

No step below was executed in this campaign.

### N0 — freeze baseline and fixtures

- Record the input-HEAD digests for flags-off default tests, the 19-row
  forcing gate, `double_fork_compact`, and both honest UNKNOWN controls.
- Port the remote-block replay in §6.2 into a checked corpus fixture only
  after the current strict verifier independently reaccepts it.
- Preserve the current recursive G2R3 path as the semantic A/B reference; do
  not delete `prove_narrow_compat` in this round.

### N1 — pure turn enumerator

Add a side-effect-free helper which emits §2's normalized `Single`, `Win1`,
`StartPair`, `TerminalSecond`, and `FrontierPair` actions. `Single` is
mandatory for every legal Opening/SecondStone placement, including a
nonwinning quiet completion such as P2's `(6,-6)`. The helper may reuse engine
legal reconstruction and coordinate ordering, but not a locality cap.

Required tests:

1. exact raw-history set equality against brute ordered apply/undo enumeration
   on every registered state, followed separately by equality of normalized
   quotient keys and game-semantic signatures (owner-labelled board,
   nonterminal next player/cadence, or terminal winner)—not literal
   `last_turn`/terminal `PositionKey` equality;
2. no duplicate normalized action and no missing Opening/SecondStone
   `Single`;
3. D6 covariance of unordered quotient keys and game-semantic outcomes;
   literal materialization order is compared only when the stabilizer does
   not swap the endpoints, otherwise both orders are retained or accepted as
   one unordered class;
4. the `(0,0);(8,0),(16,0)` dynamic fixture is present as `FrontierPair`;
5. singleton terminal prefixes have no second child;
6. the existing verifier commutation condition matrix remains green.

The helper lands test-only first. Shadow mode compares it with the recursive
G2R3 fallback before native PN consumes it.

### N2 — native Choice reveal

- Extend native Branch state with `QuietState`; keep the existing forcing
  children and priors unchanged.
- Modify `recompute` at the current Choice recurrence seam to add only the
  hidden `dn` debt.
- Insert the reveal event before df-pn threshold crossing/selection.
- Normalize complete-turn actions against existing `WidePnMove::Pair/One`
  children; append deterministically, then clear debt.
- Keep a distinct “genuine refutation versus DepthCutoff” predicate. A
  resource/cutoff flag must prevent an external hard-loss result even if the
  restricted internal `dn` reaches zero.

Stage `Off -> Shadow -> Verify -> Consume`. Off must be byte-identical.
Shadow records the supplement without changing children. Verify compares raw
sets and replays every action. Consume changes native scheduling but still
uses the unchanged certificate verifier.

### N3 — native Universal closure

- Extend Universal Branch state with `ZoneClosure` and the open `pn` debt.
- Seed `S_0` by the existing deterministic ranked candidate policy plus an
  independent legal fallback; never start with an empty closed node.
- When all current children prove, freeze exact materialization plans. A
  practical implementation may snapshot selected child-plan IDs plus an
  epoch, or materialize them into a closure-owned certificate arena. It must
  not later ask a mutable PN entry to choose a different proof.
- Derive finder-side D10/D14 data from that frozen arena using a finder helper;
  keep `verifier_zone_summary`/`verifier_uniform_zone` independent.
- Append `Zone-S` in generations until fixed. Store the final summary digest,
  zone class, build-horizon binding, and explicit coordinates before clearing
  debt.
- Teach native `WideCertBuilder::build_universal` to require `Closed` and emit
  the exact stored zone metadata. The verifier remains the acceptance oracle.

Synthetic tests must force at least two closure generations, a disappearing
fallback, an added terminal defender reply, an intermediate and final
DepthCutoff, resource failure, and a summary mutation. Any `Open` node
materialized or cached as proven is a hard failure.

### N4 — exact-zone shadow lanes

After N1–N3 are semantically green, add independent default-off selectors:

```text
UniformG2R3
ExactRoleWindow              // D15/D16 labels
FhwExactOrD22                // only verified Phase-1 class annotations
SupportReach                 // only FR-T1 class, no mixed substitution
```

The first implementation is **shadow only**. Each refined set must be a
subset of the applicable uniform set and must independently reverify when
consumed. If a certificate does not satisfy a class gate, record
`INELIGIBLE`, not zero savings and not a guessed debit.

`FhwExactOrD22` must encode the FHW-T3-R decision-tree row selected for every
`(edge,W)` and reject any all-empty direct edge with `1+q>=6`; it may not
implement the withdrawn overlapping list with source-order precedence. Every
N4 result also binds the two authority digests in the R-Z11 repair record.

### 5.1 Required telemetry schema

Emit deterministic, machine-readable records keyed by position digest,
node ply, phase, and certificate node ID.

At Choice:

```text
phase, single_count, |L_0|, |I|, raw_pair_count,
start_pair_count, terminal_second_count, frontier_pair_count,
forcing_actions_already_present, quiet_actions_added,
normalization_duplicates, reveal_count, reveal_stage, debt_before/after
```

At unforced Universal:

```text
b, k, B, |Legal|, |S_0|, closure_generations,
added_per_generation, |S_*|, role_count,
dir/seed/touch/virgin union sizes and overlaps,
uniform/exact/FHW/SR sizes and eligibility,
open_debt_events, frozen_plan_digest, finder_summary_digest,
verifier_summary_digest, strict_verdict
```

At solve level:

```text
status, nodes, wall_ms, TT_hits, peak_TT_bytes, horizon,
certificate_nodes/edges, strict_verifier_result,
count(k<b_current Universals), count(k<B_local Universals),
nonforcing_OR_edges, max_spare_turn_nesting
```

The actual λ²-node count is `k<b_current`; the plan's broader certificate proxy
is `k<B_local`, and both are reported to prevent overcounting a tight current
gate whose descendant budget is larger. Together with nonforcing OR edges and
spare nesting they form the preregistered λ-order columns. Component counts
must report both raw sums and deduplicated union; otherwise the G2R3
`19+50 -> 62` overlap is easy to misread.

## 6. Pre-registered measurement campaign — DEFERRED-NEEDS-CARGO

**Status:** design complete; executions `0`; observations from this campaign
`0`. Historical numbers below select bars but are not new measurements.

### 6.1 Cohorts fixed before implementation results

| cohort | membership | purpose | acceptance interpretation |
|---|---|---|---|
| F19 | the existing 19-entry forcing gate | dispatch regression and overhead | features should not fire; verdict/certificate invariants must hold |
| P1 | checked-in `double_fork_compact` | primary λ² positive | strict WIN historically 409 nodes at 10k |
| P2 | frozen remote-block replay below | structurally different quiet positive | must first reverify on input semantics; then strict WIN historically below 10k |
| C2 | `compact_urgent_spare`, `strongloss_a_backoff_7` | honest no-positive controls | UNKNOWN is expected; these are not LOSS oracles |
| M | dynamic-pair fixture plus synthetic debt/closure fixtures | exact state-machine coverage | set equality and invariant assertions, not solve speed |
| H1152-B | deterministic human-corpus unforced-node benchmark | fixed-key regression and materiality distributions; **not prevalence** | first 384 canonical states in each `(b,k)=(2,0),(2,1),(1,0)` stratum |
| A5 | first five opening-atlas families from plan §I.6 | capstone spot-check | descriptive; honest UNKNOWN allowed |

The H1152-B source is frozen as
`E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl`,
3,696,030 bytes, SHA-256
`54FAE7AEBCEF2A9D19D13C1946FAE36C0565E21BC726C25E2E4E230CFB42A5B7`
(6,902 games). N0 rechecks this digest; a mismatch stops sampling rather than
silently selecting from a new population.

`H1152-B` membership is an outcome-blind deterministic benchmark rule: replay every qualifying prefix,
D6-canonicalize, retain the least prefix per `(game,stratum)`, collapse equal
canonical keys to the lexicographically least `(game_index,prefix_length)`,
sort by the full canonical bytes with that tuple as tie-break, and take the
first 384. The collision-free canonical key is the lexicographically least of
the 12 D6 encodings of

```text
(current_player:u8,
 phase_tag:u8,
 transformed FirstStone coordinate when phase=SecondStone,
 placements_made:u32,
 sorted (q:i16,r:i16,owner:u8) occupied cells).
```

Integers use fixed-width big-endian two's-complement bytes after flipping the
signed coordinate high bit for lexical numeric order. Membership/dedup uses
these full bytes; SHA-256 of them is display telemetry only. If a
stratum has fewer, take all,
publish the shortfall. No result from this selection estimates corpus or
human-play prevalence, regardless of accepted-node count. The 100-node
threshold below is only a fixed-benchmark materiality denominator.

A prefix qualifies iff it is legal, nonterminal, non-Opening, the current
player has no `own_win_now`, and its current attacker-threat family has
`k<b_current`, with `k=0` when `opp_threat_count=0` and otherwise the exact
finite minimum transversal. A missing/greater-than-budget transversal is not
put in a `k=0/1` stratum.

At every sampled root set `claimant=current_player.other()` and use the
test-only fixed-claimant path (equivalently `SolveGoal::Loss` relative to the
defender-to-move root), not public `Both`. Require strict verifier acceptance
for every positive counted as a certificate node. The fixed H1152-B solve
profile is: semantic horizon `root_placements+10`, node cap `100,000`, TT cap
1 GiB, `TSS_LAZY_FRONTIER=1`, one test thread, and every optional consuming
refinement off except the single lane under comparison. Base native G2 quiet
reveal and uniform-zone closure are always Consume for H1152-B; Exact, FHW, and
SR are compared one at a time, with the other two off. The first three newly
discovered strict quiet wins in digest order may join the capstone; the
denominator and all failed candidates remain reported.

### 6.2 Frozen positive witnesses

P1 is not newly constructed here; it is the checked-in engine corpus replay
at `packages/hexfield_eq/rust/src/tss_spare_corpus.rs:262-299`:

```text
[(0,0),(-1,0),(4,1),(1,0),(2,0),(4,2),(4,3),(3,0),
 (4,6),(4,4),(4,5),(1,3),(2,3),(2,1),(5,5),(3,3),
 (0,4),(6,2),(-1,5),(0,5),(0,6),(7,6),(1,6),(5,7),
 (6,7),(6,6),(3,6),(7,7),(5,6),(-1,6),(1,4),(6,5),
 (7,4),(7,3),(7,5),(-1,2)]
```

Its landed corpus record already checks engine replay legality and strict
certificate verification. After 36 placements Player0 is at SecondStone;
the intended completion is `(4,0)`, followed by an unforced defender node
`b=2,k=1`. Historical acceptance is `WIN/409`, horizon 45, with the two zone
ratios quoted in §1.

P2 comes from the frozen, later-deleted proof artifact
`833020ed:PROOF_QUIET_LOCALITY.md` and is not yet a current corpus row:

```text
[(0,0),(-1,0),(1,-1),(1,0),(2,0),(2,-2),(3,-3),(3,0),
 (4,6),(4,-4),(5,-5),(1,3),(2,3),(2,1),(5,5),(3,3),
 (0,4),(6,2),(-1,5),(0,5),(0,6),(7,6),(1,6),(5,7),
 (6,7),(6,6),(3,6),(7,7),(5,6),(-1,6),(1,4),(6,5),
 (7,4),(7,3),(7,5),(6,0)]
```

After 36 placements Player0 is at SecondStone. The unique surviving
completion is `(6,-6)`: it is legal, distance six from the nearest attacker
stone, and outside every live attacker window. The other 537 legal moves let
Player1 immediately complete `(1,-1)..(6,-6)`. The frozen record reports a
strictly accepted 3,858-node parent certificate, child search 4,957 nodes,
and horizon 66. Because that artifact is not at input HEAD, N0 must replay and
strictly reverify it before P2 becomes an acceptance bar. Failure is reported
as provenance drift, never edited away.

These are corpus/frozen-source witnesses, not new hand positions. The only
new coordinate construction in this design is §2.2, whose cadence, inclusive
radius-eight legality, coordinate distinctness, and no-win prefixes were
checked explicitly.

### 6.3 Rung ladder and profiles

Run in this order after the cargo slot is released:

1. pure unit/property/mutation tests;
2. P1 and provisionally P2 at `10k -> 100k -> 1M`, stopping at the first
   strict-verifier-accepted positive but retaining every earlier record;
3. C2 at `10k -> 100k -> 1M`; UNKNOWN at 1M is an honest stop;
4. F19 at the plan's `10k -> 100k -> 1M -> 20M` ladder, with existing NO
   rows stopping at 1M;
5. H1152-B shadow telemetry at the `root+10`, 100k-node, 1 GiB lazy-on
   fixed-claimant profile frozen in §6.1, then only the digest-first discovery
   positives through the solve ladder;
6. A5 using the plan's recommended 1 GiB lazy-on deep profile; repeat
   headline comparisons under the legacy 2 GiB flags-off profile;
7. for every accepted capstone positive, run the outward-frontier robustness
   adapter below; and
8. owner-deferred cross-solver rows on P1, revalidated P2, and the digest-first
   human positives at matched position, horizon, and rung.

Use one test thread and record the exact compiler, target triple, environment,
semantic horizon, TT cap, and input commit. Wall comparisons use the median of
three clean repetitions; node/certificate counts come from the first accepted
run and must agree across repetitions. No rung may be silently raised after
seeing a miss.

The outward adapter is deterministic and test-only. Define

```text
Legal_R(P) = { empty c : dist(c,St(P)) <= R },
R = 9,
```

with the singleton-origin Opening unchanged. It bypasses only the production
radius-eight admission check; ownership, placement cadence, window updates,
and per-placement wins are unchanged.

The old "replace every 8 by 9 and rederive the same zones" check is **DEAD as
proof**. It may be retained only as labelled stress telemetry. The replacement
robustness oracle is an exhaustive bounded policy model checker over the same
materialized certificate:

1. begin at the certificate root and enumerate **every** `Legal_9` defender
   placement at every defender cadence state, including dynamically new
   second placements;
2. replay explicit certificate edges when present; for an omitted reply,
   execute the certificate's declared coupling/filler/escape policy but do
   not use any radius-substituted zone-containment lemma to dismiss it;
3. recursively enumerate all later `Legal_9` defender replies through the
   certificate's fixed absolute horizon, checking phase, ownership, legality,
   and termination after every placement;
4. at attacker nodes replay only the certificate's designated move or its
   finite LOSS/gate-escape adaptive choice, and reject if that move is
   illegal or the declared witness does not resolve; and
5. PASS only if every enumerated defender branch avoids a defender win and
   reaches a verifier-checked attacker win by the original horizon. Emit the
   first failing full placement trace and counts of positions, legal replies,
   and newly radius-nine-only replies.

Finiteness follows from the fixed horizon and the finite legal halo of every
finite position. Resource exhaustion is `INCONCLUSIVE`, never PASS. Failure
or inconclusive blocks only a capstone **outward-frontier robustness claim**,
not production radius-eight soundness or native Consume promotion. This
model check is SPEC-FOR-CARGO in section 6.7.

The owner-deferred cross-solver table fixes solver/version, claimant, position
digest, horizon, rung, TT/memory allowance, status, verified-certificate
availability, nodes, and wall time. Where an implementation lacks a matched
semantic control, report `NOT-COMPARABLE`; do not substitute an easier row.

### 6.4 Hard kill criteria

Any item below stops Consume promotion, returns the feature to Shadow, and
requires a written obstruction or repair before more performance tuning:

1. a produced positive certificate is rejected by the strict verifier;
2. the pure quiet enumerator misses a brute legal turn, admits an illegal
   turn, mishandles per-placement termination, or has a normalized duplicate;
3. a Hidden Choice reaches hard `dn=0`, or an Open Universal reaches `pn=0`;
   an Open Universal with only cutoff/resource-unknown children also must not
   enter `Refuted` or return a hard result despite its numeric `dn=0`;
4. a debt appears as a move/certificate node or an open node enters a proven
   TT/shared-fragment record;
5. closure repeats a set, removes an edge, changes a frozen plan, exceeds
   `|Legal\S_0|` strict additions, or materializes before fixed point;
6. finder and verifier disagree on roles, `B`, zone, horizon, or class tag;
7. a defender-terminal edge, wrong mover/budget, wrong absolute horizon, or
   radius-eight legality error survives verification;
8. any old seven-way mutation or the new early-debt-clear, plan-swap,
   dynamic-edge-drop, or illegal-commutation mutation is accepted;
9. Off mode differs byte-for-byte from the frozen baseline;
10. on the same frozen certificate and role set, an exact/FHW/`SR` mandatory
    set is not a subset of its applicable full-clock uniform set;
11. a D6 image changes legal-action coverage or a hard verified verdict;
12. a node/resource/summary failure produces a hard verdict instead of
    Unknown; or
13. either C2 control becomes WIN under the feature before independent strict
    verification and an explicit control reclassification.

Internal `dn=0` means refutation of the restricted claimant proof candidate,
not a proved Hexo loss. External hard losses still require the existing loss
certificate contract.

### 6.5 Promotion bars fixed now

Native PN may replace the recursive compatibility route only if all are true:

1. P1 and the revalidated P2 both produce strict accepted WIN at the first
   10k rung; P1 must expose at least one closed `k<b` node and P2 at least one
   revealed quiet edge;
2. each native positive uses no more than `2.00x` the nodes of its frozen
   recursive baseline and stays within the same semantic horizon;
3. F19 hard verdicts and certificate verification match; quiet reveals and
   unforced-zone closures are exactly zero there. With rows `i` and clean
   repetitions `r`, require

   ```text
   sum_i nodes_native(i) / sum_i nodes_baseline(i) <= 1.10,
   median_r [sum_i wall_native(i,r) / sum_i wall_baseline(i,r)] <= 1.10,
   max_i [median_r peak_native(i,r) / median_r peak_baseline(i,r)] <= 1.10.
   ```

   Publish every per-row delta; exact node identity is required only in Off
   mode, not after changing the native scheduler;
4. C2 yields no unverified positive through 1M;
5. all mutation, dynamic-legality, debt, closure-generation, D6, and
   flags-off identity gates pass; and
6. every materialized unforced node has `required subseteq explicit`, no
   commutations, a frozen-plan digest, and finder/verifier summary equality.

Exact-zone consumption is a separate decision. Its comparison universe is
the canonical finite index `J_zone`. Before evaluating any variant value,
freeze one strict-verifier-accepted final certificate and create one key for
each applicable ordinary zone node:

```text
J_zone key = (
  certificate_digest, certificate_node_id, position_digest,
  owner, phase, remaining_budget, child_plan_digest,
  sorted_live_role_set_digest, B_and_horizon_digest,
  finder_summary_digest
).
```

The node ID is assigned by deterministic preorder of the frozen certificate
with child edges sorted by canonical move key; a DAG node is keyed once and
also records its incoming-path count. `S_uniform(j)` and `S_variant(j)` are
the mandatory certificate sets (not heuristic supersets) recomputed from the
same frozen children, roles, clocks, and horizon at exactly that key.
Eligibility is decided solely from the frozen class tag and proof data before
either set size is read. Publish eligible, ineligible, and unmatched counts;
any missing key, changed child/role/summary digest, or nonidentical key set is
a hard comparison failure, not an exclusion. Deduplicate coordinates within
each node and define

```text
U       = sum_{j in J_zone} |S_uniform(j)|,
X       = sum_{j in J_zone} |S_variant(j)|,
G_total = 1 - X/U,                         with U>0.
```

Component gains use the same ratio of sums on their explicitly named eligible
index; they never substitute for `G_total`.

For the FHW clock ratio, construct a second canonical finite index `I_FHW`
from those same frozen certificates. Start uniform-verifier window queries at
every node in `J_zone`: all D-alive touched windows, plus every all-empty
window in the conservative finite virgin superset

```text
{ W : exists c in Legal(P_N), B(N)>=6,
      d(c,W) <= 8(B(N)-6) }.
```

Propagate each queried `W` through the old D16 recurrence to every descendant
gate until its ordinary stop. For every FHW-eligible gate reached, add

```text
I_FHW key = (certificate_digest, gate_node_id, owner,
             window_direction, window_origin).
```

`window_direction` is one of the three unoriented axial line directions and
`window_origin` is the lexicographically smaller endpoint, so each length-six
mask has one name. Deduplicate on the full key and store a touched/virgin
source bitmask rather than duplicating a key. Gate eligibility is the
verifier's FHW-T1/T2/T3-R class verdict and is frozen before clock values are
read. `E_old` and `Q_new` are then evaluated on **exactly `I_FHW`**; no
implementation-emitted query subset, gain-dependent union/intersection, or
post-result exclusion is allowed. Report gate count, key count, exclusions,
and a positive finite old denominator.

For FHW `G_total`, use the mechanically derived subset `J_zone^FHW` consisting
of exactly those `J_zone` nodes whose propagated uniform query set reaches at
least one key in `I_FHW`. Freeze that subset with the two indices before any
size is read; do not substitute all nodes or only positive-gain nodes.

- **Exact D15/D16:** require at least 100 accepted unforced nodes in each
  populated H1152-B stratum and `G_total>=0.10` versus the uniform wrapper.
  Otherwise publish a NULL materiality result and keep it default-off.
- **FHW:** require at least 30 distinct eligible
  `(certificate_digest,gate_node_id)` gates and at least a 10%
  clock reduction

  ```text
  1 - sum_{i in I_FHW} Q_new(i)
        / sum_{i in I_FHW} E_old(i) >= 0.10,
  ```

  with a positive denominator, **and** `G_total>=0.10` recomputed on
  `J_zone^FHW`. Ineligible gates are excluded with counts.
  The strict-debit worked fixture must reproduce `3/2=1.50x`.
- **Support reach:** require at least 30 nodes with a nonempty scalar seed
  component,

  ```text
  1 - sum_j |Z_seed^SR(j)| / sum_j |Z_seed_scalar(j)| >= 0.10,
  ```

  and `G_total>=0.10` on those same nodes. P1's two `Z_seed=0` nodes contribute
  neither success nor failure to this bar.

A refined lane that passes its semantic/materiality bar is not yet allowed to
Consume. On the identical matched root/profile jobs used for that lane, with
three clean repetitions and the same binary, horizon, TT cap, and frozen
cohort membership, it must also satisfy

```text
sum jobs nodes_variant / sum jobs nodes_uniform <= 1.10,
median_r(sum jobs wall_variant_r / sum jobs wall_uniform_r) <= 1.10,
max_job(median_r peak_variant / median_r peak_uniform) <= 1.10.
```

Hard statuses and strict verification must match. Failure is a valid
material-but-uneconomic result and keeps the lane in Shadow; it does not
change `J_zone`, `I_FHW`, or the 10% semantic threshold.

For every cohort, publish per-node and aggregate `|searched|/|Legal|`, median,
p90, p95, maximum, component overlap, uniform/exact delta, and the λ-order
proxy. There is no preregistered favorable ratio threshold for the descriptive
capstone headline; suppressing an unfavorable tail is forbidden. Cross-solver
wall/node comparison remains the plan's owner-deferred mission, is recorded by
§6.3 step 8, and is not a gate for this native correctness round.

### 6.6 Required future records

The future run should create, without rewriting this preregistration:

```text
.codex-group2-next/BASELINE.md
.codex-group2-next/profiles-v1.json
.codex-group2-next/radius9-v1.json
.codex-group2-next/telemetry.jsonl
.codex-group2-next/MUTATIONS.md
.codex-group2-next/GATE.md
.codex-group2-next/FRONTIER.md
.codex-group2-next/CROSS_SOLVER.md
.codex-group2-next/OBSTRUCTIONS.md   // present even if it says none
```

Every record names the exact commit and command. A failed bar is a first-class
result; it is not repaired by deleting the cohort or changing the rung.

### 6.7 Exact cargo/measurement spec

This section is the executable specification for every item labelled
SPEC-FOR-CARGO. The implementation must expose exactly two ignored test
harnesses:

```text
tss_group2_next::group2_next_gate
tss_group2_next::radius9_exhaustive_gate
```

Before any comparison result is inspected, write
`.codex-group2-next/profiles-v1.json` containing the authority digests,
implementation commit, compiler/version/target, position or corpus digest,
claimant, semantic horizon, node rung, TT bytes, every feature flag, lane,
and repetition number. Hash that manifest and copy its digest into every
telemetry row. The only permitted measurement commands from the worktree root
are:

```powershell
$env:CARGO_TARGET_DIR='.target-group2-next'
$env:TSS_GROUP2_PROFILE_MANIFEST='.codex-group2-next/profiles-v1.json'
cargo test --release -p hexfield_eq tss_group2_next::group2_next_gate -- --ignored --exact --test-threads=1 --nocapture

$env:TSS_GROUP2_R9_MANIFEST='.codex-group2-next/radius9-v1.json'
cargo test --release -p hexfield_eq tss_group2_next::radius9_exhaustive_gate -- --ignored --exact --test-threads=1 --nocapture
```

The first harness runs these frozen profiles, always matching baseline and
variant within the same release binary and repetition:

| profile | semantic horizon | node rungs | TT cap | repetitions / flags |
|---|---:|---|---:|---|
| P1 `double_fork_compact` | 45 | `10k,100k,1M` stop at first accepted WIN | 1 GiB | 3 clean; lazy on; recursive baseline versus native uniform; refined lanes one at a time |
| P2 exact 36-move replay in section 6.2 | 66 | `10k,100k,1M` stop at first accepted WIN | 1 GiB | 3 clean after mandatory strict revalidation; same lane rules as P1 |
| C2 two named rows | `placements_made+reference_plies` (both checked-in rows declare `reference_plies=2`) | `10k,100k,1M` | 1 GiB | 3 clean; UNKNOWN is retained; no unverified positive |
| F19 checked-in forcing corpus | `u32::MAX`, matching the existing corpus harness | `10k,100k,1M,20M`, NO rows stop at 1M | 2 GiB | 3 clean; exact official row set; quiet/closure fires must be zero |
| H1152-B | `root_placements+10` | 100k | 1 GiB | 3 clean; fixed claimant; lazy on; native uniform, Exact, FHW, SR one at a time |
| A5/discovery capstone | manifest-fixed per root before first run | `10k,100k,1M,20M` | 1 GiB, plus one matched legacy 2 GiB repeat | 3 clean; descriptive tail retained |

For all rows, one test thread is binding; process start is clean; no warm TT is
shared between repetitions; node and certificate counts must agree across the
three repetitions; wall is the median; peak is the median per job. Any
profile field absent from the manifest, any environment variable not recorded
there, any rung increase after a miss, or any baseline/variant manifest digest
mismatch invalidates the bar. Off mode is separately compared byte-for-byte
with input `ad606d0e` artifacts; performance comparisons use matched lanes in
the post-implementation binary.

The second harness consumes only strict-accepted capstone certificates listed
in `radius9-v1.json`, with their original absolute horizons and certificate
digests. It implements the exhaustive policy model check in section 6.3 with
no node/branch truncation. It may use an exact-state memo table capped at
2 GiB and a six-hour per-certificate watchdog, but either cap produces
`INCONCLUSIVE`, never PASS. A valid PASS record contains the full enumerated
position/reply counts and zero unchecked branches; a failure contains the
first complete countertrace. This is the only run that can license the
radius-nine robustness wording.

## 7. Hostile self-review of Phase 3

1. **Completeness before enumeration.** A first draft said “all quiet pairs”
   without naming dynamic second cells. Outcome: replaced by G2-Q1's finite
   raw index before any quotient.
2. **Turn-start commutation.** The sequence `(8,0),(16,0)` refutes a universe
   restricted to initially legal pairs. Outcome: dynamic pairs remain ordered
   and are explicitly tested.
3. **Terminal-prefix commutation.** Reversing a pair whose second cell wins as
   a singleton creates a nonexistent continuation after a win. Outcome:
   `TerminalSecond` is ordered; per-placement termination is binding. Joint
   second-placement wins with two nonwinning singletons remain commutable.
4. **Quiet locality.** A remote move might seem safely prunable because it
   advances no old attacker window. Outcome: rejected by the frozen unique
   defensive block `(6,-6)`; full legal fallback remains mandatory.
5. **One-pass zone summary.** A newly proved child can add a role or increase
   `B`. Outcome: G2-Z1 iterates to a fixed point.
6. **Assumed zone monotonicity.** The deterministic nonempty fallback can
   disappear. Outcome: no raw monotonicity premise; inflationary union retains
   old children and proves termination anyway.
7. **Mutable child proofs.** Re-selecting a shorter-looking proof can change
   the role union after children were added. Outcome: exact plans freeze before
   every summary and remain bound through materialization. Completeness across
   alternative freezes is honestly PARTIAL.
8. **Literal pseudo-children.** Debt values resemble PN children but have no
   legal move. Outcome: represented only as non-selectable node state and
   forbidden from certificates/caches.
9. **Scheduler stall.** If threshold return precedes a debt event, artificial
   `pn=1`/`dn=1` can be selected forever. Outcome: reveal/closure events have
   priority over threshold crossing and ordinary selection.
10. **Slack FHW overreach.** Treating `k<b` pressure as a forced-hit debit
    recreates Phase-1's legal two-placement completion. Outcome: the unforced
    parent costs one normally; only annotated tight descendants receive FHW.
11. **Premature hard loss.** `dn=0` after exhausting a restricted proof grammar
    is not a game loss, and a DepthCutoff itself carries numeric `dn=0`.
    Outcome: every hard-result site, including `run_until` and `work`, checks
    Open/Hidden state and genuine child status first; resource/depth/restricted
    misses stay Unknown absent the existing exact loss certificate.
12. **Finder/verifier common-mode error.** Sharing one summary routine would
    make agreement circular. Outcome: finder may use frozen certificate data,
    but the independent verifier retains its separate reconstruction.
13. **Capstone cherry-picking.** The forcing corpus cannot exercise λ² and
    discovered positives can bias distributions. Outcome: H1152-B membership,
    strata, denominators, rungs, null bars, and tail statistics are fixed
    before execution. Its lexicographic prefix is explicitly a benchmark;
    the old population-prevalence interpretation is DEAD.
14. **Empirical leakage into proof.** Historical runs motivated bars but no
    current execution exists. Outcome: all build and measurement claims remain
    **DEFERRED-NEEDS-CARGO**.
15. **Missing frontier robustness.** Production radius-eight acceptance does
    not imply resilience to a larger legal halo. Outcome: constant
    substitution is DEAD as proof; sections 6.3/6.7 instead exhaust every
    radius-nine defender reply to the fixed horizon. Failure or resource
    exhaustion blocks only that capstone claim.
16. **Infinite FHW sum.** Indexing all windows diverges. Outcome: `I_FHW`
    starts from the uniform verifier's finite B-bounded target superset,
    propagates those exact queries, canonically deduplicates masks, and uses
    the identical key set for old/new values.
17. **Certificate-switch materiality.** A variant could appear smaller by
    choosing a different child proof. Outcome: `J_zone` binds certificate,
    node, child plan, roles, clocks, horizon, and summary; unmatched keys fail.
18. **Direct-fill selector regression.** A source-ordered implementation could
    revive withdrawn FHW-T3. Outcome: N4 stores the FHW-T3-R decision-tree row
    and rejects every empty direct edge with `1+q>=6`.
19. **Underfixed A/B profile.** Recording flags after a favorable run permits
    drift. Outcome: section 6.7 freezes one manifest digest, exact horizons,
    commands, caps, lanes, and repetitions before results.
20. **Semantic gain without economics.** A 10% set reduction can still slow
    the solver or enlarge memory. Outcome: every refined Consume decision also
    has matched node, wall, and peak caps; a material-but-slow lane stays
    Shadow.

## 8. Caveat and open-question ledger

1. **PROVEN-ON-DESIGN-CLASS is conditional.** G2-NATIVE proves the stated
   state machine assuming correct implementation and the same T3 verifier
   hypotheses. It is not a test result.
2. **TSS versus unrestricted Hexo.** G2-Q1 is complete for one legal attacker
   turn. The whole solver remains a certificate search, not a complete solver
   for arbitrary infinite-board Hexo.
3. **Alternative-proof closure (OPEN).** A bad frozen child-certificate choice
   can make closure fail even when another combination would close. No
   polynomial or canonical selection theorem is known.
4. **Mixed frontier recurrence (OPEN).** Phase-2 `SR` is not yet proved through
   arbitrary D17/D22 substitutions. Use the uniform/FHW class fallback at such
   nodes.
5. **Scalar debit (OPEN).** No Phase-1 result reduces D14 `B`, LOSS remainders,
   or absolute escape horizons.
6. **P2 provenance.** The remote-block positive lives in a prior Git object,
   not the checked-in corpus. It is provisional until strict revalidation.
7. **Representation boundary.** Theory uses `Z^2`; the engine uses `i16`.
   All named coordinates are small, while future enumerators must continue to
   use checked engine coordinate construction near the boundary.
8. **Performance/materiality.** Native scheduling, closure-generation counts,
   exact-zone benchmark materiality, and all preregistered bars are unknown
   until the cargo slot is available.
9. **Robustness variant.** Radius-nine constant substitution is only stress
   telemetry. Only section 6.7's exhaustive bounded policy check can support
   the capstone robustness wording; failure or inconclusive forbids only that
   outward-frontier claim.
10. **Population inference.** H1152-B is not a probability sample. Its old
    prevalence interpretation is DEAD; its fixed-key regression/materiality
    use remains valid.

## 9. Design disposition

**Theory now:** G2-Q1/Q2, G2-Z1, G2-PN-OR, G2-PN-AND, and G2-NATIVE are
proved on their stated classes. Phase 1 licenses exact/FHW mixed-gate clocks,
including repaired FHW-T3-R's disjoint direct-fill charge;
Phase 2 licenses the branch-aware `SR` seed replacement on the exact-copy
class. Slack unforced pressure gives no generic debit.

**Build status:** **DESIGN BUILD-READY ON PAPER; EMPIRICAL PROMOTION
SPEC-FOR-CARGO.** The native state machine, FHW selector, canonical comparison
indices, fixtures, rungs, identities, economics gates, and kill criteria are
fixed. H1152 population prevalence and radius-nine constant-substitution proof
are DEAD. Native, materiality, performance, and exhaustive radius-nine
outcomes remain DEFERRED-NEEDS-CARGO; no implementation or execution was
attempted.

**Most valuable theorem:** G2-Z1. It turns certificate-dependent zone
circularity into a finite sound closure without assuming the zone function is
monotone.

**Sharpest next question:** Is there a verifier-checkable dominance order on
alternative child-certificate summaries that guarantees a canonical least
closing fixed point, or is choosing a closing family of child proofs
intrinsically exponential? An answer would decide whether native λ² closure
is merely sound-on-success or admits a useful completeness theorem.
