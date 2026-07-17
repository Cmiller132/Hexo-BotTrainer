# R-FIX1 diagnosis — bounded-horizon zone clocks

Status before implementation: **finder defect confirmed; verifier contract is
internally consistent and remains normative.** The NQ8 diagnosis is consistent
with the code. This memo was written before changing production Rust.

## Normative invariant

For every zoned `CertNode::Universal`, the verifier requires both fields of
`ZoneInfo` to be exact (`tss_verify.rs`, `verify_zone_node`, currently
lines 1216–1251):

1. `zone.build_horizon == cert.semantic_horizon`; and
2. `zone.d == verifier_zone_summary(..., node_id).local_budget`.

The second value is D14's budget of the *materialized certificate subtree*,
not the number of defender placements left before the caller's deadline.
`verifier_zone_summary` (currently lines 1019–1088) derives it recursively
from the proof DAG reached through ordinary certificate edges:

- `OrCompletion` and `Win`: `B = 0`;
- `Loss`: `B = placements_remaining(state)` at that typed leaf;
- `Choice`: `B = B(child)` (an attacker placement does not consume defender
  budget); and
- `Universal`: `B = 1 + max(B(edge.child))`, using saturating addition.

Commutation metadata does not add budget edges. Replay supplies the exact
state at each leaf, and the same routine also derives D10 protected roles for
the zone-coverage check. `R3_VERIFY_TRACE failure=clock` is emitted precisely
when either exact clock equality fails.

This matches the reviewed implementation contract recorded in
`docs/PLAN_TSS_SOLVER_UPGRADES.md`: the verifier independently derives D14
`B`, and a zone node requires an exact stored local `B` and exact
build-horizon binding (currently lines 224–240 and U13 near line 637).

## What the finder currently stamps

`NarrowCompatSearch::prove_universal` (currently lines 4240–4254) builds a
zone with

```text
d = remaining_defender_placements_for_horizon(node_state, claimant,
                                               search.semantic_horizon)
build_horizon = search.semantic_horizon
```

That helper (currently lines 5356–5394) advances the Hexo player/phase clock
from the node's placement count all the way to the caller-supplied absolute
horizon and counts placements by the defender. It returns `None` after the
count exceeds eight; `None` disables the zone and makes this node search the
full legal set.

After proof selection, both production materialization paths compact the
certificate and call `rebase_zone_distances` (pre-fix `23ffc65b`, lines
905–955, called near lines 889 and 1913). The routine correctly replays every reachable DAG
node and rejects a shared node reached at two different positions, but then
sets each `zone.d` by calling the same external-horizon counter. It never
examines the selected descendants when choosing `d`. Despite its comment
claiming an exact build-horizon rebase, it also does not assign
`zone.build_horizon`; fresh zones happen already to carry the current search
horizon.

Thus final materialization currently stamps `d_ext(node, T)`, the defender
placements on the game clock from the node to external deadline `T`, whereas
verification requires `B_DAG(node)`, the maximum defender budget actually
used below that node by the selected certificate.

## Exact divergence domain

The clocks diverge at a zoned node exactly when
`d_ext(node, cert.semantic_horizon) != B_DAG(node)`. For a proof whose typed
resolutions fit the requested horizon, the practical defect is excess slack:
`d_ext` is an admissible upper bound but is larger than the exact local `B`.
Slack does not necessarily change `d_ext` on every single added ply—the extra
ply must belong to the defender according to the node's player/phase clock—
and certificate rejection needs at least one zoned node where equality is
lost. A certificate without zones cannot fail this check.

The frozen compact reproduction is the concrete case. The root is at ply 36,
the proof resolves at ply 45, and the zoned Universal is at ply 37. With the
caller deadline 52, the materializer counts eight remaining defender
placements, while the selected proof subtree has `B_DAG = 4`; the verifier
therefore reports `stored_d=8 derived_B=4`.

Two apparently contrary profiles are explained by their horizons:

- **Fresh unbounded solve:** `semantic_horizon = u32::MAX` makes
  `remaining_defender_placements_for_horizon` exceed its eight-placement bail
  threshold and return `None`. Ranked zones are therefore not attached and
  the full legal set is searched. With no zoned node, the clock equality is
  inapplicable; NQ8's unbounded certificate verifies.
- **G2R3 consume witness:** its fixed absolute horizon is 45, exactly the
  compact proof's maximum typed resolution (root 36 + 9). At the ply-37 and
  ply-38 zone nodes the external countdown happened to equal the selected
  subtree budgets (`4` and `3` respectively). Its certificates therefore
  satisfied the verifier. G2R3 tested exact-deadline consumption, not a
  slack bounded horizon.

The defect is not specific to the interior census flag, node cap, or ladder
orchestration. It can occur in any finite-horizon materialized certificate
that carries a zone and has defender-clock slack beyond the chosen proof.
Imported finite-horizon zone fragments make final rebasing especially
important: the assembled certificate, not the source fragment or caller
deadline, owns the exact evidence labels.

## Required repair

Keep search generation conservative: using the external remaining budget can
search a superset and does not itself mint evidence. At final materialization,
replay the compact DAG, derive the same bottom-up `B` recurrence as the
verifier, and stamp every zoned node with that exact local budget plus the
assembled certificate's semantic horizon. The verifier and its checks do not
change. If the conservatively generated explicit edges fail the verifier's
zone-coverage set at the exact `B`, the finder must still return no usable
certificate; this repair only makes its evidence labels truthful.
