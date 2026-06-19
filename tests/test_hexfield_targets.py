"""M2 gates: losses / targets / shards / batching vs the restnet oracles.

- 65-bin helpers ≡ dense_cnn_restnet.losses
- segment legal-prefix CE ≡ restnet's masked dense CE on embedded rows
- masked binned losses ≡ restnet semantics incl. zero-denominator exact-0
- STV even-offset EMA + future-opponent-policy rule ≡ restnet samples helpers
- finalize invariants (hard z, moves-left countdown, truncated, fast-mask)
- expansion: policy slot mapping, D6 commutation, off-legal hard error
- hexfield_compact_v1 writer round-trip + sidecar
- legacy restnet shard cross-read with derived legality + derived win-now
- micro-bucket accumulation ≡ monolithic loss/grads (fp64 theorem)
- pair-budget bucket rule
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch

from hexfield_testkit import api, random_playout

from hexfield import constants as C
from hexfield.batching import (
    collate_training,
    pair_budget_microbuckets,
    split_stvalue_columns,
    step_global_denominators,
)
from hexfield.engine_facts import facts_from_engine, player_int
from hexfield.geometry import apply_d6, pack_action_id, unpack_action_id
from hexfield.losses import (
    binned_value_loss,
    decode_binned_value,
    decode_moves_left,
    hexfield_loss,
    scalar_to_binned_target,
    segment_policy_ce,
)
from hexfield.model import HexfieldNet
from hexfield.samples import (
    STV_HORIZONS,
    HexfieldSampleData,
    _future_opponent_policy,
    _short_term_value_targets,
    expand_sample,
    finalize_game_samples,
)
from hexfield.shards import read_compact_shard, read_legacy_restnet_shard, write_compact_shard
from hexo_engine.types import AxialCoord, PlacementAction


def _sample_from_state(state, rng: random.Random, turn_index: int) -> HexfieldSampleData:
    facts = facts_from_engine(api.to_python_state(state))
    legal = sorted(api.legal_action_ids(state))
    chosen = rng.sample(legal, k=min(3, len(legal)))
    weights = [rng.random() + 0.1 for _ in chosen]
    total = sum(weights)
    policy = tuple((aid, w / total) for aid, w in zip(chosen, weights))
    return HexfieldSampleData(
        game_id="test",
        turn_index=turn_index,
        current_player=facts.current_player,
        phase=facts.phase,
        records=facts.records,
        first_stone=facts.first_stone,
        own_hot=facts.own_hot,
        opp_hot=facts.opp_hot,
        own_win=facts.own_win,
        opp_win=facts.opp_win,
        policy=policy,
        metadata={"pcr_full": True},
    )


def _make_game(seed: int, max_plies: int = 24):
    """Pending (player, sample, root_value) decisions from one random game."""

    rng = random.Random(seed)
    state = api.new_game()
    pending = []
    winner = None
    for ply in range(max_plies):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        sample = _sample_from_state(state, rng, ply)
        pending.append((sample.current_player, sample, rng.uniform(-0.8, 0.8)))
        q, r = unpack_action_id(rng.choice(ids))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            winner = player_int(api.terminal(state).winner)
            break
    return pending, winner


def test_bin_helpers_match_restnet_oracle() -> None:
    from dense_cnn_restnet import losses as oracle

    torch.manual_seed(0)
    values = torch.rand(64) * 2.0 - 1.0
    mine = scalar_to_binned_target(values)
    theirs = oracle.scalar_to_binned_target(values)
    assert torch.allclose(mine, theirs, atol=0)

    logits = torch.randn(16, C.VALUE_BINS)
    assert torch.allclose(decode_binned_value(logits), oracle.decode_binned_value(logits), atol=1e-7)


def test_segment_ce_matches_restnet_dense_ce() -> None:
    from dense_cnn_restnet import losses as oracle

    torch.manual_seed(1)
    rng = np.random.RandomState(2)
    b, npad, area = 3, 24, 41 * 41
    legal_counts = torch.tensor([10, 17, 5])
    logits = torch.randn(b, npad)
    target = torch.zeros(b, npad)
    for g in range(b):
        l = int(legal_counts[g])
        t = torch.rand(l)
        target[g, :l] = t / t.sum()

    dense_logits = torch.randn(b, area)  # junk everywhere off the embedding
    dense_target = torch.zeros(b, area)
    dense_mask = torch.zeros(b, area, dtype=torch.bool)
    for g in range(b):
        l = int(legal_counts[g])
        cells = rng.choice(area, size=l, replace=False)
        dense_logits[g, cells] = logits[g, :l]
        dense_target[g, cells] = target[g, :l]
        dense_mask[g, cells] = True

    mine = segment_policy_ce(logits, legal_counts, target)
    theirs = oracle.soft_cross_entropy(dense_logits, dense_target, mask=dense_mask)
    assert torch.allclose(mine, theirs, atol=1e-6)


def test_binned_value_loss_matches_restnet() -> None:
    from dense_cnn_restnet import losses as oracle

    torch.manual_seed(3)
    logits = torch.randn(8, C.VALUE_BINS)
    target = torch.rand(8) * 2.0 - 1.0
    assert torch.allclose(
        binned_value_loss(logits, target), oracle.binned_value_loss(logits, target), atol=1e-6
    )
    mask = torch.tensor([1.0, 0, 1, 0, 1, 1, 0, 0])
    assert torch.allclose(
        binned_value_loss(logits, target, mask=mask),
        oracle.binned_value_loss(logits, target, mask=mask),
        atol=1e-6,
    )
    zero = binned_value_loss(logits, target, mask=torch.zeros(8))
    assert float(zero) == 0.0  # zero-denominator rows contribute exactly nothing


def test_stv_and_opp_helpers_match_restnet() -> None:
    from dense_cnn_restnet import samples as oracle

    rng = random.Random(7)
    # Synthetic decision sequence with the real 1-then-2 player structure.
    players_int = [0] + [1 if ((k - 1) // 2) % 2 == 0 else 0 for k in range(1, 23)]
    decisions_mine = []
    decisions_oracle = []
    for k, player in enumerate(players_int):
        policy = ((pack_action_id(k, -k), 0.7), (pack_action_id(k + 1, -k), 0.3))
        meta = {"pcr_full": rng.random() > 0.4}
        root_value = rng.uniform(-1.0, 1.0)
        decisions_mine.append(
            (
                player,
                HexfieldSampleData(
                    game_id="", turn_index=k, current_player=player, phase="FirstStone",
                    records=(), first_stone=None, own_hot=(), opp_hot=(), own_win=(),
                    opp_win=(), policy=policy, metadata=meta,
                ),
                root_value,
            )
        )
        decisions_oracle.append(
            (f"player{player}", SimpleNamespace(policy=policy, metadata=meta), root_value)
        )

    for index in range(len(players_int)):
        player = players_int[index]
        mine = _short_term_value_targets(decisions_mine, index, player, STV_HORIZONS)
        theirs = oracle._short_term_value_targets(
            decisions_oracle, index, f"player{player}", STV_HORIZONS
        )
        assert mine == theirs
        for mask_fast in (False, True):
            mine_opp = _future_opponent_policy(
                decisions_mine, index, player, mask_from_fast=mask_fast
            )
            theirs_opp = oracle._future_opponent_policy(
                decisions_oracle, index, f"player{player}", mask_from_fast=mask_fast
            )
            assert mine_opp == theirs_opp


def test_finalize_invariants() -> None:
    pending, winner = _make_game(11)
    assert len(pending) >= 6
    finalized = finalize_game_samples(pending, winner)
    n = len(finalized)
    for index, row in enumerate(finalized):
        z = 0.0 if winner is None else (1.0 if winner == row.current_player else -1.0)
        assert row.value == z
        assert row.moves_left == float(n - index - 1)
    truncated = finalize_game_samples(pending, None, truncated=True)
    assert all(row.moves_left == -1.0 for row in truncated)
    assert all(row.metadata["truncated"] for row in truncated)


def test_expand_policy_mapping_and_d6() -> None:
    pending, winner = _make_game(13)
    finalized = finalize_game_samples(pending, winner)
    sample = finalized[min(6, len(finalized) - 1)]
    base = expand_sample(sample, symmetry=0)
    assert np.isclose(base.policy.sum(), sum(w for _a, w in sample.policy))
    assert 0.0 <= base.opp_coverage <= 1.0

    for sym in range(12):
        rot = expand_sample(sample, symmetry=sym)
        assert rot.policy.shape == base.policy.shape
        # Each stored action's mass lands on the slot of its transformed cell.
        for action_id, weight in sample.policy:
            q, r = unpack_action_id(action_id)
            cell = apply_d6(sym, q, r)
            slot = rot.support.index[cell]
            assert slot < rot.support.legal_count
            assert rot.policy[slot] >= np.float32(weight) - 1e-7
        assert np.isclose(rot.policy.sum(), base.policy.sum())
        assert np.isclose(rot.opp_policy.sum(), base.opp_policy.sum())

    bad = HexfieldSampleData(
        game_id="", turn_index=0, current_player=sample.current_player, phase=sample.phase,
        records=sample.records, first_stone=sample.first_stone, own_hot=sample.own_hot,
        opp_hot=sample.opp_hot, own_win=sample.own_win, opp_win=sample.opp_win,
        policy=((pack_action_id(2000, 2000), 1.0),),  # nowhere near the support
    )
    try:
        expand_sample(bad)
        raise AssertionError("off-legal policy target must be a hard error")
    except ValueError:
        pass


def test_writer_roundtrip(tmp_path) -> None:
    pending, winner = _make_game(17)
    finalized = finalize_game_samples(pending, winner)
    path = tmp_path / "game.npz"
    rows = write_compact_shard(path, finalized)
    assert rows == len(finalized)
    sidecar = (tmp_path / "game.json").read_text(encoding="utf-8")
    assert '"lineage": "hexfield"' in sidecar
    assert '"hexfield_compact_v1"' in sidecar

    restored = read_compact_shard(path)
    assert len(restored) == len(finalized)
    for a, b in zip(finalized, restored):
        assert b.records == a.records
        assert b.current_player == a.current_player
        assert b.phase == a.phase
        assert b.first_stone == a.first_stone
        assert b.own_hot == a.own_hot and b.opp_hot == a.opp_hot
        assert b.own_win == a.own_win and b.opp_win == a.opp_win
        assert [aid for aid, _ in b.policy] == [aid for aid, _ in a.policy]
        assert np.allclose(
            [w for _aid, w in b.policy], np.asarray([w for _aid, w in a.policy], dtype=np.float32)
        )
        assert b.value == np.float32(a.value)
        assert b.moves_left == np.float32(a.moves_left)
        assert [h for h, _ in b.short_term_value] == [h for h, _ in a.short_term_value]


def test_legacy_crossread(tmp_path) -> None:
    from dense_cnn_restnet import compact_io as oracle_io
    from dense_cnn_restnet.samples import Model1SampleData

    rng = random.Random(19)
    state = random_playout(101, 15)
    if api.terminal(state) is not None:
        state = random_playout(103, 11)
    assert api.terminal(state) is None
    mirror = api.to_python_state(state)
    facts = facts_from_engine(mirror)
    legal = sorted(api.legal_action_ids(state))
    policy = ((legal[0], 0.6), (legal[-1], 0.4))

    legacy = Model1SampleData(
        game_id="g",
        turn_index=5,
        current_player=str(mirror.current_player),
        phase=str(mirror.phase.value),
        center=(1, -1),  # crop center: ignored by the adapter
        stones=[(c.q, c.r, str(p)) for c, p in mirror.board.stones],
        legal_action_ids=legal,  # crop-restricted at source: ignored
        placement_history=[
            (rec.coord.q, rec.coord.r, str(rec.player), None, rec.placement_index, None, None)
            for rec in mirror.placement_history
        ],
        first_stone=(
            (mirror.first_stone.q, mirror.first_stone.r) if facts.first_stone else None
        ),
        own_hot=facts.own_hot,
        opponent_hot=facts.opp_hot,
        opponent_last_turn=((0, 0),),  # ignored: derived from history instead
        policy=policy,
        root_prior_policy=(),
        opp_policy=((legal[1], 1.0),),
        value=0.5,
        short_term_value=((2, 0.25), (16, -0.5)),
        moves_left=42.0,
    )
    path = tmp_path / "legacy.npz"
    oracle_io.write_compact_shard(path, [legacy], short_term_value_horizons=STV_HORIZONS)

    rows = read_legacy_restnet_shard(path)
    assert len(rows) == 1
    row = rows[0]
    assert row.metadata["source"] == "legacy_shard"
    assert row.records == facts.records
    assert row.current_player == facts.current_player
    assert row.phase == facts.phase
    assert row.own_hot == facts.own_hot and row.opp_hot == facts.opp_hot
    # Standing-win cells derive from stones and equal the engine's view.
    assert row.own_win == facts.own_win and row.opp_win == facts.opp_win
    assert row.moves_left == 42.0
    assert dict(row.short_term_value) == {2: np.float32(0.25), 16: np.float32(-0.5)}

    # Derived legality == the engine's full legal set (not the stored column).
    expanded = expand_sample(row)
    sup_ids = [pack_action_id(q, r) for q, r in expanded.support.legal_coords().tolist()]
    assert sup_ids == legal


def _write_minimal_legacy_shard(path, *, schema_version) -> None:
    """Write a one-row restnet compact-v1 shard via the oracle writer, then
    re-save it with the ``schema_version`` column forced to ``schema_version``
    (or dropped entirely when ``None``) so the adapter's drift guard can be
    exercised without hand-rolling the legacy column layout."""

    from dense_cnn_restnet import compact_io as oracle_io
    from dense_cnn_restnet.samples import Model1SampleData

    state = random_playout(101, 15)
    if api.terminal(state) is not None:
        state = random_playout(103, 11)
    assert api.terminal(state) is None
    mirror = api.to_python_state(state)
    facts = facts_from_engine(mirror)
    legal = sorted(api.legal_action_ids(state))
    legacy = Model1SampleData(
        game_id="g",
        turn_index=5,
        current_player=str(mirror.current_player),
        phase=str(mirror.phase.value),
        center=(1, -1),
        stones=[(c.q, c.r, str(p)) for c, p in mirror.board.stones],
        legal_action_ids=legal,
        placement_history=[
            (rec.coord.q, rec.coord.r, str(rec.player), None, rec.placement_index, None, None)
            for rec in mirror.placement_history
        ],
        first_stone=(
            (mirror.first_stone.q, mirror.first_stone.r) if facts.first_stone else None
        ),
        own_hot=facts.own_hot,
        opponent_hot=facts.opp_hot,
        opponent_last_turn=((0, 0),),
        policy=((legal[0], 0.6), (legal[-1], 0.4)),
        root_prior_policy=(),
        opp_policy=((legal[1], 1.0),),
        value=0.5,
        short_term_value=((2, 0.25), (16, -0.5)),
        moves_left=42.0,
    )
    oracle_io.write_compact_shard(path, [legacy], short_term_value_horizons=STV_HORIZONS)

    with np.load(path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    if schema_version is None:
        arrays.pop("schema_version", None)
    else:
        arrays["schema_version"] = np.asarray(schema_version, dtype=np.int32)
    np.savez(path, **arrays)


def test_legacy_shard_schema_version_guard(tmp_path) -> None:
    # A wrong stored legacy schema_version is a hard error (drift guard,
    # mirroring read_compact_shard).
    bad = tmp_path / "bad.npz"
    _write_minimal_legacy_shard(bad, schema_version=99)
    try:
        read_legacy_restnet_shard(bad)
        raise AssertionError("wrong legacy schema_version must raise")
    except ValueError as exc:
        assert "schema" in str(exc) and "99" in str(exc)

    # The expected version (1) reads cleanly.
    ok = tmp_path / "ok.npz"
    _write_minimal_legacy_shard(ok, schema_version=1)
    rows = read_legacy_restnet_shard(ok)
    assert len(rows) == 1 and rows[0].metadata["source"] == "legacy_shard"

    # Lenient on absence: pre-versioning restnet shards still read.
    absent = tmp_path / "absent.npz"
    _write_minimal_legacy_shard(absent, schema_version=None)
    rows = read_legacy_restnet_shard(absent)
    assert len(rows) == 1


def test_microbucket_loss_equals_monolithic_fp64() -> None:
    pending, winner = _make_game(23, max_plies=14)
    finalized = finalize_game_samples(pending, winner)
    rows = [expand_sample(s, symmetry=i % 12) for i, s in enumerate(finalized)]
    assert len(rows) >= 6
    horizons = STV_HORIZONS
    denoms = step_global_denominators(rows, horizons)

    def loss_for(model, row_subset, pad_to=None):
        batch = collate_training(row_subset, pad_to=pad_to)
        batch = split_stvalue_columns(batch, horizons)
        batch = {k: (v.double() if v.dtype == torch.float32 else v) for k, v in batch.items()}
        out = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
        total, _ = hexfield_loss(out, batch, denominators=denoms)
        return total

    torch.manual_seed(31)
    model = HexfieldNet().double()
    loss_mono = loss_for(model, rows)
    loss_mono.backward()
    mono = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    mono_total = loss_mono.detach().item()

    model2 = HexfieldNet().double()
    model2.load_state_dict({k: v for k, v in model.state_dict().items()})
    buckets = pair_budget_microbuckets(rows, budget=2.0e5)  # tiny: force splits
    assert len(buckets) >= 2
    assert sorted(id(r) for b in buckets for r in b) == sorted(id(r) for r in rows)
    total = 0.0
    for bucket in buckets:
        loss = loss_for(model2, bucket)
        loss.backward()
        total += loss.detach().item()
    assert abs(total - mono_total) <= 1e-10 * (1.0 + abs(mono_total))
    for name, p in model2.named_parameters():
        scale = 1.0 + mono[name].abs().max().item()
        assert (p.grad - mono[name]).abs().max().item() <= 1e-10 * scale, name


def test_pair_budget_bucket_rule() -> None:
    pending, winner = _make_game(29, max_plies=20)
    rows = [expand_sample(s) for s in finalize_game_samples(pending, winner)]
    budget = 5.0e5
    buckets = pair_budget_microbuckets(rows, budget=budget)
    for bucket in buckets:
        s_pad = max(r.support.num_nodes for r in bucket) + C.NUM_TOKENS
        if len(bucket) > 1:
            assert len(bucket) * s_pad**2 <= budget


def test_decode_moves_left_median() -> None:
    # decode_moves_left is a softmax-EXPECTATION decode (NOT median) mapping the
    # 65-bin scalar support [-1, 1] onto decisions [0, MOVES_LEFT_CAP]:
    #   decisions = (scalar + 1) / 2 * MOVES_LEFT_CAP   (losses.decode_moves_left).
    # Near-one-hot logits collapse the expectation onto the peak bin's scalar.
    cap = float(C.MOVES_LEFT_CAP)
    logits = torch.full((3, C.VALUE_BINS), -40.0)
    logits[0, 0] = 40.0   # bin 0  -> scalar -1 -> 0 decisions
    logits[1, 32] = 40.0  # bin 32 -> scalar  0 -> cap/2 decisions
    logits[2, 64] = 40.0  # bin 64 -> scalar +1 -> cap decisions
    decoded = decode_moves_left(logits)
    assert torch.allclose(decoded, torch.tensor([0.0, 0.5 * cap, cap]))
