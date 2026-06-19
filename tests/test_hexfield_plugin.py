"""M7 unit gates: config parse, checkpoint strict round-trip + mismatch raise,
plugin wiring (build_model param count, optimizer decay split)."""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch

from hexfield_testkit import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield.checkpoints import HexfieldCheckpointSaver, load_into, save_checkpoint
from hexfield.config import parse_hexfield_config
from hexfield.engine_facts import facts_from_engine, player_int
from hexfield.geometry import pack_action_id
from hexfield.model import HexfieldNet
from hexfield.plugin import get_plugin
from hexfield.samples import STV_HORIZONS, HexfieldSampleData, finalize_game_samples
from hexfield.shards import write_compact_shard
from hexfield.trainer import HexfieldTrainer
from hexfield.window import load_packed_shard


def test_config_production_defaults() -> None:
    cfg = parse_hexfield_config({})
    sp = cfg.selfplay
    assert sp.search_visits == 512
    assert sp.pcr_fast_visits == 128
    assert sp.pcr_full_proportion == pytest.approx(0.33)
    # §5.1 quarantine: OFF by default.
    assert sp.root_fpu_zero_under_noise is False
    assert sp.root_policy_temperature == 1.0
    assert sp.root_policy_temperature_early == 0.0
    assert sp.search_parity_mode is False
    # temperature schedule decays from 1.0 to the floor.
    temps = cfg.temperature_by_ply()
    assert temps[0] == pytest.approx(1.0)
    assert temps[-1] == pytest.approx(sp.temperature_floor)
    assert all(t >= sp.temperature_floor for t in temps)


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown"):
        parse_hexfield_config({"selfplay": {"not_a_key": 1}})


def test_config_rejects_unknown_top_level_keys() -> None:
    # A typo'd top-level section (e.g. [model.config.slfplay]) must fail loudly
    # rather than be silently ignored, which would leave defaults in force.
    with pytest.raises(ValueError, match="unknown HexfieldConfig keys"):
        parse_hexfield_config({"slfplay": {"search_visits": 256}})
    # Sanity: a valid full config (every top-level section) still parses.
    cfg = parse_hexfield_config(
        {
            "device": "cpu",
            "selfplay": {"search_visits": 256},
            "training": {"batch_rows": 8},
            "evaluation": {"eval_visits": 64},
        }
    )
    assert cfg.device == "cpu"
    assert cfg.selfplay.search_visits == 256
    assert cfg.training.batch_rows == 8
    assert cfg.evaluation.eval_visits == 64


def test_config_override() -> None:
    cfg = parse_hexfield_config({"selfplay": {"search_visits": 256, "search_parity_mode": True}})
    assert cfg.selfplay.search_visits == 256
    assert cfg.selfplay.search_parity_mode is True


def test_checkpoint_strict_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    model = HexfieldNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(tmp_path / "ck.pt", model=model, optimizer=opt, epoch=5, extra={"run": "t"})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["meta"]["epoch"] == 5
    assert payload["meta"]["lineage"] == "hexfield"

    fresh = HexfieldNet()
    # weights differ before load
    p_fresh = dict(fresh.named_parameters())
    p_orig = dict(model.named_parameters())
    assert not torch.equal(p_fresh["stem.weight"], p_orig["stem.weight"])
    meta = load_into(fresh, payload)
    assert meta["epoch"] == 5
    for name, p in fresh.named_parameters():
        assert torch.equal(p, p_orig[name]), name


def test_checkpoint_strict_load_rejects_mismatch() -> None:
    model = HexfieldNet()
    state = model.state_dict()
    state.pop(next(iter(state)))  # drop one key
    with pytest.raises(ValueError, match="key mismatch"):
        load_into(HexfieldNet(), {"model": state, "meta": {}})


def test_plugin_builds_model_and_optimizer_split() -> None:
    plugin = get_plugin()
    assert plugin.name == "hexfield"
    model = plugin.build_model({}, {})
    assert sum(p.numel() for p in model.parameters()) == 1_591_748

    overrides = plugin.training_component_overrides(
        defaults=None, config={}, shared=None, model=model
    )
    assert overrides.uses_shared_sample_store is False
    assert overrides.trainer is not None
    assert overrides.checkpoint_loader is not None and overrides.checkpoint_saver is not None
    # Decoupled weight decay on matrix weights only: the per-block bias tables,
    # tokens, and all 1-D params (LN gains/biases, conv biases) must be in the
    # no-decay group.
    groups = overrides.optimizer.param_groups
    decay_ids = {id(p) for p in groups[0]["params"]}
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    assert groups[0]["weight_decay"] == 1e-4
    assert groups[1]["weight_decay"] == 0.0
    named = dict(model.named_parameters())
    # Per-block relative-position bias tables (bias_tables.0/.1/.2) all no-decay.
    bias_table_names = [n for n in named if n.startswith("bias_tables.")]
    assert len(bias_table_names) == len(model.bias_tables) >= 3
    for name in bias_table_names:
        assert id(named[name]) in no_decay_ids, name
    assert id(named["tokens"]) in no_decay_ids
    for name, p in named.items():
        if p.ndim <= 1:
            assert id(p) in no_decay_ids, f"{name} (1-D) should not decay"


def _decisive_game_shard(out_dir: Path) -> int:
    """Write one hexfield_compact_v1 shard from a real finished game (P0 builds
    six on the Q axis), so the trainer has Full rows to consume."""

    state = api.new_game()
    moves = [
        (0, 0),           # P0 opening
        (0, 4), (4, -4),  # P1
        (1, 0), (2, 0),   # P0
        (-4, 0), (0, -4), # P1
        (3, 0), (4, 0),   # P0 -> five
        (8, -8), (-8, 4), # P1 (declines)
        (5, 0),           # P0 -> SIX, wins
    ]
    pending = []
    winner = None
    for ply, (q, r) in enumerate(moves):
        facts = facts_from_engine(api.to_python_state(state))
        sample = HexfieldSampleData(
            game_id="g", turn_index=ply, current_player=facts.current_player,
            phase=facts.phase, records=facts.records, first_stone=facts.first_stone,
            own_hot=facts.own_hot, opp_hot=facts.opp_hot, own_win=facts.own_win,
            opp_win=facts.opp_win, policy=((pack_action_id(q, r), 1.0),),
            metadata={"pcr_full": True},
        )
        pending.append((facts.current_player, sample, 0.0))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            winner = player_int(api.terminal(state).winner)
            break
    assert winner is not None, "constructed game must finish"
    finalized = finalize_game_samples(pending, winner, STV_HORIZONS)
    out_dir.mkdir(parents=True, exist_ok=True)
    return write_compact_shard(out_dir / "game_1.npz", finalized,
                               short_term_value_horizons=STV_HORIZONS)


def test_trainer_runs_on_real_rows(tmp_path) -> None:
    """The train-on-rows path end to end: real decisive-game shard -> packed
    window -> serial expand -> micro-bucket collate -> loss -> optimizer step ->
    finite loss + steps>0. (Self-play smokes truncate on weak models, so this is
    the direct gate.)

    train_passes now consumes an opaque PackedWindow (built by load_packed_shard
    here, the production loader build_window_split uses) rather than reading a
    samples_dir itself; with no prior select_training_samples call _effective_rows_for
    falls back to min(window.n, train_samples_per_epoch), so the full window trains.
    """

    samples_dir = tmp_path / "samples"
    rows = _decisive_game_shard(samples_dir / "epoch_000001")
    assert rows >= 6

    # Pack the single decisive-game shard into the in-RAM PackedWindow the trainer
    # consumes (the production path concatenates these via build_window_split).
    window = load_packed_shard(samples_dir / "epoch_000001" / "game_1.npz")
    assert window.n == rows

    model = HexfieldNet()
    cfg = parse_hexfield_config({"device": "cpu", "training": {"batch_rows": 8}})
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = HexfieldTrainer(model=model, config=cfg, optimizer=opt)

    # train_passes reads ctx.config.run.seed (D6/permutation determinism) and
    # ctx.diagnostics_dir (the per-epoch diag write); samples_dir is unused now
    # that the packed window is passed directly.
    ctx = types.SimpleNamespace(
        config=types.SimpleNamespace(run=types.SimpleNamespace(seed=0)),
        diagnostics_dir=tmp_path / "diagnostics",
    )
    (tmp_path / "diagnostics").mkdir()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    result = trainer.train_passes(
        passes=2, sample_window=window, sample_symmetries=None,
        ctx=ctx, components=None, epoch=1,
    )
    assert result["status"] == "completed"
    assert result["window_rows"] == rows
    assert result["trained_rows"] == rows
    assert result["steps"] > 0
    assert result["grad_norm_mean"] >= 0.0
    # the optimizer actually moved the weights
    moved = any(
        not torch.equal(p, before[n]) for n, p in model.named_parameters()
    )
    assert moved, "training did not update any parameter"
