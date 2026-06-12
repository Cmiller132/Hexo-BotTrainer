"""CPU tests for the dense_cnn_restnet reference-checkpoint evaluation LADDER.

The per-epoch eval stage plays BOTH SealBot games (existing behavior) and,
when `[model.config.evaluation] reference_games > 0`, model-vs-model games
against EACH fixed anchor in `reference_checkpoints` — `reference_games` games
PER anchor (owner directive: SealBot is measured-saturated as a strength
arbiter, so the anchored rating ladder is the real progress signal).

Covers: (a) config parsing (defaults off, canonical list key, deprecated
scalar alias appends + de-dups, entry validation, live main_4 TOML parses),
(b) flag-off regression (diagnostics keys byte-identical, no
`reference`/`references` blocks), (c) two-anchor CPU smoke with real tiny
networks end to end (`references` list + `reference` first-block alias +
per-anchor ref{i}-*.hxr records), (d) fail-open (missing path, architecture
mismatch, and PARTIAL ladder failure — a bad anchor[0] must not kill
anchor[1] — SealBot block intact, the epoch never crashes).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_models", "hexo_engine", "hexo_runner", "hexo_train", "hexo_utils"):
    path = ROOT / "packages" / package / "python"
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
_RESTNET = ROOT / "packages" / "dense_cnn_restnet" / "python"
if str(_RESTNET) not in sys.path:
    sys.path.insert(0, str(_RESTNET))

torch = pytest.importorskip("torch")
pytest.importorskip("hexo_utils._rust")

config_module = importlib.import_module("dense_cnn_restnet.config")
evaluation_module = importlib.import_module("dense_cnn_restnet.evaluation")
mcts_module = importlib.import_module("dense_cnn_restnet.mcts")
plugin_module = importlib.import_module("dense_cnn_restnet.plugin")
engine = importlib.import_module("hexo_engine")
sealbot_module = importlib.import_module("hexo_runner.adapters.sealbot")


# --------------------------------------------------------------------------- #
# (a) Config parsing
# --------------------------------------------------------------------------- #
def test_reference_keys_default_off() -> None:
    cfg = config_module.parse_model1_config({})
    assert cfg.evaluation.reference_checkpoints == ()
    assert cfg.evaluation.reference_checkpoint == ""
    assert cfg.evaluation.reference_games == 0


def test_reference_scalar_alias_feeds_the_list() -> None:
    """The deprecated scalar key still parses and lands in the canonical list."""

    cfg = config_module.parse_model1_config(
        {
            "evaluation": {
                "reference_checkpoint": "/some/run/checkpoints/epoch_000010.pt",
                "reference_games": 50,
            }
        }
    )
    assert cfg.evaluation.reference_checkpoints == ("/some/run/checkpoints/epoch_000010.pt",)
    assert cfg.evaluation.reference_checkpoint == "/some/run/checkpoints/epoch_000010.pt"
    assert cfg.evaluation.reference_games == 50


def test_reference_checkpoints_list_parses_dedups_and_appends_alias() -> None:
    cfg = config_module.parse_model1_config(
        {
            "evaluation": {
                "reference_checkpoints": [
                    "/runs/a/epoch_000005.pt",
                    "/runs/a/epoch_000010.pt",
                    "/runs/a/epoch_000005.pt",  # duplicate: dropped, order preserved
                ],
                # Alias not already in the list: appended LAST.
                "reference_checkpoint": "/runs/b/epoch_000002.pt",
                "reference_games": 32,
            }
        }
    )
    assert cfg.evaluation.reference_checkpoints == (
        "/runs/a/epoch_000005.pt",
        "/runs/a/epoch_000010.pt",
        "/runs/b/epoch_000002.pt",
    )


def test_reference_checkpoints_alias_duplicate_is_dropped() -> None:
    cfg = config_module.parse_model1_config(
        {
            "evaluation": {
                "reference_checkpoints": ["/runs/a/epoch_000010.pt"],
                "reference_checkpoint": "/runs/a/epoch_000010.pt",
            }
        }
    )
    assert cfg.evaluation.reference_checkpoints == ("/runs/a/epoch_000010.pt",)


def test_reference_checkpoints_rejects_invalid_values() -> None:
    # Entries must be non-empty strings.
    with pytest.raises(ValueError, match="non-empty strings"):
        config_module.parse_model1_config({"evaluation": {"reference_checkpoints": ["/ok.pt", ""]}})
    with pytest.raises(ValueError, match="non-empty strings"):
        config_module.parse_model1_config({"evaluation": {"reference_checkpoints": ["/ok.pt", 7]}})
    # The canonical key must be an array, not a scalar path.
    with pytest.raises(ValueError, match="must be an array"):
        config_module.parse_model1_config({"evaluation": {"reference_checkpoints": "/ok.pt"}})


def test_reference_unknown_key_still_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported key"):
        config_module.parse_model1_config({"evaluation": {"reference_game": 1}})


def test_main_4_live_config_parses_with_reference_keys() -> None:
    """The live run's TOML must parse and carry the ladder activation settings."""

    tomllib = pytest.importorskip("tomllib")
    raw = tomllib.loads((ROOT / "configs" / "dense_cnn_restnet_main_4.toml").read_text(encoding="utf-8"))
    cfg = config_module.parse_model1_config(raw["model"]["config"])
    assert cfg.evaluation.games_per_epoch == 32
    assert cfg.evaluation.reference_games == 32
    assert len(cfg.evaluation.reference_checkpoints) == 2
    assert cfg.evaluation.reference_checkpoints[0].endswith(
        "dense_cnn_restnet_main_3/checkpoints/epoch_000005.pt"
    )
    assert cfg.evaluation.reference_checkpoints[1].endswith(
        "dense_cnn_restnet_main_3/checkpoints/epoch_000010.pt"
    )


# --------------------------------------------------------------------------- #
# Harness: SimpleNamespace trainer/ctx around the real evaluate_epoch
# --------------------------------------------------------------------------- #
class _Diagnostics:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        return path


def _model_config(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "device": "cpu",
        "architecture": {
            "channels": 8,
            "blocks_type": "R_R",
            "attention_heads": 2,
            "mlp_ratio": 1,
            "short_term_value_horizons": [],
        },
        "training": {"amp": False},
        "selfplay": {
            "search_visits": 8,
            "widening_max_children": 8,
            "mcts_session_cache_max_states": 4096,
            "mcts_active_root_limit": 32,
        },
        "evaluation": dict(evaluation),
    }


def _harness(tmp_path: Path, model_config: Mapping[str, Any]) -> tuple[Any, Any]:
    config = config_module.parse_model1_config(dict(model_config))
    model = plugin_module.get_plugin().build_model({}, dict(model_config))
    trainer = SimpleNamespace(
        config=config,
        device="cpu",
        inference_batch_size=16,
        mcts_virtual_batch_size=2,
    )
    components = SimpleNamespace(model=SimpleNamespace(trainer=trainer, model=model))
    ctx = SimpleNamespace(
        output_dir=tmp_path / "run",
        diagnostics=_Diagnostics(tmp_path / "run" / "diagnostics"),
        config=SimpleNamespace(run=SimpleNamespace(seed=11)),
    )
    return ctx, components


def _first_legal_action(state: object) -> object:
    from hexo_engine.types import unpack_coord_id

    return engine.PlacementAction(unpack_coord_id(int(engine.legal_action_ids(state)[0])))


def _stub_sealbot(monkeypatch: pytest.MonkeyPatch) -> None:
    """No SealBot checkout in tests: skip the probe, stub the opponent."""

    from hexo_runner.player import DecisionResult

    class FakeOpponent:
        def decide(self, state: object) -> DecisionResult:
            return DecisionResult(action=_first_legal_action(state))

        def close(self) -> None:
            return

    monkeypatch.setattr(sealbot_module.SealBotConfig, "validate", lambda self: None)
    monkeypatch.setattr(evaluation_module, "_build_opponent", lambda _config: FakeOpponent())


def _stub_dense_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """First-legal fake MCTS for tests that don't need the real search."""

    class FakeMctsSession:
        def discard(self, _game_key: int) -> None:
            return

        def run(self, _keys: Any, root_states: Any, _inference: Any, **kwargs: Any) -> list[Any]:
            results = []
            for state in root_states:
                action_id = int(engine.legal_action_ids(state)[0])
                results.append(
                    mcts_module.SearchResult(
                        action_id=action_id,
                        visit_policy={action_id: 1.0},
                        root_value=0.0,
                        visits=int(kwargs["visits"]),
                        root_prior_policy={action_id: 1.0},
                    )
                )
            return results

    monkeypatch.setattr(evaluation_module, "new_mcts_session", lambda **_kwargs: FakeMctsSession())


_SEALBOT_RESULT_KEYS = {
    "status",
    "epoch",
    "games",
    "completed",
    "wins",
    "losses",
    "mean_turns",
    "output_dir",
    "elapsed_seconds",
    "mcts_search_elapsed_seconds",
    "opponent_elapsed_seconds",
    "rounds",
    "dense_forward_batches",
    "dense_decisions",
    "diagnostics_path",
}


# --------------------------------------------------------------------------- #
# (b) Flag-off regression: diagnostics shape byte-identical, no reference block
# --------------------------------------------------------------------------- #
def test_flag_off_keeps_sealbot_diagnostics_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_sealbot(monkeypatch)
    _stub_dense_search(monkeypatch)
    epoch = 2
    ctx, components = _harness(
        tmp_path,
        _model_config({"games_per_epoch": 2, "max_actions": 8, "require_sealbot": True}),
    )

    result = evaluation_module.evaluate_epoch(ctx=ctx, components=components, epoch=epoch)

    assert result["status"] == "completed"
    assert set(result.keys()) == _SEALBOT_RESULT_KEYS
    assert "reference" not in result
    assert "references" not in result
    payload = json.loads(
        (ctx.diagnostics.root / f"dense_cnn.evaluation.epoch_{epoch:06d}.json").read_text(encoding="utf-8")
    )
    assert "reference" not in payload
    assert "references" not in payload
    assert set(payload.keys()) == _SEALBOT_RESULT_KEYS - {"diagnostics_path"}
    # No reference records appear when the feature is off (ref{i}-*.hxr).
    output_dir = ctx.output_dir / "evaluation" / f"epoch_{epoch:06d}"
    assert not list(output_dir.glob("ref*.hxr"))


# --------------------------------------------------------------------------- #
# (c) CPU smoke: real tiny networks, TWO-anchor ladder, blocks + records
# --------------------------------------------------------------------------- #
def test_reference_ladder_cpu_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_sealbot(monkeypatch)
    epoch = 3
    reference_games = 3
    ckpt_a = tmp_path / "anchor_a.pt"
    ckpt_b = tmp_path / "anchor_b.pt"

    model_config = _model_config(
        {
            "games_per_epoch": 2,
            "max_actions": 32,
            "require_sealbot": True,
            "opening_temperature": 0.6,
            "opening_moves": 2,
            "reference_checkpoints": [str(ckpt_a), str(ckpt_b)],
            "reference_games": reference_games,
        }
    )
    ctx, components = _harness(tmp_path, model_config)
    # Anchors = same architecture, different random inits (the real anchors are
    # architecture-identical by directive).
    for ckpt_path in (ckpt_a, ckpt_b):
        anchor_net = plugin_module.get_plugin().build_model({}, dict(model_config))
        torch.save({"model_state": anchor_net.state_dict()}, ckpt_path)

    result = evaluation_module.evaluate_epoch(ctx=ctx, components=components, epoch=epoch)

    # SealBot block intact.
    assert result["status"] == "completed"
    assert result["games"] == 2
    assert result["wins"] + result["losses"] <= result["completed"]

    # One block per anchor, in config order, each with plausible fields.
    references = result["references"]
    assert isinstance(references, list) and len(references) == 2
    for block, ckpt_path in zip(references, (ckpt_a, ckpt_b)):
        assert block["status"] == "completed"
        assert block["games"] == reference_games
        assert 0 <= block["completed"] <= reference_games
        assert block["wins"] >= 0 and block["losses"] >= 0
        assert block["wins"] + block["losses"] <= block["completed"]
        assert block["mean_turns"] > 0
        assert block["checkpoint"] == str(ckpt_path)
        assert block["elapsed_seconds"] >= 0.0
    # Single-anchor shape-compat alias: `reference` is the FIRST block.
    assert result["reference"] == references[0]

    # Per-anchor record prefixes (ref0-/ref1- = positional index into the
    # de-duplicated checkpoint list) land in the same eval output dir.
    output_dir = ctx.output_dir / "evaluation" / f"epoch_{epoch:06d}"
    for opponent_index in range(2):
        for index in range(reference_games):
            assert (output_dir / f"ref{opponent_index}-{epoch:06d}-{index:04d}.hxr").is_file()
    assert len(list(output_dir.glob("ref*.hxr"))) == 2 * reference_games
    for index in range(2):
        assert (output_dir / f"eval-{epoch:06d}-{index:04d}.hxr").is_file()

    # Both shapes are persisted in the same per-eval diagnostics JSON.
    payload = json.loads(
        (ctx.diagnostics.root / f"dense_cnn.evaluation.epoch_{epoch:06d}.json").read_text(encoding="utf-8")
    )
    assert [block["status"] for block in payload["references"]] == ["completed", "completed"]
    assert payload["reference"] == payload["references"][0]
    assert payload["references"][1]["wins"] == references[1]["wins"]


# --------------------------------------------------------------------------- #
# (d) Fail-open: bad reference inputs must never crash the epoch
# --------------------------------------------------------------------------- #
def test_reference_phase_fails_open_on_missing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_sealbot(monkeypatch)
    _stub_dense_search(monkeypatch)
    epoch = 2
    missing = tmp_path / "does_not_exist.pt"
    ctx, components = _harness(
        tmp_path,
        _model_config(
            {
                "games_per_epoch": 2,
                "max_actions": 8,
                "require_sealbot": True,
                "reference_checkpoint": str(missing),
                "reference_games": 4,
            }
        ),
    )

    result = evaluation_module.evaluate_epoch(ctx=ctx, components=components, epoch=epoch)

    # The epoch's SealBot results are intact...
    assert result["status"] == "completed"
    assert result["games"] == 2
    assert _SEALBOT_RESULT_KEYS <= set(result.keys())
    # ...and the (alias-fed, single-anchor) ladder reports the failure instead
    # of raising, in both the list and the first-block alias.
    assert len(result["references"]) == 1
    reference = result["reference"]
    assert reference == result["references"][0]
    assert reference["status"] == "failed"
    assert reference["games"] == 4
    assert reference["completed"] == 0
    assert reference["wins"] == 0 and reference["losses"] == 0
    assert reference["checkpoint"] == str(missing)
    assert "does_not_exist.pt" in reference["error"]
    output_dir = ctx.output_dir / "evaluation" / f"epoch_{epoch:06d}"
    assert not list(output_dir.glob("ref*.hxr"))
    payload = json.loads(
        (ctx.diagnostics.root / f"dense_cnn.evaluation.epoch_{epoch:06d}.json").read_text(encoding="utf-8")
    )
    assert payload["reference"]["status"] == "failed"
    assert payload["references"][0]["status"] == "failed"


def test_reference_phase_fails_open_on_architecture_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong-architecture payload trips the strict-load incompatibility trap
    (the value-head missing-key hazard) and fails open, not silently."""

    _stub_sealbot(monkeypatch)
    _stub_dense_search(monkeypatch)
    epoch = 2
    ckpt_path = tmp_path / "wrong_arch.pt"
    mismatched_config = _model_config({})
    mismatched_config["architecture"]["channels"] = 4
    wrong_net = plugin_module.get_plugin().build_model({}, mismatched_config)
    torch.save({"model_state": wrong_net.state_dict()}, ckpt_path)

    ctx, components = _harness(
        tmp_path,
        _model_config(
            {
                "games_per_epoch": 2,
                "max_actions": 8,
                "require_sealbot": True,
                "reference_checkpoint": str(ckpt_path),
                "reference_games": 2,
            }
        ),
    )

    result = evaluation_module.evaluate_epoch(ctx=ctx, components=components, epoch=epoch)

    assert result["status"] == "completed"
    reference = result["reference"]
    assert reference == result["references"][0]
    assert reference["status"] == "failed"
    assert "incompatible" in reference["error"]


def test_reference_ladder_partial_failure_isolates_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad anchor must not kill the others: anchor[0]'s missing checkpoint
    fails open while anchor[1] still plays its games (per-anchor try/except)."""

    _stub_sealbot(monkeypatch)
    _stub_dense_search(monkeypatch)
    epoch = 2
    reference_games = 2
    missing = tmp_path / "gone.pt"
    good = tmp_path / "good_anchor.pt"

    model_config = _model_config(
        {
            "games_per_epoch": 2,
            "max_actions": 8,
            "require_sealbot": True,
            "reference_checkpoints": [str(missing), str(good)],
            "reference_games": reference_games,
        }
    )
    ctx, components = _harness(tmp_path, model_config)
    good_net = plugin_module.get_plugin().build_model({}, dict(model_config))
    torch.save({"model_state": good_net.state_dict()}, good)

    result = evaluation_module.evaluate_epoch(ctx=ctx, components=components, epoch=epoch)

    # SealBot results intact, ladder reports both anchors in config order.
    assert result["status"] == "completed"
    references = result["references"]
    assert [block["status"] for block in references] == ["failed", "completed"]
    assert references[0]["checkpoint"] == str(missing)
    assert "gone.pt" in references[0]["error"]
    assert references[1]["checkpoint"] == str(good)
    assert references[1]["games"] == reference_games
    # The alias still mirrors the FIRST block, even when it is the failed one.
    assert result["reference"] == references[0]

    # Records: none for the failed anchor 0, one per game for anchor 1
    # (max_actions=8 + first-legal stub play -> aborted records still written).
    output_dir = ctx.output_dir / "evaluation" / f"epoch_{epoch:06d}"
    assert not list(output_dir.glob("ref0-*.hxr"))
    for index in range(reference_games):
        assert (output_dir / f"ref1-{epoch:06d}-{index:04d}.hxr").is_file()
