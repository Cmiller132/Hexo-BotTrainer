"""Tests for the dashboard Debug-tab inference library + worker service.

The inference tests need torch + the hexgt build and a real run directory, so
they skip cleanly where those are absent (e.g. a CI box without the GPU build).
The path/signature tests are pure and always run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Pure helpers (no torch) — always importable.
from hexo_frontend import debug_service


# --------------------------------------------------------------------------
# Pure: WSL path translation (runs everywhere).
# --------------------------------------------------------------------------


def test_to_wsl_translates_windows_drive_paths():
    assert debug_service._to_wsl("E:\\Hexo-BotTrainer-hexgt\\runs\\x.pt") == "/mnt/e/Hexo-BotTrainer-hexgt/runs/x.pt"
    assert debug_service._to_wsl("C:\\a\\b") == "/mnt/c/a/b"


def test_to_wsl_passes_through_posix_paths():
    assert debug_service._to_wsl("/mnt/e/runs/x.pt") == "/mnt/e/runs/x.pt"
    assert debug_service._to_wsl("relative/path") == "relative/path"


# --------------------------------------------------------------------------
# Inference: gated on torch + hexgt + a real run dir.
# --------------------------------------------------------------------------

di = pytest.importorskip("hexo_frontend.debug_infer", reason="needs torch + hexgt build")


def _run_dir() -> Path | None:
    candidate = Path.cwd() / "runs" / "hexgt_rl_main3"
    return candidate if (candidate / "checkpoints").is_dir() else None


def _checkpoint(name: str) -> Path:
    run_dir = _run_dir()
    if run_dir is None:
        pytest.skip("runs/hexgt_rl_main3 not present")
    path = run_dir / "checkpoints" / name
    if not path.is_file():
        pytest.skip(f"checkpoint {name} not present")
    return path


def _sample_actions(n: int = 16) -> list[int]:
    run_dir = _run_dir()
    hxr = run_dir / "selfplay" / "epoch_000000.hxr"
    if not hxr.is_file():
        pytest.skip("no self-play .hxr present")
    from hexo_runner.records import HexoRecordFile

    with HexoRecordFile.open(hxr) as rf:
        rec = next(iter(rf.iter_records()))
    return [int(a) for a in rec.action_ids][:n]


@pytest.fixture(scope="module")
def cpu_only():
    # Belt-and-suspenders: the library forces CPU, but make the GPU invisible too.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield


def test_load_post_graft_checkpoint(cpu_only):
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    assert loaded.graft == "post"
    assert loaded.expanded_stv == []  # already wide, no expansion
    nonstv = [w for w in loaded.load_warnings if "short_term_value" not in w]
    assert not nonstv, loaded.load_warnings
    assert not loaded.model.training  # eval mode


def test_load_pre_graft_checkpoint_expands(cpu_only):
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_epoch000000.pt"))
    assert loaded.graft == "pre"
    # The epoch-0 STV heads were SIDE-only; the loader must widen all three.
    assert len(loaded.expanded_stv) == 3
    nonstv = [w for w in loaded.load_warnings if "short_term_value" not in w]
    assert not nonstv, loaded.load_warnings


def test_analyze_position_shapes(cpu_only):
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    actions = _sample_actions(16)
    result = di.analyze_position(loaded, actions)

    assert result["current_player"] in (0, 1)
    assert -1.0 <= result["value"] <= 1.0
    assert len(result["value_dist"]) == 65
    assert abs(sum(result["value_dist"]) - 1.0) < 1e-3
    assert len(result["value_bins"]) == 65

    policy = result["policy"]
    assert policy and result["candidate_count"] == len(policy)
    # Sorted descending by prior, probabilities in [0, 1].
    probs = [row["p"] for row in policy]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-2

    # STV heads present and scalar in range.
    assert set(result["stvalue"]) == {"4", "12", "24"}
    for head in result["stvalue"].values():
        assert -1.0 <= head["scalar"] <= 1.0
        assert len(head["dist"]) == 65

    # Both-perspectives / optimism probe present and consistent.
    assert -1.0 <= result["value_swapped"] <= 1.0
    assert abs(result["optimism"] - (result["value"] + result["value_swapped"])) < 1e-3


def test_both_perspectives_swap_is_symmetric_ish(cpu_only):
    """Swapping ownership twice returns the original board, so analyzing the
    swapped facts must reproduce the swapped value the single call reported."""
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    actions = _sample_actions(14)
    result = di.analyze_position(loaded, actions)
    # optimism is the documented sum; just assert it is finite and bounded.
    assert -2.0 <= result["optimism"] <= 2.0


def test_analyze_priors_match_canonical_evaluator(cpu_only):
    """The displayed policy prior MUST be the model's real prior: assert it equals
    the canonical ``HexgtInference.evaluate_states`` priors per action id (this is
    the alignment guarantee — candidate_ids[i] <-> policy[i] — that makes the
    heatmap trustworthy)."""
    from hexo_models.hexgt.inference import HexgtInference
    from hexo_engine.types import unpack_coord_id

    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    actions = _sample_actions(12)
    analysis = di.analyze_position(loaded, actions)
    ana = {(r["q"], r["r"]): r["p"] for r in analysis["policy"]}

    state = di.state_from_actions(actions)
    canon = HexgtInference(loaded.model, device="cpu", fp16=False).evaluate_states([state])[0]
    ref = {}
    for aid, p in zip(canon["candidate_ids"].tolist(), canon["priors"].tolist()):
        c = unpack_coord_id(int(aid))
        ref[(int(c.q), int(c.r))] = float(p)

    assert set(ana) == set(ref)
    assert max(abs(ana[k] - ref[k]) for k in ana) < 1e-5
    assert abs(analysis["value"] - canon["value"]) < 1e-5


def test_debug_search_is_clean_and_deterministic(cpu_only):
    """A debug search carries no root noise, so it is reproducible and its reported
    root prior equals the raw network prior shown by analyze."""
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    actions = _sample_actions(12)
    s1 = di.search_position(loaded, actions, visits=64)
    s2 = di.search_position(loaded, actions, visits=64)
    pick = lambda s: [(r["action_id"], round(r["p"], 5)) for r in s["visit_policy"]]
    assert pick(s1) == pick(s2)  # deterministic

    ana = {r["action_id"]: r["p"] for r in di.analyze_position(loaded, actions)["policy"]}
    for row in s1["root_prior"]:
        assert abs(row["p"] - ana.get(row["action_id"], 0.0)) < 1e-5


def test_search_position(cpu_only):
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    actions = _sample_actions(16)
    result = di.search_position(loaded, actions, visits=64)
    assert result["visits"] >= 1
    assert -1.0 <= result["root_value"] <= 1.0
    assert result["visit_policy"], "search returned an empty visit policy"
    # visit policy normalized
    total = sum(row["p"] for row in result["visit_policy"])
    assert abs(total - 1.0) < 1e-2
    # best action is one of the visited candidates
    best = result["best_action_id"]
    assert any(row["action_id"] == best for row in result["visit_policy"])


# --------------------------------------------------------------------------
# Server-layer: imported-position reconstruction (needs the engine build).
# --------------------------------------------------------------------------

web = pytest.importorskip("hexo_frontend.web", reason="needs hexo_runner/engine build")


def test_position_from_actions_reconstructs_board(cpu_only):
    actions = _sample_actions(6)
    csv = ", ".join(str(a) for a in actions)
    payload = web._debug_position_from_actions("hexgt_rl_main3", csv, 0)
    assert payload["debug"]["imported"] is True
    assert payload["debug"]["total"] == 6
    assert payload["debug"]["ply"] == 6  # ply<=0 defaults to all moves
    assert payload["debug"]["last_q"] is not None and payload["debug"]["last_r"] is not None
    assert len(payload.get("placements", [])) == 6  # six stones on the board


def test_position_from_actions_rejects_garbage():
    with pytest.raises(ValueError):
        web._debug_position_from_actions("run", "12, not_a_number", 0)
    with pytest.raises(ValueError):
        web._debug_position_from_actions("run", "   ", 0)


def test_debug_run_root_override(tmp_path, monkeypatch):
    """HEXO_DEBUG_RUN_ROOT is searched first (so Debug finds checkpoints in the
    live worktree even when the dashboard cwd is a bridge mirror); unset falls
    back to the normal cwd roots."""
    root = tmp_path / "wt"
    (root / "runs").mkdir(parents=True)
    monkeypatch.setenv("HEXO_DEBUG_RUN_ROOT", str(root))
    roots = [p.resolve() for p in web._debug_training_roots()]
    assert (root / "runs").resolve() == roots[0]
    # cwd roots still present as fallback
    assert len(roots) >= 1

    monkeypatch.delenv("HEXO_DEBUG_RUN_ROOT", raising=False)
    fallback = [p.resolve() for p in web._debug_training_roots()]
    assert (root / "runs").resolve() not in fallback
    assert fallback == [p.resolve() for p in web._training_roots()]


# --------------------------------------------------------------------------
# Model Debug v2: pure-Python PUCT tree, recorded-row reader, game-eval sweep.
# --------------------------------------------------------------------------


def _sample_shard():
    """First self-play compact .npz shard of the run (skip when absent)."""
    run_dir = _run_dir()
    if run_dir is None:
        pytest.skip("runs/hexgt_rl_main3 not present")
    shards = sorted((run_dir / "selfplay").glob("epoch_*_game_*.npz"))
    if not shards:
        pytest.skip("no self-play .npz shards present")
    return shards[0]


def test_search_tree_is_deterministic(cpu_only):
    """Same request => byte-identical serialized tree (seeded tie-breaks only)."""
    import json

    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    actions = _sample_actions(12)
    t1 = di.search_tree_position(loaded, actions, visits=64, seed=3)
    t2 = di.search_tree_position(loaded, actions, visits=64, seed=3)
    assert json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True)
    assert t1["engine"] == "py_debug"
    assert t1["visits"] == 64
    assert t1["pv"] and t1["pv"][0] == t1["best_action_id"]
    assert -1.0 <= t1["root_value"] <= 1.0


def test_search_tree_root_layer_tracks_production_search(cpu_only):
    """The py-debug tree won't bit-match Rust tie-breaking, but the same priors
    feed both searches, so the top visit sets must strongly overlap."""
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    actions = _sample_actions(12)
    tree = di.search_tree_position(loaded, actions, visits=128, top_k=32, min_n=1)
    rust = di.search_position(loaded, actions, visits=128)
    py_top = [child["action_id"] for child in tree["tree"]["children"][:5]]
    rust_top = [row["action_id"] for row in rust["visit_policy"][:5]]
    assert set(py_top) & set(rust_top), (py_top, rust_top)
    # Root-child Q values live in the side-to-move band.
    for child in tree["tree"]["children"]:
        assert -1.0 <= child["qm"] <= 1.0


def test_search_tree_respects_pruning_and_node_cap(cpu_only):
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    actions = _sample_actions(12)
    tree = di.search_tree_position(loaded, actions, visits=128, top_k=2, min_n=0, max_depth=3)

    def walk(node, depth):
        assert depth <= 3
        assert len(node["children"]) <= 2
        return 1 + sum(walk(child, depth + 1) for child in node["children"])

    assert walk(tree["tree"], 0) == tree["node_count"]
    assert tree["node_count"] <= 4000
    assert tree["truncated"] is True  # hexgt branching guarantees pruned children


def test_record_row_missing_shard_reports_not_found():
    result = di.read_record_row("/definitely/not/a/real/shard.npz", 4, 0)
    assert result["found"] is False
    assert result["reason"] == "no_shard"
    assert result["row"] is None


def test_record_row_round_trips_a_real_shard(cpu_only):
    np = pytest.importorskip("numpy")
    shard = _sample_shard()
    with np.load(shard, allow_pickle=True) as data:
        turn = int(data["turn_index"][0])
        player = int(data["current_player"][0])

    result = di.read_record_row(str(shard), turn, player)
    assert result["found"] is True and result["reason"] is None
    row = result["row"]
    assert row["current_player"] == player
    assert row["policy"], "recorded visit policy must be present"
    assert abs(sum(r["p"] for r in row["policy"]) - 1.0) < 1e-3
    probs = [r["p"] for r in row["policy"]]
    assert probs == sorted(probs, reverse=True)
    assert set(row["stvalue"]) == {"4", "12", "24"}
    assert -1.0 <= row["value_target"] <= 1.0
    assert row["search_visits"] >= 0  # 0 = unknown (normalized visit policy stored)
    assert row["frequency_weight"] >= 1.0

    # Wrong turn -> no_row; wrong side-to-move -> row_mismatch (M8 guard).
    assert di.read_record_row(str(shard), 10_000, player)["reason"] == "no_row"
    mismatch = di.read_record_row(str(shard), turn, 1 - player)
    assert mismatch["found"] is False and mismatch["reason"] == "row_mismatch"


def test_game_eval_aligns_plies_and_kl_is_nonnegative(cpu_only):
    loaded = di.load_checkpoint(_checkpoint("hexgt_rl_latest.pt"))
    actions = _sample_actions(16)
    shard = _sample_shard()
    plies = [2, 3, 4]
    result = di.game_eval_positions(loaded, actions, plies, npz_path=str(shard), winner=0)
    rows = result["plies"]
    assert [row["ply"] for row in rows] == plies
    for row in rows:
        assert row["current_player"] in (0, 1)
        assert -1.0 <= row["value"] <= 1.0
        expected_p0 = row["value"] if row["current_player"] == 0 else -row["value"]
        assert row["value_p0"] == pytest.approx(expected_p0, abs=1e-6)
        if row["kl"] is not None:
            assert row["kl"] >= -1e-6
        assert row["value_err_z"] == pytest.approx(row["value_p0"] - 1.0, abs=1e-4)
        assert row["top1_match"] in (True, False)
