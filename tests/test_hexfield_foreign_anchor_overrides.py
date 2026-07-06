"""Foreign-anchor search-profile resolution for the multistage eval.

`_anchor_native_overrides` recovers each foreign anchor's OWN trained search
profile from the `_resume_config.toml` snapshot co-located with its run, so a
Gumbel-lineage anchor (main_6 / main_7) is evaluated under Gumbel and a
PUCT-lineage anchor (main_4 / main_5) under PUCT — instead of every foreign
anchor being forced onto the candidate's PUCT profile (which handicapped the
Gumbel-trained anchors). It must never raise: a bad snapshot falls back to None
so the caller can use the candidate PUCT profile.
"""

from __future__ import annotations

from hexfield.multistage_eval import _anchor_native_overrides


def _write_run(tmp_path, name: str, selfplay_toml: str):
    run = tmp_path / name
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "epoch_000067.pt"
    ckpt.write_bytes(b"")  # path only; the helper never reads the checkpoint
    (run / "_resume_config.toml").write_text(
        "[model.config.selfplay]\n" + selfplay_toml
    )
    return ckpt


def test_gumbel_lineage_anchor_resolves_to_gumbel(tmp_path):
    ckpt = _write_run(
        tmp_path,
        "hexfield_main_7",
        "gumbel_root_enabled = true\n"
        "gumbel_sequential_halving = true\n"
        "gumbel_nonroot_select = true\n",
    )
    ov = _anchor_native_overrides(ckpt)
    assert ov is not None
    assert ov["gumbel_root"] is True
    assert ov["gumbel_sequential_halving"] is True
    assert ov["gumbel_nonroot_select"] is True


def test_puct_lineage_anchor_resolves_to_puct(tmp_path):
    # main_5-style snapshot: no gumbel keys -> the SelfplayConfig defaults leave
    # every gumbel flag off.
    ckpt = _write_run(tmp_path, "hexfield_main_5", "c_scale = 0.0\n")
    ov = _anchor_native_overrides(ckpt)
    assert ov is not None
    assert ov["gumbel_root"] is False
    assert ov["gumbel_sequential_halving"] is False


def test_missing_snapshot_returns_none(tmp_path):
    # External checkpoint with no _resume_config.toml -> caller falls back.
    run = tmp_path / "external"
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "x.pt"
    ckpt.write_bytes(b"")
    assert _anchor_native_overrides(ckpt) is None


def test_unparseable_snapshot_returns_none(tmp_path):
    # A corrupt snapshot must not crash the eval.
    run = tmp_path / "hexfield_main_6"
    (run / "checkpoints").mkdir(parents=True)
    ckpt = run / "checkpoints" / "epoch_000073.pt"
    ckpt.write_bytes(b"")
    (run / "_resume_config.toml").write_text("this is = not valid toml [[[\n")
    assert _anchor_native_overrides(ckpt) is None


def test_unknown_selfplay_keys_are_ignored(tmp_path):
    # Schema drift: an old/new config carrying a key the current SelfplayConfig
    # does not know must not raise (robustness), and known keys still resolve.
    ckpt = _write_run(
        tmp_path,
        "hexfield_main_7b",
        "gumbel_root_enabled = true\nsome_removed_lever = 3\n",
    )
    ov = _anchor_native_overrides(ckpt)
    assert ov is not None
    assert ov["gumbel_root"] is True
