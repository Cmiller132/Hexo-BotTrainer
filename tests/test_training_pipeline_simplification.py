from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
for package in (
    "hexo_train",
    "hexo_utils",
    "hexo_model_resnet",
    "hexo_engine",
    "hexo_runner",
):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class TrainingPipelineSimplificationTests(unittest.TestCase):
    def test_model_config_is_the_model_owned_config_source(self) -> None:
        from hexo_train.config import normalize_training_config

        config = normalize_training_config(
            {
                "model": {
                    "name": "hexo_model_resnet",
                    "config": {"channels": 32},
                },
            },
            base_dir=ROOT,
        )

        self.assertEqual(config.model.config["channels"], 32)
        self.assertFalse(hasattr(config, "model_specific"))

    def test_top_level_model_specific_is_rejected(self) -> None:
        from hexo_train.config import normalize_training_config

        with self.assertRaisesRegex(ValueError, "model_specific"):
            normalize_training_config(
                {
                    "model": {"name": "hexo_model_resnet"},
                    "model_specific": {"channels": 32},
                },
                base_dir=ROOT,
            )

    def test_plugin_entrypoint_group_is_hexo_train_models(self) -> None:
        from hexo_train.config import ModelConfig
        from hexo_train.registry import load_model_plugin

        plugin = object()

        class FakeEntryPoint:
            name = "fake_model"

            def load(self) -> object:
                return plugin

        seen_groups: list[str] = []

        def fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
            seen_groups.append(group)
            return [FakeEntryPoint()]

        with patch("hexo_train.registry.entry_points", fake_entry_points):
            loaded = load_model_plugin(ModelConfig(name="fake_model"))

        self.assertIs(loaded, plugin)
        self.assertEqual(seen_groups, ["hexo_train.models"])

    def test_model_is_built_before_component_overrides(self) -> None:
        from hexo_train.components import ComponentOverrides, build_model_components
        from hexo_train.config import normalize_training_config
        from hexo_train.context import RunContext
        from hexo_train.defaults import build_shared_components

        class Plugin:
            def build_model(self, game_spec: object, config: object) -> str:
                return "built-model"

            def training_component_overrides(
                self,
                *,
                defaults: object,
                config: object,
                shared: object,
                model: object,
            ) -> ComponentOverrides:
                self.model_seen = model
                return ComponentOverrides(extra={"model_seen": model})

        with tempfile.TemporaryDirectory() as directory:
            config = normalize_training_config(
                {
                    "model": {"name": "fake_model", "config": {"x": 1}},
                    "run": {"output_dir": directory},
                },
                base_dir=ROOT,
            )
            ctx = RunContext.from_config(config)
            shared = build_shared_components(ctx)
            components = build_model_components(
                plugin=Plugin(),
                ctx=ctx,
                shared=shared,
            )

        self.assertEqual(components.model, "built-model")
        self.assertEqual(components.extra["model_seen"], "built-model")

    def test_d6_selection_is_deterministic_in_hexo_train(self) -> None:
        from hexo_train.symmetry import D6SymmetrySelector

        selector = D6SymmetrySelector()
        first = selector.choose(seed=7, epoch=2, sample_index=11, game_id="g", turn_index=3)
        second = selector.choose(seed=7, epoch=2, sample_index=11, game_id="g", turn_index=3)

        self.assertEqual(first, second)

    def test_policy_record_uses_parent_sample_action_order(self) -> None:
        from hexo_utils.samples import (
            PolicyOutputRecord,
            TrainingSampleRecord,
            build_legal_policy_value_target,
        )

        policy = PolicyOutputRecord(
            game_id="g",
            turn_index=4,
            model_id="m",
            logits=(0.25, 0.75),
            selected_action_id="b",
        )
        sample = TrainingSampleRecord(
            game_id="g",
            turn_index=4,
            legal_action_ids=("a", "b"),
            policy=policy,
        )

        target = build_legal_policy_value_target(sample)

        self.assertEqual(target.legal_action_ids, ("a", "b"))
        self.assertFalse(hasattr(policy, "legal_action_ids"))

    def test_resnet_plugin_is_composition_only(self) -> None:
        try:
            from hexo_model_resnet.plugin import get_plugin
            from hexo_train.components import ComponentOverrides
        except ImportError as exc:
            self.skipTest(f"ResNet plugin dependencies unavailable: {exc}")

        plugin = get_plugin()

        self.assertFalse(hasattr(plugin, "forward_inference"))
        self.assertFalse(hasattr(plugin, "loss"))
        self.assertFalse(hasattr(plugin, "augment_batch"))
        overrides = plugin.training_component_overrides(
            defaults=types.SimpleNamespace(),
            config={},
            shared=types.SimpleNamespace(),
            model=None,
        )
        self.assertIsInstance(overrides, ComponentOverrides)


if __name__ == "__main__":
    unittest.main()
