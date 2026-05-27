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

    def test_generic_stages_config_is_rejected(self) -> None:
        from hexo_train.config import normalize_training_config

        with self.assertRaisesRegex(ValueError, "stages"):
            normalize_training_config(
                {
                    "model": {"name": "hexo_model_resnet"},
                    "stages": ["train_steps"],
                },
                base_dir=ROOT,
            )

    def test_selfplay_loop_config_is_normalized(self) -> None:
        from hexo_train.config import normalize_training_config

        config = normalize_training_config(
            {
                "model": {"name": "hexo_model_resnet"},
                "loop": {"epochs": 10},
                "selfplay": {"games_per_epoch": 25},
                "samples": {"train_sample_count": 5000},
                "train": {"passes_per_epoch": 3},
            },
            base_dir=ROOT,
        )

        self.assertEqual(config.loop.epochs, 10)
        self.assertEqual(config.selfplay.games_per_epoch, 25)
        self.assertEqual(config.samples.train_sample_count, 5000)
        self.assertEqual(config.train.passes_per_epoch, 3)

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

    def test_pipeline_runs_selfplay_epochs_in_order(self) -> None:
        from hexo_train.components import ComponentOverrides
        from hexo_train.pipeline import TrainingPipeline
        from hexo_utils.samples import build_sample_window as real_build_window

        events: list[object] = []
        case = self

        class FakeFinalizer:
            def finalize(self, *, ctx: object, components: object, epoch: int) -> dict[str, object]:
                case.assertIsNotNone(components.shared.selfplay_result)
                events.append(("finalize", epoch))
                return {"epoch": epoch}

        class FakeSymmetrySelector:
            def select_for_window(self, sample_window: object, *, seed: int | None, epoch: int) -> object:
                case.assertIsNotNone(sample_window)
                events.append(("symmetry", epoch))
                return types.SimpleNamespace(
                    symmetries=tuple(range(epoch)),
                    seed=int(seed or 0),
                    epoch=epoch,
                    metadata={},
                )

        class FakeTrainer:
            def train_passes(
                self,
                *,
                passes: int,
                sample_window: object,
                sample_symmetries: object,
                ctx: object,
                components: object,
                epoch: int,
            ) -> dict[str, object]:
                case.assertIsNotNone(sample_window)
                case.assertIsNotNone(sample_symmetries)
                events.append(("train", epoch, passes, sample_window.window_size))
                return {"epoch": epoch, "passes": passes}

        class FakeSaver:
            def save(self, *, name: str, ctx: object, components: object) -> Path:
                events.append(("checkpoint", name))
                return ctx.checkpoint_dir / f"{name}.ckpt"

        class FakePlugin:
            name = "fake_model"

            def build_model(self, game_spec: object, config: object) -> str:
                return "fake-model"

            def training_component_overrides(self, **kwargs: object) -> ComponentOverrides:
                return ComponentOverrides(
                    sample_finalizer=FakeFinalizer(),
                    symmetry_selector=FakeSymmetrySelector(),
                    trainer=FakeTrainer(),
                    checkpoint_saver=FakeSaver(),
                )

            def generate_selfplay(
                self,
                *,
                ctx: object,
                components: object,
                epoch: int,
                games_per_epoch: int,
            ) -> dict[str, object]:
                events.append(("selfplay", epoch, games_per_epoch))
                return {"epoch": epoch, "games": games_per_epoch}

        module = types.SimpleNamespace(get_plugin=lambda: FakePlugin())

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "train.toml"
            output_dir = Path(directory).as_posix()
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'name = "fake_model"',
                        'module = "fake_training_plugin"',
                        "[run]",
                        f'output_dir = "{output_dir}"',
                        "seed = 7",
                        "[loop]",
                        "epochs = 10",
                        "[selfplay]",
                        "games_per_epoch = 4",
                        "[samples]",
                        "train_sample_count = 123",
                        "[train]",
                        "passes_per_epoch = 2",
                    ]
                ),
                encoding="utf-8",
            )

            def tracked_build_window(*args: object, **kwargs: object) -> object:
                events.append(("samples", kwargs.get("window_size")))
                return real_build_window(*args, **kwargs)

            with patch.dict(sys.modules, {"fake_training_plugin": module}):
                with patch("hexo_utils.samples.build_sample_window", tracked_build_window):
                    ctx = TrainingPipeline().run(config_path)

        self.assertEqual(len(ctx.epoch_outputs), 10)
        self.assertEqual(ctx.epoch_outputs[-1].epoch, 10)
        self.assertEqual(
            events[:6],
            [
                ("selfplay", 1, 4),
                ("finalize", 1),
                ("samples", 123),
                ("symmetry", 1),
                ("train", 1, 2, 123),
                ("checkpoint", "epoch_000001"),
            ],
        )
        self.assertIn(("checkpoint", "latest"), events)

    def test_pipeline_resume_continues_after_loaded_checkpoint_epoch(self) -> None:
        from hexo_train.components import ComponentOverrides
        from hexo_train.pipeline import TrainingPipeline

        events: list[object] = []

        class FakeLoader:
            def load(self, checkpoint_ref: object, *, ctx: object, components: object) -> dict[str, object]:
                events.append(("load", Path(checkpoint_ref).name))
                return {"status": "loaded", "checkpoint_ref": str(checkpoint_ref), "epoch": 2}

        class FakeFinalizer:
            def finalize(self, *, ctx: object, components: object, epoch: int) -> dict[str, object]:
                events.append(("finalize", epoch))
                return {"epoch": epoch}

        class FakeTrainer:
            def train_passes(self, **kwargs: object) -> dict[str, object]:
                epoch = int(kwargs["epoch"])
                events.append(("train", epoch))
                return {"epoch": epoch}

        class FakeSaver:
            def save(self, *, name: str, ctx: object, components: object) -> Path:
                events.append(("checkpoint", name))
                path = ctx.checkpoint_dir / f"{name}.ckpt"
                path.write_text(name, encoding="utf-8")
                return path

        class FakePlugin:
            name = "fake_model"

            def build_model(self, game_spec: object, config: object) -> str:
                return "fake-model"

            def training_component_overrides(self, **kwargs: object) -> ComponentOverrides:
                return ComponentOverrides(
                    sample_finalizer=FakeFinalizer(),
                    trainer=FakeTrainer(),
                    checkpoint_loader=FakeLoader(),
                    checkpoint_saver=FakeSaver(),
                )

            def generate_selfplay(
                self,
                *,
                ctx: object,
                components: object,
                epoch: int,
                games_per_epoch: int,
            ) -> dict[str, object]:
                events.append(("selfplay", epoch))
                return {"epoch": epoch, "games": games_per_epoch}

        module = types.SimpleNamespace(get_plugin=lambda: FakePlugin())

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "train.toml"
            output_dir = Path(directory) / "run"
            resume_path = Path(directory) / "resume.ckpt"
            resume_path.write_text("checkpoint", encoding="utf-8")
            config_path.write_text(
                "\n".join(
                    [
                        "[model]",
                        'name = "fake_model"',
                        'module = "fake_resume_plugin"',
                        "[run]",
                        f'output_dir = "{output_dir.as_posix()}"',
                        "[loop]",
                        "epochs = 4",
                        "[selfplay]",
                        "games_per_epoch = 1",
                        "[samples]",
                        "train_sample_count = 1",
                        "[checkpoint]",
                        f'resume_from = "{resume_path.as_posix()}"',
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(sys.modules, {"fake_resume_plugin": module}):
                ctx = TrainingPipeline().run(config_path)

        self.assertEqual([output.epoch for output in ctx.epoch_outputs], [3, 4])
        self.assertIn(("selfplay", 3), events)
        self.assertIn(("checkpoint", "epoch_000003"), events)
        self.assertNotIn(("selfplay", 1), events)
        self.assertNotIn(("checkpoint", "epoch_000001"), events)

    def test_d6_selection_is_deterministic_in_hexo_train(self) -> None:
        from hexo_train.symmetry import D6SymmetrySelector

        selector = D6SymmetrySelector()
        first = selector.choose(seed=7, epoch=2, sample_index=11, game_id="g", turn_index=3)
        second = selector.choose(seed=7, epoch=2, sample_index=11, game_id="g", turn_index=3)

        self.assertEqual(first, second)

    def test_d6_selection_records_epoch(self) -> None:
        from hexo_train.symmetry import D6SymmetrySelector

        selector = D6SymmetrySelector()
        sample_window = types.SimpleNamespace(
            window_size=2,
            index=types.SimpleNamespace(sample_count=2),
        )

        first = selector.select_for_window(sample_window, seed=7, epoch=1)
        second = selector.select_for_window(sample_window, seed=7, epoch=2)

        self.assertEqual(first.epoch, 1)
        self.assertEqual(second.epoch, 2)

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
            selected_action_id=2,
        )
        sample = TrainingSampleRecord(
            game_id="g",
            turn_index=4,
            legal_action_ids=(1, 2),
            policy=policy,
        )

        target = build_legal_policy_value_target(sample)

        self.assertEqual(target.legal_action_ids, (1, 2))
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
        self.assertTrue(hasattr(overrides.trainer, "train_passes"))
        self.assertFalse(hasattr(overrides.trainer, "train_steps"))


if __name__ == "__main__":
    unittest.main()
