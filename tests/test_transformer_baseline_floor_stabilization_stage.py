import unittest
from types import SimpleNamespace
from typing import Any

from transformer_baseline_floor_stabilization_stage import (
    BaselineFloorStabilizationContext,
    train_baseline_floor_stabilization_stage,
)


def make_setup(**overrides: Any) -> SimpleNamespace:
    defaults: dict[str, Any] = {
        "direct_baseline_floor_repair_anchors": ["anchor-a", "anchor-b"],
        "direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active": (
            False
        ),
        "direct_answer_baseline_floor_sequential_stabilization_active": False,
        "direct_answer_baseline_floor_calibrated_sequential_stabilization_active": (
            False
        ),
        "direct_answer_baseline_floor_profile_targeted_stabilization_active": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_context(setup: SimpleNamespace) -> BaselineFloorStabilizationContext:
    return BaselineFloorStabilizationContext(
        args=SimpleNamespace(direct_answer_learning_rate=0.25),
        direct_setup=setup,
        model=lambda: "model",
        tokenizer=lambda: "tokenizer",
        optimizer=lambda: "optimizer",
        rng="rng",
        params=lambda: "params",
        update_guard={"guard": True},
        direct_baseline={"baseline": True},
        snapshot_recorder="snapshot-recorder",
        restore_direct_update_state=lambda _model, _optimizer: None,
    )


class TransformerBaselineFloorStabilizationStageTest(unittest.TestCase):
    def test_empty_repair_anchors_skip_stabilization(self) -> None:
        calls: list[str] = []
        result = train_baseline_floor_stabilization_stage(
            make_context(make_setup(direct_baseline_floor_repair_anchors=[])),
            update_learning_rate=0.1,
            direct_step=4,
            profile_scale_stage=lambda *_args, **_kwargs: calls.append("profile"),
            sequential_stage=lambda **_kwargs: calls.append("sequential"),
            batch_stage=lambda *_args, **_kwargs: calls.append("batch"),
        )

        self.assertEqual(result, (0.0, False))
        self.assertEqual(calls, [])

    def test_profile_scale_mode_routes_to_profile_scale_stage(self) -> None:
        calls: list[tuple[str, int]] = []
        ctx = make_context(
            make_setup(
                direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active=True
            )
        )

        result = train_baseline_floor_stabilization_stage(
            ctx,
            update_learning_rate=0.1,
            direct_step=8,
            profile_scale_stage=lambda routed_ctx, direct_step: (
                calls.append(("profile", direct_step)) or (1.5, True)
            ),
            sequential_stage=lambda **_kwargs: self.fail("unexpected sequential route"),
            batch_stage=lambda *_args, **_kwargs: self.fail("unexpected batch route"),
        )

        self.assertEqual(result, (1.5, True))
        self.assertEqual(calls, [("profile", 8)])

    def test_sequential_mode_routes_with_training_context(self) -> None:
        routed: dict[str, Any] = {}
        ctx = make_context(
            make_setup(direct_answer_baseline_floor_sequential_stabilization_active=True)
        )

        def sequential_stage(**kwargs: Any) -> tuple[float, bool]:
            routed.update(kwargs)
            return 2.0, True

        result = train_baseline_floor_stabilization_stage(
            ctx,
            update_learning_rate=0.2,
            direct_step=9,
            profile_scale_stage=lambda *_args, **_kwargs: self.fail(
                "unexpected profile route"
            ),
            sequential_stage=sequential_stage,
            batch_stage=lambda *_args, **_kwargs: self.fail("unexpected batch route"),
        )

        self.assertEqual(result, (2.0, True))
        self.assertEqual(routed["model"], "model")
        self.assertEqual(routed["repair_anchors"], ["anchor-a", "anchor-b"])
        self.assertEqual(routed["update_learning_rate"], 0.2)
        self.assertEqual(routed["base_learning_rate"], 0.25)
        self.assertEqual(routed["direct_step"], 9)
        self.assertFalse(routed["calibrated"])
        self.assertEqual(routed["params"], "params")

    def test_batch_mode_uses_targeted_anchor_count(self) -> None:
        routed: dict[str, Any] = {}
        ctx = make_context(make_setup())

        def batch_stage(*args: Any, **kwargs: Any) -> tuple[float, bool]:
            routed["args"] = args
            routed["kwargs"] = kwargs
            return 3.0, False

        result = train_baseline_floor_stabilization_stage(
            ctx,
            update_learning_rate=0.3,
            direct_step=10,
            profile_scale_stage=lambda *_args, **_kwargs: self.fail(
                "unexpected profile route"
            ),
            sequential_stage=lambda **_kwargs: self.fail("unexpected sequential route"),
            batch_stage=batch_stage,
        )

        self.assertEqual(result, (3.0, False))
        self.assertEqual(
            routed["args"],
            ("model", ["anchor-a", "anchor-b"], "rng", 0.3, 2, {"guard": True}),
        )
        self.assertEqual(routed["kwargs"], {"params": "params"})


if __name__ == "__main__":
    unittest.main()
