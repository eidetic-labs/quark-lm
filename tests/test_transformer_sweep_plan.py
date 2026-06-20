from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tokenizer import CharTokenizer
from transformer_sweep_plan import (
    build_transformer_sweep_plan,
    sweep_plan_summary,
    write_transformer_sweep_plan,
)


class TransformerSweepPlanTest(unittest.TestCase):
    def test_sweep_plan_records_controlled_axes(self) -> None:
        args = _args()
        tokenizer = CharTokenizer.train("abc")

        plan = build_transformer_sweep_plan(
            args,
            tokenizer,
            recipe_id="transformer-answer:test:v0.78",
        )

        self.assertEqual(plan["kind"], "transformer_sweep_plan")
        self.assertEqual(plan["current_trial"]["tokenizer_type"], "char")
        self.assertEqual(plan["current_trial"]["transformer_profile"], "modern_small")
        self.assertEqual(plan["current_trial"]["attention_heads"], 2)
        self.assertTrue(plan["current_trial"]["use_pre_layer_norm"])
        self.assertTrue(plan["current_trial"]["use_rms_norm"])
        self.assertTrue(plan["current_trial"]["use_gated_mlp"])
        self.assertTrue(plan["current_trial"]["use_rotary_positions"])
        self.assertEqual(plan["current_trial"]["optimizer"], "adamw")
        self.assertEqual(plan["current_trial"]["gradient_clip"], 2.0)
        self.assertEqual(plan["current_trial"]["warmup_steps"], 5)
        self.assertEqual(plan["current_trial"]["decay_steps"], 40)
        self.assertEqual(plan["current_trial"]["gradient_accumulation_steps"], 2)
        self.assertEqual(
            plan["current_trial"]["direct_answer_frontier_metrics_path"],
            "runs/frontier/transformer_answer_metrics.json",
        )
        self.assertEqual(
            plan["current_trial"]["direct_answer_repair_target_profiles"],
            ["learning", "owner"],
        )
        self.assertEqual(plan["axes"]["context_size"], [16])
        self.assertEqual(
            plan["axes"]["direct_answer_repair_target_profiles"],
            [["learning", "owner"]],
        )
        self.assertEqual(
            plan["axes"]["direct_answer_frontier_metrics_path"],
            ["runs/frontier/transformer_answer_metrics.json"],
        )
        self.assertFalse(plan["pretrained_weights"])
        self.assertFalse(plan["pretrained_tokenizer"])

    def test_sweep_plan_summary_is_json_artifact_safe(self) -> None:
        plan = build_transformer_sweep_plan(
            _args(),
            CharTokenizer.train("abc"),
            recipe_id="transformer-answer:test:v0.78",
        )

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sweep_plan.json"
            write_transformer_sweep_plan(path, plan)
            written = json.loads(path.read_text(encoding="utf-8"))

        summary = sweep_plan_summary(written)
        self.assertEqual(summary["axis_count"], len(plan["axes"]))
        self.assertEqual(summary["promotion_scope"], plan["promotion_scope"])
        self.assertEqual(
            summary["direct_answer_frontier_metrics_path"],
            "runs/frontier/transformer_answer_metrics.json",
        )
        self.assertEqual(
            summary["direct_answer_repair_target_profiles"],
            ["learning", "owner"],
        )


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        run=Path("runs/sweep"),
        seed=17,
        context_size=16,
        embedding_dim=8,
        attention_heads=1,
        num_layers=2,
        feedforward_dim=24,
        use_layer_norm=False,
        use_pre_layer_norm=False,
        use_rms_norm=False,
        layer_norm_epsilon=1e-5,
        use_gated_mlp=False,
        tie_output_embeddings=False,
        use_rotary_positions=False,
        use_kv_cache_path=False,
        use_context_mean=False,
        use_context_projection=False,
        use_prompt_prefix_projection=False,
        use_prompt_position_projection=False,
        prompt_position_projection_scale=1.0,
        use_prompt_attention_summary=False,
        optimizer="sgd",
        learning_rate=0.01,
        steps=50,
        direct_answer_steps=10,
        direct_answer_mode="branch-context-profile-coverage-preserving-deficit-unlikelihood",
        direct_answer_learning_rate=0.02,
        direct_answer_repair_target_profile=["owner", "learning", "owner"],
        gradient_clip=5.0,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        warmup_steps=0,
        decay_steps=40,
        min_learning_rate=0.0,
        gradient_accumulation_steps=1,
        transformer_profile="modern_small",
        tokenizer_manifest_hash="abc123",
        direct_answer_frontier_metrics=Path(
            "runs/frontier/transformer_answer_metrics.json"
        ),
    )


if __name__ == "__main__":
    unittest.main()
