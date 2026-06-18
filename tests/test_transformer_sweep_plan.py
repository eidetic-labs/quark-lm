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
        self.assertEqual(plan["axes"]["context_size"], [16])
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


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        run=Path("runs/sweep"),
        context_size=16,
        embedding_dim=8,
        attention_heads=2,
        num_layers=2,
        feedforward_dim=24,
        optimizer="adamw",
        learning_rate=0.01,
        steps=50,
        direct_answer_steps=10,
        direct_answer_mode="branch-context-profile-coverage-preserving-deficit-unlikelihood",
        direct_answer_learning_rate=0.02,
        gradient_clip=1.0,
        warmup_steps=2,
        decay_steps=40,
        gradient_accumulation_steps=4,
        transformer_profile="modern_small",
        tokenizer_manifest_hash="abc123",
    )


if __name__ == "__main__":
    unittest.main()
