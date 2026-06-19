from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.char_model import char_model_fixture, context_and_target
from support.fake_torch import fake_torch_importer
from transformer_model import OptimizationConfig
from transformer_torch_backend import build_torch_training_parity_candidate
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchTrainingCandidateReplayControlTests(unittest.TestCase):
    def test_candidate_carries_replay_control_probe(self) -> None:
        fixture = _scalar_training_fixture()

        candidate = build_torch_training_parity_candidate(
            fixture=fixture,
            importer=fake_torch_importer(
                training_runtime=True,
                gradient_runtime=True,
            ),
        )

        probe = candidate["accumulation_replay_control_probe"]
        self.assertEqual(probe["status"], "accumulation_replay_control_recorded")
        self.assertEqual(probe["planned_microstep_count"], 2)
        self.assertEqual(
            probe["backward_pass_count"],
            fixture["training_case"]["steps"],
        )
        self.assertEqual(probe["optimizer_updates_applied"], 0)
        self.assertEqual(
            probe["gradient_signature_mismatch_count"],
            fixture["training_case"]["steps"],
        )
        self.assertFalse(probe["accumulated_gradient_parity_proven"])
        self.assertFalse(probe["final_update_parity_proven"])

        buffer_comparison = candidate["accumulation_replay_buffer_comparison"]
        self.assertEqual(
            buffer_comparison["status"],
            "replay_buffer_signature_mismatch",
        )
        self.assertEqual(
            buffer_comparison["step_count"],
            fixture["training_case"]["steps"],
        )
        self.assertFalse(buffer_comparison["buffered_gradient_parity_proven"])
        self.assertFalse(buffer_comparison["optimizer_update_parity_proven"])

        update_comparison = candidate["accumulation_replay_update_comparison"]
        self.assertEqual(update_comparison["status"], "replay_update_comparison_not_run")
        self.assertFalse(update_comparison["optimizer_update_parity_proven"])
        self.assertIn("buffer", update_comparison["reason"])


def _scalar_training_fixture() -> dict:
    tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
    context, target = context_and_target(ids, config, tokenizer)
    return build_scalar_training_parity_fixture(
        fixture_id="tiny-training-scalar",
        model=model,
        tokenizer=tokenizer,
        context=context,
        target=target,
        optimizer_config=OptimizationConfig(
            optimizer="adamw",
            gradient_accumulation_steps=2,
            warmup_steps=2,
            decay_steps=2,
            min_learning_rate=0.001,
        ),
        learning_rate=0.02,
        steps=2,
        corpus_hash="corpus-hash",
    )


if __name__ == "__main__":
    unittest.main()
