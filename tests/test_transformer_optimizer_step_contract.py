from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.char_model import char_model_fixture, context_and_target
from transformer_model import OptimizationConfig
from transformer_training_parity import (
    build_scalar_training_parity_fixture,
    validate_optimizer_step_contract,
)


class TransformerOptimizerStepContractTests(unittest.TestCase):
    def test_contract_records_accumulation_and_schedule(self) -> None:
        fixture = _scalar_training_fixture()
        contract = fixture["optimizer_step_contract"]

        validate_optimizer_step_contract(
            contract,
            training_case=fixture["training_case"],
        )
        self.assertEqual(contract["optimizer"], "adamw")
        self.assertEqual(contract["parameter_count"], fixture["parameter_manifest"]["parameter_count"])
        self.assertEqual(contract["gradient_clip"]["value"], 5.0)
        self.assertEqual(contract["gradient_accumulation_steps"], 2)
        self.assertEqual(contract["gradient_accumulation"]["steps"], 2)
        self.assertEqual(contract["gradient_accumulation"]["reduction"], "mean")
        self.assertEqual(
            contract["gradient_accumulation"]["gradient_source"],
            "clipped_microstep_gradients",
        )
        self.assertTrue(
            contract["gradient_accumulation"][
                "requires_microstep_clipping"
            ]
        )
        self.assertFalse(contract["expected_step_records"][0]["update_applied"])
        self.assertTrue(contract["expected_step_records"][1]["update_applied"])
        self.assertAlmostEqual(
            contract["expected_step_records"][0]["effective_learning_rate"],
            0.01,
        )
        self.assertEqual(
            contract["expected_final_optimizer_state"]["update_count"],
            fixture["training_case"]["optimizer_state"]["update_count"],
        )

    def test_contract_validation_rejects_step_drift(self) -> None:
        fixture = _scalar_training_fixture()
        contract = copy.deepcopy(fixture["optimizer_step_contract"])
        contract["expected_step_records"][0]["pending_accumulation_after"] = 99

        with self.assertRaisesRegex(ValueError, "pending_accumulation"):
            validate_optimizer_step_contract(
                contract,
                training_case=fixture["training_case"],
            )


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
