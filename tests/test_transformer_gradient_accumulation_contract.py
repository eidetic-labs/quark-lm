from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_training_parity import (
    GRADIENT_ACCUMULATION_CONTRACT_KIND,
    build_gradient_accumulation_contract,
    validate_gradient_accumulation_contract,
)


class TransformerGradientAccumulationContractTests(unittest.TestCase):
    def test_contract_records_mean_of_clipped_microstep_gradients(self) -> None:
        contract = build_gradient_accumulation_contract(
            optimizer_config={
                "gradient_accumulation_steps": 2,
                "gradient_clip": 5.0,
            },
        )

        validate_gradient_accumulation_contract(
            contract,
            steps=2,
            gradient_clip=5.0,
        )
        self.assertEqual(contract["kind"], GRADIENT_ACCUMULATION_CONTRACT_KIND)
        self.assertEqual(contract["steps"], 2)
        self.assertEqual(contract["reduction"], "mean")
        self.assertEqual(
            contract["gradient_source"],
            "clipped_microstep_gradients",
        )
        self.assertTrue(contract["requires_microstep_clipping"])
        self.assertTrue(
            contract["pytorch_equivalence"]["requires_clipped_gradient_buffer"]
        )
        self.assertFalse(
            contract["pytorch_equivalence"]["native_loss_scaling_sufficient"]
        )

    def test_contract_allows_native_loss_scaling_without_clipping(self) -> None:
        contract = build_gradient_accumulation_contract(
            optimizer_config={
                "gradient_accumulation_steps": 4,
                "gradient_clip": 0.0,
            },
        )

        validate_gradient_accumulation_contract(
            contract,
            steps=4,
            gradient_clip=0.0,
        )
        self.assertFalse(contract["requires_microstep_clipping"])
        self.assertFalse(
            contract["pytorch_equivalence"]["requires_clipped_gradient_buffer"]
        )
        self.assertTrue(
            contract["pytorch_equivalence"]["native_loss_scaling_sufficient"]
        )
        self.assertEqual(
            contract["pytorch_equivalence"]["loss_scale_if_no_microstep_clipping"],
            0.25,
        )

    def test_contract_does_not_require_buffer_for_single_clipped_step(self) -> None:
        contract = build_gradient_accumulation_contract(
            optimizer_config={
                "gradient_accumulation_steps": 1,
                "gradient_clip": 5.0,
            },
        )

        validate_gradient_accumulation_contract(
            contract,
            steps=1,
            gradient_clip=5.0,
        )
        self.assertTrue(contract["requires_microstep_clipping"])
        self.assertFalse(
            contract["pytorch_equivalence"]["requires_clipped_gradient_buffer"]
        )
        self.assertTrue(
            contract["pytorch_equivalence"]["native_loss_scaling_sufficient"]
        )

    def test_validation_rejects_step_mismatch(self) -> None:
        contract = build_gradient_accumulation_contract(
            optimizer_config={
                "gradient_accumulation_steps": 2,
                "gradient_clip": 5.0,
            },
        )

        with self.assertRaisesRegex(ValueError, "steps mismatch"):
            validate_gradient_accumulation_contract(
                contract,
                steps=3,
                gradient_clip=5.0,
            )


if __name__ == "__main__":
    unittest.main()
