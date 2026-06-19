from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.char_model import char_model_fixture, context_and_target
from transformer_model import OptimizationConfig
from transformer_torch_backend import (
    TORCH_PARAMETER_SIGNATURE_MATCHED_STATUS,
    TORCH_PARAMETER_SIGNATURE_MISMATCH_STATUS,
    build_torch_parameter_signature_comparison,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchParameterSignatureComparisonTests(unittest.TestCase):
    def test_comparison_matches_scalar_signature(self) -> None:
        fixture = _scalar_training_fixture()
        signature = fixture["training_case"]["parameter_signature"]

        comparison = build_torch_parameter_signature_comparison(
            expected_signature=signature,
            actual_signature=copy.deepcopy(signature),
            tolerance=fixture["tolerance"],
        )

        self.assertEqual(
            comparison["status"],
            TORCH_PARAMETER_SIGNATURE_MATCHED_STATUS,
        )
        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["failed_checks"], [])
        json.dumps(comparison)

    def test_comparison_reports_signature_mismatch(self) -> None:
        fixture = _scalar_training_fixture()
        expected = fixture["training_case"]["parameter_signature"]
        actual = copy.deepcopy(expected)
        actual["sum"] += 0.25

        comparison = build_torch_parameter_signature_comparison(
            expected_signature=expected,
            actual_signature=actual,
            tolerance=fixture["tolerance"],
        )

        self.assertEqual(
            comparison["status"],
            TORCH_PARAMETER_SIGNATURE_MISMATCH_STATUS,
        )
        self.assertFalse(comparison["passed"])
        self.assertIn("sum", comparison["failed_checks"])
        self.assertGreater(comparison["max_abs_diff"], 0.0)


def _scalar_training_fixture() -> dict:
    tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
    context, target = context_and_target(ids, config, tokenizer)
    return build_scalar_training_parity_fixture(
        fixture_id="tiny-training-scalar",
        model=model,
        tokenizer=tokenizer,
        context=context,
        target=target,
        optimizer_config=OptimizationConfig(optimizer="adamw"),
        learning_rate=0.02,
        steps=1,
        corpus_hash="corpus-hash",
    )


if __name__ == "__main__":
    unittest.main()
