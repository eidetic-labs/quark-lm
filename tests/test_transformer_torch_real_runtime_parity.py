from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt import (
    build_torch_training_parity_attempt,
)


class TransformerTorchRealRuntimeParityTests(unittest.TestCase):
    def test_real_runtime_matches_scalar_training_replay(self) -> None:
        _skip_without_torch(self)

        artifacts = build_torch_training_parity_attempt(
            corpus_dir=ROOT / "corpus",
            fixture_id="optional-real-torch-training-parity",
            requested_device="cpu",
            requested_dtype="float64",
            steps=2,
            context_index=4,
        )
        attempt = artifacts["attempt"]
        candidate = artifacts["candidate"]

        self.assertTrue(attempt["passed"])
        self.assertEqual(attempt["status"], "training_parity_matched")
        self.assertFalse(attempt["promoted_training_backend"])
        self.assertEqual(attempt["runtime"]["runtime_kind"], "pytorch")
        self.assertEqual(attempt["runtime"]["dtype"], "float64")
        self.assertTrue(attempt["training_replay_parity_gate"]["passed"])
        self.assertEqual(attempt["training_parity_report"]["failed_checks"], [])
        self.assertFalse(attempt["training_backend_promotion_gate"]["passed"])
        self.assertTrue(
            attempt["training_backend_promotion_gate"]["parity_evidence_matched"]
        )
        self.assertEqual(
            attempt["training_backend_promotion_gate"]["blockers"],
            ["fixture_scope_only", "model_quality_gate"],
        )
        self.assertEqual(attempt["next_requirements"]["stage"], "complete")
        self.assertEqual(candidate["backend"]["parity_status"], "matched")
        self.assertFalse(
            candidate["training_case"]["promoted_training_backend"]
        )
        self.assertFalse(
            attempt["closed_world_boundary"]["pretrained_weights_imported"]
        )
        self.assertFalse(
            attempt["closed_world_boundary"]["pretrained_tokenizer_imported"]
        )
        self.assertFalse(
            attempt["closed_world_boundary"]["external_embeddings_imported"]
        )


def _skip_without_torch(test_case: unittest.TestCase) -> None:
    try:
        import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
