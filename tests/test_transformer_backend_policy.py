from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_backend_policy import (
    PYTORCH_BACKEND,
    SCALAR_BACKEND,
    transformer_backend_metadata,
    validate_transformer_backend_metadata,
)


class TransformerBackendPolicyTests(unittest.TestCase):
    def test_scalar_backend_is_canonical_reference(self) -> None:
        metadata = transformer_backend_metadata(
            seed=7,
            tokenizer_type="char",
            corpus_hash="corpus",
        )

        self.assertEqual(metadata["backend"], SCALAR_BACKEND)
        self.assertEqual(metadata["backend_role"], "canonical_reference")
        self.assertEqual(metadata["planned_performance_backend"], PYTORCH_BACKEND)
        self.assertFalse(metadata["requires_scalar_parity"])
        self.assertEqual(metadata["parity_status"], "reference")
        self.assertTrue(metadata["runtime_library_allowed"])
        validate_transformer_backend_metadata(
            metadata,
            require_artifact_fields=True,
        )

    def test_pytorch_backend_is_experimental_and_requires_parity(self) -> None:
        metadata = transformer_backend_metadata(
            active_backend=PYTORCH_BACKEND,
            seed=11,
            tokenizer_type="closed-world-subword",
            corpus_hash="corpus",
            tokenizer_manifest_hash="manifest",
            device="mps",
            dtype="float32",
        )

        self.assertEqual(metadata["backend_role"], "experimental_performance")
        self.assertTrue(metadata["requires_scalar_parity"])
        self.assertEqual(metadata["parity_status"], "pending")
        self.assertFalse(metadata["purity"]["pretrained_weights"])
        self.assertFalse(metadata["purity"]["pretrained_tokenizer"])
        validate_transformer_backend_metadata(
            metadata,
            require_artifact_fields=True,
        )

    def test_pytorch_backend_cannot_claim_reference_status(self) -> None:
        metadata = transformer_backend_metadata(
            active_backend=PYTORCH_BACKEND,
            parity_status="reference",
        )

        with self.assertRaises(ValueError):
            validate_transformer_backend_metadata(metadata)

    def test_pytorch_backend_rejects_unknown_parity_status(self) -> None:
        metadata = transformer_backend_metadata(
            active_backend=PYTORCH_BACKEND,
            parity_status="almost",
        )

        with self.assertRaises(ValueError):
            validate_transformer_backend_metadata(metadata)

    def test_scalar_backend_must_keep_reference_status(self) -> None:
        metadata = transformer_backend_metadata(parity_status="matched")

        with self.assertRaises(ValueError):
            validate_transformer_backend_metadata(metadata)

    def test_artifact_validation_rejects_missing_required_fields(self) -> None:
        metadata = transformer_backend_metadata(active_backend=PYTORCH_BACKEND)

        with self.assertRaises(ValueError):
            validate_transformer_backend_metadata(
                metadata,
                require_artifact_fields=True,
            )

    def test_purity_flags_must_stay_closed_world(self) -> None:
        metadata = transformer_backend_metadata()
        metadata["purity"]["external_embeddings"] = True

        with self.assertRaises(ValueError):
            validate_transformer_backend_metadata(metadata)

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            transformer_backend_metadata(active_backend="numpy")


if __name__ == "__main__":
    unittest.main()
