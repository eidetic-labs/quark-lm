from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_backend import (
    TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM,
    TORCH_TRAINING_PARITY_ATTEMPT_FILES,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
    TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
    build_torch_training_parity_attempt_hashes,
    build_torch_training_parity_attempt_requirements,
    load_torch_training_parity_attempt_artifact_set,
    validate_torch_training_parity_attempt_requirements,
)


class TransformerTorchBackendPublicAuditTests(unittest.TestCase):
    def test_training_attempt_audit_helpers_are_public(self) -> None:
        artifacts = {
            "fixture": {"kind": "fixture"},
            "candidate": {"kind": "candidate"},
            "report": {"kind": "report"},
        }

        hashes = build_torch_training_parity_attempt_hashes(artifacts)

        self.assertEqual(
            TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM,
            "sha256-json-v1",
        )
        self.assertEqual(
            TORCH_TRAINING_PARITY_ATTEMPT_FILES["attempt"],
            "torch_training_parity_attempt.json",
        )
        self.assertEqual(
            set(hashes),
            {"fixture", "candidate", "report"},
        )
        self.assertTrue(callable(load_torch_training_parity_attempt_artifact_set))

    def test_training_attempt_requirements_contract_is_public(self) -> None:
        requirements = build_torch_training_parity_attempt_requirements(
            runtime_report={"status": "passed", "parity_attempt_allowed": True},
            candidate={"training_readiness": {"status": "ready"}},
            report={"passed": True},
        )

        self.assertEqual(
            requirements["kind"],
            TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_KIND,
        )
        self.assertEqual(
            requirements["schema_version"],
            TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENTS_SCHEMA_VERSION,
        )
        self.assertIn("complete", TORCH_TRAINING_PARITY_ATTEMPT_REQUIREMENT_STAGES)
        validate_torch_training_parity_attempt_requirements(requirements)


if __name__ == "__main__":
    unittest.main()
