from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt import (
    build_torch_training_parity_attempt,
)
from transformer_torch_training_parity_attempt_validation import (
    validate_torch_training_parity_attempt,
)


class TransformerTorchTrainingParityAttemptValidationTests(unittest.TestCase):
    def test_valid_attempt_summary_passes(self) -> None:
        validate_torch_training_parity_attempt(_attempt())

    def test_valid_json_round_tripped_attempt_summary_passes(self) -> None:
        attempt = json.loads(json.dumps(_attempt()))

        validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_missing_promotion_gate(self) -> None:
        attempt = _attempt()
        attempt.pop("training_backend_promotion_gate")

        with self.assertRaisesRegex(ValueError, "training_backend_promotion_gate"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_accidental_backend_promotion(self) -> None:
        attempt = _attempt()
        attempt["promoted_training_backend"] = True

        with self.assertRaisesRegex(ValueError, "must not promote"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_stale_attempt_passed_flag(self) -> None:
        attempt = _attempt()
        attempt["passed"] = True

        with self.assertRaisesRegex(ValueError, "passed flag"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_stale_attempt_status(self) -> None:
        attempt = _attempt()
        attempt["status"] = "training_parity_matched"

        with self.assertRaisesRegex(ValueError, "status is inconsistent"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_dirty_closed_world_boundary(self) -> None:
        attempt = _attempt()
        attempt["closed_world_boundary"]["pretrained_tokenizer_imported"] = True

        with self.assertRaisesRegex(ValueError, "pretrained_tokenizer_imported"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_external_training_text_source(self) -> None:
        attempt = _attempt()
        attempt["closed_world_boundary"]["training_text_source"] = "external_corpus"

        with self.assertRaisesRegex(ValueError, "training_text_source"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_stale_promotion_gate_boundary_status(self) -> None:
        attempt = _attempt()
        attempt["training_backend_promotion_gate"][
            "closed_world_boundary_passed"
        ] = False

        with self.assertRaisesRegex(ValueError, "boundary status"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_stale_promotion_gate_boundary_failures(self) -> None:
        attempt = _attempt()
        attempt["training_backend_promotion_gate"]["closed_world_boundary_failures"] = [
            "training_text_source"
        ]

        with self.assertRaisesRegex(ValueError, "boundary failures"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_stale_next_requirements_stage(self) -> None:
        attempt = _attempt()
        attempt["next_requirements"]["stage"] = "training_readiness"
        attempt["next_requirements"]["next_actions"] = [
            f"satisfy_training_readiness:{blocker}"
            for blocker in attempt["next_requirements"]["primary_blockers"]
        ]

        with self.assertRaisesRegex(ValueError, "next_requirements.stage"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_stale_next_requirements_schema(self) -> None:
        attempt = _attempt()
        attempt["next_requirements"]["schema_version"] = 0

        with self.assertRaisesRegex(ValueError, "next_requirements.schema_version"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_stale_next_requirements_kind(self) -> None:
        attempt = _attempt()
        attempt["next_requirements"]["kind"] = "stale"

        with self.assertRaisesRegex(ValueError, "next_requirements.kind"):
            validate_torch_training_parity_attempt(attempt)

    def test_validator_rejects_stale_next_requirements_runtime_status(self) -> None:
        attempt = _attempt()
        attempt["next_requirements"]["runtime_status"] = "blocked_dtype_unavailable"
        attempt["next_requirements"]["next_actions"] = [
            "request_available_pytorch_dtype"
        ]

        with self.assertRaisesRegex(ValueError, "runtime_status"):
            validate_torch_training_parity_attempt(attempt)

    def test_writer_validation_requires_artifact_paths(self) -> None:
        attempt = _attempt()

        with self.assertRaisesRegex(ValueError, "artifacts"):
            validate_torch_training_parity_attempt(
                attempt,
                require_artifacts=True,
            )

        attempt["artifacts"] = {
            "fixture": "fixture.json",
            "candidate": "candidate.json",
            "report": "report.json",
            "attempt": "attempt.json",
        }
        validate_torch_training_parity_attempt(
            attempt,
            require_artifacts=True,
        )


def _attempt() -> dict:
    artifacts = build_torch_training_parity_attempt(
        corpus_dir=ROOT / "corpus",
        fixture_id="validation-training-parity-attempt",
        seed=53,
        context_index=4,
        context_size=4,
        embedding_dim=4,
        feedforward_dim=8,
        steps=2,
        importer=_missing_importer,
    )
    return copy.deepcopy(artifacts["attempt"])


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)
