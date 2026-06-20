from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.char_model import char_model_fixture, context_and_target
from support.fake_torch import fake_torch_importer
from transformer_model import OptimizationConfig
from transformer_torch_backend import (
    REQUIRED_TORCH_TRAINING_CANDIDATE_KEYS,
    build_torch_training_parity_candidate,
    validate_torch_training_parity_candidate,
)
from transformer_training_parity import build_scalar_training_parity_fixture


class TransformerTorchTrainingCandidateValidationTests(unittest.TestCase):
    def test_validator_accepts_missing_runtime_candidate(self) -> None:
        candidate = _candidate(importer=_missing_importer)

        validate_torch_training_parity_candidate(candidate)

    def test_validator_accepts_pending_replay_candidate(self) -> None:
        candidate = _candidate(
            importer=fake_torch_importer(
                training_runtime=True,
                gradient_runtime=True,
            )
        )

        validate_torch_training_parity_candidate(candidate)

    def test_validator_rejects_missing_required_candidate_key(self) -> None:
        candidate = _candidate(importer=_missing_importer)
        candidate.pop("runtime_report")

        with self.assertRaisesRegex(ValueError, "candidate.runtime_report"):
            validate_torch_training_parity_candidate(candidate)

    def test_validator_rejects_extra_candidate_key(self) -> None:
        candidate = _candidate(importer=_missing_importer)
        candidate["unvalidated_extra_field"] = "drift"

        with self.assertRaisesRegex(ValueError, "candidate keys"):
            validate_torch_training_parity_candidate(candidate)

    def test_validator_rejects_stale_backend_parity_status(self) -> None:
        candidate = _candidate(importer=_missing_importer)
        candidate["backend"]["parity_status"] = "pending"

        with self.assertRaisesRegex(ValueError, "backend.parity_status"):
            validate_torch_training_parity_candidate(candidate)

    def test_validator_rejects_stale_implementation_status(self) -> None:
        candidate = _candidate(importer=_missing_importer)
        candidate["implementation_status"] = "training_replay_parity_pending"

        with self.assertRaisesRegex(ValueError, "implementation_status"):
            validate_torch_training_parity_candidate(candidate)

    def test_validator_rejects_stale_runtime_report(self) -> None:
        candidate = _candidate(importer=_missing_importer)
        candidate["runtime_report"]["runtime"] = {
            **candidate["runtime_report"]["runtime"],
            "dtype": "float16",
        }

        with self.assertRaisesRegex(ValueError, "runtime_report.runtime"):
            validate_torch_training_parity_candidate(candidate)

    def test_validator_rejects_stale_readiness_summary_count(self) -> None:
        candidate = _candidate(importer=_missing_importer)
        candidate["training_readiness"]["summary"]["check_count"] = True

        with self.assertRaisesRegex(ValueError, "training_readiness.summary"):
            validate_torch_training_parity_candidate(candidate)

    def test_validator_rejects_extra_readiness_key(self) -> None:
        candidate = _candidate(importer=_missing_importer)
        candidate["training_readiness"]["unvalidated_extra_field"] = "drift"

        with self.assertRaisesRegex(ValueError, "training_readiness keys"):
            validate_torch_training_parity_candidate(candidate)

    def test_validator_rejects_stale_replay_gate_status(self) -> None:
        candidate = _candidate(importer=_missing_importer)
        gate = candidate["training_replay_parity_gate"]
        gate["status"] = "training_replay_parity_matched"

        with self.assertRaisesRegex(ValueError, "training_replay_parity_gate.status"):
            validate_torch_training_parity_candidate(candidate)

    def test_validator_rejects_stale_replay_gate_summary_count(self) -> None:
        candidate = _candidate(importer=_missing_importer)
        candidate["training_replay_parity_gate"]["summary"]["check_count"] = True

        with self.assertRaisesRegex(ValueError, "training_replay_parity_gate.summary"):
            validate_torch_training_parity_candidate(candidate)

    def test_validator_rejects_stale_training_case_status(self) -> None:
        candidate = _candidate(importer=_missing_importer)
        candidate["training_case"]["status"] = "pending"

        with self.assertRaisesRegex(ValueError, "training_case.status"):
            validate_torch_training_parity_candidate(candidate)

    def test_validator_rejects_boolean_training_case_steps(self) -> None:
        candidate = _candidate(importer=_missing_importer)
        candidate["training_case"]["steps"] = True

        with self.assertRaisesRegex(ValueError, "training_case.steps"):
            validate_torch_training_parity_candidate(candidate)

    def test_validator_rejects_extra_training_case_key(self) -> None:
        candidate = _candidate(importer=_missing_importer)
        candidate["training_case"]["unvalidated_extra_field"] = "drift"

        with self.assertRaisesRegex(ValueError, "training_case keys"):
            validate_torch_training_parity_candidate(candidate)

    def test_public_required_key_catalog_includes_replay_gate(self) -> None:
        self.assertIn(
            "training_replay_parity_gate",
            REQUIRED_TORCH_TRAINING_CANDIDATE_KEYS,
        )


def _candidate(*, importer) -> dict:
    return build_torch_training_parity_candidate(
        fixture=_scalar_training_fixture(),
        importer=importer,
    )


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)


def _scalar_training_fixture() -> dict:
    tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=53)
    context, target = context_and_target(ids, config, tokenizer)
    return build_scalar_training_parity_fixture(
        fixture_id="candidate-validation-scalar",
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
