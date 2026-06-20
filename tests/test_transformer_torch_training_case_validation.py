from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_case_validation import validate_torch_training_case


class TransformerTorchTrainingCaseValidationTests(unittest.TestCase):
    def test_validator_accepts_pending_case(self) -> None:
        validate_torch_training_case(_case())

    def test_validator_accepts_matched_case_with_fixture_scope_evidence(self) -> None:
        case = {
            **_case(status="matched"),
            "evidence_source": "training_replay_parity_gate",
            "promoted_training_backend": False,
        }

        validate_torch_training_case(case)

    def test_validator_rejects_extra_pending_case_key(self) -> None:
        case = {**_case(), "unvalidated_extra_field": "drift"}

        with self.assertRaisesRegex(ValueError, "training_case keys"):
            validate_torch_training_case(case)

    def test_validator_rejects_extra_matched_case_key(self) -> None:
        case = {
            **_case(status="matched"),
            "evidence_source": "training_replay_parity_gate",
            "promoted_training_backend": False,
            "unvalidated_extra_field": "drift",
        }

        with self.assertRaisesRegex(ValueError, "training_case keys"):
            validate_torch_training_case(case)

    def test_validator_rejects_boolean_numeric_evidence(self) -> None:
        cases = {
            "context": [0, True, 2],
            "target": True,
            "learning_rate": True,
            "steps": True,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                case = _case()
                case[field] = value

                with self.assertRaisesRegex(ValueError, f"training_case.{field}"):
                    validate_torch_training_case(case)

    def test_validator_rejects_non_finite_or_non_positive_learning_rate(self) -> None:
        for value in (0.0, -0.01, math.inf, math.nan):
            with self.subTest(value=value):
                case = _case()
                case["learning_rate"] = value

                with self.assertRaisesRegex(ValueError, "learning_rate"):
                    validate_torch_training_case(case)

    def test_validator_rejects_invalid_context(self) -> None:
        for value in ([], [0, -1], ["0"], "01"):
            with self.subTest(value=value):
                case = _case()
                case["context"] = value

                with self.assertRaisesRegex(ValueError, "context"):
                    validate_torch_training_case(case)

    def test_validator_rejects_matched_case_without_evidence_source(self) -> None:
        case = {**_case(status="matched"), "promoted_training_backend": False}

        with self.assertRaisesRegex(ValueError, "evidence_source"):
            validate_torch_training_case(case)

    def test_validator_rejects_matched_case_that_promotes_training(self) -> None:
        case = {
            **_case(status="matched"),
            "evidence_source": "training_replay_parity_gate",
            "promoted_training_backend": True,
        }

        with self.assertRaisesRegex(ValueError, "must not promote"):
            validate_torch_training_case(case)


def _case(*, status: str = "pending") -> dict:
    return {
        "case_id": "tiny-case",
        "status": status,
        "reason": "one or more replay parity gates have not matched",
        "context": [0, 1, 2],
        "target": 1,
        "learning_rate": 0.02,
        "steps": 2,
    }


if __name__ == "__main__":
    unittest.main()
