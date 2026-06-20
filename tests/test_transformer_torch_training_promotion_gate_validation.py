from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_promotion_gate import (  # noqa: E402
    build_torch_training_backend_promotion_gate,
)
from transformer_torch_training_promotion_gate_validation import (  # noqa: E402
    validate_torch_training_backend_promotion_gate,
)


class TransformerTorchTrainingPromotionGateValidationTests(unittest.TestCase):
    def test_validator_rejects_extra_promotion_gate_key(self) -> None:
        gate = _gate()
        gate["unvalidated_extra_field"] = "drift"

        with self.assertRaisesRegex(ValueError, "promotion gate keys"):
            validate_torch_training_backend_promotion_gate(
                gate,
                closed_world_boundary=_boundary(),
            )

    def test_validator_rejects_extra_promotion_check_key(self) -> None:
        gate = _gate()
        gate["checks"][0]["unvalidated_extra_field"] = "drift"

        with self.assertRaisesRegex(ValueError, "check keys"):
            validate_torch_training_backend_promotion_gate(
                gate,
                closed_world_boundary=_boundary(),
            )

    def test_validator_rejects_stale_promotion_gate_blockers(self) -> None:
        gate = _gate()
        gate["blockers"] = ["fixture_scope_only"]

        with self.assertRaisesRegex(ValueError, "blockers"):
            validate_torch_training_backend_promotion_gate(
                gate,
                closed_world_boundary=_boundary(),
            )

    def test_validator_rejects_stale_promotion_gate_future_gates(self) -> None:
        gate = _gate()
        gate["required_future_gates"] = ["general_training_backend_gate"]

        with self.assertRaisesRegex(ValueError, "future gates"):
            validate_torch_training_backend_promotion_gate(
                gate,
                closed_world_boundary=_boundary(),
            )

    def test_validator_rejects_stale_promotion_gate_boundary_state(self) -> None:
        boundary = _boundary()
        boundary["pretrained_weights_imported"] = True

        with self.assertRaisesRegex(ValueError, "boundary status"):
            validate_torch_training_backend_promotion_gate(
                _gate(),
                closed_world_boundary=boundary,
            )

    def test_validator_rejects_non_boolean_parity_status(self) -> None:
        gate = _gate()
        gate["parity_evidence_matched"] = "false"

        with self.assertRaisesRegex(ValueError, "parity status"):
            validate_torch_training_backend_promotion_gate(
                gate,
                closed_world_boundary=_boundary(),
            )

    def test_validator_rejects_stale_parity_evidence_flag(self) -> None:
        gate = _gate()
        gate["checks"][0]["passed"] = False
        gate["blockers"] = [
            "training_parity_report",
            "fixture_scope_only",
            "model_quality_gate",
        ]

        with self.assertRaisesRegex(ValueError, "parity evidence"):
            validate_torch_training_backend_promotion_gate(
                gate,
                closed_world_boundary=_boundary(),
            )


def _gate() -> dict:
    return build_torch_training_backend_promotion_gate(
        candidate={
            "backend": {"parity_status": "matched"},
            "training_replay_parity_gate": {
                "status": "training_replay_parity_matched",
                "passed": True,
                "parity_status": "matched",
            },
        },
        report={"passed": True},
        closed_world_boundary=_boundary(),
    )


def _boundary() -> dict:
    return {
        "runtime_library_allowed": True,
        "training_text_source": "admitted_curriculum",
        "learned_assets_imported": False,
        "training_data_imported": False,
        "pretrained_weights_imported": False,
        "pretrained_tokenizer_imported": False,
        "external_embeddings_imported": False,
    }


if __name__ == "__main__":
    unittest.main()
