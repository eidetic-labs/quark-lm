from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_improve import audit_exact_promotion, audit_forgetting, promotion_gate


class SelfImprovePromotionTest(unittest.TestCase):
    def test_forgetting_audit_detects_regression(self) -> None:
        previous = {
            "responder": {"qa": {"count": 8, "exact": 8, "exact_rate": 1.0}},
            "answer_model": {
                "final": {"qa": {"count": 8, "exact": 8, "exact_rate": 1.0}}
            },
            "answer_decoder": {
                "final": {"qa": {"count": 8, "exact": 8, "exact_rate": 1.0}}
            },
        }
        current = {
            "responder": {"qa": {"count": 8, "exact": 8, "exact_rate": 1.0}},
            "answer_model": {
                "final": {"qa": {"count": 8, "exact": 8, "exact_rate": 1.0}}
            },
            "answer_decoder": {
                "final": {"qa": {"count": 8, "exact": 7, "exact_rate": 0.875}}
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "previous.json"
            path.write_text(json.dumps(previous), encoding="utf-8")

            audit = audit_forgetting(current, path)

            self.assertFalse(audit["passed"])
            failed = [check for check in audit["checks"] if not check["passed"]]
            self.assertEqual(failed[0]["component"], "answer_decoder")

    def test_promotion_gate_detects_non_exact_eval(self) -> None:
        report = {
            "responder": {"qa": {"count": 1, "exact": 1}},
            "answer_model": {"final": {"qa": {"count": 1, "exact": 0}}},
            "answer_decoder": {"final": {"qa": {"count": 1, "exact": 1}}},
            "admission_probe_audit": {"passed": True},
            "glossary_probe_audit": {"passed": True},
            "prompt_leakage_audit": {
                "heldout": {"passed": True},
                "owner_heldout": {"passed": True},
            },
            "forgetting_audit": {"passed": True},
        }
        report["exact_eval_audit"] = audit_exact_promotion(report)

        gate = promotion_gate(report)

        self.assertFalse(report["exact_eval_audit"]["passed"])
        self.assertFalse(gate["passed"])


if __name__ == "__main__":
    unittest.main()
