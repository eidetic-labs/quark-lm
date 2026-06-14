from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.self_diagnose import diagnose_report, main, parse_args


class SelfDiagnoseTest(unittest.TestCase):
    def test_clean_report_recommends_promotion_or_expansion(self) -> None:
        report = {
            "promotion_gate": {
                "passed": True,
                "checks": [{"name": "exact_eval_audit", "passed": True}],
            },
            "answer_model": {"final": {"qa": {"failed_records": []}}},
            "answer_decoder": {"final": {"qa": {"failed_records": []}}},
        }

        diagnosis = diagnose_report(report)

        self.assertFalse(diagnosis["uses_external_model"])
        self.assertEqual(diagnosis["blocker_count"], 0)
        self.assertEqual(
            diagnosis["recommended_actions"][0]["action"],
            "promote_or_expand_corpus",
        )

    def test_unknown_paraphrase_failure_recommends_unknown_bridge_lessons(self) -> None:
        report = {
            "promotion_gate": {
                "passed": False,
                "checks": [
                    {"name": "forgetting_audit", "passed": False},
                    {"name": "exact_eval_audit", "passed": False},
                ],
            },
            "answer_model": {
                "final": {
                    "paraphrases": {
                        "failed_records": [
                            {
                                "id": "para-noah-ball-place",
                                "prediction": " near the shelf.",
                                "target": " unknown.",
                            }
                        ]
                    }
                }
            },
            "answer_decoder": {"final": {}},
        }

        diagnosis = diagnose_report(report)
        actions = {action["action"] for action in diagnosis["recommended_actions"]}

        self.assertEqual(diagnosis["blocker_count"], 3)
        self.assertIn("add_or_rebalance_unknown_bridge_lessons", actions)
        self.assertIn("inspect_regressive_eval", actions)
        self.assertIn("inspect_failed_records", actions)

    def test_probe_failures_recommend_regeneration_commands(self) -> None:
        report = {
            "promotion_gate": {
                "passed": False,
                "checks": [
                    {"name": "admission_probe_audit", "passed": False},
                    {"name": "glossary_probe_audit", "passed": False},
                ],
            },
            "answer_model": {"final": {}},
            "answer_decoder": {"final": {}},
        }

        diagnosis = diagnose_report(report)
        commands = {
            action["action"]: action["command"]
            for action in diagnosis["recommended_actions"]
        }

        self.assertEqual(
            commands["regenerate_admission_probes"],
            "PYTHONPATH=src python3 -m closed_world_lm.admission_probes",
        )
        self.assertEqual(
            commands["regenerate_glossary_probes"],
            "PYTHONPATH=src python3 -m closed_world_lm.glossary_probes",
        )

    def test_json_flag_can_be_stdout_only_or_write_path(self) -> None:
        args = parse_args(["report.json", "--json"])

        self.assertEqual(args.json, Path("-"))

    def test_json_path_writes_diagnosis_file(self) -> None:
        report = {
            "promotion_gate": {"passed": True, "checks": []},
            "answer_model": {"final": {}},
            "answer_decoder": {"final": {}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report_path = tmp_path / "report.json"
            output_path = tmp_path / "diagnosis.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")

            with redirect_stdout(StringIO()):
                exit_code = main([str(report_path), "--json", str(output_path)])

            self.assertEqual(exit_code, 0)
            self.assertIn("promote_or_expand_corpus", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
