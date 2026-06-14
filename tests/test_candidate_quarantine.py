from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.candidate_quarantine import (
    build_candidate_quarantine_manifest,
    candidate_id_from_parts,
    candidate_quarantine_summary,
    candidate_record,
    transition_candidate,
    write_candidate_quarantine,
)


class CandidateQuarantineTest(unittest.TestCase):
    def test_candidate_record_has_stable_id_and_default_quarantine(self) -> None:
        first = candidate_record(
            "lesson",
            "self-diagnosis",
            prompt="What did QuarkLM learn?",
            target="It learned an admitted fact.",
        )
        second_id = candidate_id_from_parts(
            "lesson",
            "self-diagnosis",
            "What did QuarkLM learn?",
            "It learned an admitted fact.",
        )

        self.assertEqual(first["candidate_id"], second_id)
        self.assertEqual(first["state"], "quarantined")
        self.assertEqual(first["candidate_type"], "lesson")

    def test_candidate_lifecycle_transitions_are_validated(self) -> None:
        record = candidate_record("probe", "generated-probes", prompt="p", target="t")
        verified = transition_candidate(
            record,
            "verified",
            evidence=[{"check": "exact_responder", "passed": True}],
            note="Verifier accepted the closed-world answer.",
        )
        admitted = transition_candidate(
            verified,
            "admitted",
            admission_id="admission-001",
        )

        self.assertEqual(admitted["state"], "admitted")
        self.assertEqual(admitted["admission_id"], "admission-001")
        self.assertEqual(len(admitted["transitions"]), 2)
        self.assertEqual(admitted["evidence"][0]["check"], "exact_responder")

    def test_invalid_transition_is_rejected(self) -> None:
        record = candidate_record("repair_proposal", "diagnosis", proposal="Try repair.")

        with self.assertRaises(ValueError):
            transition_candidate(record, "admitted")

    def test_manifest_summary_counts_candidate_states(self) -> None:
        lesson = candidate_record("lesson", "self-diagnosis", prompt="p", target="t")
        proposal = candidate_record(
            "repair_proposal",
            "transformer-screen",
            proposal="Adjust replay floors.",
            state="proposed",
        )
        manifest = build_candidate_quarantine_manifest(
            "self-improvement-answer-cycle",
            "attempt-001",
            [lesson, proposal],
            candidate_sources=["self_diagnose.py"],
        )
        summary = candidate_quarantine_summary(manifest)

        self.assertEqual(manifest["status"], "contains_quarantined_candidates")
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["by_state"]["quarantined"], 1)
        self.assertEqual(summary["by_state"]["proposed"], 1)
        self.assertEqual(summary["not_training_eligible_count"], 2)
        self.assertFalse(summary["candidate_records_are_training_data"])

    def test_manifest_can_be_written_as_json(self) -> None:
        manifest = build_candidate_quarantine_manifest("transformer-answer-train", "run-001")

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "candidate_quarantine.json"
            write_candidate_quarantine(path, manifest)
            written = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(written["kind"], "candidate_quarantine_manifest")
        self.assertEqual(written["status"], "empty_no_candidates")


if __name__ == "__main__":
    unittest.main()
