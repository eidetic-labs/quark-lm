from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curriculum import build_curriculum, read_jsonl
from glossary_probes import DEFAULT_OUTPUT as DEFAULT_GLOSSARY_PROBES
from self_improve import evaluate_responder


def current_admission_count() -> int:
    return len(read_jsonl(ROOT / "corpus" / "admissions.jsonl"))


class SelfImproveTest(unittest.TestCase):
    def test_responder_summary_tracks_all_eval_sets(self) -> None:
        curriculum = build_curriculum(seed=3)
        summary = evaluate_responder(curriculum.train_text)

        self.assertEqual(summary["qa"]["exact_rate"], 1.0)
        self.assertEqual(summary["unknowns"]["exact_rate"], 1.0)
        self.assertEqual(summary["heldout"]["exact_rate"], 1.0)
        self.assertEqual(summary["paraphrases"]["exact_rate"], 1.0)
        self.assertEqual(summary["owner"]["exact_rate"], 1.0)
        self.assertEqual(summary["self"]["exact_rate"], 1.0)
        self.assertEqual(summary["learning"]["exact_rate"], 1.0)
        self.assertEqual(summary["admissions"]["count"], current_admission_count() * 4)
        self.assertEqual(summary["admissions"]["exact_rate"], 1.0)
        self.assertEqual(
            summary["admission_paraphrases"]["count"],
            current_admission_count() * 7,
        )
        self.assertEqual(summary["admission_paraphrases"]["exact_rate"], 1.0)
        self.assertEqual(
            summary["glossary"]["count"],
            len(read_jsonl(DEFAULT_GLOSSARY_PROBES)),
        )
        self.assertEqual(summary["glossary"]["exact_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
