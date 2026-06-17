from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curriculum import build_curriculum, read_jsonl, write_curriculum
from glossary_probes import DEFAULT_OUTPUT as DEFAULT_GLOSSARY_PROBES


def current_admission_count() -> int:
    return len(read_jsonl(ROOT / "corpus" / "admissions.jsonl"))


class CurriculumTest(unittest.TestCase):
    def test_builds_training_and_validation_text(self) -> None:
        curriculum = build_curriculum(seed=3)
        self.assertIn("glossary:", curriculum.train_text)
        self.assertIn("question: where is mia's ball?", curriculum.train_text)
        self.assertIn("fact: ivy's map is on the shelf.", curriculum.train_text)
        self.assertIn("fact: the map belongs to ivy.", curriculum.train_text)
        self.assertIn("event: I learned something new: teacher's tree is near the garden.", curriculum.train_text)
        self.assertIn("event: now teacher's tree is part of my training data.", curriculum.train_text)
        self.assertIn("event: I learned something new: child's bag is near the shelf.", curriculum.train_text)
        self.assertIn("event: now child's bag is part of my training data.", curriculum.train_text)
        self.assertIn("event: I learned something new: ivy's stone is under the table.", curriculum.train_text)
        self.assertIn("event: now ivy's stone is part of my training data.", curriculum.train_text)
        self.assertIn("fact: self dataset is the admitted corpus.", curriculum.train_text)
        self.assertIn(
            "fact: learning new_data means it becomes training data after corpus admission.",
            curriculum.train_text,
        )
        self.assertIn(
            "fact: self diagnosis_source is self-improvement reports.",
            curriculum.train_text,
        )
        self.assertIn(
            "fact: learning repair_action means from report evidence.",
            curriculum.train_text,
        )
        self.assertNotIn("question: where is ivy's map?", curriculum.train_text)
        self.assertIn("answer: unknown.", curriculum.train_text)
        self.assertGreater(curriculum.manifest["train_chars"], 1000)
        self.assertGreater(curriculum.manifest["valid_chars"], 100)
        self.assertEqual(curriculum.manifest["heldout_probe_facts"], 4)
        self.assertEqual(curriculum.manifest["admitted_facts"], current_admission_count())
        self.assertEqual(curriculum.manifest["unknown_owner_objects"], 2)
        self.assertEqual(curriculum.manifest["self_facts"], 7)
        self.assertEqual(curriculum.manifest["learning_rules"], 4)
        self.assertGreater(len(read_jsonl(DEFAULT_GLOSSARY_PROBES)), 0)

    def test_writes_curriculum_files(self) -> None:
        curriculum = build_curriculum(seed=3)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            write_curriculum(curriculum, output)
            self.assertTrue((output / "train.txt").exists())
            self.assertTrue((output / "valid.txt").exists())
            self.assertTrue((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
