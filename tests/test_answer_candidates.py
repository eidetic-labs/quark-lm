"""Per-type candidate menus route correctly and cover every eval target.

The coverage + format guard for the eval de-contamination: every probe across all
eval sets must (a) map to a known answer_type and (b) have its gold target present
in that type's corpus-answer-space menu. If this fails, either a question form is
unrouted or a menu string does not match the eval format -- both must be fixed
before per-type menus replace the global pool.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import support  # noqa: F401  (inserts src/ onto sys.path)
from answer_candidates import ABSTAIN, answer_type_for, candidates_by_type, menu_for
from corpus_responder import CorpusResponder

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_TEXT = REPO_ROOT / "build" / "train.txt"
EVAL_DIR = REPO_ROOT / "evals"

SMALL_CORPUS = (
    "fact: mia's ball is under the box.\n"
    "fact: mia's ball color is red.\n"
    "fact: the ball belongs to mia.\n"
)


class AnswerCandidatesTest(unittest.TestCase):
    def test_routing_and_menus_from_small_corpus(self) -> None:
        responder = CorpusResponder.train_from_text(SMALL_CORPUS)
        menus = candidates_by_type(responder)
        self.assertEqual(answer_type_for("question: where is mia's ball?\nanswer:"), "place")
        self.assertEqual(answer_type_for("question: what color is mia's ball?\nanswer:"), "color")
        self.assertEqual(answer_type_for("question: who has the ball?\nanswer:"), "owner")
        self.assertIn(" under the box.", menus["place"])
        self.assertIn(ABSTAIN, menus["place"])  # abstain stays in every menu
        self.assertIn(" red.", menus["color"])
        self.assertIn(" mia.", menus["owner"])

    def test_every_eval_target_is_in_its_type_menu(self) -> None:
        if not TRAIN_TEXT.exists() or not EVAL_DIR.exists():
            self.skipTest("corpus build/train.txt or evals/ not present")
        responder = CorpusResponder.load_train_text(TRAIN_TEXT)
        menus = candidates_by_type(responder)
        misses: list[tuple[str, str, str | None, str]] = []
        for path in sorted(EVAL_DIR.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                answer_type = answer_type_for(record["prompt"])
                menu = menu_for(record["prompt"], menus)
                if answer_type is None or record["target"] not in menu:
                    misses.append((path.name, record["id"], answer_type, record["target"]))
        self.assertEqual(misses, [], f"unrouted/mismatched probes: {misses[:12]}")


if __name__ == "__main__":
    unittest.main()
