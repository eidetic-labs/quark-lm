from __future__ import annotations

import json
import unittest
from pathlib import Path

import support  # noqa: F401  (inserts src/ onto sys.path)
from answer_prompt_templates import prompt_templates
from answer_unknown_augmentation import augment_unknown_examples

ROOT = Path(__file__).resolve().parents[1]


class AnswerUnknownAugmentationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = json.loads((ROOT / "corpus" / "grammar.json").read_text("utf-8"))
        self.examples = augment_unknown_examples(self.grammar)

    def test_generates_many_unknown_examples(self) -> None:
        # Far more than the ~4 declared unknown pairs, so the model can learn the
        # general "unseen pair -> unknown" pattern instead of memorizing a few.
        self.assertGreater(len(self.examples), 50)
        self.assertTrue(all(e.target == " unknown." for e in self.examples))
        self.assertTrue(all(e.source.startswith("augmented:") for e in self.examples))

    def test_no_eval_prompt_leakage(self) -> None:
        eval_prompts: set[str] = set()
        for path in sorted((ROOT / "evals").glob("*.jsonl")):
            for line in path.read_text("utf-8").splitlines():
                if line.strip():
                    eval_prompts.add(json.loads(line)["prompt"])
        augmented_prompts = {example.prompt for example in self.examples}
        leaks = sorted(augmented_prompts & eval_prompts)
        self.assertEqual(leaks, [], f"augmented prompts leak into eval sets: {leaks[:5]}")

    def test_excludes_real_and_declared_unknown_pairs(self) -> None:
        forbidden: set[str] = set()
        for fact in self.grammar["story_facts"] + self.grammar["unknown_facts"]:
            for kind in ("place", "color"):
                forbidden.update(prompt_templates(fact["person"], fact["object"], kind))
                forbidden.update(
                    prompt_templates(fact["person"], fact["object"], kind, "bridge")
                )
        augmented_prompts = {example.prompt for example in self.examples}
        self.assertEqual(augmented_prompts & forbidden, set())


if __name__ == "__main__":
    unittest.main()
