from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.answer_decoder import (
    DECODER_SELF_LEARNING_REPEATS,
    AnswerDecoder,
    build_decoder,
    decoder_training_pool,
)
from closed_world_lm.answer_model import AnswerExample, prompt_templates


class AnswerDecoderTest(unittest.TestCase):
    def test_decoder_generates_a_small_answer_set(self) -> None:
        examples = [
            AnswerExample(
                prompt=prompt_templates("mia", "ball", "place")[0],
                target=" under the box.",
                source="test",
            ),
            AnswerExample(
                prompt=prompt_templates("noah", "cup", "place")[0],
                target=" on the table.",
                source="test",
            ),
        ]
        model = build_decoder(examples, seed=1, max_answer_chars=24)
        for _ in range(260):
            for example in examples:
                model.train_example(example, learning_rate=0.08)

        self.assertEqual(model.generate(examples[0].prompt), examples[0].target)
        self.assertEqual(model.generate(examples[1].prompt), examples[1].target)

    def test_checkpoint_round_trip(self) -> None:
        examples = [
            AnswerExample(
                prompt=prompt_templates("mia", "ball", "color")[0],
                target=" red.",
                source="test",
            )
        ]
        model = build_decoder(examples, seed=1, max_answer_chars=12)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test-temp-decoder.json"
            model.save(path)
            loaded = AnswerDecoder.load(path)
            self.assertEqual(loaded.config.labels, model.config.labels)

    def test_decoder_can_generate_long_learning_answer(self) -> None:
        target = " it becomes training data after corpus admission."
        self.assertGreater(len(target), 32)
        examples = [
            AnswerExample(
                prompt="question: what happens when you learn something new?\nanswer:",
                target=target,
                source="test",
            )
        ]
        model = build_decoder(examples, seed=1, max_answer_chars=64)
        for _ in range(160):
            model.train_example(examples[0], learning_rate=0.08)

        self.assertEqual(model.generate(examples[0].prompt), target)

    def test_training_pool_upweights_learning_examples(self) -> None:
        ordinary = AnswerExample(
            prompt=prompt_templates("mia", "ball", "place")[0],
            target=" under the box.",
            source="qa:place",
        )
        unknown = AnswerExample(
            prompt="question: where is noah's ball?\nanswer:",
            target=" unknown.",
            source="unknown:place",
        )
        learning = AnswerExample(
            prompt="question: what happens when you learn something new?\nanswer:",
            target=" it becomes training data after corpus admission.",
            source="qa:learning",
        )
        glossary = AnswerExample(
            prompt="question: what does corpus mean?\nanswer:",
            target=" the admitted training data.",
            source="qa:glossary",
        )

        pool = decoder_training_pool([ordinary, unknown, learning, glossary])

        self.assertGreater(pool.count(learning), pool.count(ordinary))
        self.assertGreater(pool.count(glossary), pool.count(ordinary))
        self.assertGreater(pool.count(ordinary), pool.count(unknown))
        self.assertGreaterEqual(pool.count(learning), DECODER_SELF_LEARNING_REPEATS)


if __name__ == "__main__":
    unittest.main()
