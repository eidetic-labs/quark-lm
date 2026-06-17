from __future__ import annotations

import unittest

from support.char_model import char_model_fixture
from support.core import (
    ANSWER_TERMINATOR,
    CharTokenizer,
    GenerationConfig,
    TinyTransformerLM,
    TransformerConfig,
    generation_distribution,
    score_transformer_records,
)


class TransformerGenerationControlsTest(unittest.TestCase):
    def test_generation_controls_emit_trace_and_cache_metadata(self) -> None:
        tokenizer, _ids, _config, model = char_model_fixture(seed=55, use_kv_cache_path=True)

        generation = model.generate_with_trace(
            tokenizer,
            "abc",
            3,
            GenerationConfig(
                temperature=0.7,
                top_k=2,
                top_p=0.9,
                repetition_penalty=1.1,
                trace_top_tokens=2,
                use_kv_cache=True,
            ),
        )

        self.assertLessEqual(len(generation["trace"]), 3)
        self.assertTrue(generation["cache"]["enabled"])
        self.assertLessEqual(len(generation["trace"][0]["top_tokens"]), 2)

    def test_generation_distribution_applies_top_k_and_repetition_penalty(self) -> None:
        probs = generation_distribution(
            [0.6, 0.3, 0.1],
            [0],
            GenerationConfig(top_k=2, repetition_penalty=3.0),
        )

        self.assertEqual(probs[2], 0.0)
        self.assertGreater(probs[1], probs[0])
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_transformer_eval_scoring_returns_replayable_trace_records(self) -> None:
        example = {"id": "one", "prompt": "q:\na:", "target": " a."}
        text = example["prompt"] + example["target"] + ANSWER_TERMINATOR
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=56,
            )
        )

        records = score_transformer_records(
            model,
            tokenizer,
            [example],
            max_new_chars=2,
            generation_config=GenerationConfig(trace_top_tokens=2),
            candidates=[example["target"]],
        )

        self.assertEqual(records[0]["id"], "one")
        self.assertIn("generation_trace", records[0])
        self.assertIn("candidate_scores", records[0])

    def test_generate_uses_character_tokenizer(self) -> None:
        tokenizer, _ids, _config, model = char_model_fixture(seed=5)

        generated = model.generate(tokenizer, "abc", max_new_chars=3)

        self.assertIsInstance(generated, str)
        self.assertLessEqual(len(generated), 3)

    def test_generate_can_stop_at_admitted_terminator(self) -> None:
        text = "a\n"
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=2,
                embedding_dim=4,
                feedforward_dim=8,
                seed=6,
            )
        )
        newline_id = tokenizer.stoi[ANSWER_TERMINATOR]
        model.bout[newline_id].data = 5.0

        generated = model.generate(
            tokenizer,
            "a",
            max_new_chars=4,
            stop_at=ANSWER_TERMINATOR,
        )

        self.assertEqual(generated, "")


if __name__ == "__main__":
    unittest.main()
