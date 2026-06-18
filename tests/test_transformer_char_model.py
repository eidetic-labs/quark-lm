from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from support.char_model import char_model_fixture
from support.core import (
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
    context_before,
    continuation_nll,
)


class TransformerCharModelCoreTest(unittest.TestCase):
    def test_train_step_updates_random_transformer_weights(self) -> None:
        text = "question: where is mia's ball?\nanswer: under the box.\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=6,
            embedding_dim=4,
            feedforward_dim=8,
            seed=3,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(ids, 12, config.context_size, tokenizer.pad_id)
        target = ids[12]
        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.04)
        after = model.nll(context, target)

        self.assertGreater(before, after)

    def test_v051_architecture_options_forward_and_round_trip(self) -> None:
        text = "abcd abcd\n"
        tokenizer = CharTokenizer.train(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=54,
            attention_heads=2,
            use_pre_layer_norm=True,
            use_rms_norm=True,
            use_gated_mlp=True,
            tie_output_embeddings=True,
            use_rotary_positions=True,
            use_kv_cache_path=True,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(tokenizer.encode(text), 4, config.context_size, tokenizer.pad_id)
        probs = model.predict(context)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer, {"test": "v0.51"})
            loaded, loaded_tokenizer = TinyTransformerLM.load(path)

        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertTrue(loaded.config.tie_output_embeddings)
        self.assertEqual(loaded.config.attention_heads, 2)
        self.assertIsNotNone(loaded_tokenizer)

    def test_checkpoint_round_trip_includes_corpus_tokenizer(self) -> None:
        tokenizer, _ids, config, model = char_model_fixture(seed=1)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, loaded_tokenizer = TinyTransformerLM.load(path)

        self.assertIsNotNone(loaded_tokenizer)
        self.assertEqual(loaded.config.vocab_size, config.vocab_size)
        self.assertEqual(loaded_tokenizer.tokens, tokenizer.tokens)  # type: ignore[union-attr]

    def test_tokenizer_extension_and_vocab_resize_make_new_tokens_trainable(self) -> None:
        tokenizer, ids, config, model = char_model_fixture("abc abc\n", seed=5)
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        old_logits = model._forward_floats(context)

        extended = tokenizer.extend("abd!")
        model.resize_vocab(extended.vocab_size)
        target = extended.stoi["!"]

        self.assertEqual(extended.encode("abc"), tokenizer.encode("abc"))
        self.assertEqual(model.config.vocab_size, extended.vocab_size)
        self.assertEqual(model._forward_floats(context)[: tokenizer.vocab_size], old_logits)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.04)
        after = model.nll(context, target)

        self.assertGreater(before, after)

    def test_vocab_resize_supports_tied_output_embeddings(self) -> None:
        tokenizer, ids, config, model = char_model_fixture(
            "abc abc\n",
            seed=6,
            tie_output_embeddings=True,
        )
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        extended = tokenizer.extend("abc?")

        model.resize_vocab(extended.vocab_size)
        probs = model.predict(context)

        self.assertEqual(len(model.token_embeddings), extended.vocab_size)
        self.assertEqual(len(probs), extended.vocab_size)
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_transformer_learns_prompt_to_answer_continuation(self) -> None:
        prompt = "q:\na:"
        target = " ok"
        tokenizer = CharTokenizer.train(prompt + target)
        ids = tokenizer.encode(prompt + target)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=6,
            embedding_dim=4,
            feedforward_dim=8,
            seed=1,
        )
        model = TinyTransformerLM.init_random(config)
        answer_positions = range(len(prompt), len(ids))
        before = continuation_nll(model, tokenizer, prompt, target)

        for _ in range(40):
            for position in answer_positions:
                context = context_before(
                    ids,
                    position,
                    config.context_size,
                    tokenizer.pad_id,
                )
                model.train_step(context, ids[position], learning_rate=0.05)

        after = continuation_nll(model, tokenizer, prompt, target)

        self.assertGreater(before, after)
        self.assertEqual(model.generate(tokenizer, prompt, len(target)), target)


if __name__ == "__main__":
    unittest.main()
