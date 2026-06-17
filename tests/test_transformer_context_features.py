from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from support.char_model import char_model_fixture, context_and_target, transformer_config
from support.core import TinyTransformerLM, context_before, flatten_scalars


class TransformerContextFeatureTest(unittest.TestCase):
    def test_context_mean_transformer_trains_and_round_trips(self) -> None:
        tokenizer, ids, _config, model = char_model_fixture(seed=13, use_context_mean=True)
        context, target = context_and_target(ids, model.config, tokenizer)
        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        self.assertTrue(loaded.config.use_context_mean)
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_context_projection_starts_as_baseline_and_round_trips(self) -> None:
        tokenizer, ids, _config, baseline = char_model_fixture(seed=14)
        model = TinyTransformerLM.init_random(
            transformer_config(tokenizer, seed=14, use_context_projection=True)
        )
        context, target = context_and_target(ids, model.config, tokenizer)

        for expected, actual in zip(baseline.predict(context), model.predict(context)):
            self.assertAlmostEqual(expected, actual)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)
        projection_values = [
            value.data
            for value in (
                flatten_scalars(model.context_projection_w)
                + flatten_scalars(model.context_projection_b)
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_context_projection)
        self.assertIn("context_projection_w", weights)
        self.assertIn("context_projection_b", weights)
        self.assertTrue(any(abs(value) > 0.0 for value in projection_values))
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_prompt_prefix_projection_starts_as_baseline_and_round_trips(self) -> None:
        tokenizer, ids, _config, baseline = char_model_fixture(seed=16)
        model = TinyTransformerLM.init_random(
            transformer_config(tokenizer, seed=16, use_prompt_prefix_projection=True)
        )
        context, target = context_and_target(ids, model.config, tokenizer)

        for expected, actual in zip(baseline.predict(context), model.predict(context)):
            self.assertAlmostEqual(expected, actual)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)
        projection_values = [
            value.data
            for value in (
                flatten_scalars(model.prompt_prefix_projection_w)
                + flatten_scalars(model.prompt_prefix_projection_b)
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_prompt_prefix_projection)
        self.assertIn("prompt_prefix_projection_w", weights)
        self.assertIn("prompt_prefix_projection_b", weights)
        self.assertTrue(any(abs(value) > 0.0 for value in projection_values))
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_prompt_position_projection_starts_as_baseline_and_round_trips(self) -> None:
        tokenizer, ids, _config, baseline = char_model_fixture(seed=18)
        model = TinyTransformerLM.init_random(
            transformer_config(tokenizer, seed=18, use_prompt_position_projection=True)
        )
        context, target = context_and_target(ids, model.config, tokenizer)

        for expected, actual in zip(baseline.predict(context), model.predict(context)):
            self.assertAlmostEqual(expected, actual)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)
        projection_values = [
            value.data
            for value in (
                flatten_scalars(model.prompt_position_projection_w)
                + flatten_scalars(model.prompt_position_projection_b)
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_prompt_position_projection)
        self.assertEqual(loaded.config.prompt_position_projection_scale, 1.0)
        self.assertIn("prompt_position_projection_w", weights)
        self.assertIn("prompt_position_projection_b", weights)
        self.assertTrue(any(abs(value) > 0.0 for value in projection_values))
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_prompt_position_projection_scale_changes_nonzero_residual(self) -> None:
        tokenizer, ids, _config, base = char_model_fixture(
            seed=19,
            use_prompt_position_projection=True,
        )
        scaled = TinyTransformerLM.init_random(
            transformer_config(
                tokenizer,
                seed=19,
                use_prompt_position_projection=True,
                prompt_position_projection_scale=3.0,
            )
        )
        base.prompt_position_projection_b[0].data = 0.25
        scaled.prompt_position_projection_b[0].data = 0.25
        context = context_before(ids, 4, base.config.context_size, tokenizer.pad_id)

        base_hidden = base.final_hidden(context)
        scaled_hidden = scaled.final_hidden(context)
        base_probs = base.predict(context)
        scaled_probs = scaled.predict(context)

        self.assertNotEqual(base_hidden, scaled_hidden)
        self.assertNotEqual(base_probs, scaled_probs)

    def test_final_hidden_matches_forward_logits(self) -> None:
        tokenizer, ids, _config, model = char_model_fixture(
            seed=19,
            use_prompt_position_projection=True,
        )
        context = context_before(ids, 4, model.config.context_size, tokenizer.pad_id)

        hidden = model.final_hidden(context)
        expected_logits = model._forward_floats(context)
        actual_logits = []
        for output_index, bias in enumerate(model.bout):
            total = bias.data
            for input_index, value in enumerate(hidden):
                total += value * model.wout[input_index][output_index].data
            actual_logits.append(total)

        self.assertEqual(len(hidden), model.config.embedding_dim)
        for expected, actual in zip(expected_logits, actual_logits):
            self.assertAlmostEqual(expected, actual)

    def test_prompt_attention_summary_starts_as_baseline_and_round_trips(self) -> None:
        tokenizer, ids, _config, baseline = char_model_fixture(seed=15)
        model = TinyTransformerLM.init_random(
            transformer_config(tokenizer, seed=15, use_prompt_attention_summary=True)
        )
        context, target = context_and_target(ids, model.config, tokenizer)

        for expected, actual in zip(baseline.predict(context), model.predict(context)):
            self.assertAlmostEqual(expected, actual)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)
        projection_values = [
            value.data
            for value in (
                flatten_scalars(model.prompt_summary_w)
                + flatten_scalars(model.prompt_summary_b)
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_prompt_attention_summary)
        self.assertIn("prompt_summary_query", weights)
        self.assertIn("prompt_summary_w", weights)
        self.assertIn("prompt_summary_b", weights)
        self.assertTrue(any(abs(value) > 0.0 for value in projection_values))
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)


if __name__ == "__main__":
    unittest.main()
