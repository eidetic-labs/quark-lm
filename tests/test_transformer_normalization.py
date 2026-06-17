from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from support.char_model import char_model_fixture, context_and_target
from support.core import TinyTransformerLM, context_before


class TransformerNormalizationTest(unittest.TestCase):
    def test_layer_normalized_transformer_trains_and_round_trips(self) -> None:
        tokenizer, ids, _config, model = char_model_fixture(seed=9, use_layer_norm=True)
        context, target = context_and_target(ids, model.config, tokenizer)
        before = model.nll(context, target)
        for _ in range(30):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_layer_norm)
        self.assertIn("ln1_gain", weights)
        self.assertIn("ln2_gain", weights)
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_pre_layer_normalized_transformer_trains_and_round_trips(self) -> None:
        tokenizer, ids, _config, model = char_model_fixture(seed=11, use_pre_layer_norm=True)
        _baseline_tokenizer, _baseline_ids, _baseline_config, baseline = char_model_fixture(seed=11)
        context, target = context_and_target(ids, model.config, tokenizer)
        before = model.nll(context, target)
        params = model.parameters()

        self.assertNotEqual(baseline.final_hidden(context), model.final_hidden(context))
        self.assertTrue(any(param is model.ln1_gain[0] for param in params))
        self.assertTrue(any(param is model.ln2_gain[0] for param in params))
        self.assertTrue(any(param is model.final_ln_gain[0] for param in params))

        for _ in range(30):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_pre_layer_norm)
        self.assertIn("final_ln_gain", weights)
        self.assertIn("final_ln_bias", weights)
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_pre_layer_norm_scalar_and_float_forward_match(self) -> None:
        tokenizer, ids, _config, model = char_model_fixture(
            seed=12,
            num_layers=2,
            use_pre_layer_norm=True,
        )
        context = context_before(ids, 4, model.config.context_size, tokenizer.pad_id)
        scalar_logits = [value.data for value in model._forward_scalars(context)]
        float_logits = model._forward_floats(context)

        for expected, actual in zip(float_logits, scalar_logits):
            self.assertAlmostEqual(expected, actual)

    def test_legacy_transformer_checkpoint_defaults_pre_layer_norm_off(self) -> None:
        _tokenizer, _ids, _config, model = char_model_fixture(seed=13)
        payload = model.to_dict(_tokenizer)
        payload["config"].pop("use_pre_layer_norm", None)
        payload["weights"].pop("final_ln_gain", None)
        payload["weights"].pop("final_ln_bias", None)

        loaded, _loaded_tokenizer = TinyTransformerLM.from_dict(payload)

        self.assertFalse(loaded.config.use_pre_layer_norm)
        self.assertEqual(
            [value.data for value in loaded.final_ln_gain],
            [1.0 for _ in range(loaded.config.embedding_dim)],
        )


if __name__ == "__main__":
    unittest.main()
