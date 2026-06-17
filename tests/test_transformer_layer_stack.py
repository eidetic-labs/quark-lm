from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from support.char_model import char_model_fixture, context_and_target
from support.core import TinyTransformerLM, context_before


class TransformerLayerStackTest(unittest.TestCase):
    def test_multi_layer_transformer_trains_and_round_trips(self) -> None:
        tokenizer, ids, _config, model = char_model_fixture(seed=10, num_layers=2)
        context, target = context_and_target(ids, model.config, tokenizer)
        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertEqual(loaded.config.num_layers, 2)
        self.assertEqual(len(weights["extra_layers"]), 1)
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_multi_layer_top_layer_update_freezes_lower_layer(self) -> None:
        tokenizer, ids, config, model = char_model_fixture(seed=12, num_layers=2)
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        lower_before = model.wq[0][0].data
        top_before = model.extra_blocks[0]["wq"][0][0].data
        head_before = model.wout[0][target].data
        model.freeze_lower_layers_for_updates = True

        for _ in range(20):
            model.train_step(
                context,
                target,
                learning_rate=0.02,
                params=model.top_layer_parameters(),
            )

        self.assertEqual(model.wq[0][0].data, lower_before)
        self.assertTrue(
            model.extra_blocks[0]["wq"][0][0].data != top_before
            or model.wout[0][target].data != head_before
        )

    def test_multi_layer_final_block_matches_full_stack_logits(self) -> None:
        tokenizer, ids, config, model = char_model_fixture(seed=11, num_layers=2)
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        optimized = model._forward_floats(context)
        token_embeddings = [[value.data for value in row] for row in model.token_embeddings]
        position_embeddings = [[value.data for value in row] for row in model.position_embeddings]
        full_stack = model._forward_full_block_floats(
            [
                [
                    token_embeddings[token_id][dim] + position_embeddings[position][dim]
                    for dim in range(config.embedding_dim)
                ]
                for position, token_id in enumerate(context)
            ],
            model._block_to_floats(model.blocks[0]),
        )
        full_stack = model._forward_full_block_floats(
            full_stack,
            model._block_to_floats(model.blocks[1]),
        )
        manual = []
        for output_index, bias in enumerate([value.data for value in model.bout]):
            total = bias
            for input_index, value in enumerate(full_stack[-1]):
                total += value * model.wout[input_index][output_index].data
            manual.append(total)

        self.assertEqual(len(optimized), len(manual))
        for left, right in zip(optimized, manual):
            self.assertAlmostEqual(left, right)


if __name__ == "__main__":
    unittest.main()
