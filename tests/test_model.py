from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_char_model import CharMLP, ModelConfig, context_before
from tokenizer import CharTokenizer


class ModelTest(unittest.TestCase):
    def test_train_step_and_checkpoint_round_trip(self) -> None:
        text = "question: where is mia's ball?\nanswer: under the box.\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=8,
            embedding_dim=4,
            hidden_dim=8,
            seed=1,
        )
        model = CharMLP.init_random(config)
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        before = model.nll(context, ids[4])
        after = model.train_step(context, ids[4], learning_rate=0.1)
        self.assertGreater(before, 0)
        self.assertGreater(after, 0)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.json"
            model.save(path, tokenizer)
            loaded, loaded_tokenizer = CharMLP.load(path)
            self.assertIsNotNone(loaded_tokenizer)
            self.assertEqual(loaded.config.vocab_size, model.config.vocab_size)


if __name__ == "__main__":
    unittest.main()
