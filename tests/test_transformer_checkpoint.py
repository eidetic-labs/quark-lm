from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokenizer import CharTokenizer
from transformer_char_model import TinyTransformerLM
from transformer_checkpoint import (
    checkpoint_summary,
    load_checkpoint_payload,
    validate_checkpoint_payload,
)
from transformer_model import (
    TRANSFORMER_CHECKPOINT_FORMAT,
    TransformerConfig,
)


class TransformerCheckpointTests(unittest.TestCase):
    def test_load_checkpoint_payload_validates_identity_and_summarizes(self) -> None:
        tokenizer = CharTokenizer.train("abc abc\n")
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=3,
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer, metadata={"run_kind": "test"})

            payload = load_checkpoint_payload(path)
            summary = checkpoint_summary(payload)

        self.assertEqual(summary["checkpoint_format"], TRANSFORMER_CHECKPOINT_FORMAT)
        self.assertTrue(summary["has_tokenizer"])
        self.assertTrue(summary["has_metadata"])
        self.assertEqual(summary["vocab_size"], tokenizer.vocab_size)
        self.assertIn("token_embeddings", summary["weight_groups"])

    def test_validate_checkpoint_payload_rejects_wrong_format(self) -> None:
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(
                {
                    "architecture": "tiny-decoder-only-transformer",
                    "checkpoint_format": "external-format",
                    "config": {},
                    "weights": {},
                }
            )

    def test_load_checkpoint_payload_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps(["not", "a", "checkpoint"]), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_checkpoint_payload(path)


if __name__ == "__main__":
    unittest.main()
