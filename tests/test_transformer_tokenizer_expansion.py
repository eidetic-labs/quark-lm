from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_subword_tokenizer import ClosedWorldSubwordTokenizer, MergeRule
from tokenizer import CharTokenizer
from transformer_lm_weight_initialization import build_random_transformer_weights
from transformer_model import TransformerConfig
from transformer_tiny_lm import TinyTransformerLM
from transformer_vocab_expansion import expand_weights_for_tokenizer
from transformer_vocab_expansion_audit import audit_vocab_expansion_parity


class TransformerTokenizerExpansionTest(unittest.TestCase):
    def test_new_atomic_character_uses_existing_vocab_centroid(self) -> None:
        base = CharTokenizer.train("ab")
        expanded = base.extend("ab!")
        config = TransformerConfig(
            vocab_size=base.vocab_size,
            context_size=8,
            embedding_dim=4,
            feedforward_dim=8,
            seed=13,
        )
        weights = build_random_transformer_weights(config)

        expanded_weights = expand_weights_for_tokenizer(weights, base, expanded)

        new_id = expanded.stoi["!"]
        seed_ids = [base.stoi["a"], base.stoi["b"]]
        expected_embedding = [
            sum(weights["token_embeddings"][token_id][dim] for token_id in seed_ids)
            / len(seed_ids)
            for dim in range(config.embedding_dim)
        ]
        expected_output = [
            sum(weights["wout"][dim][token_id] for token_id in seed_ids) / len(seed_ids)
            for dim in range(config.embedding_dim)
        ]
        self.assertEqual(expanded_weights["token_embeddings"][new_id], expected_embedding)
        self.assertEqual(
            [row[new_id] for row in expanded_weights["wout"]],
            expected_output,
        )
        self.assertEqual(expanded_weights["bout"][new_id], 0.0)

    def test_new_token_embeddings_are_initialized_from_parts(self) -> None:
        base = CharTokenizer.train("kite\n")
        expanded = ClosedWorldSubwordTokenizer.from_char_tokens(base.tokens).with_merge(
            MergeRule("k", "i", "ki")
        )
        config = TransformerConfig(
            vocab_size=base.vocab_size,
            context_size=8,
            embedding_dim=4,
            feedforward_dim=8,
            seed=13,
        )
        weights = build_random_transformer_weights(config)

        expanded_weights = expand_weights_for_tokenizer(weights, base, expanded)

        new_id = expanded.stoi["ki"]
        k_id = base.stoi["k"]
        i_id = base.stoi["i"]
        expected_embedding = [
            (weights["token_embeddings"][k_id][dim] + weights["token_embeddings"][i_id][dim]) / 2
            for dim in range(config.embedding_dim)
        ]
        expected_output = [
            (weights["wout"][dim][k_id] + weights["wout"][dim][i_id]) / 2
            for dim in range(config.embedding_dim)
        ]
        self.assertEqual(expanded_weights["token_embeddings"][new_id], expected_embedding)
        self.assertEqual(
            [row[new_id] for row in expanded_weights["wout"]],
            expected_output,
        )
        self.assertEqual(expanded_weights["bout"][new_id], 0.0)
        self.assertEqual(
            expanded_weights["token_embeddings"][: base.vocab_size],
            weights["token_embeddings"],
        )

    def test_merge_with_new_atomic_character_uses_initialized_parts(self) -> None:
        base = CharTokenizer.train("hi")
        expanded_chars = ClosedWorldSubwordTokenizer.from_char_tokens(base.extend("hi!").tokens)
        expanded = expanded_chars.with_merge(MergeRule("i", "!", "i!"))
        config = TransformerConfig(
            vocab_size=base.vocab_size,
            context_size=8,
            embedding_dim=4,
            feedforward_dim=8,
            seed=13,
        )
        weights = build_random_transformer_weights(config)

        expanded_weights = expand_weights_for_tokenizer(weights, base, expanded)

        exclaim_id = expanded.stoi["!"]
        merge_id = expanded.stoi["i!"]
        i_id = base.stoi["i"]
        expected_embedding = [
            (
                expanded_weights["token_embeddings"][i_id][dim]
                + expanded_weights["token_embeddings"][exclaim_id][dim]
            )
            / 2
            for dim in range(config.embedding_dim)
        ]
        self.assertEqual(expanded_weights["token_embeddings"][merge_id], expected_embedding)

    def test_vocab_expansion_audit_preserves_old_logits(self) -> None:
        base = CharTokenizer.train("kite\n")
        expanded = ClosedWorldSubwordTokenizer.from_char_tokens(base.tokens).with_merge(
            MergeRule("k", "i", "ki")
        )
        config = TransformerConfig(
            vocab_size=base.vocab_size,
            context_size=8,
            embedding_dim=4,
            feedforward_dim=8,
            seed=13,
        )
        weights = build_random_transformer_weights(config)
        expanded_weights = expand_weights_for_tokenizer(weights, base, expanded)

        report = audit_vocab_expansion_parity(
            base_model=TinyTransformerLM(config, weights),
            expanded_model=TinyTransformerLM(
                TransformerConfig(
                    vocab_size=expanded.vocab_size,
                    context_size=8,
                    embedding_dim=4,
                    feedforward_dim=8,
                    seed=13,
                ),
                expanded_weights,
            ),
            base_tokenizer=base,
            expanded_tokenizer=expanded,
            train_text="kite\n",
            context_size=8,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["base_vocab_size"], base.vocab_size)
        self.assertEqual(report["expanded_vocab_size"], expanded.vocab_size)


if __name__ == "__main__":
    unittest.main()
