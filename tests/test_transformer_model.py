from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import transformer_char_model
from transformer_tiny_lm import TinyTransformerLM
from transformer_model import (
    TRANSFORMER_ARCHITECTURE,
    TRANSFORMER_CHECKPOINT_FORMAT,
    TRANSFORMER_TOKENIZER,
    GenerationConfig,
    OptimizationConfig,
    TransformerConfig,
    checkpoint_header,
    closed_world_dataset_metadata,
    generation_config_from_args,
    optimization_config_from_args,
    transformer_config_from_args,
    transformer_run_metadata,
    validate_generation_config,
    validate_optimization_config,
    validate_transformer_config,
)


class FakeTokenizer:
    vocab_size = 31


class FakeOptimizer:
    def summary(self) -> dict[str, int | str]:
        return {"optimizer": "sgd", "update_count": 4}


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        context_size=32,
        embedding_dim=12,
        feedforward_dim=24,
        seed=19,
        num_layers=2,
        attention_heads=3,
        use_layer_norm=True,
        use_pre_layer_norm=False,
        use_rms_norm=False,
        layer_norm_epsilon=1e-5,
        use_gated_mlp=True,
        tie_output_embeddings=False,
        use_rotary_positions=True,
        use_kv_cache_path=False,
        use_context_mean=True,
        use_context_projection=False,
        use_prompt_prefix_projection=True,
        use_prompt_position_projection=False,
        prompt_position_projection_scale=0.75,
        use_prompt_attention_summary=True,
        optimizer="adamw",
        gradient_clip=2.5,
        weight_decay=0.01,
        adam_beta1=0.8,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        warmup_steps=3,
        decay_steps=10,
        min_learning_rate=0.001,
        gradient_accumulation_steps=2,
        temperature=0.7,
        top_k=5,
        top_p=0.9,
        repetition_penalty=1.1,
        trace_top_tokens=4,
        use_kv_cache=True,
    )


class TransformerModelSurfaceTests(unittest.TestCase):
    def test_char_model_reexports_model_surface_for_compatibility(self) -> None:
        self.assertIs(transformer_char_model.TransformerConfig, TransformerConfig)
        self.assertIs(transformer_char_model.OptimizationConfig, OptimizationConfig)
        self.assertIs(transformer_char_model.GenerationConfig, GenerationConfig)
        self.assertIs(transformer_char_model.TinyTransformerLM, TinyTransformerLM)

    def test_transformer_config_from_args_and_validation(self) -> None:
        config = transformer_config_from_args(_args(), vocab_size=31)

        self.assertEqual(config.vocab_size, 31)
        self.assertEqual(config.context_size, 32)
        self.assertEqual(config.attention_heads, 3)
        self.assertTrue(config.use_prompt_attention_summary)
        validate_transformer_config(config)
        with self.assertRaises(ValueError):
            validate_transformer_config(
                TransformerConfig(vocab_size=31, embedding_dim=10, attention_heads=3)
            )

    def test_optimization_and_generation_configs_from_args(self) -> None:
        args = _args()
        optimizer_config = optimization_config_from_args(args)
        generation_config = generation_config_from_args(args)

        self.assertEqual(optimizer_config.optimizer, "adamw")
        self.assertEqual(optimizer_config.gradient_accumulation_steps, 2)
        self.assertEqual(generation_config.top_k, 5)
        self.assertTrue(generation_config.use_kv_cache)
        validate_optimization_config(optimizer_config)
        validate_generation_config(generation_config)
        with self.assertRaises(ValueError):
            validate_optimization_config(OptimizationConfig(optimizer="external"))
        with self.assertRaises(ValueError):
            validate_generation_config(GenerationConfig(top_p=0.0))

    def test_checkpoint_header_and_dataset_metadata_are_centralized(self) -> None:
        config = TransformerConfig(vocab_size=31, context_size=12, embedding_dim=6)
        header = checkpoint_header(config)
        dataset = closed_world_dataset_metadata(31)

        self.assertEqual(header["architecture"], TRANSFORMER_ARCHITECTURE)
        self.assertEqual(header["checkpoint_format"], TRANSFORMER_CHECKPOINT_FORMAT)
        self.assertEqual(header["config"], asdict(config))
        self.assertEqual(dataset["tokenizer"], TRANSFORMER_TOKENIZER)
        self.assertFalse(dataset["pretrained_weights"])
        self.assertFalse(dataset["pretrained_tokenizer"])
        self.assertFalse(dataset["external_embeddings"])

    def test_transformer_run_metadata_uses_closed_world_checkpoint_surface(self) -> None:
        metadata = transformer_run_metadata(
            _args(),
            FakeTokenizer(),
            FakeOptimizer(),
            "answer-train",
            {"resumed": False},
        )

        self.assertEqual(metadata["run_kind"], "answer-train")
        self.assertEqual(metadata["config"]["vocab_size"], 31)
        self.assertEqual(metadata["optimizer"]["update_count"], 4)
        self.assertEqual(metadata["dataset"]["tokenizer"], TRANSFORMER_TOKENIZER)
        self.assertFalse(metadata["dataset"]["external_embeddings"])


if __name__ == "__main__":
    unittest.main()
