from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_model import optimization_config_from_args, transformer_config_from_args


def _args(profile: str) -> SimpleNamespace:
    return SimpleNamespace(
        transformer_profile=profile,
        context_size=24,
        embedding_dim=8,
        feedforward_dim=16,
        seed=3,
        num_layers=1,
        attention_heads=1,
        use_layer_norm=False,
        use_pre_layer_norm=False,
        use_rms_norm=False,
        layer_norm_epsilon=1e-5,
        use_gated_mlp=False,
        tie_output_embeddings=False,
        use_rotary_positions=False,
        use_kv_cache_path=False,
        use_context_mean=False,
        use_context_projection=False,
        use_prompt_prefix_projection=False,
        use_prompt_position_projection=False,
        prompt_position_projection_scale=1.0,
        use_prompt_attention_summary=False,
        optimizer="sgd",
        gradient_clip=5.0,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        warmup_steps=0,
        decay_steps=0,
        min_learning_rate=0.0,
        gradient_accumulation_steps=1,
    )


class TransformerProfilesTest(unittest.TestCase):
    def test_modern_small_profile_enables_opt_in_modern_mechanics(self) -> None:
        config = transformer_config_from_args(_args("modern_small"), vocab_size=17)
        optimizer = optimization_config_from_args(_args("modern_small"))

        self.assertEqual(config.transformer_profile, "modern_small")
        self.assertEqual(config.attention_heads, 2)
        self.assertTrue(config.use_pre_layer_norm)
        self.assertTrue(config.use_rms_norm)
        self.assertTrue(config.use_gated_mlp)
        self.assertTrue(config.use_rotary_positions)
        self.assertEqual(optimizer.optimizer, "adamw")
        self.assertEqual(optimizer.gradient_clip, 2.0)
        self.assertEqual(optimizer.gradient_accumulation_steps, 2)

    def test_default_profile_preserves_explicit_baseline(self) -> None:
        config = transformer_config_from_args(_args("default"), vocab_size=17)
        optimizer = optimization_config_from_args(_args("default"))

        self.assertEqual(config.attention_heads, 1)
        self.assertFalse(config.use_rms_norm)
        self.assertEqual(optimizer.optimizer, "sgd")


if __name__ == "__main__":
    unittest.main()
