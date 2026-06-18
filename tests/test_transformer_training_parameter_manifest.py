from __future__ import annotations

from dataclasses import asdict
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.char_model import char_model_fixture
from transformer_training_parity import (
    TRAINING_PARAMETER_ORDER,
    build_training_parameter_manifest,
    validate_training_parameter_manifest,
)


class TransformerTrainingParameterManifestTests(unittest.TestCase):
    def test_default_manifest_matches_scalar_parameter_count_and_order(self) -> None:
        _tokenizer, _ids, config, model = char_model_fixture("abc abc\n", seed=7)

        manifest = _manifest(config, model)

        validate_training_parameter_manifest(manifest)
        self.assertEqual(manifest["parameter_order"], TRAINING_PARAMETER_ORDER)
        self.assertEqual(manifest["parameter_count"], len(model.parameters()))
        self.assertEqual(manifest["entries"][0]["name"], "token_embeddings")
        self.assertEqual(manifest["entries"][-1]["name"], "wout")
        self.assertNotIn("w_gate", _entry_names(manifest))

    def test_tied_output_manifest_excludes_output_weight_copy(self) -> None:
        _tokenizer, _ids, config, model = char_model_fixture(
            "abc abc\n",
            seed=7,
            tie_output_embeddings=True,
        )

        manifest = _manifest(config, model)

        self.assertTrue(manifest["tie_output_embeddings"])
        self.assertEqual(manifest["parameter_count"], len(model.parameters()))
        self.assertNotIn("wout", _entry_names(manifest))
        self.assertIn("bout", _entry_names(manifest))

    def test_manifest_covers_optional_and_extra_layer_parameters(self) -> None:
        _tokenizer, _ids, config, model = char_model_fixture(
            "abc abc\n",
            seed=7,
            num_layers=2,
            use_gated_mlp=True,
            use_pre_layer_norm=True,
            use_rms_norm=True,
            use_context_projection=True,
            use_prompt_prefix_projection=True,
            use_prompt_position_projection=True,
            use_prompt_attention_summary=True,
        )

        manifest = _manifest(config, model)
        names = _entry_names(manifest)

        self.assertEqual(manifest["parameter_count"], len(model.parameters()))
        self.assertIn("w_gate", names)
        self.assertIn("final_ln_gain", names)
        self.assertIn("prompt_position_projection_w", names)
        self.assertIn("extra_layers.0.wq", names)
        self.assertIn("extra_layers.0.ln2_bias", names)

    def test_manifest_validation_rejects_noncontiguous_entries(self) -> None:
        _tokenizer, _ids, config, model = char_model_fixture("abc abc\n", seed=7)
        manifest = _manifest(config, model)
        manifest["entries"][1]["index_start"] += 1

        with self.assertRaisesRegex(ValueError, "index_start"):
            validate_training_parameter_manifest(manifest)


def _manifest(config: object, model: object) -> dict:
    return build_training_parameter_manifest(
        weights=model.to_dict()["weights"],
        model_config=asdict(config),
    )


def _entry_names(manifest: dict) -> set[str]:
    return {entry["name"] for entry in manifest["entries"]}


if __name__ == "__main__":
    unittest.main()
