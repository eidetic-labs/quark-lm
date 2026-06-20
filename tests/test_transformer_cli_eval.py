from __future__ import annotations

import unittest
from pathlib import Path

from support.commands import parse_args


class TransformerCliEvalTest(unittest.TestCase):
    def test_parse_train_args_accepts_context_mean(self) -> None:
        args = parse_args(
            [
                "train",
                "--transformer-profile",
                "modern_small",
                "--tokenizer",
                "closed-world-subword",
                "--tokenizer-manifest",
                "tokenizer_manifest.json",
                "--tokenizer-report",
                "tokenizer_report.json",
                "--tokenizer-max-token-chars",
                "4",
                "--tokenizer-max-new-tokens",
                "12",
                "--use-context-mean",
                "--use-context-projection",
                "--use-prompt-prefix-projection",
                "--use-prompt-position-projection",
                "--use-pre-layer-norm",
                "--use-rms-norm",
                "--prompt-position-projection-scale",
                "4.0",
                "--use-prompt-attention-summary",
                "--attention-heads",
                "2",
                "--use-gated-mlp",
                "--tie-output-embeddings",
                "--use-rotary-positions",
                "--use-kv-cache-path",
                "--optimizer",
                "adamw",
                "--gradient-accumulation-steps",
                "3",
            ]
        )

        self.assertEqual(args.transformer_profile, "modern_small")
        self.assertEqual(args.tokenizer, "closed-world-subword")
        self.assertEqual(args.tokenizer_manifest, "tokenizer_manifest.json")
        self.assertEqual(args.tokenizer_report, "tokenizer_report.json")
        self.assertEqual(args.tokenizer_max_token_chars, 4)
        self.assertEqual(args.tokenizer_max_new_tokens, 12)
        self.assertTrue(args.use_context_mean)
        self.assertTrue(args.use_context_projection)
        self.assertTrue(args.use_prompt_prefix_projection)
        self.assertTrue(args.use_prompt_position_projection)
        self.assertTrue(args.use_pre_layer_norm)
        self.assertTrue(args.use_rms_norm)
        self.assertEqual(args.prompt_position_projection_scale, 4.0)
        self.assertTrue(args.use_prompt_attention_summary)
        self.assertEqual(args.attention_heads, 2)
        self.assertTrue(args.use_gated_mlp)
        self.assertTrue(args.tie_output_embeddings)
        self.assertTrue(args.use_rotary_positions)
        self.assertTrue(args.use_kv_cache_path)
        self.assertEqual(args.optimizer, "adamw")
        self.assertEqual(args.gradient_accumulation_steps, 3)

    def test_parse_eval_args_accepts_generation_trace_controls(self) -> None:
        args = parse_args(
            [
                "eval",
                "--temperature",
                "0.6",
                "--top-k",
                "4",
                "--top-p",
                "0.8",
                "--repetition-penalty",
                "1.3",
                "--trace-top-tokens",
                "3",
                "--use-kv-cache",
                "--samples-jsonl",
                "samples.jsonl",
            ]
        )

        self.assertEqual(args.temperature, 0.6)
        self.assertEqual(args.top_k, 4)
        self.assertEqual(args.top_p, 0.8)
        self.assertEqual(args.repetition_penalty, 1.3)
        self.assertEqual(args.trace_top_tokens, 3)
        self.assertTrue(args.use_kv_cache)
        self.assertEqual(args.samples_jsonl, Path("samples.jsonl"))

    def test_parse_incremental_update_args_accepts_guard_inputs(self) -> None:
        args = parse_args(
            [
                "incremental-update",
                "--base-checkpoint",
                "base.json",
                "--candidate-checkpoint",
                "candidate.json",
                "--accepted-checkpoint",
                "accepted.json",
                "--report",
                "report.json",
                "--new-lesson-probe",
                "new.jsonl",
                "--regression-probe",
                "regression.jsonl",
                "--nll-tolerance",
                "0.125",
                "--trace-top-tokens",
                "3",
            ]
        )

        self.assertEqual(args.base_checkpoint, Path("base.json"))
        self.assertEqual(args.candidate_checkpoint, Path("candidate.json"))
        self.assertEqual(args.accepted_checkpoint, Path("accepted.json"))
        self.assertEqual(args.report, Path("report.json"))
        self.assertEqual(args.new_lesson_probe, [Path("new.jsonl")])
        self.assertEqual(args.regression_probe, [Path("regression.jsonl")])
        self.assertEqual(args.nll_tolerance, 0.125)
        self.assertEqual(args.trace_top_tokens, 3)


if __name__ == "__main__":
    unittest.main()
