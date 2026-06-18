from __future__ import annotations

import unittest
from pathlib import Path

from support.cli_eval import DIRECT_ANSWER_MODE_CASES, parse_direct_answer_mode_case
from support.commands import parse_args


class TransformerCliEvalTest(unittest.TestCase):
    def test_direct_answer_modes_include_rollout_and_hybrid(self) -> None:
        for mode in DIRECT_ANSWER_MODE_CASES:
            args = parse_direct_answer_mode_case(mode)
            self.assertEqual(args.direct_answer_mode, mode)
            self.assertEqual(args.direct_answer_rollout_interval, 4)
            self.assertEqual(args.direct_answer_positive_weight, 1.5)
            self.assertEqual(args.direct_answer_contrast_weight, 1.25)
            self.assertEqual(args.direct_answer_recovery_steps, 2)
            self.assertEqual(args.direct_answer_branch_position, 1)
            self.assertEqual(args.direct_answer_branch_span, 3)
            self.assertEqual(args.direct_answer_branch_batch_size, 5)
            self.assertEqual(args.direct_answer_hard_negatives, 7)
            self.assertEqual(args.temperature, 0.8)
            self.assertEqual(args.top_k, 3)
            self.assertEqual(args.top_p, 0.75)
            self.assertEqual(args.repetition_penalty, 1.2)
            self.assertEqual(args.trace_top_tokens, 4)
            self.assertTrue(args.use_kv_cache)
            self.assertEqual(args.direct_answer_snapshot_mode, "branch-only")
            self.assertTrue(args.direct_answer_train_top_layer_only)
            self.assertTrue(args.direct_answer_freeze_output_bias)
            self.assertTrue(args.direct_answer_restore_best_branch_snapshot)
            self.assertTrue(args.direct_answer_require_branch_context_gate)
            self.assertTrue(args.skip_post_direct_snapshot)
            self.assertEqual(args.num_layers, 2)
            self.assertEqual(args.attention_heads, 2)
            self.assertTrue(args.use_layer_norm)
            self.assertTrue(args.use_pre_layer_norm)
            self.assertTrue(args.use_rms_norm)
            self.assertEqual(args.layer_norm_epsilon, 0.0001)
            self.assertTrue(args.use_gated_mlp)
            self.assertTrue(args.tie_output_embeddings)
            self.assertTrue(args.use_rotary_positions)
            self.assertTrue(args.use_kv_cache_path)
            self.assertTrue(args.use_context_mean)
            self.assertTrue(args.use_context_projection)
            self.assertTrue(args.use_prompt_prefix_projection)
            self.assertTrue(args.use_prompt_position_projection)
            self.assertEqual(args.prompt_position_projection_scale, 12.5)
            self.assertTrue(args.use_prompt_attention_summary)
            self.assertEqual(args.direct_answer_sequence_interval, 6)
            self.assertEqual(args.optimizer, "adamw")
            self.assertEqual(args.gradient_clip, 3.5)
            self.assertEqual(args.weight_decay, 0.01)
            self.assertEqual(args.adam_beta1, 0.8)
            self.assertEqual(args.adam_beta2, 0.95)
            self.assertEqual(args.adam_epsilon, 0.0000001)
            self.assertEqual(args.warmup_steps, 2)
            self.assertEqual(args.decay_steps, 7)
            self.assertEqual(args.min_learning_rate, 0.001)
            self.assertEqual(args.gradient_accumulation_steps, 2)

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

    def test_parse_answer_args_accepts_experiment_contract(self) -> None:
        args = parse_args(
            [
                "answer-train",
                "--experiment-version",
                "v0.71",
                "--experiment-hypothesis",
                "A bounded screen can explain itself.",
                "--experiment-acceptance-gate",
                "custom_gate:Custom rule.",
                "--experiment-failure-criterion",
                "Custom failure.",
                "--experiment-note",
                "Custom note.",
                "--tokenizer",
                "closed-world-subword",
                "--tokenizer-manifest",
                "answer_tokenizer_manifest.json",
                "--tokenizer-report",
                "answer_tokenizer_report.json",
                "--tokenizer-max-token-chars",
                "3",
                "--tokenizer-max-new-tokens",
                "9",
            ]
        )

        self.assertEqual(args.experiment_version, "v0.71")
        self.assertEqual(args.experiment_hypothesis, "A bounded screen can explain itself.")
        self.assertEqual(args.experiment_acceptance_gate, ["custom_gate:Custom rule."])
        self.assertEqual(args.experiment_failure_criterion, ["Custom failure."])
        self.assertEqual(args.experiment_note, ["Custom note."])
        self.assertEqual(args.tokenizer, "closed-world-subword")
        self.assertEqual(args.tokenizer_manifest, "answer_tokenizer_manifest.json")
        self.assertEqual(args.tokenizer_report, "answer_tokenizer_report.json")
        self.assertEqual(args.tokenizer_max_token_chars, 3)
        self.assertEqual(args.tokenizer_max_new_tokens, 9)

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
