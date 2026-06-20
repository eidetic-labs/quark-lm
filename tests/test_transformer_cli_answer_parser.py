from __future__ import annotations

import unittest
from pathlib import Path

from support.cli_eval import DIRECT_ANSWER_MODE_CASES, parse_direct_answer_mode_case
from support.commands import parse_args


class TransformerCliAnswerParserTest(unittest.TestCase):
    def test_direct_answer_modes_include_rollout_and_hybrid(self) -> None:
        for mode in DIRECT_ANSWER_MODE_CASES:
            args = parse_direct_answer_mode_case(mode)
            self.assertEqual(args.direct_answer_mode, mode)
            self.assertEqual(args.direct_answer_rollout_interval, 4)
            self.assertEqual(args.direct_answer_positive_weight, 1.5)
            self.assertEqual(args.direct_answer_contrast_weight, 1.25)
            self.assertEqual(args.direct_answer_representation_weight, 0.75)
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
                "--direct-answer-frontier-metrics",
                "runs/frontier/transformer_answer_metrics.json",
                "--direct-answer-repair-target-profile",
                "learning",
                "--direct-answer-repair-target-profile",
                "owner",
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
        self.assertEqual(
            args.direct_answer_frontier_metrics,
            Path("runs/frontier/transformer_answer_metrics.json"),
        )
        self.assertEqual(
            args.direct_answer_repair_target_profile,
            ["learning", "owner"],
        )

    def test_parse_answer_sweep_args_accepts_declared_axes(self) -> None:
        args = parse_args(
            [
                "answer-sweep",
                "--run",
                "runs/sweep",
                "--sweep-axis",
                "tokenizer=char,closed-world-subword",
                "--sweep-axis",
                "embedding_dim=4,8",
                "--sweep-max-trials",
                "4",
                "--sweep-frontier-metrics",
                "runs/frontier/transformer_answer_metrics.json",
                "--sweep-existing-report",
                "runs/previous/sweep_report.json",
                "--sweep-dry-run",
            ]
        )

        self.assertEqual(args.command, "answer-sweep")
        self.assertEqual(args.run, Path("runs/sweep"))
        self.assertEqual(
            args.sweep_axis,
            ["tokenizer=char,closed-world-subword", "embedding_dim=4,8"],
        )
        self.assertEqual(args.sweep_max_trials, 4)
        self.assertEqual(
            args.sweep_frontier_metrics,
            Path("runs/frontier/transformer_answer_metrics.json"),
        )
        self.assertEqual(
            args.sweep_existing_report,
            Path("runs/previous/sweep_report.json"),
        )
        self.assertTrue(args.sweep_dry_run)


if __name__ == "__main__":
    unittest.main()
