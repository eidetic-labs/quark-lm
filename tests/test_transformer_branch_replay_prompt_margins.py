from __future__ import annotations

import unittest

from support.core import TinyTransformerLM, TransformerConfig


class TransformerBranchReplayPromptMarginsTest(unittest.TestCase):
    def test_profile_prompt_ownership_margin_lifts_context_specific_target(
        self,
    ) -> None:
        def initialized_model() -> TinyTransformerLM:
            initialized = TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=4,
                    context_size=1,
                    embedding_dim=3,
                    feedforward_dim=4,
                    seed=83,
                )
            )
            initialized.bout[1].data = 3.0
            initialized.bout[2].data = -3.0
            return initialized

        def ownership_margin(
            scored_model: TinyTransformerLM,
            context: list[int],
            target: int,
            rival: int,
        ) -> float:
            logits = scored_model._forward_floats(context)
            return logits[target] - logits[rival]

        baseline_model = initialized_model()
        prompt_model = initialized_model()
        first_context = [0]
        second_context = [3]
        first_branch = (first_context, 1, 1, "qa:mixed")
        second_branch = (second_context, 2, 1, "qa:mixed")
        replay_branches = [first_branch, second_branch]
        before_second_margin = ownership_margin(prompt_model, second_context, 2, 1)

        for _ in range(40):
            baseline_model.train_step_with_branch_context_replay_coverage(
                replay_branches,
                replay_branches,
                learning_rate=0.03,
                negative_weight=0.0,
                positive_weight=0.0,
                replay_weight=1.0,
                hard_negative_count=0,
                profile_aware_targets=True,
                balance_profile_target_shares=True,
            )
            prompt_model.train_step_with_branch_context_replay_coverage(
                replay_branches,
                replay_branches,
                learning_rate=0.03,
                negative_weight=0.0,
                positive_weight=0.0,
                replay_weight=1.0,
                hard_negative_count=0,
                profile_aware_targets=True,
                balance_profile_target_shares=True,
                enforce_prompt_target_margins=True,
            )

        self.assertGreater(
            ownership_margin(prompt_model, second_context, 2, 1),
            before_second_margin,
        )
        self.assertGreater(
            ownership_margin(prompt_model, second_context, 2, 1),
            ownership_margin(baseline_model, second_context, 2, 1),
        )


if __name__ == "__main__":
    unittest.main()
