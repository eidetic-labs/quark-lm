from __future__ import annotations

import unittest

from support.core import TinyTransformerLM, TransformerConfig


class TransformerBranchReplayBalancingTest(unittest.TestCase):
    def test_profile_target_share_balancing_lifts_minority_replay_target(
        self,
    ) -> None:
        def initialized_model() -> TinyTransformerLM:
            initialized = TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=4,
                    context_size=1,
                    embedding_dim=3,
                    feedforward_dim=4,
                    seed=71,
                )
            )
            initialized.bout[1].data = 2.5
            initialized.bout[2].data = -2.5
            return initialized

        baseline_model = initialized_model()
        balanced_model = initialized_model()
        majority_branch = ([0], 1, 1, "qa:mixed")
        minority_branch = ([0], 2, 1, "qa:mixed")
        replay_branches = [majority_branch] * 8 + [minority_branch]

        def minority_target_share(scored_model: TinyTransformerLM) -> float:
            probs = scored_model.predict([0])
            return probs[2] / (probs[1] + probs[2])

        before_share = minority_target_share(balanced_model)

        for _ in range(40):
            baseline_model.train_step_with_branch_context_replay_coverage(
                replay_branches,
                replay_branches,
                learning_rate=0.04,
                negative_weight=0.0,
                positive_weight=0.0,
                replay_weight=1.0,
                hard_negative_count=0,
                profile_aware_targets=True,
            )
            balanced_model.train_step_with_branch_context_replay_coverage(
                replay_branches,
                replay_branches,
                learning_rate=0.04,
                negative_weight=0.0,
                positive_weight=0.0,
                replay_weight=1.0,
                hard_negative_count=0,
                profile_aware_targets=True,
                balance_profile_target_shares=True,
            )

        self.assertGreater(minority_target_share(balanced_model), before_share)
        self.assertGreater(
            minority_target_share(balanced_model),
            minority_target_share(baseline_model),
        )


if __name__ == "__main__":
    unittest.main()
