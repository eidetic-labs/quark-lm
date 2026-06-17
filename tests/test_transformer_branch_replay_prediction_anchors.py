from __future__ import annotations

import unittest

from support.core import TinyTransformerLM, TransformerConfig


class TransformerBranchReplayPredictionAnchorsTest(unittest.TestCase):
    def test_baseline_prediction_anchor_protects_covered_target_after_drift(
        self,
    ) -> None:
        def initialized_model() -> TinyTransformerLM:
            initialized = TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=4,
                    context_size=1,
                    embedding_dim=3,
                    feedforward_dim=4,
                    seed=85,
                )
            )
            initialized.bout[1].data = -2.0
            initialized.bout[2].data = 2.0
            return initialized

        def covered_probability(scored_model: TinyTransformerLM) -> float:
            probs = scored_model.predict([0])
            return probs[1]

        dynamic_model = initialized_model()
        anchored_model = initialized_model()
        dynamic_replay = [([0], 2, 2, "qa:mixed"), ([3], 1, 2, "qa:mixed")]
        anchored_replay = [([0], 2, 1, "qa:mixed"), ([3], 1, 2, "qa:mixed")]
        before_anchor_probability = covered_probability(anchored_model)

        for _ in range(30):
            dynamic_model.train_step_with_branch_context_replay_coverage(
                dynamic_replay,
                dynamic_replay,
                learning_rate=0.03,
                negative_weight=0.0,
                positive_weight=0.0,
                replay_weight=1.0,
                hard_negative_count=0,
                preserve_predicted_target_coverage=True,
                profile_aware_targets=True,
                balance_profile_target_shares=True,
                enforce_prompt_target_margins=True,
            )
            anchored_model.train_step_with_branch_context_replay_coverage(
                dynamic_replay,
                anchored_replay,
                learning_rate=0.03,
                negative_weight=0.0,
                positive_weight=0.0,
                replay_weight=1.0,
                hard_negative_count=0,
                preserve_predicted_target_coverage=True,
                profile_aware_targets=True,
                balance_profile_target_shares=True,
                enforce_prompt_target_margins=True,
            )

        self.assertGreater(covered_probability(anchored_model), before_anchor_probability)
        self.assertGreater(
            covered_probability(anchored_model),
            covered_probability(dynamic_model),
        )


if __name__ == "__main__":
    unittest.main()
