from __future__ import annotations

import random
import unittest

from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from support.direct_answer import (
    direct_answer_branch_context,
    direct_answer_profiled_branch_batch,
)


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

    def test_profiled_branch_batch_can_use_baseline_prediction_overrides(
        self,
    ) -> None:
        green = AnswerExample(
            prompt="q: color?\na:",
            target=" green.",
            source="qa:color",
        )
        blue = AnswerExample(
            prompt="q: color?\na:",
            target=" blue.",
            source="qa:color",
        )
        tokenizer = CharTokenizer.train(
            green.prompt + green.target + blue.prompt + blue.target + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=84,
            )
        )
        model.bout[tokenizer.stoi["g"]].data = 4.0
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            green,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(branch)
        context, target_id, _position = branch
        override_id = tokenizer.stoi["b"]
        current_prediction = max(
            range(tokenizer.vocab_size),
            key=lambda index: model.predict(context)[index],
        )

        batch = direct_answer_profiled_branch_batch(
            model,
            tokenizer,
            green,
            [green, blue],
            random.Random(84),
            branch_position=1,
            batch_size=1,
            terminator=ANSWER_TERMINATOR,
            prediction_overrides={
                (tuple(context), target_id, "qa:color"): override_id,
            },
        )

        self.assertNotEqual(current_prediction, override_id)
        self.assertEqual(batch, [(context, target_id, override_id, "qa:color")])

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
