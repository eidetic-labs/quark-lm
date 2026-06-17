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
from support.direct_answer import direct_answer_target_balanced_branch_diversity_batch


class TransformerBranchReplayCoverageTest(unittest.TestCase):
    def test_branch_context_coverage_deficit_lifts_missing_target_probability(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )

        def initialized_model() -> TinyTransformerLM:
            initialized = TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=tokenizer.vocab_size,
                    context_size=8,
                    embedding_dim=4,
                    feedforward_dim=8,
                    seed=61,
                )
            )
            initialized.bout[tokenizer.stoi["n"]].data = 5.0
            initialized.bout[tokenizer.stoi["."]].data = 4.0
            return initialized

        baseline_model = initialized_model()
        deficit_model = initialized_model()
        replay_branches = direct_answer_target_balanced_branch_diversity_batch(
            baseline_model,
            tokenizer,
            near,
            examples,
            random.Random(16),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        replay_targets = sorted(
            {target for _context, target, _predicted in replay_branches}
        )
        replay_target_set = set(replay_targets)
        predicted_replay_targets = {
            predicted
            for _context, _target, predicted in replay_branches
            if predicted in replay_target_set
        }
        deficit_targets = replay_target_set - predicted_replay_targets
        self.assertTrue(deficit_targets)
        deficit_context, deficit_target, _predicted = next(
            branch
            for branch in replay_branches
            if branch[1] in deficit_targets
        )
        branch_batch = direct_answer_target_balanced_branch_diversity_batch(
            baseline_model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=2,
            terminator=ANSWER_TERMINATOR,
        )

        def deficit_target_probability(scored_model: TinyTransformerLM) -> float:
            probs = scored_model.predict(deficit_context)
            hard_candidates = [
                index
                for index in sorted(
                    range(len(probs)),
                    key=lambda item: probs[item],
                    reverse=True,
                )
                if index not in replay_target_set
            ][:5]
            candidate_ids = [*replay_targets, *hard_candidates]
            denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
            return probs[deficit_target] / denominator

        before_probability = deficit_target_probability(deficit_model)

        for _ in range(48):
            baseline_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
            )
            deficit_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
                focus_uncovered_targets=True,
            )

        self.assertGreater(
            deficit_target_probability(deficit_model),
            before_probability,
        )
        self.assertGreater(
            deficit_target_probability(deficit_model),
            deficit_target_probability(baseline_model),
        )

    def test_branch_context_coverage_preserving_deficit_protects_represented_target(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        examples = [near, green, tree]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + ANSWER_TERMINATOR
        )

        def initialized_model() -> TinyTransformerLM:
            initialized = TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=tokenizer.vocab_size,
                    context_size=8,
                    embedding_dim=4,
                    feedforward_dim=8,
                    seed=62,
                )
            )
            initialized.bout[tokenizer.stoi["n"]].data = 5.0
            initialized.bout[tokenizer.stoi["."]].data = 4.0
            return initialized

        deficit_only_model = initialized_model()
        preserving_model = initialized_model()
        replay_branches = direct_answer_target_balanced_branch_diversity_batch(
            deficit_only_model,
            tokenizer,
            near,
            examples,
            random.Random(16),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        replay_targets = sorted(
            {target for _context, target, _predicted in replay_branches}
        )
        replay_target_set = set(replay_targets)
        predicted_replay_targets = {
            predicted
            for _context, _target, predicted in replay_branches
            if predicted in replay_target_set
        }
        deficit_targets = replay_target_set - predicted_replay_targets
        self.assertTrue(deficit_targets)
        represented_context, _represented_target, represented_prediction = next(
            branch
            for branch in replay_branches
            if branch[2] in predicted_replay_targets
        )
        deficit_context, deficit_target, _predicted = next(
            branch
            for branch in replay_branches
            if branch[1] in deficit_targets
        )
        branch_batch = direct_answer_target_balanced_branch_diversity_batch(
            deficit_only_model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=2,
            terminator=ANSWER_TERMINATOR,
        )

        def target_probability(
            scored_model: TinyTransformerLM,
            context: list[int],
            target: int,
        ) -> float:
            probs = scored_model.predict(context)
            hard_candidates = [
                index
                for index in sorted(
                    range(len(probs)),
                    key=lambda item: probs[item],
                    reverse=True,
                )
                if index not in replay_target_set
            ][:5]
            candidate_ids = [*replay_targets, *hard_candidates]
            denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
            return probs[target] / denominator

        before_deficit_probability = target_probability(
            preserving_model,
            deficit_context,
            deficit_target,
        )

        for _ in range(48):
            deficit_only_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
                focus_uncovered_targets=True,
            )
            preserving_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
                focus_uncovered_targets=True,
                preserve_predicted_target_coverage=True,
                balance_deficit_targets=True,
            )

        self.assertGreater(
            target_probability(preserving_model, deficit_context, deficit_target),
            before_deficit_probability,
        )
        self.assertGreater(
            target_probability(
                preserving_model,
                represented_context,
                represented_prediction,
            ),
            target_probability(
                deficit_only_model,
                represented_context,
                represented_prediction,
            ),
        )


if __name__ == "__main__":
    unittest.main()
