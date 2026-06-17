from __future__ import annotations

import unittest

from support.branch_replay_coverage import (
    branch_training_batch,
    initialized_coverage_model,
    replay_coverage_fixture,
    replay_deficit_scenario,
    target_probability,
    train_deficit_focus_comparison_steps,
    train_preserving_deficit_steps,
)


class TransformerBranchReplayCoverageTest(unittest.TestCase):
    def test_branch_context_coverage_deficit_lifts_missing_target_probability(
        self,
    ) -> None:
        near, examples, tokenizer = replay_coverage_fixture()
        baseline_model = initialized_coverage_model(tokenizer, seed=61)
        deficit_model = initialized_coverage_model(tokenizer, seed=61)
        scenario = replay_deficit_scenario(
            baseline_model,
            tokenizer,
            near,
            examples,
        )
        branch_batch = branch_training_batch(
            baseline_model,
            tokenizer,
            near,
            examples,
        )

        before_probability = target_probability(
            deficit_model,
            scenario,
            scenario.deficit_context,
            scenario.deficit_target,
        )

        train_deficit_focus_comparison_steps(
            baseline_model,
            deficit_model,
            branch_batch,
            scenario.replay_branches,
        )

        self.assertGreater(
            target_probability(
                deficit_model,
                scenario,
                scenario.deficit_context,
                scenario.deficit_target,
            ),
            before_probability,
        )
        self.assertGreater(
            target_probability(
                deficit_model,
                scenario,
                scenario.deficit_context,
                scenario.deficit_target,
            ),
            target_probability(
                baseline_model,
                scenario,
                scenario.deficit_context,
                scenario.deficit_target,
            ),
        )

    def test_branch_context_coverage_preserving_deficit_protects_represented_target(
        self,
    ) -> None:
        near, examples, tokenizer = replay_coverage_fixture()
        deficit_only_model = initialized_coverage_model(tokenizer, seed=62)
        preserving_model = initialized_coverage_model(tokenizer, seed=62)
        scenario = replay_deficit_scenario(
            deficit_only_model,
            tokenizer,
            near,
            examples,
        )
        branch_batch = branch_training_batch(
            deficit_only_model,
            tokenizer,
            near,
            examples,
        )

        before_deficit_probability = target_probability(
            preserving_model,
            scenario,
            scenario.deficit_context,
            scenario.deficit_target,
        )

        train_preserving_deficit_steps(
            deficit_only_model,
            preserving_model,
            branch_batch,
            scenario.replay_branches,
        )

        self.assertGreater(
            target_probability(
                preserving_model,
                scenario,
                scenario.deficit_context,
                scenario.deficit_target,
            ),
            before_deficit_probability,
        )
        self.assertGreater(
            target_probability(
                preserving_model,
                scenario,
                scenario.represented_context,
                scenario.represented_prediction,
            ),
            target_probability(
                deficit_only_model,
                scenario,
                scenario.represented_context,
                scenario.represented_prediction,
            ),
        )


if __name__ == "__main__":
    unittest.main()
