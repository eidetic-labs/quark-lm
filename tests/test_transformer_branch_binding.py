from __future__ import annotations

import unittest

from support.branch_binding import (
    average_target_context_ownership,
    average_target_rank,
    branch_binding_fixture,
    branch_targets_from_batch,
    initialized_branch_binding_model,
    restricted_probabilities,
    target_balanced_branch_batch,
    train_bidirectional_binding_steps,
    train_coverage_binding_steps,
    train_rank_margin_steps,
)


class TransformerBranchBindingTest(unittest.TestCase):
    def test_balanced_branch_rank_margin_uses_target_balanced_batches(self) -> None:
        near, examples, tokenizer = branch_binding_fixture()
        model = initialized_branch_binding_model(
            tokenizer,
            seed=51,
            token_biases={".": 5.0},
        )
        batch = target_balanced_branch_batch(
            model,
            tokenizer,
            near,
            examples,
            rng_seed=15,
        )

        before_rank = average_target_rank(model, batch)

        train_rank_margin_steps(
            model,
            tokenizer,
            near,
            examples,
            repeat=48,
            rng_seed=16,
        )

        self.assertLess(average_target_rank(model, batch), before_rank)

    def test_branch_bidirectional_binding_lifts_target_context_ownership(self) -> None:
        near, examples, tokenizer = branch_binding_fixture()
        model = initialized_branch_binding_model(
            tokenizer,
            seed=53,
            token_biases={".": 5.0},
        )
        batch = target_balanced_branch_batch(
            model,
            tokenizer,
            near,
            examples,
            rng_seed=15,
        )
        branch_targets = branch_targets_from_batch(batch)
        self.assertGreater(len(branch_targets), 1)

        before_ownership = average_target_context_ownership(
            model,
            batch,
            branch_targets,
        )

        train_bidirectional_binding_steps(
            model,
            tokenizer,
            near,
            examples,
            repeat=48,
            rng_seed=16,
        )

        self.assertGreater(
            average_target_context_ownership(model, batch, branch_targets),
            before_ownership,
        )

    def test_branch_coverage_binding_lifts_target_set_against_hard_wrong_tokens(
        self,
    ) -> None:
        near, examples, tokenizer = branch_binding_fixture()
        model = initialized_branch_binding_model(
            tokenizer,
            seed=54,
            token_biases={".": 5.0},
        )
        batch = target_balanced_branch_batch(
            model,
            tokenizer,
            near,
            examples,
            rng_seed=15,
        )
        branch_targets = branch_targets_from_batch(batch)
        self.assertGreater(len(branch_targets), 1)

        before_target_set, before_target = restricted_probabilities(
            model,
            batch,
            branch_targets,
        )

        train_coverage_binding_steps(
            model,
            tokenizer,
            near,
            examples,
            repeat=48,
            rng_seed=16,
        )

        after_target_set, after_target = restricted_probabilities(
            model,
            batch,
            branch_targets,
        )
        self.assertGreater(after_target_set, before_target_set)
        self.assertGreater(after_target, before_target)


if __name__ == "__main__":
    unittest.main()
