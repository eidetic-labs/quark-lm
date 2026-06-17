from __future__ import annotations

import unittest

from support.branch_target_coverage import (
    branch_target_fixture,
    branch_targets_from_batch,
    initialized_target_model,
    replay_target_ids,
    replay_target_metrics,
    restricted_target_metrics,
    restricted_target_set_mass,
    target_balanced_branch_batch,
    train_target_diversity_steps,
    train_target_replay_coverage_steps,
    train_target_set_coverage_steps,
)


class TransformerBranchTargetCoverageTest(unittest.TestCase):
    def test_branch_target_set_coverage_lifts_set_without_exact_sharpening(
        self,
    ) -> None:
        near, examples, tokenizer = branch_target_fixture()
        model = initialized_target_model(tokenizer, seed=55, token_biases={".": 5.0})
        batch = target_balanced_branch_batch(
            model,
            tokenizer,
            near,
            examples,
            rng_seed=15,
            batch_size=3,
        )
        branch_targets = branch_targets_from_batch(batch)
        self.assertGreater(len(branch_targets), 1)

        before_mass = restricted_target_set_mass(model, batch, branch_targets)

        train_target_set_coverage_steps(
            model,
            tokenizer,
            near,
            examples,
            repeat=48,
            rng_seed=16,
        )

        self.assertGreater(
            restricted_target_set_mass(model, batch, branch_targets),
            before_mass,
        )

    def test_branch_target_diversity_lifts_set_and_target_share_balance(
        self,
    ) -> None:
        near, examples, tokenizer = branch_target_fixture()
        model = initialized_target_model(
            tokenizer,
            seed=56,
            token_biases={".": 5.0, "n": 4.0},
        )
        batch = target_balanced_branch_batch(
            model,
            tokenizer,
            near,
            examples,
            rng_seed=15,
            batch_size=3,
        )
        branch_targets = branch_targets_from_batch(batch)
        self.assertGreater(len(branch_targets), 1)

        before_mass, before_min_share = restricted_target_metrics(
            model,
            batch,
            branch_targets,
        )

        train_target_diversity_steps(
            model,
            tokenizer,
            near,
            examples,
            repeat=48,
            rng_seed=16,
        )

        after_mass, after_min_share = restricted_target_metrics(
            model,
            batch,
            branch_targets,
        )
        self.assertGreater(after_mass, before_mass)
        self.assertGreater(after_min_share, before_min_share)

    def test_branch_target_replay_coverage_uses_pool_targets_beyond_batch(
        self,
    ) -> None:
        near, examples, tokenizer = branch_target_fixture(include_blue=True)
        model = initialized_target_model(
            tokenizer,
            seed=57,
            token_biases={".": 5.0, "n": 4.0},
        )
        batch = target_balanced_branch_batch(
            model,
            tokenizer,
            near,
            examples,
            rng_seed=15,
            batch_size=2,
        )
        replay_targets = replay_target_ids(model, tokenizer, examples)
        batch_target_set = set(branch_targets_from_batch(batch))
        missing_targets = set(replay_targets) - batch_target_set
        self.assertEqual(len(batch_target_set), 2)
        self.assertGreater(len(replay_targets), len(batch_target_set))
        self.assertTrue(missing_targets)

        before_mass, before_missing_share = replay_target_metrics(
            model,
            batch,
            replay_targets,
            missing_targets,
        )

        train_target_replay_coverage_steps(
            model,
            tokenizer,
            near,
            examples,
            repeat=64,
            rng_seed=16,
        )

        after_mass, after_missing_share = replay_target_metrics(
            model,
            batch,
            replay_targets,
            missing_targets,
        )
        self.assertGreater(after_mass, before_mass)
        self.assertGreater(after_missing_share, before_missing_share)


if __name__ == "__main__":
    unittest.main()
