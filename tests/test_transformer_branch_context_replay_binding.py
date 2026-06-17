from __future__ import annotations

import unittest

from support.branch_context_replay_binding import (
    branch_binding_fixture,
    branch_diversity_batch,
    covered_replay_targets,
    initialized_replay_model,
    replay_context_metrics,
    replay_target_ids,
    target_balanced_branch_batch,
    target_normalized_probability,
    train_anchor_comparison_steps,
    train_replay_coverage_steps,
)


class TransformerBranchContextReplayBindingTest(unittest.TestCase):
    def test_branch_context_replay_coverage_lifts_owned_replay_targets(
        self,
    ) -> None:
        near, examples, tokenizer = branch_binding_fixture(include_blue=True)
        model = initialized_replay_model(
            tokenizer,
            seed=58,
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
        replay_branches = target_balanced_branch_batch(
            model,
            tokenizer,
            near,
            examples,
            rng_seed=16,
            batch_size=4,
        )
        batch_targets = {target for _context, target, _predicted in batch}
        replay_targets = replay_target_ids(replay_branches)
        self.assertEqual(len(batch_targets), 2)
        self.assertGreater(len(replay_targets), len(batch_targets))

        before_mass, before_owned_share = replay_context_metrics(
            model,
            replay_branches,
            replay_targets,
        )

        train_replay_coverage_steps(
            model,
            tokenizer,
            near,
            examples,
            repeat=80,
            rng_seed=17,
        )

        after_mass, after_owned_share = replay_context_metrics(
            model,
            replay_branches,
            replay_targets,
        )
        self.assertGreater(after_mass, before_mass)
        self.assertGreater(after_owned_share, before_owned_share)

    def test_branch_context_coverage_anchor_lifts_covered_target_probability(
        self,
    ) -> None:
        near, examples, tokenizer = branch_binding_fixture()
        model = initialized_replay_model(
            tokenizer,
            seed=59,
            token_biases={"n": 5.0, ".": 4.0},
        )
        anchored_model = initialized_replay_model(
            tokenizer,
            seed=59,
            token_biases={"n": 5.0, ".": 4.0},
        )
        covered_branch = branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            rng_seed=15,
            batch_size=1,
        )[0]
        context, target, predicted = covered_branch
        self.assertEqual(target, predicted)
        replay_branches = target_balanced_branch_batch(
            model,
            tokenizer,
            near,
            examples,
            rng_seed=16,
            batch_size=3,
        )
        replay_targets = replay_target_ids(replay_branches)

        before_probability = target_normalized_probability(
            model,
            context,
            target,
            replay_targets,
        )

        train_replay_coverage_steps(
            model,
            tokenizer,
            near,
            examples,
            repeat=48,
            rng_seed=17,
        )
        train_replay_coverage_steps(
            anchored_model,
            tokenizer,
            near,
            examples,
            repeat=48,
            rng_seed=17,
            preserve_covered_targets=True,
        )

        self.assertLess(
            target_normalized_probability(model, context, target, replay_targets),
            before_probability,
        )
        self.assertGreater(
            target_normalized_probability(
                anchored_model,
                context,
                target,
                replay_targets,
            ),
            target_normalized_probability(model, context, target, replay_targets),
        )

    def test_branch_context_target_balanced_anchor_skips_single_covered_target(
        self,
    ) -> None:
        near, examples, tokenizer = branch_binding_fixture()
        unanchored_model = initialized_replay_model(
            tokenizer,
            seed=60,
            token_biases={"n": 5.0, ".": 4.0},
        )
        global_anchor_model = initialized_replay_model(
            tokenizer,
            seed=60,
            token_biases={"n": 5.0, ".": 4.0},
        )
        balanced_anchor_model = initialized_replay_model(
            tokenizer,
            seed=60,
            token_biases={"n": 5.0, ".": 4.0},
        )
        replay_branches = target_balanced_branch_batch(
            unanchored_model,
            tokenizer,
            near,
            examples,
            rng_seed=16,
            batch_size=3,
        )
        self.assertEqual(covered_replay_targets(replay_branches), {tokenizer.stoi["n"]})
        branch_batch = target_balanced_branch_batch(
            unanchored_model,
            tokenizer,
            near,
            examples,
            rng_seed=15,
            batch_size=2,
        )
        replay_targets = replay_target_ids(replay_branches)
        context, target, _predicted = replay_branches[0]

        before_probability = target_normalized_probability(
            unanchored_model,
            context,
            target,
            replay_targets,
        )

        train_anchor_comparison_steps(
            unanchored_model,
            global_anchor_model,
            balanced_anchor_model,
            branch_batch,
            replay_branches,
            repeat=48,
        )

        self.assertLess(
            target_normalized_probability(
                unanchored_model,
                context,
                target,
                replay_targets,
            ),
            before_probability,
        )
        self.assertGreater(
            target_normalized_probability(
                global_anchor_model,
                context,
                target,
                replay_targets,
            ),
            target_normalized_probability(
                unanchored_model,
                context,
                target,
                replay_targets,
            ),
        )
        self.assertAlmostEqual(
            target_normalized_probability(
                balanced_anchor_model,
                context,
                target,
                replay_targets,
            ),
            target_normalized_probability(
                unanchored_model,
                context,
                target,
                replay_targets,
            ),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
