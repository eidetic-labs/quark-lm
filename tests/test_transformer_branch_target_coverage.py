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
    direct_answer_branch_target_ids,
    direct_answer_lesson,
    direct_answer_target_balanced_branch_diversity_batch,
    train_direct_answer_branch_target_diversity_unlikelihood,
    train_direct_answer_branch_target_replay_coverage_unlikelihood,
    train_direct_answer_branch_target_set_coverage_unlikelihood,
)


class TransformerBranchTargetCoverageTest(unittest.TestCase):
    def test_branch_target_set_coverage_lifts_set_without_exact_sharpening(
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
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=55,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = sorted({target for _context, target, _predicted in batch})
        branch_target_set = set(branch_targets)
        self.assertGreater(len(branch_targets), 1)

        def restricted_target_set_mass() -> float:
            total = 0.0
            for context, _target, _predicted in batch:
                probs = model.predict(context)
                hard_candidates = [
                    index
                    for index in sorted(
                        range(len(probs)),
                        key=lambda item: probs[item],
                        reverse=True,
                    )
                    if index not in branch_target_set
                ][:5]
                candidate_ids = [*branch_targets, *hard_candidates]
                denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
                total += (
                    sum(probs[branch_target] for branch_target in branch_targets)
                    / denominator
                )
            return total / len(batch)

        before_mass = restricted_target_set_mass()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_target_set_coverage_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                coverage_weight=2.0,
                branch_position=1,
                batch_size=3,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        self.assertGreater(restricted_target_set_mass(), before_mass)

    def test_branch_target_diversity_lifts_set_and_target_share_balance(
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
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=56,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        model.bout[tokenizer.stoi["n"]].data = 4.0
        batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = sorted({target for _context, target, _predicted in batch})
        branch_target_set = set(branch_targets)
        self.assertGreater(len(branch_targets), 1)

        def restricted_target_metrics() -> tuple[float, float]:
            target_set_total = 0.0
            target_share_totals = [0.0 for _branch_target in branch_targets]
            for context, _target, _predicted in batch:
                probs = model.predict(context)
                hard_candidates = [
                    index
                    for index in sorted(
                        range(len(probs)),
                        key=lambda item: probs[item],
                        reverse=True,
                    )
                    if index not in branch_target_set
                ][:5]
                candidate_ids = [*branch_targets, *hard_candidates]
                denominator = sum(probs[candidate_id] for candidate_id in candidate_ids)
                target_values = [
                    probs[branch_target] / denominator
                    for branch_target in branch_targets
                ]
                target_set_mass = sum(target_values)
                target_set_total += target_set_mass
                for offset, target_value in enumerate(target_values):
                    target_share_totals[offset] += target_value / target_set_mass
            average_target_shares = [
                target_share_total / len(batch)
                for target_share_total in target_share_totals
            ]
            return target_set_total / len(batch), min(average_target_shares)

        before_mass, before_min_share = restricted_target_metrics()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_target_diversity_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                diversity_weight=2.0,
                branch_position=1,
                batch_size=3,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        after_mass, after_min_share = restricted_target_metrics()
        self.assertGreater(after_mass, before_mass)
        self.assertGreater(after_min_share, before_min_share)

    def test_branch_target_replay_coverage_uses_pool_targets_beyond_batch(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        blue = AnswerExample(prompt="q: thing?\na:", target=" blue.", source="qa:thing")
        examples = [near, green, tree, blue]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + green.prompt
            + green.target
            + tree.prompt
            + tree.target
            + blue.prompt
            + blue.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=57,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        model.bout[tokenizer.stoi["n"]].data = 4.0
        batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=2,
            terminator=ANSWER_TERMINATOR,
        )
        replay_targets = direct_answer_branch_target_ids(
            model,
            tokenizer,
            examples,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        batch_target_set = {target for _context, target, _predicted in batch}
        replay_target_set = set(replay_targets)
        missing_targets = replay_target_set - batch_target_set
        self.assertEqual(len(batch_target_set), 2)
        self.assertGreater(len(replay_targets), len(batch_target_set))
        self.assertTrue(missing_targets)

        def replay_target_metrics() -> tuple[float, float]:
            target_set_total = 0.0
            missing_share_totals = [0.0 for _missing_target in missing_targets]
            missing_offsets = [
                offset
                for offset, replay_target in enumerate(replay_targets)
                if replay_target in missing_targets
            ]
            for context, _target, _predicted in batch:
                probs = model.predict(context)
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
                target_values = [
                    probs[replay_target] / denominator
                    for replay_target in replay_targets
                ]
                target_set_mass = sum(target_values)
                target_set_total += target_set_mass
                for missing_index, target_offset in enumerate(missing_offsets):
                    missing_share_totals[missing_index] += (
                        target_values[target_offset] / target_set_mass
                    )
            average_missing_shares = [
                missing_share_total / len(batch)
                for missing_share_total in missing_share_totals
            ]
            return target_set_total / len(batch), min(average_missing_shares)

        before_mass, before_missing_share = replay_target_metrics()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(64):
            train_direct_answer_branch_target_replay_coverage_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                branch_position=1,
                batch_size=2,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        after_mass, after_missing_share = replay_target_metrics()
        self.assertGreater(after_mass, before_mass)
        self.assertGreater(after_missing_share, before_missing_share)

if __name__ == "__main__":
    unittest.main()
