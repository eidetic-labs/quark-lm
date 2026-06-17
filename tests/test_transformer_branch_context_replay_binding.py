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
    direct_answer_branch_diversity_batch,
    direct_answer_lesson,
    direct_answer_target_balanced_branch_diversity_batch,
    train_direct_answer_branch_context_replay_coverage_unlikelihood,
)


class TransformerBranchContextReplayBindingTest(unittest.TestCase):
    def test_branch_context_replay_coverage_lifts_owned_replay_targets(
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
                seed=58,
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
        replay_branches = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(16),
            branch_position=1,
            batch_size=4,
            terminator=ANSWER_TERMINATOR,
        )
        batch_targets = {target for _context, target, _predicted in batch}
        replay_targets = sorted(
            {target for _context, target, _predicted in replay_branches}
        )
        replay_target_set = set(replay_targets)
        self.assertEqual(len(batch_targets), 2)
        self.assertGreater(len(replay_targets), len(batch_targets))

        def replay_context_metrics() -> tuple[float, float]:
            target_set_total = 0.0
            owned_shares = []
            for context, target, _predicted in replay_branches:
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
                target_offset = replay_targets.index(target)
                owned_shares.append(target_values[target_offset] / target_set_mass)
            return target_set_total / len(replay_branches), min(owned_shares)

        before_mass, before_owned_share = replay_context_metrics()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(17)

        for _ in range(80):
            train_direct_answer_branch_context_replay_coverage_unlikelihood(
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

        after_mass, after_owned_share = replay_context_metrics()
        self.assertGreater(after_mass, before_mass)
        self.assertGreater(after_owned_share, before_owned_share)

    def test_branch_context_coverage_anchor_lifts_covered_target_probability(
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
                    seed=59,
                )
            )
            initialized.bout[tokenizer.stoi["n"]].data = 5.0
            initialized.bout[tokenizer.stoi["."]].data = 4.0
            return initialized

        model = initialized_model()
        anchored_model = initialized_model()
        covered_branch = direct_answer_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=1,
            terminator=ANSWER_TERMINATOR,
        )[0]
        context, target, predicted = covered_branch
        self.assertEqual(target, predicted)
        replay_branches = direct_answer_target_balanced_branch_diversity_batch(
            model,
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

        def covered_anchor_probability(scored_model: TinyTransformerLM) -> float:
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

        before_probability = covered_anchor_probability(model)
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        unanchored_rng = random.Random(17)
        anchored_rng = random.Random(17)

        for _ in range(48):
            train_direct_answer_branch_context_replay_coverage_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                unanchored_rng,
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
        for _ in range(48):
            train_direct_answer_branch_context_replay_coverage_unlikelihood(
                anchored_model,
                tokenizer,
                near,
                examples,
                lesson,
                anchored_rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                branch_position=1,
                batch_size=2,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
                preserve_covered_targets=True,
            )

        self.assertLess(covered_anchor_probability(model), before_probability)
        self.assertGreater(
            covered_anchor_probability(anchored_model),
            covered_anchor_probability(model),
        )

    def test_branch_context_target_balanced_anchor_skips_single_covered_target(
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
                    seed=60,
                )
            )
            initialized.bout[tokenizer.stoi["n"]].data = 5.0
            initialized.bout[tokenizer.stoi["."]].data = 4.0
            return initialized

        unanchored_model = initialized_model()
        global_anchor_model = initialized_model()
        balanced_anchor_model = initialized_model()
        replay_branches = direct_answer_target_balanced_branch_diversity_batch(
            unanchored_model,
            tokenizer,
            near,
            examples,
            random.Random(16),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        covered_targets = {
            target
            for _context, target, predicted in replay_branches
            if target == predicted
        }
        self.assertEqual(covered_targets, {tokenizer.stoi["n"]})
        branch_batch = direct_answer_target_balanced_branch_diversity_batch(
            unanchored_model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=2,
            terminator=ANSWER_TERMINATOR,
        )
        replay_targets = sorted(
            {target for _context, target, _predicted in replay_branches}
        )
        replay_target_set = set(replay_targets)
        context, target, _predicted = replay_branches[0]

        def covered_anchor_probability(scored_model: TinyTransformerLM) -> float:
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

        before_probability = covered_anchor_probability(unanchored_model)

        for _ in range(48):
            unanchored_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
            )
            global_anchor_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
                preserve_covered_targets=True,
            )
            balanced_anchor_model.train_step_with_branch_context_replay_coverage(
                branch_batch,
                replay_branches,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=0.0,
                replay_weight=2.0,
                hard_negative_count=5,
                preserve_covered_targets=True,
                balance_covered_target_anchors=True,
            )

        self.assertLess(
            covered_anchor_probability(unanchored_model),
            before_probability,
        )
        self.assertGreater(
            covered_anchor_probability(global_anchor_model),
            covered_anchor_probability(unanchored_model),
        )
        self.assertAlmostEqual(
            covered_anchor_probability(balanced_anchor_model),
            covered_anchor_probability(unanchored_model),
            places=12,
        )

if __name__ == "__main__":
    unittest.main()
