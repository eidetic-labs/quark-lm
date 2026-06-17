from __future__ import annotations

import math
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
    direct_answer_lesson,
    direct_answer_target_balanced_branch_diversity_batch,
    train_direct_answer_branch_bidirectional_binding_unlikelihood,
    train_direct_answer_branch_coverage_binding_unlikelihood,
    train_direct_answer_branch_rank_margin_unlikelihood,
)


class TransformerBranchBindingTest(unittest.TestCase):
    def test_balanced_branch_rank_margin_uses_target_balanced_batches(self) -> None:
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
                seed=51,
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

        def average_target_rank() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                probs = model.predict(context)
                ranked = sorted(
                    range(len(probs)),
                    key=lambda index: probs[index],
                    reverse=True,
                )
                total += ranked.index(target) + 1
            return total / len(batch)

        before_rank = average_target_rank()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_rank_margin_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=1.0,
                margin_weight=2.0,
                branch_position=1,
                batch_size=3,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        self.assertLess(average_target_rank(), before_rank)

    def test_branch_bidirectional_binding_lifts_target_context_ownership(self) -> None:
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
                seed=53,
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
        self.assertGreater(len(branch_targets), 1)

        def average_target_context_ownership() -> float:
            total = 0.0
            for branch_target in branch_targets:
                target_logits = [
                    model._forward_floats(context)[branch_target]
                    for context, _target, _predicted in batch
                ]
                max_logit = max(target_logits)
                exp_scores = [
                    math.exp(target_logit - max_logit)
                    for target_logit in target_logits
                ]
                denominator = sum(exp_scores)
                owned_mass = 0.0
                for exp_score, (_context, target, _predicted) in zip(exp_scores, batch):
                    if target == branch_target:
                        owned_mass += exp_score / denominator
                total += owned_mass
            return total / len(branch_targets)

        before_ownership = average_target_context_ownership()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_bidirectional_binding_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=1.0,
                binding_weight=2.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        self.assertGreater(average_target_context_ownership(), before_ownership)

    def test_branch_coverage_binding_lifts_target_set_against_hard_wrong_tokens(
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
                seed=54,
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

        def restricted_probabilities() -> tuple[float, float]:
            target_set_total = 0.0
            target_total = 0.0
            for context, target, _predicted in batch:
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
                target_set_total += (
                    sum(probs[branch_target] for branch_target in branch_targets)
                    / denominator
                )
                target_total += probs[target] / denominator
            return target_set_total / len(batch), target_total / len(batch)

        before_target_set, before_target = restricted_probabilities()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_coverage_binding_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=1.0,
                binding_weight=2.0,
                branch_position=1,
                batch_size=3,
                hard_negative_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        after_target_set, after_target = restricted_probabilities()
        self.assertGreater(after_target_set, before_target_set)
        self.assertGreater(after_target, before_target)

if __name__ == "__main__":
    unittest.main()
