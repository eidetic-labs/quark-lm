from __future__ import annotations

import random
import unittest

from support.branch_training import branch_training_fixture
from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from support.direct_answer import (
    audit_prompt_context_coverage,
    direct_answer_branch_context,
    direct_answer_hard_branch_contrast,
    direct_answer_lesson,
    direct_answer_target_balanced_branch_diversity_batch,
    train_direct_answer_branch_contrast_unlikelihood,
    train_direct_answer_branch_topk_softmax_unlikelihood,
    train_direct_answer_hard_branch_contrast_unlikelihood,
)


class TransformerDirectAnswerBranchContrastRepairsTest(unittest.TestCase):
    def test_branch_topk_softmax_lifts_target_within_hard_candidate_set(self) -> None:
        fixture = branch_training_fixture(seed=52)
        fixture.model.bout[fixture.tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_target_balanced_branch_diversity_batch(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            fixture.examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        def restricted_target_probability() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                probs = fixture.model.predict(context)
                hard_candidates = [
                    index
                    for index in sorted(
                        range(len(probs)),
                        key=lambda item: probs[item],
                        reverse=True,
                    )
                    if index != target
                ][:5]
                denominator = probs[target] + sum(
                    probs[candidate] for candidate in hard_candidates
                )
                total += probs[target] / denominator
            return total / len(batch)

        before_probability = restricted_target_probability()
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_topk_softmax_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.near,
                fixture.examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=1.0,
                positive_weight=1.0,
                candidate_weight=2.0,
                branch_position=1,
                batch_size=3,
                candidate_count=5,
                terminator=ANSWER_TERMINATOR,
                balance_targets=True,
            )

        self.assertGreater(restricted_target_probability(), before_probability)

    def test_prompt_context_coverage_marks_truncated_semantic_prompt(self) -> None:
        records = [
            {
                "id": "color-teacher-tree",
                "prompt": "which color belongs to teacher tree\nanswer:",
                "target": " green.",
            }
        ]
        narrow = audit_prompt_context_coverage(records, context_size=32)
        wide = audit_prompt_context_coverage(records, context_size=64)

        self.assertEqual(narrow["semantic_records"], 1)
        self.assertEqual(narrow["missing"], 1)
        self.assertIn("intent:color", narrow["missing_records"][0]["missing_features"])
        self.assertEqual(wide["covered"], 1)
        self.assertEqual(wide["missing_records"], [])

    def test_direct_answer_branch_contrast_separates_prompt_branches(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tokenizer = CharTokenizer.train(
            near.prompt + near.target + green.prompt + green.target + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=36,
            )
        )
        near_branch = direct_answer_branch_context(
            model,
            tokenizer,
            near,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        green_branch = direct_answer_branch_context(
            model,
            tokenizer,
            green,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(near_branch)
        self.assertIsNotNone(green_branch)
        near_context, near_target, _near_position = near_branch
        green_context, green_target, _green_position = green_branch
        near_lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        before_target = model.nll(near_context, near_target) + model.nll(
            green_context,
            green_target,
        )
        near_probs = model.predict(near_context)
        green_probs = model.predict(green_context)
        before_margin = (
            near_probs[near_target]
            - near_probs[green_target]
            + green_probs[green_target]
            - green_probs[near_target]
        )
        rng = random.Random(14)

        for _ in range(96):
            train_direct_answer_branch_contrast_unlikelihood(
                model,
                tokenizer,
                near,
                [green],
                near_lesson,
                rng,
                learning_rate=0.05,
                negative_weight=1.0,
                contrast_weight=1.0,
                branch_position=1,
                terminator=ANSWER_TERMINATOR,
            )

        after_target = model.nll(near_context, near_target) + model.nll(
            green_context,
            green_target,
        )
        near_probs = model.predict(near_context)
        green_probs = model.predict(green_context)
        after_margin = (
            near_probs[near_target]
            - near_probs[green_target]
            + green_probs[green_target]
            - green_probs[near_target]
        )
        self.assertEqual(tokenizer.itos[near_target], "n")
        self.assertEqual(tokenizer.itos[green_target], "g")
        self.assertGreater(before_target, after_target)
        self.assertGreater(after_margin, before_margin)

    def test_direct_answer_hard_branch_contrast_selects_confused_branch(self) -> None:
        fixture = branch_training_fixture(seed=37)
        fixture.model.bout[fixture.tokenizer.stoi["t"]].data = 4.0
        fixture.model.bout[fixture.tokenizer.stoi["g"]].data = 1.0
        contrast = direct_answer_hard_branch_contrast(
            fixture.model,
            fixture.tokenizer,
            fixture.near,
            [fixture.green, fixture.tree],
            random.Random(15),
            branch_position=1,
            hard_negative_count=0,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(contrast)
        context, target_id, _contrast_context, contrast_target = contrast
        before_wrong = fixture.model.predict(context)[contrast_target]
        lesson = direct_answer_lesson(
            fixture.tokenizer,
            fixture.model.config.context_size,
            fixture.near,
            ANSWER_TERMINATOR,
        )

        for _ in range(24):
            train_direct_answer_hard_branch_contrast_unlikelihood(
                fixture.model,
                fixture.tokenizer,
                fixture.near,
                [fixture.green, fixture.tree],
                lesson,
                random.Random(16),
                learning_rate=0.05,
                negative_weight=1.0,
                positive_weight=1.0,
                contrast_weight=1.0,
                branch_position=1,
                hard_negative_count=0,
                terminator=ANSWER_TERMINATOR,
            )

        self.assertEqual(fixture.tokenizer.itos[target_id], "n")
        self.assertEqual(fixture.tokenizer.itos[contrast_target], "t")
        self.assertGreater(before_wrong, fixture.model.predict(context)[contrast_target])


if __name__ == "__main__":
    unittest.main()
