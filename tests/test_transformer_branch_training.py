from __future__ import annotations

from transformer_char_model_test_support import *  # noqa: F401,F403


class TransformerBranchTrainingTest(unittest.TestCase):
    def test_branch_batch_selects_distinct_branch_targets(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
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
                seed=40,
            )
        )

        batch = direct_answer_branch_batch(
            model,
            tokenizer,
            near,
            [near, green, tree],
            random.Random(11),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(len(batch), 3)
        self.assertEqual(
            {tokenizer.itos[target] for _context, target in batch},
            {"n", "g", "t"},
        )

    def test_target_balanced_branch_batch_samples_rare_targets(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        blue = AnswerExample(prompt="q: blue?\na:", target=" blue.", source="qa:color")
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
                seed=40,
            )
        )
        skewed_examples = [near for _ in range(20)] + [green, tree, blue]

        batch = direct_answer_target_balanced_branch_batch(
            model,
            tokenizer,
            near,
            skewed_examples,
            random.Random(11),
            branch_position=1,
            batch_size=4,
            terminator=ANSWER_TERMINATOR,
        )
        diversity_batch = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            near,
            skewed_examples,
            random.Random(11),
            branch_position=1,
            batch_size=4,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(len(batch), 4)
        self.assertEqual(
            {tokenizer.itos[target] for _context, target in batch},
            {"n", "g", "t", "b"},
        )
        self.assertEqual(
            {tokenizer.itos[target] for _context, target, _predicted in diversity_batch},
            {"n", "g", "t", "b"},
        )

    def test_branch_diversity_batch_records_current_predictions(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
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
                seed=44,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0

        batch = direct_answer_branch_diversity_batch(
            model,
            tokenizer,
            near,
            [near, green, tree],
            random.Random(14),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(len(batch), 3)
        self.assertEqual(
            {tokenizer.itos[target] for _context, target, _predicted in batch},
            {"n", "g", "t"},
        )
        self.assertEqual(
            {tokenizer.itos[predicted] for _context, _target, predicted in batch},
            {"."},
        )

    def test_branch_batch_contrast_improves_prompt_branch_margin(self) -> None:
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
                seed=41,
            )
        )
        batch = direct_answer_branch_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(12),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )
        branch_targets = {target for _context, target in batch}

        def branch_margin() -> float:
            total = 0.0
            for context, target in batch:
                probs = model.predict(context)
                strongest_other = max(
                    probs[other]
                    for other in branch_targets
                    if other != target
                )
                total += probs[target] - strongest_other
            return total

        before = branch_margin()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(13)

        for _ in range(64):
            train_direct_answer_branch_batch_contrast_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.06,
                negative_weight=1.0,
                positive_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        after = branch_margin()
        self.assertGreater(after, before)

    def test_branch_diversity_unlikelihood_suppresses_global_wrong_token(self) -> None:
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
                seed=45,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0
        batch = direct_answer_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        def batch_scores() -> tuple[float, float]:
            wrong_total = 0.0
            target_total = 0.0
            for context, target, _predicted in batch:
                probs = model.predict(context)
                wrong_total += probs[wrong_id]
                target_total += probs[target]
            return wrong_total, target_total

        before_wrong, before_target = batch_scores()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_diversity_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.06,
                negative_weight=1.0,
                positive_weight=1.0,
                contrast_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        after_wrong, after_target = batch_scores()
        self.assertLess(after_wrong, before_wrong)
        self.assertGreater(after_target, before_target)

    def test_branch_diversity_training_can_freeze_output_bias(self) -> None:
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
                seed=45,
            )
        )
        wrong_id = tokenizer.stoi["."]
        model.bout[wrong_id].data = 5.0
        params = exclude_scalars(model.parameters(), model.bout)
        before_bout = [value.data for value in model.bout]
        before_wout = model.wout[0][wrong_id].data
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )

        train_direct_answer_branch_diversity_unlikelihood(
            model,
            tokenizer,
            near,
            examples,
            lesson,
            random.Random(16),
            learning_rate=0.06,
            negative_weight=1.0,
            positive_weight=1.0,
            contrast_weight=1.0,
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
            params=params,
        )

        self.assertEqual([value.data for value in model.bout], before_bout)
        self.assertNotEqual(model.wout[0][wrong_id].data, before_wout)

    def test_branch_target_softmax_improves_restricted_branch_choice(self) -> None:
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
                seed=45,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_branch_diversity_batch(
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

        def restricted_target_probability() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                probs = model.predict(context)
                denominator = sum(probs[branch_target] for branch_target in branch_targets)
                total += probs[target] / denominator
            return total

        before = restricted_target_probability()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_target_softmax_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.06,
                negative_weight=1.0,
                positive_weight=1.0,
                target_softmax_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        after = restricted_target_probability()
        self.assertGreater(after, before)

    def test_branch_target_margin_improves_restricted_logit_gap(self) -> None:
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
                seed=42,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_branch_diversity_batch(
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

        def restricted_logit_gap() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                logits = model._forward_floats(context)
                strongest_other = max(
                    logits[other]
                    for other in branch_targets
                    if other != target
                )
                total += logits[target] - strongest_other
            return total

        before = restricted_logit_gap()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_target_margin_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.02,
                negative_weight=1.0,
                positive_weight=1.0,
                margin_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        after = restricted_logit_gap()
        self.assertGreater(after, before)

    def test_branch_hidden_projection_margin_improves_projection_gap(self) -> None:
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
                seed=43,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_branch_diversity_batch(
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

        def projection(context: list[int], token_id: int) -> float:
            hidden = model.final_hidden(context)
            output_weights = model._output_weights_floats()
            return sum(
                hidden[dim] * output_weights[dim][token_id]
                for dim in range(len(hidden))
            )

        def restricted_projection_gap() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                strongest_other = max(
                    projection(context, other)
                    for other in branch_targets
                    if other != target
                )
                total += projection(context, target) - strongest_other
            return total

        before = restricted_projection_gap()
        before_bias = [value.data for value in model.bout]
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_hidden_projection_margin_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.03,
                negative_weight=0.0,
                positive_weight=0.0,
                margin_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        after = restricted_projection_gap()
        after_bias = [value.data for value in model.bout]
        self.assertGreater(after, before)
        self.assertEqual(after_bias, before_bias)

    def test_branch_representation_profile_reports_hidden_distances(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        records = [
            {"id": "near", "prompt": near.prompt, "target": near.target},
            {"id": "green", "prompt": green.prompt, "target": green.target},
            {"id": "tree", "prompt": tree.prompt, "target": tree.target},
        ]
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
                seed=47,
            )
        )

        profile = direct_answer_branch_representation_profile(
            model,
            tokenizer,
            records,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(profile["count"], 3)
        self.assertEqual(profile["skipped"], 0)
        self.assertEqual(profile["target_unique"], 3)
        self.assertEqual(profile["different_target_pairwise_distance"]["count"], 3)
        self.assertGreater(profile["different_target_pairwise_distance"]["avg"], 0.0)
        self.assertEqual(len(profile["target_centroids"]), 3)
        self.assertEqual(profile["target_centroid_distance"]["count"], 3)
        self.assertEqual(profile["target_centroid_margin"]["count"], 3)

    def test_branch_logit_prior_profile_decomposes_dominant_bias(self) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        green = AnswerExample(prompt="q: color?\na:", target=" green.", source="qa:color")
        tree = AnswerExample(prompt="q: owner?\na:", target=" tree.", source="qa:owner")
        records = [
            {"id": "near", "prompt": near.prompt, "target": near.target},
            {"id": "green", "prompt": green.prompt, "target": green.target},
            {"id": "tree", "prompt": tree.prompt, "target": tree.target},
        ]
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
                seed=48,
            )
        )
        model.bout[tokenizer.stoi["n"]].data = 5.0

        profile = direct_answer_branch_logit_prior_profile(
            model,
            tokenizer,
            records,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(profile["count"], 3)
        self.assertEqual(profile["dominant_predicted_token"], "n")
        self.assertEqual(profile["dominant_token_bias_rank"], 1)
        missing_tokens = {item["value"] for item in profile["missing_target_tokens"]}
        self.assertIn("g", missing_tokens)
        self.assertIn("t", missing_tokens)
        failed = profile["dominant_vs_target_decomposition"]["failed_records"]
        self.assertEqual(failed["primary_pressure"], "output_bias")
        self.assertEqual(failed["count"], 2)
        self.assertGreater(failed["avg_bias_advantage"], 0.0)
        self.assertGreater(profile["dominant_vs_missing_bias_advantage"], 0.0)

    def test_branch_representation_contrast_increases_hidden_distance(self) -> None:
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
                seed=48,
            )
        )
        batch = direct_answer_branch_diversity_batch(
            model,
            tokenizer,
            near,
            examples,
            random.Random(15),
            branch_position=1,
            batch_size=3,
            terminator=ANSWER_TERMINATOR,
        )

        def average_hidden_distance() -> float:
            distances = []
            for left_index, (left_context, left_target, _left_predicted) in enumerate(batch):
                left_hidden = model.final_hidden(left_context)
                for right_context, right_target, _right_predicted in batch[left_index + 1:]:
                    if left_target == right_target:
                        continue
                    right_hidden = model.final_hidden(right_context)
                    distances.append(
                        sum(
                            (left_value - right_value) ** 2
                            for left_value, right_value in zip(left_hidden, right_hidden)
                        )
                    )
            return sum(distances) / len(distances)

        before = average_hidden_distance()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_representation_contrast_unlikelihood(
                model,
                tokenizer,
                near,
                examples,
                lesson,
                rng,
                learning_rate=0.04,
                negative_weight=0.0,
                positive_weight=0.0,
                representation_weight=1.0,
                branch_position=1,
                batch_size=3,
                terminator=ANSWER_TERMINATOR,
            )

        after = average_hidden_distance()
        self.assertGreater(after, before)

    def test_branch_output_binding_improves_rank_and_hidden_distance(self) -> None:
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
                seed=49,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_branch_diversity_batch(
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

        def restricted_target_probability() -> float:
            total = 0.0
            for context, target, _predicted in batch:
                probs = model.predict(context)
                denominator = sum(probs[branch_target] for branch_target in branch_targets)
                total += probs[target] / denominator
            return total

        def average_hidden_distance() -> float:
            distances = []
            for left_index, (left_context, left_target, _left_predicted) in enumerate(batch):
                left_hidden = model.final_hidden(left_context)
                for right_context, right_target, _right_predicted in batch[left_index + 1:]:
                    if left_target == right_target:
                        continue
                    right_hidden = model.final_hidden(right_context)
                    distances.append(
                        sum(
                            (left_value - right_value) ** 2
                            for left_value, right_value in zip(left_hidden, right_hidden)
                        )
                    )
            return sum(distances) / len(distances)

        before_probability = restricted_target_probability()
        before_distance = average_hidden_distance()
        lesson = direct_answer_lesson(
            tokenizer,
            model.config.context_size,
            near,
            ANSWER_TERMINATOR,
        )
        rng = random.Random(16)

        for _ in range(48):
            train_direct_answer_branch_output_binding_unlikelihood(
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
            )

        self.assertGreater(restricted_target_probability(), before_probability)
        self.assertGreater(average_hidden_distance(), before_distance)

    def test_branch_rank_margin_lifts_targets_above_hard_negatives(self) -> None:
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
                seed=50,
            )
        )
        model.bout[tokenizer.stoi["."]].data = 5.0
        batch = direct_answer_branch_diversity_batch(
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
            )

        self.assertLess(average_target_rank(), before_rank)


if __name__ == "__main__":
    unittest.main()
