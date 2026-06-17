from __future__ import annotations

from transformer_char_model_test_support import *  # noqa: F401,F403


class TransformerBranchReplayTest(unittest.TestCase):
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

    def test_branch_replay_plan_tracks_profile_deficits_independently(
        self,
    ) -> None:
        replay_branches = [
            ([0], 1, 1, "qa:place"),
            ([0], 2, 1, "qa:place"),
            ([0], 2, 2, "qa:color"),
        ]

        global_plan = branch_replay_plan(
            replay_branches,
            replay_branches,
            profile_aware_targets=False,
        )
        profile_plan = branch_replay_plan(
            replay_branches,
            replay_branches,
            profile_aware_targets=True,
        )

        self.assertEqual(
            global_plan["profiles"]["__all__"]["missing_target_ids"],
            [],
        )
        self.assertEqual(
            profile_plan["profiles"]["qa:place"]["missing_target_ids"],
            [2],
        )
        self.assertEqual(
            profile_plan["profiles"]["qa:color"]["missing_target_ids"],
            [],
        )
        self.assertEqual(
            profile_plan["profiles"]["qa:place"]["coverage_floor"],
            0.5,
        )
        with tempfile.TemporaryDirectory() as temp:
            plan_path = Path(temp) / "direct_answer_replay_plan.json"
            plan_path.write_text(
                json.dumps(profile_plan, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            loaded_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertTrue(loaded_plan["profile_aware_targets"])
        self.assertEqual(
            loaded_plan["profiles"]["qa:place"]["missing_target_count"],
            1,
        )

    def test_profiled_replay_records_preserve_sources_for_shared_targets(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        nine = AnswerExample(prompt="q: number?\na:", target=" nine.", source="qa:number")
        examples = [near, nine]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + nine.prompt
            + nine.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=63,
            )
        )

        records = direct_answer_profiled_replay_records(
            model,
            tokenizer,
            examples,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        target_ids_by_profile = {
            profile: target for _context, target, _predicted, profile in records
        }
        self.assertEqual(set(target_ids_by_profile), {"qa:place", "qa:number"})
        self.assertEqual(
            target_ids_by_profile["qa:place"],
            target_ids_by_profile["qa:number"],
        )

    def test_branch_replay_coverage_falls_back_to_sampled_branches(
        self,
    ) -> None:
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=3,
                context_size=4,
                embedding_dim=3,
                feedforward_dim=4,
                seed=67,
            )
        )

        loss = model.train_step_with_branch_context_replay_coverage(
            [([0, 0, 0, 0], 1, 2)],
            [],
            learning_rate=0.01,
            negative_weight=0.0,
            positive_weight=0.0,
            replay_weight=1.0,
            hard_negative_count=1,
        )

        self.assertGreater(loss, 0.0)

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

    def test_profile_target_share_balancing_lifts_minority_replay_target(
        self,
    ) -> None:
        def initialized_model() -> TinyTransformerLM:
            initialized = TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=4,
                    context_size=1,
                    embedding_dim=3,
                    feedforward_dim=4,
                    seed=71,
                )
            )
            initialized.bout[1].data = 2.5
            initialized.bout[2].data = -2.5
            return initialized

        baseline_model = initialized_model()
        balanced_model = initialized_model()
        majority_branch = ([0], 1, 1, "qa:mixed")
        minority_branch = ([0], 2, 1, "qa:mixed")
        replay_branches = [majority_branch] * 8 + [minority_branch]

        def minority_target_share(scored_model: TinyTransformerLM) -> float:
            probs = scored_model.predict([0])
            return probs[2] / (probs[1] + probs[2])

        before_share = minority_target_share(balanced_model)

        for _ in range(40):
            baseline_model.train_step_with_branch_context_replay_coverage(
                replay_branches,
                replay_branches,
                learning_rate=0.04,
                negative_weight=0.0,
                positive_weight=0.0,
                replay_weight=1.0,
                hard_negative_count=0,
                profile_aware_targets=True,
            )
            balanced_model.train_step_with_branch_context_replay_coverage(
                replay_branches,
                replay_branches,
                learning_rate=0.04,
                negative_weight=0.0,
                positive_weight=0.0,
                replay_weight=1.0,
                hard_negative_count=0,
                profile_aware_targets=True,
                balance_profile_target_shares=True,
            )

        self.assertGreater(minority_target_share(balanced_model), before_share)
        self.assertGreater(
            minority_target_share(balanced_model),
            minority_target_share(baseline_model),
        )

    def test_profile_prompt_ownership_margin_lifts_context_specific_target(
        self,
    ) -> None:
        def initialized_model() -> TinyTransformerLM:
            initialized = TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=4,
                    context_size=1,
                    embedding_dim=3,
                    feedforward_dim=4,
                    seed=83,
                )
            )
            initialized.bout[1].data = 3.0
            initialized.bout[2].data = -3.0
            return initialized

        def ownership_margin(
            scored_model: TinyTransformerLM,
            context: list[int],
            target: int,
            rival: int,
        ) -> float:
            logits = scored_model._forward_floats(context)
            return logits[target] - logits[rival]

        baseline_model = initialized_model()
        prompt_model = initialized_model()
        first_context = [0]
        second_context = [3]
        first_branch = (first_context, 1, 1, "qa:mixed")
        second_branch = (second_context, 2, 1, "qa:mixed")
        replay_branches = [first_branch, second_branch]
        before_second_margin = ownership_margin(prompt_model, second_context, 2, 1)

        for _ in range(40):
            baseline_model.train_step_with_branch_context_replay_coverage(
                replay_branches,
                replay_branches,
                learning_rate=0.03,
                negative_weight=0.0,
                positive_weight=0.0,
                replay_weight=1.0,
                hard_negative_count=0,
                profile_aware_targets=True,
                balance_profile_target_shares=True,
            )
            prompt_model.train_step_with_branch_context_replay_coverage(
                replay_branches,
                replay_branches,
                learning_rate=0.03,
                negative_weight=0.0,
                positive_weight=0.0,
                replay_weight=1.0,
                hard_negative_count=0,
                profile_aware_targets=True,
                balance_profile_target_shares=True,
                enforce_prompt_target_margins=True,
            )

        self.assertGreater(
            ownership_margin(prompt_model, second_context, 2, 1),
            before_second_margin,
        )
        self.assertGreater(
            ownership_margin(prompt_model, second_context, 2, 1),
            ownership_margin(baseline_model, second_context, 2, 1),
        )

    def test_profiled_branch_batch_can_use_baseline_prediction_overrides(
        self,
    ) -> None:
        green = AnswerExample(
            prompt="q: color?\na:",
            target=" green.",
            source="qa:color",
        )
        blue = AnswerExample(
            prompt="q: color?\na:",
            target=" blue.",
            source="qa:color",
        )
        tokenizer = CharTokenizer.train(
            green.prompt + green.target + blue.prompt + blue.target + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=84,
            )
        )
        model.bout[tokenizer.stoi["g"]].data = 4.0
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            green,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )
        self.assertIsNotNone(branch)
        context, target_id, _position = branch  # type: ignore[misc]
        override_id = tokenizer.stoi["b"]
        current_prediction = max(
            range(tokenizer.vocab_size),
            key=lambda index: model.predict(context)[index],
        )

        batch = direct_answer_profiled_branch_batch(
            model,
            tokenizer,
            green,
            [green, blue],
            random.Random(84),
            branch_position=1,
            batch_size=1,
            terminator=ANSWER_TERMINATOR,
            prediction_overrides={
                (tuple(context), target_id, "qa:color"): override_id,
            },
        )

        self.assertNotEqual(current_prediction, override_id)
        self.assertEqual(batch, [(context, target_id, override_id, "qa:color")])

    def test_baseline_prediction_anchor_protects_covered_target_after_drift(
        self,
    ) -> None:
        def initialized_model() -> TinyTransformerLM:
            initialized = TinyTransformerLM.init_random(
                TransformerConfig(
                    vocab_size=4,
                    context_size=1,
                    embedding_dim=3,
                    feedforward_dim=4,
                    seed=85,
                )
            )
            initialized.bout[1].data = -2.0
            initialized.bout[2].data = 2.0
            return initialized

        def covered_probability(scored_model: TinyTransformerLM) -> float:
            probs = scored_model.predict([0])
            return probs[1]

        dynamic_model = initialized_model()
        anchored_model = initialized_model()
        dynamic_replay = [([0], 2, 2, "qa:mixed"), ([3], 1, 2, "qa:mixed")]
        anchored_replay = [([0], 2, 1, "qa:mixed"), ([3], 1, 2, "qa:mixed")]
        before_anchor_probability = covered_probability(anchored_model)

        for _ in range(30):
            dynamic_model.train_step_with_branch_context_replay_coverage(
                dynamic_replay,
                dynamic_replay,
                learning_rate=0.03,
                negative_weight=0.0,
                positive_weight=0.0,
                replay_weight=1.0,
                hard_negative_count=0,
                preserve_predicted_target_coverage=True,
                profile_aware_targets=True,
                balance_profile_target_shares=True,
                enforce_prompt_target_margins=True,
            )
            anchored_model.train_step_with_branch_context_replay_coverage(
                dynamic_replay,
                anchored_replay,
                learning_rate=0.03,
                negative_weight=0.0,
                positive_weight=0.0,
                replay_weight=1.0,
                hard_negative_count=0,
                preserve_predicted_target_coverage=True,
                profile_aware_targets=True,
                balance_profile_target_shares=True,
                enforce_prompt_target_margins=True,
            )

        self.assertGreater(covered_probability(anchored_model), before_anchor_probability)
        self.assertGreater(
            covered_probability(anchored_model),
            covered_probability(dynamic_model),
        )

    def test_baseline_floor_gated_prompt_mode_records_update_guard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "baseline-floor-gated-screen"),
                    "--steps",
                    "0",
                    "--eval-every",
                    "0",
                    "--candidate-scope",
                    "eval",
                    "--direct-answer-steps",
                    "1",
                    "--direct-answer-eval-every",
                    "1",
                    "--direct-answer-mode",
                    (
                        "branch-balanced-context-profile-baseline-floor-gated-"
                        "prompt-ownership-target-share-preserving-deficit-"
                        "unlikelihood"
                    ),
                    "--direct-answer-snapshot-mode",
                    "branch-only",
                    "--direct-answer-branch-batch-size",
                    "2",
                    "--direct-answer-hard-negatives",
                    "1",
                    "--skip-post-direct-snapshot",
                    "--embedding-dim",
                    "2",
                    "--feedforward-dim",
                    "4",
                    "--context-size",
                    "80",
                ]
            )

            metrics = train_transformer_answers(args)

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        self.assertTrue(direct_answer["direct_answer_replay_prediction_anchors_active"])
        self.assertTrue(direct_answer["direct_answer_baseline_floor_update_gate_active"])
        self.assertTrue(guard["active"])
        self.assertEqual(guard["checked_steps"], 1)
        self.assertEqual(
            guard["accepted_steps"] + guard["rejected_steps"],
            guard["checked_steps"],
        )

    def test_baseline_floor_adaptive_prompt_mode_records_retry_guard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "baseline-floor-adaptive-screen"),
                    "--steps",
                    "0",
                    "--eval-every",
                    "0",
                    "--candidate-scope",
                    "eval",
                    "--direct-answer-steps",
                    "1",
                    "--direct-answer-eval-every",
                    "1",
                    "--direct-answer-mode",
                    (
                        "branch-balanced-context-profile-baseline-floor-adaptive-"
                        "prompt-ownership-target-share-preserving-deficit-"
                        "unlikelihood"
                    ),
                    "--direct-answer-snapshot-mode",
                    "branch-only",
                    "--direct-answer-branch-batch-size",
                    "2",
                    "--direct-answer-hard-negatives",
                    "1",
                    "--skip-post-direct-snapshot",
                    "--embedding-dim",
                    "2",
                    "--feedforward-dim",
                    "4",
                    "--context-size",
                    "80",
                ]
            )

            metrics = train_transformer_answers(args)

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        self.assertTrue(direct_answer["direct_answer_replay_prediction_anchors_active"])
        self.assertTrue(direct_answer["direct_answer_baseline_floor_update_gate_active"])
        self.assertTrue(
            direct_answer["direct_answer_baseline_floor_adaptive_updates_active"]
        )
        self.assertTrue(guard["active"])
        self.assertTrue(guard["adaptive"])
        self.assertEqual(guard["checked_steps"], 1)
        self.assertGreaterEqual(guard["attempted_updates"], guard["checked_steps"])
        self.assertEqual(
            guard["accepted_steps"] + guard["rejected_steps"],
            guard["checked_steps"],
        )
        self.assertEqual(
            guard["accepted_attempts"] + guard["rejected_attempts"],
            guard["attempted_updates"],
        )

    def test_baseline_floor_repair_anchor_records_keep_covered_predictions(
        self,
    ) -> None:
        anchors = baseline_floor_repair_anchor_records(
            [
                ([1, 2], 4, 4, "qa:place"),
                ([1, 3], 5, 4, "qa:place"),
                ([1, 4], 6, 9, "qa:place"),
                ([2, 2], 7, 8, "qa:owner"),
                ([2, 3], 8, 8, "qa:owner"),
            ]
        )

        self.assertEqual(
            anchors,
            [
                ([1, 2], 4, 4, "qa:place"),
                ([1, 3], 4, 4, "qa:place"),
                ([2, 2], 8, 8, "qa:owner"),
                ([2, 3], 8, 8, "qa:owner"),
            ],
        )

    def test_baseline_floor_objective_anchor_batch_balances_profile_targets(
        self,
    ) -> None:
        anchors = [
            ([1, 2], 4, 4, "qa:place"),
            ([1, 3], 4, 4, "qa:place"),
            ([2, 2], 8, 8, "qa:owner"),
            ([2, 3], 9, 9, "qa:owner"),
            ([3, 3], 9, 9, "qa:owner"),
        ]

        batch = baseline_floor_objective_anchor_batch(
            anchors,
            random.Random(11),
            batch_size=10,
        )

        profile_targets = {
            (profile, target)
            for _context, target, _predicted, profile in batch
        }
        self.assertEqual(
            profile_targets,
            {("qa:place", 4), ("qa:owner", 8), ("qa:owner", 9)},
        )
        self.assertEqual(baseline_floor_anchor_profile_target_count(anchors), 3)
        self.assertEqual(
            {
                profile: len(group)
                for profile, group in baseline_floor_anchor_profile_groups(
                    anchors
                ).items()
            },
            {"qa:owner": 3, "qa:place": 2},
        )

    def test_baseline_floor_anchor_batch_update_lowers_anchor_nll(self) -> None:
        text = "where? near.\nwho? owner.\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=89,
        )
        model = TinyTransformerLM.init_random(config)
        target = ids[4]
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        before = model.nll(context, target)

        for _step in range(10):
            train_direct_answer_baseline_floor_anchor_batch(
                model,
                [(context, target, target, "qa:place")],
                learning_rate=0.05,
            )

        self.assertLess(model.nll(context, target), before)

    def test_baseline_floor_repaired_prompt_mode_records_repair_guard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(
                [
                    "answer-train",
                    "--run",
                    str(Path(temp) / "baseline-floor-repaired-screen"),
                    "--steps",
                    "0",
                    "--eval-every",
                    "0",
                    "--candidate-scope",
                    "eval",
                    "--direct-answer-steps",
                    "1",
                    "--direct-answer-eval-every",
                    "1",
                    "--direct-answer-mode",
                    (
                        "branch-balanced-context-profile-baseline-floor-repaired-"
                        "prompt-ownership-target-share-preserving-deficit-"
                        "unlikelihood"
                    ),
                    "--direct-answer-snapshot-mode",
                    "branch-only",
                    "--direct-answer-branch-batch-size",
                    "2",
                    "--direct-answer-hard-negatives",
                    "1",
                    "--skip-post-direct-snapshot",
                    "--embedding-dim",
                    "2",
                    "--feedforward-dim",
                    "4",
                    "--context-size",
                    "80",
                ]
            )

            metrics = train_transformer_answers(args)

        direct_answer = metrics["direct_answer"]
        guard = direct_answer["direct_answer_update_guard"]
        replay_plan = direct_answer["direct_answer_replay_plan_summary"]
        self.assertTrue(direct_answer["direct_answer_replay_prediction_anchors_active"])
        self.assertTrue(direct_answer["direct_answer_baseline_floor_update_gate_active"])
        self.assertTrue(
            direct_answer["direct_answer_baseline_floor_adaptive_updates_active"]
        )
        self.assertTrue(
            direct_answer["direct_answer_baseline_floor_repaired_updates_active"]
        )
        self.assertTrue(guard["active"])
        self.assertTrue(guard["adaptive"])
        self.assertTrue(guard["repair_active"])
        self.assertGreaterEqual(guard["repair_anchor_count"], 0)
        self.assertEqual(
            guard["repair_anchor_count"],
            replay_plan["baseline_floor_repair_anchor_count"],
        )
        self.assertEqual(guard["repair_steps_per_attempt"], 1)
        self.assertEqual(guard["checked_steps"], 1)
        self.assertGreaterEqual(guard["attempted_updates"], guard["checked_steps"])
        self.assertEqual(
            guard["accepted_steps"] + guard["rejected_steps"],
            guard["checked_steps"],
        )
        self.assertEqual(
            guard["accepted_attempts"] + guard["rejected_attempts"],
            guard["attempted_updates"],
        )


if __name__ == "__main__":
    unittest.main()
