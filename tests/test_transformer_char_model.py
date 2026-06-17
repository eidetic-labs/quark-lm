from __future__ import annotations

from transformer_char_model_test_support import *  # noqa: F401,F403


class TransformerCharModelCoreTest(unittest.TestCase):
    def test_train_step_updates_random_transformer_weights(self) -> None:
        text = "question: where is mia's ball?\nanswer: under the box.\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=6,
            embedding_dim=4,
            feedforward_dim=8,
            seed=3,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(ids, 12, config.context_size, tokenizer.pad_id)
        target = ids[12]
        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.04)
        after = model.nll(context, target)

        self.assertGreater(before, after)

    def test_adamw_optimizer_accumulates_gradients_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=53,
        )
        model = TinyTransformerLM.init_random(config)
        optimizer = ScalarOptimizer(
            OptimizationConfig(
                optimizer="adamw",
                gradient_accumulation_steps=2,
                warmup_steps=2,
                decay_steps=2,
                min_learning_rate=0.001,
            )
        )
        model.active_optimizer = optimizer
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        before = model.nll(context, target)

        model.train_step(context, target, learning_rate=0.02)
        self.assertEqual(optimizer.update_count, 0)
        self.assertEqual(optimizer.pending_accumulation, 1)
        model.train_step(context, target, learning_rate=0.02)
        self.assertEqual(optimizer.update_count, 1)
        self.assertEqual(optimizer.pending_accumulation, 0)
        self.assertGreater(before, model.nll(context, target))

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "optimizer.json"
            save_optimizer_state(path, optimizer)
            loaded = load_optimizer_state(path, optimizer.config)

        self.assertEqual(loaded.update_count, optimizer.update_count)
        self.assertEqual(loaded.config.optimizer, "adamw")
        self.assertEqual(len(loaded.first_moment), len(optimizer.first_moment))

    def test_v051_architecture_options_forward_and_round_trip(self) -> None:
        text = "abcd abcd\n"
        tokenizer = CharTokenizer.train(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=54,
            attention_heads=2,
            use_pre_layer_norm=True,
            use_rms_norm=True,
            use_gated_mlp=True,
            tie_output_embeddings=True,
            use_rotary_positions=True,
            use_kv_cache_path=True,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(tokenizer.encode(text), 4, config.context_size, tokenizer.pad_id)
        probs = model.predict(context)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer, {"test": "v0.51"})
            loaded, loaded_tokenizer = TinyTransformerLM.load(path)

        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertTrue(loaded.config.tie_output_embeddings)
        self.assertEqual(loaded.config.attention_heads, 2)
        self.assertIsNotNone(loaded_tokenizer)

    def test_generation_controls_emit_trace_and_cache_metadata(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=55,
                use_kv_cache_path=True,
            )
        )
        generation = model.generate_with_trace(
            tokenizer,
            "abc",
            3,
            GenerationConfig(
                temperature=0.7,
                top_k=2,
                top_p=0.9,
                repetition_penalty=1.1,
                trace_top_tokens=2,
                use_kv_cache=True,
            ),
        )

        self.assertLessEqual(len(generation["trace"]), 3)
        self.assertTrue(generation["cache"]["enabled"])
        self.assertLessEqual(len(generation["trace"][0]["top_tokens"]), 2)

    def test_generation_distribution_applies_top_k_and_repetition_penalty(self) -> None:
        probs = generation_distribution(
            [0.6, 0.3, 0.1],
            [0],
            GenerationConfig(top_k=2, repetition_penalty=3.0),
        )

        self.assertEqual(probs[2], 0.0)
        self.assertGreater(probs[1], probs[0])
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_transformer_eval_scoring_returns_replayable_trace_records(self) -> None:
        example = {"id": "one", "prompt": "q:\na:", "target": " a."}
        text = example["prompt"] + example["target"] + ANSWER_TERMINATOR
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=56,
            )
        )

        records = score_transformer_records(
            model,
            tokenizer,
            [example],
            max_new_chars=2,
            generation_config=GenerationConfig(trace_top_tokens=2),
            candidates=[example["target"]],
        )

        self.assertEqual(records[0]["id"], "one")
        self.assertIn("generation_trace", records[0])
        self.assertIn("candidate_scores", records[0])

    def test_checkpoint_round_trip_includes_corpus_tokenizer(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=1,
        )
        model = TinyTransformerLM.init_random(config)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, loaded_tokenizer = TinyTransformerLM.load(path)

        self.assertIsNotNone(loaded_tokenizer)
        self.assertEqual(loaded.config.vocab_size, tokenizer.vocab_size)
        self.assertEqual(loaded_tokenizer.tokens, tokenizer.tokens)  # type: ignore[union-attr]

    def test_layer_normalized_transformer_trains_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=9,
            use_layer_norm=True,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        before = model.nll(context, target)
        for _ in range(30):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_layer_norm)
        self.assertIn("ln1_gain", weights)
        self.assertIn("ln2_gain", weights)
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_pre_layer_normalized_transformer_trains_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=11,
            use_pre_layer_norm=True,
        )
        model = TinyTransformerLM.init_random(config)
        baseline = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=11,
            )
        )
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        before = model.nll(context, target)
        params = model.parameters()

        self.assertNotEqual(baseline.final_hidden(context), model.final_hidden(context))
        self.assertTrue(any(param is model.ln1_gain[0] for param in params))
        self.assertTrue(any(param is model.ln2_gain[0] for param in params))
        self.assertTrue(any(param is model.final_ln_gain[0] for param in params))

        for _ in range(30):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_pre_layer_norm)
        self.assertIn("final_ln_gain", weights)
        self.assertIn("final_ln_bias", weights)
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_pre_layer_norm_scalar_and_float_forward_match(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=12,
                num_layers=2,
                use_pre_layer_norm=True,
            )
        )
        context = context_before(ids, 4, model.config.context_size, tokenizer.pad_id)
        scalar_logits = [value.data for value in model._forward_scalars(context)]
        float_logits = model._forward_floats(context)

        for expected, actual in zip(float_logits, scalar_logits):
            self.assertAlmostEqual(expected, actual)

    def test_legacy_transformer_checkpoint_defaults_pre_layer_norm_off(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=13,
            )
        )
        payload = model.to_dict(tokenizer)
        payload["config"].pop("use_pre_layer_norm", None)
        payload["weights"].pop("final_ln_gain", None)
        payload["weights"].pop("final_ln_bias", None)

        loaded, _loaded_tokenizer = TinyTransformerLM.from_dict(payload)

        self.assertFalse(loaded.config.use_pre_layer_norm)
        self.assertEqual(
            [value.data for value in loaded.final_ln_gain],
            [1.0 for _ in range(loaded.config.embedding_dim)],
        )

    def test_multi_layer_transformer_trains_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=10,
            num_layers=2,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertEqual(loaded.config.num_layers, 2)
        self.assertEqual(len(weights["extra_layers"]), 1)
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_context_mean_transformer_trains_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=13,
            use_context_mean=True,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        self.assertTrue(loaded.config.use_context_mean)
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_context_projection_starts_as_baseline_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        base_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=14,
        )
        projection_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=14,
            use_context_projection=True,
        )
        baseline = TinyTransformerLM.init_random(base_config)
        model = TinyTransformerLM.init_random(projection_config)
        context = context_before(ids, 4, projection_config.context_size, tokenizer.pad_id)
        target = ids[4]

        for expected, actual in zip(baseline.predict(context), model.predict(context)):
            self.assertAlmostEqual(expected, actual)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)
        projection_values = [
            value.data
            for value in (
                flatten_scalars(model.context_projection_w)
                + flatten_scalars(model.context_projection_b)
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_context_projection)
        self.assertIn("context_projection_w", weights)
        self.assertIn("context_projection_b", weights)
        self.assertTrue(any(abs(value) > 0.0 for value in projection_values))
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_prompt_prefix_projection_starts_as_baseline_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        base_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=16,
        )
        prefix_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=16,
            use_prompt_prefix_projection=True,
        )
        baseline = TinyTransformerLM.init_random(base_config)
        model = TinyTransformerLM.init_random(prefix_config)
        context = context_before(ids, 4, prefix_config.context_size, tokenizer.pad_id)
        target = ids[4]

        for expected, actual in zip(baseline.predict(context), model.predict(context)):
            self.assertAlmostEqual(expected, actual)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)
        projection_values = [
            value.data
            for value in (
                flatten_scalars(model.prompt_prefix_projection_w)
                + flatten_scalars(model.prompt_prefix_projection_b)
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_prompt_prefix_projection)
        self.assertIn("prompt_prefix_projection_w", weights)
        self.assertIn("prompt_prefix_projection_b", weights)
        self.assertTrue(any(abs(value) > 0.0 for value in projection_values))
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_prompt_position_projection_starts_as_baseline_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        base_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=18,
        )
        position_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=18,
            use_prompt_position_projection=True,
        )
        baseline = TinyTransformerLM.init_random(base_config)
        model = TinyTransformerLM.init_random(position_config)
        context = context_before(ids, 4, position_config.context_size, tokenizer.pad_id)
        target = ids[4]

        for expected, actual in zip(baseline.predict(context), model.predict(context)):
            self.assertAlmostEqual(expected, actual)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)
        projection_values = [
            value.data
            for value in (
                flatten_scalars(model.prompt_position_projection_w)
                + flatten_scalars(model.prompt_position_projection_b)
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_prompt_position_projection)
        self.assertEqual(loaded.config.prompt_position_projection_scale, 1.0)
        self.assertIn("prompt_position_projection_w", weights)
        self.assertIn("prompt_position_projection_b", weights)
        self.assertTrue(any(abs(value) > 0.0 for value in projection_values))
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_prompt_position_projection_scale_changes_nonzero_residual(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        base_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=19,
            use_prompt_position_projection=True,
        )
        scaled_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=19,
            use_prompt_position_projection=True,
            prompt_position_projection_scale=3.0,
        )
        base = TinyTransformerLM.init_random(base_config)
        scaled = TinyTransformerLM.init_random(scaled_config)
        base.prompt_position_projection_b[0].data = 0.25
        scaled.prompt_position_projection_b[0].data = 0.25
        context = context_before(ids, 4, base_config.context_size, tokenizer.pad_id)

        base_hidden = base.final_hidden(context)
        scaled_hidden = scaled.final_hidden(context)
        base_probs = base.predict(context)
        scaled_probs = scaled.predict(context)

        self.assertNotEqual(base_hidden, scaled_hidden)
        self.assertNotEqual(base_probs, scaled_probs)

    def test_final_hidden_matches_forward_logits(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=19,
                use_prompt_position_projection=True,
            )
        )
        context = context_before(ids, 4, model.config.context_size, tokenizer.pad_id)

        hidden = model.final_hidden(context)
        expected_logits = model._forward_floats(context)
        actual_logits = []
        for output_index, bias in enumerate(model.bout):
            total = bias.data
            for input_index, value in enumerate(hidden):
                total += value * model.wout[input_index][output_index].data
            actual_logits.append(total)

        self.assertEqual(len(hidden), model.config.embedding_dim)
        for expected, actual in zip(expected_logits, actual_logits):
            self.assertAlmostEqual(expected, actual)

    def test_prompt_attention_summary_starts_as_baseline_and_round_trips(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        base_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=15,
        )
        summary_config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=15,
            use_prompt_attention_summary=True,
        )
        baseline = TinyTransformerLM.init_random(base_config)
        model = TinyTransformerLM.init_random(summary_config)
        context = context_before(ids, 4, summary_config.context_size, tokenizer.pad_id)
        target = ids[4]

        for expected, actual in zip(baseline.predict(context), model.predict(context)):
            self.assertAlmostEqual(expected, actual)

        before = model.nll(context, target)
        for _ in range(20):
            model.train_step(context, target, learning_rate=0.02)
        after = model.nll(context, target)
        projection_values = [
            value.data
            for value in (
                flatten_scalars(model.prompt_summary_w)
                + flatten_scalars(model.prompt_summary_b)
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "transformer.json"
            model.save(path, tokenizer)
            loaded, _loaded_tokenizer = TinyTransformerLM.load(path)

        weights = loaded.to_dict()["weights"]
        self.assertTrue(loaded.config.use_prompt_attention_summary)
        self.assertIn("prompt_summary_query", weights)
        self.assertIn("prompt_summary_w", weights)
        self.assertIn("prompt_summary_b", weights)
        self.assertTrue(any(abs(value) > 0.0 for value in projection_values))
        self.assertGreater(before, after)
        self.assertAlmostEqual(sum(model.predict(context)), 1.0)

    def test_multi_layer_top_layer_update_freezes_lower_layer(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        ids = tokenizer.encode(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=12,
            num_layers=2,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(ids, 4, config.context_size, tokenizer.pad_id)
        target = ids[4]
        lower_before = model.wq[0][0].data
        top_before = model.extra_blocks[0]["wq"][0][0].data
        head_before = model.wout[0][target].data
        model.freeze_lower_layers_for_updates = True

        for _ in range(20):
            model.train_step(
                context,
                target,
                learning_rate=0.02,
                params=model.top_layer_parameters(),
            )

        self.assertEqual(model.wq[0][0].data, lower_before)
        self.assertTrue(
            model.extra_blocks[0]["wq"][0][0].data != top_before
            or model.wout[0][target].data != head_before
        )

    def test_multi_layer_final_block_matches_full_stack_logits(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=11,
            num_layers=2,
        )
        model = TinyTransformerLM.init_random(config)
        context = context_before(tokenizer.encode(text), 4, config.context_size, tokenizer.pad_id)
        optimized = model._forward_floats(context)
        token_embeddings = [[value.data for value in row] for row in model.token_embeddings]
        position_embeddings = [[value.data for value in row] for row in model.position_embeddings]
        full_stack = model._forward_full_block_floats(
            [
                [
                    token_embeddings[token_id][dim] + position_embeddings[position][dim]
                    for dim in range(config.embedding_dim)
                ]
                for position, token_id in enumerate(context)
            ],
            model._block_to_floats(model.blocks[0]),
        )
        full_stack = model._forward_full_block_floats(
            full_stack,
            model._block_to_floats(model.blocks[1]),
        )
        manual = []
        for output_index, bias in enumerate([value.data for value in model.bout]):
            total = bias
            for input_index, value in enumerate(full_stack[-1]):
                total += value * model.wout[input_index][output_index].data
            manual.append(total)

        self.assertEqual(len(optimized), len(manual))
        for left, right in zip(optimized, manual):
            self.assertAlmostEqual(left, right)

    def test_generate_uses_character_tokenizer(self) -> None:
        text = "abc abc\n"
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=5,
            )
        )

        generated = model.generate(tokenizer, "abc", max_new_chars=3)

        self.assertIsInstance(generated, str)
        self.assertLessEqual(len(generated), 3)

    def test_generate_can_stop_at_admitted_terminator(self) -> None:
        text = "a\n"
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=2,
                embedding_dim=4,
                feedforward_dim=8,
                seed=6,
            )
        )
        newline_id = tokenizer.stoi[ANSWER_TERMINATOR]
        model.bout[newline_id].data = 5.0

        generated = model.generate(
            tokenizer,
            "a",
            max_new_chars=4,
            stop_at=ANSWER_TERMINATOR,
        )

        self.assertEqual(generated, "")


if __name__ == "__main__":
    unittest.main()
