"""Tier-2 batched (B,C,D) torch forward parity (real-torch-only, validated band).

torch_batched_logits(contexts)[b] must match per-position torch_minimal_logits on
the SAME weights within the dtype parity band (float64 1e-6 / float32 3e-3), rank
candidates identically (zero-tolerance), respect the leading-pad mask, cover every
batched-supported profile flag, handle an odd head_dim RoPE case, and leak nothing
across duplicate batch rows. use_prompt_position_projection is fail-closed (Tier-1).
"""

from __future__ import annotations

import random
import unittest
from dataclasses import asdict
from importlib import import_module

import support  # noqa: F401  (puts src/ on sys.path)
from support.core import CharTokenizer, TransformerConfig
from neural_char_ops import make_context
from transformer_parity_contract import assert_numeric_parity, assert_rank_invariant
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_batched_block import torch_batched_logits
from transformer_torch_minimal_block import torch_minimal_logits
from transformer_torch_profile_support import batched_forward_unsupported_reason

PROBES = [
    ("mia ball", [" box", " cup", " red", " unknown"]),
    ("noah cup", [" red", " box", " unknown", " cup"]),
    ("ava book", [" unknown", " shelf", " box", " red"]),
    ("leo", [" box", " unknown", " cup"]),
    ("mia", [" red", " cup", " box", " unknown"]),
    ("the red ball is", [" box", " here", " unknown", " gone"]),
    ("noah", [" cup", " box", " unknown", " red"]),
    ("ball box red cup", [" unknown", " mia", " noah"]),
    ("ava ball cup", [" box", " unknown", " red", " shelf"]),
    ("leo book shelf", [" red", " box", " unknown", " here"]),
]
ALL_TEXT = "".join(p + "".join(c) for p, c in PROBES) + "\n"
CPU64 = {"dtype": "float64", "device": "cpu"}
CPU32 = {"dtype": "float32", "device": "cpu"}


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


def _forward_fixture(model: TinyTransformerLM) -> dict:
    return {"weights": model.to_dict()["weights"], "model_config": asdict(model.config)}


def _fill(value, rand):
    if isinstance(value, list):
        return [_fill(item, rand) for item in value]
    return rand.uniform(-0.3, 0.3)


def _model(tokenizer, **overrides) -> TinyTransformerLM:
    defaults = dict(
        vocab_size=tokenizer.vocab_size, context_size=6, embedding_dim=4,
        feedforward_dim=8, attention_heads=2, seed=7,
    )
    defaults.update(overrides)
    return TinyTransformerLM.init_random(TransformerConfig(**defaults))


def _randomize_summary_weights(fixture: dict, seed: int) -> None:
    """Give the active summary projection weights non-zero values (init is zero)."""

    rand = random.Random(seed)
    weights = fixture["weights"]
    for key in (
        "context_projection_w", "context_projection_b",
        "prompt_prefix_projection_w", "prompt_prefix_projection_b",
        "prompt_summary_query", "prompt_summary_w", "prompt_summary_b",
    ):
        weights[key] = _fill(weights[key], rand)


def _contexts(tokenizer, context_size):
    out = []
    for prompt, _ in PROBES:
        ids = tokenizer.encode(prompt)
        out.append(make_context(ids, context_size, tokenizer.pad_id))
    return out


def _assert_batched_matches_per_position(test, torch, model, contexts, runtime):
    fixture = _forward_fixture(model)
    batched = torch_batched_logits(contexts, fixture, torch, runtime)
    for index, context in enumerate(contexts):
        single = torch_minimal_logits(context, fixture, torch, runtime)
        assert_numeric_parity(
            [float(v) for v in single.detach().cpu().tolist()],
            [float(v) for v in batched[index].detach().cpu().tolist()],
            dtype=runtime["dtype"],
        )


class BatchedForwardParityTest(unittest.TestCase):
    def test_batched_matches_per_position_distinct_and_duplicate_rows(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(ALL_TEXT)
        model = _model(tokenizer)
        contexts = _contexts(tokenizer, model.config.context_size)
        # Duplicate rows prove no cross-batch leakage.
        contexts = contexts + [contexts[0], contexts[0]]
        for runtime in (CPU64, CPU32):
            _assert_batched_matches_per_position(self, torch, model, contexts, runtime)

    def test_rank_invariance_over_probes(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(ALL_TEXT)
        model = _model(tokenizer)
        fixture = _forward_fixture(model)
        contexts = _contexts(tokenizer, model.config.context_size)
        for runtime in (CPU64, CPU32):
            batched = torch_batched_logits(contexts, fixture, torch, runtime)
            for index, context in enumerate(contexts):
                single = torch_minimal_logits(context, fixture, torch, runtime)
                assert_rank_invariant(
                    [float(v) for v in single.detach().cpu().tolist()],
                    [float(v) for v in batched[index].detach().cpu().tolist()],
                )

    def test_leading_pad_mask_matches(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(ALL_TEXT)
        model = _model(tokenizer, use_prompt_prefix_projection=True)
        fixture = _forward_fixture(model)
        _randomize_summary_weights(fixture, seed=21)
        size = model.config.context_size
        # Mixed leading-pad lengths in one batch (pad_id == 0).
        contexts = [
            make_context(tokenizer.encode("mia"), size, 0),
            make_context(tokenizer.encode("noah cup"), size, 0),
            make_context(tokenizer.encode("ava ball box"), size, 0),
        ]
        for runtime in (CPU64, CPU32):
            batched = torch_batched_logits(contexts, fixture, torch, runtime)
            for index, context in enumerate(contexts):
                single = torch_minimal_logits(context, fixture, torch, runtime)
                assert_numeric_parity(
                    [float(v) for v in single.detach().cpu().tolist()],
                    [float(v) for v in batched[index].detach().cpu().tolist()],
                    dtype=runtime["dtype"],
                )

    def test_profile_coverage(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(ALL_TEXT)
        profiles = [
            {"use_rotary_positions": True},
            {"use_pre_layer_norm": True},
            {"use_pre_layer_norm": True, "use_rms_norm": True},
            {"use_gated_mlp": True},
            {"tie_output_embeddings": True},
            {"use_context_mean": True},
            {"use_context_projection": True},
            {"use_prompt_prefix_projection": True},
            {"use_prompt_attention_summary": True},
            {"use_layer_norm": True},
            # Mixed combo across orthogonal axes.
            {
                "use_rotary_positions": True, "use_pre_layer_norm": True,
                "use_rms_norm": True, "use_gated_mlp": True,
                "use_prompt_attention_summary": True,
            },
        ]
        for seed, overrides in enumerate(profiles):
            model = _model(tokenizer, **overrides)
            fixture = _forward_fixture(model)
            _randomize_summary_weights(fixture, seed=100 + seed)
            contexts = _contexts(tokenizer, model.config.context_size)[:4]
            for runtime in (CPU64, CPU32):
                batched = torch_batched_logits(contexts, fixture, torch, runtime)
                for index, context in enumerate(contexts):
                    single = torch_minimal_logits(context, fixture, torch, runtime)
                    assert_numeric_parity(
                        [float(v) for v in single.detach().cpu().tolist()],
                        [float(v) for v in batched[index].detach().cpu().tolist()],
                        dtype=runtime["dtype"],
                    )

    def test_odd_head_dim_rotary(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train(ALL_TEXT)
        # embedding_dim 6 / 2 heads -> head_dim 3 (odd); the tail dim is untouched.
        model = _model(
            tokenizer, embedding_dim=6, attention_heads=2, feedforward_dim=8,
            use_rotary_positions=True,
        )
        self.assertEqual(model.config.embedding_dim // model.config.attention_heads, 3)
        contexts = _contexts(tokenizer, model.config.context_size)[:3]
        for runtime in (CPU64, CPU32):
            _assert_batched_matches_per_position(self, torch, model, contexts, runtime)

    def test_prompt_position_projection_is_fail_closed(self) -> None:
        _torch_or_skip(self)
        reason = batched_forward_unsupported_reason(
            {"use_prompt_position_projection": True}
        )
        self.assertIsNotNone(reason)
        self.assertIsNone(batched_forward_unsupported_reason({}))


if __name__ == "__main__":
    unittest.main()
