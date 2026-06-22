"""Opt-in torch.compile of the Tier-2 batched forward (real-torch, validated band).

runtime['use_compile'] routes torch_batched_logits through torch.compile (the
device-agnostic backend compiler). It computes the same forward, so on identical
weights it agrees with the eager path within the parity band. Default off returns
the eager function unchanged; the batched path is never reached under the
fake-torch double, so torch.compile (absent there) is never invoked.
"""

from __future__ import annotations

import unittest
from dataclasses import asdict
from importlib import import_module

import support  # noqa: F401  (puts src/ on sys.path)
from neural_char_ops import make_context
from tokenizer import CharTokenizer
from transformer_model import TransformerConfig
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_batched_block import batched_logits_fn, torch_batched_logits

CPU64 = {"dtype": "float64", "device": "cpu"}


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


class TorchCompileBatchedTest(unittest.TestCase):
    def test_compile_off_returns_eager_function(self) -> None:
        torch = _torch_or_skip(self)
        self.assertIs(batched_logits_fn(torch, {}), torch_batched_logits)
        self.assertIs(batched_logits_fn(torch, {"use_compile": False}), torch_batched_logits)

    def test_compiled_matches_eager(self) -> None:
        torch = _torch_or_skip(self)
        tokenizer = CharTokenizer.train("mia ball box red cup noah shelf\n")
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size, context_size=6, embedding_dim=4,
                feedforward_dim=8, attention_heads=2, seed=7,
            )
        )
        fixture = {"weights": model.to_dict()["weights"], "model_config": asdict(model.config)}
        contexts = [
            make_context(tokenizer.encode("mia"), 6, tokenizer.pad_id),
            make_context(tokenizer.encode("noah"), 6, tokenizer.pad_id),
        ]
        eager = batched_logits_fn(torch, {"use_compile": False})(contexts, fixture, torch, CPU64)
        compiled = batched_logits_fn(torch, {"use_compile": True})(contexts, fixture, torch, CPU64)
        self.assertLess(float((eager - compiled).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
