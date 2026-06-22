"""Device/dtype policy for the torch runtime + the CUDA-host validation leg.

QuarkLM is meant to run on GPU, NPU, or CPU. The torch paths are device-agnostic
(tensors created on ``runtime['device']``; fused AdamW guarded to CUDA; SDPA dispatches
to FlashAttention on CUDA), but two things must hold for that to be trustworthy:

  1. configure_torch_runtime FAILS LOUD on a silent-downcast combo. MPS (Apple GPUs)
     has no float64 -- requesting it would downcast to float32 and void the float64
     1e-6 parity guarantee, looking green while lying. The guard rejects it.
  2. CUDA is a VALIDATED path, not merely compiled code. The parity leg below runs the
     batched forward at float32 on real CUDA and asserts it ranks identically to the
     CPU float64 reference. It SKIPS where CUDA is absent (e.g. this Apple host), so it
     turns green only on a CUDA host -- making "supports CUDA" a checked claim there.
"""

from __future__ import annotations

import unittest
from dataclasses import asdict
from importlib import import_module

import support  # noqa: F401  (puts src/ on sys.path)
from neural_char_ops import make_context
from tokenizer import CharTokenizer
from transformer_model import TransformerConfig
from transformer_parity_contract import assert_rank_invariant
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_runtime import configure_torch_runtime

CORPUS = "mia ball box red cup noah shelf\n"


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


class DeviceDtypePolicyTest(unittest.TestCase):
    def test_mps_float64_is_rejected(self) -> None:
        torch = _torch_or_skip(self)
        with self.assertRaises(ValueError):
            configure_torch_runtime(torch, {"dtype": "float64", "device": "mps"})
        # Qualified device string is canonicalized before the lookup.
        with self.assertRaises(ValueError):
            configure_torch_runtime(torch, {"dtype": "float64", "device": "mps:0"})

    def test_mps_float32_is_allowed(self) -> None:
        torch = _torch_or_skip(self)
        configure_torch_runtime(torch, {"dtype": "float32", "device": "mps"})

    def test_cpu_float64_is_allowed(self) -> None:
        torch = _torch_or_skip(self)
        configure_torch_runtime(torch, {"dtype": "float64", "device": "cpu"})

    def test_cuda_permits_both_dtypes(self) -> None:
        torch = _torch_or_skip(self)
        # Policy is string-level (torch.device('cuda') is constructible without a GPU),
        # so this checks the documented CUDA dtype policy, not hardware presence.
        configure_torch_runtime(torch, {"dtype": "float32", "device": "cuda"})
        configure_torch_runtime(torch, {"dtype": "float64", "device": "cuda"})


class CudaParityLegTest(unittest.TestCase):
    def test_cuda_float32_ranks_like_cpu_float64(self) -> None:
        torch = _torch_or_skip(self)
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            self.skipTest("CUDA not available on this host")
        from transformer_torch_batched_block import torch_batched_logits

        tokenizer = CharTokenizer.train(CORPUS)
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
        cpu = {"dtype": "float64", "device": "cpu"}
        gpu = {"dtype": "float32", "device": "cuda"}
        configure_torch_runtime(torch, cpu)
        configure_torch_runtime(torch, gpu)
        reference = torch_batched_logits(contexts, fixture, torch, cpu)
        on_cuda = torch_batched_logits(contexts, fixture, torch, gpu)
        for index in range(len(contexts)):
            assert_rank_invariant(
                [float(v) for v in reference[index].detach().cpu().tolist()],
                [float(v) for v in on_cuda[index].detach().cpu().tolist()],
            )


if __name__ == "__main__":
    unittest.main()
