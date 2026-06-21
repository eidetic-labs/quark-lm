"""Determinism floor: torch runtime is validated, TF32-off, optionally seeded."""

from __future__ import annotations

import unittest
from importlib import import_module

import support  # noqa: F401  (inserts src/ onto sys.path)
from transformer_torch_runtime import configure_torch_runtime


def _torch_or_skip(test_case: unittest.TestCase):
    try:
        return import_module("torch")
    except ModuleNotFoundError:
        test_case.skipTest("optional PyTorch runtime is not installed")
        return None


class TorchRuntimeTest(unittest.TestCase):
    def test_valid_runtime_disables_tf32_and_seeds(self) -> None:
        torch = _torch_or_skip(self)
        configure_torch_runtime(torch, {"dtype": "float64", "device": "cpu"}, seed=17)
        matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
        if matmul is not None and hasattr(matmul, "allow_tf32"):
            self.assertFalse(matmul.allow_tf32)
        # seeding makes torch RNG reproducible
        configure_torch_runtime(torch, {"dtype": "float32", "device": "cpu"}, seed=3)
        a = torch.rand(4)
        configure_torch_runtime(torch, {"dtype": "float32", "device": "cpu"}, seed=3)
        b = torch.rand(4)
        self.assertTrue(torch.equal(a, b))

    def test_rejects_bad_dtype_and_device(self) -> None:
        torch = _torch_or_skip(self)
        with self.assertRaises(ValueError):
            configure_torch_runtime(torch, {"dtype": "float8", "device": "cpu"})
        with self.assertRaises(ValueError):
            configure_torch_runtime(torch, {"dtype": "float64", "device": "nonsense-device"})


if __name__ == "__main__":
    unittest.main()
