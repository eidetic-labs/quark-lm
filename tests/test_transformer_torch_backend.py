from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_char_ops import make_context
from tokenizer import CharTokenizer
from transformer_backend_parity import build_backend_parity_report
from transformer_backend_parity_fixture import build_scalar_backend_parity_fixture
from transformer_model import TransformerConfig
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_backend import (
    TORCH_PARITY_CANDIDATE_KIND,
    build_torch_backend_parity_candidate,
    torch_runtime_status,
)


class TransformerTorchBackendTests(unittest.TestCase):
    def test_runtime_status_reports_unavailable_without_hard_dependency(self) -> None:
        status = torch_runtime_status(importer=_missing_importer)

        self.assertFalse(status["available"])
        self.assertEqual(status["backend"], "pytorch")
        self.assertEqual(status["device"], "cpu")
        self.assertIn("ModuleNotFoundError", status["error"])

    def test_runtime_status_uses_fake_torch_device_capabilities(self) -> None:
        status = torch_runtime_status(
            importer=_fake_torch_importer(cuda=True, mps=True),
            requested_device="auto",
            requested_dtype="float32",
        )

        self.assertTrue(status["available"])
        self.assertEqual(status["version"], "fake-torch")
        self.assertEqual(status["device"], "cuda")
        self.assertEqual(status["available_devices"], ["cpu", "cuda", "mps"])
        self.assertTrue(status["dtype_available"])

    def test_runtime_status_falls_back_to_cpu_for_unavailable_device(self) -> None:
        status = torch_runtime_status(
            importer=_fake_torch_importer(cuda=False, mps=False),
            requested_device="mps",
        )

        self.assertTrue(status["available"])
        self.assertEqual(status["device"], "cpu")

    def test_torch_candidate_marks_unavailable_runtime_as_failed(self) -> None:
        fixture = _scalar_fixture()

        candidate = build_torch_backend_parity_candidate(
            fixture=fixture,
            importer=_missing_importer,
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(candidate["kind"], TORCH_PARITY_CANDIDATE_KIND)
        self.assertFalse(candidate["runtime"]["available"])
        self.assertEqual(candidate["backend"]["parity_status"], "failed")
        self.assertFalse(report["passed"])
        self.assertNotIn("backend_metadata", report["summary"]["failed_checks"])

    def test_torch_candidate_matches_scalar_fixture_for_minimal_profile(self) -> None:
        fixture = _scalar_fixture()

        candidate = build_torch_backend_parity_candidate(
            fixture=fixture,
            importer=_fake_torch_importer(),
            requested_device="cpu",
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertTrue(candidate["runtime"]["available"])
        self.assertEqual(candidate["backend"]["parity_status"], "matched")
        self.assertEqual(candidate["implementation_status"], "minimal_forward")
        self.assertEqual(candidate["forward_cases"][0]["status"], "computed")
        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"]["failed_checks"], [])

    def test_torch_candidate_reports_unsupported_profile_without_drifting(self) -> None:
        fixture = _scalar_fixture(use_layer_norm=True)

        candidate = build_torch_backend_parity_candidate(
            fixture=fixture,
            importer=_fake_torch_importer(),
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(candidate["implementation_status"], "unsupported_profile")
        self.assertEqual(candidate["backend"]["parity_status"], "pending")
        self.assertEqual(candidate["forward_cases"][0]["status"], "pending")
        self.assertFalse(report["passed"])

    def test_torch_candidate_reports_unavailable_dtype_without_crashing(self) -> None:
        fixture = _scalar_fixture()

        candidate = build_torch_backend_parity_candidate(
            fixture=fixture,
            importer=_fake_torch_importer(),
            requested_dtype="bfloat16",
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(candidate["implementation_status"], "dtype_unavailable")
        self.assertEqual(candidate["backend"]["parity_status"], "pending")
        self.assertFalse(candidate["runtime"]["dtype_available"])
        self.assertFalse(report["passed"])


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)


def _scalar_fixture(*, use_layer_norm: bool = False) -> dict:
    tokenizer = CharTokenizer.train("abc ")
    model = TinyTransformerLM.init_random(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=17,
            use_layer_norm=use_layer_norm,
        )
    )
    context = make_context(tokenizer.encode("ab"), 4, tokenizer.pad_id)
    return build_scalar_backend_parity_fixture(
        fixture_id="tiny-scalar",
        model=model,
        tokenizer=tokenizer,
        contexts=[context],
        targets=[tokenizer.stoi["c"]],
        prompts=["ab"],
        corpus_hash="corpus-hash",
        max_new_chars=2,
    )


def _fake_torch_importer(
    *,
    cuda: bool = False,
    mps: bool = False,
) -> object:
    fake = types.SimpleNamespace(
        __version__="fake-torch",
        float32="float32",
        float64="float64",
        tensor=lambda value, dtype=None, device=None: FakeTensor(value),
        stack=lambda values: FakeTensor([_raw(value) for value in values]),
        tanh=lambda value: FakeTensor(_map_unary(_raw(value), math.tanh)),
        softmax=lambda value, dim=0: FakeTensor(_softmax(_raw(value))),
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps),
        ),
    )

    def importer(name: str) -> object:
        if name != "torch":
            raise ModuleNotFoundError(name)
        return fake

    return importer


class FakeTensor:
    def __init__(self, value: object) -> None:
        self.value = _copy_raw(value)

    def __iter__(self):
        return (FakeTensor(item) for item in self.value)

    def __getitem__(self, key: int | slice):
        return FakeTensor(self.value[key])

    def __add__(self, other: object):
        return FakeTensor(_binary(_raw(self), _raw(other), lambda left, right: left + right))

    def __radd__(self, other: object):
        return self + other

    def __mul__(self, other: object):
        return FakeTensor(_binary(_raw(self), _raw(other), lambda left, right: left * right))

    def __rmul__(self, other: object):
        return self * other

    def __matmul__(self, other: object):
        vector = _raw(self)
        matrix = _raw(other)
        return FakeTensor(
            [
                sum(vector[row] * matrix[row][column] for row in range(len(vector)))
                for column in range(len(matrix[0]))
            ]
        )

    def sum(self):
        return FakeTensor(sum(_raw(self)))

    def tolist(self):
        return _copy_raw(self.value)


def _raw(value: object) -> object:
    return value.value if isinstance(value, FakeTensor) else value


def _copy_raw(value: object) -> object:
    value = _raw(value)
    if isinstance(value, list):
        return [_copy_raw(item) for item in value]
    return float(value) if isinstance(value, int | float) else value


def _binary(left: object, right: object, op):
    if isinstance(left, list) and isinstance(right, list):
        return [_binary(left_item, right_item, op) for left_item, right_item in zip(left, right)]
    if isinstance(left, list):
        return [_binary(item, right, op) for item in left]
    if isinstance(right, list):
        return [_binary(left, item, op) for item in right]
    return op(float(left), float(right))


def _map_unary(value: object, op):
    if isinstance(value, list):
        return [_map_unary(item, op) for item in value]
    return op(float(value))


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


if __name__ == "__main__":
    unittest.main()
