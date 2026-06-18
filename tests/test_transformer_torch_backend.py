from __future__ import annotations

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

    def test_torch_candidate_is_pending_until_backend_math_exists(self) -> None:
        fixture = _scalar_fixture()

        candidate = build_torch_backend_parity_candidate(
            fixture=fixture,
            importer=_fake_torch_importer(),
            requested_device="cpu",
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertTrue(candidate["runtime"]["available"])
        self.assertEqual(candidate["backend"]["parity_status"], "pending")
        self.assertEqual(candidate["implementation_status"], "skeleton")
        self.assertEqual(candidate["forward_cases"][0]["status"], "pending")
        self.assertFalse(report["passed"])
        self.assertIn(
            "forward_logits:forward-01",
            report["summary"]["failed_checks"],
        )


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)


def _scalar_fixture() -> dict:
    tokenizer = CharTokenizer.train("abc ")
    model = TinyTransformerLM.init_random(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            feedforward_dim=8,
            seed=17,
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
        float32=object(),
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


if __name__ == "__main__":
    unittest.main()
