from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_char_ops import make_context
from support.fake_torch import fake_torch_importer
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
            importer=fake_torch_importer(cuda=True, mps=True),
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
            importer=fake_torch_importer(cuda=False, mps=False),
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
            importer=fake_torch_importer(),
            requested_device="cpu",
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertTrue(candidate["runtime"]["available"])
        self.assertEqual(candidate["backend"]["parity_status"], "matched")
        self.assertEqual(candidate["implementation_status"], "minimal_forward")
        self.assertEqual(candidate["forward_cases"][0]["status"], "computed")
        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"]["failed_checks"], [])

    def test_torch_candidate_matches_scalar_fixture_for_layer_norm_profile(self) -> None:
        fixture = _scalar_fixture(use_layer_norm=True)

        candidate = build_torch_backend_parity_candidate(
            fixture=fixture,
            importer=fake_torch_importer(),
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(candidate["backend"]["parity_status"], "matched")
        self.assertEqual(candidate["implementation_status"], "minimal_forward")
        self.assertTrue(report["passed"])

    def test_torch_candidate_matches_scalar_fixture_for_pre_layer_norm_profile(self) -> None:
        fixture = _scalar_fixture(use_pre_layer_norm=True)

        candidate = build_torch_backend_parity_candidate(
            fixture=fixture,
            importer=fake_torch_importer(),
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(candidate["backend"]["parity_status"], "matched")
        self.assertTrue(report["passed"])

    def test_torch_candidate_matches_scalar_fixture_for_pre_rms_norm_profile(self) -> None:
        fixture = _scalar_fixture(use_pre_layer_norm=True, use_rms_norm=True)

        candidate = build_torch_backend_parity_candidate(
            fixture=fixture,
            importer=fake_torch_importer(),
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(candidate["backend"]["parity_status"], "matched")
        self.assertTrue(report["passed"])

    def test_torch_candidate_matches_scalar_fixture_for_gated_mlp_profile(self) -> None:
        fixture = _scalar_fixture(use_gated_mlp=True)

        candidate = build_torch_backend_parity_candidate(
            fixture=fixture,
            importer=fake_torch_importer(),
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(candidate["backend"]["parity_status"], "matched")
        self.assertTrue(report["passed"])

    def test_torch_candidate_matches_scalar_fixture_for_multi_head_profile(self) -> None:
        fixture = _scalar_fixture(attention_heads=2)

        candidate = build_torch_backend_parity_candidate(
            fixture=fixture,
            importer=fake_torch_importer(),
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(candidate["backend"]["parity_status"], "matched")
        self.assertTrue(report["passed"])

    def test_torch_candidate_matches_scalar_fixture_for_rotary_profile(self) -> None:
        fixture = _scalar_fixture(use_rotary_positions=True, attention_heads=2)

        candidate = build_torch_backend_parity_candidate(
            fixture=fixture,
            importer=fake_torch_importer(),
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(candidate["backend"]["parity_status"], "matched")
        self.assertTrue(report["passed"])

    def test_torch_candidate_reports_unsupported_profile_without_drifting(self) -> None:
        fixture = _scalar_fixture(num_layers=2)

        candidate = build_torch_backend_parity_candidate(
            fixture=fixture,
            importer=fake_torch_importer(),
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
            importer=fake_torch_importer(),
            requested_dtype="bfloat16",
        )
        report = build_backend_parity_report(fixture=fixture, candidate=candidate)

        self.assertEqual(candidate["implementation_status"], "dtype_unavailable")
        self.assertEqual(candidate["backend"]["parity_status"], "pending")
        self.assertFalse(candidate["runtime"]["dtype_available"])
        self.assertFalse(report["passed"])


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)


def _scalar_fixture(
    *,
    use_layer_norm: bool = False,
    use_pre_layer_norm: bool = False,
    use_rms_norm: bool = False,
    use_gated_mlp: bool = False,
    use_rotary_positions: bool = False,
    attention_heads: int = 1,
    num_layers: int = 1,
) -> dict:
    tokenizer = CharTokenizer.train("abc ")
    model = TinyTransformerLM.init_random(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=4,
            embedding_dim=4,
            attention_heads=attention_heads,
            num_layers=num_layers,
            feedforward_dim=8,
            seed=17,
            use_layer_norm=use_layer_norm,
            use_pre_layer_norm=use_pre_layer_norm,
            use_rms_norm=use_rms_norm,
            use_gated_mlp=use_gated_mlp,
            use_rotary_positions=use_rotary_positions,
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


if __name__ == "__main__":
    unittest.main()
