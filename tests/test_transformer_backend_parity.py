from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_char_ops import make_context
from tokenizer import CharTokenizer
from transformer_backend_parity import (
    PARITY_FIXTURE_KIND,
    build_backend_parity_report,
    build_scalar_backend_parity_fixture,
    validate_backend_parity_fixture,
)
from transformer_backend_policy import (
    PYTORCH_BACKEND,
    transformer_backend_metadata,
)
from transformer_model import GenerationConfig, TransformerConfig
from transformer_tiny_lm import TinyTransformerLM


class TransformerBackendParityTests(unittest.TestCase):
    def test_scalar_fixture_records_reference_math_and_generation(self) -> None:
        model, tokenizer, context, target = _fixture_model()

        fixture = build_scalar_backend_parity_fixture(
            fixture_id="tiny-scalar",
            model=model,
            tokenizer=tokenizer,
            contexts=[context],
            targets=[target],
            prompts=["ab"],
            corpus_hash="corpus-hash",
            generation_config=GenerationConfig(trace_top_tokens=3),
            max_new_chars=2,
        )

        validate_backend_parity_fixture(fixture)
        forward_case = fixture["forward_cases"][0]
        generation_case = fixture["generation_cases"][0]
        self.assertEqual(fixture["kind"], PARITY_FIXTURE_KIND)
        self.assertEqual(fixture["reference_backend"]["backend"], "scalar_python")
        self.assertEqual(fixture["reference_backend"]["corpus_hash"], "corpus-hash")
        self.assertIn("token_embeddings", fixture["weights"])
        self.assertEqual(forward_case["context"], context)
        self.assertEqual(forward_case["target"], target)
        self.assertEqual(len(forward_case["logits"]), tokenizer.vocab_size)
        self.assertAlmostEqual(sum(forward_case["probabilities"]), 1.0)
        self.assertAlmostEqual(forward_case["loss"], model.nll(context, target))
        self.assertEqual(generation_case["prompt_ids"], tokenizer.encode("ab"))
        self.assertEqual(len(generation_case["token_ids"]), 2)
        self.assertEqual(generation_case["cache"]["mode"], "disabled")

    def test_report_passes_for_matching_candidate_backend_outputs(self) -> None:
        fixture = _scalar_fixture()
        candidate = _matching_candidate(fixture)

        report = build_backend_parity_report(
            fixture=fixture,
            candidate=candidate,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["candidate_backend"], PYTORCH_BACKEND)
        self.assertEqual(report["summary"]["failed_checks"], [])

    def test_report_fails_when_candidate_logits_or_generation_drift(self) -> None:
        fixture = _scalar_fixture()
        candidate = _matching_candidate(fixture)
        candidate["forward_cases"][0]["logits"][0] += 0.01
        candidate["generation_cases"][0]["text"] = "drift"
        candidate["generation_cases"][0]["cache"]["enabled"] = True

        report = build_backend_parity_report(
            fixture=fixture,
            candidate=candidate,
        )

        failed = set(report["summary"]["failed_checks"])
        self.assertFalse(report["passed"])
        self.assertIn("forward_logits:forward-01", failed)
        self.assertIn("generation_text:generation-01", failed)
        self.assertIn("generation_cache:generation-01", failed)

    def test_report_fails_when_pytorch_metadata_lacks_artifact_fields(self) -> None:
        fixture = _scalar_fixture()
        candidate = _matching_candidate(fixture)
        candidate["backend"] = transformer_backend_metadata(
            active_backend=PYTORCH_BACKEND,
            seed=17,
            tokenizer_type="char",
        )

        report = build_backend_parity_report(
            fixture=fixture,
            candidate=candidate,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["summary"]["failed_checks"], ["backend_metadata"])

    def test_report_requires_consistent_pytorch_runtime_report(self) -> None:
        fixture = _scalar_fixture()
        candidate = _matching_candidate(fixture)
        candidate["runtime_report"]["runtime"] = {"device": "drift"}

        report = build_backend_parity_report(
            fixture=fixture,
            candidate=candidate,
        )

        self.assertFalse(report["passed"])
        self.assertIn("runtime_report", report["summary"]["failed_checks"])

    def test_fixture_validation_requires_scalar_reference_backend(self) -> None:
        fixture = _scalar_fixture()
        fixture["reference_backend"] = transformer_backend_metadata(
            active_backend=PYTORCH_BACKEND,
            seed=17,
            tokenizer_type="char",
            corpus_hash="corpus-hash",
            parity_status="pending",
        )

        with self.assertRaises(ValueError):
            validate_backend_parity_fixture(fixture)

    def test_fixture_builder_rejects_context_target_mismatch(self) -> None:
        model, tokenizer, context, _target = _fixture_model()

        with self.assertRaises(ValueError):
            build_scalar_backend_parity_fixture(
                fixture_id="bad",
                model=model,
                tokenizer=tokenizer,
                contexts=[context],
                targets=[],
                prompts=[],
                corpus_hash="corpus-hash",
            )


def _fixture_model() -> tuple[TinyTransformerLM, CharTokenizer, list[int], int]:
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
    return model, tokenizer, context, tokenizer.stoi["c"]


def _scalar_fixture() -> dict:
    model, tokenizer, context, target = _fixture_model()
    return build_scalar_backend_parity_fixture(
        fixture_id="tiny-scalar",
        model=model,
        tokenizer=tokenizer,
        contexts=[context],
        targets=[target],
        prompts=["ab"],
        corpus_hash="corpus-hash",
        max_new_chars=2,
    )


def _matching_candidate(fixture: dict) -> dict:
    runtime = {"device": "cpu", "dtype": "float32", "runtime_kind": "test_double"}
    return {
        "backend": transformer_backend_metadata(
            active_backend=PYTORCH_BACKEND,
            seed=17,
            tokenizer_type="char",
            corpus_hash="corpus-hash",
            device="cpu",
            dtype="float32",
            parity_status="matched",
        ),
        "runtime": runtime,
        "runtime_report": _runtime_report(runtime, training_allowed=False),
        "forward_cases": [
            {
                "case_id": case["case_id"],
                "logits": list(case["logits"]),
                "loss": case["loss"],
            }
            for case in fixture["forward_cases"]
        ],
        "generation_cases": [
            {
                "case_id": case["case_id"],
                "text": case["text"],
                "token_ids": copy.deepcopy(case["token_ids"]),
                "cache": copy.deepcopy(case["cache"]),
            }
            for case in fixture["generation_cases"]
        ],
    }


def _runtime_report(runtime: dict, *, training_allowed: bool) -> dict:
    return {
        "kind": "transformer_torch_runtime_report",
        "runtime": copy.deepcopy(runtime),
        "status": (
            "ready_for_pytorch_parity"
            if training_allowed
            else "blocked_test_double_runtime"
        ),
        "evidence_scope": "runtime_preflight_only",
        "parity_attempt_allowed": training_allowed,
        "training_evidence_allowed": training_allowed,
        "closed_world_boundary": {
            "runtime_library_allowed": True,
            "learned_assets_imported": False,
            "training_data_imported": False,
            "pretrained_weights_imported": False,
            "pretrained_tokenizer_imported": False,
            "external_embeddings_imported": False,
        },
    }


if __name__ == "__main__":
    unittest.main()
