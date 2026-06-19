"""Build optional PyTorch training parity attempt artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from corpus_artifacts import SCHEMA_VERSION
from curriculum import DEFAULT_CORPUS_DIR, PROJECT_DIR, build_curriculum
from neural_char_ops import context_before
from tokenizer import CharTokenizer
from transformer_model import OptimizationConfig, TransformerConfig
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_runtime import TorchImporter
from transformer_torch_training_attempt_boundary import (
    build_torch_training_attempt_boundary,
)
from transformer_torch_training_candidate import build_torch_training_parity_candidate
from transformer_torch_training_parity_attempt_requirements import (
    build_torch_training_parity_attempt_requirements,
)
from transformer_torch_training_parity_attempt_validation import (
    validate_torch_training_parity_attempt,
)
from transformer_torch_training_parity_attempt_writer import (
    write_torch_training_parity_attempt,
)
from transformer_torch_training_promotion_gate import (
    build_torch_training_backend_promotion_gate,
)
from transformer_training_parity import (
    build_scalar_training_parity_fixture,
    build_training_parity_report,
)


DEFAULT_OUTPUT_DIR = PROJECT_DIR / "build" / "torch_training_parity_attempt"
TORCH_TRAINING_PARITY_ATTEMPT_KIND = "transformer_torch_training_parity_attempt"


def build_torch_training_parity_attempt(
    *,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    fixture_id: str = "admitted-curriculum-training-parity",
    seed: int = 53,
    context_index: int = 4,
    context_size: int = 4,
    embedding_dim: int = 4,
    feedforward_dim: int = 8,
    learning_rate: float = 0.02,
    steps: int = 2,
    gradient_accumulation_steps: int = 2,
    requested_device: str = "cpu",
    requested_dtype: str = "float64",
    importer: TorchImporter | None = None,
) -> dict[str, Any]:
    """Build scalar fixture, PyTorch candidate, report, and attempt summary."""

    curriculum = build_curriculum(corpus_dir=corpus_dir, seed=seed)
    tokenizer = CharTokenizer.train(curriculum.train_text)
    ids = tokenizer.encode(curriculum.train_text)
    context, target = _context_and_target(
        ids=ids,
        context_index=context_index,
        context_size=context_size,
        pad_id=tokenizer.pad_id,
    )
    model = TinyTransformerLM.init_random(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=context_size,
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim,
            seed=seed,
        )
    )
    fixture = build_scalar_training_parity_fixture(
        fixture_id=fixture_id,
        model=model,
        tokenizer=tokenizer,
        context=context,
        target=target,
        optimizer_config=OptimizationConfig(
            optimizer="adamw",
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=steps,
            decay_steps=steps,
            min_learning_rate=0.001,
        ),
        learning_rate=learning_rate,
        steps=steps,
        corpus_hash=_text_hash(curriculum.train_text),
    )
    candidate_kwargs: dict[str, Any] = {
        "fixture": fixture,
        "requested_device": requested_device,
        "requested_dtype": requested_dtype,
    }
    if importer is not None:
        candidate_kwargs["importer"] = importer
    candidate = build_torch_training_parity_candidate(**candidate_kwargs)
    report = build_training_parity_report(fixture=fixture, candidate=candidate)
    attempt = _attempt_summary(
        corpus_dir=corpus_dir,
        curriculum_manifest=curriculum.manifest,
        train_text=curriculum.train_text,
        fixture=fixture,
        candidate=candidate,
        report=report,
    )
    validate_torch_training_parity_attempt(attempt)
    return {
        "attempt": attempt,
        "fixture": fixture,
        "candidate": candidate,
        "report": report,
    }


def _context_and_target(
    *,
    ids: list[int],
    context_index: int,
    context_size: int,
    pad_id: int,
) -> tuple[list[int], int]:
    if not ids:
        raise ValueError("training curriculum must not be empty")
    if context_index < 0 or context_index >= len(ids):
        raise ValueError("context_index must point inside the training curriculum")
    return context_before(ids, context_index, context_size, pad_id), ids[context_index]


def _attempt_summary(
    *,
    corpus_dir: Path,
    curriculum_manifest: dict[str, Any],
    train_text: str,
    fixture: dict[str, Any],
    candidate: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    runtime_report = candidate.get("runtime_report", {})
    gate = candidate.get("training_replay_parity_gate", {})
    closed_world_boundary = build_torch_training_attempt_boundary()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": TORCH_TRAINING_PARITY_ATTEMPT_KIND,
        "fixture_id": fixture["fixture_id"],
        "status": _attempt_status(runtime_report, gate, report),
        "passed": bool(report.get("passed")),
        "promoted_training_backend": False,
        "evidence_scope": "training_parity_attempt_only",
        "corpus": _corpus_summary(corpus_dir, curriculum_manifest, train_text),
        "runtime": _runtime_summary(runtime_report),
        "candidate": _candidate_summary(candidate),
        "training_replay_parity_gate": _gate_summary(gate),
        "training_parity_report": _report_summary(report),
        "training_backend_promotion_gate": build_torch_training_backend_promotion_gate(
            candidate=candidate,
            report=report,
            closed_world_boundary=closed_world_boundary,
        ),
        "next_requirements": build_torch_training_parity_attempt_requirements(
            runtime_report=runtime_report,
            candidate=candidate,
            report=report,
        ),
        "closed_world_boundary": closed_world_boundary,
    }


def _attempt_status(
    runtime_report: dict[str, Any],
    gate: dict[str, Any],
    report: dict[str, Any],
) -> str:
    if report.get("passed") is True:
        return "training_parity_matched"
    if runtime_report.get("parity_attempt_allowed") is not True:
        return str(runtime_report.get("status", "blocked_pytorch_runtime"))
    return str(gate.get("status", "training_parity_pending"))


def _corpus_summary(
    corpus_dir: Path,
    manifest: dict[str, Any],
    train_text: str,
) -> dict[str, Any]:
    return {
        "corpus_dir": str(corpus_dir),
        "train_sha256": _text_hash(train_text),
        "train_chars": len(train_text),
        "manifest": dict(manifest),
    }


def _runtime_summary(runtime_report: dict[str, Any]) -> dict[str, Any]:
    runtime = runtime_report.get("runtime", {})
    return {
        "status": runtime_report.get("status"),
        "passed": runtime_report.get("passed"),
        "parity_attempt_allowed": runtime_report.get("parity_attempt_allowed"),
        "runtime_kind": runtime.get("runtime_kind"),
        "device": runtime.get("device"),
        "dtype": runtime.get("dtype"),
    }


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    backend = candidate.get("backend", {})
    return {
        "implementation_status": candidate.get("implementation_status"),
        "parity_status": backend.get("parity_status"),
        "training_readiness_status": candidate.get("training_readiness", {}).get(
            "status"
        ),
        "training_case_status": candidate.get("training_case", {}).get("status"),
    }


def _gate_summary(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": gate.get("status"),
        "passed": gate.get("passed"),
        "failed_checks": gate.get("summary", {}).get("failed_checks", []),
    }


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": report.get("passed"),
        "failed_checks": report.get("summary", {}).get("failed_checks", []),
    }


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
