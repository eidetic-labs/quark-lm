"""Artifact-set validation for PyTorch training parity attempts."""

from __future__ import annotations

from typing import Any

from transformer_training_parity_report import build_training_parity_report
from transformer_training_parity_report_validation import validate_training_parity_report
from transformer_torch_training_parity_attempt_hashes import (
    TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM,
    build_torch_training_parity_attempt_hashes,
)
from transformer_torch_training_parity_attempt_requirements import (
    build_torch_training_parity_attempt_requirements,
)
from transformer_torch_training_parity_attempt_validation import (
    validate_torch_training_parity_attempt,
)
from transformer_torch_training_parity_attempt_summaries import (
    build_torch_attempt_candidate_summary,
    build_torch_attempt_replay_gate_summary,
    build_torch_attempt_report_summary,
    build_torch_attempt_runtime_summary,
)
from transformer_torch_training_candidate_validation import (
    validate_torch_training_parity_candidate,
)
from transformer_torch_training_promotion_gate import (
    build_torch_training_backend_promotion_gate,
)


REQUIRED_TORCH_TRAINING_ATTEMPT_ARTIFACTS = (
    "attempt",
    "fixture",
    "candidate",
    "report",
)


def validate_torch_training_parity_attempt_artifact_set(
    artifacts: dict[str, Any],
    *,
    require_artifact_paths: bool = False,
    require_artifact_hashes: bool = False,
) -> None:
    """Validate that a training parity attempt artifact set is coherent."""

    payloads = _required_payloads(artifacts)
    validate_torch_training_parity_attempt(
        payloads["attempt"],
        require_artifacts=require_artifact_paths,
    )
    _validate_fixture_ids(payloads)
    _validate_corpus_summary(payloads)
    _validate_summary(
        "runtime",
        payloads["attempt"].get("runtime"),
        build_torch_attempt_runtime_summary(
            payloads["candidate"].get("runtime_report", {})
        ),
    )
    _validate_summary(
        "candidate",
        payloads["attempt"].get("candidate"),
        build_torch_attempt_candidate_summary(payloads["candidate"]),
    )
    _validate_summary(
        "training_replay_parity_gate",
        payloads["attempt"].get("training_replay_parity_gate"),
        build_torch_attempt_replay_gate_summary(
            payloads["candidate"].get("training_replay_parity_gate", {})
        ),
    )
    _validate_summary(
        "training_parity_report",
        payloads["attempt"].get("training_parity_report"),
        build_torch_attempt_report_summary(payloads["report"]),
    )
    validate_torch_training_parity_candidate(payloads["candidate"])
    _validate_report_payload(payloads)
    _validate_promotion_gate_payload(payloads)
    _validate_next_requirements_payload(payloads)
    _validate_artifact_hashes(
        payloads,
        require_artifact_hashes=require_artifact_hashes,
    )


def _required_payloads(artifacts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(artifacts, dict):
        raise ValueError("training parity artifacts must be a dict")
    payloads = {}
    for key in REQUIRED_TORCH_TRAINING_ATTEMPT_ARTIFACTS:
        value = artifacts.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"artifacts.{key} must be a dict")
        payloads[key] = value
    return payloads


def _validate_fixture_ids(payloads: dict[str, dict[str, Any]]) -> None:
    fixture_id = payloads["attempt"].get("fixture_id")
    for key in ("fixture", "candidate", "report"):
        if payloads[key].get("fixture_id") != fixture_id:
            raise ValueError(f"artifacts.{key}.fixture_id is inconsistent")


def _validate_corpus_summary(payloads: dict[str, dict[str, Any]]) -> None:
    corpus = payloads["attempt"].get("corpus", {})
    if not isinstance(corpus.get("train_sha256"), str) or not corpus["train_sha256"]:
        raise ValueError("attempt.corpus.train_sha256 is invalid")
    expected_hash = payloads["fixture"].get("reference_backend", {}).get("corpus_hash")
    if corpus["train_sha256"] != expected_hash:
        raise ValueError("attempt.corpus.train_sha256 is inconsistent")
    candidate_hash = payloads["candidate"].get("backend", {}).get("corpus_hash")
    if candidate_hash != expected_hash:
        raise ValueError("artifacts.candidate.backend.corpus_hash is inconsistent")


def _validate_summary(name: str, actual: Any, expected: dict[str, Any]) -> None:
    if actual != expected:
        raise ValueError(f"attempt.{name} summary is inconsistent")


def _validate_report_payload(payloads: dict[str, dict[str, Any]]) -> None:
    validate_training_parity_report(payloads["report"])
    expected = build_training_parity_report(
        fixture=payloads["fixture"],
        candidate=payloads["candidate"],
    )
    if payloads["report"] != expected:
        raise ValueError("artifacts.report payload is inconsistent")


def _validate_promotion_gate_payload(payloads: dict[str, dict[str, Any]]) -> None:
    expected = build_torch_training_backend_promotion_gate(
        candidate=payloads["candidate"],
        report=payloads["report"],
        closed_world_boundary=payloads["attempt"]["closed_world_boundary"],
    )
    if payloads["attempt"]["training_backend_promotion_gate"] != expected:
        raise ValueError("attempt.training_backend_promotion_gate is inconsistent")


def _validate_next_requirements_payload(payloads: dict[str, dict[str, Any]]) -> None:
    expected = build_torch_training_parity_attempt_requirements(
        runtime_report=payloads["candidate"].get("runtime_report", {}),
        candidate=payloads["candidate"],
        report=payloads["report"],
    )
    if payloads["attempt"]["next_requirements"] != expected:
        raise ValueError("attempt.next_requirements is inconsistent")


def _validate_artifact_hashes(
    payloads: dict[str, dict[str, Any]],
    *,
    require_artifact_hashes: bool,
) -> None:
    attempt = payloads["attempt"]
    has_hashes = (
        "artifact_hash_algorithm" in attempt or "artifact_hashes" in attempt
    )
    if not require_artifact_hashes and not has_hashes:
        return
    if attempt.get("artifact_hash_algorithm") != TORCH_TRAINING_ATTEMPT_HASH_ALGORITHM:
        raise ValueError("attempt.artifact_hash_algorithm is inconsistent")
    expected = build_torch_training_parity_attempt_hashes(payloads)
    if attempt.get("artifact_hashes") != expected:
        raise ValueError("attempt.artifact_hashes is inconsistent")
