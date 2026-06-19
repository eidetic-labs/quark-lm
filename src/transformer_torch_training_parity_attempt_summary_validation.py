"""Compact summary validation for PyTorch training parity attempts."""

from __future__ import annotations

from typing import Any


def validate_torch_training_parity_attempt_summaries(
    attempt: dict[str, Any],
) -> None:
    """Validate JSON-safe summary sections before trusting attempt evidence."""

    _validate_corpus_summary(attempt["corpus"])
    _validate_runtime_summary(attempt["runtime"])
    _validate_candidate_summary(attempt["candidate"])
    _validate_gate_summary(
        "training_replay_parity_gate",
        attempt["training_replay_parity_gate"],
    )
    _validate_report_summary(attempt["training_parity_report"])


def _validate_corpus_summary(corpus: dict[str, Any]) -> None:
    _require_non_empty_string(corpus, "corpus", "corpus_dir")
    train_hash = corpus.get("train_sha256")
    if not isinstance(train_hash, str) or len(train_hash) != 64:
        raise ValueError("corpus.train_sha256 is invalid")
    if any(char not in "0123456789abcdef" for char in train_hash):
        raise ValueError("corpus.train_sha256 is invalid")
    train_chars = corpus.get("train_chars")
    if not isinstance(train_chars, int) or train_chars <= 0:
        raise ValueError("corpus.train_chars is invalid")
    if not isinstance(corpus.get("manifest"), dict):
        raise ValueError("corpus.manifest is invalid")


def _validate_runtime_summary(runtime: dict[str, Any]) -> None:
    _require_non_empty_string(runtime, "runtime", "status")
    _require_bool(runtime, "runtime", "passed")
    _require_bool(runtime, "runtime", "parity_attempt_allowed")
    for key in ("runtime_kind", "device", "dtype"):
        _require_non_empty_string(runtime, "runtime", key)
    _require_sha256(runtime, "runtime", "runtime_report_sha256")


def _validate_candidate_summary(candidate: dict[str, Any]) -> None:
    for key in (
        "implementation_status",
        "parity_status",
        "training_readiness_status",
        "training_case_status",
    ):
        _require_non_empty_string(candidate, "candidate", key)
    _require_sha256(candidate, "candidate", "candidate_sha256")


def _validate_gate_summary(name: str, gate: dict[str, Any]) -> None:
    _require_non_empty_string(gate, name, "status")
    _require_bool(gate, name, "passed")
    _require_string_list(gate, name, "failed_checks")
    _require_sha256(gate, name, "training_replay_parity_gate_sha256")


def _validate_report_summary(report: dict[str, Any]) -> None:
    _require_bool(report, "training_parity_report", "passed")
    _require_string_list(report, "training_parity_report", "failed_checks")
    _require_sha256(
        report,
        "training_parity_report",
        "training_parity_report_sha256",
    )


def _require_bool(record: dict[str, Any], label: str, key: str) -> None:
    if not isinstance(record.get(key), bool):
        raise ValueError(f"{label}.{key} is invalid")


def _require_non_empty_string(record: dict[str, Any], label: str, key: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} is invalid")


def _require_sha256(record: dict[str, Any], label: str, key: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label}.{key} is invalid")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label}.{key} is invalid")


def _require_string_list(record: dict[str, Any], label: str, key: str) -> None:
    value = record.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{label}.{key} is invalid")
