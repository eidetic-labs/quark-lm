"""Standalone validation for closed-world tokenizer artifacts."""

from __future__ import annotations

from typing import Any

from closed_world_subword_tokenizer import SUBWORD_TOKENIZER_TYPE
from tokenizer_artifacts import TOKENIZER_ARTIFACT_VERSION, stable_json_hash


def validate_tokenizer_artifacts(
    manifest: dict[str, Any],
    report: dict[str, Any],
    *,
    manifest_hash: str | None = None,
    require_no_full_answer_tokens: bool = True,
) -> None:
    """Validate tokenizer manifest/report evidence before trusting it."""

    if not isinstance(manifest, dict):
        raise ValueError("tokenizer manifest must be a dict")
    if not isinstance(report, dict):
        raise ValueError("tokenizer report must be a dict")
    _validate_manifest(manifest)
    _validate_report(report)
    if report.get("tokenizer_type") != manifest.get("tokenizer_type"):
        raise ValueError("tokenizer report tokenizer_type is inconsistent")
    if require_no_full_answer_tokens and report.get("full_answer_tokens") != []:
        raise ValueError("tokenizer report full_answer_tokens must be empty")
    if manifest_hash is not None and manifest_hash != stable_json_hash(manifest):
        raise ValueError("tokenizer manifest_hash is inconsistent")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("version") != TOKENIZER_ARTIFACT_VERSION:
        raise ValueError("tokenizer manifest version is inconsistent")
    if manifest.get("tokenizer_type") != SUBWORD_TOKENIZER_TYPE:
        raise ValueError("tokenizer manifest tokenizer_type is unsupported")
    _require_sha256(manifest, "tokenizer manifest", "corpus_hash")
    _require_string_list(manifest, "tokenizer manifest", "source_files")
    _require_non_empty_list(manifest, "tokens")
    if not isinstance(manifest.get("merge_rules"), list):
        raise ValueError("tokenizer manifest merge_rules must be a list")
    if not isinstance(manifest.get("candidate_scores"), list):
        raise ValueError("tokenizer manifest candidate_scores must be a list")
    if not isinstance(manifest.get("rejected_candidates"), list):
        raise ValueError("tokenizer manifest rejected_candidates must be a list")
    _validate_purity(manifest.get("purity"))


def _validate_report(report: dict[str, Any]) -> None:
    if report.get("version") != TOKENIZER_ARTIFACT_VERSION:
        raise ValueError("tokenizer report version is inconsistent")
    if report.get("tokenizer_type") != SUBWORD_TOKENIZER_TYPE:
        raise ValueError("tokenizer report tokenizer_type is unsupported")
    if report.get("round_trip_ok") is not True:
        raise ValueError("tokenizer report round_trip_ok must be true")
    _require_non_negative_int(report, "char_token_count")
    _require_non_negative_int(report, "subword_token_count")
    expected_savings = report["char_token_count"] - report["subword_token_count"]
    if report.get("token_count_savings") != expected_savings:
        raise ValueError("tokenizer report token_count_savings is inconsistent")
    _require_number(report, "compression_ratio")
    _require_non_negative_int(report, "accepted_token_count")
    _require_non_negative_int(report, "max_target_token_chars")
    _require_number(report, "branch_diversity_score")
    if not isinstance(report.get("full_answer_tokens"), list):
        raise ValueError("tokenizer report full_answer_tokens must be a list")
    if not isinstance(report.get("long_answer_effect"), dict):
        raise ValueError("tokenizer report long_answer_effect must be a dict")


def _validate_purity(purity: Any) -> None:
    if not isinstance(purity, dict):
        raise ValueError("tokenizer manifest purity must be a dict")
    expected = {
        "pretrained_tokenizer": False,
        "external_vocabulary": False,
        "normalization": "none",
        "admitted_corpus_only": True,
    }
    for key, value in expected.items():
        if purity.get(key) != value:
            raise ValueError(f"tokenizer manifest purity.{key} is inconsistent")


def _require_sha256(record: dict[str, Any], label: str, key: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} {key} is invalid")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} {key} is invalid")


def _require_string_list(record: dict[str, Any], label: str, key: str) -> None:
    value = record.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{label} {key} must be a list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} {key} must contain non-empty strings")


def _require_non_empty_list(record: dict[str, Any], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"tokenizer manifest {key} must be a non-empty list")


def _require_non_negative_int(record: dict[str, Any], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"tokenizer report {key} must be a non-negative int")


def _require_number(record: dict[str, Any], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"tokenizer report {key} must be numeric")
