"""Tokenizer candidate artifacts for self-improvement cycles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tokenizer_artifacts import (
    propose_closed_world_subword_tokenizer,
    write_tokenizer_artifacts,
)
from tokenizer_artifact_validation import validate_tokenizer_artifacts


DEFAULT_TOKENIZER_MAX_TOKEN_CHARS = 4
DEFAULT_TOKENIZER_MAX_NEW_TOKENS = 32


def build_self_improvement_tokenizer_candidate(
    train_text_path: Path,
    training_examples: list[Any],
    manifest_path: Path,
    report_path: Path,
    *,
    max_token_chars: int = DEFAULT_TOKENIZER_MAX_TOKEN_CHARS,
    max_new_tokens: int = DEFAULT_TOKENIZER_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    train_text = train_text_path.read_text(encoding="utf-8")
    protected_answers = protected_answer_texts(training_examples)
    proposal = propose_closed_world_subword_tokenizer(
        train_text,
        source_files=[str(train_text_path)],
        protected_answers=protected_answers,
        max_token_chars=max_token_chars,
        max_new_tokens=max_new_tokens,
    )
    validate_tokenizer_artifacts(
        proposal["manifest"],
        proposal["report"],
        manifest_hash=proposal["manifest_hash"],
    )
    write_tokenizer_artifacts(
        manifest_path,
        report_path,
        proposal["manifest"],
        proposal["report"],
    )
    return tokenizer_candidate_record(
        manifest_path,
        report_path,
        proposal["manifest_hash"],
        proposal["manifest"],
        proposal["report"],
    )


def protected_answer_texts(training_examples: list[Any]) -> set[str]:
    return {
        target
        for target in (getattr(example, "target", None) for example in training_examples)
        if isinstance(target, str) and target
    }


def tokenizer_candidate_record(
    manifest_path: Path,
    report_path: Path,
    manifest_hash: str,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    summary = tokenizer_candidate_summary(manifest_hash, manifest, report)
    return {
        "status": "candidate_generated",
        "promotion_status": "candidate_only_not_promoted",
        "rule": (
            "Tokenizer candidates are scored and recorded before training, but "
            "cannot become the active self-improvement tokenizer until model "
            "evaluation proves no retained behavior, unknown-policy, leakage, "
            "or branch-diversity regression."
        ),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "manifest_hash": manifest_hash,
        "summary": summary,
        "manifest": manifest,
        "report": report,
    }


def tokenizer_candidate_summary(
    manifest_hash: str,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    rejected_candidates = manifest.get("rejected_candidates", [])
    full_answer_tokens = report.get("full_answer_tokens", [])
    return {
        "tokenizer_type": manifest.get("tokenizer_type"),
        "manifest_hash": manifest_hash,
        "corpus_hash": manifest.get("corpus_hash"),
        "accepted_token_count": report.get("accepted_token_count", 0),
        "rejected_candidate_count": len(rejected_candidates),
        "round_trip_ok": report.get("round_trip_ok") is True,
        "token_count_savings": report.get("token_count_savings", 0),
        "compression_ratio": report.get("compression_ratio", 1.0),
        "branch_diversity_score": report.get("branch_diversity_score", 0.0),
        "full_answer_token_count": len(full_answer_tokens),
        "full_answer_tokens": list(full_answer_tokens),
        "purity": dict(manifest.get("purity", {})),
    }


def tokenizer_candidate_guard(candidate: dict[str, Any]) -> dict[str, Any]:
    summary = dict(candidate.get("summary", {}))
    purity = dict(summary.get("purity", {}))
    checks = [
        _guard_check(
            "candidate_recorded",
            candidate.get("status") == "candidate_generated",
            "Tokenizer candidate must be generated as an attempt artifact.",
        ),
        _guard_check(
            "active_tokenizer_unchanged",
            candidate.get("promotion_status") == "candidate_only_not_promoted",
            "Self-improvement cannot silently promote a tokenizer candidate.",
        ),
        _tokenizer_artifact_validation_check(candidate),
        _guard_check(
            "round_trip",
            summary.get("round_trip_ok") is True,
            "Candidate tokenizer must exactly round-trip admitted training text.",
        ),
        _guard_check(
            "no_full_answer_tokens",
            summary.get("full_answer_token_count") == 0,
            "Candidate tokenizer must not add protected full-answer tokens.",
        ),
        _guard_check(
            "no_pretrained_tokenizer",
            purity.get("pretrained_tokenizer") is False,
            "Candidate tokenizer must not use pretrained tokenizer state.",
        ),
        _guard_check(
            "no_external_vocabulary",
            purity.get("external_vocabulary") is False,
            "Candidate tokenizer vocabulary must come only from admitted corpus text.",
        ),
        _guard_check(
            "admitted_corpus_only",
            purity.get("admitted_corpus_only") is True,
            "Candidate tokenizer must declare admitted-corpus-only construction.",
        ),
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "rule": "Tokenizer candidates must be corpus-only, round-trip safe, and candidate-only before model evidence can evaluate promotion.",
        "checks": checks,
        "summary": summary,
    }


def _guard_check(name: str, passed: bool, rule: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "status": "passed" if passed else "failed",
        "rule": rule,
    }


def _tokenizer_artifact_validation_check(candidate: dict[str, Any]) -> dict[str, Any]:
    try:
        validate_tokenizer_artifacts(
            candidate.get("manifest"),
            candidate.get("report"),
            manifest_hash=candidate.get("manifest_hash"),
        )
    except ValueError as exc:
        check = _guard_check(
            "validated_artifacts",
            False,
            "Tokenizer candidate manifest/report artifacts must validate before promotion checks can trust them.",
        )
        check["error"] = str(exc)
        return check
    return _guard_check(
        "validated_artifacts",
        True,
        "Tokenizer candidate manifest/report artifacts validated before promotion checks used their summaries.",
    )
