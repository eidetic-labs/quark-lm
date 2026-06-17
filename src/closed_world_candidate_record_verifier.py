"""Closed-world verifier checks for candidate quarantine records."""

from __future__ import annotations

from typing import Any

from candidate_quarantine import TRAINING_ELIGIBLE_STATES, validate_candidate_record
from closed_world_verifier_reports import verifier_check, verifier_report
from corpus_responder import CorpusResponder


def verify_candidate_record(
    record: dict[str, Any],
    responder: CorpusResponder | None = None,
) -> dict[str, Any]:
    validation_error: str | None = None
    try:
        validate_candidate_record(record)
    except ValueError as exc:
        validation_error = str(exc)

    candidate_type = record.get("candidate_type")
    prompt = record.get("prompt", "")
    target = record.get("target", "")
    proposal = record.get("proposal", "")
    state = record.get("state", "")
    admission_id = record.get("admission_id")
    checks = [
        verifier_check(
            "candidate_record_valid",
            validation_error is None,
            "Candidate records must satisfy the quarantine schema.",
            {"error": validation_error},
        ),
        verifier_check(
            "source_label_present",
            isinstance(record.get("source"), str)
            and bool(record.get("source", "").strip()),
            "Every candidate must retain a non-empty origin label.",
            {"source": record.get("source")},
        ),
    ]

    payload_present = _candidate_payload_present(
        candidate_type,
        prompt,
        target,
        proposal,
    )
    checks.append(
        verifier_check(
            "candidate_payload_present",
            payload_present,
            "A candidate must carry the prompt, target, or proposal fields required by its type.",
            {
                "candidate_type": candidate_type,
                "has_prompt": bool(prompt),
                "has_target": bool(target),
                "has_proposal": bool(proposal),
            },
        )
    )

    if prompt and target:
        _append_exact_answer_check(checks, prompt, target, responder)

    eligible = state in TRAINING_ELIGIBLE_STATES
    checks.append(
        verifier_check(
            "training_eligible_state_has_admission",
            not eligible
            or (isinstance(admission_id, str) and bool(admission_id.strip())),
            "Training-eligible candidate states must link to a ledger admission id.",
            {"state": state, "admission_id": admission_id},
        )
    )

    return verifier_report(
        component="candidate_quarantine",
        run_id=str(record.get("candidate_id", "candidate")),
        subject_kind="candidate_record",
        checks=checks,
    )


def _append_exact_answer_check(
    checks: list[dict[str, Any]],
    prompt: str,
    target: str,
    responder: CorpusResponder | None,
) -> None:
    if responder is None:
        checks.append(
            verifier_check(
                "exact_answer_consistency",
                True,
                "Prompt-target candidates can be checked against a closed-world responder before admission.",
                {"reason": "responder_missing"},
            )
        )
        return
    answer = responder.answer_prompt(prompt)
    checks.append(
        verifier_check(
            "exact_answer_consistency",
            answer == target,
            "Candidate targets must match the deterministic responder trained from admitted text.",
            {"answer": answer, "target": target},
        )
    )


def _candidate_payload_present(
    candidate_type: Any,
    prompt: Any,
    target: Any,
    proposal: Any,
) -> bool:
    has_prompt_target = (
        isinstance(prompt, str)
        and bool(prompt.strip())
        and isinstance(target, str)
        and bool(target.strip())
    )
    has_proposal = isinstance(proposal, str) and bool(proposal.strip())
    if candidate_type in ("lesson", "probe"):
        return has_prompt_target
    if candidate_type in ("repair_proposal", "diagnosis", "memory"):
        return has_proposal or has_prompt_target
    return False
