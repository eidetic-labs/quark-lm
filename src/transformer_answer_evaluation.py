"""Answer candidate and completion evaluation helpers."""

from __future__ import annotations

from typing import Any

from neural_char_metrics import continuation_nll
from probes import summarize
from tokenizer_protocol import TokenizerProtocol
from transformer_answer_selector import AnswerCandidateSelector
from transformer_eval import score_transformer_records
from transformer_model import GenerationConfig


def evaluate_answer_records(
    model: Any,
    tokenizer: TokenizerProtocol,
    records: list[dict[str, Any]],
    candidates: list[str],
    max_new_chars: int,
    include_completions: bool = True,
    selector: AnswerCandidateSelector | None = None,
    emit_selected_candidate: bool = False,
    generation_config: GenerationConfig | None = None,
) -> dict[str, Any]:
    if not include_completions:
        return evaluate_answer_candidates(
            model,
            tokenizer,
            records,
            candidates,
            selector,
            emit_selected_candidate=emit_selected_candidate,
        )
    scored = score_transformer_records(
        model,
        tokenizer,
        records,
        max_new_chars=max_new_chars,
        generation_config=generation_config or GenerationConfig(),
        candidates=candidates,
    )
    summary = summarize(scored)
    failed_exact = [record for record in scored if not record["exact_match"]]
    failed_candidate = [record for record in scored if not record["candidate_match"]]
    return {
        **summary,
        "failed_records": failed_exact,
        "failed_candidate_records": failed_candidate,
    }

def evaluate_answer_candidates(
    model: Any,
    tokenizer: TokenizerProtocol,
    records: list[dict[str, Any]],
    candidates: list[str],
    selector: AnswerCandidateSelector | None = None,
    emit_selected_candidate: bool = False,
) -> dict[str, Any]:
    if emit_selected_candidate and selector is None:
        raise ValueError("selector-assisted emission requires a selector")
    scored: list[dict[str, Any]] = []
    for record in records:
        if selector is None:
            candidate_scores = [
                {
                    "target": candidate,
                    "target_nll": continuation_nll(
                        model,
                        tokenizer,
                        record["prompt"],
                        candidate,
                    ),
                }
                for candidate in candidates
            ]
            predicted_candidate = min(
                candidate_scores,
                key=lambda item: float(item["target_nll"]),
            )["target"]
            candidate_scorer = "transformer_nll"
            if record["target"] in candidates:
                target_nll = next(
                    float(item["target_nll"])
                    for item in candidate_scores
                    if item["target"] == record["target"]
                )
            else:
                target_nll = continuation_nll(
                    model,
                    tokenizer,
                    record["prompt"],
                    record["target"],
                )
        else:
            predicted_candidate = selector.predict(record["prompt"], candidates)
            candidate_scorer = "answer_candidate_selector"
            target_nll = continuation_nll(
                model,
                tokenizer,
                record["prompt"],
                record["target"],
            )
        completion = predicted_candidate if emit_selected_candidate else None
        exact_match = completion == record["target"] if completion is not None else False
        scored.append(
            {
                "id": record["id"],
                "target": record["target"],
                "completion": completion,
                "exact_match": exact_match,
                "candidate_match": predicted_candidate == record["target"],
                "predicted_candidate": predicted_candidate,
                "candidate_scorer": candidate_scorer,
                "completion_source": "selector_candidate"
                if emit_selected_candidate
                else None,
                "target_selector_score": selector.score(record["prompt"], record["target"])
                if selector is not None
                else None,
                "target_nll": target_nll,
            }
        )
    summary = summarize(scored)
    failed_exact = [record for record in scored if not record["exact_match"]]
    failed_candidate = [record for record in scored if not record["candidate_match"]]
    return {
        **summary,
        "exact": summary["exact"] if emit_selected_candidate else None,
        "exact_rate": summary["exact_rate"] if emit_selected_candidate else None,
        "failed_records": failed_exact if emit_selected_candidate else [],
        "failed_candidate_records": failed_candidate,
    }
