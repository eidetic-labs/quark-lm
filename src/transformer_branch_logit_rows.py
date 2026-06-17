"""Row collection for branch logit-prior diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from answer_model import AnswerExample
from tokenizer import CharTokenizer
from transformer_direct_answer_core import direct_answer_branch_context
from transformer_direct_modes import ANSWER_TERMINATOR


@dataclass(frozen=True)
class BranchLogitPriorRows:
    rows: list[dict[str, Any]]
    skipped: int
    predicted_counts: Counter[int]
    target_counts: Counter[int]


def collect_branch_logit_prior_rows(
    model: Any,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> BranchLogitPriorRows:
    rows: list[dict[str, Any]] = []
    skipped = 0
    predicted_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()

    for record in records:
        example = AnswerExample(
            prompt=record["prompt"],
            target=record["target"],
            source=f"eval:{record['id']}",
        )
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            example,
            branch_position,
            terminator,
        )
        if branch is None:
            skipped += 1
            continue
        context, target_id, position = branch
        logits = model._forward_floats(context)
        hidden = model.final_hidden(context)
        ranked_ids = sorted(
            range(len(logits)),
            key=lambda index: (-logits[index], tokenizer.itos[index], index),
        )
        predicted_id = ranked_ids[0]
        predicted_counts[predicted_id] += 1
        target_counts[target_id] += 1
        rows.append(
            {
                "id": record["id"],
                "position": position,
                "target_id": target_id,
                "predicted_id": predicted_id,
                "target_rank": ranked_ids.index(target_id) + 1,
                "logits": logits,
                "hidden": hidden,
            }
        )

    return BranchLogitPriorRows(
        rows=rows,
        skipped=skipped,
        predicted_counts=predicted_counts,
        target_counts=target_counts,
    )
