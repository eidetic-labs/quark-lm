"""Command handlers for answer model workflows."""

from __future__ import annotations

import argparse
import json
from typing import Any

from answer_model_constants import DEFAULT_EVALS
from answer_model_evaluation import evaluate_records
from answer_model_softmax import AnswerSoftmax
from probes import read_jsonl


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    model = AnswerSoftmax.load(args.checkpoint)
    result = {
        path.stem: evaluate_records(model, read_jsonl(path))
        for path in DEFAULT_EVALS
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    summary = {
        name: {
            "count": value["count"],
            "exact": value["exact"],
            "exact_rate": value["exact_rate"],
            "avg_target_loss": value["avg_target_loss"],
        }
        for name, value in result.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result
