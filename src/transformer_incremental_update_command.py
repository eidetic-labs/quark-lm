"""CLI command adapter for guarded incremental transformer updates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from probes import read_jsonl
from transformer_incremental_update import guarded_incremental_update
from transformer_model import generation_config_from_args


def incremental_update_command(args: argparse.Namespace, model_cls: Any) -> dict[str, Any]:
    return guarded_incremental_update(
        model_cls=model_cls,
        base_checkpoint=args.base_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        accepted_checkpoint=args.accepted_checkpoint,
        new_lesson_records=_read_probe_records(args.new_lesson_probe),
        regression_records=_read_probe_records(args.regression_probe),
        nll_tolerance=args.nll_tolerance,
        generation_config=generation_config_from_args(args),
        report_path=args.report,
    )


def _read_probe_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_jsonl(path))
    return records
