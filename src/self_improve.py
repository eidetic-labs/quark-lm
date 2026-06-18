"""Run and record a closed-world self-improvement cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from admission_probes import audit_all_admission_probes
from answer_decoder import train_decoder
from answer_model import (
    train_model,
)
from curriculum import DEFAULT_CORPUS_DIR
from experiment_registry import (
    record_experiment_decision,
)
from glossary_probes import audit_glossary_probes
from provenance import corpus_diff_for_report
from self_diagnose import diagnose_report
from constraint_first_report import write_constraint_first_report
from self_improvement_constraints import self_improvement_constraint_report
from self_improvement_audits import (
    audit_all_protected_prompts,
    audit_exact_promotion,
    audit_forgetting,
    evaluate_responder,
    promotion_gate,
)
from self_improvement_artifacts import next_attempt, write_report_artifacts
from self_improvement_cycle_setup import prepare_answer_cycle_setup
from self_improvement_experiment import (
    self_improvement_experiment_decision,
    self_improvement_experiment_intent,
)
from self_improvement_tokenizer import tokenizer_candidate_guard


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = PROJECT_DIR / "build"
DEFAULT_RUN_DIR = PROJECT_DIR / "runs" / "self-improve-latest"

__all__ = [
    "audit_exact_promotion",
    "audit_forgetting",
    "evaluate_responder",
    "next_attempt",
    "promotion_gate",
    "self_improvement_experiment_intent",
    "write_report_artifacts",
]


def run_answer_cycle(args: argparse.Namespace) -> dict[str, Any]:
    setup = prepare_answer_cycle_setup(args)

    answer_metrics = train_model(
        SimpleNamespace(
            train_text=setup.train_text_path,
            corpus_dir=args.corpus_dir,
            run=setup.answer_run,
            steps=args.steps,
            learning_rate=args.learning_rate,
            eval_every=args.eval_every,
            seed=args.seed,
        )
    )
    decoder_metrics = train_decoder(
        SimpleNamespace(
            train_text=setup.train_text_path,
            corpus_dir=args.corpus_dir,
            run=setup.decoder_run,
            steps=args.decoder_steps,
            learning_rate=args.decoder_learning_rate,
            eval_every=args.decoder_eval_every,
            seed=args.seed,
            max_answer_chars=args.max_answer_chars,
        )
    )
    report = {
        "cycle": "answer",
        "goal": "Improve exact closed-world response reliability and generative answer reliability while preserving dataset exclusivity.",
        "dataset_exclusivity": {
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "training_text": str(setup.train_text_path),
            "lesson_sources": [
                str(setup.answer_run / "answer_lessons.jsonl"),
                str(setup.decoder_run / "decoder_lessons.jsonl"),
            ],
        },
        "memory_admission": {
            "source": str(args.corpus_dir / "admissions.jsonl"),
            "admitted_facts": setup.curriculum.manifest.get("admitted_facts", 0),
            "rule": "New knowledge becomes learnable only after it is admitted to the ledgered corpus, converted into lessons, and trained into versioned weights.",
            "status": "included_before_weight_update",
        },
        "admission_probe_audit": audit_all_admission_probes(
            args.corpus_dir / "admissions.jsonl"
        ),
        "glossary_probe_audit": audit_glossary_probes(args.corpus_dir / "glossary.json"),
        "prompt_leakage_audit": audit_all_protected_prompts(
            [
                setup.answer_run / "answer_lessons.jsonl",
                setup.decoder_run / "decoder_lessons.jsonl",
            ]
        ),
        "weight_updates": [
            {
                "component": "answer_model",
                "updates_weights": True,
                "initialization": "random softmax weights",
                "checkpoint": answer_metrics["checkpoint"],
                "promotion_rule": "Compare step-0 baseline evals with final evals; promote only when metrics improve or hold under stricter evals.",
                "silent_in_place_mutation": False,
            },
            {
                "component": "answer_decoder",
                "updates_weights": True,
                "initialization": "random prompt-conditioned character decoder weights",
                "checkpoint": decoder_metrics["checkpoint"],
                "promotion_rule": "Compare step-0 generated-answer exact rates with final generated-answer exact rates.",
                "silent_in_place_mutation": False,
            },
        ],
        "curriculum_manifest": setup.curriculum.manifest,
        "corpus_hygiene": setup.corpus_hygiene,
        "training_plan": setup.training_plan,
        "training_recipe": setup.training_recipe,
        "candidate_quarantine": setup.candidate_quarantine,
        "tokenizer_candidate": setup.tokenizer_candidate,
        "closed_world_verifier": setup.closed_world_verifier,
        "corpus_snapshot": setup.snapshot,
        "corpus_diff": corpus_diff_for_report(setup.snapshot, args.compare_report),
        "responder": evaluate_responder(setup.curriculum.train_text),
        "answer_model": {
            "checkpoint": answer_metrics["checkpoint"],
            "history": answer_metrics["history"],
            "baseline": answer_metrics["baseline"]["evals"],
            "final": answer_metrics["final"]["evals"],
            "examples": answer_metrics["examples"],
            "training_examples": answer_metrics["training_examples"],
            "features": answer_metrics["features"],
            "labels": answer_metrics["labels"],
        },
        "answer_decoder": {
            "checkpoint": decoder_metrics["checkpoint"],
            "history": decoder_metrics["history"],
            "baseline": decoder_metrics["baseline"]["evals"],
            "final": decoder_metrics["final"]["evals"],
            "examples": decoder_metrics["examples"],
            "training_examples": decoder_metrics["training_examples"],
            "features": decoder_metrics["features"],
            "labels": decoder_metrics["labels"],
        },
        "experiment_intent": setup.experiment_intent,
    }
    report["tokenizer_candidate_guard"] = tokenizer_candidate_guard(
        setup.tokenizer_candidate
    )
    report["forgetting_audit"] = audit_forgetting(report, args.compare_report)
    report["exact_eval_audit"] = audit_exact_promotion(report)
    report["constraint_first_promotion"] = self_improvement_constraint_report(report)
    write_constraint_first_report(
        setup.constraint_first_path,
        report["constraint_first_promotion"],
    )
    report["promotion_gate"] = promotion_gate(report)
    report["self_diagnosis"] = diagnose_report(report)
    status, summary, evidence = self_improvement_experiment_decision(report)
    report["experiment_intent"] = record_experiment_decision(
        setup.experiment_intent,
        status,
        summary,
        evidence,
    )
    setup.run_dir.mkdir(parents=True, exist_ok=True)
    write_report_artifacts(
        report,
        setup.run_dir,
        setup.attempt_dir,
        setup.attempt_number,
    )
    print(json.dumps({"answer_model": report["answer_model"]["final"]}, indent=2, sort_keys=True))
    print(json.dumps({"answer_decoder": report["answer_decoder"]["final"]}, indent=2, sort_keys=True))
    print(f"wrote {setup.attempt_dir / 'self_improvement_report.json'}")
    print(f"updated {setup.run_dir / 'self_improvement_report.json'}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    answer = subparsers.add_parser("answer-cycle")
    answer.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    answer.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    answer.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    answer.add_argument("--steps", type=int, default=3600)
    answer.add_argument("--learning-rate", type=float, default=0.08)
    answer.add_argument("--eval-every", type=int, default=400)
    answer.add_argument("--decoder-steps", type=int, default=40000)
    answer.add_argument("--decoder-learning-rate", type=float, default=0.035)
    answer.add_argument("--decoder-eval-every", type=int, default=2000)
    answer.add_argument("--max-answer-chars", type=int, default=64)
    answer.add_argument("--seed", type=int, default=7)
    answer.add_argument("--compare-report", type=Path, default=None)
    answer.add_argument("--experiment-version", default="v0.77")
    answer.add_argument("--experiment-hypothesis", default=None)
    answer.add_argument("--experiment-note", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "answer-cycle":
        report = run_answer_cycle(args)
        return 0 if report["promotion_gate"]["passed"] else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
