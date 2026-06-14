"""Run and record a closed-world self-improvement cycle."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .admission_probes import audit_all_admission_probes
from .answer_decoder import train_decoder
from .answer_model import (
    DEFAULT_EVALS,
    answer_training_pool,
    load_training_examples,
    train_model,
)
from .candidate_quarantine import (
    build_candidate_quarantine_manifest,
    candidate_quarantine_summary,
    write_candidate_quarantine,
)
from .corpus_hygiene import (
    build_corpus_hygiene_report,
    build_training_plan,
    write_json_artifact,
)
from .curriculum import DEFAULT_CORPUS_DIR, build_curriculum, write_json, write_curriculum
from .experiment_registry import (
    ExperimentIntent,
    record_experiment_decision,
    write_experiment_intent,
)
from .glossary_probes import audit_glossary_probes
from .probes import read_jsonl
from .provenance import corpus_diff_for_report, corpus_snapshot
from .respond import CorpusResponder
from .self_diagnose import diagnose_report


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_DIR = PROJECT_DIR / "build"
DEFAULT_RUN_DIR = PROJECT_DIR / "runs" / "self-improve-latest"
ATTEMPT_DIR_RE = re.compile(r"^attempt-(?P<number>\d+)$")


def evaluate_responder(train_text: str) -> dict[str, Any]:
    responder = CorpusResponder.train_from_text(train_text)
    return {
        path.stem: summarize_exact(responder.evaluate(read_jsonl(path)))
        for path in DEFAULT_EVALS
    }


def summarize_exact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "count": result["count"],
        "exact": result["exact"],
        "exact_rate": result["exact_rate"],
    }


def audit_prompt_leakage(
    lesson_paths: list[Path],
    eval_path: Path,
    protected_id_contains: str | None = None,
) -> dict[str, Any]:
    records = read_jsonl(eval_path)
    if protected_id_contains is not None:
        records = [record for record in records if protected_id_contains in record["id"]]
    eval_prompts = {record["prompt"] for record in records}
    leaked: list[dict[str, str]] = []
    for lesson_path in lesson_paths:
        if not lesson_path.exists():
            leaked.append({"lesson_source": str(lesson_path), "prompt": "<missing lesson file>"})
            continue
        with lesson_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("prompt") in eval_prompts:
                    leaked.append(
                        {
                            "lesson_source": str(lesson_path),
                            "prompt": record["prompt"],
                        }
                    )
    return {
        "eval_source": str(eval_path),
        "protected_id_contains": protected_id_contains,
        "lesson_sources": [str(path) for path in lesson_paths],
        "leaked_prompts": leaked,
        "passed": not leaked,
    }


def audit_all_protected_prompts(lesson_paths: list[Path]) -> dict[str, Any]:
    return {
        "heldout": audit_prompt_leakage(
            lesson_paths,
            PROJECT_DIR / "evals" / "heldout.jsonl",
        ),
        "owner_heldout": audit_prompt_leakage(
            lesson_paths,
            PROJECT_DIR / "evals" / "owner.jsonl",
            protected_id_contains="-heldout-",
        ),
    }


def read_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def component_final_evals(report: dict[str, Any], component: str) -> dict[str, Any]:
    if component == "responder":
        return report.get("responder", {})
    return report.get(component, {}).get("final", {})


def audit_forgetting(
    current_report: dict[str, Any],
    previous_report_path: Path | None,
) -> dict[str, Any]:
    if previous_report_path is None:
        return {
            "mode": "previous_report",
            "compare_report": None,
            "passed": True,
            "status": "not_evaluated_no_previous_report",
            "checks": [],
        }

    previous_report = read_report(previous_report_path)
    checks: list[dict[str, Any]] = []
    for component in ("responder", "answer_model", "answer_decoder"):
        previous_evals = component_final_evals(previous_report, component)
        current_evals = component_final_evals(current_report, component)
        for eval_name in sorted(set(previous_evals) | set(current_evals)):
            if eval_name not in previous_evals:
                checks.append(
                    {
                        "component": component,
                        "eval": eval_name,
                        "status": "new_eval",
                        "passed": True,
                    }
                )
                continue
            if eval_name not in current_evals:
                checks.append(
                    {
                        "component": component,
                        "eval": eval_name,
                        "status": "missing_current_eval",
                        "passed": False,
                    }
                )
                continue

            previous = previous_evals[eval_name]
            current = current_evals[eval_name]
            current_count = current.get("count", 0)
            previous_count = previous.get("count", 0)
            current_exact = current.get("exact", 0)
            previous_exact = previous.get("exact", 0)
            current_rate = current.get("exact_rate", 0.0)
            previous_rate = previous.get("exact_rate", 0.0)
            passed = (
                current_count >= previous_count
                and current_exact >= previous_exact
                and current_rate >= previous_rate
            )
            checks.append(
                {
                    "component": component,
                    "eval": eval_name,
                    "status": "compared",
                    "previous": {
                        "count": previous_count,
                        "exact": previous_exact,
                        "exact_rate": previous_rate,
                    },
                    "current": {
                        "count": current_count,
                        "exact": current_exact,
                        "exact_rate": current_rate,
                    },
                    "passed": passed,
                }
            )

    return {
        "mode": "previous_report",
        "compare_report": str(previous_report_path),
        "passed": all(check["passed"] for check in checks),
        "status": "evaluated",
        "rule": "For every shared eval set, the current run must keep at least the previous count, exact matches, and exact rate.",
        "checks": checks,
    }


def audit_exact_promotion(report: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for component, evals in {
        "responder": report.get("responder", {}),
        "answer_model": report.get("answer_model", {}).get("final", {}),
        "answer_decoder": report.get("answer_decoder", {}).get("final", {}),
    }.items():
        for eval_name, metrics in sorted(evals.items()):
            count = metrics.get("count", 0)
            exact = metrics.get("exact", 0)
            checks.append(
                {
                    "component": component,
                    "eval": eval_name,
                    "count": count,
                    "exact": exact,
                    "passed": count > 0 and exact == count,
                }
            )
    return {
        "passed": all(check["passed"] for check in checks),
        "rule": "Every responder, classifier, and decoder eval must be non-empty and exact before promotion.",
        "checks": checks,
    }


def promotion_gate(report: dict[str, Any]) -> dict[str, Any]:
    prompt_leakage = report["prompt_leakage_audit"]
    checks = [
        {
            "name": "admission_probe_audit",
            "passed": report["admission_probe_audit"]["passed"],
        },
        {
            "name": "glossary_probe_audit",
            "passed": report["glossary_probe_audit"]["passed"],
        },
        {
            "name": "heldout_prompt_leakage",
            "passed": prompt_leakage["heldout"]["passed"],
        },
        {
            "name": "owner_heldout_prompt_leakage",
            "passed": prompt_leakage["owner_heldout"]["passed"],
        },
        {
            "name": "forgetting_audit",
            "passed": report["forgetting_audit"]["passed"],
        },
        {
            "name": "exact_eval_audit",
            "passed": report["exact_eval_audit"]["passed"],
        },
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "rule": "Promote only when generated probes, prompt leakage, forgetting, and exact eval audits all pass.",
        "checks": checks,
    }


def next_attempt(run_dir: Path) -> tuple[int, Path]:
    attempts_dir = run_dir / "attempts"
    numbers: list[int] = []
    if attempts_dir.exists():
        for child in attempts_dir.iterdir():
            match = ATTEMPT_DIR_RE.match(child.name)
            if match and child.is_dir():
                numbers.append(int(match["number"]))
    number = max(numbers, default=0) + 1
    return number, attempts_dir / f"attempt-{number:03d}"


def write_report_artifacts(
    report: dict[str, Any],
    run_dir: Path,
    attempt_dir: Path,
    attempt_number: int,
) -> None:
    report["attempt"] = {
        "index": attempt_number,
        "path": str(attempt_dir),
        "report": str(attempt_dir / "self_improvement_report.json"),
        "latest_report": str(run_dir / "self_improvement_report.json"),
    }
    write_json(attempt_dir / "corpus_snapshot.json", report["corpus_snapshot"])
    write_json(attempt_dir / "corpus_diff.json", report["corpus_diff"])
    if "corpus_hygiene" in report:
        write_json_artifact(attempt_dir / "corpus_hygiene.json", report["corpus_hygiene"])
    if "training_plan" in report:
        write_json_artifact(attempt_dir / "training_plan.json", report["training_plan"])
    if "candidate_quarantine" in report:
        write_candidate_quarantine(
            attempt_dir / "candidate_quarantine.json",
            report["candidate_quarantine"],
        )
    if "experiment_intent" in report:
        write_experiment_intent(
            attempt_dir / "experiment_intent.json",
            report["experiment_intent"],
        )
    write_json(attempt_dir / "self_improvement_report.json", report)
    write_json(run_dir / "corpus_snapshot.json", report["corpus_snapshot"])
    write_json(run_dir / "corpus_diff.json", report["corpus_diff"])
    if "corpus_hygiene" in report:
        write_json_artifact(run_dir / "corpus_hygiene.json", report["corpus_hygiene"])
    if "training_plan" in report:
        write_json_artifact(run_dir / "training_plan.json", report["training_plan"])
    if "candidate_quarantine" in report:
        write_candidate_quarantine(
            run_dir / "candidate_quarantine.json",
            report["candidate_quarantine"],
        )
    if "experiment_intent" in report:
        write_experiment_intent(
            run_dir / "experiment_intent.json",
            report["experiment_intent"],
        )
    write_json(run_dir / "self_improvement_report.json", report)


def self_improvement_experiment_intent(
    args: argparse.Namespace,
    run_dir: Path,
    attempt_dir: Path,
    train_text_path: Path,
) -> dict[str, Any]:
    hypothesis = getattr(args, "experiment_hypothesis", None) or (
        "A closed-world answer-cycle can update answer_model and answer_decoder "
        "weights from admitted corpus lessons while preserving exact evals, "
        "prompt-leakage controls, and forgetting gates."
    )
    notes = getattr(args, "experiment_note", None) or []
    intent = ExperimentIntent(
        version=getattr(args, "experiment_version", "v0.75"),
        run_id=attempt_dir.name or run_dir.name,
        component="self-improvement-answer-cycle",
        hypothesis=hypothesis,
        allowed_data_sources=[
            str(args.corpus_dir / "admissions.jsonl"),
            str(args.corpus_dir / "glossary.json"),
            str(args.corpus_dir / "grammar.json"),
            str(train_text_path),
        ],
        planned_artifacts=[
            str(attempt_dir / "answer" / "answer_model.json"),
            str(attempt_dir / "decoder" / "answer_decoder.json"),
            str(attempt_dir / "corpus_snapshot.json"),
            str(attempt_dir / "corpus_diff.json"),
            str(attempt_dir / "corpus_hygiene.json"),
            str(attempt_dir / "training_plan.json"),
            str(attempt_dir / "candidate_quarantine.json"),
            str(attempt_dir / "experiment_intent.json"),
            str(attempt_dir / "self_improvement_report.json"),
            str(run_dir / "corpus_hygiene.json"),
            str(run_dir / "training_plan.json"),
            str(run_dir / "candidate_quarantine.json"),
            str(run_dir / "experiment_intent.json"),
            str(run_dir / "self_improvement_report.json"),
        ],
        training_recipe_id="self-improve-answer-cycle:v0.75",
        acceptance_gates=[
            {
                "name": "admission_probe_audit",
                "rule": "Generated probes for every admitted fact must pass.",
                "required": True,
            },
            {
                "name": "glossary_probe_audit",
                "rule": "Glossary-derived probes must remain exact.",
                "required": True,
            },
            {
                "name": "heldout_prompt_leakage",
                "rule": "Heldout prompts must not appear in training lessons.",
                "required": True,
            },
            {
                "name": "owner_heldout_prompt_leakage",
                "rule": "Protected owner heldout prompts must not appear in lessons.",
                "required": True,
            },
            {
                "name": "forgetting_audit",
                "rule": "Current evals may not regress against the comparison report.",
                "required": True,
            },
            {
                "name": "exact_eval_audit",
                "rule": "Responder, answer model, and decoder evals must be exact.",
                "required": True,
            },
            {
                "name": "promotion_gate",
                "rule": "All required audits must pass before the result can promote.",
                "required": True,
            },
        ],
        failure_criteria=[
            "Any required probe, leakage, forgetting, or exact-eval audit fails.",
            "Training writes checkpoints without a matching report and intent artifact.",
            "The run uses pretrained weights, pretrained tokenizers, or external embeddings.",
        ],
        notes=notes,
    )
    return intent.to_record()


def self_improvement_experiment_decision(
    report: dict[str, Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    gate = report["promotion_gate"]
    evidence = [
        {
            "name": "admission_probe_audit",
            "passed": report["admission_probe_audit"]["passed"],
        },
        {
            "name": "glossary_probe_audit",
            "passed": report["glossary_probe_audit"]["passed"],
        },
        {
            "name": "heldout_prompt_leakage",
            "passed": report["prompt_leakage_audit"]["heldout"]["passed"],
        },
        {
            "name": "owner_heldout_prompt_leakage",
            "passed": report["prompt_leakage_audit"]["owner_heldout"]["passed"],
        },
        {"name": "forgetting_audit", "passed": report["forgetting_audit"]["passed"]},
        {"name": "exact_eval_audit", "passed": report["exact_eval_audit"]["passed"]},
        {"name": "promotion_gate", "passed": gate["passed"]},
    ]
    if gate["passed"]:
        return (
            "promoted",
            "Self-improvement run passed all declared gates and is eligible for promotion.",
            evidence,
        )
    return (
        "rejected",
        "Self-improvement run was recorded as evidence but failed at least one declared gate.",
        evidence,
    )


def run_answer_cycle(args: argparse.Namespace) -> dict[str, Any]:
    curriculum = build_curriculum(args.corpus_dir, args.seed)
    write_curriculum(curriculum, args.build_dir)
    snapshot = corpus_snapshot(args.corpus_dir)
    train_text_path = args.build_dir / "train.txt"
    training_examples = load_training_examples(train_text_path, args.corpus_dir)
    scheduled_training_examples = answer_training_pool(training_examples)
    run_dir = args.run
    attempt_number, attempt_dir = next_attempt(run_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    answer_run = attempt_dir / "answer"
    decoder_run = attempt_dir / "decoder"
    experiment_intent = self_improvement_experiment_intent(
        args,
        run_dir,
        attempt_dir,
        train_text_path,
    )
    write_experiment_intent(attempt_dir / "experiment_intent.json", experiment_intent)
    hygiene_path = attempt_dir / "corpus_hygiene.json"
    training_plan_path = attempt_dir / "training_plan.json"
    candidate_quarantine_path = attempt_dir / "candidate_quarantine.json"
    candidate_quarantine = build_candidate_quarantine_manifest(
        "self-improvement-answer-cycle",
        attempt_dir.name,
    )
    write_candidate_quarantine(candidate_quarantine_path, candidate_quarantine)
    candidate_summary = candidate_quarantine_summary(candidate_quarantine)
    corpus_hygiene = build_corpus_hygiene_report(
        "self-improvement-answer-cycle",
        args.corpus_dir,
        train_text_path,
        DEFAULT_EVALS,
        training_examples,
    )
    training_plan = build_training_plan(
        "self-improvement-answer-cycle",
        attempt_dir.name,
        train_text_path,
        args.corpus_dir,
        DEFAULT_EVALS,
        training_examples,
        scheduled_training_examples,
        hygiene_path,
        planned_artifacts=[
            answer_run / "answer_model.json",
            decoder_run / "answer_decoder.json",
            hygiene_path,
            training_plan_path,
            candidate_quarantine_path,
            attempt_dir / "self_improvement_report.json",
        ],
        candidate_quarantine_path=candidate_quarantine_path,
        candidate_quarantine_summary=candidate_summary,
    )
    write_json_artifact(hygiene_path, corpus_hygiene)
    write_json_artifact(training_plan_path, training_plan)

    answer_metrics = train_model(
        SimpleNamespace(
            train_text=train_text_path,
            corpus_dir=args.corpus_dir,
            run=answer_run,
            steps=args.steps,
            learning_rate=args.learning_rate,
            eval_every=args.eval_every,
            seed=args.seed,
        )
    )
    decoder_metrics = train_decoder(
        SimpleNamespace(
            train_text=train_text_path,
            corpus_dir=args.corpus_dir,
            run=decoder_run,
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
            "training_text": str(train_text_path),
            "lesson_sources": [
                str(answer_run / "answer_lessons.jsonl"),
                str(decoder_run / "decoder_lessons.jsonl"),
            ],
        },
        "memory_admission": {
            "source": str(args.corpus_dir / "admissions.jsonl"),
            "admitted_facts": curriculum.manifest.get("admitted_facts", 0),
            "rule": "New knowledge becomes learnable only after it is admitted to the ledgered corpus, converted into lessons, and trained into versioned weights.",
            "status": "included_before_weight_update",
        },
        "admission_probe_audit": audit_all_admission_probes(
            args.corpus_dir / "admissions.jsonl"
        ),
        "glossary_probe_audit": audit_glossary_probes(args.corpus_dir / "glossary.json"),
        "prompt_leakage_audit": audit_all_protected_prompts(
            [
                answer_run / "answer_lessons.jsonl",
                decoder_run / "decoder_lessons.jsonl",
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
        "curriculum_manifest": curriculum.manifest,
        "corpus_hygiene": corpus_hygiene,
        "training_plan": training_plan,
        "candidate_quarantine": candidate_quarantine,
        "corpus_snapshot": snapshot,
        "corpus_diff": corpus_diff_for_report(snapshot, args.compare_report),
        "responder": evaluate_responder(curriculum.train_text),
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
        "experiment_intent": experiment_intent,
    }
    report["forgetting_audit"] = audit_forgetting(report, args.compare_report)
    report["exact_eval_audit"] = audit_exact_promotion(report)
    report["promotion_gate"] = promotion_gate(report)
    report["self_diagnosis"] = diagnose_report(report)
    status, summary, evidence = self_improvement_experiment_decision(report)
    report["experiment_intent"] = record_experiment_decision(
        experiment_intent,
        status,
        summary,
        evidence,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_report_artifacts(report, run_dir, attempt_dir, attempt_number)
    print(json.dumps({"answer_model": report["answer_model"]["final"]}, indent=2, sort_keys=True))
    print(json.dumps({"answer_decoder": report["answer_decoder"]["final"]}, indent=2, sort_keys=True))
    print(f"wrote {attempt_dir / 'self_improvement_report.json'}")
    print(f"updated {run_dir / 'self_improvement_report.json'}")
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
    answer.add_argument("--experiment-version", default="v0.75")
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
