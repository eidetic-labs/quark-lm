"""Rule-based diagnosis for closed-world self-improvement reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def failed_gate_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    gate = report.get("promotion_gate", {})
    return [
        {
            "kind": "promotion_gate",
            "name": check["name"],
            "component": None,
            "eval": None,
            "record_id": None,
            "target": None,
            "prediction": None,
        }
        for check in gate.get("checks", [])
        if not check.get("passed", False)
    ]


def failed_eval_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for component in ("answer_model", "answer_decoder"):
        final = report.get(component, {}).get("final", {})
        for eval_name, metrics in final.items():
            for record in metrics.get("failed_records", []):
                failures.append(
                    {
                        "kind": "failed_record",
                        "name": None,
                        "component": component,
                        "eval": eval_name,
                        "record_id": record.get("id"),
                        "target": record.get("target"),
                        "prediction": record.get("prediction"),
                    }
                )
    return failures


def action_for_blocker(blocker: dict[str, Any]) -> dict[str, Any]:
    if blocker["kind"] == "promotion_gate":
        return gate_action(blocker["name"])
    return eval_action(blocker)


def gate_action(name: str) -> dict[str, Any]:
    if name == "admission_probe_audit":
        return {
            "action": "regenerate_admission_probes",
            "command": "PYTHONPATH=src python3 -m admission_probes",
            "rationale": "Admission probes must be derived from corpus/admissions.jsonl before promotion.",
        }
    if name == "glossary_probe_audit":
        return {
            "action": "regenerate_glossary_probes",
            "command": "PYTHONPATH=src python3 -m glossary_probes",
            "rationale": "Glossary probes must be derived from corpus/glossary.json before promotion.",
        }
    if "prompt_leakage" in name:
        return {
            "action": "remove_protected_prompt_leakage",
            "command": None,
            "rationale": "Protected held-out prompts must not appear as training lessons.",
        }
    if name == "forgetting_audit":
        return {
            "action": "inspect_regressive_eval",
            "command": None,
            "rationale": "A current eval regressed against the previous promoted report.",
        }
    if name == "exact_eval_audit":
        return {
            "action": "inspect_failed_records",
            "command": None,
            "rationale": "Every eval must be non-empty and exact before promotion.",
        }
    return {
        "action": "inspect_failed_gate",
        "command": None,
        "rationale": f"The promotion gate check {name!r} failed.",
    }


def eval_action(blocker: dict[str, Any]) -> dict[str, Any]:
    component = blocker["component"]
    eval_name = blocker["eval"]
    target = blocker.get("target")
    prediction = blocker.get("prediction")

    if eval_name == "paraphrases" and target == " unknown." and prediction != " unknown.":
        return {
            "action": "add_or_rebalance_unknown_bridge_lessons",
            "command": None,
            "rationale": "The learner overgeneralized from known facts into an unknown paraphrase surface form.",
        }
    if eval_name in {"self", "learning"}:
        return {
            "action": f"rebalance_{component}_self_learning_lessons",
            "command": None,
            "rationale": "Operational self and learning-policy answers must stay stable as the corpus grows.",
        }
    if eval_name == "glossary":
        return {
            "action": f"rebalance_{component}_glossary_lessons",
            "command": None,
            "rationale": "Glossary definitions should remain learnable from the admitted glossary.",
        }
    if eval_name in {"admissions", "admission_paraphrases"}:
        return {
            "action": "verify_admitted_memory_lesson_coverage",
            "command": None,
            "rationale": "Admitted memories need direct, paraphrase, and training-data answer coverage.",
        }
    if eval_name == "heldout":
        return {
            "action": "strengthen_non_heldout_bridge_transfer",
            "command": None,
            "rationale": "Held-out facts may use corpus-derived bridge lessons, but not exact protected prompts.",
        }
    return {
        "action": "inspect_failed_eval_record",
        "command": None,
        "rationale": f"{component} failed {eval_name}; improve only with admitted or corpus-derived lessons.",
    }


def unique_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for action in actions:
        key = action["action"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    return unique


def diagnose_report(report: dict[str, Any]) -> dict[str, Any]:
    blockers = [*failed_gate_checks(report), *failed_eval_records(report)]
    actions = unique_actions([action_for_blocker(blocker) for blocker in blockers])
    if not blockers:
        actions = [
            {
                "action": "promote_or_expand_corpus",
                "command": None,
                "rationale": "The report passed; it can become the next comparison baseline or seed a new corpus expansion.",
            }
        ]
    return {
        "schema_version": 1,
        "uses_external_model": False,
        "rule_source": "self_diagnose",
        "promotion_gate_passed": report.get("promotion_gate", {}).get("passed", False),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "recommended_actions": actions,
    }


def read_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--json",
        nargs="?",
        const=Path("-"),
        default=None,
        type=Path,
        help="Optionally also write the diagnosis JSON to a path; stdout is always JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    diagnosis = diagnose_report(read_report(args.report))
    if args.json and args.json != Path("-"):
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(diagnosis, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(diagnosis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
