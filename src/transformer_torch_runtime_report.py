"""Preflight report for optional PyTorch parity evidence."""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from corpus_artifacts import SCHEMA_VERSION, write_json_artifact
from curriculum import PROJECT_DIR
from transformer_torch_runtime import (
    TORCH_RUNTIME_KIND_PYTORCH,
    TORCH_RUNTIME_KIND_TEST_DOUBLE,
    TorchImporter,
    torch_runtime_status,
)


DEFAULT_OUTPUT = PROJECT_DIR / "build" / "torch_runtime_report.json"
TORCH_RUNTIME_REPORT_KIND = "transformer_torch_runtime_report"


def build_torch_runtime_report(
    *,
    importer: TorchImporter = import_module,
    requested_device: str = "auto",
    requested_dtype: str = "float32",
) -> dict[str, Any]:
    """Build JSON-safe evidence for optional PyTorch parity attempts."""

    runtime = torch_runtime_status(
        importer=importer,
        requested_device=requested_device,
        requested_dtype=requested_dtype,
    )
    checks = _checks(runtime)
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": TORCH_RUNTIME_REPORT_KIND,
        "status": _status(runtime, passed),
        "passed": passed,
        "runtime": runtime,
        "checks": checks,
        "summary": _summary(checks),
        "evidence_scope": "runtime_preflight_only",
        "parity_attempt_allowed": passed,
        "closed_world_boundary": {
            "runtime_library_allowed": True,
            "learned_assets_imported": False,
            "training_data_imported": False,
            "pretrained_weights_imported": False,
            "pretrained_tokenizer_imported": False,
            "external_embeddings_imported": False,
        },
        "training_evidence_allowed": passed,
        "reason": _reason(runtime, passed),
    }


def write_torch_runtime_report(path: Path, report: dict[str, Any]) -> None:
    write_json_artifact(path, report)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requested-device", default="auto")
    parser.add_argument("--requested-dtype", default="float32")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_torch_runtime_report(
        requested_device=args.requested_device,
        requested_dtype=args.requested_dtype,
    )
    write_torch_runtime_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def _checks(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _bool_check("runtime_available", runtime.get("available")),
        _status_check(
            "runtime_kind",
            runtime.get("runtime_kind"),
            TORCH_RUNTIME_KIND_PYTORCH,
        ),
        _bool_check("dtype_available", runtime.get("dtype_available")),
    ]


def _bool_check(name: str, value: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(value), "actual": bool(value)}


def _status_check(name: str, actual: Any, expected: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": actual == expected,
        "expected": expected,
        "actual": actual,
    }


def _summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [check["name"] for check in checks if check["passed"] is not True]
    return {
        "check_count": len(checks),
        "passed_check_count": len(checks) - len(failed),
        "failed_checks": failed,
    }


def _status(runtime: dict[str, Any], passed: bool) -> str:
    if passed:
        return "ready_for_pytorch_parity"
    if not runtime.get("available"):
        return "blocked_runtime_unavailable"
    if runtime.get("runtime_kind") == TORCH_RUNTIME_KIND_TEST_DOUBLE:
        return "blocked_test_double_runtime"
    if not runtime.get("dtype_available"):
        return "blocked_dtype_unavailable"
    return "blocked_pytorch_runtime"


def _reason(runtime: dict[str, Any], passed: bool) -> str:
    if passed:
        return "real PyTorch runtime is available for parity attempts"
    if not runtime.get("available"):
        return "PyTorch is not installed in this environment"
    if runtime.get("runtime_kind") == TORCH_RUNTIME_KIND_TEST_DOUBLE:
        return "test doubles can validate wiring but not training parity"
    if not runtime.get("dtype_available"):
        return "requested PyTorch dtype is unavailable"
    return "PyTorch runtime did not satisfy parity preflight checks"


if __name__ == "__main__":
    raise SystemExit(main())
