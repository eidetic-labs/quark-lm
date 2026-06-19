from __future__ import annotations

import copy


def runtime_report(runtime: dict, *, training_allowed: bool) -> dict:
    checks = runtime_checks(runtime)
    failed = [check["name"] for check in checks if check["passed"] is not True]
    return {
        "schema_version": 1,
        "kind": "transformer_torch_runtime_report",
        "runtime": copy.deepcopy(runtime),
        "passed": training_allowed,
        "status": (
            "ready_for_pytorch_parity"
            if training_allowed
            else "blocked_test_double_runtime"
        ),
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "passed_check_count": len(checks) - len(failed),
            "failed_checks": failed,
        },
        "evidence_scope": "runtime_preflight_only",
        "parity_attempt_allowed": training_allowed,
        "training_evidence_allowed": training_allowed,
        "closed_world_boundary": {
            "runtime_library_allowed": True,
            "learned_assets_imported": False,
            "training_data_imported": False,
            "pretrained_weights_imported": False,
            "pretrained_tokenizer_imported": False,
            "external_embeddings_imported": False,
        },
        "reason": (
            "real PyTorch runtime is available for parity attempts"
            if training_allowed
            else "test doubles can validate wiring but not training parity"
        ),
    }


def runtime(*, runtime_kind: str) -> dict:
    return {
        "available": True,
        "runtime_kind": runtime_kind,
        "device": "cpu",
        "dtype": "float32",
        "dtype_available": True,
    }


def runtime_checks(runtime: dict) -> list[dict]:
    return [
        {
            "name": "runtime_available",
            "passed": bool(runtime["available"]),
            "actual": bool(runtime["available"]),
        },
        {
            "name": "runtime_kind",
            "passed": runtime["runtime_kind"] == "pytorch",
            "expected": "pytorch",
            "actual": runtime["runtime_kind"],
        },
        {
            "name": "dtype_available",
            "passed": bool(runtime["dtype_available"]),
            "actual": bool(runtime["dtype_available"]),
        },
    ]
