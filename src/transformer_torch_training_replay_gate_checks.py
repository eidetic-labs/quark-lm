"""Check builders for PyTorch training replay parity gates."""

from __future__ import annotations

from typing import Any


def probe_status(probes: dict[str, Any], name: str) -> Any:
    probe = probes.get(name, {})
    return probe.get("status") if isinstance(probe, dict) else None


def build_bool_check(name: str, value: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(value), "actual": bool(value)}


def build_status_check(name: str, actual: Any, expected: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": actual == expected,
        "expected": expected,
        "actual": actual,
    }


def build_replay_control_count_check(
    name: str,
    probe: dict[str, Any],
    expected_schema_version: int,
) -> dict[str, Any]:
    schema_version = probe.get("schema_version")
    counts = {
        "match_count": _count_field(probe, "gradient_signature_match_count"),
        "mismatch_count": _count_field(probe, "gradient_signature_mismatch_count"),
        "planned_count": _count_field(probe, "planned_microstep_count"),
        "executed_count": _count_field(probe, "executed_microstep_count"),
        "backward_count": _count_field(probe, "backward_pass_count"),
        "microstep_count": _microstep_count(probe),
    }
    count_values = {key: value for key, (value, _) in counts.items()}
    count_types_valid = all(valid for _, valid in counts.values())
    passed = (
        schema_version == expected_schema_version
        and count_types_valid
        and count_values["planned_count"] > 0
        and count_values["executed_count"] == count_values["planned_count"]
        and count_values["backward_count"] == count_values["planned_count"]
        and count_values["match_count"] == count_values["planned_count"]
        and count_values["mismatch_count"] == 0
        and count_values["microstep_count"] == count_values["planned_count"]
    )
    return {
        "name": name,
        "passed": passed,
        "schema_version": schema_version,
        "expected_schema_version": expected_schema_version,
        "count_types_valid": count_types_valid,
        **count_values,
    }


def build_replay_probe_check(
    name: str,
    probe: dict[str, Any],
    expected_status: str,
    expected_schema_version: int,
    required_proof_flags: list[str],
) -> dict[str, Any]:
    actual_status = probe.get("status")
    schema_version = probe.get("schema_version")
    proof_flags = {
        flag: probe.get(flag) is True
        for flag in required_proof_flags
    }
    return {
        "name": name,
        "passed": (
            bool(probe.get("passed"))
            and actual_status == expected_status
            and schema_version == expected_schema_version
            and all(proof_flags.values())
        ),
        "expected": expected_status,
        "status": actual_status,
        "schema_version": schema_version,
        "expected_schema_version": expected_schema_version,
        "proof_flags": proof_flags,
    }


def _count_field(probe: dict[str, Any], name: str) -> tuple[int, bool]:
    value = probe.get(name)
    valid = type(value) is int and value >= 0
    return (value if valid else 0, valid)


def _microstep_count(probe: dict[str, Any]) -> tuple[int, bool]:
    microsteps = probe.get("microsteps", [])
    return (len(microsteps), isinstance(microsteps, list))
