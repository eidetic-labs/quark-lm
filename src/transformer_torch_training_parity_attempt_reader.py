"""Load and validate written PyTorch training parity attempt artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from corpus_artifacts import read_json
from transformer_torch_training_parity_attempt_artifact_set import (
    validate_torch_training_parity_attempt_artifact_set,
)


TORCH_TRAINING_PARITY_ATTEMPT_FILES = {
    "fixture": "scalar_training_fixture.json",
    "candidate": "torch_training_candidate.json",
    "report": "training_parity_report.json",
    "attempt": "torch_training_parity_attempt.json",
}


def load_torch_training_parity_attempt_artifact_set(
    output_dir: Path,
) -> dict[str, Any]:
    """Load a written attempt artifact set and verify its persisted contract."""

    paths = _artifact_paths(output_dir)
    artifacts = {
        name: _read_required_json(path, name) for name, path in paths.items()
    }
    _validate_attempt_paths(artifacts["attempt"], paths)
    validate_torch_training_parity_attempt_artifact_set(
        artifacts,
        require_artifact_paths=True,
        require_artifact_hashes=True,
    )
    return artifacts


def _read_required_json(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"artifacts.{name} path is missing")
    payload = read_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"artifacts.{name} must be a non-empty JSON object")
    return payload


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        name: output_dir / filename
        for name, filename in TORCH_TRAINING_PARITY_ATTEMPT_FILES.items()
    }


def _validate_attempt_paths(attempt: dict[str, Any], paths: dict[str, Path]) -> None:
    recorded = attempt.get("artifacts")
    if not isinstance(recorded, dict):
        raise ValueError("attempt.artifacts must be a dict")
    for name, path in paths.items():
        if Path(str(recorded.get(name))).resolve() != path.resolve():
            raise ValueError(f"attempt.artifacts.{name} path is inconsistent")
