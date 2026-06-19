"""Write optional PyTorch training parity attempt artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from corpus_artifacts import write_json_artifact
from transformer_torch_training_parity_attempt_validation import (
    validate_torch_training_parity_attempt,
)


def write_torch_training_parity_attempt(
    output_dir: Path,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Write all attempt artifacts and return the path-enriched summary."""

    paths = {
        "fixture": output_dir / "scalar_training_fixture.json",
        "candidate": output_dir / "torch_training_candidate.json",
        "report": output_dir / "training_parity_report.json",
        "attempt": output_dir / "torch_training_parity_attempt.json",
    }
    attempt = {
        **artifacts["attempt"],
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    validate_torch_training_parity_attempt(attempt, require_artifacts=True)
    write_json_artifact(paths["fixture"], artifacts["fixture"])
    write_json_artifact(paths["candidate"], artifacts["candidate"])
    write_json_artifact(paths["report"], artifacts["report"])
    write_json_artifact(paths["attempt"], attempt)
    return attempt
