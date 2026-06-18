"""Build scalar-reference fixtures for transformer training parity."""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

from transformer_backend_policy import SCALAR_BACKEND, transformer_backend_metadata
from transformer_model import (
    TRANSFORMER_ARCHITECTURE,
    OptimizationConfig,
)
from transformer_optimizer import ScalarOptimizer
from transformer_training_parity_schema import (
    TRAINING_PARITY_ABSOLUTE_TOLERANCE,
    TRAINING_PARITY_FIXTURE_KIND,
    TRAINING_PARITY_RELATIVE_TOLERANCE,
    TRAINING_PARITY_SCHEMA_VERSION,
)


def build_scalar_training_parity_fixture(
    *,
    fixture_id: str,
    model: Any,
    tokenizer: Any,
    context: list[int],
    target: int,
    optimizer_config: OptimizationConfig,
    learning_rate: float,
    steps: int,
    corpus_hash: str,
) -> dict[str, Any]:
    """Capture scalar training behavior a future backend must match."""

    if not fixture_id:
        raise ValueError("fixture_id is required")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    initial_payload = copy.deepcopy(model.to_dict())
    trained_model, _tokenizer = type(model).from_dict(initial_payload)
    optimizer = ScalarOptimizer(optimizer_config)
    trained_model.active_optimizer = optimizer
    initial_logits = trained_model._forward_floats(context)
    initial_loss = trained_model.nll(context, target)
    step_records = [
        _training_step(trained_model, optimizer, context, target, learning_rate, step)
        for step in range(1, steps + 1)
    ]
    final_logits = trained_model._forward_floats(context)
    final_loss = trained_model.nll(context, target)
    fixture = {
        "schema_version": TRAINING_PARITY_SCHEMA_VERSION,
        "kind": TRAINING_PARITY_FIXTURE_KIND,
        "fixture_id": fixture_id,
        "architecture": TRANSFORMER_ARCHITECTURE,
        "reference_backend": transformer_backend_metadata(
            active_backend=SCALAR_BACKEND,
            seed=model.config.seed,
            tokenizer_type=getattr(tokenizer, "tokenizer_type", "char"),
            corpus_hash=corpus_hash,
        ),
        "model_config": asdict(model.config),
        "tokenizer": _tokenizer_summary(tokenizer),
        "initial_weights": initial_payload["weights"],
        "optimizer_config": asdict(optimizer_config),
        "tolerance": {
            "absolute": TRAINING_PARITY_ABSOLUTE_TOLERANCE,
            "relative": TRAINING_PARITY_RELATIVE_TOLERANCE,
        },
        "training_case": {
            "case_id": "training-01",
            "context": list(context),
            "target": target,
            "learning_rate": learning_rate,
            "steps": steps,
            "initial_loss": initial_loss,
            "initial_logits": initial_logits,
            "step_records": step_records,
            "final_loss": final_loss,
            "final_logits": final_logits,
            "optimizer_state": optimizer.to_dict(),
            "parameter_signature": _parameter_signature(
                trained_model.to_dict()["weights"]
            ),
        },
    }
    validate_training_parity_fixture(fixture)
    return fixture


def validate_training_parity_fixture(fixture: dict[str, Any]) -> None:
    if fixture.get("schema_version") != TRAINING_PARITY_SCHEMA_VERSION:
        raise ValueError("unsupported training parity fixture schema_version")
    if fixture.get("kind") != TRAINING_PARITY_FIXTURE_KIND:
        raise ValueError(f"kind must be {TRAINING_PARITY_FIXTURE_KIND}")
    if fixture.get("architecture") != TRANSFORMER_ARCHITECTURE:
        raise ValueError(f"architecture must be {TRANSFORMER_ARCHITECTURE}")
    if fixture.get("reference_backend", {}).get("backend") != SCALAR_BACKEND:
        raise ValueError("reference_backend must be scalar_python")
    if not isinstance(fixture.get("initial_weights"), dict):
        raise ValueError("initial_weights must be a dict")
    if not isinstance(fixture.get("optimizer_config"), dict):
        raise ValueError("optimizer_config must be a dict")
    _validate_training_case(fixture.get("training_case"))


def _training_step(
    model: Any,
    optimizer: ScalarOptimizer,
    context: list[int],
    target: int,
    learning_rate: float,
    step: int,
) -> dict[str, Any]:
    loss = model.train_step(context, target, learning_rate)
    return {
        "step": step,
        "loss": loss,
        "optimizer_summary": optimizer.summary(),
    }


def _tokenizer_summary(tokenizer: Any) -> dict[str, Any]:
    return {
        "tokenizer_type": getattr(tokenizer, "tokenizer_type", "char"),
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "tokens": list(getattr(tokenizer, "tokens", [])),
    }


def _parameter_signature(weights: dict[str, Any]) -> dict[str, float | int]:
    values = list(_numbers(weights))
    return {
        "count": len(values),
        "sum": sum(values),
        "abs_sum": sum(abs(value) for value in values),
        "square_sum": sum(value * value for value in values),
    }


def _numbers(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, int | float):
        yield float(value)


def _validate_training_case(case: Any) -> None:
    if not isinstance(case, dict):
        raise ValueError("training_case must be a dict")
    required = {
        "case_id",
        "context",
        "target",
        "learning_rate",
        "steps",
        "initial_loss",
        "initial_logits",
        "step_records",
        "final_loss",
        "final_logits",
        "optimizer_state",
        "parameter_signature",
    }
    for key in required:
        if key not in case:
            raise ValueError(f"training_case missing {key}")
