"""In-loop combined does-both eval bridge for the torch mixed-objective trainer.

Bridges the live torch training state back to a TinyTransformerLM (the SAME
torch_trained_weights + from_dict bridge the answer-train stage uses) and scores it
against a validation slice via the epistemic spine, so the training loop can pick the
checkpoint that BOTH abstains and generates concrete answers. The eval MUST run with a
CorpusResponder (per-type menus) -- without one, run_epistemic_eval falls back to the
contaminated global candidate pool and the in-loop F1 is poisoned, so the provenance is
asserted per_type and a global_pool result fails closed.
"""

from __future__ import annotations

from typing import Any, Callable

from epistemic_eval_runner import run_epistemic_eval
from transformer_tiny_lm import TinyTransformerLM
from transformer_torch_training_loop import torch_trained_weights


def evaluate_combined_does_both(
    *,
    fixture: dict[str, Any],
    state: dict[str, Any],
    tokenizer: Any,
    torch: Any,
    validation_probe_paths: list[Any],
    eval_responder: Any,
    max_new_chars: int,
) -> tuple[float, float]:
    """Score the live state on the validation slice; return (abstention_f1, concrete_gen).

    Runs under torch.no_grad(); raises if the report's candidate-menu provenance is
    the contaminated global pool rather than the de-contaminated per-type menus.
    """

    with torch.no_grad():
        weights = torch_trained_weights(fixture=fixture, state=state)
        model, _ = TinyTransformerLM.from_dict(
            {"config": fixture["model_config"], "weights": weights}
        )
        report = run_epistemic_eval(
            model=model,
            tokenizer=tokenizer,
            probe_paths=validation_probe_paths,
            responder=eval_responder,
            max_new_chars=max_new_chars,
        )
    return _read_does_both(report)


def _read_does_both(report: dict[str, Any]) -> tuple[float, float]:
    menus = report.get("provenance", {}).get("candidate_menus")
    if menus != "per_type":
        raise ValueError(
            "in-loop combined eval requires per_type candidate menus "
            f"(got {menus!r}); pass a CorpusResponder so the F1 is not poisoned "
            "by the contaminated global candidate pool"
        )
    head = report["headline"]
    abstention_f1 = head.get("abstention_f1") or 0.0
    concrete_gen = head.get("concrete_generation_exact_rate") or 0.0
    return float(abstention_f1), float(concrete_gen)


# Type of the optional in-loop eval seam (tests inject a deterministic report).
CombinedEvalFn = Callable[[int], dict[str, Any]]
