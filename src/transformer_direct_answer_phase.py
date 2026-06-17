"""Compatibility facade for the direct-answer phase workflow."""

from __future__ import annotations

from transformer_direct_answer_phase_completion import complete_direct_answer_phase
from transformer_direct_answer_phase_loop import run_direct_answer_training_loop
from transformer_direct_answer_phase_setup import initialize_direct_answer_phase
from transformer_direct_answer_phase_types import (
    DirectAnswerLoopResult,
    DirectAnswerPhaseResult,
    DirectAnswerPhaseRuntime,
)


__all__ = [
    "DirectAnswerLoopResult",
    "DirectAnswerPhaseResult",
    "DirectAnswerPhaseRuntime",
    "complete_direct_answer_phase",
    "initialize_direct_answer_phase",
    "run_direct_answer_training_loop",
]
