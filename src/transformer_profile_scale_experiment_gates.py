"""Profile-scale direct-answer experiment acceptance gate dispatch."""

from __future__ import annotations

from typing import Any

from transformer_profile_scale_gate_catalog import profile_scale_catalog_gate
from transformer_profile_scale_memory_gates import profile_scale_memory_gate


def profile_scale_experiment_gates(direct_answer_mode: str) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    catalog_gate = profile_scale_catalog_gate(direct_answer_mode)
    if catalog_gate is not None:
        gates.append(catalog_gate)
    memory_gate = profile_scale_memory_gate(direct_answer_mode)
    if memory_gate is not None:
        gates.append(memory_gate)
    return gates
