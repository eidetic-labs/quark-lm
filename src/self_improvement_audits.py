"""Compatibility exports for self-improvement audits and promotion checks."""

from __future__ import annotations

from self_improvement_forgetting_audit import (
    audit_forgetting,
    component_final_evals,
    read_report,
)
from self_improvement_promotion_audit import audit_exact_promotion, promotion_gate
from self_improvement_prompt_leakage import (
    PROJECT_DIR,
    audit_all_protected_prompts,
    audit_prompt_leakage,
)
from self_improvement_responder_eval import evaluate_responder, summarize_exact


__all__ = [
    "PROJECT_DIR",
    "audit_all_protected_prompts",
    "audit_exact_promotion",
    "audit_forgetting",
    "audit_prompt_leakage",
    "component_final_evals",
    "evaluate_responder",
    "promotion_gate",
    "read_report",
    "summarize_exact",
]
