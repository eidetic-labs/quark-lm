"""Compatibility exports for training recipe and constraint reports."""

from __future__ import annotations

from constraint_first_report import (
    CONSTRAINT_REPORT_KIND,
    QUALITY_POLICY,
    build_constraint_first_promotion_report,
    constraint_first_summary,
    promotion_check,
    validate_constraint_first_promotion_report,
    write_constraint_first_report,
)
from self_improvement_constraints import self_improvement_constraint_report
from training_recipe_core import (
    TRAINING_RECIPE_KIND,
    attach_recipe_summary,
    build_training_recipe,
    training_recipe_summary,
    validate_training_recipe,
    write_training_recipe,
)
from training_recipe_validation import SCHEMA_VERSION
from transformer_constraints import transformer_constraint_report

__all__ = [
    "CONSTRAINT_REPORT_KIND",
    "QUALITY_POLICY",
    "SCHEMA_VERSION",
    "TRAINING_RECIPE_KIND",
    "attach_recipe_summary",
    "build_constraint_first_promotion_report",
    "build_training_recipe",
    "constraint_first_summary",
    "promotion_check",
    "self_improvement_constraint_report",
    "training_recipe_summary",
    "transformer_constraint_report",
    "validate_constraint_first_promotion_report",
    "validate_training_recipe",
    "write_constraint_first_report",
    "write_training_recipe",
]
