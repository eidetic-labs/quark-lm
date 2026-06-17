"""Baseline-floor direct-answer tuning defaults."""

from __future__ import annotations


BASELINE_FLOOR_ADAPTIVE_LEARNING_RATE_SCALES = (1.0, 0.25, 0.05, 0.01)
BASELINE_FLOOR_CALIBRATED_ADAPTIVE_LEARNING_RATE_SCALES = (
    1.0,
    0.25,
    0.05,
    0.01,
    0.0025,
    0.0005,
    0.0001,
)
BASELINE_FLOOR_COVERAGE_RECOVERY_LEARNING_RATE_SCALES = (1.0, 0.25, 0.05)
BASELINE_FLOOR_BRANCH_DIVERSITY_RECOVERY_LEARNING_RATE_SCALES = (0.25, 0.05, 0.01)
BASELINE_FLOOR_COLLAPSED_PROFILE_BINDING_LEARNING_RATE_SCALES = (0.25, 0.05, 0.01)
BASELINE_FLOOR_MISSING_FIRST_TOKEN_LEARNING_RATE_SCALES = (0.25, 0.05, 0.01)
BASELINE_FLOOR_REMAINING_PROFILE_BINDING_TARGET_PROFILES = (
    "learning",
    "owner",
    "paraphrases",
)
BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES = (
    "owner",
    "paraphrases",
)
BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES = ("learning",)
BASELINE_FLOOR_REMAINING_PROFILE_BINDING_PARAPHRASE_SOURCE_LABELS = (
    "color",
    "owner",
    "place",
    "training_data",
)
BASELINE_FLOOR_REMAINING_PROFILE_BINDING_TARGET_SOURCE_LABELS = {
    "admission_paraphrases": ("color", "owner", "place", "training_data"),
    "admissions": ("color", "owner", "place", "training_data"),
    "glossary": ("glossary",),
    "heldout": ("color", "owner", "place"),
    "learning": ("learning",),
    "owner": ("owner",),
    "paraphrases": BASELINE_FLOOR_REMAINING_PROFILE_BINDING_PARAPHRASE_SOURCE_LABELS,
    "qa": ("color", "owner", "place"),
    "self": ("self",),
}
BASELINE_FLOOR_REPAIR_STEPS = 1
BASELINE_FLOOR_OBJECTIVE_ANCHOR_BATCH_SIZE = 32
BASELINE_FLOOR_OBJECTIVE_ANCHOR_WEIGHT = 10.0
BASELINE_FLOOR_STABILIZATION_ANCHOR_BATCH_SIZE = 32
