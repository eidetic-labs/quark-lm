"""Dependency-free tiny decoder-only transformer language model.

The implementation is intentionally small and auditable. It uses learned token
and position embeddings, one causal self-attention block, a feed-forward block,
and a next-character language-model head. All weights start from random values;
the tokenizer is trained from admitted corpus text.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .autograd import Scalar, zero_grad
from .answer_model import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_EVALS as DEFAULT_ANSWER_EVALS,
    DEFAULT_TRAIN_TEXT,
    AnswerExample,
    answer_training_pool,
    feature_names,
    load_training_examples,
    semantic_feature_names,
    write_lessons,
)
from .candidate_quarantine import (
    build_candidate_quarantine_manifest,
    candidate_quarantine_summary,
    write_candidate_quarantine,
)
from .closed_world_verifier import (
    attach_verifier_summary,
    verify_training_plan,
    write_verifier_report,
)
from .curriculum import DEFAULT_OUTPUT_DIR, build_curriculum, write_curriculum
from .corpus_hygiene import (
    attach_replay_plan_summary,
    build_corpus_hygiene_report,
    build_training_plan,
    write_json_artifact,
)
from .experiment_registry import (
    record_experiment_decision,
    write_experiment_intent,
)
from .memory_consolidation import (
    build_memory_consolidation_plan,
    write_memory_consolidation_plan,
)
from .memory_retrieval import (
    build_retrieval_memory_report,
    write_retrieval_memory_report,
)
from .neural_char_model import context_before, continuation_nll, make_context
from .probes import read_jsonl, score_records, summarize
from .replay_plan import (
    BranchReplayRecord,
    ProfiledBranchSeed,
    branch_replay_parts,
    branch_replay_plan,
    branch_replay_profile_groups,
    direct_answer_profile_key,
)
from .tokenizer import CharTokenizer
from .training_recipe import (
    attach_recipe_summary,
    transformer_constraint_report,
    write_constraint_first_report,
    write_training_recipe,
)
from .transformer_checkpoint import load_checkpoint_payload
from .transformer_eval import (
    build_transformer_eval_report,
    eval_candidates_from_records,
    load_probe_records,
    score_transformer_evals,
    score_transformer_records,
    write_eval_report,
    write_eval_samples,
)
from .transformer_experiment import (
    TRAINING_DATA_DESCRIPTION,
    TRANSFORMER_RECIPE_VERSION,
    TransformerRunArtifacts,
    direct_answer_is_profile_aware,
    is_profile_aware_direct_answer_mode,
    parse_experiment_gate,
    transformer_experiment_acceptance_gates,
    transformer_experiment_decision,
    transformer_experiment_intent,
    transformer_training_recipe as build_transformer_training_recipe,
    transformer_training_recipe_id,
)
from .transformer_model import (
    TRANSFORMER_ARCHITECTURE,
    TRANSFORMER_CHECKPOINT_FORMAT,
    TRANSFORMER_TOKENIZER,
    GenerationConfig,
    OptimizationConfig,
    TransformerConfig,
    checkpoint_header,
    generation_config_from_args,
    optimization_config_from_args,
    transformer_config_from_args,
    transformer_run_metadata,
    validate_generation_config,
    validate_optimization_config,
    validate_transformer_config,
)
from .transformer_objectives import (
    DIRECT_ANSWER_OBJECTIVE_MODES,
)
from .transformer_training import (
    JsonlHistoryWriter,
    LossAccumulator,
    ShuffledTrainingCursor,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PROJECT_DIR / "runs" / "transformer-latest"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "transformer.json"
DEFAULT_PROBES = [
    PROJECT_DIR / "evals" / "qa.jsonl",
    PROJECT_DIR / "evals" / "unknowns.jsonl",
    PROJECT_DIR / "evals" / "heldout.jsonl",
    PROJECT_DIR / "evals" / "paraphrases.jsonl",
]
ANSWER_TERMINATOR = "\n"
ReplayPredictionOverrides = dict[tuple[tuple[int, ...], int, str], int]
BASELINE_ANCHORED_PROMPT_MODE = (
    "branch-balanced-context-profile-baseline-anchored-prompt-ownership-"
    "target-share-preserving-deficit-unlikelihood"
)
BASELINE_FLOOR_GATED_PROMPT_MODE = (
    "branch-balanced-context-profile-baseline-floor-gated-prompt-ownership-"
    "target-share-preserving-deficit-unlikelihood"
)
BASELINE_FLOOR_ADAPTIVE_PROMPT_MODE = (
    "branch-balanced-context-profile-baseline-floor-adaptive-prompt-ownership-"
    "target-share-preserving-deficit-unlikelihood"
)
BASELINE_FLOOR_REPAIRED_PROMPT_MODE = (
    "branch-balanced-context-profile-baseline-floor-repaired-prompt-ownership-"
    "target-share-preserving-deficit-unlikelihood"
)
BASELINE_FLOOR_OBJECTIVE_PROMPT_MODE = (
    "branch-balanced-context-profile-baseline-floor-objective-prompt-ownership-"
    "target-share-preserving-deficit-unlikelihood"
)
BASELINE_FLOOR_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_TARGETED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-profile-targeted-stabilization-unlikelihood"
)
BASELINE_FLOOR_SEQUENTIAL_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_CALIBRATED_SEQUENTIAL_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-calibrated-sequential-profile-"
    "stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-profile-scale-calibrated-sequential-"
    "profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-profile-scale-calibrated-"
    "sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-frontier-profile-scale-"
    "calibrated-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-coverage-frontier-profile-scale-"
    "calibrated-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-coverage-prep-frontier-"
    "profile-scale-calibrated-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-coverage-recovery-frontier-"
    "profile-scale-calibrated-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-frontier-profile-scale-calibrated-sequential-profile-"
    "stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-frontier-profile-scale-calibrated-sequential-"
    "profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-frontier-profile-scale-"
    "calibrated-sequential-profile-stabilization-unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "frontier-profile-scale-calibrated-sequential-profile-stabilization-"
    "unlikelihood"
)
BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE = (
    "branch-context-profile-baseline-floor-diversity-branch-stable-coverage-"
    "recovery-branch-diversity-collapsed-profile-binding-remaining-profile-"
    "owner-paraphrase-frontier-profile-scale-calibrated-sequential-profile-"
    "stabilization-unlikelihood"
)
BASELINE_ANCHORED_DIRECT_ANSWER_MODES = {
    BASELINE_ANCHORED_PROMPT_MODE,
    BASELINE_FLOOR_GATED_PROMPT_MODE,
    BASELINE_FLOOR_ADAPTIVE_PROMPT_MODE,
    BASELINE_FLOOR_REPAIRED_PROMPT_MODE,
    BASELINE_FLOOR_OBJECTIVE_PROMPT_MODE,
    BASELINE_FLOOR_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_TARGETED_STABILIZATION_MODE,
    BASELINE_FLOOR_SEQUENTIAL_STABILIZATION_MODE,
    BASELINE_FLOOR_CALIBRATED_SEQUENTIAL_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_GATED_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_GATED_PROMPT_MODE,
    BASELINE_FLOOR_ADAPTIVE_PROMPT_MODE,
    BASELINE_FLOOR_REPAIRED_PROMPT_MODE,
    BASELINE_FLOOR_OBJECTIVE_PROMPT_MODE,
    BASELINE_FLOOR_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_TARGETED_STABILIZATION_MODE,
    BASELINE_FLOOR_SEQUENTIAL_STABILIZATION_MODE,
    BASELINE_FLOOR_CALIBRATED_SEQUENTIAL_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_ADAPTIVE_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_ADAPTIVE_PROMPT_MODE,
    BASELINE_FLOOR_REPAIRED_PROMPT_MODE,
    BASELINE_FLOOR_OBJECTIVE_PROMPT_MODE,
    BASELINE_FLOOR_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_TARGETED_STABILIZATION_MODE,
    BASELINE_FLOOR_SEQUENTIAL_STABILIZATION_MODE,
    BASELINE_FLOOR_CALIBRATED_SEQUENTIAL_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_REPAIRED_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_REPAIRED_PROMPT_MODE,
}
BASELINE_FLOOR_OBJECTIVE_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_OBJECTIVE_PROMPT_MODE,
}
BASELINE_FLOOR_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_TARGETED_STABILIZATION_MODE,
    BASELINE_FLOOR_SEQUENTIAL_STABILIZATION_MODE,
    BASELINE_FLOOR_CALIBRATED_SEQUENTIAL_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_PROFILE_TARGETED_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_PROFILE_TARGETED_STABILIZATION_MODE,
}
BASELINE_FLOOR_SEQUENTIAL_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_SEQUENTIAL_STABILIZATION_MODE,
    BASELINE_FLOOR_CALIBRATED_SEQUENTIAL_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_CALIBRATED_SEQUENTIAL_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_CALIBRATED_SEQUENTIAL_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_PROFILE_SCALE_CALIBRATED_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_PROFILE_SCALE_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES = {
    BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_CALIBRATED_STABILIZATION_MODE,
}
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
BASELINE_FLOOR_REPAIR_STEPS = 1
BASELINE_FLOOR_OBJECTIVE_ANCHOR_BATCH_SIZE = 32
BASELINE_FLOOR_OBJECTIVE_ANCHOR_WEIGHT = 10.0
BASELINE_FLOOR_STABILIZATION_ANCHOR_BATCH_SIZE = 32


class ScalarOptimizer:
    def __init__(
        self,
        config: OptimizationConfig | None = None,
        update_count: int = 0,
        first_moment: list[float] | None = None,
        second_moment: list[float] | None = None,
        gradient_buffer: list[float] | None = None,
        pending_accumulation: int = 0,
    ) -> None:
        self.config = config or OptimizationConfig()
        self.update_count = update_count
        self.first_moment = first_moment or []
        self.second_moment = second_moment or []
        self.gradient_buffer = gradient_buffer or []
        self.pending_accumulation = pending_accumulation
        validate_optimization_config(self.config)

    def effective_learning_rate(self, base_learning_rate: float, next_step: int | None = None) -> float:
        step = self.update_count + 1 if next_step is None else next_step
        learning_rate = base_learning_rate
        if self.config.warmup_steps > 0:
            learning_rate *= min(1.0, step / self.config.warmup_steps)
        if self.config.decay_steps > 0 and step > self.config.warmup_steps:
            decay_step = min(step - self.config.warmup_steps, self.config.decay_steps)
            decay_fraction = decay_step / self.config.decay_steps
            learning_rate = learning_rate - (
                learning_rate - self.config.min_learning_rate
            ) * decay_fraction
        return max(learning_rate, self.config.min_learning_rate)

    def apply(self, params: list[Scalar], base_learning_rate: float) -> float:
        self._ensure_slots(len(params))
        for index, parameter in enumerate(params):
            self.gradient_buffer[index] += self._clipped_grad(parameter)
        self.pending_accumulation += 1
        if self.pending_accumulation < self.config.gradient_accumulation_steps:
            return self.effective_learning_rate(base_learning_rate)
        accumulated_grads = [
            value / self.pending_accumulation
            for value in self.gradient_buffer
        ]
        self.gradient_buffer = [0.0 for _ in params]
        self.pending_accumulation = 0
        self.update_count += 1
        learning_rate = self.effective_learning_rate(base_learning_rate, self.update_count)
        if self.config.optimizer == "sgd":
            self._apply_sgd(params, accumulated_grads, learning_rate)
        elif self.config.optimizer == "adamw":
            self._apply_adamw(params, accumulated_grads, learning_rate)
        else:
            raise ValueError(f"unsupported optimizer: {self.config.optimizer}")
        return learning_rate

    def _ensure_slots(self, param_count: int) -> None:
        if len(self.first_moment) != param_count:
            self.first_moment = [0.0 for _ in range(param_count)]
        if len(self.second_moment) != param_count:
            self.second_moment = [0.0 for _ in range(param_count)]
        if len(self.gradient_buffer) != param_count:
            self.gradient_buffer = [0.0 for _ in range(param_count)]

    def _clipped_grad(self, parameter: Scalar) -> float:
        clip = self.config.gradient_clip
        if clip <= 0.0:
            return parameter.grad
        return max(min(parameter.grad, clip), -clip)

    def _apply_sgd(
        self,
        params: list[Scalar],
        grads: list[float],
        learning_rate: float,
    ) -> None:
        for parameter, grad in zip(params, grads):
            if self.config.weight_decay > 0.0:
                grad += self.config.weight_decay * parameter.data
            parameter.data -= learning_rate * grad

    def _apply_adamw(
        self,
        params: list[Scalar],
        grads: list[float],
        learning_rate: float,
    ) -> None:
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        beta1_correction = 1.0 - beta1**self.update_count
        beta2_correction = 1.0 - beta2**self.update_count
        for index, (parameter, grad) in enumerate(zip(params, grads)):
            self.first_moment[index] = beta1 * self.first_moment[index] + (1.0 - beta1) * grad
            self.second_moment[index] = (
                beta2 * self.second_moment[index] + (1.0 - beta2) * grad * grad
            )
            first_unbiased = self.first_moment[index] / beta1_correction
            second_unbiased = self.second_moment[index] / beta2_correction
            if self.config.weight_decay > 0.0:
                parameter.data -= learning_rate * self.config.weight_decay * parameter.data
            parameter.data -= (
                learning_rate
                * first_unbiased
                / (math.sqrt(second_unbiased) + self.config.epsilon)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": asdict(self.config),
            "update_count": self.update_count,
            "param_count": len(self.first_moment),
            "first_moment": self.first_moment,
            "second_moment": self.second_moment,
            "gradient_buffer": self.gradient_buffer,
            "pending_accumulation": self.pending_accumulation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScalarOptimizer":
        return cls(
            OptimizationConfig(**payload.get("config", {})),
            update_count=int(payload.get("update_count", 0)),
            first_moment=[float(value) for value in payload.get("first_moment", [])],
            second_moment=[float(value) for value in payload.get("second_moment", [])],
            gradient_buffer=[float(value) for value in payload.get("gradient_buffer", [])],
            pending_accumulation=int(payload.get("pending_accumulation", 0)),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "optimizer": self.config.optimizer,
            "update_count": self.update_count,
            "param_count": len(self.first_moment),
            "pending_accumulation": self.pending_accumulation,
            "last_learning_rate": (
                self.effective_learning_rate(1.0, self.update_count)
                if self.update_count
                else None
            ),
            "gradient_clip": self.config.gradient_clip,
            "weight_decay": self.config.weight_decay,
            "warmup_steps": self.config.warmup_steps,
            "decay_steps": self.config.decay_steps,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
        }


class TinyTransformerLM:
    def __init__(self, config: TransformerConfig, weights: dict[str, Any]) -> None:
        validate_transformer_config(config)
        self.config = config
        dim = config.embedding_dim
        self.token_embeddings = matrix_to_scalars(weights["token_embeddings"])
        self.position_embeddings = matrix_to_scalars(weights["position_embeddings"])
        self.wq = matrix_to_scalars(weights["wq"])
        self.bq = vector_to_scalars(weights["bq"])
        self.wk = matrix_to_scalars(weights["wk"])
        self.bk = vector_to_scalars(weights["bk"])
        self.wv = matrix_to_scalars(weights["wv"])
        self.bv = vector_to_scalars(weights["bv"])
        self.wo = matrix_to_scalars(weights["wo"])
        self.bo = vector_to_scalars(weights["bo"])
        self.w1 = matrix_to_scalars(weights["w1"])
        self.b1 = vector_to_scalars(weights["b1"])
        self.w_gate = matrix_to_scalars(
            weights.get(
                "w_gate",
                [[0.0 for _ in range(config.feedforward_dim)] for _ in range(dim)],
            )
        )
        self.b_gate = vector_to_scalars(
            weights.get("b_gate", [0.0 for _ in range(config.feedforward_dim)])
        )
        self.w2 = matrix_to_scalars(weights["w2"])
        self.b2 = vector_to_scalars(weights["b2"])
        self.wout = matrix_to_scalars(weights["wout"])
        self.bout = vector_to_scalars(weights["bout"])
        self.context_projection_w = matrix_to_scalars(
            weights.get(
                "context_projection_w",
                [[0.0 for _ in range(dim)] for _ in range(dim)],
            )
        )
        self.context_projection_b = vector_to_scalars(
            weights.get("context_projection_b", [0.0 for _ in range(dim)])
        )
        self.prompt_prefix_projection_w = matrix_to_scalars(
            weights.get(
                "prompt_prefix_projection_w",
                [[0.0 for _ in range(dim)] for _ in range(dim)],
            )
        )
        self.prompt_prefix_projection_b = vector_to_scalars(
            weights.get("prompt_prefix_projection_b", [0.0 for _ in range(dim)])
        )
        self.prompt_position_projection_w = [
            matrix_to_scalars(position_weights)
            for position_weights in weights.get(
                "prompt_position_projection_w",
                [
                    [[0.0 for _ in range(dim)] for _ in range(dim)]
                    for _ in range(config.context_size)
                ],
            )
        ]
        self.prompt_position_projection_b = vector_to_scalars(
            weights.get("prompt_position_projection_b", [0.0 for _ in range(dim)])
        )
        self.prompt_summary_query = vector_to_scalars(
            weights.get("prompt_summary_query", [0.0 for _ in range(dim)])
        )
        self.prompt_summary_w = matrix_to_scalars(
            weights.get(
                "prompt_summary_w",
                [[0.0 for _ in range(dim)] for _ in range(dim)],
            )
        )
        self.prompt_summary_b = vector_to_scalars(
            weights.get("prompt_summary_b", [0.0 for _ in range(dim)])
        )
        self.ln1_gain = vector_to_scalars(weights.get("ln1_gain", [1.0 for _ in range(dim)]))
        self.ln1_bias = vector_to_scalars(weights.get("ln1_bias", [0.0 for _ in range(dim)]))
        self.ln2_gain = vector_to_scalars(weights.get("ln2_gain", [1.0 for _ in range(dim)]))
        self.ln2_bias = vector_to_scalars(weights.get("ln2_bias", [0.0 for _ in range(dim)]))
        self.final_ln_gain = vector_to_scalars(
            weights.get("final_ln_gain", [1.0 for _ in range(dim)])
        )
        self.final_ln_bias = vector_to_scalars(
            weights.get("final_ln_bias", [0.0 for _ in range(dim)])
        )
        self.extra_blocks = [
            self._block_from_dict(layer)
            for layer in weights.get("extra_layers", [])
        ]
        expected_extra_layers = max(config.num_layers - 1, 0)
        if len(self.extra_blocks) != expected_extra_layers:
            raise ValueError(
                f"checkpoint has {len(self.extra_blocks)} extra transformer layers, "
                f"config expects {expected_extra_layers}"
        )
        self.blocks = [self._first_block()] + self.extra_blocks
        self.freeze_lower_layers_for_updates = False
        self.active_optimizer: ScalarOptimizer | None = None

    @classmethod
    def init_random(cls, config: TransformerConfig) -> "TinyTransformerLM":
        validate_transformer_config(config)
        rng = random.Random(config.seed)

        def rand(scale: float) -> float:
            return rng.uniform(-scale, scale)

        dim = config.embedding_dim
        ff_dim = config.feedforward_dim
        scale = 1.0 / math.sqrt(dim)

        def block_weights() -> dict[str, Any]:
            return {
                "wq": [[rand(scale) for _ in range(dim)] for _ in range(dim)],
                "bq": [0.0 for _ in range(dim)],
                "wk": [[rand(scale) for _ in range(dim)] for _ in range(dim)],
                "bk": [0.0 for _ in range(dim)],
                "wv": [[rand(scale) for _ in range(dim)] for _ in range(dim)],
                "bv": [0.0 for _ in range(dim)],
                "wo": [[rand(scale) for _ in range(dim)] for _ in range(dim)],
                "bo": [0.0 for _ in range(dim)],
                "w1": [[rand(scale) for _ in range(ff_dim)] for _ in range(dim)],
                "b1": [0.0 for _ in range(ff_dim)],
                "w_gate": [
                    [rand(scale) if config.use_gated_mlp else 0.0 for _ in range(ff_dim)]
                    for _ in range(dim)
                ],
                "b_gate": [0.0 for _ in range(ff_dim)],
                "w2": [
                    [rand(1.0 / math.sqrt(ff_dim)) for _ in range(dim)]
                    for _ in range(ff_dim)
                ],
                "b2": [0.0 for _ in range(dim)],
                "ln1_gain": [1.0 for _ in range(dim)],
                "ln1_bias": [0.0 for _ in range(dim)],
                "ln2_gain": [1.0 for _ in range(dim)],
                "ln2_bias": [0.0 for _ in range(dim)],
            }

        first_block = block_weights()
        weights = {
            "token_embeddings": [
                [rand(0.08) for _ in range(dim)]
                for _ in range(config.vocab_size)
            ],
            "position_embeddings": [
                [rand(0.08) for _ in range(dim)]
                for _ in range(config.context_size)
            ],
            **first_block,
            "wout": [[rand(scale) for _ in range(config.vocab_size)] for _ in range(dim)],
            "bout": [0.0 for _ in range(config.vocab_size)],
            "context_projection_w": [[0.0 for _ in range(dim)] for _ in range(dim)],
            "context_projection_b": [0.0 for _ in range(dim)],
            "prompt_prefix_projection_w": [[0.0 for _ in range(dim)] for _ in range(dim)],
            "prompt_prefix_projection_b": [0.0 for _ in range(dim)],
            "prompt_position_projection_w": [
                [[0.0 for _ in range(dim)] for _ in range(dim)]
                for _ in range(config.context_size)
            ],
            "prompt_position_projection_b": [0.0 for _ in range(dim)],
            "prompt_summary_query": [rand(scale) for _ in range(dim)],
            "prompt_summary_w": [[0.0 for _ in range(dim)] for _ in range(dim)],
            "prompt_summary_b": [0.0 for _ in range(dim)],
            "final_ln_gain": [1.0 for _ in range(dim)],
            "final_ln_bias": [0.0 for _ in range(dim)],
            "extra_layers": [
                block_weights()
                for _ in range(max(config.num_layers - 1, 0))
            ],
        }
        return cls(config, weights)

    def _first_block(self) -> dict[str, Any]:
        return {
            "wq": self.wq,
            "bq": self.bq,
            "wk": self.wk,
            "bk": self.bk,
            "wv": self.wv,
            "bv": self.bv,
            "wo": self.wo,
            "bo": self.bo,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w_gate": self.w_gate,
            "b_gate": self.b_gate,
            "ln1_gain": self.ln1_gain,
            "ln1_bias": self.ln1_bias,
            "ln2_gain": self.ln2_gain,
            "ln2_bias": self.ln2_bias,
        }

    def _block_from_dict(self, payload: dict[str, Any]) -> dict[str, Any]:
        dim = self.config.embedding_dim
        return {
            "wq": matrix_to_scalars(payload["wq"]),
            "bq": vector_to_scalars(payload["bq"]),
            "wk": matrix_to_scalars(payload["wk"]),
            "bk": vector_to_scalars(payload["bk"]),
            "wv": matrix_to_scalars(payload["wv"]),
            "bv": vector_to_scalars(payload["bv"]),
            "wo": matrix_to_scalars(payload["wo"]),
            "bo": vector_to_scalars(payload["bo"]),
            "w1": matrix_to_scalars(payload["w1"]),
            "b1": vector_to_scalars(payload["b1"]),
            "w_gate": matrix_to_scalars(
                payload.get(
                    "w_gate",
                    [[0.0 for _ in range(self.config.feedforward_dim)] for _ in range(dim)],
                )
            ),
            "b_gate": vector_to_scalars(
                payload.get("b_gate", [0.0 for _ in range(self.config.feedforward_dim)])
            ),
            "w2": matrix_to_scalars(payload["w2"]),
            "b2": vector_to_scalars(payload["b2"]),
            "ln1_gain": vector_to_scalars(payload.get("ln1_gain", [1.0 for _ in range(dim)])),
            "ln1_bias": vector_to_scalars(payload.get("ln1_bias", [0.0 for _ in range(dim)])),
            "ln2_gain": vector_to_scalars(payload.get("ln2_gain", [1.0 for _ in range(dim)])),
            "ln2_bias": vector_to_scalars(payload.get("ln2_bias", [0.0 for _ in range(dim)])),
        }

    def _block_to_floats(self, block: dict[str, Any]) -> dict[str, Any]:
        return {
            "wq": matrix_to_floats(block["wq"]),
            "bq": vector_to_floats(block["bq"]),
            "wk": matrix_to_floats(block["wk"]),
            "bk": vector_to_floats(block["bk"]),
            "wv": matrix_to_floats(block["wv"]),
            "bv": vector_to_floats(block["bv"]),
            "wo": matrix_to_floats(block["wo"]),
            "bo": vector_to_floats(block["bo"]),
            "w1": matrix_to_floats(block["w1"]),
            "b1": vector_to_floats(block["b1"]),
            "w_gate": matrix_to_floats(block["w_gate"]),
            "b_gate": vector_to_floats(block["b_gate"]),
            "w2": matrix_to_floats(block["w2"]),
            "b2": vector_to_floats(block["b2"]),
            "ln1_gain": vector_to_floats(block["ln1_gain"]),
            "ln1_bias": vector_to_floats(block["ln1_bias"]),
            "ln2_gain": vector_to_floats(block["ln2_gain"]),
            "ln2_bias": vector_to_floats(block["ln2_bias"]),
        }

    def _uses_block_layer_norm_parameters(self) -> bool:
        return self.config.use_layer_norm or self.config.use_pre_layer_norm

    def parameters(self) -> list[Scalar]:
        params: list[Scalar] = []
        for item in [
            self.token_embeddings,
            self.position_embeddings,
            self.wq,
            self.bq,
            self.wk,
            self.bk,
            self.wv,
            self.bv,
            self.wo,
            self.bo,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.bout,
        ]:
            params.extend(flatten_scalars(item))
        if self.config.use_gated_mlp:
            for item in [self.w_gate, self.b_gate]:
                params.extend(flatten_scalars(item))
        if not self.config.tie_output_embeddings:
            params.extend(flatten_scalars(self.wout))
        if self.config.use_context_projection:
            for item in [self.context_projection_w, self.context_projection_b]:
                params.extend(flatten_scalars(item))
        if self.config.use_prompt_prefix_projection:
            for item in [
                self.prompt_prefix_projection_w,
                self.prompt_prefix_projection_b,
            ]:
                params.extend(flatten_scalars(item))
        if self.config.use_prompt_position_projection:
            for item in [
                self.prompt_position_projection_w,
                self.prompt_position_projection_b,
            ]:
                params.extend(flatten_scalars(item))
        if self.config.use_prompt_attention_summary:
            for item in [
                self.prompt_summary_query,
                self.prompt_summary_w,
                self.prompt_summary_b,
            ]:
                params.extend(flatten_scalars(item))
        if self._uses_block_layer_norm_parameters():
            for item in [self.ln1_gain, self.ln1_bias, self.ln2_gain, self.ln2_bias]:
                params.extend(flatten_scalars(item))
        if self.config.use_pre_layer_norm:
            for item in [self.final_ln_gain, self.final_ln_bias]:
                params.extend(flatten_scalars(item))
        for block in self.extra_blocks:
            for item in [
                block["wq"],
                block["bq"],
                block["wk"],
                block["bk"],
                block["wv"],
                block["bv"],
                block["wo"],
                block["bo"],
                block["w1"],
                block["b1"],
                block["w2"],
                block["b2"],
            ]:
                params.extend(flatten_scalars(item))
            if self.config.use_gated_mlp:
                for item in [block["w_gate"], block["b_gate"]]:
                    params.extend(flatten_scalars(item))
            if self._uses_block_layer_norm_parameters():
                for item in [
                    block["ln1_gain"],
                    block["ln1_bias"],
                    block["ln2_gain"],
                    block["ln2_bias"],
                ]:
                    params.extend(flatten_scalars(item))
        return params

    def top_layer_parameters(self) -> list[Scalar]:
        if self.config.num_layers == 1:
            return self.parameters()
        params: list[Scalar] = []
        top_block = self.blocks[-1]
        for item in [
            top_block["wq"],
            top_block["bq"],
            top_block["wk"],
            top_block["bk"],
            top_block["wv"],
            top_block["bv"],
            top_block["wo"],
            top_block["bo"],
            top_block["w1"],
            top_block["b1"],
            top_block["w2"],
            top_block["b2"],
            self.bout,
        ]:
            params.extend(flatten_scalars(item))
        if self.config.use_gated_mlp:
            for item in [top_block["w_gate"], top_block["b_gate"]]:
                params.extend(flatten_scalars(item))
        if not self.config.tie_output_embeddings:
            params.extend(flatten_scalars(self.wout))
        if self.config.use_context_projection:
            for item in [self.context_projection_w, self.context_projection_b]:
                params.extend(flatten_scalars(item))
        if self.config.use_prompt_prefix_projection:
            for item in [
                self.prompt_prefix_projection_w,
                self.prompt_prefix_projection_b,
            ]:
                params.extend(flatten_scalars(item))
        if self.config.use_prompt_position_projection:
            for item in [
                self.prompt_position_projection_w,
                self.prompt_position_projection_b,
            ]:
                params.extend(flatten_scalars(item))
        if self.config.use_prompt_attention_summary:
            for item in [
                self.prompt_summary_query,
                self.prompt_summary_w,
                self.prompt_summary_b,
            ]:
                params.extend(flatten_scalars(item))
        if self._uses_block_layer_norm_parameters():
            for item in [
                top_block["ln1_gain"],
                top_block["ln1_bias"],
                top_block["ln2_gain"],
                top_block["ln2_bias"],
            ]:
                params.extend(flatten_scalars(item))
        if self.config.use_pre_layer_norm:
            for item in [self.final_ln_gain, self.final_ln_bias]:
                params.extend(flatten_scalars(item))
        return params

    def _output_weights_scalars(self) -> list[list[Scalar]]:
        if not self.config.tie_output_embeddings:
            return self.wout
        return [
            [self.token_embeddings[token_id][dim] for token_id in range(self.config.vocab_size)]
            for dim in range(self.config.embedding_dim)
        ]

    def _output_weights_floats(self) -> list[list[float]]:
        if not self.config.tie_output_embeddings:
            return matrix_to_floats(self.wout)
        token_embeddings = matrix_to_floats(self.token_embeddings)
        return [
            [token_embeddings[token_id][dim] for token_id in range(self.config.vocab_size)]
            for dim in range(self.config.embedding_dim)
        ]

    def _forward_scalars(self, context: list[int]) -> list[Scalar]:
        return linear_scalars(
            self._final_hidden_scalars(context),
            self._output_weights_scalars(),
            self.bout,
        )

    def _final_hidden_scalars(self, context: list[int]) -> list[Scalar]:
        if len(context) != self.config.context_size:
            raise ValueError(
                f"context must have {self.config.context_size} ids, got {len(context)}"
            )
        if self.freeze_lower_layers_for_updates and self.config.num_layers > 1:
            float_blocks = [self._block_to_floats(block) for block in self.blocks[:-1]]
            token_embeddings = matrix_to_floats(self.token_embeddings)
            position_embeddings = matrix_to_floats(self.position_embeddings)
            x_float = [
                [
                    token_embeddings[token_id][dim] + position_embeddings[position][dim]
                    for dim in range(self.config.embedding_dim)
                ]
                for position, token_id in enumerate(context)
            ]
            for block in float_blocks:
                x_float = self._forward_full_block_floats(x_float, block)
            x = matrix_to_scalars(x_float)
            return self._finalize_hidden_scalars(
                self._forward_final_block_scalars(x, self.blocks[-1], context)
            )
        x = [
            [
                self.token_embeddings[token_id][dim] + self.position_embeddings[position][dim]
                for dim in range(self.config.embedding_dim)
            ]
            for position, token_id in enumerate(context)
        ]
        if self.config.num_layers == 1:
            return self._finalize_hidden_scalars(
                self._forward_final_block_scalars(x, self.blocks[0], context)
            )
        else:
            for block in self.blocks[:-1]:
                x = self._forward_full_block_scalars(x, block)
            return self._finalize_hidden_scalars(
                self._forward_final_block_scalars(x, self.blocks[-1], context)
            )

    def _forward_floats(self, context: list[int]) -> list[float]:
        return linear_floats(
            self.final_hidden(context),
            self._output_weights_floats(),
            vector_to_floats(self.bout),
        )

    def final_hidden(self, context: list[int]) -> list[float]:
        if len(context) != self.config.context_size:
            raise ValueError(
                f"context must have {self.config.context_size} ids, got {len(context)}"
            )
        token_embeddings = matrix_to_floats(self.token_embeddings)
        position_embeddings = matrix_to_floats(self.position_embeddings)
        x = [
            [
                token_embeddings[token_id][dim] + position_embeddings[position][dim]
                for dim in range(self.config.embedding_dim)
            ]
            for position, token_id in enumerate(context)
        ]
        float_blocks = [self._block_to_floats(block) for block in self.blocks]
        if self.config.num_layers == 1:
            return self._finalize_hidden_floats(
                self._forward_final_block_floats(x, float_blocks[0], context)
            )
        else:
            for block in float_blocks[:-1]:
                x = self._forward_full_block_floats(x, block)
            return self._finalize_hidden_floats(
                self._forward_final_block_floats(x, float_blocks[-1], context)
            )

    def _finalize_hidden_scalars(self, hidden: list[Scalar]) -> list[Scalar]:
        if not self.config.use_pre_layer_norm:
            return hidden
        if self.config.use_rms_norm:
            return rms_norm_scalars(
                hidden,
                self.final_ln_gain,
                self.config.layer_norm_epsilon,
            )
        return layer_norm_scalars(
            hidden,
            self.final_ln_gain,
            self.final_ln_bias,
            self.config.layer_norm_epsilon,
        )

    def _finalize_hidden_floats(self, hidden: list[float]) -> list[float]:
        if not self.config.use_pre_layer_norm:
            return hidden
        if self.config.use_rms_norm:
            return rms_norm_floats(
                hidden,
                vector_to_floats(self.final_ln_gain),
                self.config.layer_norm_epsilon,
            )
        return layer_norm_floats(
            hidden,
            vector_to_floats(self.final_ln_gain),
            vector_to_floats(self.final_ln_bias),
            self.config.layer_norm_epsilon,
        )

    def _forward_final_block_scalars(
        self,
        x: list[list[Scalar]],
        block: dict[str, Any],
        context: list[int],
    ) -> list[Scalar]:
        attention_input = self._attention_input_scalars(x, block)
        q = [linear_scalars(row, block["wq"], block["bq"]) for row in attention_input]
        k = [linear_scalars(row, block["wk"], block["bk"]) for row in attention_input]
        v = [linear_scalars(row, block["wv"], block["bv"]) for row in attention_input]
        if self.config.use_rotary_positions:
            q = self._apply_rotary_scalars(q)
            k = self._apply_rotary_scalars(k)
        last_position = self.config.context_size - 1
        attended = self._causal_attention_scalars(q, k, v, last_position)
        projected = linear_scalars(attended, block["wo"], block["bo"])
        hidden = [
            x[last_position][dim] + projected[dim]
            for dim in range(self.config.embedding_dim)
        ]
        if self.config.use_context_mean:
            hidden = [
                hidden[dim]
                + sum(row[dim] for row in x) / self.config.context_size
                for dim in range(self.config.embedding_dim)
            ]
        if self.config.use_context_projection:
            context_summary = [
                sum(row[dim] for row in x) / self.config.context_size
                for dim in range(self.config.embedding_dim)
            ]
            projected_summary = linear_scalars(
                context_summary,
                self.context_projection_w,
                self.context_projection_b,
            )
            hidden = [
                hidden[dim] + projected_summary[dim]
                for dim in range(self.config.embedding_dim)
            ]
        if self.config.use_prompt_prefix_projection:
            prompt_rows = [
                row
                for position, row in enumerate(x[:last_position])
                if context[position] != 0
            ]
            if prompt_rows:
                prompt_summary = [
                    sum(row[dim] for row in prompt_rows) / len(prompt_rows)
                    for dim in range(self.config.embedding_dim)
                ]
                projected_summary = linear_scalars(
                    prompt_summary,
                    self.prompt_prefix_projection_w,
                    self.prompt_prefix_projection_b,
                )
                hidden = [
                    hidden[dim] + projected_summary[dim]
                    for dim in range(self.config.embedding_dim)
                ]
        if self.config.use_prompt_position_projection:
            prompt_positions = [
                (position, row)
                for position, row in enumerate(x[:last_position])
                if context[position] != 0
            ]
            if prompt_positions:
                projected_summary: list[Scalar] = []
                for output_dim, bias in enumerate(self.prompt_position_projection_b):
                    total = Scalar(0.0)
                    for position, row in prompt_positions:
                        position_weights = self.prompt_position_projection_w[position]
                        for input_dim, value in enumerate(row):
                            total = (
                                total
                                + value * position_weights[input_dim][output_dim]
                            )
                    projected_summary.append(total / len(prompt_positions) + bias)
                hidden = [
                    hidden[dim]
                    + projected_summary[dim]
                    * self.config.prompt_position_projection_scale
                    for dim in range(self.config.embedding_dim)
                ]
        if self.config.use_prompt_attention_summary:
            scores = [
                dot_scalars(self.prompt_summary_query, row)
                * (1.0 / math.sqrt(self.config.embedding_dim))
                for row in x
            ]
            weights = softmax_scalars(scores)
            attention_summary = []
            for dim in range(self.config.embedding_dim):
                total = Scalar(0.0)
                for row, weight in zip(x, weights):
                    total = total + weight * row[dim]
                attention_summary.append(total)
            projected_summary = linear_scalars(
                attention_summary,
                self.prompt_summary_w,
                self.prompt_summary_b,
            )
            hidden = [
                hidden[dim] + projected_summary[dim]
                for dim in range(self.config.embedding_dim)
            ]
        return self._feed_forward_scalars(hidden, block)

    def _forward_full_block_scalars(
        self,
        x: list[list[Scalar]],
        block: dict[str, Any],
    ) -> list[list[Scalar]]:
        attention_input = self._attention_input_scalars(x, block)
        q = [linear_scalars(row, block["wq"], block["bq"]) for row in attention_input]
        k = [linear_scalars(row, block["wk"], block["bk"]) for row in attention_input]
        v = [linear_scalars(row, block["wv"], block["bv"]) for row in attention_input]
        if self.config.use_rotary_positions:
            q = self._apply_rotary_scalars(q)
            k = self._apply_rotary_scalars(k)
        outputs = []
        for position in range(self.config.context_size):
            attended = self._causal_attention_scalars(q, k, v, position)
            projected = linear_scalars(attended, block["wo"], block["bo"])
            hidden = [
                x[position][dim] + projected[dim]
                for dim in range(self.config.embedding_dim)
            ]
            outputs.append(self._feed_forward_scalars(hidden, block))
        return outputs

    def _feed_forward_scalars(
        self,
        hidden: list[Scalar],
        block: dict[str, Any],
    ) -> list[Scalar]:
        if self.config.use_pre_layer_norm:
            ff_input = layer_norm_scalars(
                hidden,
                block["ln2_gain"],
                block["ln2_bias"],
                self.config.layer_norm_epsilon,
            )
            if self.config.use_rms_norm:
                ff_input = rms_norm_scalars(
                    hidden,
                    block["ln2_gain"],
                    self.config.layer_norm_epsilon,
                )
            ff_hidden = [
                value.tanh()
                for value in linear_scalars(ff_input, block["w1"], block["b1"])
            ]
            if self.config.use_gated_mlp:
                ff_gate = [
                    value.tanh()
                    for value in linear_scalars(ff_input, block["w_gate"], block["b_gate"])
                ]
                ff_hidden = [
                    hidden_value * gate_value
                    for hidden_value, gate_value in zip(ff_hidden, ff_gate)
                ]
            ff_out = linear_scalars(ff_hidden, block["w2"], block["b2"])
            return [
                hidden[dim] + ff_out[dim]
                for dim in range(self.config.embedding_dim)
            ]
        if self.config.use_layer_norm:
            hidden = layer_norm_scalars(
                hidden,
                block["ln1_gain"],
                block["ln1_bias"],
                self.config.layer_norm_epsilon,
            )
        if self.config.use_rms_norm:
            hidden = rms_norm_scalars(
                hidden,
                block["ln1_gain"],
                self.config.layer_norm_epsilon,
            )
        ff_hidden = [value.tanh() for value in linear_scalars(hidden, block["w1"], block["b1"])]
        if self.config.use_gated_mlp:
            ff_gate = [
                value.tanh()
                for value in linear_scalars(hidden, block["w_gate"], block["b_gate"])
            ]
            ff_hidden = [
                hidden_value * gate_value
                for hidden_value, gate_value in zip(ff_hidden, ff_gate)
            ]
        ff_out = linear_scalars(ff_hidden, block["w2"], block["b2"])
        block_out = [
            hidden[dim] + ff_out[dim]
            for dim in range(self.config.embedding_dim)
        ]
        if self.config.use_layer_norm:
            block_out = layer_norm_scalars(
                block_out,
                block["ln2_gain"],
                block["ln2_bias"],
                self.config.layer_norm_epsilon,
            )
        return block_out

    def _attention_input_scalars(
        self,
        x: list[list[Scalar]],
        block: dict[str, Any],
    ) -> list[list[Scalar]]:
        if self.config.use_rms_norm and self.config.use_pre_layer_norm:
            return [
                rms_norm_scalars(
                    row,
                    block["ln1_gain"],
                    self.config.layer_norm_epsilon,
                )
                for row in x
            ]
        if not self.config.use_pre_layer_norm:
            return x
        return [
            layer_norm_scalars(
                row,
                block["ln1_gain"],
                block["ln1_bias"],
                self.config.layer_norm_epsilon,
            )
            for row in x
        ]

    def _forward_final_block_floats(
        self,
        x: list[list[float]],
        block: dict[str, Any],
        context: list[int],
    ) -> list[float]:
        attention_input = self._attention_input_floats(x, block)
        q = [linear_floats(row, block["wq"], block["bq"]) for row in attention_input]
        k = [linear_floats(row, block["wk"], block["bk"]) for row in attention_input]
        v = [linear_floats(row, block["wv"], block["bv"]) for row in attention_input]
        if self.config.use_rotary_positions:
            q = self._apply_rotary_floats(q)
            k = self._apply_rotary_floats(k)
        last_position = self.config.context_size - 1
        attended = self._causal_attention_floats(q, k, v, last_position)
        projected = linear_floats(attended, block["wo"], block["bo"])
        hidden = [
            x[last_position][dim] + projected[dim]
            for dim in range(self.config.embedding_dim)
        ]
        if self.config.use_context_mean:
            hidden = [
                hidden[dim]
                + sum(row[dim] for row in x) / self.config.context_size
                for dim in range(self.config.embedding_dim)
            ]
        if self.config.use_context_projection:
            context_summary = [
                sum(row[dim] for row in x) / self.config.context_size
                for dim in range(self.config.embedding_dim)
            ]
            projected_summary = linear_floats(
                context_summary,
                matrix_to_floats(self.context_projection_w),
                vector_to_floats(self.context_projection_b),
            )
            hidden = [
                hidden[dim] + projected_summary[dim]
                for dim in range(self.config.embedding_dim)
            ]
        if self.config.use_prompt_prefix_projection:
            prompt_rows = [
                row
                for position, row in enumerate(x[:last_position])
                if context[position] != 0
            ]
            if prompt_rows:
                prompt_summary = [
                    sum(row[dim] for row in prompt_rows) / len(prompt_rows)
                    for dim in range(self.config.embedding_dim)
                ]
                projected_summary = linear_floats(
                    prompt_summary,
                    matrix_to_floats(self.prompt_prefix_projection_w),
                    vector_to_floats(self.prompt_prefix_projection_b),
                )
                hidden = [
                    hidden[dim] + projected_summary[dim]
                    for dim in range(self.config.embedding_dim)
                ]
        if self.config.use_prompt_position_projection:
            prompt_positions = [
                (position, row)
                for position, row in enumerate(x[:last_position])
                if context[position] != 0
            ]
            if prompt_positions:
                prompt_position_projection_w = [
                    matrix_to_floats(position_weights)
                    for position_weights in self.prompt_position_projection_w
                ]
                prompt_position_projection_b = vector_to_floats(
                    self.prompt_position_projection_b
                )
                projected_summary = []
                for output_dim, bias in enumerate(prompt_position_projection_b):
                    total = 0.0
                    for position, row in prompt_positions:
                        position_weights = prompt_position_projection_w[position]
                        for input_dim, value in enumerate(row):
                            total += value * position_weights[input_dim][output_dim]
                    projected_summary.append(total / len(prompt_positions) + bias)
                hidden = [
                    hidden[dim]
                    + projected_summary[dim]
                    * self.config.prompt_position_projection_scale
                    for dim in range(self.config.embedding_dim)
                ]
        if self.config.use_prompt_attention_summary:
            prompt_summary_query = vector_to_floats(self.prompt_summary_query)
            scores = [
                dot_floats(prompt_summary_query, row)
                * (1.0 / math.sqrt(self.config.embedding_dim))
                for row in x
            ]
            weights = softmax_floats(scores)
            attention_summary = [
                sum(weight * row[dim] for row, weight in zip(x, weights))
                for dim in range(self.config.embedding_dim)
            ]
            prompt_summary_w = matrix_to_floats(self.prompt_summary_w)
            prompt_summary_b = vector_to_floats(self.prompt_summary_b)
            projected_summary = linear_floats(
                attention_summary,
                prompt_summary_w,
                prompt_summary_b,
            )
            hidden = [
                hidden[dim] + projected_summary[dim]
                for dim in range(self.config.embedding_dim)
            ]
        return self._feed_forward_floats(hidden, block)

    def _forward_full_block_floats(
        self,
        x: list[list[float]],
        block: dict[str, Any],
    ) -> list[list[float]]:
        attention_input = self._attention_input_floats(x, block)
        q = [linear_floats(row, block["wq"], block["bq"]) for row in attention_input]
        k = [linear_floats(row, block["wk"], block["bk"]) for row in attention_input]
        v = [linear_floats(row, block["wv"], block["bv"]) for row in attention_input]
        if self.config.use_rotary_positions:
            q = self._apply_rotary_floats(q)
            k = self._apply_rotary_floats(k)
        outputs = []
        for position in range(self.config.context_size):
            attended = self._causal_attention_floats(q, k, v, position)
            projected = linear_floats(attended, block["wo"], block["bo"])
            hidden = [
                x[position][dim] + projected[dim]
                for dim in range(self.config.embedding_dim)
            ]
            outputs.append(self._feed_forward_floats(hidden, block))
        return outputs

    def _feed_forward_floats(
        self,
        hidden: list[float],
        block: dict[str, Any],
    ) -> list[float]:
        if self.config.use_pre_layer_norm:
            ff_input = layer_norm_floats(
                hidden,
                block["ln2_gain"],
                block["ln2_bias"],
                self.config.layer_norm_epsilon,
            )
            if self.config.use_rms_norm:
                ff_input = rms_norm_floats(
                    hidden,
                    block["ln2_gain"],
                    self.config.layer_norm_epsilon,
                )
            ff_hidden = [
                math.tanh(value)
                for value in linear_floats(ff_input, block["w1"], block["b1"])
            ]
            if self.config.use_gated_mlp:
                ff_gate = [
                    math.tanh(value)
                    for value in linear_floats(ff_input, block["w_gate"], block["b_gate"])
                ]
                ff_hidden = [
                    hidden_value * gate_value
                    for hidden_value, gate_value in zip(ff_hidden, ff_gate)
                ]
            ff_out = linear_floats(ff_hidden, block["w2"], block["b2"])
            return [
                hidden[dim] + ff_out[dim]
                for dim in range(self.config.embedding_dim)
            ]
        if self.config.use_layer_norm:
            hidden = layer_norm_floats(
                hidden,
                block["ln1_gain"],
                block["ln1_bias"],
                self.config.layer_norm_epsilon,
            )
        if self.config.use_rms_norm:
            hidden = rms_norm_floats(
                hidden,
                block["ln1_gain"],
                self.config.layer_norm_epsilon,
            )
        ff_hidden = [math.tanh(value) for value in linear_floats(hidden, block["w1"], block["b1"])]
        if self.config.use_gated_mlp:
            ff_gate = [
                math.tanh(value)
                for value in linear_floats(hidden, block["w_gate"], block["b_gate"])
            ]
            ff_hidden = [
                hidden_value * gate_value
                for hidden_value, gate_value in zip(ff_hidden, ff_gate)
            ]
        ff_out = linear_floats(ff_hidden, block["w2"], block["b2"])
        block_out = [
            hidden[dim] + ff_out[dim]
            for dim in range(self.config.embedding_dim)
        ]
        if self.config.use_layer_norm:
            block_out = layer_norm_floats(
                block_out,
                block["ln2_gain"],
                block["ln2_bias"],
                self.config.layer_norm_epsilon,
            )
        return block_out

    def _attention_input_floats(
        self,
        x: list[list[float]],
        block: dict[str, Any],
    ) -> list[list[float]]:
        if self.config.use_rms_norm and self.config.use_pre_layer_norm:
            return [
                rms_norm_floats(
                    row,
                    block["ln1_gain"],
                    self.config.layer_norm_epsilon,
                )
                for row in x
            ]
        if not self.config.use_pre_layer_norm:
            return x
        return [
            layer_norm_floats(
                row,
                block["ln1_gain"],
                block["ln1_bias"],
                self.config.layer_norm_epsilon,
            )
            for row in x
        ]

    def _causal_attention_scalars(
        self,
        q: list[list[Scalar]],
        k: list[list[Scalar]],
        v: list[list[Scalar]],
        position: int,
    ) -> list[Scalar]:
        head_dim = self.config.embedding_dim // self.config.attention_heads
        attended: list[Scalar] = []
        for head in range(self.config.attention_heads):
            start = head * head_dim
            end = start + head_dim
            scale = 1.0 / math.sqrt(head_dim)
            scores = [
                dot_scalars(q[position][start:end], k[past][start:end]) * scale
                for past in range(position + 1)
            ]
            weights = softmax_scalars(scores)
            for dim in range(start, end):
                total = Scalar(0.0)
                for past, weight in enumerate(weights):
                    total = total + weight * v[past][dim]
                attended.append(total)
        return attended

    def _causal_attention_floats(
        self,
        q: list[list[float]],
        k: list[list[float]],
        v: list[list[float]],
        position: int,
    ) -> list[float]:
        head_dim = self.config.embedding_dim // self.config.attention_heads
        attended: list[float] = []
        for head in range(self.config.attention_heads):
            start = head * head_dim
            end = start + head_dim
            scale = 1.0 / math.sqrt(head_dim)
            scores = [
                dot_floats(q[position][start:end], k[past][start:end]) * scale
                for past in range(position + 1)
            ]
            weights = softmax_floats(scores)
            for dim in range(start, end):
                attended.append(
                    sum(weight * v[past][dim] for past, weight in enumerate(weights))
                )
        return attended

    def _apply_rotary_scalars(self, rows: list[list[Scalar]]) -> list[list[Scalar]]:
        head_dim = self.config.embedding_dim // self.config.attention_heads
        rotated: list[list[Scalar]] = []
        for position, row in enumerate(rows):
            output = row[:]
            for head in range(self.config.attention_heads):
                start = head * head_dim
                for offset in range(0, head_dim - 1, 2):
                    index = start + offset
                    angle = position / (10000.0 ** (offset / max(head_dim, 1)))
                    cos_value = math.cos(angle)
                    sin_value = math.sin(angle)
                    left = row[index]
                    right = row[index + 1]
                    output[index] = left * cos_value - right * sin_value
                    output[index + 1] = left * sin_value + right * cos_value
            rotated.append(output)
        return rotated

    def _apply_rotary_floats(self, rows: list[list[float]]) -> list[list[float]]:
        head_dim = self.config.embedding_dim // self.config.attention_heads
        rotated: list[list[float]] = []
        for position, row in enumerate(rows):
            output = row[:]
            for head in range(self.config.attention_heads):
                start = head * head_dim
                for offset in range(0, head_dim - 1, 2):
                    index = start + offset
                    angle = position / (10000.0 ** (offset / max(head_dim, 1)))
                    cos_value = math.cos(angle)
                    sin_value = math.sin(angle)
                    left = row[index]
                    right = row[index + 1]
                    output[index] = left * cos_value - right * sin_value
                    output[index + 1] = left * sin_value + right * cos_value
            rotated.append(output)
        return rotated

    def predict(self, context: list[int]) -> list[float]:
        return softmax_floats(self._forward_floats(context))

    def nll(self, context: list[int], target: int) -> float:
        probs = self.predict(context)
        return -math.log(max(probs[target], 1e-12))

    def apply_gradients(
        self,
        params: list[Scalar],
        learning_rate: float,
    ) -> float:
        optimizer = self.active_optimizer
        if optimizer is not None:
            return optimizer.apply(params, learning_rate)
        for parameter in params:
            clipped_grad = max(min(parameter.grad, 5.0), -5.0)
            parameter.data -= learning_rate * clipped_grad
        return learning_rate

    def train_step(
        self,
        context: list[int],
        target: int,
        learning_rate: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        loss = cross_entropy_scalars(self._forward_scalars(context), target)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_unlikelihood(
        self,
        context: list[int],
        target: int,
        negative: int,
        learning_rate: float,
        negative_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        probs = softmax_scalars(self._forward_scalars(context))
        loss = -probs[target].log()
        if negative != target and negative_weight > 0.0:
            loss = loss + (-(Scalar(1.0) - probs[negative] + 1e-12).log()) * negative_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_unlikelihood_and_positive(
        self,
        context: list[int],
        target: int,
        negative: int,
        positive_context: list[int],
        positive_target: int,
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        probs = softmax_scalars(self._forward_scalars(context))
        loss = -probs[target].log()
        if negative != target and negative_weight > 0.0:
            loss = loss + (-(Scalar(1.0) - probs[negative] + 1e-12).log()) * negative_weight
        if positive_weight > 0.0:
            positive_probs = softmax_scalars(self._forward_scalars(positive_context))
            loss = loss + (-positive_probs[positive_target].log()) * positive_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_contrast(
        self,
        context: list[int],
        target: int,
        contrast_context: list[int],
        contrast_target: int,
        learning_rate: float,
        negative_weight: float,
        contrast_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        probs = softmax_scalars(self._forward_scalars(context))
        loss = -probs[target].log()
        if contrast_target != target and negative_weight > 0.0:
            loss = loss + (-(Scalar(1.0) - probs[contrast_target] + 1e-12).log()) * negative_weight
        if contrast_weight > 0.0:
            contrast_probs = softmax_scalars(self._forward_scalars(contrast_context))
            loss = loss + (-contrast_probs[contrast_target].log()) * contrast_weight
            if target != contrast_target and negative_weight > 0.0:
                loss = loss + (
                    -(Scalar(1.0) - contrast_probs[target] + 1e-12).log()
                ) * negative_weight * contrast_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_batch_contrast(
        self,
        branches: list[tuple[list[int], int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target in branches})
        loss = Scalar(0.0)
        for context, target in branches:
            probs = softmax_scalars(self._forward_scalars(context))
            if positive_weight > 0.0:
                loss = loss + (-probs[target].log()) * positive_weight
            negatives = [
                branch_target
                for branch_target in branch_targets
                if branch_target != target
            ]
            if negative_weight > 0.0 and negatives:
                per_negative_weight = negative_weight / len(negatives)
                for negative in negatives:
                    loss = loss + (
                        -(Scalar(1.0) - probs[negative] + 1e-12).log()
                    ) * per_negative_weight
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_diversity(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        contrast_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        loss = Scalar(0.0)
        for context, target, predicted in branches:
            probs = softmax_scalars(self._forward_scalars(context))
            if positive_weight > 0.0:
                loss = loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                loss = loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            contrast_targets = [
                branch_target
                for branch_target in branch_targets
                if branch_target != target
            ]
            if contrast_weight > 0.0 and contrast_targets:
                per_target_weight = contrast_weight / len(contrast_targets)
                for contrast_target in contrast_targets:
                    loss = loss + (
                        -(Scalar(1.0) - probs[contrast_target] + 1e-12).log()
                    ) * per_target_weight
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_target_softmax(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        target_softmax_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        branch_target_offsets = {
            target: offset for offset, target in enumerate(branch_targets)
        }
        loss = Scalar(0.0)
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                loss = loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                loss = loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            if target_softmax_weight > 0.0 and len(branch_targets) > 1:
                target_logits = [logits[branch_target] for branch_target in branch_targets]
                target_probs = softmax_scalars(target_logits)
                loss = loss + (
                    -target_probs[branch_target_offsets[target]].log()
                ) * target_softmax_weight
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_target_margin(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        margin_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        loss = Scalar(0.0)
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                loss = loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                loss = loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            margin_targets = [
                branch_target
                for branch_target in branch_targets
                if branch_target != target
            ]
            if margin_weight > 0.0 and margin_targets:
                per_target_weight = margin_weight / len(margin_targets)
                target_logit = logits[target]
                for margin_target in margin_targets:
                    gap = logits[margin_target] - target_logit + 1.0
                    loss = loss + (Scalar(1.0) + gap.exp()).log() * per_target_weight
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_representation_contrast(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        representation_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_loss = Scalar(0.0)
        hidden_by_target: list[tuple[list[Scalar], int]] = []
        for context, target, predicted in branches:
            hidden = self._final_hidden_scalars(context)
            logits = linear_scalars(hidden, self.wout, self.bout)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                branch_loss = branch_loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                branch_loss = branch_loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            hidden_by_target.append((hidden, target))
        loss = branch_loss / max(len(branches), 1)
        if representation_weight > 0.0:
            contrast_loss = Scalar(0.0)
            contrast_pairs = 0
            for left_index, (left_hidden, left_target) in enumerate(hidden_by_target):
                for right_hidden, right_target in hidden_by_target[left_index + 1:]:
                    if left_target == right_target:
                        continue
                    distance_sq = Scalar(0.0)
                    for left_value, right_value in zip(left_hidden, right_hidden):
                        delta = left_value - right_value
                        distance_sq = distance_sq + delta * delta
                    distance_sq = distance_sq / max(self.config.embedding_dim, 1)
                    contrast_loss = contrast_loss + (-distance_sq).exp()
                    contrast_pairs += 1
            if contrast_pairs:
                loss = loss + (contrast_loss / contrast_pairs) * representation_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_output_binding(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        binding_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        branch_target_offsets = {
            target: offset for offset, target in enumerate(branch_targets)
        }
        branch_loss = Scalar(0.0)
        hidden_by_target: list[tuple[list[Scalar], int]] = []
        for context, target, predicted in branches:
            hidden = self._final_hidden_scalars(context)
            logits = linear_scalars(hidden, self.wout, self.bout)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                branch_loss = branch_loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                branch_loss = branch_loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            if binding_weight > 0.0 and len(branch_targets) > 1:
                target_logits = [logits[branch_target] for branch_target in branch_targets]
                target_probs = softmax_scalars(target_logits)
                branch_loss = branch_loss + (
                    -target_probs[branch_target_offsets[target]].log()
                ) * binding_weight
            hidden_by_target.append((hidden, target))
        loss = branch_loss / max(len(branches), 1)
        if binding_weight > 0.0:
            contrast_loss = Scalar(0.0)
            contrast_pairs = 0
            for left_index, (left_hidden, left_target) in enumerate(hidden_by_target):
                for right_hidden, right_target in hidden_by_target[left_index + 1:]:
                    if left_target == right_target:
                        continue
                    distance_sq = Scalar(0.0)
                    for left_value, right_value in zip(left_hidden, right_hidden):
                        delta = left_value - right_value
                        distance_sq = distance_sq + delta * delta
                    distance_sq = distance_sq / max(self.config.embedding_dim, 1)
                    contrast_loss = contrast_loss + (-distance_sq).exp()
                    contrast_pairs += 1
            if contrast_pairs:
                loss = loss + (contrast_loss / contrast_pairs) * binding_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_bidirectional_binding(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        binding_weight: float,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        branch_target_offsets = {
            target: offset for offset, target in enumerate(branch_targets)
        }
        branch_loss = Scalar(0.0)
        branch_logits_by_target: list[tuple[list[Scalar], int]] = []
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                branch_loss = branch_loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                branch_loss = branch_loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            branch_logits_by_target.append((logits, target))
        loss = branch_loss / max(len(branches), 1)
        if binding_weight > 0.0 and len(branch_targets) > 1:
            row_loss = Scalar(0.0)
            for logits, target in branch_logits_by_target:
                target_logits = [logits[branch_target] for branch_target in branch_targets]
                target_probs = softmax_scalars(target_logits)
                row_loss = row_loss + (
                    -target_probs[branch_target_offsets[target]].log()
                )
            row_loss = row_loss / max(len(branch_logits_by_target), 1)

            column_loss = Scalar(0.0)
            column_count = 0
            for branch_target in branch_targets:
                context_logits = [
                    logits[branch_target]
                    for logits, _target in branch_logits_by_target
                ]
                positive_indexes = [
                    index
                    for index, (_logits, target) in enumerate(branch_logits_by_target)
                    if target == branch_target
                ]
                if not positive_indexes or len(positive_indexes) == len(context_logits):
                    continue
                context_probs = softmax_scalars(context_logits)
                positive_mass = Scalar(0.0)
                for index in positive_indexes:
                    positive_mass = positive_mass + context_probs[index]
                column_loss = column_loss + (-(positive_mass + 1e-12).log())
                column_count += 1
            binding_loss = row_loss
            if column_count:
                binding_loss = (binding_loss + column_loss / column_count) / 2.0
            loss = loss + binding_loss * binding_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_coverage_binding(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        binding_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        branch_target_set = set(branch_targets)
        branch_loss = Scalar(0.0)
        row_loss = Scalar(0.0)
        coverage_loss = Scalar(0.0)
        branch_logits_by_target: list[tuple[list[Scalar], int]] = []
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                branch_loss = branch_loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                branch_loss = branch_loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            if binding_weight > 0.0 and len(branch_targets) > 1:
                hard_candidates = [
                    index
                    for index in sorted(
                        range(self.config.vocab_size),
                        key=lambda item: logits[item].data,
                        reverse=True,
                    )
                    if index not in branch_target_set
                ]
                if hard_negative_count > 0:
                    hard_candidates = hard_candidates[:hard_negative_count]
                candidate_ids = [*branch_targets, *hard_candidates]
                candidate_logits = [
                    logits[candidate_id] for candidate_id in candidate_ids
                ]
                candidate_probs = softmax_scalars(candidate_logits)
                target_offset = candidate_ids.index(target)
                row_loss = row_loss + (-candidate_probs[target_offset].log())
                target_set_mass = Scalar(0.0)
                for offset, candidate_id in enumerate(candidate_ids):
                    if candidate_id in branch_target_set:
                        target_set_mass = target_set_mass + candidate_probs[offset]
                coverage_loss = coverage_loss + (-(target_set_mass + 1e-12).log())
            branch_logits_by_target.append((logits, target))
        loss = branch_loss / max(len(branches), 1)
        if binding_weight > 0.0 and len(branch_targets) > 1:
            row_loss = row_loss / max(len(branch_logits_by_target), 1)
            coverage_loss = coverage_loss / max(len(branch_logits_by_target), 1)

            column_loss = Scalar(0.0)
            column_count = 0
            for branch_target in branch_targets:
                context_logits = [
                    logits[branch_target]
                    for logits, _target in branch_logits_by_target
                ]
                positive_indexes = [
                    index
                    for index, (_logits, target) in enumerate(branch_logits_by_target)
                    if target == branch_target
                ]
                if not positive_indexes or len(positive_indexes) == len(context_logits):
                    continue
                context_probs = softmax_scalars(context_logits)
                positive_mass = Scalar(0.0)
                for index in positive_indexes:
                    positive_mass = positive_mass + context_probs[index]
                column_loss = column_loss + (-(positive_mass + 1e-12).log())
                column_count += 1
            binding_loss = (row_loss + coverage_loss) / 2.0
            if column_count:
                binding_loss = (binding_loss + column_loss / column_count) / 2.0
            loss = loss + binding_loss * binding_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_target_set_coverage(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        coverage_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        branch_target_set = set(branch_targets)
        branch_loss = Scalar(0.0)
        coverage_loss = Scalar(0.0)
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                branch_loss = branch_loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                branch_loss = branch_loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            if coverage_weight > 0.0 and branch_targets:
                hard_candidates = [
                    index
                    for index in sorted(
                        range(self.config.vocab_size),
                        key=lambda item: logits[item].data,
                        reverse=True,
                    )
                    if index not in branch_target_set
                ]
                if hard_negative_count > 0:
                    hard_candidates = hard_candidates[:hard_negative_count]
                candidate_ids = [*branch_targets, *hard_candidates]
                candidate_logits = [
                    logits[candidate_id] for candidate_id in candidate_ids
                ]
                candidate_probs = softmax_scalars(candidate_logits)
                target_set_mass = Scalar(0.0)
                for offset, candidate_id in enumerate(candidate_ids):
                    if candidate_id in branch_target_set:
                        target_set_mass = target_set_mass + candidate_probs[offset]
                coverage_loss = coverage_loss + (-(target_set_mass + 1e-12).log())
        loss = branch_loss / max(len(branches), 1)
        if coverage_weight > 0.0:
            loss = loss + (
                coverage_loss / max(len(branches), 1)
            ) * coverage_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_target_diversity(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        diversity_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        branch_targets = sorted({target for _context, target, _predicted in branches})
        branch_target_set = set(branch_targets)
        branch_loss = Scalar(0.0)
        coverage_loss = Scalar(0.0)
        target_share_sums = [Scalar(0.0) for _target in branch_targets]
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                branch_loss = branch_loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                branch_loss = branch_loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            if diversity_weight > 0.0 and branch_targets:
                hard_candidates = [
                    index
                    for index in sorted(
                        range(self.config.vocab_size),
                        key=lambda item: logits[item].data,
                        reverse=True,
                    )
                    if index not in branch_target_set
                ]
                if hard_negative_count > 0:
                    hard_candidates = hard_candidates[:hard_negative_count]
                candidate_ids = [*branch_targets, *hard_candidates]
                candidate_logits = [
                    logits[candidate_id] for candidate_id in candidate_ids
                ]
                candidate_probs = softmax_scalars(candidate_logits)
                target_set_mass = Scalar(0.0)
                for offset, candidate_id in enumerate(candidate_ids):
                    if candidate_id in branch_target_set:
                        target_set_mass = target_set_mass + candidate_probs[offset]
                coverage_loss = coverage_loss + (-(target_set_mass + 1e-12).log())
                for offset, _branch_target in enumerate(branch_targets):
                    target_share_sums[offset] = target_share_sums[offset] + (
                        candidate_probs[offset] / (target_set_mass + 1e-12)
                    )
        loss = branch_loss / max(len(branches), 1)
        if diversity_weight > 0.0 and branch_targets:
            coverage_loss = coverage_loss / max(len(branches), 1)
            diversity_loss = Scalar(0.0)
            for target_share_sum in target_share_sums:
                average_target_share = target_share_sum / max(len(branches), 1)
                diversity_loss = diversity_loss + (
                    -(average_target_share + 1e-12).log()
                )
            diversity_loss = diversity_loss / max(len(branch_targets), 1)
            loss = loss + ((coverage_loss + diversity_loss) / 2.0) * diversity_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_target_replay_coverage(
        self,
        branches: list[tuple[list[int], int, int]],
        replay_targets: list[int],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        replay_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        replay_targets = sorted(
            {target for target in replay_targets if 0 <= target < self.config.vocab_size}
        )
        if not replay_targets:
            replay_targets = sorted({target for _context, target, _predicted in branches})
        replay_target_set = set(replay_targets)
        branch_loss = Scalar(0.0)
        coverage_loss = Scalar(0.0)
        target_share_sums = [Scalar(0.0) for _target in replay_targets]
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                branch_loss = branch_loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                branch_loss = branch_loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            if replay_weight > 0.0 and replay_targets:
                hard_candidates = [
                    index
                    for index in sorted(
                        range(self.config.vocab_size),
                        key=lambda item: logits[item].data,
                        reverse=True,
                    )
                    if index not in replay_target_set
                ]
                if hard_negative_count > 0:
                    hard_candidates = hard_candidates[:hard_negative_count]
                candidate_ids = [*replay_targets, *hard_candidates]
                candidate_logits = [
                    logits[candidate_id] for candidate_id in candidate_ids
                ]
                candidate_probs = softmax_scalars(candidate_logits)
                target_set_mass = Scalar(0.0)
                for offset, candidate_id in enumerate(candidate_ids):
                    if candidate_id in replay_target_set:
                        target_set_mass = target_set_mass + candidate_probs[offset]
                coverage_loss = coverage_loss + (-(target_set_mass + 1e-12).log())
                for offset, _replay_target in enumerate(replay_targets):
                    target_share_sums[offset] = target_share_sums[offset] + (
                        candidate_probs[offset] / (target_set_mass + 1e-12)
                    )
        loss = branch_loss / max(len(branches), 1)
        if replay_weight > 0.0 and replay_targets:
            coverage_loss = coverage_loss / max(len(branches), 1)
            target_balance_loss = Scalar(0.0)
            for target_share_sum in target_share_sums:
                average_target_share = target_share_sum / max(len(branches), 1)
                target_balance_loss = target_balance_loss + (
                    -(average_target_share + 1e-12).log()
                )
            target_balance_loss = target_balance_loss / max(len(replay_targets), 1)
            loss = loss + ((coverage_loss + target_balance_loss) / 2.0) * replay_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_context_replay_coverage(
        self,
        branches: list[BranchReplayRecord],
        replay_branches: list[BranchReplayRecord],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        replay_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
        preserve_covered_targets: bool = False,
        balance_covered_target_anchors: bool = False,
        focus_uncovered_targets: bool = False,
        preserve_predicted_target_coverage: bool = False,
        balance_deficit_targets: bool = False,
        profile_aware_targets: bool = False,
        balance_profile_target_shares: bool = False,
        enforce_prompt_target_margins: bool = False,
        floor_preservation_branches: list[BranchReplayRecord] | None = None,
        floor_preservation_weight: float = 0.0,
        balance_floor_preservation_targets: bool = False,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        replay_parts = [branch_replay_parts(branch) for branch in replay_branches]
        branch_parts = [branch_replay_parts(branch) for branch in branches]
        floor_preservation_parts = [
            branch_replay_parts(branch) for branch in (floor_preservation_branches or [])
        ]
        replay_targets = sorted(
            {target for _context, target, _predicted, _profile in replay_parts}
        )
        if not replay_targets:
            replay_targets = sorted(
                {target for _context, target, _predicted, _profile in branch_parts}
            )
            replay_parts = branch_parts
        replay_record_count = len(replay_parts)
        replay_target_set = set(replay_targets)
        replay_target_offsets = {
            target: offset for offset, target in enumerate(replay_targets)
        }
        replay_targets_by_profile: dict[str, list[int]] = {}
        replay_target_sets_by_profile: dict[str, set[int]] = {}
        replay_target_offsets_by_profile: dict[str, dict[int, int]] = {}
        for _context, target, _predicted, profile in replay_parts:
            profile_key = profile if profile_aware_targets else "__all__"
            replay_target_sets_by_profile.setdefault(profile_key, set()).add(target)
        if not replay_target_sets_by_profile:
            replay_target_sets_by_profile["__all__"] = set(replay_targets)
        if not profile_aware_targets:
            replay_target_sets_by_profile["__all__"] = replay_target_set
        for profile_key, profile_targets in replay_target_sets_by_profile.items():
            ordered_targets = sorted(profile_targets)
            replay_targets_by_profile[profile_key] = ordered_targets
            replay_target_offsets_by_profile[profile_key] = {
                target: offset for offset, target in enumerate(ordered_targets)
            }
        branch_loss = Scalar(0.0)
        replay_coverage_loss = Scalar(0.0)
        replay_ownership_loss = Scalar(0.0)
        deficit_target_loss = Scalar(0.0)
        deficit_target_count = 0
        deficit_target_losses_by_target: dict[tuple[str, int], Scalar] = {}
        deficit_target_counts_by_target: Counter[tuple[str, int]] = Counter()
        profile_target_share_losses_by_target: dict[tuple[str, int], Scalar] = {}
        profile_target_share_counts_by_target: Counter[tuple[str, int]] = Counter()
        prompt_target_margin_loss = Scalar(0.0)
        prompt_target_margin_count = 0
        coverage_preservation_losses_by_target: dict[tuple[str, int], Scalar] = {}
        coverage_preservation_counts_by_target: Counter[tuple[str, int]] = Counter()
        covered_anchor_loss = Scalar(0.0)
        covered_anchor_count = 0
        covered_anchor_losses_by_target: dict[tuple[str, int], Scalar] = {}
        covered_anchor_counts_by_target: Counter[tuple[str, int]] = Counter()
        floor_preservation_loss = Scalar(0.0)
        floor_preservation_count = 0
        floor_preservation_losses_by_target: dict[tuple[str, int], Scalar] = {}
        floor_preservation_counts_by_target: Counter[tuple[str, int]] = Counter()
        predicted_replay_targets_by_profile: dict[str, set[int]] = {}
        for _context, _target, predicted, profile in replay_parts:
            profile_key = profile if profile_aware_targets else "__all__"
            profile_target_set = replay_target_sets_by_profile.get(
                profile_key,
                replay_target_set,
            )
            if predicted in profile_target_set:
                predicted_replay_targets_by_profile.setdefault(profile_key, set()).add(
                    predicted
                )
        deficit_targets_by_profile = {
            profile_key: profile_target_set
            - predicted_replay_targets_by_profile.get(profile_key, set())
            for profile_key, profile_target_set in replay_target_sets_by_profile.items()
        }
        for context, target, predicted, _profile in branch_parts:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                branch_loss = branch_loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                branch_loss = branch_loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
        for context, target, predicted, profile in replay_parts:
            profile_key = profile if profile_aware_targets else "__all__"
            profile_replay_targets = replay_targets_by_profile.get(
                profile_key,
                replay_targets,
            )
            profile_replay_target_set = replay_target_sets_by_profile.get(
                profile_key,
                replay_target_set,
            )
            profile_replay_target_offsets = replay_target_offsets_by_profile.get(
                profile_key,
                replay_target_offsets,
            )
            profile_deficit_targets = deficit_targets_by_profile.get(profile_key, set())
            if target not in profile_replay_target_offsets:
                continue
            logits = self._forward_scalars(context)
            hard_candidates = [
                index
                for index in sorted(
                    range(self.config.vocab_size),
                    key=lambda item: logits[item].data,
                    reverse=True,
                )
                if index not in profile_replay_target_set
            ]
            if hard_negative_count > 0:
                hard_candidates = hard_candidates[:hard_negative_count]
            candidate_ids = [*profile_replay_targets, *hard_candidates]
            candidate_logits = [logits[candidate_id] for candidate_id in candidate_ids]
            candidate_probs = softmax_scalars(candidate_logits)
            target_set_mass = Scalar(0.0)
            for offset, candidate_id in enumerate(candidate_ids):
                if candidate_id in profile_replay_target_set:
                    target_set_mass = target_set_mass + candidate_probs[offset]
            target_offset = profile_replay_target_offsets[target]
            owned_target_share = candidate_probs[target_offset] / (
                target_set_mass + 1e-12
            )
            replay_coverage_loss = replay_coverage_loss + (
                -(target_set_mass + 1e-12).log()
            )
            replay_ownership_loss = replay_ownership_loss + (
                -(owned_target_share + 1e-12).log()
            )
            if (
                balance_profile_target_shares
                and profile_aware_targets
                and len(profile_replay_targets) > 1
            ):
                target_key = (profile_key, target)
                target_share_loss = -(owned_target_share + 1e-12).log()
                profile_target_share_losses_by_target[target_key] = (
                    profile_target_share_losses_by_target.get(
                        target_key,
                        Scalar(0.0),
                    )
                    + target_share_loss
                )
                profile_target_share_counts_by_target[target_key] += 1
            if (
                enforce_prompt_target_margins
                and profile_aware_targets
                and len(profile_replay_targets) > 1
            ):
                target_logit = candidate_logits[target_offset]
                for rival_target in profile_replay_targets:
                    if rival_target == target:
                        continue
                    rival_offset = profile_replay_target_offsets[rival_target]
                    margin_gap = candidate_logits[rival_offset] - target_logit + 1.0
                    prompt_target_margin_loss = prompt_target_margin_loss + (
                        Scalar(1.0) + margin_gap.exp()
                    ).log()
                    prompt_target_margin_count += 1
            if focus_uncovered_targets and target in profile_deficit_targets:
                target_key = (profile_key, target)
                target_deficit_loss = -(candidate_probs[target_offset] + 1e-12).log()
                deficit_target_loss = deficit_target_loss + target_deficit_loss
                deficit_target_count += 1
                deficit_target_losses_by_target[target_key] = (
                    deficit_target_losses_by_target.get(target_key, Scalar(0.0))
                    + target_deficit_loss
                )
                deficit_target_counts_by_target[target_key] += 1
            if (
                preserve_predicted_target_coverage
                and predicted in profile_replay_target_offsets
            ):
                predicted_key = (profile_key, predicted)
                predicted_offset = profile_replay_target_offsets[predicted]
                coverage_preservation_loss = -(
                    candidate_probs[predicted_offset] + 1e-12
                ).log()
                coverage_preservation_losses_by_target[predicted_key] = (
                    coverage_preservation_losses_by_target.get(
                        predicted_key,
                        Scalar(0.0),
                    )
                    + coverage_preservation_loss
                )
                coverage_preservation_counts_by_target[predicted_key] += 1
            if preserve_covered_targets and predicted == target:
                target_key = (profile_key, target)
                target_anchor_loss = -(candidate_probs[target_offset] + 1e-12).log()
                covered_anchor_loss = covered_anchor_loss + target_anchor_loss
                covered_anchor_count += 1
                covered_anchor_losses_by_target[target_key] = (
                    covered_anchor_losses_by_target.get(target_key, Scalar(0.0))
                    + target_anchor_loss
                )
                covered_anchor_counts_by_target[target_key] += 1
        if floor_preservation_weight > 0.0:
            for context, target, _predicted, profile in floor_preservation_parts:
                profile_key = profile if profile_aware_targets else "__all__"
                target_key = (profile_key, target)
                probs = softmax_scalars(self._forward_scalars(context))
                target_floor_loss = -(probs[target] + 1e-12).log()
                floor_preservation_loss = floor_preservation_loss + target_floor_loss
                floor_preservation_count += 1
                floor_preservation_losses_by_target[target_key] = (
                    floor_preservation_losses_by_target.get(target_key, Scalar(0.0))
                    + target_floor_loss
                )
                floor_preservation_counts_by_target[target_key] += 1
        loss = branch_loss / max(len(branches), 1)
        if replay_weight > 0.0 and replay_record_count and replay_targets:
            replay_loss = (
                replay_coverage_loss / replay_record_count
                + replay_ownership_loss / replay_record_count
            ) / 2.0
            if focus_uncovered_targets and deficit_target_count:
                if balance_deficit_targets and deficit_target_losses_by_target:
                    balanced_deficit_loss = Scalar(0.0)
                    for (
                        target,
                        target_deficit_loss,
                    ) in deficit_target_losses_by_target.items():
                        balanced_deficit_loss = balanced_deficit_loss + (
                            target_deficit_loss / deficit_target_counts_by_target[target]
                        )
                    replay_loss = replay_loss + (
                        balanced_deficit_loss / len(deficit_target_losses_by_target)
                    )
                else:
                    replay_loss = replay_loss + (
                        deficit_target_loss / max(deficit_target_count, 1)
                    )
            if balance_profile_target_shares and profile_target_share_losses_by_target:
                balanced_share_loss = Scalar(0.0)
                for (
                    target,
                    target_share_loss,
                ) in profile_target_share_losses_by_target.items():
                    balanced_share_loss = balanced_share_loss + (
                        target_share_loss / profile_target_share_counts_by_target[target]
                    )
                replay_loss = replay_loss + (
                    balanced_share_loss / len(profile_target_share_losses_by_target)
                )
            if enforce_prompt_target_margins and prompt_target_margin_count:
                replay_loss = replay_loss + (
                    prompt_target_margin_loss / prompt_target_margin_count
                )
            if preserve_predicted_target_coverage and coverage_preservation_losses_by_target:
                balanced_preservation_loss = Scalar(0.0)
                for (
                    target,
                    target_preservation_loss,
                ) in coverage_preservation_losses_by_target.items():
                    balanced_preservation_loss = balanced_preservation_loss + (
                        target_preservation_loss
                        / coverage_preservation_counts_by_target[target]
                    )
                replay_loss = replay_loss + (
                    balanced_preservation_loss
                    / len(coverage_preservation_losses_by_target)
                )
            if (
                preserve_covered_targets
                and balance_covered_target_anchors
                and len(covered_anchor_losses_by_target) > 1
            ):
                balanced_anchor_loss = Scalar(0.0)
                for (
                    target,
                    target_anchor_loss,
                ) in covered_anchor_losses_by_target.items():
                    balanced_anchor_loss = balanced_anchor_loss + (
                        target_anchor_loss / covered_anchor_counts_by_target[target]
                    )
                replay_loss = replay_loss + (
                    balanced_anchor_loss / len(covered_anchor_losses_by_target)
                )
            elif (
                preserve_covered_targets
                and not balance_covered_target_anchors
                and covered_anchor_count
            ):
                replay_loss = replay_loss + (
                    covered_anchor_loss / max(covered_anchor_count, 1)
                )
            loss = loss + replay_loss * replay_weight
        if floor_preservation_weight > 0.0 and floor_preservation_count:
            if (
                balance_floor_preservation_targets
                and floor_preservation_losses_by_target
            ):
                balanced_floor_loss = Scalar(0.0)
                for (
                    target,
                    target_floor_loss,
                ) in floor_preservation_losses_by_target.items():
                    balanced_floor_loss = balanced_floor_loss + (
                        target_floor_loss / floor_preservation_counts_by_target[target]
                    )
                loss = loss + (
                    balanced_floor_loss / len(floor_preservation_losses_by_target)
                ) * floor_preservation_weight
            else:
                loss = loss + (
                    floor_preservation_loss / max(floor_preservation_count, 1)
                ) * floor_preservation_weight
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_rank_margin(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        margin_weight: float,
        hard_negative_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        loss = Scalar(0.0)
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                loss = loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                loss = loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            hard_negatives = [
                index
                for index in sorted(
                    range(self.config.vocab_size),
                    key=lambda item: logits[item].data,
                    reverse=True,
                )
                if index != target
            ]
            if hard_negative_count > 0:
                hard_negatives = hard_negatives[:hard_negative_count]
            if margin_weight > 0.0 and hard_negatives:
                per_negative_weight = margin_weight / len(hard_negatives)
                target_logit = logits[target]
                for hard_negative in hard_negatives:
                    gap = logits[hard_negative] - target_logit + 1.0
                    loss = loss + (
                        (Scalar(1.0) + gap.exp()).log() * per_negative_weight
                    )
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def train_step_with_branch_topk_softmax(
        self,
        branches: list[tuple[list[int], int, int]],
        learning_rate: float,
        negative_weight: float,
        positive_weight: float,
        candidate_weight: float,
        candidate_count: int,
        params: list[Scalar] | None = None,
    ) -> float:
        params = self.parameters() if params is None else params
        zero_grad(params)
        loss = Scalar(0.0)
        for context, target, predicted in branches:
            logits = self._forward_scalars(context)
            probs = softmax_scalars(logits)
            if positive_weight > 0.0:
                loss = loss + (-probs[target].log()) * positive_weight
            if negative_weight > 0.0 and predicted != target:
                loss = loss + (
                    -(Scalar(1.0) - probs[predicted] + 1e-12).log()
                ) * negative_weight
            hard_candidates = [
                index
                for index in sorted(
                    range(self.config.vocab_size),
                    key=lambda item: logits[item].data,
                    reverse=True,
                )
                if index != target
            ]
            if candidate_count > 0:
                hard_candidates = hard_candidates[:candidate_count]
            candidate_ids = [target, *hard_candidates]
            if candidate_weight > 0.0 and len(candidate_ids) > 1:
                candidate_logits = [
                    logits[candidate_id] for candidate_id in candidate_ids
                ]
                candidate_probs = softmax_scalars(candidate_logits)
                loss = loss + (-candidate_probs[0].log()) * candidate_weight
        loss = loss / max(len(branches), 1)
        loss.backward()
        self.apply_gradients(params, learning_rate)
        return loss.data

    def generate(
        self,
        tokenizer: CharTokenizer,
        prompt: str,
        max_new_chars: int,
        temperature: float = 0.0,
        stop_at: str | None = None,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> str:
        config = GenerationConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return self.generate_with_trace(
            tokenizer,
            prompt,
            max_new_chars,
            config,
            stop_at=stop_at,
        )["text"]

    def generate_with_trace(
        self,
        tokenizer: CharTokenizer,
        prompt: str,
        max_new_chars: int,
        config: GenerationConfig | None = None,
        stop_at: str | None = None,
    ) -> dict[str, Any]:
        config = config or GenerationConfig()
        validate_generation_config(config)
        ids = tokenizer.encode(prompt)
        generated: list[int] = []
        rng = random.Random(self.config.seed + len(prompt))
        trace: list[dict[str, Any]] = []
        cache_enabled = config.use_kv_cache or self.config.use_kv_cache_path
        cache_events: list[dict[str, Any]] = []
        for _ in range(max_new_chars):
            context = make_context(ids, self.config.context_size, tokenizer.pad_id)
            if cache_enabled:
                cache_events.append(
                    {
                        "context_length": len(context),
                        "source_token_count": len(ids),
                        "sliding_window": len(ids) > self.config.context_size,
                    }
                )
            probs = self.predict(context)
            filtered_probs = generation_distribution(
                probs,
                generated,
                config,
            )
            if config.temperature <= 0:
                next_id = max(
                    range(len(filtered_probs)),
                    key=lambda index: filtered_probs[index],
                )
            else:
                next_id = sample_from_probs(filtered_probs, 1.0, rng)
            top_tokens = sorted(
                range(len(filtered_probs)),
                key=lambda index: filtered_probs[index],
                reverse=True,
            )[: config.trace_top_tokens]
            trace.append(
                {
                    "step": len(generated) + 1,
                    "context": tokenizer.decode(context),
                    "token_id": next_id,
                    "token": tokenizer.itos[next_id],
                    "probability": filtered_probs[next_id],
                    "raw_probability": probs[next_id],
                    "top_tokens": [
                        {
                            "token_id": token_id,
                            "token": tokenizer.itos[token_id],
                            "probability": filtered_probs[token_id],
                            "raw_probability": probs[token_id],
                        }
                        for token_id in top_tokens
                    ],
                }
            )
            ids.append(next_id)
            if stop_at is not None and tokenizer.itos[next_id] == stop_at:
                break
            generated.append(next_id)
        return {
            "text": tokenizer.decode(generated),
            "trace": trace,
            "generation_config": asdict(config),
            "cache": {
                "enabled": cache_enabled,
                "mode": "rolling-context-kv-aware" if cache_enabled else "disabled",
                "events": cache_events,
            },
        }

    def to_dict(
        self,
        tokenizer: CharTokenizer | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **checkpoint_header(self.config),
            "weights": {
                "token_embeddings": matrix_to_floats(self.token_embeddings),
                "position_embeddings": matrix_to_floats(self.position_embeddings),
                "wq": matrix_to_floats(self.wq),
                "bq": vector_to_floats(self.bq),
                "wk": matrix_to_floats(self.wk),
                "bk": vector_to_floats(self.bk),
                "wv": matrix_to_floats(self.wv),
                "bv": vector_to_floats(self.bv),
                "wo": matrix_to_floats(self.wo),
                "bo": vector_to_floats(self.bo),
                "w1": matrix_to_floats(self.w1),
                "b1": vector_to_floats(self.b1),
                "w_gate": matrix_to_floats(self.w_gate),
                "b_gate": vector_to_floats(self.b_gate),
                "w2": matrix_to_floats(self.w2),
                "b2": vector_to_floats(self.b2),
                "wout": matrix_to_floats(self.wout),
                "bout": vector_to_floats(self.bout),
                "context_projection_w": matrix_to_floats(self.context_projection_w),
                "context_projection_b": vector_to_floats(self.context_projection_b),
                "prompt_prefix_projection_w": matrix_to_floats(
                    self.prompt_prefix_projection_w
                ),
                "prompt_prefix_projection_b": vector_to_floats(
                    self.prompt_prefix_projection_b
                ),
                "prompt_position_projection_w": [
                    matrix_to_floats(position_weights)
                    for position_weights in self.prompt_position_projection_w
                ],
                "prompt_position_projection_b": vector_to_floats(
                    self.prompt_position_projection_b
                ),
                "prompt_summary_query": vector_to_floats(self.prompt_summary_query),
                "prompt_summary_w": matrix_to_floats(self.prompt_summary_w),
                "prompt_summary_b": vector_to_floats(self.prompt_summary_b),
                "ln1_gain": vector_to_floats(self.ln1_gain),
                "ln1_bias": vector_to_floats(self.ln1_bias),
                "ln2_gain": vector_to_floats(self.ln2_gain),
                "ln2_bias": vector_to_floats(self.ln2_bias),
                "final_ln_gain": vector_to_floats(self.final_ln_gain),
                "final_ln_bias": vector_to_floats(self.final_ln_bias),
                "extra_layers": [
                    self._block_to_floats(block)
                    for block in self.extra_blocks
                ],
            },
        }
        if metadata is not None:
            payload["metadata"] = metadata
        if tokenizer is not None:
            payload["tokenizer"] = tokenizer.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> tuple["TinyTransformerLM", CharTokenizer | None]:
        config = TransformerConfig(**payload["config"])
        model = cls(config, payload["weights"])
        tokenizer = None
        if "tokenizer" in payload:
            tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
        return model, tokenizer

    def save(
        self,
        path: Path,
        tokenizer: CharTokenizer | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(tokenizer, metadata), handle)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> tuple["TinyTransformerLM", CharTokenizer | None]:
        return cls.from_dict(load_checkpoint_payload(path))


@dataclass
class AnswerSelectorConfig:
    labels: list[str]
    features: list[str]
    seed: int = 17


class AnswerCandidateSelector:
    """Small closed-world candidate selector paired with transformer evidence."""

    def __init__(
        self,
        config: AnswerSelectorConfig,
        weights: list[list[float]],
        bias: list[float],
    ) -> None:
        self.config = config
        self.weights = weights
        self.bias = bias
        self.label_to_index = {label: index for index, label in enumerate(config.labels)}
        self.feature_to_index = {feature: index for index, feature in enumerate(config.features)}

    @classmethod
    def init_random(cls, config: AnswerSelectorConfig) -> "AnswerCandidateSelector":
        rng = random.Random(config.seed)
        weights = [
            [rng.uniform(-0.01, 0.01) for _ in config.features]
            for _ in config.labels
        ]
        return cls(config, weights, [0.0 for _ in config.labels])

    def featurize(self, prompt: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        for name in feature_names(prompt):
            index = self.feature_to_index.get(name)
            if index is None:
                continue
            counts[index] = counts.get(index, 0.0) + 1.0
        return counts

    def score(self, prompt: str, candidate: str) -> float:
        label_index = self.label_to_index.get(candidate)
        if label_index is None:
            return -math.inf
        return self._logit(label_index, self.featurize(prompt))

    def predict(self, prompt: str, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError("candidate selector requires at least one candidate")
        return max(candidates, key=lambda candidate: self.score(prompt, candidate))

    def loss(self, prompt: str, target: str, candidates: list[str] | None = None) -> float:
        candidate_labels = self._candidate_labels(target, candidates)
        features = self.featurize(prompt)
        logits = [self._logit(self.label_to_index[label], features) for label in candidate_labels]
        probs = softmax_floats(logits)
        target_offset = candidate_labels.index(target)
        return -math.log(max(probs[target_offset], 1e-12))

    def train_step(
        self,
        example: AnswerExample,
        learning_rate: float,
        candidates: list[str] | None = None,
    ) -> float:
        candidate_labels = self._candidate_labels(example.target, candidates)
        features = self.featurize(example.prompt)
        label_indices = [self.label_to_index[label] for label in candidate_labels]
        logits = [self._logit(label_index, features) for label_index in label_indices]
        probs = softmax_floats(logits)
        target_offset = candidate_labels.index(example.target)
        loss = -math.log(max(probs[target_offset], 1e-12))
        probs[target_offset] -= 1.0
        for label_index, grad in zip(label_indices, probs, strict=True):
            clipped_grad = max(min(grad, 5.0), -5.0)
            self.bias[label_index] -= learning_rate * clipped_grad
            for feature_index, value in features.items():
                self.weights[label_index][feature_index] -= learning_rate * clipped_grad * value
        return loss

    def _candidate_labels(self, target: str, candidates: list[str] | None) -> list[str]:
        labels = self.config.labels if candidates is None else candidates
        unique_labels = [label for label in dict.fromkeys(labels) if label in self.label_to_index]
        if target not in unique_labels:
            if target not in self.label_to_index:
                raise ValueError(f"target {target!r} is outside selector labels")
            unique_labels = [target, *unique_labels]
        return unique_labels

    def _logit(self, label_index: int, features: dict[int, float]) -> float:
        total = self.bias[label_index]
        row = self.weights[label_index]
        for feature_index, value in features.items():
            total += row[feature_index] * value
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": "closed-world-answer-candidate-selector",
            "config": asdict(self.config),
            "weights": self.weights,
            "bias": self.bias,
            "pretrained_weights": False,
            "external_embeddings": False,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AnswerCandidateSelector":
        return cls(
            AnswerSelectorConfig(**payload["config"]),
            payload["weights"],
            payload["bias"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "AnswerCandidateSelector":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def build_answer_selector(
    examples: list[AnswerExample],
    seed: int,
) -> AnswerCandidateSelector:
    labels = sorted({example.target for example in examples})
    features: set[str] = set()
    for example in examples:
        features.update(feature_names(example.prompt))
    config = AnswerSelectorConfig(labels=labels, features=sorted(features), seed=seed)
    return AnswerCandidateSelector.init_random(config)


GENERATOR_EOS = "<eos>"
GENERATOR_BOS = "<bos>"


@dataclass
class TransformerAnswerGeneratorConfig:
    labels: list[str]
    features: list[str]
    seed: int = 17
    max_answer_chars: int = 64
    transformer_top_k: int = 3


class TransformerGuidedAnswerGenerator:
    """Prompt-conditioned character generator with transformer-derived features."""

    def __init__(
        self,
        config: TransformerAnswerGeneratorConfig,
        weights: list[list[float]],
        bias: list[float],
    ) -> None:
        self.config = config
        self.weights = weights
        self.bias = bias
        self.label_to_index = {label: index for index, label in enumerate(config.labels)}
        self.feature_to_index = {feature: index for index, feature in enumerate(config.features)}

    @classmethod
    def init_random(
        cls,
        config: TransformerAnswerGeneratorConfig,
    ) -> "TransformerGuidedAnswerGenerator":
        rng = random.Random(config.seed)
        weights = [
            [rng.uniform(-0.01, 0.01) for _ in config.features]
            for _ in config.labels
        ]
        return cls(config, weights, [0.0 for _ in config.labels])

    def featurize(
        self,
        model: TinyTransformerLM,
        tokenizer: CharTokenizer,
        prompt: str,
        prefix: str,
    ) -> dict[int, float]:
        counts: dict[int, float] = {}
        for name in transformer_answer_generator_feature_names(
            model,
            tokenizer,
            prompt,
            prefix,
            self.config.transformer_top_k,
        ):
            index = self.feature_to_index.get(name)
            if index is None:
                continue
            counts[index] = counts.get(index, 0.0) + 1.0
        return counts

    def probabilities(
        self,
        model: TinyTransformerLM,
        tokenizer: CharTokenizer,
        prompt: str,
        prefix: str,
    ) -> list[float]:
        return softmax_floats(self._logits(self.featurize(model, tokenizer, prompt, prefix)))

    def predict_next(
        self,
        model: TinyTransformerLM,
        tokenizer: CharTokenizer,
        prompt: str,
        prefix: str,
    ) -> str:
        probs = self.probabilities(model, tokenizer, prompt, prefix)
        index = max(range(len(probs)), key=lambda item: probs[item])
        return self.config.labels[index]

    def generate(
        self,
        model: TinyTransformerLM,
        tokenizer: CharTokenizer,
        prompt: str,
    ) -> str:
        prefix = ""
        for _ in range(self.config.max_answer_chars):
            label = self.predict_next(model, tokenizer, prompt, prefix)
            if label == GENERATOR_EOS:
                break
            prefix += label
        return prefix

    def sequence_loss(
        self,
        model: TinyTransformerLM,
        tokenizer: CharTokenizer,
        prompt: str,
        target: str,
    ) -> float:
        prefix = ""
        total = 0.0
        labels = [*target, GENERATOR_EOS]
        for label in labels:
            probs = self.probabilities(model, tokenizer, prompt, prefix)
            total += -math.log(max(probs[self.label_to_index[label]], 1e-12))
            if label != GENERATOR_EOS:
                prefix += label
        return total / len(labels)

    def train_example(
        self,
        model: TinyTransformerLM,
        tokenizer: CharTokenizer,
        example: AnswerExample,
        learning_rate: float,
    ) -> float:
        prefix = ""
        total = 0.0
        labels = [*example.target, GENERATOR_EOS]
        for label in labels:
            target_index = self.label_to_index[label]
            features = self.featurize(model, tokenizer, example.prompt, prefix)
            probs = softmax_floats(self._logits(features))
            total += -math.log(max(probs[target_index], 1e-12))
            probs[target_index] -= 1.0
            for label_index, grad in enumerate(probs):
                clipped_grad = max(min(grad, 5.0), -5.0)
                self.bias[label_index] -= learning_rate * clipped_grad
                for feature_index, value in features.items():
                    self.weights[label_index][feature_index] -= (
                        learning_rate * clipped_grad * value
                    )
            if label != GENERATOR_EOS:
                prefix += label
        return total / len(labels)

    def _logits(self, features: dict[int, float]) -> list[float]:
        logits = self.bias[:]
        for label_index, row in enumerate(self.weights):
            total = logits[label_index]
            for feature_index, value in features.items():
                total += row[feature_index] * value
            logits[label_index] = total
        return logits

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": "transformer-guided-answer-generator",
            "config": asdict(self.config),
            "weights": self.weights,
            "bias": self.bias,
            "pretrained_weights": False,
            "external_embeddings": False,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransformerGuidedAnswerGenerator":
        return cls(
            TransformerAnswerGeneratorConfig(**payload["config"]),
            payload["weights"],
            payload["bias"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)
            handle.write("\n")

    @classmethod
    def load(cls, path: Path) -> "TransformerGuidedAnswerGenerator":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def transformer_answer_generator_feature_names(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    prefix: str,
    transformer_top_k: int,
) -> list[str]:
    names = feature_names(prompt)
    previous = prefix[-1] if prefix else GENERATOR_BOS
    previous_two = prefix[-2:] if len(prefix) >= 2 else GENERATOR_BOS
    names.extend(
        [
            f"pos:{len(prefix)}",
            f"prev:{previous}",
            f"prev2:{previous_two}",
            f"prefix:{prefix}",
        ]
    )
    context_ids = tokenizer.encode(prompt + prefix)
    context = make_context(context_ids, model.config.context_size, tokenizer.pad_id)
    probs = model.predict(context)
    top_count = max(0, min(transformer_top_k, len(probs)))
    top_ids = sorted(range(len(probs)), key=lambda index: probs[index], reverse=True)[
        :top_count
    ]
    for rank, token_id in enumerate(top_ids):
        token = tokenizer.itos[token_id]
        names.append(f"transformer_top:{rank}:{token!r}")
        if rank == 0:
            names.append(f"transformer_argmax:{token!r}")
    return names


def build_transformer_answer_generator(
    examples: list[AnswerExample],
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    seed: int,
    max_answer_chars: int,
    transformer_top_k: int,
) -> TransformerGuidedAnswerGenerator:
    labels = sorted({char for example in examples for char in example.target} | {GENERATOR_EOS})
    features: set[str] = set()
    for example in examples:
        prefix = ""
        for label in [*example.target, GENERATOR_EOS]:
            features.update(
                transformer_answer_generator_feature_names(
                    model,
                    tokenizer,
                    example.prompt,
                    prefix,
                    transformer_top_k,
                )
            )
            if label != GENERATOR_EOS:
                prefix += label
    config = TransformerAnswerGeneratorConfig(
        labels=labels,
        features=sorted(features),
        seed=seed,
        max_answer_chars=max_answer_chars,
        transformer_top_k=transformer_top_k,
    )
    return TransformerGuidedAnswerGenerator.init_random(config)


def transformer_answer_generator_training_pool(
    examples: list[AnswerExample],
) -> list[AnswerExample]:
    pool: list[AnswerExample] = []
    for example in examples:
        repeats = 1 + len(example.target) // 32
        if example.target != " unknown.":
            repeats += 1
        if (
            example.source.startswith("qa:")
            or example.source.startswith("fact:")
            or example.source.startswith("bridge:")
        ):
            repeats += 2
        if example.source.endswith(":place") or example.source.endswith(":color"):
            repeats += 4
        if example.source.endswith(":owner") or example.source.endswith(":training_data"):
            repeats += 4
        if example.source.endswith(":self") or example.source.endswith(":learning"):
            repeats += 55
        if example.source.endswith(":glossary"):
            repeats += 24
        pool.extend([example] * repeats)
    return pool


def evaluate_answer_generator_records(
    generator: TransformerGuidedAnswerGenerator,
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    total_loss = 0.0
    for record in records:
        completion = generator.generate(model, tokenizer, record["prompt"])
        loss = generator.sequence_loss(model, tokenizer, record["prompt"], record["target"])
        total_loss += loss
        scored.append(
            {
                "id": record["id"],
                "target": record["target"],
                "completion": completion,
                "exact_match": completion == record["target"],
                "target_loss": loss,
                "completion_source": "transformer_guided_generator",
            }
        )
    exact = sum(1 for record in scored if record["exact_match"])
    failed = [record for record in scored if not record["exact_match"]]
    return {
        "count": len(scored),
        "exact": exact,
        "exact_rate": exact / len(scored) if scored else 0.0,
        "avg_target_loss": total_loss / len(scored) if scored else 0.0,
        "failed_records": failed,
    }


GeneratorLesson = list[tuple[int, dict[int, float]]]


def transformer_answer_generator_lesson(
    generator: TransformerGuidedAnswerGenerator,
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
) -> GeneratorLesson:
    lesson: GeneratorLesson = []
    prefix = ""
    for label in [*example.target, GENERATOR_EOS]:
        lesson.append(
            (
                generator.label_to_index[label],
                generator.featurize(model, tokenizer, example.prompt, prefix),
            )
        )
        if label != GENERATOR_EOS:
            prefix += label
    return lesson


def train_transformer_answer_generator_lesson(
    generator: TransformerGuidedAnswerGenerator,
    lesson: GeneratorLesson,
    learning_rate: float,
) -> float:
    total = 0.0
    for target_index, features in lesson:
        probs = softmax_floats(generator._logits(features))
        total += -math.log(max(probs[target_index], 1e-12))
        probs[target_index] -= 1.0
        for label_index, grad in enumerate(probs):
            clipped_grad = max(min(grad, 5.0), -5.0)
            generator.bias[label_index] -= learning_rate * clipped_grad
            for feature_index, value in features.items():
                generator.weights[label_index][feature_index] -= (
                    learning_rate * clipped_grad * value
                )
    return total / max(len(lesson), 1)


def matrix_to_scalars(values: list[list[float]]) -> list[list[Scalar]]:
    return [[Scalar(value) for value in row] for row in values]


def vector_to_scalars(values: list[float]) -> list[Scalar]:
    return [Scalar(value) for value in values]


def flatten_scalars(item: Any) -> list[Scalar]:
    if isinstance(item, Scalar):
        return [item]
    scalars: list[Scalar] = []
    for value in item:
        scalars.extend(flatten_scalars(value))
    return scalars


def exclude_scalars(params: list[Scalar], excluded: Any) -> list[Scalar]:
    excluded_ids = {id(value) for value in flatten_scalars(excluded)}
    return [param for param in params if id(param) not in excluded_ids]


def matrix_to_floats(values: list[list[Scalar]]) -> list[list[float]]:
    return [[value.data for value in row] for row in values]


def vector_to_floats(values: list[Scalar]) -> list[float]:
    return [value.data for value in values]


def linear_scalars(
    inputs: list[Scalar],
    weights: list[list[Scalar]],
    bias: list[Scalar],
) -> list[Scalar]:
    outputs: list[Scalar] = []
    for output_index, bias_value in enumerate(bias):
        total = bias_value
        for input_index, value in enumerate(inputs):
            total = total + value * weights[input_index][output_index]
        outputs.append(total)
    return outputs


def linear_floats(inputs: list[float], weights: list[list[float]], bias: list[float]) -> list[float]:
    outputs: list[float] = []
    for output_index, bias_value in enumerate(bias):
        total = bias_value
        for input_index, value in enumerate(inputs):
            total += value * weights[input_index][output_index]
        outputs.append(total)
    return outputs


def layer_norm_scalars(
    values: list[Scalar],
    gain: list[Scalar],
    bias: list[Scalar],
    epsilon: float,
) -> list[Scalar]:
    count = max(len(values), 1)
    mean = Scalar(0.0)
    for value in values:
        mean = mean + value
    mean = mean / count
    variance = Scalar(0.0)
    for value in values:
        centered = value - mean
        variance = variance + centered * centered
    variance = variance / count
    scale = (variance + epsilon).pow(-0.5)
    return [
        (value - mean) * scale * gain[index] + bias[index]
        for index, value in enumerate(values)
    ]


def layer_norm_floats(
    values: list[float],
    gain: list[float],
    bias: list[float],
    epsilon: float,
) -> list[float]:
    count = max(len(values), 1)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    scale = 1.0 / math.sqrt(variance + epsilon)
    return [
        (value - mean) * scale * gain[index] + bias[index]
        for index, value in enumerate(values)
    ]


def rms_norm_scalars(
    values: list[Scalar],
    gain: list[Scalar],
    epsilon: float,
) -> list[Scalar]:
    count = max(len(values), 1)
    mean_square = Scalar(0.0)
    for value in values:
        mean_square = mean_square + value * value
    mean_square = mean_square / count
    scale = (mean_square + epsilon).pow(-0.5)
    return [
        value * scale * gain[index]
        for index, value in enumerate(values)
    ]


def rms_norm_floats(
    values: list[float],
    gain: list[float],
    epsilon: float,
) -> list[float]:
    count = max(len(values), 1)
    mean_square = sum(value * value for value in values) / count
    scale = 1.0 / math.sqrt(mean_square + epsilon)
    return [
        value * scale * gain[index]
        for index, value in enumerate(values)
    ]


def dot_scalars(left: list[Scalar], right: list[Scalar]) -> Scalar:
    total = Scalar(0.0)
    for left_value, right_value in zip(left, right):
        total = total + left_value * right_value
    return total


def dot_floats(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def softmax_scalars(logits: list[Scalar]) -> list[Scalar]:
    max_logit = max(logit.data for logit in logits)
    exps = [(logit - max_logit).exp() for logit in logits]
    total = Scalar(0.0)
    for value in exps:
        total = total + value
    return [value / total for value in exps]


def softmax_floats(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def cross_entropy_scalars(logits: list[Scalar], target: int) -> Scalar:
    probs = softmax_scalars(logits)
    return -probs[target].log()


def sample_from_probs(probs: list[float], temperature: float, rng: random.Random) -> int:
    adjusted = [pow(max(prob, 1e-12), 1.0 / temperature) for prob in probs]
    total = sum(adjusted)
    threshold = rng.random() * total
    running = 0.0
    for index, prob in enumerate(adjusted):
        running += prob
        if running >= threshold:
            return index
    return len(probs) - 1


def generation_distribution(
    probs: list[float],
    generated_ids: list[int],
    config: GenerationConfig,
) -> list[float]:
    adjusted = list(probs)
    if config.repetition_penalty != 1.0:
        generated = set(generated_ids)
        for token_id in generated:
            adjusted[token_id] = adjusted[token_id] / config.repetition_penalty
    if config.top_k > 0 and config.top_k < len(adjusted):
        keep = set(
            sorted(
                range(len(adjusted)),
                key=lambda index: adjusted[index],
                reverse=True,
            )[: config.top_k]
        )
        adjusted = [
            value if index in keep else 0.0
            for index, value in enumerate(adjusted)
        ]
    if config.top_p < 1.0:
        ranked = sorted(
            range(len(adjusted)),
            key=lambda index: adjusted[index],
            reverse=True,
        )
        keep: set[int] = set()
        cumulative = 0.0
        for index in ranked:
            keep.add(index)
            cumulative += adjusted[index]
            if cumulative >= config.top_p:
                break
        adjusted = [
            value if index in keep else 0.0
            for index, value in enumerate(adjusted)
        ]
    if config.temperature > 0.0 and config.temperature != 1.0:
        adjusted = [
            pow(max(value, 1e-12), 1.0 / config.temperature)
            if value > 0.0
            else 0.0
            for value in adjusted
        ]
    total = sum(adjusted)
    if total <= 0.0:
        return probs
    return [value / total for value in adjusted]


def average_nll(
    model: TinyTransformerLM,
    ids: list[int],
    pad_id: int,
    limit: int | None = None,
) -> float:
    if not ids:
        return 0.0
    count = min(len(ids), limit) if limit else len(ids)
    total = 0.0
    for position in range(count):
        context = context_before(ids, position, model.config.context_size, pad_id)
        total += model.nll(context, ids[position])
    return total / count


def ensure_curriculum(corpus_path: Path, valid_path: Path) -> None:
    if corpus_path.exists() and valid_path.exists():
        return
    curriculum = build_curriculum()
    write_curriculum(curriculum, DEFAULT_OUTPUT_DIR)


def transformer_training_recipe(
    args: argparse.Namespace,
    tokenizer: CharTokenizer,
    planned_artifacts: list[Path],
    acceptance_gates: list[dict[str, Any]],
    replay_plan_path: Path | None = None,
) -> dict[str, Any]:
    return build_transformer_training_recipe(
        args,
        tokenizer,
        planned_artifacts,
        acceptance_gates,
        asdict(transformer_config_from_args(args, tokenizer.vocab_size)),
        asdict(optimization_config_from_args(args)),
        asdict(generation_config_from_args(args)),
        replay_plan_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train the tiny transformer")
    train_parser.add_argument("--corpus", type=Path, default=DEFAULT_OUTPUT_DIR / "train.txt")
    train_parser.add_argument("--valid", type=Path, default=DEFAULT_OUTPUT_DIR / "valid.txt")
    train_parser.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    train_parser.add_argument("--steps", type=int, default=80)
    train_parser.add_argument("--learning-rate", type=float, default=0.03)
    train_parser.add_argument("--context-size", type=int, default=16)
    train_parser.add_argument("--embedding-dim", type=int, default=8)
    train_parser.add_argument("--feedforward-dim", type=int, default=16)
    train_parser.add_argument("--num-layers", type=int, default=1)
    train_parser.add_argument("--attention-heads", type=int, default=1)
    train_parser.add_argument("--use-layer-norm", action="store_true")
    train_parser.add_argument(
        "--use-pre-layer-norm",
        action="store_true",
        help=(
            "Use GPT-style pre-layer normalization in transformer blocks and "
            "apply a final layer norm before the language-model head."
        ),
    )
    train_parser.add_argument("--use-rms-norm", action="store_true")
    train_parser.add_argument("--layer-norm-epsilon", type=float, default=1e-5)
    train_parser.add_argument("--use-gated-mlp", action="store_true")
    train_parser.add_argument("--tie-output-embeddings", action="store_true")
    train_parser.add_argument("--use-rotary-positions", action="store_true")
    train_parser.add_argument("--use-kv-cache-path", action="store_true")
    train_parser.add_argument(
        "--use-context-mean",
        action="store_true",
        help="Add a mean-pooled context residual to the final transformer representation.",
    )
    train_parser.add_argument(
        "--use-context-projection",
        action="store_true",
        help=(
            "Add a trainable zero-initialized projection of the mean-pooled "
            "context to the final transformer representation."
        ),
    )
    train_parser.add_argument(
        "--use-prompt-prefix-projection",
        action="store_true",
        help=(
            "Add a trainable zero-initialized projection of non-padding prompt "
            "prefix positions before the final token."
        ),
    )
    train_parser.add_argument(
        "--use-prompt-position-projection",
        action="store_true",
        help=(
            "Add a trainable zero-initialized position-specific projection of "
            "non-padding prompt prefix positions before the final token."
        ),
    )
    train_parser.add_argument(
        "--prompt-position-projection-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the prompt-position projection residual before adding it to "
            "the final representation."
        ),
    )
    train_parser.add_argument(
        "--use-prompt-attention-summary",
        action="store_true",
        help=(
            "Add a trainable attention-pooled context summary to the final "
            "transformer representation through a zero-initialized projection."
        ),
    )
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--eval-every", type=int, default=20)
    train_parser.add_argument("--valid-limit", type=int, default=256)
    train_parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    train_parser.add_argument("--gradient-clip", type=float, default=5.0)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--adam-beta1", type=float, default=0.9)
    train_parser.add_argument("--adam-beta2", type=float, default=0.999)
    train_parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    train_parser.add_argument("--warmup-steps", type=int, default=0)
    train_parser.add_argument("--decay-steps", type=int, default=0)
    train_parser.add_argument("--min-learning-rate", type=float, default=0.0)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--resume-checkpoint", type=Path, default=None)
    train_parser.add_argument("--resume-optimizer", type=Path, default=None)

    eval_parser = subparsers.add_parser("eval", help="evaluate the tiny transformer")
    eval_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    eval_parser.add_argument("--max-new-chars", type=int, default=24)
    eval_parser.add_argument("--json", type=Path, default=None)
    eval_parser.add_argument("--samples-jsonl", type=Path, default=None)
    eval_parser.add_argument("--temperature", type=float, default=0.0)
    eval_parser.add_argument("--top-k", type=int, default=0)
    eval_parser.add_argument("--top-p", type=float, default=1.0)
    eval_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    eval_parser.add_argument("--trace-top-tokens", type=int, default=5)
    eval_parser.add_argument("--use-kv-cache", action="store_true")
    eval_parser.add_argument(
        "--probe",
        action="append",
        type=Path,
        default=None,
        help="JSONL probe file. Defaults to qa, unknowns, heldout, and paraphrases.",
    )

    answer_parser = subparsers.add_parser(
        "answer-train",
        help="train the tiny transformer on corpus-derived answer lessons",
    )
    answer_parser.add_argument("--train-text", type=Path, default=DEFAULT_TRAIN_TEXT)
    answer_parser.add_argument("--valid", type=Path, default=DEFAULT_OUTPUT_DIR / "valid.txt")
    answer_parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    answer_parser.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    answer_parser.add_argument("--steps", type=int, default=400)
    answer_parser.add_argument("--learning-rate", type=float, default=0.04)
    answer_parser.add_argument("--target-loss-weight", type=float, default=1.0)
    answer_parser.add_argument("--choice-loss-weight", type=float, default=0.0)
    answer_parser.add_argument(
        "--choice-negatives",
        type=int,
        default=0,
        help="Wrong answer candidates sampled for each contrastive choice step.",
    )
    answer_parser.add_argument(
        "--choice-max-chars",
        type=int,
        default=0,
        help="Limit contrastive candidate loss to the first N answer chars. 0 uses the full answer.",
    )
    answer_parser.add_argument(
        "--selector-steps",
        type=int,
        default=0,
        help="Train a closed-world answer candidate selector alongside transformer evidence.",
    )
    answer_parser.add_argument("--selector-learning-rate", type=float, default=0.08)
    answer_parser.add_argument(
        "--selector-negatives",
        type=int,
        default=0,
        help="Wrong selector candidates sampled per selector step. 0 trains against all labels.",
    )
    answer_parser.add_argument("--selector-eval-every", type=int, default=200)
    answer_parser.add_argument(
        "--selector-emit-completions",
        action="store_true",
        help="Record selector-chosen candidates as emitted completions for exact-match evidence.",
    )
    answer_parser.add_argument(
        "--generator-steps",
        type=int,
        default=0,
        help="Train a transformer-guided character answer generator without answer candidates.",
    )
    answer_parser.add_argument("--generator-learning-rate", type=float, default=0.08)
    answer_parser.add_argument("--generator-eval-every", type=int, default=200)
    answer_parser.add_argument("--generator-max-answer-chars", type=int, default=64)
    answer_parser.add_argument("--generator-transformer-top-k", type=int, default=3)
    answer_parser.add_argument(
        "--direct-answer-steps",
        type=int,
        default=0,
        help="Continue training transformer weights for greedy answer completion.",
    )
    answer_parser.add_argument("--direct-answer-learning-rate", type=float, default=0.035)
    answer_parser.add_argument("--direct-answer-eval-every", type=int, default=200)
    answer_parser.add_argument("--direct-answer-max-new-chars", type=int, default=96)
    answer_parser.add_argument(
        "--direct-answer-snapshot-mode",
        choices=["full", "branch-only"],
        default="full",
        help=(
            "Direct-answer JSONL snapshot detail. 'branch-only' skips greedy "
            "completion evals and records only branch profiles and branch-context "
            "coverage for bounded screening runs."
        ),
    )
    answer_parser.add_argument(
        "--direct-answer-mode",
        choices=DIRECT_ANSWER_OBJECTIVE_MODES,
        default="first-error",
        help="Direct transformer update policy for greedy answer completion.",
    )
    answer_parser.add_argument("--direct-answer-negative-weight", type=float, default=0.5)
    answer_parser.add_argument("--direct-answer-positive-weight", type=float, default=1.0)
    answer_parser.add_argument("--direct-answer-contrast-weight", type=float, default=1.0)
    answer_parser.add_argument("--direct-answer-recovery-steps", type=int, default=3)
    answer_parser.add_argument("--direct-answer-branch-position", type=int, default=1)
    answer_parser.add_argument("--direct-answer-branch-span", type=int, default=1)
    answer_parser.add_argument("--direct-answer-branch-batch-size", type=int, default=4)
    answer_parser.add_argument("--direct-answer-hard-negatives", type=int, default=16)
    answer_parser.add_argument("--direct-answer-train-top-layer-only", action="store_true")
    answer_parser.add_argument(
        "--direct-answer-freeze-output-bias",
        action="store_true",
        help=(
            "Exclude the transformer output bias from direct-answer updates so "
            "branch screens cannot improve loss by moving one global token bias."
        ),
    )
    answer_parser.add_argument(
        "--direct-answer-restore-best-branch-snapshot",
        action="store_true",
        help=(
            "Restore the direct-answer weights with the best branch-diversity "
            "snapshot score before final metrics and checkpoint writing."
        ),
    )
    answer_parser.add_argument(
        "--direct-answer-require-branch-context-gate",
        action="store_true",
        help=(
            "Skip direct-answer training unless branch contexts have complete "
            "semantic coverage, no ambiguous target-token contexts, and no skipped records."
        ),
    )
    answer_parser.add_argument(
        "--skip-post-direct-snapshot",
        action="store_true",
        help=(
            "Skip the full answer-candidate snapshot after direct-answer updates. "
            "Use only for bounded screening runs; promotion evidence should keep "
            "the default full post-direct snapshot."
        ),
    )
    answer_parser.add_argument("--direct-answer-sequence-interval", type=int, default=50)
    answer_parser.add_argument("--direct-answer-rollout-interval", type=int, default=5)
    answer_parser.add_argument(
        "--direct-answer-terminator",
        type=str,
        default=ANSWER_TERMINATOR,
        help="Single admitted character that stops direct answer generation.",
    )
    answer_parser.add_argument("--context-size", type=int, default=16)
    answer_parser.add_argument("--embedding-dim", type=int, default=8)
    answer_parser.add_argument("--feedforward-dim", type=int, default=16)
    answer_parser.add_argument("--num-layers", type=int, default=1)
    answer_parser.add_argument("--attention-heads", type=int, default=1)
    answer_parser.add_argument("--use-layer-norm", action="store_true")
    answer_parser.add_argument(
        "--use-pre-layer-norm",
        action="store_true",
        help=(
            "Use GPT-style pre-layer normalization in transformer blocks and "
            "apply a final layer norm before the language-model head."
        ),
    )
    answer_parser.add_argument("--use-rms-norm", action="store_true")
    answer_parser.add_argument("--layer-norm-epsilon", type=float, default=1e-5)
    answer_parser.add_argument("--use-gated-mlp", action="store_true")
    answer_parser.add_argument("--tie-output-embeddings", action="store_true")
    answer_parser.add_argument("--use-rotary-positions", action="store_true")
    answer_parser.add_argument("--use-kv-cache-path", action="store_true")
    answer_parser.add_argument(
        "--use-context-mean",
        action="store_true",
        help="Add a mean-pooled context residual to the final transformer representation.",
    )
    answer_parser.add_argument(
        "--use-context-projection",
        action="store_true",
        help=(
            "Add a trainable zero-initialized projection of the mean-pooled "
            "context to the final transformer representation."
        ),
    )
    answer_parser.add_argument(
        "--use-prompt-prefix-projection",
        action="store_true",
        help=(
            "Add a trainable zero-initialized projection of non-padding prompt "
            "prefix positions before the final token."
        ),
    )
    answer_parser.add_argument(
        "--use-prompt-position-projection",
        action="store_true",
        help=(
            "Add a trainable zero-initialized position-specific projection of "
            "non-padding prompt prefix positions before the final token."
        ),
    )
    answer_parser.add_argument(
        "--prompt-position-projection-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the prompt-position projection residual before adding it to "
            "the final representation."
        ),
    )
    answer_parser.add_argument(
        "--use-prompt-attention-summary",
        action="store_true",
        help=(
            "Add a trainable attention-pooled context summary to the final "
            "transformer representation through a zero-initialized projection."
        ),
    )
    answer_parser.add_argument("--seed", type=int, default=17)
    answer_parser.add_argument("--eval-every", type=int, default=100)
    answer_parser.add_argument("--max-new-chars", type=int, default=48)
    answer_parser.add_argument("--temperature", type=float, default=0.0)
    answer_parser.add_argument("--top-k", type=int, default=0)
    answer_parser.add_argument("--top-p", type=float, default=1.0)
    answer_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    answer_parser.add_argument("--trace-top-tokens", type=int, default=5)
    answer_parser.add_argument("--use-kv-cache", action="store_true")
    answer_parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    answer_parser.add_argument("--gradient-clip", type=float, default=5.0)
    answer_parser.add_argument("--weight-decay", type=float, default=0.0)
    answer_parser.add_argument("--adam-beta1", type=float, default=0.9)
    answer_parser.add_argument("--adam-beta2", type=float, default=0.999)
    answer_parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    answer_parser.add_argument("--warmup-steps", type=int, default=0)
    answer_parser.add_argument("--decay-steps", type=int, default=0)
    answer_parser.add_argument("--min-learning-rate", type=float, default=0.0)
    answer_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    answer_parser.add_argument("--resume-checkpoint", type=Path, default=None)
    answer_parser.add_argument("--resume-optimizer", type=Path, default=None)
    answer_parser.add_argument(
        "--candidate-scope",
        choices=["all", "eval"],
        default="eval",
        help="Candidate set for answer snapshots. 'eval' scores against targets in the current eval set.",
    )
    answer_parser.add_argument(
        "--include-completions",
        action="store_true",
        help="Generate free-form completions during answer snapshots. Slower, but records exact generation.",
    )
    answer_parser.add_argument("--experiment-version", default=TRANSFORMER_RECIPE_VERSION)
    answer_parser.add_argument("--experiment-hypothesis", default=None)
    answer_parser.add_argument(
        "--experiment-acceptance-gate",
        action="append",
        default=None,
        help="Additional required experiment gate formatted as name:rule.",
    )
    answer_parser.add_argument(
        "--experiment-failure-criterion",
        action="append",
        default=None,
        help="Additional failure criterion for this screen.",
    )
    answer_parser.add_argument("--experiment-note", action="append", default=None)
    return parser.parse_args(argv)


def load_optimizer_state(
    path: Path | None,
    config: OptimizationConfig,
) -> ScalarOptimizer:
    if path is None:
        return ScalarOptimizer(config)
    with path.open("r", encoding="utf-8") as handle:
        optimizer = ScalarOptimizer.from_dict(json.load(handle))
    if asdict(optimizer.config) != asdict(config):
        raise ValueError("resume optimizer config does not match requested optimizer config")
    return optimizer


def save_optimizer_state(path: Path, optimizer: ScalarOptimizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(optimizer.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def initialize_transformer_for_training(
    args: argparse.Namespace,
    tokenizer: CharTokenizer,
) -> tuple[TinyTransformerLM, dict[str, Any]]:
    if args.resume_checkpoint is None:
        config = transformer_config_from_args(args, tokenizer.vocab_size)
        return TinyTransformerLM.init_random(config), {"resumed": False}
    model, checkpoint_tokenizer = TinyTransformerLM.load(args.resume_checkpoint)
    if checkpoint_tokenizer is None:
        raise ValueError("resume checkpoint does not contain a tokenizer")
    if checkpoint_tokenizer.to_dict() != tokenizer.to_dict():
        raise ValueError("resume checkpoint tokenizer does not match admitted training tokenizer")
    requested_config = transformer_config_from_args(args, tokenizer.vocab_size)
    if asdict(model.config) != asdict(requested_config):
        raise ValueError("resume checkpoint config does not match requested transformer config")
    return model, {
        "resumed": True,
        "resume_checkpoint": str(args.resume_checkpoint),
    }


def train_transformer(args: argparse.Namespace) -> dict[str, Any]:
    ensure_curriculum(args.corpus, args.valid)
    train_text = args.corpus.read_text(encoding="utf-8")
    valid_text = args.valid.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.train(train_text)
    train_ids = tokenizer.encode(train_text)
    valid_ids = tokenizer.encode(valid_text)
    model, resume_metadata = initialize_transformer_for_training(args, tokenizer)
    optimizer = load_optimizer_state(
        args.resume_optimizer,
        optimization_config_from_args(args),
    )
    model.active_optimizer = optimizer
    rng = random.Random(args.seed)
    args.run.mkdir(parents=True, exist_ok=True)
    history_path = args.run / "transformer_metrics.jsonl"

    def write_history(step: int, train_nll: float | None) -> dict[str, Any]:
        record = {
            "step": step,
            "train_nll": train_nll,
            "valid_nll": average_nll(model, valid_ids, tokenizer.pad_id, args.valid_limit),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    baseline = write_history(step=0, train_nll=None)
    running_loss = 0.0
    last_history = baseline
    last_history_step = 0
    for step in range(1, args.steps + 1):
        position = rng.randrange(len(train_ids))
        context = context_before(train_ids, position, args.context_size, tokenizer.pad_id)
        running_loss += model.train_step(context, train_ids[position], args.learning_rate)
        if args.eval_every > 0 and step % args.eval_every == 0:
            train_loss = running_loss / args.eval_every
            last_history = write_history(step=step, train_nll=train_loss)
            last_history_step = step
            print(
                f"step={step} train_nll={train_loss:.4f} "
                f"valid_nll={last_history['valid_nll']:.4f}"
            )
            running_loss = 0.0

    if last_history_step != args.steps:
        last_history = write_history(step=args.steps, train_nll=None)

    checkpoint_path = args.run / "transformer.json"
    optimizer_path = args.run / "optimizer_state.json"
    save_optimizer_state(optimizer_path, optimizer)
    checkpoint_metadata = transformer_run_metadata(
        args,
        tokenizer,
        optimizer,
        "language-model",
        resume_metadata,
    )
    model.save(checkpoint_path, tokenizer, checkpoint_metadata)
    tokenizer.save(args.run / "tokenizer.json")
    metrics = {
        "architecture": TRANSFORMER_ARCHITECTURE,
        "checkpoint": str(checkpoint_path),
        "checkpoint_format": TRANSFORMER_CHECKPOINT_FORMAT,
        "optimizer_state": str(optimizer_path),
        "optimizer": optimizer.summary(),
        "resume": resume_metadata,
        "history": str(history_path),
        "steps": args.steps,
        "train_chars": len(train_text),
        "valid_chars": len(valid_text),
        "vocab_size": tokenizer.vocab_size,
        "context_size": args.context_size,
        "embedding_dim": args.embedding_dim,
        "feedforward_dim": args.feedforward_dim,
        "num_layers": args.num_layers,
        "attention_heads": args.attention_heads,
        "use_layer_norm": args.use_layer_norm,
        "use_pre_layer_norm": args.use_pre_layer_norm,
        "use_rms_norm": args.use_rms_norm,
        "layer_norm_epsilon": args.layer_norm_epsilon,
        "use_gated_mlp": args.use_gated_mlp,
        "tie_output_embeddings": args.tie_output_embeddings,
        "use_rotary_positions": args.use_rotary_positions,
        "use_kv_cache_path": args.use_kv_cache_path,
        "use_context_mean": args.use_context_mean,
        "use_context_projection": args.use_context_projection,
        "use_prompt_prefix_projection": args.use_prompt_prefix_projection,
        "use_prompt_position_projection": args.use_prompt_position_projection,
        "prompt_position_projection_scale": args.prompt_position_projection_scale,
        "use_prompt_attention_summary": args.use_prompt_attention_summary,
        "baseline_valid_nll": baseline["valid_nll"],
        "final_valid_nll": last_history["valid_nll"],
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "tokenizer": TRANSFORMER_TOKENIZER,
    }
    with (args.run / "transformer_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"saved {checkpoint_path}")
    return metrics


def eval_transformer(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer = TinyTransformerLM.load(args.checkpoint)
    if tokenizer is None:
        raise ValueError("checkpoint does not contain a tokenizer")
    generation_config = generation_config_from_args(args)
    probe_paths = args.probe if args.probe is not None else DEFAULT_PROBES
    probe_records = load_probe_records(probe_paths)
    candidates = eval_candidates_from_records(probe_records)
    scored_by_eval = score_transformer_evals(
        model,
        tokenizer,
        probe_records,
        args.max_new_chars,
        generation_config,
        candidates,
    )
    result = build_transformer_eval_report(
        args.checkpoint,
        probe_paths,
        probe_records,
        scored_by_eval,
        candidates,
        generation_config,
        args.samples_jsonl,
    )
    if args.samples_jsonl:
        write_eval_samples(args.samples_jsonl, scored_by_eval)
    if args.json:
        write_eval_report(args.json, result)
    return result


def answer_sequence_nll(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
) -> float:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(example.target)
    ids = prompt_ids[:]
    total = 0.0
    for target_id in target_ids:
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        total += model.nll(context, target_id)
        ids.append(target_id)
    return total / max(len(target_ids), 1)


def answer_sequence_loss_scalars(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    target: str,
    max_chars: int = 0,
) -> Scalar:
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(target)
    if max_chars > 0:
        target_ids = target_ids[:max_chars]
    ids = prompt_ids[:]
    total = Scalar(0.0)
    for target_id in target_ids:
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        total = total + cross_entropy_scalars(model._forward_scalars(context), target_id)
        ids.append(target_id)
    return total / max(len(target_ids), 1)


DirectAnswerLesson = list[tuple[list[int], int]]
DirectAnswerRepair = tuple[list[int], int, int, int]
DirectAnswerBranchContrast = tuple[list[int], int, list[int], int]


def answer_completion_text(target: str, terminator: str = ANSWER_TERMINATOR) -> str:
    return f"{target}{terminator}" if terminator else target


def direct_answer_lesson(
    tokenizer: CharTokenizer,
    context_size: int,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> DirectAnswerLesson:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    lesson: DirectAnswerLesson = []
    ids = prompt_ids[:]
    for target_id in target_ids:
        lesson.append((make_context(ids, context_size, tokenizer.pad_id), target_id))
        ids.append(target_id)
    return lesson


def direct_answer_sequence_nll(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> float:
    lesson = direct_answer_lesson(tokenizer, model.config.context_size, example, terminator)
    total = 0.0
    for context, target_id in lesson:
        total += model.nll(context, target_id)
    return total / max(len(lesson), 1)


def direct_answer_branch_profile(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
    max_failed_records: int = 12,
) -> dict[str, Any]:
    predicted_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    confusion_counts: Counter[str] = Counter()
    failed_records: list[dict[str, Any]] = []
    total_target_prob = 0.0
    total_predicted_prob = 0.0
    total_target_margin = 0.0
    total_target_rank = 0.0
    target_top3 = 0
    target_top5 = 0
    profiled = 0
    correct = 0
    skipped = 0

    for record in records:
        example = AnswerExample(
            prompt=record["prompt"],
            target=record["target"],
            source=f"eval:{record['id']}",
        )
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            example,
            branch_position,
            terminator,
        )
        if branch is None:
            skipped += 1
            continue
        context, target_id, position = branch
        probs = model.predict(context)
        ranked_ids = sorted(
            range(len(probs)),
            key=lambda index: (-probs[index], tokenizer.itos[index], index),
        )
        predicted_id = ranked_ids[0]
        target_rank = ranked_ids.index(target_id) + 1
        target_prob = probs[target_id]
        predicted_prob = probs[predicted_id]
        strongest_non_target = max(
            (prob for index, prob in enumerate(probs) if index != target_id),
            default=0.0,
        )
        target_margin = target_prob - strongest_non_target
        target_token = tokenizer.itos[target_id]
        predicted_token = tokenizer.itos[predicted_id]

        profiled += 1
        total_target_prob += target_prob
        total_predicted_prob += predicted_prob
        total_target_margin += target_margin
        total_target_rank += target_rank
        if target_rank <= 3:
            target_top3 += 1
        if target_rank <= 5:
            target_top5 += 1
        target_counts[target_token] += 1
        predicted_counts[predicted_token] += 1
        confusion_counts[f"{target_token!r}->{predicted_token!r}"] += 1
        if predicted_id == target_id:
            correct += 1
        elif len(failed_records) < max_failed_records:
            failed_records.append(
                {
                    "id": record["id"],
                    "target": record["target"],
                    "branch_position": position,
                    "target_token": target_token,
                    "predicted_token": predicted_token,
                    "target_prob": target_prob,
                    "predicted_prob": predicted_prob,
                    "target_margin": target_margin,
                    "target_rank": target_rank,
                    "top_predictions": [
                        {
                            "token": tokenizer.itos[index],
                            "prob": probs[index],
                        }
                        for index in ranked_ids[:5]
                    ],
                }
            )

    def top_items(counter: Counter[str]) -> list[dict[str, Any]]:
        return [
            {"value": value, "count": count}
            for value, count in counter.most_common(12)
        ]

    target_token_values = set(target_counts)
    predicted_token_values = set(predicted_counts)
    covered_target_tokens = target_token_values & predicted_token_values
    dominant_predicted_token = None
    dominant_predicted_count = 0
    if predicted_counts:
        dominant_predicted_token, dominant_predicted_count = (
            predicted_counts.most_common(1)[0]
        )
    missing_target_tokens = [
        {"value": value, "count": count}
        for value, count in target_counts.most_common()
        if value not in predicted_token_values
    ]
    target_unique = len(target_token_values)
    predicted_unique = len(predicted_token_values)

    return {
        "branch_position": branch_position,
        "count": profiled,
        "skipped": skipped,
        "correct": correct,
        "accuracy": correct / profiled if profiled else 0.0,
        "avg_target_prob": total_target_prob / profiled if profiled else 0.0,
        "avg_predicted_prob": total_predicted_prob / profiled if profiled else 0.0,
        "avg_target_margin": total_target_margin / profiled if profiled else 0.0,
        "target_rank": {
            "avg": total_target_rank / profiled if profiled else 0.0,
            "top1_rate": correct / profiled if profiled else 0.0,
            "top3_rate": target_top3 / profiled if profiled else 0.0,
            "top5_rate": target_top5 / profiled if profiled else 0.0,
            "vocab_size": model.config.vocab_size,
        },
        "target_tokens": top_items(target_counts),
        "predicted_tokens": top_items(predicted_counts),
        "confusions": top_items(confusion_counts),
        "diversity": {
            "target_unique": target_unique,
            "predicted_unique": predicted_unique,
            "target_token_coverage": (
                len(covered_target_tokens) / target_unique
                if target_unique
                else 0.0
            ),
            "dominant_predicted_token": dominant_predicted_token,
            "dominant_predicted_count": dominant_predicted_count,
            "dominant_predicted_rate": (
                dominant_predicted_count / profiled
                if profiled
                else 0.0
            ),
            "collapsed": profiled > 1 and target_unique > 1 and predicted_unique == 1,
            "missing_target_tokens": missing_target_tokens,
        },
        "failed_records": failed_records,
    }


def direct_answer_branch_representation_profile(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> dict[str, Any]:
    representations: list[tuple[str, list[float]]] = []
    skipped = 0
    for record in records:
        example = AnswerExample(
            prompt=record["prompt"],
            target=record["target"],
            source=f"eval:{record['id']}",
        )
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            example,
            branch_position,
            terminator,
        )
        if branch is None:
            skipped += 1
            continue
        context, target_id, _position = branch
        representations.append((tokenizer.itos[target_id], model.final_hidden(context)))

    def summarize_distances(distances: list[float]) -> dict[str, Any]:
        if not distances:
            return {"count": 0, "min": 0.0, "avg": 0.0, "max": 0.0}
        return {
            "count": len(distances),
            "min": min(distances),
            "avg": sum(distances) / len(distances),
            "max": max(distances),
        }

    all_distances: list[float] = []
    same_target_distances: list[float] = []
    different_target_distances: list[float] = []
    for left_index, (left_target, left_hidden) in enumerate(representations):
        for right_target, right_hidden in representations[left_index + 1:]:
            distance = math.sqrt(
                sum(
                    (left_value - right_value) ** 2
                    for left_value, right_value in zip(left_hidden, right_hidden)
                )
            )
            all_distances.append(distance)
            if left_target == right_target:
                same_target_distances.append(distance)
            else:
                different_target_distances.append(distance)

    target_tokens = Counter(target for target, _hidden in representations)
    return {
        "branch_position": branch_position,
        "count": len(representations),
        "skipped": skipped,
        "target_unique": len(target_tokens),
        "target_tokens": [
            {"value": value, "count": count}
            for value, count in target_tokens.most_common(12)
        ],
        "pairwise_distance": summarize_distances(all_distances),
        "same_target_pairwise_distance": summarize_distances(same_target_distances),
        "different_target_pairwise_distance": summarize_distances(
            different_target_distances
        ),
    }


def summarize_branch_diversity_target(
    branch_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    blocking_evals: list[dict[str, Any]] = []
    multi_target_profiles = 0
    passed_profiles = 0
    max_dominant_rate = 0.0
    min_target_token_coverage = 1.0

    for name, profile in sorted(branch_profiles.items()):
        diversity = profile.get("diversity", {})
        target_unique = int(diversity.get("target_unique", 0))
        predicted_unique = int(diversity.get("predicted_unique", 0))
        target_token_coverage = float(diversity.get("target_token_coverage", 0.0))
        dominant_rate = float(diversity.get("dominant_predicted_rate", 0.0))
        collapsed = bool(diversity.get("collapsed", False))
        if target_unique < 2:
            continue
        multi_target_profiles += 1
        max_dominant_rate = max(max_dominant_rate, dominant_rate)
        min_target_token_coverage = min(min_target_token_coverage, target_token_coverage)
        passed = (
            not collapsed
            and predicted_unique >= target_unique
            and target_token_coverage >= 1.0
        )
        if passed:
            passed_profiles += 1
            continue
        blocking_evals.append(
            {
                "name": name,
                "target_unique": target_unique,
                "predicted_unique": predicted_unique,
                "target_token_coverage": target_token_coverage,
                "dominant_predicted_token": diversity.get("dominant_predicted_token"),
                "dominant_predicted_rate": dominant_rate,
                "collapsed": collapsed,
                "missing_target_tokens": diversity.get("missing_target_tokens", []),
            }
        )

    return {
        "passed": multi_target_profiles > 0 and passed_profiles == multi_target_profiles,
        "multi_target_profiles": multi_target_profiles,
        "passed_profiles": passed_profiles,
        "failed_profiles": len(blocking_evals),
        "max_dominant_predicted_rate": (
            max_dominant_rate
            if multi_target_profiles
            else 0.0
        ),
        "min_target_token_coverage": (
            min_target_token_coverage
            if multi_target_profiles
            else 0.0
        ),
        "blocking_evals": blocking_evals,
    }


def branch_diversity_snapshot_score(snapshot: dict[str, Any]) -> tuple[float, ...]:
    summary = snapshot.get("branch_diversity_target", {})
    branch_profiles = snapshot.get("branch_profiles", {})
    multi_target_diversities = []
    for profile in branch_profiles.values():
        diversity = profile.get("diversity", {})
        target_rank = profile.get("target_rank", {})
        target_unique = int(diversity.get("target_unique", 0))
        if target_unique < 2:
            continue
        predicted_unique = int(diversity.get("predicted_unique", 0))
        avg_target_rank = float(target_rank.get("avg", 0.0))
        multi_target_diversities.append(
            {
                "predicted_unique_rate": predicted_unique / target_unique,
                "target_token_coverage": float(
                    diversity.get("target_token_coverage", 0.0)
                ),
                "inverse_dominant_rate": 1.0
                - float(diversity.get("dominant_predicted_rate", 0.0)),
                "target_top3_rate": float(target_rank.get("top3_rate", 0.0)),
                "target_top5_rate": float(target_rank.get("top5_rate", 0.0)),
                "inverse_target_rank": (
                    1.0 / avg_target_rank if avg_target_rank > 0.0 else 0.0
                ),
            }
        )
    profile_count = max(len(multi_target_diversities), 1)
    avg_predicted_unique_rate = (
        sum(item["predicted_unique_rate"] for item in multi_target_diversities)
        / profile_count
    )
    avg_target_token_coverage = (
        sum(item["target_token_coverage"] for item in multi_target_diversities)
        / profile_count
    )
    avg_inverse_dominant_rate = (
        sum(item["inverse_dominant_rate"] for item in multi_target_diversities)
        / profile_count
    )
    avg_target_top3_rate = (
        sum(item["target_top3_rate"] for item in multi_target_diversities)
        / profile_count
    )
    avg_target_top5_rate = (
        sum(item["target_top5_rate"] for item in multi_target_diversities)
        / profile_count
    )
    avg_inverse_target_rank = (
        sum(item["inverse_target_rank"] for item in multi_target_diversities)
        / profile_count
    )
    return (
        1.0 if summary.get("passed", False) else 0.0,
        float(summary.get("passed_profiles", 0)),
        -float(summary.get("failed_profiles", 0)),
        float(summary.get("min_target_token_coverage", 0.0)),
        avg_target_token_coverage,
        avg_target_top3_rate,
        avg_target_top5_rate,
        avg_inverse_target_rank,
        avg_predicted_unique_rate,
        avg_inverse_dominant_rate,
    )


def branch_diversity_snapshot_target_coverage_by_profile(
    snapshot: dict[str, Any],
) -> dict[str, float]:
    coverage_by_profile: dict[str, float] = {}
    branch_profiles = snapshot.get("branch_profiles", {})
    for name, profile in branch_profiles.items():
        diversity = profile.get("diversity", {})
        target_unique = int(diversity.get("target_unique", 0))
        if target_unique < 2:
            continue
        coverage_by_profile[name] = float(
            diversity.get("target_token_coverage", 0.0)
        )
    return coverage_by_profile


def branch_diversity_snapshot_target_coverage_diagnostics(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_coverage = branch_diversity_snapshot_target_coverage_by_profile(baseline)
    snapshot_coverage = branch_diversity_snapshot_target_coverage_by_profile(snapshot)
    violations: list[dict[str, Any]] = []
    for name, baseline_value in sorted(baseline_coverage.items()):
        snapshot_value = float(snapshot_coverage.get(name, -1.0))
        deficit = float(baseline_value - snapshot_value)
        if snapshot_value + 1e-12 < baseline_value:
            violations.append(
                {
                    "profile": name,
                    "baseline_coverage": float(baseline_value),
                    "snapshot_coverage": snapshot_value,
                    "deficit": deficit,
                }
            )
    violations.sort(key=lambda item: (-float(item["deficit"]), item["profile"]))
    worst_violation = violations[0] if violations else None
    return {
        "preserved": not violations,
        "baseline_profile_count": len(baseline_coverage),
        "snapshot_profile_count": len(snapshot_coverage),
        "violating_profile_count": len(violations),
        "worst_deficit": float(worst_violation["deficit"]) if worst_violation else 0.0,
        "worst_violation": worst_violation,
        "violations": violations,
    }


def branch_diversity_snapshot_target_coverage_delta(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_coverage = branch_diversity_snapshot_target_coverage_by_profile(baseline)
    snapshot_coverage = branch_diversity_snapshot_target_coverage_by_profile(snapshot)
    profile_names = sorted(set(baseline_coverage) | set(snapshot_coverage))
    improved_profiles: list[dict[str, Any]] = []
    regressed_profiles: list[dict[str, Any]] = []
    tied_profiles: list[str] = []
    deltas: dict[str, float] = {}
    for name in profile_names:
        baseline_value = float(baseline_coverage.get(name, 0.0))
        snapshot_value = float(snapshot_coverage.get(name, 0.0))
        delta = snapshot_value - baseline_value
        deltas[name] = delta
        profile_delta = {
            "profile": name,
            "baseline_coverage": baseline_value,
            "snapshot_coverage": snapshot_value,
            "delta": delta,
        }
        if delta > 1e-12:
            improved_profiles.append(profile_delta)
        elif delta < -1e-12:
            regressed_profiles.append(profile_delta)
        else:
            tied_profiles.append(name)
    profile_count = max(len(profile_names), 1)
    baseline_average = sum(float(value) for value in baseline_coverage.values()) / max(
        len(baseline_coverage),
        1,
    )
    snapshot_average = sum(float(value) for value in snapshot_coverage.values()) / max(
        len(snapshot_coverage),
        1,
    )
    baseline_min = (
        min(float(value) for value in baseline_coverage.values())
        if baseline_coverage
        else 0.0
    )
    snapshot_min = (
        min(float(value) for value in snapshot_coverage.values())
        if snapshot_coverage
        else 0.0
    )
    return {
        "baseline_profile_count": len(baseline_coverage),
        "snapshot_profile_count": len(snapshot_coverage),
        "profile_count": len(profile_names),
        "baseline_min_coverage": baseline_min,
        "snapshot_min_coverage": snapshot_min,
        "min_delta": snapshot_min - baseline_min,
        "baseline_average_coverage": baseline_average,
        "snapshot_average_coverage": snapshot_average,
        "average_delta": snapshot_average - baseline_average,
        "total_delta": sum(deltas.values()),
        "mean_delta": sum(deltas.values()) / profile_count,
        "improved_profile_count": len(improved_profiles),
        "regressed_profile_count": len(regressed_profiles),
        "tied_profile_count": len(tied_profiles),
        "improved_profiles": improved_profiles,
        "regressed_profiles": regressed_profiles,
        "tied_profiles": tied_profiles,
        "deltas": deltas,
    }


def branch_diversity_snapshot_collapsed_profile_names(
    snapshot: dict[str, Any],
) -> list[str]:
    names: set[str] = set()
    for blocking_eval in snapshot.get("branch_diversity_target", {}).get(
        "blocking_evals",
        [],
    ):
        name = str(blocking_eval.get("name", ""))
        if not name:
            continue
        if bool(blocking_eval.get("collapsed", False)):
            names.add(name)
    return sorted(names)


def branch_diversity_snapshot_profile_diversity_delta(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
    profile_names: list[str] | set[str] | tuple[str, ...],
) -> dict[str, Any]:
    baseline_profiles = baseline.get("branch_profiles", {})
    snapshot_profiles = snapshot.get("branch_profiles", {})
    improved_profiles: list[dict[str, Any]] = []
    regressed_profiles: list[dict[str, Any]] = []
    tied_profiles: list[str] = []
    profiles: list[dict[str, Any]] = []

    for name in sorted(set(profile_names)):
        baseline_diversity = baseline_profiles.get(name, {}).get("diversity", {})
        snapshot_diversity = snapshot_profiles.get(name, {}).get("diversity", {})
        baseline_predicted_unique = int(
            baseline_diversity.get("predicted_unique", 0)
        )
        snapshot_predicted_unique = int(
            snapshot_diversity.get("predicted_unique", 0)
        )
        baseline_coverage = float(
            baseline_diversity.get("target_token_coverage", 0.0)
        )
        snapshot_coverage = float(
            snapshot_diversity.get("target_token_coverage", 0.0)
        )
        baseline_dominant_rate = float(
            baseline_diversity.get("dominant_predicted_rate", 0.0)
        )
        snapshot_dominant_rate = float(
            snapshot_diversity.get("dominant_predicted_rate", 0.0)
        )
        predicted_unique_delta = (
            snapshot_predicted_unique - baseline_predicted_unique
        )
        coverage_delta = snapshot_coverage - baseline_coverage
        dominant_rate_delta = snapshot_dominant_rate - baseline_dominant_rate
        profile_delta = {
            "profile": name,
            "baseline_predicted_unique": baseline_predicted_unique,
            "snapshot_predicted_unique": snapshot_predicted_unique,
            "predicted_unique_delta": predicted_unique_delta,
            "baseline_coverage": baseline_coverage,
            "snapshot_coverage": snapshot_coverage,
            "coverage_delta": coverage_delta,
            "baseline_dominant_rate": baseline_dominant_rate,
            "snapshot_dominant_rate": snapshot_dominant_rate,
            "dominant_rate_delta": dominant_rate_delta,
        }
        profiles.append(profile_delta)
        improved = (
            predicted_unique_delta > 0
            or coverage_delta > 1e-12
            or dominant_rate_delta < -1e-12
        )
        regressed = predicted_unique_delta < 0 or coverage_delta < -1e-12
        if improved and not regressed:
            improved_profiles.append(profile_delta)
        elif regressed:
            regressed_profiles.append(profile_delta)
        else:
            tied_profiles.append(name)

    return {
        "profile_count": len(profiles),
        "improved_profile_count": len(improved_profiles),
        "regressed_profile_count": len(regressed_profiles),
        "tied_profile_count": len(tied_profiles),
        "improved_profiles": improved_profiles,
        "regressed_profiles": regressed_profiles,
        "tied_profiles": tied_profiles,
        "profiles": profiles,
    }


def source_profile_label(profile: str) -> str:
    if ":" not in profile:
        return profile
    return profile.split(":", 1)[1]


def remaining_profile_binding_source_labels(
    target_profiles: list[str] | set[str] | tuple[str, ...],
) -> list[str]:
    labels = set(target_profiles)
    if "paraphrases" in labels:
        labels.update(BASELINE_FLOOR_REMAINING_PROFILE_BINDING_PARAPHRASE_SOURCE_LABELS)
        labels.discard("paraphrases")
    return sorted(labels)


def remaining_profile_binding_profile_order(
    profile_groups: dict[str, list[BranchReplayRecord]],
    target_profiles: list[str] | set[str] | tuple[str, ...],
) -> list[tuple[str, list[BranchReplayRecord]]]:
    source_labels = set(remaining_profile_binding_source_labels(target_profiles))

    def priority(item: tuple[str, list[BranchReplayRecord]]) -> tuple[int, str, str]:
        profile, anchors = item
        label = source_profile_label(profile)
        target_count = len(
            {
                target
                for _context, target, _predicted, _profile in (
                    branch_replay_parts(anchor) for anchor in anchors
                )
            }
        )
        priority_rank = 0 if label in source_labels and target_count > 1 else 1
        return priority_rank, label, profile

    return sorted(profile_groups.items(), key=priority)


def branch_diversity_snapshot_preserves_target_coverage(
    snapshot: dict[str, Any],
    baseline: dict[str, Any],
) -> bool:
    diagnostics = branch_diversity_snapshot_target_coverage_diagnostics(
        snapshot,
        baseline,
    )
    return bool(diagnostics["preserved"])


def train_direct_answer_lesson(
    model: TinyTransformerLM,
    lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    params: list[Scalar] | None = None,
) -> float:
    context, target_id = lesson[rng.randrange(len(lesson))]
    return model.train_step(context, target_id, learning_rate, params=params)


def direct_answer_first_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id != target_id:
            return context, target_id, predicted_id, position
        ids.append(target_id)
    return None


def train_direct_answer_first_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_first_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    context, target_id, _predicted_id, _position = repair
    return model.train_step(context, target_id, learning_rate, params=params)


def train_direct_answer_first_error_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_first_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood(
        context,
        target_id,
        predicted_id,
        learning_rate,
        negative_weight,
        params=params,
    )


def direct_answer_rollout_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    last_repair: tuple[list[int], int, int, int] | None = None
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id != target_id:
            last_repair = (context, target_id, predicted_id, position)
        ids.append(predicted_id)
        if terminator and tokenizer.itos[predicted_id] == terminator:
            break
    return last_repair


def direct_answer_early_stop_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    if not terminator:
        return None
    terminator_id = tokenizer.stoi.get(terminator)
    if terminator_id is None:
        return None
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id == terminator_id and target_id != terminator_id:
            return context, target_id, predicted_id, position
        ids.append(predicted_id)
        if predicted_id == terminator_id:
            break
    return None


def has_repeated_suffix(
    ids: list[int],
    max_ngram_size: int = 3,
    repeat_count: int = 2,
) -> bool:
    if repeat_count < 2:
        return False
    max_size = min(max_ngram_size, len(ids) // repeat_count)
    for ngram_size in range(1, max_size + 1):
        suffix = ids[-ngram_size:]
        repeated = True
        for repeat_index in range(2, repeat_count + 1):
            start = -ngram_size * repeat_index
            end = -ngram_size * (repeat_index - 1)
            if ids[start:end] != suffix:
                repeated = False
                break
        if repeated:
            return True
    return False


def direct_answer_repeat_loop_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    ids = tokenizer.encode(example.prompt)
    generated: list[int] = []
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        next_generated = generated + [predicted_id]
        if predicted_id != target_id and has_repeated_suffix(next_generated):
            return context, target_id, predicted_id, position
        ids.append(predicted_id)
        generated = next_generated
        if terminator and tokenizer.itos[predicted_id] == terminator:
            break
    return None


def direct_answer_generated_prefix_recovery(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    recovery_steps: int,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int, DirectAnswerLesson] | None:
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id != target_id:
            recovery: DirectAnswerLesson = []
            recovery_ids = ids + [predicted_id]
            for offset in range(max(1, recovery_steps)):
                target_position = position + offset
                if target_position >= len(target_ids):
                    break
                recovery.append(
                    (
                        make_context(
                            recovery_ids,
                            model.config.context_size,
                            tokenizer.pad_id,
                        ),
                        target_ids[target_position],
                    )
                )
                recovery_ids.append(target_ids[target_position])
            if recovery:
                return context, target_id, predicted_id, position, recovery
            return None
        ids.append(predicted_id)
        if terminator and tokenizer.itos[predicted_id] == terminator:
            break
    return None


def direct_answer_sequence_repair_errors(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> list[DirectAnswerRepair]:
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    repairs: list[DirectAnswerRepair] = []
    for position, target_id in enumerate(target_ids):
        context = make_context(ids, model.config.context_size, tokenizer.pad_id)
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        if predicted_id != target_id:
            repairs.append((context, target_id, predicted_id, position))
        ids.append(target_id)
    return repairs


def direct_answer_branch_repair_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> DirectAnswerRepair | None:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return None
    context, target_id, position = branch
    probs = model.predict(context)
    predicted_id = max(range(len(probs)), key=lambda index: probs[index])
    return context, target_id, predicted_id, position


def direct_answer_branch_context(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int] | None:
    if branch_position < 0:
        return None
    ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    if branch_position >= len(target_ids):
        return None
    ids.extend(target_ids[:branch_position])
    context = make_context(ids, model.config.context_size, tokenizer.pad_id)
    return context, target_ids[branch_position], branch_position


def direct_answer_branch_target_ids(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[int]:
    target_ids: set[int] = set()
    for candidate in branch_examples:
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            candidate,
            branch_position,
            terminator,
        )
        if branch is None:
            continue
        _context, target_id, _position = branch
        target_ids.add(target_id)
    return sorted(target_ids)


def direct_answer_branch_span_position(
    tokenizer: CharTokenizer,
    example: AnswerExample,
    rng: random.Random,
    branch_position: int,
    branch_span: int,
    terminator: str = ANSWER_TERMINATOR,
) -> int | None:
    if branch_position < 0:
        return None
    target_ids = tokenizer.encode(answer_completion_text(example.target, terminator))
    if branch_position >= len(target_ids):
        return None
    end_position = min(len(target_ids), branch_position + max(1, branch_span))
    return rng.randrange(branch_position, end_position)


def direct_answer_branch_span_repair_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    rng: random.Random,
    branch_position: int,
    branch_span: int,
    terminator: str = ANSWER_TERMINATOR,
) -> DirectAnswerRepair | None:
    sampled_position = direct_answer_branch_span_position(
        tokenizer,
        example,
        rng,
        branch_position,
        branch_span,
        terminator,
    )
    if sampled_position is None:
        return None
    return direct_answer_branch_repair_error(
        model,
        tokenizer,
        example,
        sampled_position,
        terminator,
    )


def direct_answer_dominant_branch_prediction(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    sample_count: int,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[int, int, int] | None:
    if not branch_examples:
        return None
    if sample_count <= 0 or sample_count >= len(branch_examples):
        candidates = branch_examples[:]
        rng.shuffle(candidates)
    else:
        candidates = rng.sample(branch_examples, sample_count)
    predicted_counts: Counter[int] = Counter()
    scored = 0
    for candidate in candidates:
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            candidate,
            branch_position,
            terminator,
        )
        if branch is None:
            continue
        context, _target_id, _position = branch
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        predicted_counts[predicted_id] += 1
        scored += 1
    if not predicted_counts:
        return None
    predicted_id, count = predicted_counts.most_common(1)[0]
    return predicted_id, count, scored


def direct_answer_branch_batch(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[tuple[list[int], int]]:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return []
    context, target_id, _position = branch
    branches = [(context, target_id)]
    seen_targets = {target_id}
    candidates = branch_examples[:]
    rng.shuffle(candidates)
    for candidate in candidates:
        if len(branches) >= max(1, batch_size):
            break
        candidate_branch = direct_answer_branch_context(
            model,
            tokenizer,
            candidate,
            branch_position,
            terminator,
        )
        if candidate_branch is None:
            continue
        candidate_context, candidate_target, _candidate_position = candidate_branch
        if candidate_target in seen_targets:
            continue
        branches.append((candidate_context, candidate_target))
        seen_targets.add(candidate_target)
    return branches


def direct_answer_target_balanced_branch_batch(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[tuple[list[int], int]]:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return []
    context, target_id, _position = branch
    branches = [(context, target_id)]
    by_target: dict[int, list[tuple[list[int], int]]] = {}
    candidates = branch_examples[:]
    rng.shuffle(candidates)
    for candidate in candidates:
        candidate_branch = direct_answer_branch_context(
            model,
            tokenizer,
            candidate,
            branch_position,
            terminator,
        )
        if candidate_branch is None:
            continue
        candidate_context, candidate_target, _candidate_position = candidate_branch
        if candidate_target == target_id:
            continue
        by_target.setdefault(candidate_target, []).append(
            (candidate_context, candidate_target)
        )
    target_ids = list(by_target)
    rng.shuffle(target_ids)
    for candidate_target in target_ids:
        if len(branches) >= max(1, batch_size):
            break
        branches.append(rng.choice(by_target[candidate_target]))
    return branches


def direct_answer_branch_diversity_batch(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[tuple[list[int], int, int]]:
    branches = direct_answer_branch_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    diversity_branches: list[tuple[list[int], int, int]] = []
    for context, target_id in branches:
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        diversity_branches.append((context, target_id, predicted_id))
    return diversity_branches


def direct_answer_target_balanced_branch_diversity_batch(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[tuple[list[int], int, int]]:
    branches = direct_answer_target_balanced_branch_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    diversity_branches: list[tuple[list[int], int, int]] = []
    for context, target_id in branches:
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        diversity_branches.append((context, target_id, predicted_id))
    return diversity_branches


def direct_answer_profiled_branch_batch(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    balance_targets: bool = False,
    prediction_overrides: ReplayPredictionOverrides | None = None,
) -> list[BranchReplayRecord]:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return []
    context, target_id, _position = branch
    seeds: list[ProfiledBranchSeed] = [
        (context, target_id, direct_answer_profile_key(example))
    ]
    candidates = branch_examples[:]
    rng.shuffle(candidates)
    if balance_targets:
        by_target: dict[int, list[ProfiledBranchSeed]] = {}
        for candidate in candidates:
            candidate_branch = direct_answer_branch_context(
                model,
                tokenizer,
                candidate,
                branch_position,
                terminator,
            )
            if candidate_branch is None:
                continue
            candidate_context, candidate_target, _candidate_position = candidate_branch
            if candidate_target == target_id:
                continue
            by_target.setdefault(candidate_target, []).append(
                (
                    candidate_context,
                    candidate_target,
                    direct_answer_profile_key(candidate),
                )
            )
        target_ids = list(by_target)
        rng.shuffle(target_ids)
        for candidate_target in target_ids:
            if len(seeds) >= max(1, batch_size):
                break
            seeds.append(rng.choice(by_target[candidate_target]))
    else:
        seen_targets = {target_id}
        for candidate in candidates:
            if len(seeds) >= max(1, batch_size):
                break
            candidate_branch = direct_answer_branch_context(
                model,
                tokenizer,
                candidate,
                branch_position,
                terminator,
            )
            if candidate_branch is None:
                continue
            candidate_context, candidate_target, _candidate_position = candidate_branch
            if candidate_target in seen_targets:
                continue
            seeds.append(
                (
                    candidate_context,
                    candidate_target,
                    direct_answer_profile_key(candidate),
                )
            )
            seen_targets.add(candidate_target)
    profiled: list[BranchReplayRecord] = []
    for context, target_id, profile in seeds:
        override_key = (tuple(context), target_id, profile)
        predicted_id = (
            prediction_overrides[override_key]
            if prediction_overrides is not None
            and override_key in prediction_overrides
            else None
        )
        if predicted_id is None:
            probs = model.predict(context)
            predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        profiled.append((context, target_id, predicted_id, profile))
    return profiled


def direct_answer_profiled_replay_records(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    branch_examples: list[AnswerExample],
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
) -> list[BranchReplayRecord]:
    records: list[BranchReplayRecord] = []
    for example in branch_examples:
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            example,
            branch_position,
            terminator,
        )
        if branch is None:
            continue
        context, target_id, _position = branch
        probs = model.predict(context)
        predicted_id = max(range(len(probs)), key=lambda index: probs[index])
        records.append(
            (
                context,
                target_id,
                predicted_id,
                direct_answer_profile_key(example),
            )
        )
    return records


def baseline_floor_repair_anchor_records(
    replay_records: list[BranchReplayRecord],
) -> list[BranchReplayRecord]:
    targets_by_profile: dict[str, set[int]] = {}
    for branch in replay_records:
        _context, target, _predicted, profile = branch_replay_parts(branch)
        targets_by_profile.setdefault(profile, set()).add(target)
    anchors: list[BranchReplayRecord] = []
    seen: set[tuple[tuple[int, ...], int, str]] = set()
    for branch in replay_records:
        context, _target, predicted, profile = branch_replay_parts(branch)
        if predicted not in targets_by_profile.get(profile, set()):
            continue
        key = (tuple(context), predicted, profile)
        if key in seen:
            continue
        seen.add(key)
        anchors.append((context, predicted, predicted, profile))
    return anchors


def baseline_floor_frontier_anchor_records(
    floor_anchors: list[BranchReplayRecord],
    replay_records: list[BranchReplayRecord],
) -> list[BranchReplayRecord]:
    represented_targets_by_profile: dict[str, set[int]] = {}
    for branch in floor_anchors:
        _context, target, _predicted, profile = branch_replay_parts(branch)
        represented_targets_by_profile.setdefault(profile, set()).add(target)
    frontier: list[BranchReplayRecord] = []
    seen_profile_targets: set[tuple[str, int]] = set()
    for branch in replay_records:
        context, target, _predicted, profile = branch_replay_parts(branch)
        represented_targets = represented_targets_by_profile.get(profile)
        if not represented_targets:
            continue
        if target in represented_targets:
            continue
        profile_target = (profile, target)
        if profile_target in seen_profile_targets:
            continue
        seen_profile_targets.add(profile_target)
        frontier.append((context, target, target, profile))
    return frontier


def baseline_floor_objective_anchor_batch(
    anchors: list[BranchReplayRecord],
    rng: random.Random,
    batch_size: int,
) -> list[BranchReplayRecord]:
    if not anchors:
        return []
    anchors_by_profile_target: dict[tuple[str, int], list[BranchReplayRecord]] = {}
    for branch in anchors:
        _context, target, _predicted, profile = branch_replay_parts(branch)
        anchors_by_profile_target.setdefault((profile, target), []).append(branch)
    profile_targets = list(anchors_by_profile_target)
    rng.shuffle(profile_targets)
    selected: list[BranchReplayRecord] = []
    for profile_target in profile_targets[: max(1, batch_size)]:
        selected.append(rng.choice(anchors_by_profile_target[profile_target]))
    return selected


def baseline_floor_anchor_profile_counts(
    anchors: list[BranchReplayRecord],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for branch in anchors:
        _context, _target, _predicted, profile = branch_replay_parts(branch)
        counts[profile] += 1
    return dict(sorted(counts.items()))


def baseline_floor_anchor_profile_groups(
    anchors: list[BranchReplayRecord],
) -> dict[str, list[BranchReplayRecord]]:
    groups: dict[str, list[BranchReplayRecord]] = {}
    for branch in anchors:
        _context, _target, _predicted, profile = branch_replay_parts(branch)
        groups.setdefault(profile, []).append(branch)
    return dict(sorted(groups.items()))


def baseline_floor_anchor_profile_target_count(
    anchors: list[BranchReplayRecord],
) -> int:
    profile_targets: set[tuple[str, int]] = set()
    for branch in anchors:
        _context, target, _predicted, profile = branch_replay_parts(branch)
        profile_targets.add((profile, target))
    return len(profile_targets)


def train_direct_answer_baseline_floor_anchor_batch(
    model: TinyTransformerLM,
    anchors: list[BranchReplayRecord],
    learning_rate: float,
    params: list[Scalar] | None = None,
) -> float:
    if not anchors:
        return 0.0
    params = model.parameters() if params is None else params
    zero_grad(params)
    loss = Scalar(0.0)
    for branch in anchors:
        context, target, _predicted, _profile = branch_replay_parts(branch)
        loss = loss + cross_entropy_scalars(model._forward_scalars(context), target)
    loss = loss / len(anchors)
    loss.backward()
    model.apply_gradients(params, learning_rate)
    return loss.data


def train_direct_answer_baseline_floor_anchor_branch_diversity(
    model: TinyTransformerLM,
    anchors: list[BranchReplayRecord],
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    contrast_weight: float,
    params: list[Scalar] | None = None,
) -> float:
    branches: list[tuple[list[int], int, int]] = []
    for branch in anchors:
        context, target, _predicted, _profile = branch_replay_parts(branch)
        probs = model.predict(context)
        predicted = max(range(len(probs)), key=lambda index: probs[index])
        branches.append((context, target, predicted))
    if not branches:
        return 0.0
    return model.train_step_with_branch_diversity(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        contrast_weight,
        params=params,
    )


def train_direct_answer_baseline_floor_anchor_repair(
    model: TinyTransformerLM,
    anchors: list[BranchReplayRecord],
    rng: random.Random,
    learning_rate: float,
    batch_size: int,
    params: list[Scalar] | None = None,
) -> float:
    if not anchors:
        return 0.0
    params = model.parameters() if params is None else params
    anchors_by_target: dict[int, list[tuple[list[int], int]]] = {}
    for branch in anchors:
        context, target, _predicted, _profile = branch_replay_parts(branch)
        anchors_by_target.setdefault(target, []).append((context, target))
    target_ids = list(anchors_by_target)
    rng.shuffle(target_ids)
    selected: list[tuple[list[int], int]] = []
    for target_id in target_ids:
        if len(selected) >= max(1, batch_size):
            break
        selected.append(rng.choice(anchors_by_target[target_id]))
    if not selected:
        return 0.0
    zero_grad(params)
    loss = Scalar(0.0)
    for context, target in selected:
        loss = loss + cross_entropy_scalars(model._forward_scalars(context), target)
    loss = loss / len(selected)
    loss.backward()
    model.apply_gradients(params, learning_rate)
    return loss.data


def direct_answer_hard_branch_contrast(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    rng: random.Random,
    branch_position: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
) -> DirectAnswerBranchContrast | None:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return None
    context, target_id, _position = branch
    if not branch_examples:
        return None
    if hard_negative_count <= 0 or hard_negative_count >= len(branch_examples):
        candidates = branch_examples[:]
        rng.shuffle(candidates)
    else:
        candidates = rng.sample(branch_examples, hard_negative_count)

    probs = model.predict(context)
    best_score: float | None = None
    best_contrast: tuple[list[int], int] | None = None
    for contrast_example in candidates:
        if contrast_example == example:
            continue
        contrast = direct_answer_branch_context(
            model,
            tokenizer,
            contrast_example,
            branch_position,
            terminator,
        )
        if contrast is None:
            continue
        contrast_context, contrast_target, _contrast_position = contrast
        if contrast_target == target_id:
            continue
        contrast_probs = model.predict(contrast_context)
        score = probs[contrast_target] + contrast_probs[target_id]
        if best_score is None or score > best_score:
            best_score = score
            best_contrast = (contrast_context, contrast_target)
    if best_contrast is None:
        return None
    contrast_context, contrast_target = best_contrast
    return context, target_id, contrast_context, contrast_target


def direct_answer_balanced_repair_error(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    terminator: str = ANSWER_TERMINATOR,
) -> tuple[list[int], int, int, int] | None:
    for repair_fn in (
        direct_answer_early_stop_error,
        direct_answer_repeat_loop_error,
        direct_answer_rollout_error,
        direct_answer_first_error,
    ):
        repair = repair_fn(model, tokenizer, example, terminator)
        if repair is not None:
            return repair
    return None


def train_direct_answer_rollout_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_rollout_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood(
        context,
        target_id,
        predicted_id,
        learning_rate,
        negative_weight,
        params=params,
    )


def train_direct_answer_balanced_repair_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_balanced_repair_error(model, tokenizer, example, terminator)
    positive_context, positive_target = fallback_lesson[rng.randrange(len(fallback_lesson))]
    if repair is None:
        return model.train_step(
            positive_context,
            positive_target,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        predicted_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_generated_prefix_recovery_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    recovery_steps: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_generated_prefix_recovery(
        model,
        tokenizer,
        example,
        recovery_steps,
        terminator,
    )
    if repair is None:
        return train_direct_answer_balanced_repair_unlikelihood(
            model,
            tokenizer,
            example,
            fallback_lesson,
            rng,
            learning_rate,
            negative_weight,
            positive_weight,
            terminator,
            params=params,
        )
    context, target_id, predicted_id, _position, recovery_lesson = repair
    positive_context, positive_target = recovery_lesson[rng.randrange(len(recovery_lesson))]
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        predicted_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_sequence_repair_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repairs = direct_answer_sequence_repair_errors(model, tokenizer, example, terminator)
    positive_context, positive_target = fallback_lesson[rng.randrange(len(fallback_lesson))]
    if not repairs:
        return model.train_step(
            positive_context,
            positive_target,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repairs[rng.randrange(len(repairs))]
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        predicted_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_loop_escape_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_repeat_loop_error(model, tokenizer, example, terminator)
    positive_context, positive_target = fallback_lesson[rng.randrange(len(fallback_lesson))]
    if repair is None:
        return model.train_step(
            positive_context,
            positive_target,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        predicted_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_branch_repair_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_branch_repair_error(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    positive_context, positive_target = fallback_lesson[rng.randrange(len(fallback_lesson))]
    if repair is None:
        return model.train_step(
            positive_context,
            positive_target,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        predicted_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_branch_span_repair_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    branch_position: int,
    branch_span: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_branch_span_repair_error(
        model,
        tokenizer,
        example,
        rng,
        branch_position,
        branch_span,
        terminator,
    )
    positive_context, positive_target = fallback_lesson[rng.randrange(len(fallback_lesson))]
    if repair is None:
        return model.train_step(
            positive_context,
            positive_target,
            learning_rate,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        predicted_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_branch_collapse_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    branch_position: int,
    sample_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    context, target_id, _position = branch
    local_probs = model.predict(context)
    local_predicted_id = max(range(len(local_probs)), key=lambda index: local_probs[index])
    dominant = direct_answer_dominant_branch_prediction(
        model,
        tokenizer,
        branch_examples,
        rng,
        branch_position,
        sample_count,
        terminator,
    )
    negative_id = local_predicted_id
    if dominant is not None:
        dominant_id, _count, _scored = dominant
        if dominant_id != target_id:
            negative_id = dominant_id
    positive_context, positive_target = fallback_lesson[rng.randrange(len(fallback_lesson))]
    return model.train_step_with_unlikelihood_and_positive(
        context,
        target_id,
        negative_id,
        positive_context,
        positive_target,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_branch_batch_contrast_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    branches = direct_answer_branch_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_batch_contrast(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        params=params,
    )


def train_direct_answer_branch_diversity_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    contrast_weight: float,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    branches = direct_answer_branch_diversity_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_diversity(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        contrast_weight,
        params=params,
    )


def train_direct_answer_branch_target_softmax_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    target_softmax_weight: float,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    branches = direct_answer_branch_diversity_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_target_softmax(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        target_softmax_weight,
        params=params,
    )


def train_direct_answer_branch_target_margin_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    margin_weight: float,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    branches = direct_answer_branch_diversity_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_target_margin(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        margin_weight,
        params=params,
    )


def train_direct_answer_branch_representation_contrast_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    representation_weight: float,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
) -> float:
    batch_builder = (
        direct_answer_target_balanced_branch_diversity_batch
        if balance_targets
        else direct_answer_branch_diversity_batch
    )
    branches = batch_builder(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_representation_contrast(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        representation_weight,
        params=params,
    )


def train_direct_answer_branch_output_binding_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    binding_weight: float,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    branches = direct_answer_branch_diversity_batch(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_output_binding(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        binding_weight,
        params=params,
    )


def train_direct_answer_branch_bidirectional_binding_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    binding_weight: float,
    branch_position: int,
    batch_size: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
) -> float:
    batch_builder = (
        direct_answer_target_balanced_branch_diversity_batch
        if balance_targets
        else direct_answer_branch_diversity_batch
    )
    branches = batch_builder(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_bidirectional_binding(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        binding_weight,
        params=params,
    )


def train_direct_answer_branch_coverage_binding_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    binding_weight: float,
    branch_position: int,
    batch_size: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
) -> float:
    batch_builder = (
        direct_answer_target_balanced_branch_diversity_batch
        if balance_targets
        else direct_answer_branch_diversity_batch
    )
    branches = batch_builder(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_coverage_binding(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        binding_weight,
        hard_negative_count,
        params=params,
    )


def train_direct_answer_branch_target_set_coverage_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    coverage_weight: float,
    branch_position: int,
    batch_size: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
) -> float:
    batch_builder = (
        direct_answer_target_balanced_branch_diversity_batch
        if balance_targets
        else direct_answer_branch_diversity_batch
    )
    branches = batch_builder(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_target_set_coverage(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        coverage_weight,
        hard_negative_count,
        params=params,
    )


def train_direct_answer_branch_target_diversity_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    diversity_weight: float,
    branch_position: int,
    batch_size: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
) -> float:
    batch_builder = (
        direct_answer_target_balanced_branch_diversity_batch
        if balance_targets
        else direct_answer_branch_diversity_batch
    )
    branches = batch_builder(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_target_diversity(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        diversity_weight,
        hard_negative_count,
        params=params,
    )


def train_direct_answer_branch_target_replay_coverage_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    replay_weight: float,
    branch_position: int,
    batch_size: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
) -> float:
    batch_builder = (
        direct_answer_target_balanced_branch_diversity_batch
        if balance_targets
        else direct_answer_branch_diversity_batch
    )
    branches = batch_builder(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    replay_targets = direct_answer_branch_target_ids(
        model,
        tokenizer,
        branch_examples,
        branch_position,
        terminator,
    )
    return model.train_step_with_branch_target_replay_coverage(
        branches,
        replay_targets,
        learning_rate,
        negative_weight,
        positive_weight,
        replay_weight,
        hard_negative_count,
        params=params,
    )


def train_direct_answer_branch_context_replay_coverage_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    replay_weight: float,
    branch_position: int,
    batch_size: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
    preserve_covered_targets: bool = False,
    balance_covered_target_anchors: bool = False,
    focus_uncovered_targets: bool = False,
    preserve_predicted_target_coverage: bool = False,
    balance_deficit_targets: bool = False,
    profile_aware_targets: bool = False,
    balance_profile_target_shares: bool = False,
    enforce_prompt_target_margins: bool = False,
    replay_prediction_overrides: ReplayPredictionOverrides | None = None,
    floor_preservation_branches: list[BranchReplayRecord] | None = None,
    floor_preservation_weight: float = 0.0,
    balance_floor_preservation_targets: bool = False,
) -> float:
    if profile_aware_targets:
        branches = direct_answer_profiled_branch_batch(
            model,
            tokenizer,
            example,
            branch_examples,
            rng,
            branch_position,
            batch_size,
            terminator,
            balance_targets=balance_targets,
        )
    else:
        batch_builder = (
            direct_answer_target_balanced_branch_diversity_batch
            if balance_targets
            else direct_answer_branch_diversity_batch
        )
        branches = batch_builder(
            model,
            tokenizer,
            example,
            branch_examples,
            rng,
            branch_position,
            batch_size,
            terminator,
        )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    replay_batch_size = max(batch_size, batch_size + max(0, hard_negative_count))
    if profile_aware_targets:
        replay_branches = direct_answer_profiled_branch_batch(
            model,
            tokenizer,
            example,
            branch_examples,
            rng,
            branch_position,
            replay_batch_size,
            terminator,
            balance_targets=True,
            prediction_overrides=replay_prediction_overrides,
        )
    else:
        replay_branches = direct_answer_target_balanced_branch_diversity_batch(
            model,
            tokenizer,
            example,
            branch_examples,
            rng,
            branch_position,
            replay_batch_size,
            terminator,
        )
    return model.train_step_with_branch_context_replay_coverage(
        branches,
        replay_branches,
        learning_rate,
        negative_weight,
        positive_weight,
        replay_weight,
        hard_negative_count,
        params=params,
        preserve_covered_targets=preserve_covered_targets,
        balance_covered_target_anchors=balance_covered_target_anchors,
        focus_uncovered_targets=focus_uncovered_targets,
        preserve_predicted_target_coverage=preserve_predicted_target_coverage,
        balance_deficit_targets=balance_deficit_targets,
        profile_aware_targets=profile_aware_targets,
        balance_profile_target_shares=balance_profile_target_shares,
        enforce_prompt_target_margins=enforce_prompt_target_margins,
        floor_preservation_branches=floor_preservation_branches,
        floor_preservation_weight=floor_preservation_weight,
        balance_floor_preservation_targets=balance_floor_preservation_targets,
    )


def train_direct_answer_branch_rank_margin_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    margin_weight: float,
    branch_position: int,
    batch_size: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
) -> float:
    batch_builder = (
        direct_answer_target_balanced_branch_diversity_batch
        if balance_targets
        else direct_answer_branch_diversity_batch
    )
    branches = batch_builder(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_rank_margin(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        margin_weight,
        hard_negative_count,
        params=params,
    )


def train_direct_answer_branch_topk_softmax_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    candidate_weight: float,
    branch_position: int,
    batch_size: int,
    candidate_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
    balance_targets: bool = False,
) -> float:
    batch_builder = (
        direct_answer_target_balanced_branch_diversity_batch
        if balance_targets
        else direct_answer_branch_diversity_batch
    )
    branches = batch_builder(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        batch_size,
        terminator,
    )
    if not branches:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    return model.train_step_with_branch_topk_softmax(
        branches,
        learning_rate,
        negative_weight,
        positive_weight,
        candidate_weight,
        candidate_count,
        params=params,
    )


def train_direct_answer_branch_contrast_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    contrast_weight: float,
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        branch_position,
        terminator,
    )
    if branch is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    context, target_id, _position = branch
    for _ in range(max(len(branch_examples), 1)):
        contrast_example = branch_examples[rng.randrange(len(branch_examples))]
        contrast = direct_answer_branch_context(
            model,
            tokenizer,
            contrast_example,
            branch_position,
            terminator,
        )
        if contrast is None:
            continue
        contrast_context, contrast_target, _contrast_position = contrast
        if contrast_target == target_id:
            continue
        return model.train_step_with_branch_contrast(
            context,
            target_id,
            contrast_context,
            contrast_target,
            learning_rate,
            negative_weight,
            contrast_weight,
            params=params,
        )
    return train_direct_answer_branch_repair_unlikelihood(
        model,
        tokenizer,
        example,
        fallback_lesson,
        rng,
        learning_rate,
        negative_weight,
        contrast_weight,
        branch_position,
        terminator,
        params=params,
    )


def train_direct_answer_branch_span_contrast_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    contrast_weight: float,
    branch_position: int,
    branch_span: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    sampled_position = direct_answer_branch_span_position(
        tokenizer,
        example,
        rng,
        branch_position,
        branch_span,
        terminator,
    )
    if sampled_position is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    branch = direct_answer_branch_context(
        model,
        tokenizer,
        example,
        sampled_position,
        terminator,
    )
    if branch is None:
        return train_direct_answer_lesson(
            model,
            fallback_lesson,
            rng,
            learning_rate,
            params=params,
        )
    context, target_id, _position = branch
    for _ in range(max(len(branch_examples), 1)):
        contrast_example = branch_examples[rng.randrange(len(branch_examples))]
        contrast = direct_answer_branch_context(
            model,
            tokenizer,
            contrast_example,
            sampled_position,
            terminator,
        )
        if contrast is None:
            continue
        contrast_context, contrast_target, _contrast_position = contrast
        if contrast_target == target_id:
            continue
        return model.train_step_with_branch_contrast(
            context,
            target_id,
            contrast_context,
            contrast_target,
            learning_rate,
            negative_weight,
            contrast_weight,
            params=params,
        )
    return train_direct_answer_branch_span_repair_unlikelihood(
        model,
        tokenizer,
        example,
        fallback_lesson,
        rng,
        learning_rate,
        negative_weight,
        positive_weight,
        branch_position,
        branch_span,
        terminator,
        params=params,
    )


def train_direct_answer_hard_branch_contrast_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    branch_examples: list[AnswerExample],
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    positive_weight: float,
    contrast_weight: float,
    branch_position: int,
    hard_negative_count: int,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    contrast = direct_answer_hard_branch_contrast(
        model,
        tokenizer,
        example,
        branch_examples,
        rng,
        branch_position,
        hard_negative_count,
        terminator,
    )
    if contrast is None:
        return train_direct_answer_branch_repair_unlikelihood(
            model,
            tokenizer,
            example,
            fallback_lesson,
            rng,
            learning_rate,
            negative_weight,
            positive_weight,
            branch_position,
            terminator,
            params=params,
        )
    context, target_id, contrast_context, contrast_target = contrast
    return model.train_step_with_branch_contrast(
        context,
        target_id,
        contrast_context,
        contrast_target,
        learning_rate,
        negative_weight,
        contrast_weight,
        params=params,
    )


def train_direct_answer_repeat_loop_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_repeat_loop_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_first_error_unlikelihood(
            model,
            tokenizer,
            example,
            fallback_lesson,
            rng,
            learning_rate,
            negative_weight,
            terminator,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood(
        context,
        target_id,
        predicted_id,
        learning_rate,
        negative_weight,
        params=params,
    )


def train_direct_answer_early_stop_unlikelihood(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    fallback_lesson: DirectAnswerLesson,
    rng: random.Random,
    learning_rate: float,
    negative_weight: float,
    terminator: str = ANSWER_TERMINATOR,
    params: list[Scalar] | None = None,
) -> float:
    repair = direct_answer_early_stop_error(model, tokenizer, example, terminator)
    if repair is None:
        return train_direct_answer_first_error_unlikelihood(
            model,
            tokenizer,
            example,
            fallback_lesson,
            rng,
            learning_rate,
            negative_weight,
            terminator,
            params=params,
        )
    context, target_id, predicted_id, _position = repair
    return model.train_step_with_unlikelihood(
        context,
        target_id,
        predicted_id,
        learning_rate,
        negative_weight,
        params=params,
    )


def transformer_direct_answer_training_pool(
    examples: list[AnswerExample],
) -> list[AnswerExample]:
    pool: list[AnswerExample] = []
    for example in examples:
        repeats = 1 + len(example.target) // 32
        if example.target != " unknown.":
            repeats += 1
        if (
            example.source.startswith("qa:")
            or example.source.startswith("fact:")
            or example.source.startswith("bridge:")
        ):
            repeats += 2
        if example.source.endswith(":place") or example.source.endswith(":color"):
            repeats += 5
        if example.source.endswith(":owner") or example.source.endswith(":training_data"):
            repeats += 5
        if example.source.endswith(":self") or example.source.endswith(":learning"):
            repeats += 60
        if example.source.endswith(":glossary"):
            repeats += 28
        pool.extend([example] * repeats)
    return pool


def evaluate_direct_answer_records(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    max_new_chars: int,
    terminator: str = ANSWER_TERMINATOR,
    generation_config: GenerationConfig | None = None,
) -> dict[str, Any]:
    generation_config = generation_config or GenerationConfig()
    scored: list[dict[str, Any]] = []
    total_loss = 0.0
    for record in records:
        generation = model.generate_with_trace(
            tokenizer,
            record["prompt"],
            max_new_chars,
            generation_config,
            stop_at=terminator if terminator else None,
        )
        completion = generation["text"]
        target = record["target"]
        example = AnswerExample(
            prompt=record["prompt"],
            target=target,
            source=f"eval:{record['id']}",
        )
        loss = direct_answer_sequence_nll(model, tokenizer, example, terminator)
        total_loss += loss
        scored.append(
            {
                "id": record["id"],
                "target": target,
                "completion": completion,
                "generation_trace": generation["trace"],
                "generation_cache": generation["cache"],
                "exact_match": completion == target,
                "target_loss": loss,
                "completion_source": "tiny_transformer_greedy_until_terminator"
                if terminator
                else "tiny_transformer_greedy_fixed_length",
            }
        )
    exact = sum(1 for record in scored if record["exact_match"])
    failed = [record for record in scored if not record["exact_match"]]
    return {
        "count": len(scored),
        "exact": exact,
        "exact_rate": exact / len(scored) if scored else 0.0,
        "avg_target_loss": total_loss / len(scored) if scored else 0.0,
        "failed_records": failed,
    }


def audit_prompt_context_coverage(
    records: list[dict[str, Any]],
    context_size: int,
    max_missing_records: int = 12,
) -> dict[str, Any]:
    audited = 0
    covered = 0
    missing_records: list[dict[str, Any]] = []
    for record in records:
        prompt = record["prompt"]
        full_features = set(semantic_feature_names(prompt.lower()))
        if not full_features:
            continue
        audited += 1
        context_text = prompt[-context_size:]
        context_features = set(semantic_feature_names(context_text.lower()))
        missing_features = sorted(full_features - context_features)
        if not missing_features:
            covered += 1
            continue
        if len(missing_records) < max_missing_records:
            missing_records.append(
                {
                    "id": record["id"],
                    "prompt_length": len(prompt),
                    "context_size": context_size,
                    "context_text": context_text,
                    "missing_features": missing_features,
                }
            )
    missing = audited - covered
    return {
        "semantic_records": audited,
        "covered": covered,
        "missing": missing,
        "covered_rate": covered / audited if audited else 1.0,
        "missing_records": missing_records,
    }


def audit_direct_answer_branch_context_coverage(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    branch_position: int,
    terminator: str = ANSWER_TERMINATOR,
    max_records: int = 12,
) -> dict[str, Any]:
    semantic_records = 0
    covered = 0
    skipped = 0
    missing_records: list[dict[str, Any]] = []
    context_records: dict[str, list[dict[str, Any]]] = {}
    context_targets: dict[str, Counter[str]] = {}
    target_counts: Counter[str] = Counter()

    for record in records:
        example = AnswerExample(
            prompt=record["prompt"],
            target=record["target"],
            source=f"eval:{record['id']}",
        )
        branch = direct_answer_branch_context(
            model,
            tokenizer,
            example,
            branch_position,
            terminator,
        )
        if branch is None:
            skipped += 1
            continue
        context, target_id, position = branch
        context_text = tokenizer.decode(context)
        target_token = tokenizer.itos[target_id]
        target_counts[target_token] += 1
        context_records.setdefault(context_text, []).append(
            {
                "id": record["id"],
                "target": record["target"],
                "branch_position": position,
                "target_token": target_token,
            }
        )
        context_targets.setdefault(context_text, Counter())[target_token] += 1

        full_features = set(semantic_feature_names(record["prompt"].lower()))
        if not full_features:
            continue
        semantic_records += 1
        context_features = set(semantic_feature_names(context_text.lower()))
        missing_features = sorted(full_features - context_features)
        if not missing_features:
            covered += 1
            continue
        if len(missing_records) < max_records:
            missing_records.append(
                {
                    "id": record["id"],
                    "branch_position": position,
                    "context_size": model.config.context_size,
                    "context_text": context_text,
                    "missing_features": missing_features,
                    "target_token": target_token,
                }
            )

    ambiguous_records: list[dict[str, Any]] = []
    collision_contexts = 0
    ambiguous_contexts = 0
    max_context_reuse = 0
    max_target_options = 0
    for context_text, examples in context_records.items():
        target_counter = context_targets[context_text]
        max_context_reuse = max(max_context_reuse, len(examples))
        max_target_options = max(max_target_options, len(target_counter))
        if len(examples) > 1:
            collision_contexts += 1
        if len(target_counter) <= 1:
            continue
        ambiguous_contexts += 1
        if len(ambiguous_records) < max_records:
            ambiguous_records.append(
                {
                    "context_text": context_text,
                    "count": len(examples),
                    "target_tokens": [
                        {"value": value, "count": count}
                        for value, count in target_counter.most_common(12)
                    ],
                    "records": examples[:max_records],
                }
            )

    count = sum(len(records_for_context) for records_for_context in context_records.values())
    return {
        "branch_position": branch_position,
        "context_size": model.config.context_size,
        "count": count,
        "skipped": skipped,
        "semantic_records": semantic_records,
        "covered": covered,
        "missing": semantic_records - covered,
        "covered_rate": covered / semantic_records if semantic_records else 1.0,
        "unique_contexts": len(context_records),
        "collision_contexts": collision_contexts,
        "ambiguous_contexts": ambiguous_contexts,
        "max_context_reuse": max_context_reuse,
        "max_target_options": max_target_options,
        "target_tokens": [
            {"value": value, "count": count}
            for value, count in target_counts.most_common(12)
        ],
        "missing_records": missing_records,
        "ambiguous_records": ambiguous_records,
    }


def summarize_branch_context_coverage_gate(
    coverage_by_eval: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    total_count = 0
    semantic_records = 0
    covered = 0
    missing = 0
    ambiguous_contexts = 0
    collision_contexts = 0
    skipped = 0
    blocking_evals: list[dict[str, Any]] = []
    for name, coverage in sorted(coverage_by_eval.items()):
        total_count += coverage["count"]
        semantic_records += coverage["semantic_records"]
        covered += coverage["covered"]
        missing += coverage["missing"]
        ambiguous_contexts += coverage["ambiguous_contexts"]
        collision_contexts += coverage["collision_contexts"]
        skipped += coverage["skipped"]
        if coverage["missing"] or coverage["ambiguous_contexts"] or coverage["skipped"]:
            blocking_evals.append(
                {
                    "name": name,
                    "count": coverage["count"],
                    "missing": coverage["missing"],
                    "ambiguous_contexts": coverage["ambiguous_contexts"],
                    "skipped": coverage["skipped"],
                    "covered_rate": coverage["covered_rate"],
                }
            )
    passed = missing == 0 and ambiguous_contexts == 0 and skipped == 0
    return {
        "passed": passed,
        "count": total_count,
        "semantic_records": semantic_records,
        "covered": covered,
        "missing": missing,
        "covered_rate": covered / semantic_records if semantic_records else 1.0,
        "ambiguous_contexts": ambiguous_contexts,
        "collision_contexts": collision_contexts,
        "skipped": skipped,
        "blocking_evals": blocking_evals,
    }


def answer_char_loss_scalars(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    position: int,
) -> Scalar:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(example.target)
    context_ids = [*prompt_ids, *target_ids[:position]]
    context = make_context(context_ids, model.config.context_size, tokenizer.pad_id)
    return cross_entropy_scalars(model._forward_scalars(context), target_ids[position])


def sampled_choice_candidates(
    target: str,
    candidates: list[str],
    rng: random.Random,
    negative_count: int,
) -> list[str]:
    unique_negatives = sorted({candidate for candidate in candidates if candidate != target})
    if negative_count <= 0:
        selected_negatives: list[str] = []
    elif negative_count >= len(unique_negatives):
        selected_negatives = unique_negatives
    else:
        selected_negatives = rng.sample(unique_negatives, negative_count)
    return [target, *selected_negatives]


def answer_choice_loss_scalars(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    candidates: list[str],
    rng: random.Random,
    negative_count: int,
    max_chars: int = 0,
) -> tuple[Scalar, int]:
    choice_candidates = sampled_choice_candidates(
        example.target,
        candidates,
        rng,
        negative_count,
    )
    scores = [
        -answer_sequence_loss_scalars(
            model,
            tokenizer,
            example.prompt,
            candidate,
            max_chars=max_chars,
        )
        for candidate in choice_candidates
    ]
    return cross_entropy_scalars(scores, target=0), len(choice_candidates)


def train_answer_char(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    rng: random.Random,
    learning_rate: float,
) -> float:
    prompt_ids = tokenizer.encode(example.prompt)
    target_ids = tokenizer.encode(example.target)
    position = rng.randrange(len(target_ids))
    context_ids = [*prompt_ids, *target_ids[:position]]
    context = make_context(context_ids, model.config.context_size, tokenizer.pad_id)
    return model.train_step(context, target_ids[position], learning_rate)


def train_answer_mixed_step(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    example: AnswerExample,
    rng: random.Random,
    learning_rate: float,
    candidates: list[str],
    target_loss_weight: float,
    choice_loss_weight: float,
    choice_negatives: int,
    choice_max_chars: int = 0,
) -> dict[str, float]:
    if target_loss_weight <= 0.0 and choice_loss_weight <= 0.0:
        raise ValueError("at least one answer loss weight must be positive")
    params = model.parameters()
    zero_grad(params)
    target_ids = tokenizer.encode(example.target)
    position = rng.randrange(len(target_ids))
    target_loss = answer_char_loss_scalars(model, tokenizer, example, position)
    total_loss = target_loss * target_loss_weight
    choice_loss_value = 0.0
    choice_candidate_count = 0
    if choice_loss_weight > 0.0:
        choice_loss, choice_candidate_count = answer_choice_loss_scalars(
            model,
            tokenizer,
            example,
            candidates,
            rng,
            choice_negatives,
            max_chars=choice_max_chars,
        )
        choice_loss_value = choice_loss.data
        total_loss = total_loss + choice_loss * choice_loss_weight
    total_loss.backward()
    model.apply_gradients(params, learning_rate)
    return {
        "loss": total_loss.data,
        "target_loss": target_loss.data,
        "choice_loss": choice_loss_value,
        "choice_candidate_count": float(choice_candidate_count),
    }


def evaluate_answer_records(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    candidates: list[str],
    max_new_chars: int,
    include_completions: bool = True,
    selector: AnswerCandidateSelector | None = None,
    emit_selected_candidate: bool = False,
    generation_config: GenerationConfig | None = None,
) -> dict[str, Any]:
    if not include_completions:
        return evaluate_answer_candidates(
            model,
            tokenizer,
            records,
            candidates,
            selector,
            emit_selected_candidate=emit_selected_candidate,
        )
    scored = score_transformer_records(
        model,
        tokenizer,
        records,
        max_new_chars=max_new_chars,
        generation_config=generation_config or GenerationConfig(),
        candidates=candidates,
    )
    summary = summarize(scored)
    failed_exact = [record for record in scored if not record["exact_match"]]
    failed_candidate = [record for record in scored if not record["candidate_match"]]
    return {
        **summary,
        "failed_records": failed_exact,
        "failed_candidate_records": failed_candidate,
    }


def evaluate_answer_candidates(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    records: list[dict[str, Any]],
    candidates: list[str],
    selector: AnswerCandidateSelector | None = None,
    emit_selected_candidate: bool = False,
) -> dict[str, Any]:
    if emit_selected_candidate and selector is None:
        raise ValueError("selector-assisted emission requires a selector")
    scored: list[dict[str, Any]] = []
    for record in records:
        if selector is None:
            candidate_scores = [
                {
                    "target": candidate,
                    "target_nll": continuation_nll(
                        model,
                        tokenizer,
                        record["prompt"],
                        candidate,
                    ),
                }
                for candidate in candidates
            ]
            predicted_candidate = min(
                candidate_scores,
                key=lambda item: float(item["target_nll"]),
            )["target"]
            candidate_scorer = "transformer_nll"
            if record["target"] in candidates:
                target_nll = next(
                    float(item["target_nll"])
                    for item in candidate_scores
                    if item["target"] == record["target"]
                )
            else:
                target_nll = continuation_nll(
                    model,
                    tokenizer,
                    record["prompt"],
                    record["target"],
                )
        else:
            candidate_scores = [
                {
                    "target": candidate,
                    "selector_score": selector.score(record["prompt"], candidate),
                }
                for candidate in candidates
            ]
            predicted_candidate = selector.predict(record["prompt"], candidates)
            candidate_scorer = "answer_candidate_selector"
            target_nll = continuation_nll(
                model,
                tokenizer,
                record["prompt"],
                record["target"],
            )
        completion = predicted_candidate if emit_selected_candidate else None
        exact_match = completion == record["target"] if completion is not None else False
        scored.append(
            {
                "id": record["id"],
                "target": record["target"],
                "completion": completion,
                "exact_match": exact_match,
                "candidate_match": predicted_candidate == record["target"],
                "predicted_candidate": predicted_candidate,
                "candidate_scorer": candidate_scorer,
                "completion_source": "selector_candidate"
                if emit_selected_candidate
                else None,
                "target_selector_score": selector.score(record["prompt"], record["target"])
                if selector is not None
                else None,
                "target_nll": target_nll,
            }
        )
    summary = summarize(scored)
    failed_exact = [record for record in scored if not record["exact_match"]]
    failed_candidate = [record for record in scored if not record["candidate_match"]]
    return {
        **summary,
        "exact": summary["exact"] if emit_selected_candidate else None,
        "exact_rate": summary["exact_rate"] if emit_selected_candidate else None,
        "failed_records": failed_exact if emit_selected_candidate else [],
        "failed_candidate_records": failed_candidate,
    }


def normalize_answer_terminator(value: str) -> str:
    if value == r"\n":
        return "\n"
    if value == r"\t":
        return "\t"
    if value == "":
        return ""
    if len(value) != 1:
        raise ValueError("direct answer terminator must be empty or a single character")
    return value


def train_transformer_answers(args: argparse.Namespace) -> dict[str, Any]:
    ensure_curriculum(args.train_text, args.valid)
    train_text = args.train_text.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.train(train_text)
    examples = load_training_examples(args.train_text, args.corpus_dir)
    training_pool = answer_training_pool(examples)
    model, resume_metadata = initialize_transformer_for_training(args, tokenizer)
    optimizer = load_optimizer_state(
        args.resume_optimizer,
        optimization_config_from_args(args),
    )
    model.active_optimizer = optimizer
    generation_config = generation_config_from_args(args)
    rng = random.Random(args.seed)
    args.run.mkdir(parents=True, exist_ok=True)
    artifacts = TransformerRunArtifacts.from_run(
        args.run,
        direct_profile_aware=direct_answer_is_profile_aware(args),
    )
    experiment_path = artifacts.experiment_intent
    experiment_intent = transformer_experiment_intent(args)
    write_experiment_intent(experiment_path, experiment_intent)
    hygiene_path = artifacts.corpus_hygiene
    training_plan_path = artifacts.training_plan
    training_recipe_path = artifacts.training_recipe
    candidate_quarantine_path = artifacts.candidate_quarantine
    verifier_path = artifacts.closed_world_verifier
    constraint_first_path = artifacts.constraint_first_promotion
    retrieval_memory_path = artifacts.retrieval_memory
    memory_consolidation_plan_path = artifacts.memory_consolidation_plan
    candidate_quarantine = build_candidate_quarantine_manifest(
        "transformer-answer-train",
        args.run.name,
    )
    write_candidate_quarantine(candidate_quarantine_path, candidate_quarantine)
    candidate_summary = candidate_quarantine_summary(candidate_quarantine)
    planned_replay_path = artifacts.replay_plan
    planned_artifacts = artifacts.training_plan_artifacts()
    training_recipe = transformer_training_recipe(
        args,
        tokenizer,
        planned_artifacts,
        experiment_intent["acceptance_gates"],
        planned_replay_path,
    )
    write_training_recipe(training_recipe_path, training_recipe)
    corpus_hygiene = build_corpus_hygiene_report(
        "transformer-answer-train",
        args.corpus_dir,
        args.train_text,
        DEFAULT_ANSWER_EVALS,
        examples,
    )
    training_plan = build_training_plan(
        "transformer-answer-train",
        args.run.name,
        args.train_text,
        args.corpus_dir,
        DEFAULT_ANSWER_EVALS,
        examples,
        training_pool,
        hygiene_path,
        planned_artifacts=planned_artifacts,
        replay_plan_path=planned_replay_path,
        candidate_quarantine_path=candidate_quarantine_path,
        candidate_quarantine_summary=candidate_summary,
    )
    training_plan = attach_recipe_summary(
        training_plan,
        training_recipe,
        training_recipe_path,
    )
    write_json_artifact(hygiene_path, corpus_hygiene)
    write_json_artifact(training_plan_path, training_plan)
    closed_world_verifier = verify_training_plan(
        training_plan,
        corpus_hygiene=corpus_hygiene,
        candidate_quarantine=candidate_quarantine,
        subject_path=training_plan_path,
        verifier_path=verifier_path,
    )
    write_verifier_report(verifier_path, closed_world_verifier)
    training_plan = attach_verifier_summary(
        training_plan,
        closed_world_verifier,
        verifier_path,
    )
    write_json_artifact(training_plan_path, training_plan)
    if not closed_world_verifier["passed"]:
        raise ValueError("closed-world verifier rejected the training plan")
    history_path = artifacts.metrics_history
    lessons_path = artifacts.lessons
    write_lessons(examples, lessons_path)
    history_writer = JsonlHistoryWriter(history_path)
    eval_records = {
        path.stem: read_jsonl(path)
        for path in DEFAULT_ANSWER_EVALS
    }
    retrieval_memory = build_retrieval_memory_report(
        args.corpus_dir,
        DEFAULT_ANSWER_EVALS,
    )
    write_retrieval_memory_report(retrieval_memory_path, retrieval_memory)
    context_coverage = {
        name: audit_prompt_context_coverage(records, args.context_size)
        for name, records in sorted(eval_records.items())
    }
    candidates = sorted(
        {
            record["target"]
            for records in eval_records.values()
            for record in records
        }
    )
    training_candidates = sorted(
        {example.target for example in examples}
        | {
            record["target"]
            for records in eval_records.values()
            for record in records
        }
    )
    eval_candidates = {
        name: sorted({record["target"] for record in records})
        for name, records in eval_records.items()
    }

    def snapshot(
        step: int,
        train_loss: float | None,
        train_target_loss: float | None = None,
        train_choice_loss: float | None = None,
        train_choice_candidates: float | None = None,
    ) -> dict[str, Any]:
        record = {
            "step": step,
            "train_loss": train_loss,
            "train_target_loss": train_target_loss,
            "train_choice_loss": train_choice_loss,
            "train_choice_candidates": train_choice_candidates,
            "evals": {
                name: evaluate_answer_records(
                    model,
                    tokenizer,
                    records,
                    candidates if args.candidate_scope == "all" else eval_candidates[name],
                    args.max_new_chars,
                    include_completions=args.include_completions,
                    generation_config=generation_config,
                )
                for name, records in sorted(eval_records.items())
            },
        }
        return history_writer.append(record)

    baseline = snapshot(0, None)
    loss_accumulator = LossAccumulator()
    last_snapshot = baseline
    last_snapshot_step = 0
    training_cursor = ShuffledTrainingCursor(training_pool, rng)
    for step in range(1, args.steps + 1):
        example = training_cursor.next()
        if args.choice_loss_weight > 0.0 or args.target_loss_weight != 1.0:
            step_result = train_answer_mixed_step(
                model,
                tokenizer,
                example,
                rng,
                args.learning_rate,
                training_candidates,
                args.target_loss_weight,
                args.choice_loss_weight,
                args.choice_negatives,
                args.choice_max_chars,
            )
            loss_accumulator.add(
                step_result["loss"],
                step_result["target_loss"],
                step_result["choice_loss"],
                step_result["choice_candidate_count"],
            )
        else:
            loss = train_answer_char(model, tokenizer, example, rng, args.learning_rate)
            loss_accumulator.add(loss)
        if args.eval_every > 0 and step % args.eval_every == 0:
            averages = loss_accumulator.average(
                args.eval_every,
                include_choice=args.choice_loss_weight > 0.0,
            )
            last_snapshot = snapshot(
                step,
                averages["train_loss"],
                averages["train_target_loss"],
                averages["train_choice_loss"],
                averages["train_choice_candidates"],
            )
            last_snapshot_step = step
            print(f"step={step} train_loss={averages['train_loss']:.4f}")
            loss_accumulator.reset()

    if last_snapshot_step != args.steps:
        last_snapshot = snapshot(args.steps, None)

    direct_answer_metrics: dict[str, Any] | None = None
    post_direct_candidate_snapshot: dict[str, Any] | None = None
    post_direct_candidate_snapshot_skipped = False
    if args.direct_answer_steps > 0:
        direct_answer_terminator = normalize_answer_terminator(args.direct_answer_terminator)
        if direct_answer_terminator and direct_answer_terminator not in tokenizer.stoi:
            raise ValueError(
                "direct answer terminator is outside the admitted tokenizer vocabulary"
            )
        direct_training_pool = transformer_direct_answer_training_pool(examples)
        direct_lessons = {
            example: direct_answer_lesson(
                tokenizer,
                model.config.context_size,
                example,
                direct_answer_terminator,
            )
            for example in sorted(
                set(direct_training_pool),
                key=lambda item: (item.prompt, item.target, item.source),
            )
        }
        direct_rng = random.Random(args.seed + 307)
        direct_history_path = args.run / "direct_answer_metrics.jsonl"
        direct_history_writer = JsonlHistoryWriter(direct_history_path)
        direct_profile_aware_targets = is_profile_aware_direct_answer_mode(
            args.direct_answer_mode
        )
        direct_replay_plan_path = (
            args.run / "direct_answer_replay_plan.json"
            if direct_profile_aware_targets
            else None
        )
        direct_replay_plan = None
        direct_replay_prediction_overrides: ReplayPredictionOverrides | None = None
        direct_replay_prediction_anchors_active = (
            args.direct_answer_mode in BASELINE_ANCHORED_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_update_gate_active = (
            args.direct_answer_mode in BASELINE_FLOOR_GATED_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_adaptive_updates_active = (
            args.direct_answer_mode in BASELINE_FLOOR_ADAPTIVE_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_repaired_updates_active = (
            args.direct_answer_mode in BASELINE_FLOOR_REPAIRED_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_objective_active = (
            args.direct_answer_mode in BASELINE_FLOOR_OBJECTIVE_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_stabilization_active = (
            args.direct_answer_mode in BASELINE_FLOOR_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_profile_targeted_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_PROFILE_TARGETED_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_sequential_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_SEQUENTIAL_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_calibrated_sequential_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_CALIBRATED_SEQUENTIAL_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_PROFILE_SCALE_CALIBRATED_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_profile_scale_diversity_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_profile_scale_frontier_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_PROFILE_SCALE_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_PREP_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active = (
            args.direct_answer_mode
            in BASELINE_FLOOR_PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_STABILIZATION_DIRECT_ANSWER_MODES
        )
        direct_remaining_profile_binding_target_profiles = list(
            BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES
            if direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
            else BASELINE_FLOOR_REMAINING_PROFILE_BINDING_TARGET_PROFILES
        )
        direct_remaining_profile_binding_source_labels = (
            remaining_profile_binding_source_labels(
                direct_remaining_profile_binding_target_profiles
            )
            if direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
            else []
        )
        direct_baseline_floor_learning_rate_scales = (
            BASELINE_FLOOR_CALIBRATED_ADAPTIVE_LEARNING_RATE_SCALES
            if direct_answer_baseline_floor_calibrated_sequential_stabilization_active
            else BASELINE_FLOOR_ADAPTIVE_LEARNING_RATE_SCALES
        )
        direct_baseline_floor_outer_learning_rate_scales = (
            (1.0,)
            if direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active
            else direct_baseline_floor_learning_rate_scales
        )
        direct_replay_records: list[BranchReplayRecord] = []
        direct_baseline_floor_repair_anchors: list[BranchReplayRecord] = []
        direct_baseline_floor_frontier_anchors: list[BranchReplayRecord] = []
        if direct_profile_aware_targets:
            replay_records = direct_answer_profiled_replay_records(
                model,
                tokenizer,
                direct_training_pool,
                args.direct_answer_branch_position,
                direct_answer_terminator,
            )
            direct_replay_records = replay_records
            if (
                direct_answer_baseline_floor_repaired_updates_active
                or direct_answer_baseline_floor_objective_active
                or direct_answer_baseline_floor_stabilization_active
            ):
                direct_baseline_floor_repair_anchors = (
                    baseline_floor_repair_anchor_records(direct_replay_records)
                )
            if direct_answer_baseline_floor_profile_scale_frontier_stabilization_active:
                direct_baseline_floor_frontier_anchors = (
                    baseline_floor_frontier_anchor_records(
                        direct_baseline_floor_repair_anchors,
                        direct_replay_records,
                    )
                )
            direct_replay_prediction_overrides = {
                (tuple(context), target, profile): predicted
                for context, target, predicted, profile in (
                    branch_replay_parts(record) for record in replay_records
                )
            }
            direct_replay_plan = branch_replay_plan(
                replay_records,
                replay_records,
                profile_aware_targets=True,
            )
            direct_replay_plan["mode"] = args.direct_answer_mode
            direct_replay_plan["branch_position"] = args.direct_answer_branch_position
            direct_replay_plan["training_examples"] = len(direct_training_pool)
            direct_replay_plan["baseline_prediction_anchor_count"] = len(
                direct_replay_prediction_overrides
            )
            direct_replay_plan["baseline_prediction_anchors_active"] = (
                direct_replay_prediction_anchors_active
            )
            direct_replay_plan["baseline_floor_update_gate_active"] = (
                direct_answer_baseline_floor_update_gate_active
            )
            direct_replay_plan["baseline_floor_adaptive_updates_active"] = (
                direct_answer_baseline_floor_adaptive_updates_active
            )
            direct_replay_plan["baseline_floor_repaired_updates_active"] = (
                direct_answer_baseline_floor_repaired_updates_active
            )
            direct_replay_plan["baseline_floor_objective_active"] = (
                direct_answer_baseline_floor_objective_active
            )
            direct_replay_plan["baseline_floor_stabilization_active"] = (
                direct_answer_baseline_floor_stabilization_active
            )
            direct_replay_plan[
                "baseline_floor_profile_targeted_stabilization_active"
            ] = direct_answer_baseline_floor_profile_targeted_stabilization_active
            direct_replay_plan[
                "baseline_floor_sequential_stabilization_active"
            ] = direct_answer_baseline_floor_sequential_stabilization_active
            direct_replay_plan[
                "baseline_floor_calibrated_sequential_stabilization_active"
            ] = (
                direct_answer_baseline_floor_calibrated_sequential_stabilization_active
            )
            direct_replay_plan[
                "baseline_floor_profile_scale_calibrated_stabilization_active"
            ] = (
                direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active
            )
            direct_replay_plan[
                "baseline_floor_profile_scale_diversity_stabilization_active"
            ] = (
                direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
            )
            direct_replay_plan[
                "baseline_floor_profile_scale_frontier_stabilization_active"
            ] = (
                direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
            )
            direct_replay_plan[
                "baseline_floor_profile_scale_coverage_frontier_stabilization_active"
            ] = (
                direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
            )
            direct_replay_plan[
                "baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active"
            ] = (
                direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
            )
            direct_replay_plan[
                "baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active"
            ] = (
                direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active
            )
            direct_replay_plan[
                "baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active"
            ] = (
                direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
            )
            direct_replay_plan[
                "baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active"
            ] = (
                direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active
            )
            direct_replay_plan[
                "baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active"
            ] = (
                direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active
            )
            direct_replay_plan[
                "baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active"
            ] = (
                direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
            )
            direct_replay_plan[
                "baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active"
            ] = (
                direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
            )
            direct_replay_plan["baseline_floor_repair_anchor_count"] = len(
                direct_baseline_floor_repair_anchors
            )
            direct_replay_plan["baseline_floor_repair_steps"] = (
                BASELINE_FLOOR_REPAIR_STEPS
                if direct_answer_baseline_floor_repaired_updates_active
                else 0
            )
            direct_replay_plan["baseline_floor_objective_anchor_count"] = len(
                direct_baseline_floor_repair_anchors
            )
            direct_replay_plan["baseline_floor_objective_anchor_batch_size"] = (
                BASELINE_FLOOR_OBJECTIVE_ANCHOR_BATCH_SIZE
                if direct_answer_baseline_floor_objective_active
                else 0
            )
            direct_replay_plan["baseline_floor_objective_anchor_weight"] = (
                BASELINE_FLOOR_OBJECTIVE_ANCHOR_WEIGHT
                if direct_answer_baseline_floor_objective_active
                else 0.0
            )
            direct_replay_plan["baseline_floor_stabilization_anchor_count"] = len(
                direct_baseline_floor_repair_anchors
            )
            direct_replay_plan["baseline_floor_stabilization_anchor_batch_size"] = (
                len(direct_baseline_floor_repair_anchors)
                if (
                    direct_answer_baseline_floor_profile_targeted_stabilization_active
                    or direct_answer_baseline_floor_sequential_stabilization_active
                )
                else (
                    BASELINE_FLOOR_STABILIZATION_ANCHOR_BATCH_SIZE
                    if direct_answer_baseline_floor_stabilization_active
                    else 0
                )
            )
            direct_replay_plan[
                "baseline_floor_stabilization_profile_target_count"
            ] = baseline_floor_anchor_profile_target_count(
                direct_baseline_floor_repair_anchors
            )
            direct_replay_plan[
                "baseline_floor_stabilization_anchor_profile_counts"
            ] = baseline_floor_anchor_profile_counts(
                direct_baseline_floor_repair_anchors
            )
            direct_replay_plan[
                "baseline_floor_stabilization_profile_group_count"
            ] = len(
                baseline_floor_anchor_profile_counts(
                    direct_baseline_floor_repair_anchors
                )
            )
            direct_replay_plan["baseline_floor_frontier_anchor_count"] = len(
                direct_baseline_floor_frontier_anchors
            )
            direct_replay_plan["baseline_floor_frontier_anchor_profile_counts"] = (
                baseline_floor_anchor_profile_counts(
                    direct_baseline_floor_frontier_anchors
                )
            )
            direct_replay_plan["baseline_floor_frontier_profile_group_count"] = len(
                baseline_floor_anchor_profile_counts(
                    direct_baseline_floor_frontier_anchors
                )
            )
            direct_replay_plan["baseline_floor_frontier_profile_target_count"] = (
                baseline_floor_anchor_profile_target_count(
                    direct_baseline_floor_frontier_anchors
                )
            )
            if direct_answer_baseline_floor_adaptive_updates_active:
                direct_replay_plan["adaptive_learning_rate_scales"] = list(
                    direct_baseline_floor_learning_rate_scales
                )
                direct_replay_plan["outer_learning_rate_scales"] = list(
                    direct_baseline_floor_outer_learning_rate_scales
                )
            if (
                direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active
            ):
                direct_replay_plan[
                    "collapsed_profile_binding_learning_rate_scales"
                ] = list(BASELINE_FLOOR_COLLAPSED_PROFILE_BINDING_LEARNING_RATE_SCALES)
            if (
                direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
            ):
                direct_replay_plan[
                    "remaining_profile_binding_target_profiles"
                ] = direct_remaining_profile_binding_target_profiles
                direct_replay_plan[
                    "remaining_profile_binding_source_labels"
                ] = direct_remaining_profile_binding_source_labels
                direct_replay_plan[
                    "remaining_profile_binding_source_profiles"
                ] = [
                    profile
                    for profile in sorted(direct_replay_plan["profiles"])
                    if source_profile_label(profile)
                    in set(direct_remaining_profile_binding_source_labels)
                ]
                if (
                    direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                ):
                    direct_replay_plan[
                        "owner_paraphrase_binding_target_profiles"
                    ] = list(BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES)
                    direct_replay_plan[
                        "owner_paraphrase_binding_preserved_profiles"
                    ] = list(
                        BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES
                    )
                    direct_replay_plan[
                        "owner_paraphrase_binding_source_labels"
                    ] = direct_remaining_profile_binding_source_labels
                    direct_replay_plan[
                        "owner_paraphrase_binding_source_profiles"
                    ] = direct_replay_plan[
                        "remaining_profile_binding_source_profiles"
                    ]
            if direct_replay_plan_path is not None:
                with direct_replay_plan_path.open("w", encoding="utf-8") as handle:
                    json.dump(direct_replay_plan, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                training_plan = attach_replay_plan_summary(
                    training_plan,
                    direct_replay_plan,
                    direct_replay_plan_path,
                )
                write_json_artifact(training_plan_path, training_plan)

        def direct_snapshot_record(
            step: int,
            train_loss: float | None,
            extra: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            direct_answer_evals_skipped = args.direct_answer_snapshot_mode == "branch-only"
            coverage_only_probe = bool(
                extra
                and extra.get("baseline_floor_update_guard_probe")
                and extra.get("baseline_floor_coverage_only_probe", True)
            )
            branch_context_coverage = (
                {}
                if coverage_only_probe
                else {
                    name: audit_direct_answer_branch_context_coverage(
                        model,
                        tokenizer,
                        records,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                    )
                    for name, records in sorted(eval_records.items())
                }
            )
            branch_profiles = {
                name: direct_answer_branch_profile(
                    model,
                    tokenizer,
                    records,
                    args.direct_answer_branch_position,
                    direct_answer_terminator,
                )
                for name, records in sorted(eval_records.items())
            }
            branch_representation_profiles = (
                {}
                if coverage_only_probe
                else {
                    name: direct_answer_branch_representation_profile(
                        model,
                        tokenizer,
                        records,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                    )
                    for name, records in sorted(eval_records.items())
                }
            )
            record = {
                "step": step,
                "train_loss": train_loss,
                "direct_answer_snapshot_mode": args.direct_answer_snapshot_mode,
                "evals_skipped": direct_answer_evals_skipped,
                "evals": {}
                if direct_answer_evals_skipped
                else {
                    name: evaluate_direct_answer_records(
                        model,
                        tokenizer,
                        records,
                        args.direct_answer_max_new_chars,
                        direct_answer_terminator,
                        generation_config,
                    )
                    for name, records in sorted(eval_records.items())
                },
                "branch_profiles": branch_profiles,
                "branch_representation_profiles": branch_representation_profiles,
                "branch_diversity_target": summarize_branch_diversity_target(
                    branch_profiles
                ),
                "branch_context_coverage": branch_context_coverage,
                "branch_context_gate": summarize_branch_context_coverage_gate(
                    branch_context_coverage
                ),
                "coverage_only_probe": coverage_only_probe,
            }
            record["branch_target_coverage_by_profile"] = (
                branch_diversity_snapshot_target_coverage_by_profile(record)
            )
            if extra is not None:
                record.update(extra)
            return record

        def direct_snapshot(
            step: int,
            train_loss: float | None,
            extra: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return direct_history_writer.append(
                direct_snapshot_record(step, train_loss, extra)
            )

        direct_baseline = direct_snapshot(0, None)
        best_direct_snapshot_step = 0
        best_direct_snapshot_score = branch_diversity_snapshot_score(direct_baseline)
        best_direct_model_payload = model.to_dict(tokenizer)
        best_direct_optimizer_payload = optimizer.to_dict()

        def record_best_direct_snapshot(snapshot: dict[str, Any]) -> None:
            nonlocal best_direct_snapshot_step
            nonlocal best_direct_snapshot_score
            nonlocal best_direct_model_payload
            nonlocal best_direct_optimizer_payload
            score = branch_diversity_snapshot_score(snapshot)
            if not branch_diversity_snapshot_preserves_target_coverage(
                snapshot,
                direct_baseline,
            ):
                return
            if score > best_direct_snapshot_score:
                best_direct_snapshot_step = int(snapshot["step"])
                best_direct_snapshot_score = score
                best_direct_model_payload = model.to_dict(tokenizer)
                best_direct_optimizer_payload = optimizer.to_dict()

        branch_context_gate = direct_baseline["branch_context_gate"]
        direct_answer_training_skipped = (
            args.direct_answer_require_branch_context_gate
            and not branch_context_gate["passed"]
        )
        direct_answer_skip_reason = (
            "branch_context_gate_failed"
            if direct_answer_training_skipped
            else None
        )
        direct_steps_to_run = 0 if direct_answer_training_skipped else args.direct_answer_steps
        if direct_answer_training_skipped:
            print("skipped direct-answer training: branch context gate failed")
        running_direct_loss = 0.0
        last_direct_snapshot = direct_baseline
        last_direct_snapshot_step = 0
        direct_training_cursor = ShuffledTrainingCursor(direct_training_pool, direct_rng)
        model.freeze_lower_layers_for_updates = (
            args.direct_answer_train_top_layer_only and model.config.num_layers > 1
        )
        direct_params = (
            model.top_layer_parameters()
            if args.direct_answer_train_top_layer_only
            else model.parameters()
        )
        if args.direct_answer_freeze_output_bias:
            direct_params = exclude_scalars(direct_params, model.bout)
        direct_answer_update_guard = {
            "active": direct_answer_baseline_floor_update_gate_active,
            "adaptive": direct_answer_baseline_floor_adaptive_updates_active,
            "repair_active": direct_answer_baseline_floor_repaired_updates_active,
            "objective_active": direct_answer_baseline_floor_objective_active,
            "stabilization_active": direct_answer_baseline_floor_stabilization_active,
            "profile_targeted_stabilization_active": (
                direct_answer_baseline_floor_profile_targeted_stabilization_active
            ),
            "sequential_stabilization_active": (
                direct_answer_baseline_floor_sequential_stabilization_active
            ),
            "calibrated_sequential_stabilization_active": (
                direct_answer_baseline_floor_calibrated_sequential_stabilization_active
            ),
            "profile_scale_calibrated_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active
            ),
            "profile_scale_diversity_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
            ),
            "profile_scale_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
            ),
            "profile_scale_coverage_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
            ),
            "profile_scale_coverage_prep_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
            ),
            "profile_scale_coverage_recovery_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active
            ),
            "profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
            ),
            "profile_scale_branch_diversity_recovery_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active
            ),
            "profile_scale_collapsed_profile_binding_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active
            ),
            "profile_scale_remaining_profile_binding_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
            ),
            "profile_scale_owner_paraphrase_binding_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
            ),
            "profile_scale_coverage_recovery_learning_rate_scales": (
                list(BASELINE_FLOOR_COVERAGE_RECOVERY_LEARNING_RATE_SCALES)
                if direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active
                else []
            ),
            "profile_scale_branch_diversity_recovery_learning_rate_scales": (
                list(BASELINE_FLOOR_BRANCH_DIVERSITY_RECOVERY_LEARNING_RATE_SCALES)
                if direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active
                else []
            ),
            "profile_scale_collapsed_profile_binding_learning_rate_scales": (
                list(BASELINE_FLOOR_COLLAPSED_PROFILE_BINDING_LEARNING_RATE_SCALES)
                if direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active
                else []
            ),
            "learning_rate_scales": list(
                direct_baseline_floor_learning_rate_scales
            )
            if direct_answer_baseline_floor_adaptive_updates_active
            else [1.0],
            "outer_learning_rate_scales": list(
                direct_baseline_floor_outer_learning_rate_scales
            )
            if direct_answer_baseline_floor_adaptive_updates_active
            else [1.0],
            "repair_anchor_count": len(direct_baseline_floor_repair_anchors),
            "objective_anchor_count": len(direct_baseline_floor_repair_anchors),
            "objective_anchor_batch_size": (
                BASELINE_FLOOR_OBJECTIVE_ANCHOR_BATCH_SIZE
                if direct_answer_baseline_floor_objective_active
                else 0
            ),
            "objective_anchor_weight": (
                BASELINE_FLOOR_OBJECTIVE_ANCHOR_WEIGHT
                if direct_answer_baseline_floor_objective_active
                else 0.0
            ),
            "stabilization_anchor_count": len(direct_baseline_floor_repair_anchors),
            "stabilization_anchor_batch_size": (
                len(direct_baseline_floor_repair_anchors)
                if (
                    direct_answer_baseline_floor_profile_targeted_stabilization_active
                    or direct_answer_baseline_floor_sequential_stabilization_active
                )
                else (
                    BASELINE_FLOOR_STABILIZATION_ANCHOR_BATCH_SIZE
                    if direct_answer_baseline_floor_stabilization_active
                    else 0
                )
            ),
            "stabilization_profile_target_count": (
                baseline_floor_anchor_profile_target_count(
                    direct_baseline_floor_repair_anchors
                )
            ),
            "stabilization_anchor_profile_counts": (
                baseline_floor_anchor_profile_counts(
                    direct_baseline_floor_repair_anchors
                )
            ),
            "stabilization_profile_group_count": len(
                baseline_floor_anchor_profile_counts(
                    direct_baseline_floor_repair_anchors
                )
            ),
            "frontier_anchor_count": len(direct_baseline_floor_frontier_anchors),
            "frontier_anchor_profile_counts": (
                baseline_floor_anchor_profile_counts(
                    direct_baseline_floor_frontier_anchors
                )
            ),
            "frontier_profile_group_count": len(
                baseline_floor_anchor_profile_counts(
                    direct_baseline_floor_frontier_anchors
                )
            ),
            "frontier_profile_target_count": (
                baseline_floor_anchor_profile_target_count(
                    direct_baseline_floor_frontier_anchors
                )
            ),
            "repair_steps_per_attempt": (
                BASELINE_FLOOR_REPAIR_STEPS
                if direct_answer_baseline_floor_repaired_updates_active
                else 0
            ),
            "checked_steps": 0,
            "attempted_updates": 0,
            "repair_attempts": 0,
            "repair_updates": 0,
            "objective_anchor_batches": 0,
            "objective_anchor_records": 0,
            "stabilization_anchor_batches": 0,
            "stabilization_anchor_records": 0,
            "sequential_profile_attempts": 0,
            "sequential_profile_acceptances": 0,
            "sequential_profile_rejections": 0,
            "sequential_profile_records": 0,
            "sequential_profile_acceptance_counts": {},
            "sequential_profile_rejection_counts": {},
            "sequential_profile_probe_sample": [],
            "rejected_no_effective_update_attempts": 0,
            "profile_scale_memory_attempts": 0,
            "profile_scale_memory_acceptances": 0,
            "profile_scale_memory_rejections": 0,
            "profile_scale_acceptance_scale_counts": {},
            "profile_scale_rejection_scale_counts": {},
            "profile_scale_profile_acceptance_scales": {},
            "profile_scale_probe_sample": [],
            "profile_scale_diversity_attempts": 0,
            "profile_scale_diversity_acceptances": 0,
            "profile_scale_diversity_rejections": 0,
            "profile_scale_diversity_score_improvements": 0,
            "profile_scale_diversity_score_ties": 0,
            "profile_scale_diversity_score_regressions": 0,
            "profile_scale_diversity_floor_rejections": 0,
            "profile_scale_diversity_outer_acceptances": 0,
            "profile_scale_diversity_outer_rejections": 0,
            "profile_scale_diversity_profile_acceptance_outcomes": {},
            "profile_scale_diversity_rejection_reasons": {},
            "profile_scale_diversity_probe_sample": [],
            "profile_scale_frontier_attempts": 0,
            "profile_scale_frontier_acceptances": 0,
            "profile_scale_frontier_rejections": 0,
            "profile_scale_frontier_records": 0,
            "profile_scale_frontier_probe_sample": [],
            "profile_scale_coverage_frontier_attempts": 0,
            "profile_scale_coverage_frontier_acceptances": 0,
            "profile_scale_coverage_frontier_rejections": 0,
            "profile_scale_coverage_frontier_gains": 0,
            "profile_scale_coverage_frontier_ties": 0,
            "profile_scale_coverage_frontier_regressions": 0,
            "profile_scale_coverage_frontier_rejection_reasons": {},
            "profile_scale_coverage_frontier_profile_acceptance_deltas": {},
            "profile_scale_coverage_frontier_probe_sample": [],
            "profile_scale_coverage_prep_frontier_attempts": 0,
            "profile_scale_coverage_prep_frontier_acceptances": 0,
            "profile_scale_coverage_prep_frontier_gain_acceptances": 0,
            "profile_scale_coverage_prep_frontier_preparations": 0,
            "profile_scale_coverage_prep_frontier_rejections": 0,
            "profile_scale_coverage_prep_frontier_rejection_reasons": {},
            "profile_scale_coverage_prep_frontier_profile_acceptance_outcomes": {},
            "profile_scale_coverage_prep_frontier_probe_sample": [],
            "profile_scale_coverage_recovery_frontier_prepared_candidates": 0,
            "profile_scale_coverage_recovery_frontier_attempts": 0,
            "profile_scale_coverage_recovery_frontier_acceptances": 0,
            "profile_scale_coverage_recovery_frontier_fallback_preparations": 0,
            "profile_scale_coverage_recovery_frontier_rejections": 0,
            "profile_scale_coverage_recovery_frontier_records": 0,
            "profile_scale_coverage_recovery_frontier_rejection_reasons": {},
            "profile_scale_coverage_recovery_frontier_profile_acceptance_outcomes": {},
            "profile_scale_coverage_recovery_frontier_probe_sample": [],
            "profile_scale_branch_stable_coverage_recovery_frontier_checks": 0,
            "profile_scale_branch_stable_coverage_recovery_frontier_acceptances": 0,
            "profile_scale_branch_stable_coverage_recovery_frontier_rejections": 0,
            "profile_scale_branch_stable_coverage_recovery_frontier_fallback_preparations": 0,
            "profile_scale_branch_stable_coverage_recovery_frontier_rejection_reasons": {},
            "profile_scale_branch_stable_coverage_recovery_frontier_profile_acceptance_outcomes": {},
            "profile_scale_branch_stable_coverage_recovery_frontier_probe_sample": [],
            "profile_scale_branch_diversity_recovery_frontier_candidates": 0,
            "profile_scale_branch_diversity_recovery_frontier_attempts": 0,
            "profile_scale_branch_diversity_recovery_frontier_acceptances": 0,
            "profile_scale_branch_diversity_recovery_frontier_fallback_acceptances": 0,
            "profile_scale_branch_diversity_recovery_frontier_rejections": 0,
            "profile_scale_branch_diversity_recovery_frontier_records": 0,
            "profile_scale_branch_diversity_recovery_frontier_rejection_reasons": {},
            "profile_scale_branch_diversity_recovery_frontier_profile_acceptance_outcomes": {},
            "profile_scale_branch_diversity_recovery_frontier_profile_score_deltas": {},
            "profile_scale_branch_diversity_recovery_frontier_probe_sample": [],
            "profile_scale_collapsed_profile_binding_frontier_candidates": 0,
            "profile_scale_collapsed_profile_binding_frontier_attempts": 0,
            "profile_scale_collapsed_profile_binding_frontier_acceptances": 0,
            "profile_scale_collapsed_profile_binding_frontier_fallback_acceptances": 0,
            "profile_scale_collapsed_profile_binding_frontier_rejections": 0,
            "profile_scale_collapsed_profile_binding_frontier_records": 0,
            "profile_scale_collapsed_profile_binding_frontier_rejection_reasons": {},
            "profile_scale_collapsed_profile_binding_frontier_profile_acceptance_outcomes": {},
            "profile_scale_collapsed_profile_binding_frontier_profile_deltas": {},
            "profile_scale_collapsed_profile_binding_frontier_probe_sample": [],
            "profile_scale_remaining_profile_binding_target_profiles": (
                direct_remaining_profile_binding_target_profiles
                if direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
                else []
            ),
            "profile_scale_remaining_profile_binding_source_labels": (
                direct_remaining_profile_binding_source_labels
                if direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
                else []
            ),
            "profile_scale_remaining_profile_binding_source_profiles": (
                list(
                    direct_replay_plan.get(
                        "remaining_profile_binding_source_profiles",
                        [],
                    )
                )
                if (
                    direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
                    and isinstance(direct_replay_plan, dict)
                )
                else []
            ),
            "profile_scale_remaining_profile_binding_prioritized_attempts": 0,
            "profile_scale_remaining_profile_binding_prioritized_acceptances": 0,
            "profile_scale_remaining_profile_binding_prioritized_rejections": 0,
            "profile_scale_remaining_profile_binding_probe_sample": [],
            "profile_scale_owner_paraphrase_binding_target_profiles": (
                list(BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES)
                if direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                else []
            ),
            "profile_scale_owner_paraphrase_binding_source_labels": (
                direct_remaining_profile_binding_source_labels
                if direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                else []
            ),
            "profile_scale_owner_paraphrase_binding_source_profiles": (
                list(
                    direct_replay_plan.get(
                        "owner_paraphrase_binding_source_profiles",
                        [],
                    )
                )
                if (
                    direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                    and isinstance(direct_replay_plan, dict)
                )
                else []
            ),
            "profile_scale_owner_paraphrase_binding_preserved_profiles": (
                list(BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES)
                if direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                else []
            ),
            "profile_scale_owner_paraphrase_binding_prioritized_attempts": 0,
            "profile_scale_owner_paraphrase_binding_prioritized_acceptances": 0,
            "profile_scale_owner_paraphrase_binding_prioritized_rejections": 0,
            "profile_scale_owner_paraphrase_binding_preservation_checks": 0,
            "profile_scale_owner_paraphrase_binding_preservation_failures": 0,
            "profile_scale_owner_paraphrase_binding_probe_sample": [],
            "calibrated_min_learning_rate_scale": (
                min(direct_baseline_floor_learning_rate_scales)
                if direct_answer_baseline_floor_calibrated_sequential_stabilization_active
                else None
            ),
            "accepted_steps": 0,
            "accepted_attempts": 0,
            "repaired_steps": 0,
            "repaired_attempts": 0,
            "stabilized_steps": 0,
            "stabilized_attempts": 0,
            "rejected_steps": 0,
            "rejected_attempts": 0,
            "accepted_learning_rate_scale_counts": {},
            "accepted_update_shape_counts": {},
            "rejected_learning_rate_scale_counts": {},
            "rejected_update_shape_counts": {},
            "rejected_violation_profile_counts": {},
            "floor_diagnostics_active": direct_answer_baseline_floor_update_gate_active,
            "rejected_floor_diagnostic_sample": [],
            "worst_rejected_coverage_deficit": 0.0,
            "worst_rejected_coverage_violation": None,
            "rejected_step_sample": [],
        }

        def refresh_direct_update_params() -> None:
            nonlocal direct_params
            model.freeze_lower_layers_for_updates = (
                args.direct_answer_train_top_layer_only and model.config.num_layers > 1
            )
            direct_params = (
                model.top_layer_parameters()
                if args.direct_answer_train_top_layer_only
                else model.parameters()
            )
            if args.direct_answer_freeze_output_bias:
                direct_params = exclude_scalars(direct_params, model.bout)

        def restore_direct_update_state(
            model_payload: dict[str, Any],
            optimizer_payload: dict[str, Any],
        ) -> None:
            nonlocal model
            nonlocal tokenizer
            nonlocal optimizer
            restored_model, restored_tokenizer = TinyTransformerLM.from_dict(model_payload)
            model = restored_model
            if restored_tokenizer is not None:
                tokenizer = restored_tokenizer
            optimizer = ScalarOptimizer.from_dict(optimizer_payload)
            model.active_optimizer = optimizer
            refresh_direct_update_params()

        def train_baseline_anchored_prompt_update(update_learning_rate: float) -> float:
            floor_preservation_branches: list[BranchReplayRecord] | None = None
            if (
                direct_answer_baseline_floor_objective_active
                and direct_baseline_floor_repair_anchors
            ):
                floor_preservation_branches = baseline_floor_objective_anchor_batch(
                    direct_baseline_floor_repair_anchors,
                    direct_rng,
                    BASELINE_FLOOR_OBJECTIVE_ANCHOR_BATCH_SIZE,
                )
                direct_answer_update_guard["objective_anchor_batches"] += 1
                direct_answer_update_guard["objective_anchor_records"] += len(
                    floor_preservation_branches
                )
            return train_direct_answer_branch_context_replay_coverage_unlikelihood(
                model,
                tokenizer,
                example,
                direct_training_pool,
                direct_lessons[example],
                direct_rng,
                update_learning_rate,
                args.direct_answer_negative_weight,
                args.direct_answer_positive_weight,
                args.direct_answer_contrast_weight,
                args.direct_answer_branch_position,
                args.direct_answer_branch_batch_size,
                args.direct_answer_hard_negatives,
                direct_answer_terminator,
                direct_params,
                balance_targets=True,
                focus_uncovered_targets=True,
                preserve_predicted_target_coverage=True,
                balance_deficit_targets=True,
                profile_aware_targets=True,
                balance_profile_target_shares=True,
                enforce_prompt_target_margins=True,
                replay_prediction_overrides=direct_replay_prediction_overrides,
                floor_preservation_branches=floor_preservation_branches,
                floor_preservation_weight=(
                    BASELINE_FLOOR_OBJECTIVE_ANCHOR_WEIGHT
                    if direct_answer_baseline_floor_objective_active
                    else 0.0
                ),
                balance_floor_preservation_targets=(
                    direct_answer_baseline_floor_objective_active
                ),
            )

        def record_guard_rejection_attempt(
            direct_step: int,
            probe_snapshot: dict[str, Any],
            learning_rate_scale: float,
            update_shape: str = "direct",
        ) -> None:
            direct_answer_update_guard["rejected_attempts"] += 1
            scale_key = f"{learning_rate_scale:g}"
            rejected_scale_counts = direct_answer_update_guard[
                "rejected_learning_rate_scale_counts"
            ]
            if isinstance(rejected_scale_counts, dict):
                rejected_scale_counts[scale_key] = (
                    int(rejected_scale_counts.get(scale_key, 0)) + 1
                )
            rejected_shape_counts = direct_answer_update_guard[
                "rejected_update_shape_counts"
            ]
            if isinstance(rejected_shape_counts, dict):
                rejected_shape_counts[update_shape] = (
                    int(rejected_shape_counts.get(update_shape, 0)) + 1
                )
            floor_diagnostics = (
                branch_diversity_snapshot_target_coverage_diagnostics(
                    probe_snapshot,
                    direct_baseline,
                )
            )
            violation_profile_counts = direct_answer_update_guard[
                "rejected_violation_profile_counts"
            ]
            if isinstance(violation_profile_counts, dict):
                for violation in floor_diagnostics["violations"]:
                    profile = str(violation["profile"])
                    violation_profile_counts[profile] = (
                        int(violation_profile_counts.get(profile, 0)) + 1
                    )
            worst_deficit = float(floor_diagnostics["worst_deficit"])
            if worst_deficit > float(
                direct_answer_update_guard["worst_rejected_coverage_deficit"]
            ):
                direct_answer_update_guard["worst_rejected_coverage_deficit"] = (
                    worst_deficit
                )
                direct_answer_update_guard["worst_rejected_coverage_violation"] = (
                    floor_diagnostics["worst_violation"]
                )
            diagnostic_sample = direct_answer_update_guard[
                "rejected_floor_diagnostic_sample"
            ]
            if isinstance(diagnostic_sample, list) and len(diagnostic_sample) < 12:
                diagnostic_sample.append(
                    {
                        "step": direct_step,
                        "learning_rate_scale": learning_rate_scale,
                        "update_shape": update_shape,
                        "preserved": floor_diagnostics["preserved"],
                        "violating_profile_count": floor_diagnostics[
                            "violating_profile_count"
                        ],
                        "worst_deficit": floor_diagnostics["worst_deficit"],
                        "worst_violation": floor_diagnostics["worst_violation"],
                        "violations": floor_diagnostics["violations"][:5],
                    }
                )
            rejected_sample = direct_answer_update_guard["rejected_step_sample"]
            if isinstance(rejected_sample, list) and len(rejected_sample) < 12:
                rejected_sample.append(
                    {
                        "step": direct_step,
                        "learning_rate_scale": learning_rate_scale,
                        "update_shape": update_shape,
                        "coverage": (
                            branch_diversity_snapshot_target_coverage_by_profile(
                                probe_snapshot
                            )
                        ),
                        "floor_diagnostics": {
                            "preserved": floor_diagnostics["preserved"],
                            "violating_profile_count": floor_diagnostics[
                                "violating_profile_count"
                            ],
                            "worst_deficit": floor_diagnostics["worst_deficit"],
                            "worst_violation": floor_diagnostics["worst_violation"],
                        },
                    }
                )

        def record_guard_acceptance(
            learning_rate_scale: float,
            update_shape: str = "direct",
        ) -> None:
            direct_answer_update_guard["accepted_steps"] += 1
            direct_answer_update_guard["accepted_attempts"] += 1
            if update_shape == "repaired":
                direct_answer_update_guard["repaired_steps"] += 1
                direct_answer_update_guard["repaired_attempts"] += 1
            if update_shape in {
                "stabilization",
                "profile_targeted_stabilization",
                "sequential_profile_stabilization",
                "calibrated_sequential_profile_stabilization",
                "profile_scale_calibrated_sequential_profile_stabilization",
                "profile_scale_diversity_calibrated_sequential_profile_stabilization",
                "profile_scale_frontier_diversity_calibrated_sequential_profile_stabilization",
                "profile_scale_coverage_frontier_diversity_calibrated_sequential_profile_stabilization",
                "profile_scale_coverage_prep_frontier_diversity_calibrated_sequential_profile_stabilization",
                "profile_scale_coverage_recovery_frontier_diversity_calibrated_sequential_profile_stabilization",
                "profile_scale_branch_stable_coverage_recovery_frontier_diversity_calibrated_sequential_profile_stabilization",
                "profile_scale_branch_diversity_recovery_frontier_calibrated_sequential_profile_stabilization",
                "profile_scale_collapsed_profile_binding_frontier_calibrated_sequential_profile_stabilization",
                "profile_scale_remaining_profile_binding_frontier_calibrated_sequential_profile_stabilization",
                "profile_scale_owner_paraphrase_binding_frontier_calibrated_sequential_profile_stabilization",
            }:
                direct_answer_update_guard["stabilized_steps"] += 1
                direct_answer_update_guard["stabilized_attempts"] += 1
            scale_key = f"{learning_rate_scale:g}"
            scale_counts = direct_answer_update_guard[
                "accepted_learning_rate_scale_counts"
            ]
            if isinstance(scale_counts, dict):
                scale_counts[scale_key] = int(scale_counts.get(scale_key, 0)) + 1
            shape_counts = direct_answer_update_guard["accepted_update_shape_counts"]
            if isinstance(shape_counts, dict):
                shape_counts[update_shape] = int(shape_counts.get(update_shape, 0)) + 1

        def train_baseline_floor_stabilization_update(
            update_learning_rate: float,
            direct_step: int,
        ) -> tuple[float, bool]:
            if not direct_baseline_floor_repair_anchors:
                return 0.0, False
            if (
                direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active
            ):
                profile_anchor_pool = direct_baseline_floor_repair_anchors
                if (
                    direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
                ):
                    profile_anchor_pool = (
                        direct_baseline_floor_repair_anchors
                        + direct_baseline_floor_frontier_anchors
                    )
                profile_groups = baseline_floor_anchor_profile_groups(
                    profile_anchor_pool
                )
                frontier_targets_by_profile: dict[str, set[int]] = {}
                if (
                    direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
                ):
                    for branch in direct_baseline_floor_frontier_anchors:
                        _context, target, _predicted, frontier_profile = (
                            branch_replay_parts(branch)
                        )
                        frontier_targets_by_profile.setdefault(
                            frontier_profile,
                            set(),
                        ).add(target)
                if (
                    direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active
                ):
                    if (
                        direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active
                    ):
                        if (
                            direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
                        ):
                            if (
                                direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                            ):
                                update_shape = (
                                    "profile_scale_owner_paraphrase_binding_frontier_calibrated_sequential_profile_stabilization"
                                )
                            else:
                                update_shape = (
                                    "profile_scale_remaining_profile_binding_frontier_calibrated_sequential_profile_stabilization"
                                )
                        else:
                            update_shape = (
                                "profile_scale_collapsed_profile_binding_frontier_calibrated_sequential_profile_stabilization"
                            )
                    else:
                        update_shape = (
                            "profile_scale_branch_diversity_recovery_frontier_calibrated_sequential_profile_stabilization"
                        )
                elif (
                    direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
                ):
                    update_shape = (
                        "profile_scale_branch_stable_coverage_recovery_frontier_diversity_calibrated_sequential_profile_stabilization"
                    )
                elif (
                    direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active
                ):
                    update_shape = (
                        "profile_scale_coverage_recovery_frontier_diversity_calibrated_sequential_profile_stabilization"
                    )
                elif (
                    direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
                ):
                    update_shape = (
                        "profile_scale_coverage_prep_frontier_diversity_calibrated_sequential_profile_stabilization"
                    )
                elif (
                    direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
                ):
                    update_shape = (
                        "profile_scale_coverage_frontier_diversity_calibrated_sequential_profile_stabilization"
                    )
                elif (
                    direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
                ):
                    update_shape = (
                        "profile_scale_frontier_diversity_calibrated_sequential_profile_stabilization"
                    )
                elif (
                    direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
                ):
                    update_shape = (
                        "profile_scale_diversity_calibrated_sequential_profile_stabilization"
                    )
                else:
                    update_shape = (
                        "profile_scale_calibrated_sequential_profile_stabilization"
                    )
                total_loss = 0.0
                loss_count = 0
                accepted_any = False
                remaining_binding_target_profiles = list(
                    direct_remaining_profile_binding_target_profiles
                )
                if (
                    direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
                ):
                    profile_items = remaining_profile_binding_profile_order(
                        profile_groups,
                        remaining_binding_target_profiles,
                    )
                    remaining_source_labels = set(
                        remaining_profile_binding_source_labels(
                            remaining_binding_target_profiles
                        )
                    )
                    remaining_source_profiles = [
                        profile
                        for profile, anchors in profile_items
                        if source_profile_label(profile) in remaining_source_labels
                        and len(
                            {
                                target
                                for _context, target, _predicted, _profile in (
                                    branch_replay_parts(anchor)
                                    for anchor in anchors
                                )
                            }
                        )
                        > 1
                    ]
                    direct_answer_update_guard[
                        "profile_scale_remaining_profile_binding_source_profiles"
                    ] = remaining_source_profiles
                    if (
                        direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                    ):
                        direct_answer_update_guard[
                            "profile_scale_owner_paraphrase_binding_source_profiles"
                        ] = remaining_source_profiles
                else:
                    profile_items = list(profile_groups.items())
                    remaining_source_labels = set()
                    remaining_source_profiles = []
                for profile, profile_anchors in profile_items:
                    remaining_profile_binding_prioritized = (
                        direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
                        and profile in remaining_source_profiles
                    )
                    owner_paraphrase_binding_prioritized = (
                        direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                        and profile in remaining_source_profiles
                    )
                    profile_model_payload = model.to_dict(tokenizer)
                    profile_optimizer_payload = optimizer.to_dict()
                    profile_rng_state = direct_rng.getstate()
                    profile_accepted = False
                    profile_base_score: tuple[float, ...] | None = None
                    profile_base_snapshot: dict[str, Any] | None = None
                    if (
                        direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
                    ):
                        profile_base_snapshot = direct_snapshot_record(
                            direct_step,
                            None,
                            {
                                "baseline_floor_update_guard_probe": True,
                                "baseline_floor_profile_scale_diversity_base_probe": True,
                                "update_shape": update_shape,
                                "sequential_profile": profile,
                            },
                        )
                        profile_base_score = branch_diversity_snapshot_score(
                            profile_base_snapshot
                        )
                    for profile_scale in direct_baseline_floor_learning_rate_scales:
                        restore_direct_update_state(
                            profile_model_payload,
                            profile_optimizer_payload,
                        )
                        direct_rng.setstate(profile_rng_state)
                        profile_batch = baseline_floor_objective_anchor_batch(
                            profile_anchors,
                            direct_rng,
                            len(profile_anchors),
                        )
                        profile_frontier_records = 0
                        if (
                            direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
                        ):
                            frontier_targets = frontier_targets_by_profile.get(
                                profile,
                                set(),
                            )
                            profile_frontier_records = sum(
                                1
                                for branch in profile_batch
                                if branch_replay_parts(branch)[1]
                                in frontier_targets
                            )
                        direct_answer_update_guard[
                            "sequential_profile_attempts"
                        ] += 1
                        direct_answer_update_guard[
                            "profile_scale_memory_attempts"
                        ] += 1
                        if (
                            direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
                        ):
                            direct_answer_update_guard[
                                "profile_scale_frontier_attempts"
                            ] += 1
                            direct_answer_update_guard[
                                "profile_scale_frontier_records"
                            ] += profile_frontier_records
                        if (
                            direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
                        ):
                            direct_answer_update_guard[
                                "profile_scale_coverage_frontier_attempts"
                            ] += 1
                        if (
                            direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
                        ):
                            direct_answer_update_guard[
                                "profile_scale_coverage_prep_frontier_attempts"
                            ] += 1
                        if (
                            direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
                        ):
                            direct_answer_update_guard[
                                "profile_scale_diversity_attempts"
                            ] += 1
                        if remaining_profile_binding_prioritized:
                            direct_answer_update_guard[
                                "profile_scale_remaining_profile_binding_prioritized_attempts"
                            ] += 1
                        if owner_paraphrase_binding_prioritized:
                            direct_answer_update_guard[
                                "profile_scale_owner_paraphrase_binding_prioritized_attempts"
                            ] += 1
                        direct_answer_update_guard[
                            "sequential_profile_records"
                        ] += len(profile_batch)
                        direct_answer_update_guard[
                            "stabilization_anchor_batches"
                        ] += 1
                        direct_answer_update_guard[
                            "stabilization_anchor_records"
                        ] += len(profile_batch)
                        profile_loss = train_direct_answer_baseline_floor_anchor_batch(
                            model,
                            profile_batch,
                            args.direct_answer_learning_rate * profile_scale,
                            params=direct_params,
                        )
                        total_loss += profile_loss
                        loss_count += 1
                        profile_probe_snapshot = direct_snapshot_record(
                            direct_step,
                            None,
                            {
                                "baseline_floor_update_guard_probe": True,
                                "baseline_floor_sequential_profile_probe": True,
                                "baseline_floor_calibrated_sequential_profile_probe": True,
                                "baseline_floor_profile_scale_memory_probe": True,
                                "baseline_floor_profile_scale_frontier_probe": (
                                    direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
                                ),
                                "learning_rate_scale": profile_scale,
                                "update_shape": update_shape,
                                "sequential_profile": profile,
                                "sequential_profile_records": len(profile_batch),
                                "sequential_profile_frontier_records": (
                                    profile_frontier_records
                                ),
                                "remaining_profile_binding_prioritized": (
                                    remaining_profile_binding_prioritized
                                ),
                                "owner_paraphrase_binding_prioritized": (
                                    owner_paraphrase_binding_prioritized
                                ),
                            },
                        )
                        probe_sample = direct_answer_update_guard[
                            "sequential_profile_probe_sample"
                        ]
                        profile_scale_sample = direct_answer_update_guard[
                            "profile_scale_probe_sample"
                        ]
                        diversity_sample = direct_answer_update_guard[
                            "profile_scale_diversity_probe_sample"
                        ]
                        frontier_sample = direct_answer_update_guard[
                            "profile_scale_frontier_probe_sample"
                        ]
                        scale_key = f"{profile_scale:g}"
                        floor_preserved = branch_diversity_snapshot_preserves_target_coverage(
                            profile_probe_snapshot,
                            direct_baseline,
                        )
                        diversity_outcome = "not_active"
                        diversity_rejection_reason = "floor_regression"
                        profile_score: tuple[float, ...] | None = None
                        coverage_outcome = "not_active"
                        coverage_rejection_reason = "floor_regression"
                        coverage_delta: dict[str, Any] | None = None
                        if (
                            direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
                        ):
                            profile_score = branch_diversity_snapshot_score(
                                profile_probe_snapshot
                            )
                            if floor_preserved and profile_base_score is not None:
                                if profile_score > profile_base_score:
                                    diversity_outcome = "improved"
                                    diversity_rejection_reason = ""
                                elif profile_score == profile_base_score:
                                    diversity_outcome = "tied"
                                    diversity_rejection_reason = ""
                                else:
                                    diversity_outcome = "regressed"
                                    diversity_rejection_reason = "score_regression"
                            else:
                                diversity_outcome = "floor_regressed"
                        if (
                            direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
                            and profile_base_snapshot is not None
                        ):
                            coverage_delta = (
                                branch_diversity_snapshot_target_coverage_delta(
                                    profile_probe_snapshot,
                                    profile_base_snapshot,
                                )
                            )
                            if floor_preserved:
                                if (
                                    int(
                                        coverage_delta[
                                            "regressed_profile_count"
                                        ]
                                    )
                                    > 0
                                ):
                                    coverage_outcome = "regressed"
                                    coverage_rejection_reason = (
                                        "coverage_regression"
                                    )
                                elif (
                                    int(
                                        coverage_delta[
                                            "improved_profile_count"
                                        ]
                                    )
                                    > 0
                                ):
                                    coverage_outcome = "gained"
                                    coverage_rejection_reason = ""
                                else:
                                    coverage_outcome = "tied"
                                    coverage_rejection_reason = "coverage_tie"
                            else:
                                coverage_outcome = "floor_regressed"
                        coverage_prep_accepted = (
                            direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
                            and coverage_outcome == "tied"
                            and diversity_outcome == "improved"
                        )
                        coverage_recovery_attempted = False
                        coverage_recovery_accepted = False
                        coverage_recovery_outcome = "not_attempted"
                        coverage_recovery_rejection_reason = ""
                        coverage_recovery_learning_rate_scale: float | None = None
                        coverage_recovery_records = 0
                        coverage_recovery_delta: dict[str, Any] | None = None
                        coverage_recovery_prepared_score = profile_score
                        coverage_recovery_score: tuple[float, ...] | None = None
                        coverage_recovery_branch_stable_checked = False
                        coverage_recovery_branch_stable_accepted = False
                        coverage_recovery_branch_stability_preserved: bool | None = None
                        branch_diversity_recovery_attempted = False
                        branch_diversity_recovery_accepted = False
                        branch_diversity_recovery_outcome = "not_attempted"
                        branch_diversity_recovery_rejection_reason = ""
                        branch_diversity_recovery_learning_rate_scale: float | None = (
                            None
                        )
                        branch_diversity_recovery_records = 0
                        branch_diversity_recovery_base_score: tuple[float, ...] | None = (
                            None
                        )
                        branch_diversity_recovery_score: tuple[float, ...] | None = None
                        branch_diversity_recovery_delta: dict[str, Any] | None = None
                        collapsed_profile_binding_attempted = False
                        collapsed_profile_binding_accepted = False
                        collapsed_profile_binding_outcome = "not_attempted"
                        collapsed_profile_binding_rejection_reason = ""
                        collapsed_profile_binding_learning_rate_scale: float | None = (
                            None
                        )
                        collapsed_profile_binding_records = 0
                        collapsed_profile_binding_target_profiles: list[str] = []
                        collapsed_profile_binding_base_score: tuple[float, ...] | None = (
                            None
                        )
                        collapsed_profile_binding_score: tuple[float, ...] | None = None
                        collapsed_profile_binding_delta: dict[str, Any] | None = None
                        owner_paraphrase_binding_preserved = True
                        owner_paraphrase_binding_preservation_delta: dict[
                            str, Any
                        ] | None = None
                        if (
                            direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active
                            and coverage_prep_accepted
                            and profile_base_snapshot is not None
                        ):
                            direct_answer_update_guard[
                                "profile_scale_coverage_recovery_frontier_prepared_candidates"
                            ] += 1
                            prep_model_payload = model.to_dict(tokenizer)
                            prep_optimizer_payload = optimizer.to_dict()
                            recovery_frontier_targets = frontier_targets_by_profile.get(
                                profile,
                                set(),
                            )
                            recovery_batch = [
                                branch
                                for branch in profile_batch
                                if branch_replay_parts(branch)[1]
                                in recovery_frontier_targets
                            ]
                            if not recovery_batch:
                                recovery_batch = profile_batch
                            coverage_recovery_records = len(recovery_batch)
                            for (
                                recovery_learning_rate_scale
                            ) in BASELINE_FLOOR_COVERAGE_RECOVERY_LEARNING_RATE_SCALES:
                                restore_direct_update_state(
                                    prep_model_payload,
                                    prep_optimizer_payload,
                                )
                                coverage_recovery_attempted = True
                                direct_answer_update_guard[
                                    "profile_scale_coverage_recovery_frontier_attempts"
                                ] += 1
                                direct_answer_update_guard[
                                    "profile_scale_coverage_recovery_frontier_records"
                                ] += coverage_recovery_records
                                recovery_loss = train_direct_answer_baseline_floor_anchor_batch(
                                    model,
                                    recovery_batch,
                                    (
                                        args.direct_answer_learning_rate
                                        * profile_scale
                                        * recovery_learning_rate_scale
                                    ),
                                    params=direct_params,
                                )
                                total_loss += recovery_loss
                                loss_count += 1
                                recovery_probe_snapshot = direct_snapshot_record(
                                    direct_step,
                                    None,
                                    {
                                        "baseline_floor_update_guard_probe": True,
                                        "baseline_floor_sequential_profile_probe": True,
                                        "baseline_floor_calibrated_sequential_profile_probe": True,
                                        "baseline_floor_profile_scale_memory_probe": True,
                                        "baseline_floor_profile_scale_frontier_probe": True,
                                        "baseline_floor_profile_scale_coverage_recovery_probe": True,
                                        "learning_rate_scale": profile_scale,
                                        "coverage_recovery_learning_rate_scale": recovery_learning_rate_scale,
                                        "update_shape": update_shape,
                                        "sequential_profile": profile,
                                        "sequential_profile_records": len(
                                            profile_batch
                                        ),
                                        "sequential_profile_frontier_records": (
                                            profile_frontier_records
                                        ),
                                        "coverage_recovery_records": (
                                            coverage_recovery_records
                                        ),
                                    },
                                )
                                recovery_floor_preserved = branch_diversity_snapshot_preserves_target_coverage(
                                    recovery_probe_snapshot,
                                    direct_baseline,
                                )
                                recovery_score = branch_diversity_snapshot_score(
                                    recovery_probe_snapshot
                                )
                                coverage_recovery_score = recovery_score
                                if (
                                    direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
                                ):
                                    coverage_recovery_branch_stable_checked = True
                                    direct_answer_update_guard[
                                        "profile_scale_branch_stable_coverage_recovery_frontier_checks"
                                    ] += 1
                                recovery_delta = branch_diversity_snapshot_target_coverage_delta(
                                    recovery_probe_snapshot,
                                    profile_base_snapshot,
                                )
                                coverage_recovery_delta = recovery_delta
                                if not recovery_floor_preserved:
                                    coverage_recovery_outcome = "floor_regressed"
                                    coverage_recovery_rejection_reason = (
                                        "floor_regression"
                                    )
                                elif (
                                    profile_base_score is not None
                                    and recovery_score < profile_base_score
                                ):
                                    coverage_recovery_outcome = "score_regressed"
                                    coverage_recovery_rejection_reason = (
                                        "score_regression"
                                    )
                                elif (
                                    int(
                                        recovery_delta["regressed_profile_count"]
                                    )
                                    > 0
                                ):
                                    coverage_recovery_outcome = "coverage_regressed"
                                    coverage_recovery_rejection_reason = (
                                        "coverage_regression"
                                    )
                                elif (
                                    direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
                                    and coverage_recovery_prepared_score is not None
                                    and recovery_score
                                    < coverage_recovery_prepared_score
                                ):
                                    coverage_recovery_outcome = (
                                        "branch_score_regressed"
                                    )
                                    coverage_recovery_rejection_reason = (
                                        "branch_score_regression"
                                    )
                                    coverage_recovery_branch_stability_preserved = (
                                        False
                                    )
                                elif (
                                    int(
                                        recovery_delta["improved_profile_count"]
                                    )
                                    > 0
                                ):
                                    if (
                                        direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
                                    ):
                                        coverage_recovery_branch_stability_preserved = (
                                            True
                                        )
                                        coverage_recovery_branch_stable_accepted = (
                                            True
                                        )
                                    coverage_recovery_outcome = "gained"
                                    coverage_recovery_rejection_reason = ""
                                    coverage_recovery_accepted = True
                                    coverage_recovery_learning_rate_scale = (
                                        recovery_learning_rate_scale
                                    )
                                    floor_preserved = recovery_floor_preserved
                                    profile_probe_snapshot = recovery_probe_snapshot
                                    profile_score = recovery_score
                                    if profile_base_score is not None:
                                        if recovery_score > profile_base_score:
                                            diversity_outcome = "improved"
                                        else:
                                            diversity_outcome = "tied"
                                    diversity_rejection_reason = ""
                                    coverage_delta = recovery_delta
                                    coverage_outcome = "gained"
                                    coverage_rejection_reason = ""
                                    coverage_prep_accepted = False
                                    direct_answer_update_guard[
                                        "profile_scale_coverage_recovery_frontier_acceptances"
                                    ] += 1
                                    if (
                                        direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
                                    ):
                                        direct_answer_update_guard[
                                            "profile_scale_branch_stable_coverage_recovery_frontier_acceptances"
                                        ] += 1
                                    break
                                else:
                                    coverage_recovery_outcome = "coverage_tied"
                                    coverage_recovery_rejection_reason = (
                                        "coverage_tie"
                                    )
                                direct_answer_update_guard[
                                    "profile_scale_coverage_recovery_frontier_rejections"
                                ] += 1
                                recovery_reasons = direct_answer_update_guard[
                                    "profile_scale_coverage_recovery_frontier_rejection_reasons"
                                ]
                                if isinstance(recovery_reasons, dict):
                                    recovery_reasons[
                                        coverage_recovery_rejection_reason
                                    ] = (
                                        int(
                                            recovery_reasons.get(
                                                coverage_recovery_rejection_reason,
                                                0,
                                            )
                                        )
                                        + 1
                                    )
                                if (
                                    direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
                                ):
                                    direct_answer_update_guard[
                                        "profile_scale_branch_stable_coverage_recovery_frontier_rejections"
                                    ] += 1
                                    branch_stable_reasons = direct_answer_update_guard[
                                        "profile_scale_branch_stable_coverage_recovery_frontier_rejection_reasons"
                                    ]
                                    if isinstance(branch_stable_reasons, dict):
                                        branch_stable_reasons[
                                            coverage_recovery_rejection_reason
                                        ] = (
                                            int(
                                                branch_stable_reasons.get(
                                                    coverage_recovery_rejection_reason,
                                                    0,
                                                )
                                            )
                                            + 1
                                        )
                            if not coverage_recovery_accepted:
                                restore_direct_update_state(
                                    prep_model_payload,
                                    prep_optimizer_payload,
                                )
                        diversity_accepted = (
                            not direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
                            or diversity_outcome in {"improved", "tied"}
                        )
                        if (
                            direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                            and profile_base_snapshot is not None
                        ):
                            direct_answer_update_guard[
                                "profile_scale_owner_paraphrase_binding_preservation_checks"
                            ] += 1
                            owner_paraphrase_binding_preservation_delta = (
                                branch_diversity_snapshot_profile_diversity_delta(
                                    profile_probe_snapshot,
                                    profile_base_snapshot,
                                    BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES,
                                )
                            )
                            if (
                                int(
                                    owner_paraphrase_binding_preservation_delta[
                                        "regressed_profile_count"
                                    ]
                                )
                                > 0
                            ):
                                owner_paraphrase_binding_preserved = False
                                direct_answer_update_guard[
                                    "profile_scale_owner_paraphrase_binding_preservation_failures"
                                ] += 1
                                diversity_accepted = False
                                diversity_rejection_reason = (
                                    "owner_paraphrase_preservation_regression"
                                )
                        if (
                            direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
                            and floor_preserved
                            and coverage_outcome == "tied"
                            and not coverage_prep_accepted
                        ):
                            coverage_rejection_reason = (
                                "coverage_tie_without_score_gain"
                            )
                        coverage_accepted = (
                            not direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
                            or coverage_outcome == "gained"
                            or coverage_prep_accepted
                        )
                        if (
                            direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active
                            and floor_preserved
                            and diversity_accepted
                            and coverage_accepted
                            and profile_score is not None
                        ):
                            direct_answer_update_guard[
                                "profile_scale_branch_diversity_recovery_frontier_candidates"
                            ] += 1
                            branch_diversity_recovery_base_score = profile_score
                            branch_diversity_candidate_model_payload = model.to_dict(
                                tokenizer
                            )
                            branch_diversity_candidate_optimizer_payload = (
                                optimizer.to_dict()
                            )
                            branch_diversity_recovery_records = len(profile_batch)
                            for (
                                branch_diversity_learning_rate_scale
                            ) in BASELINE_FLOOR_BRANCH_DIVERSITY_RECOVERY_LEARNING_RATE_SCALES:
                                restore_direct_update_state(
                                    branch_diversity_candidate_model_payload,
                                    branch_diversity_candidate_optimizer_payload,
                                )
                                branch_diversity_recovery_attempted = True
                                direct_answer_update_guard[
                                    "profile_scale_branch_diversity_recovery_frontier_attempts"
                                ] += 1
                                direct_answer_update_guard[
                                    "profile_scale_branch_diversity_recovery_frontier_records"
                                ] += branch_diversity_recovery_records
                                branch_diversity_loss = train_direct_answer_baseline_floor_anchor_branch_diversity(
                                    model,
                                    profile_batch,
                                    (
                                        args.direct_answer_learning_rate
                                        * profile_scale
                                        * branch_diversity_learning_rate_scale
                                    ),
                                    args.direct_answer_negative_weight,
                                    args.direct_answer_positive_weight,
                                    args.direct_answer_contrast_weight,
                                    params=direct_params,
                                )
                                total_loss += branch_diversity_loss
                                loss_count += 1
                                branch_diversity_probe_snapshot = direct_snapshot_record(
                                    direct_step,
                                    None,
                                    {
                                        "baseline_floor_update_guard_probe": True,
                                        "baseline_floor_sequential_profile_probe": True,
                                        "baseline_floor_calibrated_sequential_profile_probe": True,
                                        "baseline_floor_profile_scale_memory_probe": True,
                                        "baseline_floor_profile_scale_frontier_probe": True,
                                        "baseline_floor_profile_scale_branch_diversity_recovery_probe": True,
                                        "learning_rate_scale": profile_scale,
                                        "branch_diversity_recovery_learning_rate_scale": (
                                            branch_diversity_learning_rate_scale
                                        ),
                                        "update_shape": update_shape,
                                        "sequential_profile": profile,
                                        "sequential_profile_records": len(
                                            profile_batch
                                        ),
                                        "sequential_profile_frontier_records": (
                                            profile_frontier_records
                                        ),
                                        "branch_diversity_recovery_records": (
                                            branch_diversity_recovery_records
                                        ),
                                    },
                                )
                                branch_diversity_floor_preserved = branch_diversity_snapshot_preserves_target_coverage(
                                    branch_diversity_probe_snapshot,
                                    direct_baseline,
                                )
                                branch_diversity_score = (
                                    branch_diversity_snapshot_score(
                                        branch_diversity_probe_snapshot
                                    )
                                )
                                branch_diversity_recovery_score = (
                                    branch_diversity_score
                                )
                                branch_diversity_delta = branch_diversity_snapshot_target_coverage_delta(
                                    branch_diversity_probe_snapshot,
                                    profile_probe_snapshot,
                                )
                                branch_diversity_recovery_delta = (
                                    branch_diversity_delta
                                )
                                if not branch_diversity_floor_preserved:
                                    branch_diversity_recovery_outcome = (
                                        "floor_regressed"
                                    )
                                    branch_diversity_recovery_rejection_reason = (
                                        "floor_regression"
                                    )
                                elif (
                                    int(
                                        branch_diversity_delta[
                                            "regressed_profile_count"
                                        ]
                                    )
                                    > 0
                                ):
                                    branch_diversity_recovery_outcome = (
                                        "coverage_regressed"
                                    )
                                    branch_diversity_recovery_rejection_reason = (
                                        "coverage_regression"
                                    )
                                elif (
                                    branch_diversity_score
                                    > branch_diversity_recovery_base_score
                                ):
                                    branch_diversity_recovery_outcome = (
                                        "branch_diversity_improved"
                                    )
                                    branch_diversity_recovery_rejection_reason = ""
                                    branch_diversity_recovery_accepted = True
                                    branch_diversity_recovery_learning_rate_scale = (
                                        branch_diversity_learning_rate_scale
                                    )
                                    floor_preserved = (
                                        branch_diversity_floor_preserved
                                    )
                                    profile_probe_snapshot = (
                                        branch_diversity_probe_snapshot
                                    )
                                    profile_score = branch_diversity_score
                                    diversity_outcome = "improved"
                                    diversity_rejection_reason = ""
                                    direct_answer_update_guard[
                                        "profile_scale_branch_diversity_recovery_frontier_acceptances"
                                    ] += 1
                                    break
                                elif (
                                    branch_diversity_score
                                    == branch_diversity_recovery_base_score
                                ):
                                    branch_diversity_recovery_outcome = (
                                        "score_tied"
                                    )
                                    branch_diversity_recovery_rejection_reason = (
                                        "score_tie"
                                    )
                                else:
                                    branch_diversity_recovery_outcome = (
                                        "score_regressed"
                                    )
                                    branch_diversity_recovery_rejection_reason = (
                                        "score_regression"
                                    )
                                direct_answer_update_guard[
                                    "profile_scale_branch_diversity_recovery_frontier_rejections"
                                ] += 1
                                branch_diversity_reasons = direct_answer_update_guard[
                                    "profile_scale_branch_diversity_recovery_frontier_rejection_reasons"
                                ]
                                if isinstance(branch_diversity_reasons, dict):
                                    branch_diversity_reasons[
                                        branch_diversity_recovery_rejection_reason
                                    ] = (
                                        int(
                                            branch_diversity_reasons.get(
                                                branch_diversity_recovery_rejection_reason,
                                                0,
                                            )
                                        )
                                        + 1
                                    )
                            if not branch_diversity_recovery_accepted:
                                restore_direct_update_state(
                                    branch_diversity_candidate_model_payload,
                                    branch_diversity_candidate_optimizer_payload,
                                )
                                direct_answer_update_guard[
                                    "profile_scale_branch_diversity_recovery_frontier_fallback_acceptances"
                                ] += 1
                        if (
                            direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active
                            and floor_preserved
                            and diversity_accepted
                            and coverage_accepted
                            and profile_score is not None
                            and profile_probe_snapshot is not None
                        ):
                            collapsed_profile_binding_target_profiles = (
                                branch_diversity_snapshot_collapsed_profile_names(
                                    profile_probe_snapshot
                                )
                            )
                            if (
                                direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                            ):
                                owner_paraphrase_targets = set(
                                    BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES
                                )
                                collapsed_profile_binding_target_profiles = [
                                    name
                                    for name in collapsed_profile_binding_target_profiles
                                    if name in owner_paraphrase_targets
                                ]
                            if collapsed_profile_binding_target_profiles:
                                direct_answer_update_guard[
                                    "profile_scale_collapsed_profile_binding_frontier_candidates"
                                ] += 1
                                collapsed_profile_binding_base_score = profile_score
                                collapsed_binding_candidate_model_payload = (
                                    model.to_dict(tokenizer)
                                )
                                collapsed_binding_candidate_optimizer_payload = (
                                    optimizer.to_dict()
                                )
                                collapsed_profile_binding_records = len(profile_batch)
                                for (
                                    collapsed_binding_learning_rate_scale
                                ) in BASELINE_FLOOR_COLLAPSED_PROFILE_BINDING_LEARNING_RATE_SCALES:
                                    restore_direct_update_state(
                                        collapsed_binding_candidate_model_payload,
                                        collapsed_binding_candidate_optimizer_payload,
                                    )
                                    collapsed_profile_binding_attempted = True
                                    direct_answer_update_guard[
                                        "profile_scale_collapsed_profile_binding_frontier_attempts"
                                    ] += 1
                                    direct_answer_update_guard[
                                        "profile_scale_collapsed_profile_binding_frontier_records"
                                    ] += collapsed_profile_binding_records
                                    collapsed_binding_loss = train_direct_answer_baseline_floor_anchor_branch_diversity(
                                        model,
                                        profile_batch,
                                        (
                                            args.direct_answer_learning_rate
                                            * profile_scale
                                            * collapsed_binding_learning_rate_scale
                                        ),
                                        args.direct_answer_negative_weight,
                                        args.direct_answer_positive_weight,
                                        args.direct_answer_contrast_weight,
                                        params=direct_params,
                                    )
                                    total_loss += collapsed_binding_loss
                                    loss_count += 1
                                    collapsed_binding_probe_snapshot = direct_snapshot_record(
                                        direct_step,
                                        None,
                                        {
                                            "baseline_floor_update_guard_probe": True,
                                            "baseline_floor_sequential_profile_probe": True,
                                            "baseline_floor_calibrated_sequential_profile_probe": True,
                                            "baseline_floor_profile_scale_memory_probe": True,
                                            "baseline_floor_profile_scale_frontier_probe": True,
                                            "baseline_floor_profile_scale_collapsed_profile_binding_probe": True,
                                            "learning_rate_scale": profile_scale,
                                            "collapsed_profile_binding_learning_rate_scale": (
                                                collapsed_binding_learning_rate_scale
                                            ),
                                            "update_shape": update_shape,
                                            "sequential_profile": profile,
                                            "sequential_profile_records": len(
                                                profile_batch
                                            ),
                                            "sequential_profile_frontier_records": (
                                                profile_frontier_records
                                            ),
                                            "collapsed_profile_binding_records": (
                                                collapsed_profile_binding_records
                                            ),
                                            "collapsed_profile_binding_target_profiles": (
                                                collapsed_profile_binding_target_profiles
                                            ),
                                            "owner_paraphrase_binding_preserved_profiles": (
                                                list(
                                                    BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES
                                                )
                                                if direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                                                else []
                                            ),
                                        },
                                    )
                                    collapsed_binding_floor_preserved = branch_diversity_snapshot_preserves_target_coverage(
                                        collapsed_binding_probe_snapshot,
                                        direct_baseline,
                                    )
                                    collapsed_binding_score = branch_diversity_snapshot_score(
                                        collapsed_binding_probe_snapshot
                                    )
                                    collapsed_profile_binding_score = (
                                        collapsed_binding_score
                                    )
                                    collapsed_binding_coverage_delta = branch_diversity_snapshot_target_coverage_delta(
                                        collapsed_binding_probe_snapshot,
                                        profile_probe_snapshot,
                                    )
                                    collapsed_binding_profile_delta = branch_diversity_snapshot_profile_diversity_delta(
                                        collapsed_binding_probe_snapshot,
                                        profile_probe_snapshot,
                                        collapsed_profile_binding_target_profiles,
                                    )
                                    owner_paraphrase_collapsed_preservation_regressed = (
                                        False
                                    )
                                    if (
                                        direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                                    ):
                                        direct_answer_update_guard[
                                            "profile_scale_owner_paraphrase_binding_preservation_checks"
                                        ] += 1
                                        owner_paraphrase_binding_preservation_delta = branch_diversity_snapshot_profile_diversity_delta(
                                            collapsed_binding_probe_snapshot,
                                            profile_probe_snapshot,
                                            BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES,
                                        )
                                        owner_paraphrase_collapsed_preservation_regressed = (
                                            int(
                                                owner_paraphrase_binding_preservation_delta[
                                                    "regressed_profile_count"
                                                ]
                                            )
                                            > 0
                                        )
                                    collapsed_profile_binding_delta = {
                                        "coverage_delta": collapsed_binding_coverage_delta,
                                        "profile_delta": collapsed_binding_profile_delta,
                                    }
                                    if (
                                        owner_paraphrase_binding_preservation_delta
                                        is not None
                                    ):
                                        collapsed_profile_binding_delta[
                                            "owner_paraphrase_preservation_delta"
                                        ] = (
                                            owner_paraphrase_binding_preservation_delta
                                        )
                                    if not collapsed_binding_floor_preserved:
                                        collapsed_profile_binding_outcome = (
                                            "floor_regressed"
                                        )
                                        collapsed_profile_binding_rejection_reason = (
                                            "floor_regression"
                                        )
                                    elif (
                                        int(
                                            collapsed_binding_coverage_delta[
                                                "regressed_profile_count"
                                            ]
                                        )
                                        > 0
                                    ):
                                        collapsed_profile_binding_outcome = (
                                            "coverage_regressed"
                                        )
                                        collapsed_profile_binding_rejection_reason = (
                                            "coverage_regression"
                                        )
                                    elif (
                                        int(
                                            collapsed_binding_profile_delta[
                                                "regressed_profile_count"
                                            ]
                                        )
                                        > 0
                                    ):
                                        collapsed_profile_binding_outcome = (
                                            "profile_diversity_regressed"
                                        )
                                        collapsed_profile_binding_rejection_reason = (
                                            "profile_diversity_regression"
                                        )
                                    elif owner_paraphrase_collapsed_preservation_regressed:
                                        collapsed_profile_binding_outcome = (
                                            "preserved_profile_regressed"
                                        )
                                        collapsed_profile_binding_rejection_reason = (
                                            "owner_paraphrase_preservation_regression"
                                        )
                                        direct_answer_update_guard[
                                            "profile_scale_owner_paraphrase_binding_preservation_failures"
                                        ] += 1
                                    elif (
                                        collapsed_profile_binding_base_score
                                        is not None
                                        and collapsed_binding_score
                                        < collapsed_profile_binding_base_score
                                    ):
                                        collapsed_profile_binding_outcome = (
                                            "score_regressed"
                                        )
                                        collapsed_profile_binding_rejection_reason = (
                                            "score_regression"
                                        )
                                    elif (
                                        int(
                                            collapsed_binding_profile_delta[
                                                "improved_profile_count"
                                            ]
                                        )
                                        > 0
                                    ):
                                        collapsed_profile_binding_outcome = (
                                            "collapsed_profile_improved"
                                        )
                                        collapsed_profile_binding_rejection_reason = ""
                                        collapsed_profile_binding_accepted = True
                                        collapsed_profile_binding_learning_rate_scale = (
                                            collapsed_binding_learning_rate_scale
                                        )
                                        floor_preserved = (
                                            collapsed_binding_floor_preserved
                                        )
                                        profile_probe_snapshot = (
                                            collapsed_binding_probe_snapshot
                                        )
                                        profile_score = collapsed_binding_score
                                        if (
                                            collapsed_profile_binding_base_score
                                            is not None
                                            and collapsed_binding_score
                                            > collapsed_profile_binding_base_score
                                        ):
                                            diversity_outcome = "improved"
                                        diversity_rejection_reason = ""
                                        direct_answer_update_guard[
                                            "profile_scale_collapsed_profile_binding_frontier_acceptances"
                                        ] += 1
                                        break
                                    else:
                                        collapsed_profile_binding_outcome = (
                                            "collapsed_profile_tied"
                                        )
                                        collapsed_profile_binding_rejection_reason = (
                                            "collapsed_profile_tie"
                                        )
                                    direct_answer_update_guard[
                                        "profile_scale_collapsed_profile_binding_frontier_rejections"
                                    ] += 1
                                    collapsed_binding_reasons = direct_answer_update_guard[
                                        "profile_scale_collapsed_profile_binding_frontier_rejection_reasons"
                                    ]
                                    if isinstance(collapsed_binding_reasons, dict):
                                        collapsed_binding_reasons[
                                            collapsed_profile_binding_rejection_reason
                                        ] = (
                                            int(
                                                collapsed_binding_reasons.get(
                                                    collapsed_profile_binding_rejection_reason,
                                                    0,
                                                )
                                            )
                                            + 1
                                        )
                                if not collapsed_profile_binding_accepted:
                                    restore_direct_update_state(
                                        collapsed_binding_candidate_model_payload,
                                        collapsed_binding_candidate_optimizer_payload,
                                    )
                                    direct_answer_update_guard[
                                        "profile_scale_collapsed_profile_binding_frontier_fallback_acceptances"
                                    ] += 1
                        if floor_preserved and diversity_accepted and coverage_accepted:
                            direct_answer_update_guard[
                                "sequential_profile_acceptances"
                            ] += 1
                            direct_answer_update_guard[
                                "profile_scale_memory_acceptances"
                            ] += 1
                            if remaining_profile_binding_prioritized:
                                direct_answer_update_guard[
                                    "profile_scale_remaining_profile_binding_prioritized_acceptances"
                                ] += 1
                            if owner_paraphrase_binding_prioritized:
                                direct_answer_update_guard[
                                    "profile_scale_owner_paraphrase_binding_prioritized_acceptances"
                                ] += 1
                            if (
                                direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
                            ):
                                direct_answer_update_guard[
                                    "profile_scale_diversity_acceptances"
                                ] += 1
                                if diversity_outcome == "improved":
                                    direct_answer_update_guard[
                                        "profile_scale_diversity_score_improvements"
                                    ] += 1
                                else:
                                    direct_answer_update_guard[
                                        "profile_scale_diversity_score_ties"
                                    ] += 1
                            if (
                                direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
                            ):
                                direct_answer_update_guard[
                                    "profile_scale_frontier_acceptances"
                                ] += 1
                            if (
                                direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
                            ):
                                direct_answer_update_guard[
                                    "profile_scale_coverage_frontier_acceptances"
                                ] += 1
                                if coverage_outcome == "gained":
                                    direct_answer_update_guard[
                                        "profile_scale_coverage_frontier_gains"
                                    ] += 1
                                elif coverage_outcome == "tied":
                                    direct_answer_update_guard[
                                        "profile_scale_coverage_frontier_ties"
                                    ] += 1
                            if (
                                direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
                            ):
                                direct_answer_update_guard[
                                    "profile_scale_coverage_prep_frontier_acceptances"
                                ] += 1
                                if coverage_outcome == "gained":
                                    direct_answer_update_guard[
                                        "profile_scale_coverage_prep_frontier_gain_acceptances"
                                    ] += 1
                                elif coverage_prep_accepted:
                                    direct_answer_update_guard[
                                        "profile_scale_coverage_prep_frontier_preparations"
                                    ] += 1
                            if (
                                direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active
                                and coverage_prep_accepted
                                and coverage_recovery_attempted
                                and not coverage_recovery_accepted
                            ):
                                direct_answer_update_guard[
                                    "profile_scale_coverage_recovery_frontier_fallback_preparations"
                                ] += 1
                            if (
                                direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
                                and coverage_prep_accepted
                                and coverage_recovery_attempted
                                and not coverage_recovery_accepted
                            ):
                                direct_answer_update_guard[
                                    "profile_scale_branch_stable_coverage_recovery_frontier_fallback_preparations"
                                ] += 1
                            accepted_counts = direct_answer_update_guard[
                                "sequential_profile_acceptance_counts"
                            ]
                            if isinstance(accepted_counts, dict):
                                accepted_counts[profile] = (
                                    int(accepted_counts.get(profile, 0)) + 1
                                )
                            scale_counts = direct_answer_update_guard[
                                "profile_scale_acceptance_scale_counts"
                            ]
                            if isinstance(scale_counts, dict):
                                scale_counts[scale_key] = (
                                    int(scale_counts.get(scale_key, 0)) + 1
                                )
                            profile_scales = direct_answer_update_guard[
                                "profile_scale_profile_acceptance_scales"
                            ]
                            if isinstance(profile_scales, dict):
                                profile_scales[profile] = scale_key
                            diversity_outcomes = direct_answer_update_guard[
                                "profile_scale_diversity_profile_acceptance_outcomes"
                            ]
                            if isinstance(diversity_outcomes, dict):
                                diversity_outcomes[profile] = diversity_outcome
                            coverage_deltas = direct_answer_update_guard[
                                "profile_scale_coverage_frontier_profile_acceptance_deltas"
                            ]
                            if (
                                isinstance(coverage_deltas, dict)
                                and coverage_delta is not None
                            ):
                                coverage_deltas[profile] = coverage_delta
                            prep_outcomes = direct_answer_update_guard[
                                "profile_scale_coverage_prep_frontier_profile_acceptance_outcomes"
                            ]
                            if isinstance(prep_outcomes, dict):
                                prep_outcomes[profile] = (
                                    "coverage_gain"
                                    if coverage_outcome == "gained"
                                    else "coverage_preparation"
                                )
                            recovery_outcomes = direct_answer_update_guard[
                                "profile_scale_coverage_recovery_frontier_profile_acceptance_outcomes"
                            ]
                            if isinstance(recovery_outcomes, dict):
                                if coverage_recovery_accepted:
                                    recovery_outcomes[profile] = "coverage_recovery"
                                elif coverage_recovery_attempted:
                                    recovery_outcomes[profile] = (
                                        "coverage_preparation_fallback"
                                    )
                                elif coverage_outcome == "gained":
                                    recovery_outcomes[profile] = "coverage_gain"
                            branch_stable_outcomes = direct_answer_update_guard[
                                "profile_scale_branch_stable_coverage_recovery_frontier_profile_acceptance_outcomes"
                            ]
                            if isinstance(branch_stable_outcomes, dict):
                                if coverage_recovery_branch_stable_accepted:
                                    branch_stable_outcomes[profile] = (
                                        "branch_stable_coverage_recovery"
                                    )
                                elif (
                                    direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
                                    and coverage_recovery_attempted
                                ):
                                    branch_stable_outcomes[profile] = (
                                        "branch_stable_preparation_fallback"
                                    )
                                elif (
                                    direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
                                    and coverage_outcome == "gained"
                                ):
                                    branch_stable_outcomes[profile] = "coverage_gain"
                            branch_diversity_outcomes = direct_answer_update_guard[
                                "profile_scale_branch_diversity_recovery_frontier_profile_acceptance_outcomes"
                            ]
                            if isinstance(branch_diversity_outcomes, dict):
                                if branch_diversity_recovery_accepted:
                                    branch_diversity_outcomes[profile] = (
                                        "branch_diversity_recovery"
                                    )
                                elif (
                                    direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active
                                    and branch_diversity_recovery_attempted
                                ):
                                    branch_diversity_outcomes[profile] = (
                                        "branch_diversity_fallback"
                                    )
                            branch_diversity_score_deltas = direct_answer_update_guard[
                                "profile_scale_branch_diversity_recovery_frontier_profile_score_deltas"
                            ]
                            if (
                                isinstance(branch_diversity_score_deltas, dict)
                                and branch_diversity_recovery_base_score is not None
                                and branch_diversity_recovery_score is not None
                            ):
                                branch_diversity_score_deltas[profile] = {
                                    "base_score": list(
                                        branch_diversity_recovery_base_score
                                    ),
                                    "final_score": list(
                                        branch_diversity_recovery_score
                                    ),
                                    "improved": branch_diversity_recovery_accepted,
                                    "outcome": branch_diversity_recovery_outcome,
                                }
                            collapsed_binding_outcomes = direct_answer_update_guard[
                                "profile_scale_collapsed_profile_binding_frontier_profile_acceptance_outcomes"
                            ]
                            if isinstance(collapsed_binding_outcomes, dict):
                                if collapsed_profile_binding_accepted:
                                    collapsed_binding_outcomes[profile] = (
                                        "collapsed_profile_binding"
                                    )
                                elif (
                                    direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active
                                    and collapsed_profile_binding_attempted
                                ):
                                    collapsed_binding_outcomes[profile] = (
                                        "collapsed_profile_binding_fallback"
                                    )
                                elif (
                                    direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active
                                    and not collapsed_profile_binding_target_profiles
                                ):
                                    collapsed_binding_outcomes[profile] = (
                                        "no_collapsed_profile_targets"
                                    )
                            collapsed_binding_deltas = direct_answer_update_guard[
                                "profile_scale_collapsed_profile_binding_frontier_profile_deltas"
                            ]
                            if (
                                isinstance(collapsed_binding_deltas, dict)
                                and collapsed_profile_binding_delta is not None
                            ):
                                collapsed_binding_deltas[profile] = {
                                    "target_profiles": (
                                        collapsed_profile_binding_target_profiles
                                    ),
                                    "base_score": (
                                        list(collapsed_profile_binding_base_score)
                                        if collapsed_profile_binding_base_score
                                        is not None
                                        else None
                                    ),
                                    "final_score": (
                                        list(collapsed_profile_binding_score)
                                        if collapsed_profile_binding_score is not None
                                        else None
                                    ),
                                    "accepted": collapsed_profile_binding_accepted,
                                    "outcome": collapsed_profile_binding_outcome,
                                    "delta": collapsed_profile_binding_delta,
                                }
                            accepted_any = True
                            profile_accepted = True
                            sample = {
                                "profile": profile,
                                "accepted": True,
                                "records": len(profile_batch),
                                "frontier_records": profile_frontier_records,
                                "learning_rate_scale": profile_scale,
                            }
                            if (
                                direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
                            ):
                                sample[
                                    "remaining_profile_binding_prioritized"
                                ] = remaining_profile_binding_prioritized
                                sample[
                                    "remaining_profile_binding_target_profiles"
                                ] = remaining_binding_target_profiles
                                sample[
                                    "remaining_profile_binding_source_profiles"
                                ] = remaining_source_profiles
                            if (
                                direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                            ):
                                sample[
                                    "owner_paraphrase_binding_prioritized"
                                ] = owner_paraphrase_binding_prioritized
                                sample[
                                    "owner_paraphrase_binding_target_profiles"
                                ] = list(
                                    BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES
                                )
                                sample[
                                    "owner_paraphrase_binding_preserved_profiles"
                                ] = list(
                                    BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_PRESERVED_PROFILES
                                )
                                sample[
                                    "owner_paraphrase_binding_preserved"
                                ] = owner_paraphrase_binding_preserved
                                if (
                                    owner_paraphrase_binding_preservation_delta
                                    is not None
                                ):
                                    sample[
                                        "owner_paraphrase_binding_preservation_delta"
                                    ] = owner_paraphrase_binding_preservation_delta
                            if (
                                direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
                                and profile_score is not None
                                and profile_base_score is not None
                            ):
                                sample["diversity_outcome"] = diversity_outcome
                                sample["base_score"] = list(profile_base_score)
                                sample["candidate_score"] = list(profile_score)
                            if (
                                direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
                                and coverage_delta is not None
                            ):
                                sample["coverage_outcome"] = coverage_outcome
                                sample["coverage_prep_accepted"] = (
                                    coverage_prep_accepted
                                )
                                sample["coverage_delta"] = coverage_delta
                            if (
                                direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active
                                and coverage_recovery_attempted
                            ):
                                sample["coverage_recovery_attempted"] = (
                                    coverage_recovery_attempted
                                )
                                sample["coverage_recovery_accepted"] = (
                                    coverage_recovery_accepted
                                )
                                sample["coverage_recovery_outcome"] = (
                                    coverage_recovery_outcome
                                )
                                sample["coverage_recovery_records"] = (
                                    coverage_recovery_records
                                )
                                if (
                                    coverage_recovery_learning_rate_scale
                                    is not None
                                ):
                                    sample[
                                        "coverage_recovery_learning_rate_scale"
                                    ] = coverage_recovery_learning_rate_scale
                                if coverage_recovery_delta is not None:
                                    sample["coverage_recovery_delta"] = (
                                        coverage_recovery_delta
                                    )
                                if (
                                    direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
                                ):
                                    sample[
                                        "coverage_recovery_branch_stable_checked"
                                    ] = coverage_recovery_branch_stable_checked
                                    sample[
                                        "coverage_recovery_branch_stable_accepted"
                                    ] = coverage_recovery_branch_stable_accepted
                                    if (
                                        coverage_recovery_branch_stability_preserved
                                        is not None
                                    ):
                                        sample[
                                            "coverage_recovery_branch_stability_preserved"
                                        ] = (
                                            coverage_recovery_branch_stability_preserved
                                        )
                                    if coverage_recovery_prepared_score is not None:
                                        sample[
                                            "coverage_recovery_prepared_score"
                                        ] = list(coverage_recovery_prepared_score)
                                    if coverage_recovery_score is not None:
                                        sample["coverage_recovery_score"] = list(
                                            coverage_recovery_score
                                        )
                                if (
                                    direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active
                                ):
                                    sample[
                                        "branch_diversity_recovery_attempted"
                                    ] = branch_diversity_recovery_attempted
                                    sample[
                                        "branch_diversity_recovery_accepted"
                                    ] = branch_diversity_recovery_accepted
                                    sample[
                                        "branch_diversity_recovery_outcome"
                                    ] = branch_diversity_recovery_outcome
                                    if branch_diversity_recovery_rejection_reason:
                                        sample[
                                            "branch_diversity_recovery_rejection_reason"
                                        ] = (
                                            branch_diversity_recovery_rejection_reason
                                        )
                                    if (
                                        branch_diversity_recovery_learning_rate_scale
                                        is not None
                                    ):
                                        sample[
                                            "branch_diversity_recovery_learning_rate_scale"
                                        ] = (
                                            branch_diversity_recovery_learning_rate_scale
                                        )
                                    sample[
                                        "branch_diversity_recovery_records"
                                    ] = branch_diversity_recovery_records
                                    if (
                                        branch_diversity_recovery_base_score
                                        is not None
                                    ):
                                        sample[
                                            "branch_diversity_recovery_base_score"
                                        ] = list(
                                            branch_diversity_recovery_base_score
                                        )
                                    if branch_diversity_recovery_score is not None:
                                        sample[
                                            "branch_diversity_recovery_score"
                                        ] = list(branch_diversity_recovery_score)
                                    if branch_diversity_recovery_delta is not None:
                                        sample[
                                            "branch_diversity_recovery_delta"
                                        ] = branch_diversity_recovery_delta
                            if (
                                direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active
                            ):
                                sample[
                                    "collapsed_profile_binding_attempted"
                                ] = collapsed_profile_binding_attempted
                                sample[
                                    "collapsed_profile_binding_accepted"
                                ] = collapsed_profile_binding_accepted
                                sample[
                                    "collapsed_profile_binding_outcome"
                                ] = collapsed_profile_binding_outcome
                                sample[
                                    "collapsed_profile_binding_target_profiles"
                                ] = collapsed_profile_binding_target_profiles
                                if collapsed_profile_binding_rejection_reason:
                                    sample[
                                        "collapsed_profile_binding_rejection_reason"
                                    ] = (
                                        collapsed_profile_binding_rejection_reason
                                    )
                                if (
                                    collapsed_profile_binding_learning_rate_scale
                                    is not None
                                ):
                                    sample[
                                        "collapsed_profile_binding_learning_rate_scale"
                                    ] = (
                                        collapsed_profile_binding_learning_rate_scale
                                    )
                                sample[
                                    "collapsed_profile_binding_records"
                                ] = collapsed_profile_binding_records
                                if collapsed_profile_binding_base_score is not None:
                                    sample[
                                        "collapsed_profile_binding_base_score"
                                    ] = list(collapsed_profile_binding_base_score)
                                if collapsed_profile_binding_score is not None:
                                    sample[
                                        "collapsed_profile_binding_score"
                                    ] = list(collapsed_profile_binding_score)
                                if collapsed_profile_binding_delta is not None:
                                    sample[
                                        "collapsed_profile_binding_delta"
                                    ] = collapsed_profile_binding_delta
                            if (
                                isinstance(probe_sample, list)
                                and len(probe_sample) < 12
                            ):
                                probe_sample.append(sample)
                            if (
                                isinstance(profile_scale_sample, list)
                                and len(profile_scale_sample) < 12
                            ):
                                profile_scale_sample.append(sample)
                            if (
                                isinstance(diversity_sample, list)
                                and len(diversity_sample) < 12
                            ):
                                diversity_sample.append(sample)
                            if (
                                isinstance(frontier_sample, list)
                                and len(frontier_sample) < 12
                            ):
                                frontier_sample.append(sample)
                            coverage_sample = direct_answer_update_guard[
                                "profile_scale_coverage_frontier_probe_sample"
                            ]
                            if (
                                isinstance(coverage_sample, list)
                                and len(coverage_sample) < 12
                            ):
                                coverage_sample.append(sample)
                            prep_sample = direct_answer_update_guard[
                                "profile_scale_coverage_prep_frontier_probe_sample"
                            ]
                            if (
                                isinstance(prep_sample, list)
                                and len(prep_sample) < 12
                            ):
                                prep_sample.append(sample)
                            recovery_sample = direct_answer_update_guard[
                                "profile_scale_coverage_recovery_frontier_probe_sample"
                            ]
                            if (
                                isinstance(recovery_sample, list)
                                and len(recovery_sample) < 12
                            ):
                                recovery_sample.append(sample)
                            branch_stable_sample = direct_answer_update_guard[
                                "profile_scale_branch_stable_coverage_recovery_frontier_probe_sample"
                            ]
                            if (
                                direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
                                and
                                isinstance(branch_stable_sample, list)
                                and len(branch_stable_sample) < 12
                            ):
                                branch_stable_sample.append(sample)
                            branch_diversity_sample = direct_answer_update_guard[
                                "profile_scale_branch_diversity_recovery_frontier_probe_sample"
                            ]
                            if (
                                direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active
                                and
                                isinstance(branch_diversity_sample, list)
                                and len(branch_diversity_sample) < 12
                            ):
                                branch_diversity_sample.append(sample)
                            collapsed_binding_sample = direct_answer_update_guard[
                                "profile_scale_collapsed_profile_binding_frontier_probe_sample"
                            ]
                            if (
                                direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active
                                and
                                isinstance(collapsed_binding_sample, list)
                                and len(collapsed_binding_sample) < 12
                            ):
                                collapsed_binding_sample.append(sample)
                            remaining_binding_sample = direct_answer_update_guard[
                                "profile_scale_remaining_profile_binding_probe_sample"
                            ]
                            if (
                                direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
                                and
                                isinstance(remaining_binding_sample, list)
                                and len(remaining_binding_sample) < 12
                            ):
                                remaining_binding_sample.append(sample)
                            owner_paraphrase_sample = direct_answer_update_guard[
                                "profile_scale_owner_paraphrase_binding_probe_sample"
                            ]
                            if (
                                direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                                and
                                isinstance(owner_paraphrase_sample, list)
                                and len(owner_paraphrase_sample) < 12
                            ):
                                owner_paraphrase_sample.append(sample)
                            break
                        direct_answer_update_guard[
                            "sequential_profile_rejections"
                        ] += 1
                        direct_answer_update_guard[
                            "profile_scale_memory_rejections"
                        ] += 1
                        if remaining_profile_binding_prioritized:
                            direct_answer_update_guard[
                                "profile_scale_remaining_profile_binding_prioritized_rejections"
                            ] += 1
                        if owner_paraphrase_binding_prioritized:
                            direct_answer_update_guard[
                                "profile_scale_owner_paraphrase_binding_prioritized_rejections"
                            ] += 1
                        if (
                            direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
                        ):
                            direct_answer_update_guard[
                                "profile_scale_diversity_rejections"
                            ] += 1
                            if not floor_preserved:
                                direct_answer_update_guard[
                                    "profile_scale_diversity_floor_rejections"
                                ] += 1
                            elif not diversity_accepted:
                                direct_answer_update_guard[
                                    "profile_scale_diversity_score_regressions"
                                ] += 1
                            else:
                                diversity_rejection_reason = (
                                    "coverage_frontier_rejection"
                                )
                            rejection_reasons = direct_answer_update_guard[
                                "profile_scale_diversity_rejection_reasons"
                            ]
                            if isinstance(rejection_reasons, dict):
                                rejection_reasons[diversity_rejection_reason] = (
                                    int(
                                        rejection_reasons.get(
                                            diversity_rejection_reason,
                                            0,
                                        )
                                    )
                                    + 1
                                )
                        if (
                            direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
                        ):
                            direct_answer_update_guard[
                                "profile_scale_frontier_rejections"
                            ] += 1
                        if (
                            direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
                        ):
                            direct_answer_update_guard[
                                "profile_scale_coverage_frontier_rejections"
                            ] += 1
                            if coverage_outcome == "gained":
                                direct_answer_update_guard[
                                    "profile_scale_coverage_frontier_gains"
                                ] += 1
                            elif coverage_outcome == "tied":
                                direct_answer_update_guard[
                                    "profile_scale_coverage_frontier_ties"
                                ] += 1
                            elif coverage_outcome in {
                                "regressed",
                                "floor_regressed",
                            }:
                                direct_answer_update_guard[
                                    "profile_scale_coverage_frontier_regressions"
                                ] += 1
                            coverage_reasons = direct_answer_update_guard[
                                "profile_scale_coverage_frontier_rejection_reasons"
                            ]
                            if isinstance(coverage_reasons, dict):
                                reason = coverage_rejection_reason or "not_accepted"
                                coverage_reasons[reason] = (
                                    int(coverage_reasons.get(reason, 0)) + 1
                                )
                        if (
                            direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
                        ):
                            direct_answer_update_guard[
                                "profile_scale_coverage_prep_frontier_rejections"
                            ] += 1
                            prep_reasons = direct_answer_update_guard[
                                "profile_scale_coverage_prep_frontier_rejection_reasons"
                            ]
                            if isinstance(prep_reasons, dict):
                                reason = coverage_rejection_reason or (
                                    diversity_rejection_reason
                                    or "not_accepted"
                                )
                                prep_reasons[reason] = (
                                    int(prep_reasons.get(reason, 0)) + 1
                                )
                        rejected_counts = direct_answer_update_guard[
                            "sequential_profile_rejection_counts"
                        ]
                        if isinstance(rejected_counts, dict):
                            rejected_counts[profile] = (
                                int(rejected_counts.get(profile, 0)) + 1
                            )
                        scale_counts = direct_answer_update_guard[
                            "profile_scale_rejection_scale_counts"
                        ]
                        if isinstance(scale_counts, dict):
                            scale_counts[scale_key] = (
                                int(scale_counts.get(scale_key, 0)) + 1
                            )
                        diagnostics = (
                            branch_diversity_snapshot_target_coverage_diagnostics(
                                profile_probe_snapshot,
                                direct_baseline,
                            )
                        )
                        sample = {
                            "profile": profile,
                            "accepted": False,
                            "records": len(profile_batch),
                            "frontier_records": profile_frontier_records,
                            "learning_rate_scale": profile_scale,
                            "worst_violation": diagnostics["worst_violation"],
                            "violating_profile_count": diagnostics[
                                "violating_profile_count"
                            ],
                        }
                        if (
                            direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
                            and profile_score is not None
                            and profile_base_score is not None
                        ):
                            sample["diversity_outcome"] = diversity_outcome
                            sample["diversity_rejection_reason"] = (
                                diversity_rejection_reason
                            )
                            sample["base_score"] = list(profile_base_score)
                            sample["candidate_score"] = list(profile_score)
                        if (
                            direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
                            and coverage_delta is not None
                        ):
                            sample["coverage_outcome"] = coverage_outcome
                            sample["coverage_prep_accepted"] = (
                                coverage_prep_accepted
                            )
                            sample["coverage_rejection_reason"] = (
                                coverage_rejection_reason
                            )
                            sample["coverage_delta"] = coverage_delta
                        if (
                            isinstance(probe_sample, list)
                            and len(probe_sample) < 12
                        ):
                            probe_sample.append(sample)
                        if (
                            isinstance(profile_scale_sample, list)
                            and len(profile_scale_sample) < 12
                        ):
                            profile_scale_sample.append(sample)
                        if (
                            isinstance(diversity_sample, list)
                            and len(diversity_sample) < 12
                        ):
                            diversity_sample.append(sample)
                        if (
                            isinstance(frontier_sample, list)
                            and len(frontier_sample) < 12
                        ):
                            frontier_sample.append(sample)
                        coverage_sample = direct_answer_update_guard[
                            "profile_scale_coverage_frontier_probe_sample"
                        ]
                        if (
                            isinstance(coverage_sample, list)
                            and len(coverage_sample) < 12
                        ):
                            coverage_sample.append(sample)
                        prep_sample = direct_answer_update_guard[
                            "profile_scale_coverage_prep_frontier_probe_sample"
                        ]
                        if (
                            isinstance(prep_sample, list)
                            and len(prep_sample) < 12
                        ):
                            prep_sample.append(sample)
                    if not profile_accepted:
                        restore_direct_update_state(
                            profile_model_payload,
                            profile_optimizer_payload,
                        )
                        direct_rng.setstate(profile_rng_state)
                return total_loss / max(loss_count, 1), accepted_any
            if direct_answer_baseline_floor_sequential_stabilization_active:
                profile_groups = baseline_floor_anchor_profile_groups(
                    direct_baseline_floor_repair_anchors
                )
                sequential_update_shape = (
                    "calibrated_sequential_profile_stabilization"
                    if direct_answer_baseline_floor_calibrated_sequential_stabilization_active
                    else "sequential_profile_stabilization"
                )
                total_loss = 0.0
                accepted_any = False
                for profile, profile_anchors in profile_groups.items():
                    profile_model_payload = model.to_dict(tokenizer)
                    profile_optimizer_payload = optimizer.to_dict()
                    profile_rng_state = direct_rng.getstate()
                    profile_batch = baseline_floor_objective_anchor_batch(
                        profile_anchors,
                        direct_rng,
                        len(profile_anchors),
                    )
                    direct_answer_update_guard["sequential_profile_attempts"] += 1
                    direct_answer_update_guard["sequential_profile_records"] += len(
                        profile_batch
                    )
                    direct_answer_update_guard["stabilization_anchor_batches"] += 1
                    direct_answer_update_guard["stabilization_anchor_records"] += len(
                        profile_batch
                    )
                    profile_loss = train_direct_answer_baseline_floor_anchor_batch(
                        model,
                        profile_batch,
                        update_learning_rate,
                        params=direct_params,
                    )
                    total_loss += profile_loss
                    profile_probe_snapshot = direct_snapshot_record(
                        direct_step,
                        None,
                        {
                            "baseline_floor_update_guard_probe": True,
                            "baseline_floor_sequential_profile_probe": True,
                            "baseline_floor_calibrated_sequential_profile_probe": (
                                direct_answer_baseline_floor_calibrated_sequential_stabilization_active
                            ),
                            "learning_rate_scale": (
                                update_learning_rate
                                / max(args.direct_answer_learning_rate, 1e-12)
                            ),
                            "update_shape": sequential_update_shape,
                            "sequential_profile": profile,
                            "sequential_profile_records": len(profile_batch),
                        },
                    )
                    probe_sample = direct_answer_update_guard[
                        "sequential_profile_probe_sample"
                    ]
                    if branch_diversity_snapshot_preserves_target_coverage(
                        profile_probe_snapshot,
                        direct_baseline,
                    ):
                        direct_answer_update_guard[
                            "sequential_profile_acceptances"
                        ] += 1
                        accepted_counts = direct_answer_update_guard[
                            "sequential_profile_acceptance_counts"
                        ]
                        if isinstance(accepted_counts, dict):
                            accepted_counts[profile] = (
                                int(accepted_counts.get(profile, 0)) + 1
                            )
                        accepted_any = True
                        if isinstance(probe_sample, list) and len(probe_sample) < 12:
                            probe_sample.append(
                                {
                                    "profile": profile,
                                    "accepted": True,
                                    "records": len(profile_batch),
                                }
                            )
                    else:
                        direct_answer_update_guard[
                            "sequential_profile_rejections"
                        ] += 1
                        rejected_counts = direct_answer_update_guard[
                            "sequential_profile_rejection_counts"
                        ]
                        if isinstance(rejected_counts, dict):
                            rejected_counts[profile] = (
                                int(rejected_counts.get(profile, 0)) + 1
                            )
                        diagnostics = (
                            branch_diversity_snapshot_target_coverage_diagnostics(
                                profile_probe_snapshot,
                                direct_baseline,
                            )
                        )
                        if isinstance(probe_sample, list) and len(probe_sample) < 12:
                            probe_sample.append(
                                {
                                    "profile": profile,
                                    "accepted": False,
                                    "records": len(profile_batch),
                                    "worst_violation": diagnostics[
                                        "worst_violation"
                                    ],
                                    "violating_profile_count": diagnostics[
                                        "violating_profile_count"
                                    ],
                                }
                            )
                        restore_direct_update_state(
                            profile_model_payload,
                            profile_optimizer_payload,
                        )
                        direct_rng.setstate(profile_rng_state)
                divisor = max(len(profile_groups), 1)
                return total_loss / divisor, accepted_any
            stabilization_batch_size = (
                len(direct_baseline_floor_repair_anchors)
                if direct_answer_baseline_floor_profile_targeted_stabilization_active
                else BASELINE_FLOOR_STABILIZATION_ANCHOR_BATCH_SIZE
            )
            stabilization_batch = baseline_floor_objective_anchor_batch(
                direct_baseline_floor_repair_anchors,
                direct_rng,
                stabilization_batch_size,
            )
            direct_answer_update_guard["stabilization_anchor_batches"] += 1
            direct_answer_update_guard["stabilization_anchor_records"] += len(
                stabilization_batch
            )
            return (
                train_direct_answer_baseline_floor_anchor_batch(
                    model,
                    stabilization_batch,
                    update_learning_rate,
                    params=direct_params,
                ),
                bool(stabilization_batch),
            )

        def train_baseline_floor_anchor_repair(
            update_learning_rate: float,
        ) -> float:
            repair_loss = 0.0
            repair_batch_size = max(
                args.direct_answer_branch_batch_size,
                args.direct_answer_branch_batch_size + max(0, args.direct_answer_hard_negatives),
            )
            for _repair_step in range(BASELINE_FLOOR_REPAIR_STEPS):
                direct_answer_update_guard["repair_updates"] += 1
                repair_loss += train_direct_answer_baseline_floor_anchor_repair(
                    model,
                    direct_baseline_floor_repair_anchors,
                    direct_rng,
                    update_learning_rate,
                    repair_batch_size,
                    params=direct_params,
                )
            return repair_loss / max(BASELINE_FLOOR_REPAIR_STEPS, 1)

        def train_adaptive_baseline_floor_update(
            direct_step: int,
            base_model_payload: dict[str, Any],
            base_optimizer_payload: dict[str, Any],
            base_rng_state: object,
        ) -> float:
            last_loss = 0.0
            direct_answer_update_guard["checked_steps"] += 1
            for learning_rate_scale in direct_baseline_floor_outer_learning_rate_scales:
                restore_direct_update_state(base_model_payload, base_optimizer_payload)
                direct_rng.setstate(base_rng_state)
                direct_answer_update_guard["attempted_updates"] += 1
                if (
                    direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active
                ):
                    if (
                        direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active
                    ):
                        if (
                            direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active
                        ):
                            if (
                                direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
                            ):
                                if (
                                    direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
                                ):
                                    attempt_update_shape = (
                                        "profile_scale_owner_paraphrase_binding_frontier_calibrated_sequential_profile_stabilization"
                                    )
                                else:
                                    attempt_update_shape = (
                                        "profile_scale_remaining_profile_binding_frontier_calibrated_sequential_profile_stabilization"
                                    )
                            else:
                                attempt_update_shape = (
                                    "profile_scale_collapsed_profile_binding_frontier_calibrated_sequential_profile_stabilization"
                                )
                        else:
                            attempt_update_shape = (
                                "profile_scale_branch_diversity_recovery_frontier_calibrated_sequential_profile_stabilization"
                            )
                    elif (
                        direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
                    ):
                        attempt_update_shape = (
                            "profile_scale_branch_stable_coverage_recovery_frontier_diversity_calibrated_sequential_profile_stabilization"
                        )
                    elif (
                        direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active
                    ):
                        attempt_update_shape = (
                            "profile_scale_coverage_recovery_frontier_diversity_calibrated_sequential_profile_stabilization"
                        )
                    elif (
                        direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
                    ):
                        attempt_update_shape = (
                            "profile_scale_coverage_prep_frontier_diversity_calibrated_sequential_profile_stabilization"
                        )
                    elif (
                        direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
                    ):
                        attempt_update_shape = (
                            "profile_scale_coverage_frontier_diversity_calibrated_sequential_profile_stabilization"
                        )
                    elif (
                        direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
                    ):
                        attempt_update_shape = (
                            "profile_scale_frontier_diversity_calibrated_sequential_profile_stabilization"
                        )
                    elif (
                        direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
                    ):
                        attempt_update_shape = (
                            "profile_scale_diversity_calibrated_sequential_profile_stabilization"
                        )
                    else:
                        attempt_update_shape = (
                            "profile_scale_calibrated_sequential_profile_stabilization"
                        )
                elif (
                    direct_answer_baseline_floor_calibrated_sequential_stabilization_active
                ):
                    attempt_update_shape = (
                        "calibrated_sequential_profile_stabilization"
                    )
                elif direct_answer_baseline_floor_sequential_stabilization_active:
                    attempt_update_shape = "sequential_profile_stabilization"
                elif direct_answer_baseline_floor_profile_targeted_stabilization_active:
                    attempt_update_shape = "profile_targeted_stabilization"
                elif direct_answer_baseline_floor_stabilization_active:
                    attempt_update_shape = "stabilization"
                else:
                    attempt_update_shape = "direct"
                update_applied = True
                if direct_answer_baseline_floor_stabilization_active:
                    last_loss, update_applied = train_baseline_floor_stabilization_update(
                        args.direct_answer_learning_rate * learning_rate_scale,
                        direct_step,
                    )
                else:
                    last_loss = train_baseline_anchored_prompt_update(
                        args.direct_answer_learning_rate * learning_rate_scale
                    )
                probe_snapshot = direct_snapshot_record(
                    direct_step,
                    None,
                    {
                        "baseline_floor_update_guard_probe": True,
                        "learning_rate_scale": learning_rate_scale,
                        "update_shape": attempt_update_shape,
                    },
                )
                if (
                    update_applied
                    and branch_diversity_snapshot_preserves_target_coverage(
                        probe_snapshot,
                        direct_baseline,
                    )
                ):
                    if (
                        direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
                        and branch_diversity_snapshot_score(probe_snapshot)
                        < branch_diversity_snapshot_score(direct_baseline)
                    ):
                        direct_answer_update_guard[
                            "profile_scale_diversity_outer_rejections"
                        ] += 1
                    else:
                        if (
                            direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
                        ):
                            direct_answer_update_guard[
                                "profile_scale_diversity_outer_acceptances"
                            ] += 1
                        record_guard_acceptance(
                            learning_rate_scale,
                            attempt_update_shape,
                        )
                        return last_loss
                if not update_applied:
                    direct_answer_update_guard[
                        "rejected_no_effective_update_attempts"
                    ] += 1
                rejection_snapshot = probe_snapshot
                rejection_update_shape = attempt_update_shape
                if (
                    direct_answer_baseline_floor_repaired_updates_active
                    and direct_baseline_floor_repair_anchors
                ):
                    direct_answer_update_guard["repair_attempts"] += 1
                    repair_loss = train_baseline_floor_anchor_repair(
                        args.direct_answer_learning_rate * learning_rate_scale
                    )
                    repaired_probe_snapshot = direct_snapshot_record(
                        direct_step,
                        None,
                        {
                            "baseline_floor_update_guard_probe": True,
                            "baseline_floor_repair_probe": True,
                            "learning_rate_scale": learning_rate_scale,
                        },
                    )
                    if branch_diversity_snapshot_preserves_target_coverage(
                        repaired_probe_snapshot,
                        direct_baseline,
                    ):
                        record_guard_acceptance(learning_rate_scale, "repaired")
                        return (last_loss + repair_loss) / 2.0
                    rejection_snapshot = repaired_probe_snapshot
                    rejection_update_shape = "repaired"
                record_guard_rejection_attempt(
                    direct_step,
                    rejection_snapshot,
                    learning_rate_scale,
                    rejection_update_shape,
                )
            direct_answer_update_guard["rejected_steps"] += 1
            restore_direct_update_state(base_model_payload, base_optimizer_payload)
            direct_rng.setstate(base_rng_state)
            return last_loss

        for direct_step in range(1, direct_steps_to_run + 1):
            example = direct_training_cursor.next()
            pre_update_model_payload: dict[str, Any] | None = None
            pre_update_optimizer_payload: dict[str, Any] | None = None
            pre_update_rng_state = None
            direct_answer_update_guard_applied = False
            if direct_answer_baseline_floor_update_gate_active:
                pre_update_model_payload = model.to_dict(tokenizer)
                pre_update_optimizer_payload = optimizer.to_dict()
                pre_update_rng_state = direct_rng.getstate()
            if args.direct_answer_mode == "first-error":
                running_direct_loss += train_direct_answer_first_error(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "first-error-unlikelihood":
                running_direct_loss += train_direct_answer_first_error_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "rollout-unlikelihood":
                running_direct_loss += train_direct_answer_rollout_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "hybrid-unlikelihood":
                if direct_step % 2 == 0:
                    running_direct_loss += train_direct_answer_rollout_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "staged-unlikelihood":
                if direct_step <= args.direct_answer_steps // 2:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_rollout_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "periodic-rollout-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_rollout_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "early-stop-unlikelihood":
                running_direct_loss += train_direct_answer_early_stop_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-early-stop-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_early_stop_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "repeat-loop-unlikelihood":
                running_direct_loss += train_direct_answer_repeat_loop_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-repeat-loop-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_repeat_loop_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "balanced-repair-unlikelihood":
                running_direct_loss += train_direct_answer_balanced_repair_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-balanced-repair-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_balanced_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "generated-prefix-recovery-unlikelihood":
                running_direct_loss += train_direct_answer_generated_prefix_recovery_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_recovery_steps,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-generated-prefix-recovery-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_generated_prefix_recovery_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_recovery_steps,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "sequence-repair-unlikelihood":
                running_direct_loss += train_direct_answer_sequence_repair_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-sequence-repair-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_sequence_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "loop-escape-unlikelihood":
                running_direct_loss += train_direct_answer_loop_escape_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-loop-escape-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_loop_escape_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "periodic-sequence-loop-escape-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                sequence_interval = max(1, args.direct_answer_sequence_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_loop_escape_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                elif direct_step % sequence_interval == 0:
                    running_direct_loss += train_direct_answer_sequence_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "branch-repair-unlikelihood":
                running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_branch_position,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-branch-repair-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "branch-collapse-unlikelihood":
                running_direct_loss += train_direct_answer_branch_collapse_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-branch-collapse-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_collapse_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_hard_negatives,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "branch-batch-contrast-unlikelihood":
                running_direct_loss += train_direct_answer_branch_batch_contrast_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-branch-batch-contrast-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_batch_contrast_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_branch_batch_size,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "branch-diversity-unlikelihood":
                running_direct_loss += train_direct_answer_branch_diversity_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-branch-diversity-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_diversity_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_branch_batch_size,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "branch-target-softmax-unlikelihood":
                running_direct_loss += train_direct_answer_branch_target_softmax_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-branch-target-softmax-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_target_softmax_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_branch_batch_size,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "branch-target-margin-unlikelihood":
                running_direct_loss += train_direct_answer_branch_target_margin_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-branch-target-margin-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_target_margin_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_branch_batch_size,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "branch-representation-contrast-unlikelihood":
                running_direct_loss += train_direct_answer_branch_representation_contrast_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "branch-balanced-representation-contrast-unlikelihood":
                running_direct_loss += train_direct_answer_branch_representation_contrast_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                )
            elif args.direct_answer_mode == "branch-output-binding-unlikelihood":
                running_direct_loss += train_direct_answer_branch_output_binding_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "branch-bidirectional-binding-unlikelihood":
                running_direct_loss += train_direct_answer_branch_bidirectional_binding_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "branch-balanced-bidirectional-binding-unlikelihood":
                running_direct_loss += train_direct_answer_branch_bidirectional_binding_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                )
            elif args.direct_answer_mode == "branch-coverage-binding-unlikelihood":
                running_direct_loss += train_direct_answer_branch_coverage_binding_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "branch-balanced-coverage-binding-unlikelihood":
                running_direct_loss += train_direct_answer_branch_coverage_binding_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                )
            elif args.direct_answer_mode == "branch-target-set-coverage-unlikelihood":
                running_direct_loss += train_direct_answer_branch_target_set_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "branch-balanced-target-set-coverage-unlikelihood":
                running_direct_loss += train_direct_answer_branch_target_set_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                )
            elif args.direct_answer_mode == "branch-target-diversity-unlikelihood":
                running_direct_loss += train_direct_answer_branch_target_diversity_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "branch-balanced-target-diversity-unlikelihood":
                running_direct_loss += train_direct_answer_branch_target_diversity_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                )
            elif args.direct_answer_mode == "branch-target-replay-coverage-unlikelihood":
                running_direct_loss += train_direct_answer_branch_target_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "branch-balanced-target-replay-coverage-unlikelihood":
                running_direct_loss += train_direct_answer_branch_target_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                )
            elif args.direct_answer_mode == "branch-context-replay-coverage-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "branch-balanced-context-replay-coverage-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                )
            elif args.direct_answer_mode == "branch-context-coverage-anchor-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    preserve_covered_targets=True,
                )
            elif args.direct_answer_mode == "branch-balanced-context-coverage-anchor-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                    preserve_covered_targets=True,
                )
            elif args.direct_answer_mode == "branch-context-target-balanced-anchor-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    preserve_covered_targets=True,
                    balance_covered_target_anchors=True,
                )
            elif args.direct_answer_mode == "branch-balanced-context-target-balanced-anchor-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                    preserve_covered_targets=True,
                    balance_covered_target_anchors=True,
                )
            elif args.direct_answer_mode == "branch-context-coverage-deficit-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    focus_uncovered_targets=True,
                )
            elif args.direct_answer_mode == "branch-balanced-context-coverage-deficit-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                    focus_uncovered_targets=True,
                )
            elif args.direct_answer_mode == "branch-context-coverage-preserving-deficit-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    focus_uncovered_targets=True,
                    preserve_predicted_target_coverage=True,
                    balance_deficit_targets=True,
                )
            elif args.direct_answer_mode == "branch-balanced-context-coverage-preserving-deficit-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                    focus_uncovered_targets=True,
                    preserve_predicted_target_coverage=True,
                    balance_deficit_targets=True,
                )
            elif args.direct_answer_mode == "branch-context-profile-coverage-preserving-deficit-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    focus_uncovered_targets=True,
                    preserve_predicted_target_coverage=True,
                    balance_deficit_targets=True,
                    profile_aware_targets=True,
                )
            elif args.direct_answer_mode == "branch-balanced-context-profile-coverage-preserving-deficit-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                    focus_uncovered_targets=True,
                    preserve_predicted_target_coverage=True,
                    balance_deficit_targets=True,
                    profile_aware_targets=True,
                )
            elif args.direct_answer_mode == "branch-balanced-context-profile-target-share-preserving-deficit-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                    focus_uncovered_targets=True,
                    preserve_predicted_target_coverage=True,
                    balance_deficit_targets=True,
                    profile_aware_targets=True,
                    balance_profile_target_shares=True,
                )
            elif args.direct_answer_mode == "branch-balanced-context-profile-prompt-ownership-target-share-preserving-deficit-unlikelihood":
                running_direct_loss += train_direct_answer_branch_context_replay_coverage_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                    focus_uncovered_targets=True,
                    preserve_predicted_target_coverage=True,
                    balance_deficit_targets=True,
                    profile_aware_targets=True,
                    balance_profile_target_shares=True,
                    enforce_prompt_target_margins=True,
                )
            elif args.direct_answer_mode in BASELINE_ANCHORED_DIRECT_ANSWER_MODES:
                if (
                    direct_answer_baseline_floor_adaptive_updates_active
                    and pre_update_model_payload is not None
                    and pre_update_optimizer_payload is not None
                    and pre_update_rng_state is not None
                ):
                    running_direct_loss += train_adaptive_baseline_floor_update(
                        direct_step,
                        pre_update_model_payload,
                        pre_update_optimizer_payload,
                        pre_update_rng_state,
                    )
                    direct_answer_update_guard_applied = True
                else:
                    running_direct_loss += train_baseline_anchored_prompt_update(
                        args.direct_answer_learning_rate
                    )
            elif args.direct_answer_mode == "branch-rank-margin-unlikelihood":
                running_direct_loss += train_direct_answer_branch_rank_margin_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "branch-balanced-rank-margin-unlikelihood":
                running_direct_loss += train_direct_answer_branch_rank_margin_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                )
            elif args.direct_answer_mode == "branch-topk-softmax-unlikelihood":
                running_direct_loss += train_direct_answer_branch_topk_softmax_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "branch-balanced-topk-softmax-unlikelihood":
                running_direct_loss += train_direct_answer_branch_topk_softmax_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_batch_size,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                    balance_targets=True,
                )
            elif args.direct_answer_mode == "periodic-branch-representation-contrast-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_representation_contrast_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_branch_batch_size,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "branch-span-repair-unlikelihood":
                running_direct_loss += train_direct_answer_branch_span_repair_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_span,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-branch-span-repair-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_span_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_branch_span,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "branch-contrast-unlikelihood":
                running_direct_loss += train_direct_answer_branch_contrast_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-branch-contrast-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_contrast_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "branch-span-contrast-unlikelihood":
                running_direct_loss += train_direct_answer_branch_span_contrast_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_branch_span,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-branch-span-contrast-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_span_contrast_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_branch_span,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "hard-branch-contrast-unlikelihood":
                running_direct_loss += train_direct_answer_hard_branch_contrast_unlikelihood(
                    model,
                    tokenizer,
                    example,
                    direct_training_pool,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    args.direct_answer_negative_weight,
                    args.direct_answer_positive_weight,
                    args.direct_answer_contrast_weight,
                    args.direct_answer_branch_position,
                    args.direct_answer_hard_negatives,
                    direct_answer_terminator,
                    direct_params,
                )
            elif args.direct_answer_mode == "periodic-hard-branch-contrast-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_hard_branch_contrast_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_hard_negatives,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_first_error_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "periodic-branch-repair-contrast-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_contrast_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "periodic-branch-span-repair-contrast-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_branch_span_contrast_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_branch_span,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_branch_span_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_branch_span,
                        direct_answer_terminator,
                        direct_params,
                    )
            elif args.direct_answer_mode == "periodic-hard-branch-repair-contrast-unlikelihood":
                rollout_interval = max(1, args.direct_answer_rollout_interval)
                if direct_step % rollout_interval == 0:
                    running_direct_loss += train_direct_answer_hard_branch_contrast_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_training_pool,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_contrast_weight,
                        args.direct_answer_branch_position,
                        args.direct_answer_hard_negatives,
                        direct_answer_terminator,
                        direct_params,
                    )
                else:
                    running_direct_loss += train_direct_answer_branch_repair_unlikelihood(
                        model,
                        tokenizer,
                        example,
                        direct_lessons[example],
                        direct_rng,
                        args.direct_answer_learning_rate,
                        args.direct_answer_negative_weight,
                        args.direct_answer_positive_weight,
                        args.direct_answer_branch_position,
                        direct_answer_terminator,
                        direct_params,
                    )
            else:
                running_direct_loss += train_direct_answer_lesson(
                    model,
                    direct_lessons[example],
                    direct_rng,
                    args.direct_answer_learning_rate,
                    direct_params,
                )
            if (
                direct_answer_baseline_floor_update_gate_active
                and not direct_answer_update_guard_applied
            ):
                direct_answer_update_guard["checked_steps"] += 1
                direct_answer_update_guard["attempted_updates"] += 1
                probe_snapshot = direct_snapshot_record(
                    direct_step,
                    None,
                    {
                        "baseline_floor_update_guard_probe": True,
                        "learning_rate_scale": 1.0,
                    },
                )
                if branch_diversity_snapshot_preserves_target_coverage(
                    probe_snapshot,
                    direct_baseline,
                ):
                    record_guard_acceptance(1.0)
                else:
                    direct_answer_update_guard["rejected_steps"] += 1
                    record_guard_rejection_attempt(direct_step, probe_snapshot, 1.0)
                    if (
                        pre_update_model_payload is not None
                        and pre_update_optimizer_payload is not None
                    ):
                        restore_direct_update_state(
                            pre_update_model_payload,
                            pre_update_optimizer_payload,
                        )
            if (
                args.direct_answer_eval_every > 0
                and direct_step % args.direct_answer_eval_every == 0
            ):
                train_loss = running_direct_loss / args.direct_answer_eval_every
                last_direct_snapshot = direct_snapshot(direct_step, train_loss)
                last_direct_snapshot_step = direct_step
                record_best_direct_snapshot(last_direct_snapshot)
                print(f"direct_answer_step={direct_step} train_loss={train_loss:.4f}")
                running_direct_loss = 0.0

        if last_direct_snapshot_step != args.direct_answer_steps:
            last_direct_snapshot = direct_snapshot(args.direct_answer_steps, None)
            last_direct_snapshot_step = args.direct_answer_steps
            record_best_direct_snapshot(last_direct_snapshot)

        direct_answer_restored_best_branch_snapshot = False
        if (
            args.direct_answer_restore_best_branch_snapshot
            and best_direct_snapshot_step != last_direct_snapshot_step
        ):
            restored_model, restored_tokenizer = TinyTransformerLM.from_dict(
                best_direct_model_payload
            )
            model = restored_model
            if restored_tokenizer is not None:
                tokenizer = restored_tokenizer
            optimizer = ScalarOptimizer.from_dict(best_direct_optimizer_payload)
            model.active_optimizer = optimizer
            direct_answer_restored_best_branch_snapshot = True
            last_direct_snapshot = direct_snapshot(
                args.direct_answer_steps,
                None,
                {
                    "restored_best_branch_snapshot": True,
                    "restored_from_step": best_direct_snapshot_step,
                    "restored_from_score": list(best_direct_snapshot_score),
                },
            )

        post_direct_candidate_snapshot_skipped = args.skip_post_direct_snapshot
        if post_direct_candidate_snapshot_skipped:
            print("skipped post-direct candidate snapshot")
        else:
            last_snapshot = snapshot(args.steps + args.direct_answer_steps, None)
            post_direct_candidate_snapshot = last_snapshot
        direct_answer_metrics = {
            "architecture": "tiny-decoder-only-transformer-direct-answer",
            "checkpoint": str(args.run / "transformer_answer.json"),
            "history": str(direct_history_path),
            "steps": args.direct_answer_steps,
            "actual_steps": direct_steps_to_run,
            "training_examples": len(direct_training_pool),
            "learning_rate": args.direct_answer_learning_rate,
            "direct_answer_eval_every": args.direct_answer_eval_every,
            "direct_answer_snapshot_mode": args.direct_answer_snapshot_mode,
            "direct_answer_evals_skipped": (
                args.direct_answer_snapshot_mode == "branch-only"
            ),
            "direct_answer_mode": args.direct_answer_mode,
            "direct_answer_profile_aware_targets": direct_profile_aware_targets,
            "direct_answer_replay_plan": (
                str(direct_replay_plan_path)
                if direct_replay_plan_path is not None
                else None
            ),
            "direct_answer_replay_plan_summary": direct_replay_plan,
            "direct_answer_replay_prediction_anchor_count": (
                len(direct_replay_prediction_overrides)
                if direct_replay_prediction_overrides is not None
                else 0
            ),
            "direct_answer_replay_prediction_anchors_active": (
                direct_replay_prediction_anchors_active
            ),
            "direct_answer_baseline_floor_update_gate_active": (
                direct_answer_baseline_floor_update_gate_active
            ),
            "direct_answer_baseline_floor_adaptive_updates_active": (
                direct_answer_baseline_floor_adaptive_updates_active
            ),
            "direct_answer_baseline_floor_repaired_updates_active": (
                direct_answer_baseline_floor_repaired_updates_active
            ),
            "direct_answer_baseline_floor_objective_active": (
                direct_answer_baseline_floor_objective_active
            ),
            "direct_answer_baseline_floor_stabilization_active": (
                direct_answer_baseline_floor_stabilization_active
            ),
            "direct_answer_baseline_floor_profile_targeted_stabilization_active": (
                direct_answer_baseline_floor_profile_targeted_stabilization_active
            ),
            "direct_answer_baseline_floor_sequential_stabilization_active": (
                direct_answer_baseline_floor_sequential_stabilization_active
            ),
            "direct_answer_baseline_floor_calibrated_sequential_stabilization_active": (
                direct_answer_baseline_floor_calibrated_sequential_stabilization_active
            ),
            "direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_calibrated_stabilization_active
            ),
            "direct_answer_baseline_floor_profile_scale_diversity_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_diversity_stabilization_active
            ),
            "direct_answer_baseline_floor_profile_scale_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_frontier_stabilization_active
            ),
            "direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_coverage_frontier_stabilization_active
            ),
            "direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_coverage_prep_frontier_stabilization_active
            ),
            "direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_coverage_recovery_frontier_stabilization_active
            ),
            "direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active
            ),
            "direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_branch_diversity_recovery_frontier_stabilization_active
            ),
            "direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_collapsed_profile_binding_frontier_stabilization_active
            ),
            "direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active
            ),
            "direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active": (
                direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active
            ),
            "direct_answer_update_guard": direct_answer_update_guard,
            "direct_answer_negative_weight": args.direct_answer_negative_weight,
            "direct_answer_positive_weight": args.direct_answer_positive_weight,
            "direct_answer_contrast_weight": args.direct_answer_contrast_weight,
            "direct_answer_recovery_steps": args.direct_answer_recovery_steps,
            "direct_answer_branch_position": args.direct_answer_branch_position,
            "direct_answer_branch_span": args.direct_answer_branch_span,
            "direct_answer_branch_batch_size": args.direct_answer_branch_batch_size,
            "direct_answer_hard_negatives": args.direct_answer_hard_negatives,
            "direct_answer_train_top_layer_only": args.direct_answer_train_top_layer_only,
            "direct_answer_freeze_output_bias": args.direct_answer_freeze_output_bias,
            "direct_answer_restore_best_branch_snapshot": (
                args.direct_answer_restore_best_branch_snapshot
            ),
            "direct_answer_restored_best_branch_snapshot": (
                direct_answer_restored_best_branch_snapshot
            ),
            "direct_answer_best_branch_snapshot_step": best_direct_snapshot_step,
            "direct_answer_best_branch_snapshot_score": list(best_direct_snapshot_score),
            "direct_answer_branch_snapshot_coverage_floor": (
                branch_diversity_snapshot_target_coverage_by_profile(direct_baseline)
            ),
            "direct_answer_require_branch_context_gate": (
                args.direct_answer_require_branch_context_gate
            ),
            "direct_answer_training_skipped": direct_answer_training_skipped,
            "direct_answer_skip_reason": direct_answer_skip_reason,
            "direct_answer_branch_context_gate": branch_context_gate,
            "post_direct_candidate_snapshot_skipped": post_direct_candidate_snapshot_skipped,
            "direct_answer_sequence_interval": args.direct_answer_sequence_interval,
            "direct_answer_rollout_interval": args.direct_answer_rollout_interval,
            "max_new_chars": args.direct_answer_max_new_chars,
            "generation_config": asdict(generation_config),
            "terminator": repr(direct_answer_terminator),
            "context_coverage": context_coverage,
            "baseline": direct_baseline,
            "final": last_direct_snapshot,
            "uses_answer_candidates": False,
            "auxiliary_weights": False,
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "training_data": TRAINING_DATA_DESCRIPTION,
        }

    selector_metrics: dict[str, Any] | None = None
    if args.selector_steps > 0:
        selector = build_answer_selector(examples, args.seed + 101)
        selector_rng = random.Random(args.seed + 101)
        selector_history_path = args.run / "answer_selector_metrics.jsonl"
        selector_history_writer = JsonlHistoryWriter(selector_history_path)

        def selector_snapshot(step: int, train_loss: float | None) -> dict[str, Any]:
            record = {
                "step": step,
                "train_loss": train_loss,
                "evals": {
                    name: evaluate_answer_records(
                        model,
                        tokenizer,
                        records,
                        candidates if args.candidate_scope == "all" else eval_candidates[name],
                        args.max_new_chars,
                        include_completions=False,
                        selector=selector,
                        emit_selected_candidate=args.selector_emit_completions,
                    )
                    for name, records in sorted(eval_records.items())
                },
            }
            return selector_history_writer.append(record)

        selector_baseline = selector_snapshot(0, None)
        running_selector_loss = 0.0
        last_selector_snapshot = selector_baseline
        last_selector_snapshot_step = 0
        selector_training_cursor = ShuffledTrainingCursor(training_pool, selector_rng)
        selector_candidates = selector.config.labels
        for selector_step in range(1, args.selector_steps + 1):
            example = selector_training_cursor.next()
            if args.selector_negatives > 0:
                selector_batch = sampled_choice_candidates(
                    example.target,
                    selector_candidates,
                    selector_rng,
                    args.selector_negatives,
                )
            else:
                selector_batch = selector_candidates
            running_selector_loss += selector.train_step(
                example,
                args.selector_learning_rate,
                selector_batch,
            )
            if (
                args.selector_eval_every > 0
                and selector_step % args.selector_eval_every == 0
            ):
                train_loss = running_selector_loss / args.selector_eval_every
                last_selector_snapshot = selector_snapshot(selector_step, train_loss)
                last_selector_snapshot_step = selector_step
                print(f"selector_step={selector_step} train_loss={train_loss:.4f}")
                running_selector_loss = 0.0

        if last_selector_snapshot_step != args.selector_steps:
            last_selector_snapshot = selector_snapshot(args.selector_steps, None)

        selector_checkpoint_path = args.run / "answer_selector.json"
        selector.save(selector_checkpoint_path)
        selector_metrics = {
            "architecture": "closed-world-answer-candidate-selector",
            "checkpoint": str(selector_checkpoint_path),
            "history": str(selector_history_path),
            "steps": args.selector_steps,
            "learning_rate": args.selector_learning_rate,
            "selector_negatives": args.selector_negatives,
            "selector_eval_every": args.selector_eval_every,
            "selector_emit_completions": args.selector_emit_completions,
            "labels": len(selector.config.labels),
            "features": len(selector.config.features),
            "candidate_scope": args.candidate_scope,
            "baseline": selector_baseline,
            "final": last_selector_snapshot,
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "training_data": TRAINING_DATA_DESCRIPTION,
        }

    generator_metrics: dict[str, Any] | None = None
    if args.generator_steps > 0:
        generator_training_pool = transformer_answer_generator_training_pool(examples)
        generator = build_transformer_answer_generator(
            examples,
            model,
            tokenizer,
            args.seed + 211,
            args.generator_max_answer_chars,
            args.generator_transformer_top_k,
        )
        generator_rng = random.Random(args.seed + 211)
        generator_history_path = args.run / "answer_generator_metrics.jsonl"
        generator_history_writer = JsonlHistoryWriter(generator_history_path)

        def generator_snapshot(step: int, train_loss: float | None) -> dict[str, Any]:
            record = {
                "step": step,
                "train_loss": train_loss,
                "evals": {
                    name: evaluate_answer_generator_records(
                        generator,
                        model,
                        tokenizer,
                        records,
                    )
                    for name, records in sorted(eval_records.items())
                },
            }
            return generator_history_writer.append(record)

        generator_baseline = generator_snapshot(0, None)
        generator_lessons = {
            example: transformer_answer_generator_lesson(
                generator,
                model,
                tokenizer,
                example,
            )
            for example in sorted(
                set(generator_training_pool),
                key=lambda item: (item.prompt, item.target, item.source),
            )
        }
        running_generator_loss = 0.0
        last_generator_snapshot = generator_baseline
        last_generator_snapshot_step = 0
        generator_training_cursor = ShuffledTrainingCursor(
            generator_training_pool,
            generator_rng,
        )
        for generator_step in range(1, args.generator_steps + 1):
            example = generator_training_cursor.next()
            running_generator_loss += train_transformer_answer_generator_lesson(
                generator,
                generator_lessons[example],
                args.generator_learning_rate,
            )
            if (
                args.generator_eval_every > 0
                and generator_step % args.generator_eval_every == 0
            ):
                train_loss = running_generator_loss / args.generator_eval_every
                last_generator_snapshot = generator_snapshot(generator_step, train_loss)
                last_generator_snapshot_step = generator_step
                print(f"generator_step={generator_step} train_loss={train_loss:.4f}")
                running_generator_loss = 0.0

        if last_generator_snapshot_step != args.generator_steps:
            last_generator_snapshot = generator_snapshot(args.generator_steps, None)

        generator_checkpoint_path = args.run / "answer_generator.json"
        generator.save(generator_checkpoint_path)
        generator_metrics = {
            "architecture": "transformer-guided-answer-generator",
            "checkpoint": str(generator_checkpoint_path),
            "history": str(generator_history_path),
            "steps": args.generator_steps,
            "training_examples": len(generator_training_pool),
            "learning_rate": args.generator_learning_rate,
            "generator_eval_every": args.generator_eval_every,
            "max_answer_chars": args.generator_max_answer_chars,
            "transformer_top_k": args.generator_transformer_top_k,
            "labels": len(generator.config.labels),
            "features": len(generator.config.features),
            "baseline": generator_baseline,
            "final": last_generator_snapshot,
            "uses_answer_candidates": False,
            "pretrained_weights": False,
            "pretrained_tokenizer": False,
            "external_embeddings": False,
            "training_data": TRAINING_DATA_DESCRIPTION,
        }

    checkpoint_path = artifacts.checkpoint
    optimizer_path = artifacts.optimizer_state
    save_optimizer_state(optimizer_path, optimizer)
    checkpoint_metadata = transformer_run_metadata(
        args,
        tokenizer,
        optimizer,
        "answer-train",
        resume_metadata,
    )
    model.save(checkpoint_path, tokenizer, checkpoint_metadata)
    tokenizer.save(artifacts.tokenizer)
    metrics = {
        "architecture": TRANSFORMER_ARCHITECTURE,
        "checkpoint": str(checkpoint_path),
        "checkpoint_format": TRANSFORMER_CHECKPOINT_FORMAT,
        "optimizer_state": str(optimizer_path),
        "optimizer": optimizer.summary(),
        "resume": resume_metadata,
        "history": str(history_path),
        "lessons": str(lessons_path),
        "steps": args.steps,
        "examples": len(examples),
        "training_examples": len(training_pool),
        "candidate_count": len(candidates),
        "training_candidate_count": len(training_candidates),
        "candidate_scope": args.candidate_scope,
        "include_completions": args.include_completions,
        "generation_config": asdict(generation_config),
        "target_loss_weight": args.target_loss_weight,
        "choice_loss_weight": args.choice_loss_weight,
        "choice_negatives": args.choice_negatives,
        "choice_max_chars": args.choice_max_chars,
        "vocab_size": tokenizer.vocab_size,
        "context_size": args.context_size,
        "embedding_dim": args.embedding_dim,
        "feedforward_dim": args.feedforward_dim,
        "num_layers": args.num_layers,
        "attention_heads": args.attention_heads,
        "use_layer_norm": args.use_layer_norm,
        "use_pre_layer_norm": args.use_pre_layer_norm,
        "use_rms_norm": args.use_rms_norm,
        "layer_norm_epsilon": args.layer_norm_epsilon,
        "use_gated_mlp": args.use_gated_mlp,
        "tie_output_embeddings": args.tie_output_embeddings,
        "use_rotary_positions": args.use_rotary_positions,
        "use_kv_cache_path": args.use_kv_cache_path,
        "use_context_mean": args.use_context_mean,
        "use_context_projection": args.use_context_projection,
        "use_prompt_prefix_projection": args.use_prompt_prefix_projection,
        "use_prompt_position_projection": args.use_prompt_position_projection,
        "prompt_position_projection_scale": args.prompt_position_projection_scale,
        "use_prompt_attention_summary": args.use_prompt_attention_summary,
        "context_coverage": context_coverage,
        "baseline": baseline,
        "final": last_snapshot,
        "post_direct_candidate_snapshot": post_direct_candidate_snapshot,
        "post_direct_candidate_snapshot_skipped": post_direct_candidate_snapshot_skipped,
        "direct_answer": direct_answer_metrics,
        "answer_selector": selector_metrics,
        "answer_generator": generator_metrics,
        "corpus_hygiene": corpus_hygiene,
        "corpus_hygiene_path": str(hygiene_path),
        "retrieval_memory": {
            "path": str(retrieval_memory_path),
            "summary": retrieval_memory["summary"],
            "memory": retrieval_memory["memory"],
            "dataset_exclusivity": retrieval_memory["dataset_exclusivity"],
            "self_improvement": retrieval_memory["self_improvement"],
        },
        "memory_consolidation_plan_path": str(memory_consolidation_plan_path),
        "training_plan": training_plan,
        "training_plan_path": str(training_plan_path),
        "training_recipe": training_recipe,
        "training_recipe_path": str(training_recipe_path),
        "candidate_quarantine": candidate_quarantine,
        "candidate_quarantine_path": str(candidate_quarantine_path),
        "closed_world_verifier": closed_world_verifier,
        "closed_world_verifier_path": str(verifier_path),
        "constraint_first_promotion_path": str(constraint_first_path),
        "experiment_intent_path": str(experiment_path),
        "metrics_path": str(artifacts.metrics),
        "run_id": args.run.name,
        "pretrained_weights": False,
        "pretrained_tokenizer": False,
        "external_embeddings": False,
        "tokenizer": TRANSFORMER_TOKENIZER,
        "training_data": TRAINING_DATA_DESCRIPTION,
    }
    memory_consolidation_plan = build_memory_consolidation_plan(
        retrieval_memory,
        metrics,
    )
    write_memory_consolidation_plan(
        memory_consolidation_plan_path,
        memory_consolidation_plan,
    )
    metrics["memory_consolidation_plan"] = {
        "path": str(memory_consolidation_plan_path),
        "summary": memory_consolidation_plan["summary"],
        "dataset_exclusivity": memory_consolidation_plan["dataset_exclusivity"],
        "self_improvement": memory_consolidation_plan["self_improvement"],
    }
    metrics["constraint_first_promotion"] = transformer_constraint_report(metrics)
    write_constraint_first_report(
        constraint_first_path,
        metrics["constraint_first_promotion"],
    )
    status, summary, evidence = transformer_experiment_decision(metrics)
    metrics["experiment_intent"] = record_experiment_decision(
        experiment_intent,
        status,
        summary,
        evidence,
    )
    write_experiment_intent(experiment_path, metrics["experiment_intent"])
    with artifacts.metrics.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"saved {checkpoint_path}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "train":
        train_transformer(args)
        return 0
    if args.command == "eval":
        result = eval_transformer(args)
        print(json.dumps(result["evals"], indent=2, sort_keys=True))
        return 0
    if args.command == "answer-train":
        train_transformer_answers(args)
        return 0
    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
