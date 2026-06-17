"""Static profile-scale experiment gate catalog."""

from __future__ import annotations

from typing import Any

from transformer_experiment_modes import (
    PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_MODE,
    PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_MODE,
    PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_MODE,
    PROFILE_SCALE_COVERAGE_FRONTIER_MODE,
    PROFILE_SCALE_COVERAGE_PREP_FRONTIER_MODE,
    PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_MODE,
    PROFILE_SCALE_DIVERSITY_MODE,
    PROFILE_SCALE_FRONTIER_MODE,
    PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_MODE,
    PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_MODE,
)
from transformer_profile_scale_gate_common import required_gate


PROFILE_SCALE_GATE_SPECS: dict[str, tuple[str, str]] = {
    PROFILE_SCALE_DIVERSITY_MODE: (
        "baseline_floor_profile_scale_diversity_calibrated_"
        "sequential_stabilization_screen",
        "Run records diversity-aware profile-scale search activation, "
        "outer/search scales, source-profile scale attempts, accepted/rejected "
        "diversity outcome counts, score regression rejections, floor "
        "regression rejections, accepted profile scales, update-shape counts, "
        "replay plan, branch-context gate, coverage floor, diversity target, "
        "recipe, verifier, and constraint-first promotion artifacts.",
    ),
    PROFILE_SCALE_FRONTIER_MODE: (
        "baseline_floor_profile_scale_frontier_calibrated_"
        "sequential_stabilization_screen",
        "Run records frontier target-anchor search activation, outer/search "
        "scales, source-profile scale attempts, frontier anchor counts, "
        "accepted/rejected diversity outcome counts, score regression "
        "rejections, floor regression rejections, accepted profile scales, "
        "update-shape counts, replay plan, branch-context gate, coverage floor, "
        "diversity target, recipe, verifier, and constraint-first promotion "
        "artifacts.",
    ),
    PROFILE_SCALE_COVERAGE_FRONTIER_MODE: (
        "baseline_floor_profile_scale_coverage_frontier_calibrated_"
        "sequential_stabilization_screen",
        "Run records coverage-frontier target-anchor search activation, "
        "outer/search scales, source-profile scale attempts, frontier anchor "
        "counts, coverage gain/tie/regression counts, coverage rejection "
        "reasons, accepted coverage deltas, diversity outcome counts, floor "
        "regression rejections, accepted profile scales, update-shape counts, "
        "replay plan, branch-context gate, coverage floor, diversity target, "
        "recipe, verifier, and constraint-first promotion artifacts.",
    ),
    PROFILE_SCALE_COVERAGE_PREP_FRONTIER_MODE: (
        "baseline_floor_profile_scale_coverage_prep_frontier_calibrated_"
        "sequential_stabilization_screen",
        "Run records coverage-prep frontier activation, outer/search scales, "
        "source-profile scale attempts, frontier anchor counts, coverage "
        "gain/tie/regression counts, coverage-preparation acceptances, coverage "
        "rejection reasons, accepted preparation outcomes, diversity outcome "
        "counts, floor regression rejections, accepted profile scales, "
        "update-shape counts, replay plan, branch-context gate, coverage floor, "
        "diversity target, recipe, verifier, and constraint-first promotion "
        "artifacts.",
    ),
    PROFILE_SCALE_COVERAGE_RECOVERY_FRONTIER_MODE: (
        "baseline_floor_profile_scale_coverage_recovery_frontier_calibrated_"
        "sequential_stabilization_screen",
        "Run records coverage-recovery frontier activation, outer/search scales, "
        "source-profile scale attempts, frontier anchor counts, coverage-"
        "preparation candidates, recovery retry scales, recovery attempts, "
        "recovery acceptances, fallback preparations, recovery rejection "
        "reasons, accepted recovery outcomes, coverage gain/tie/regression "
        "counts, diversity outcome counts, floor regression rejections, accepted "
        "profile scales, update-shape counts, replay plan, branch-context gate, "
        "coverage floor, diversity target, recipe, verifier, and "
        "constraint-first promotion artifacts.",
    ),
    PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_MODE: (
        "baseline_floor_profile_scale_branch_stable_coverage_recovery_frontier_"
        "calibrated_sequential_stabilization_screen",
        "Run records branch-stable coverage-recovery frontier activation, "
        "outer/search scales, source-profile scale attempts, frontier anchor "
        "counts, coverage-preparation candidates, recovery retry scales, "
        "recovery attempts, branch-stability checks, branch-stable recovery "
        "acceptances, fallback preparations, branch-stability rejection reasons, "
        "accepted branch-stable outcomes, coverage gain/tie/regression counts, "
        "diversity outcome counts, floor regression rejections, accepted profile "
        "scales, update-shape counts, replay plan, branch-context gate, coverage "
        "floor, diversity target, recipe, verifier, and constraint-first "
        "promotion artifacts.",
    ),
    PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_MODE: (
        "baseline_floor_profile_scale_branch_diversity_recovery_frontier_"
        "calibrated_sequential_stabilization_screen",
        "Run records branch-diversity recovery frontier activation, outer/search "
        "scales, source-profile scale attempts, frontier anchor counts, "
        "coverage-preparation candidates, recovery retry scales, "
        "branch-stability checks, branch-stable recovery acceptances, "
        "branch-diversity recovery candidates, attempts, acceptances, fallback "
        "acceptances, rejection reasons, score deltas, accepted branch-diversity "
        "outcomes, coverage gain/tie/regression counts, diversity outcome "
        "counts, floor regression rejections, accepted profile scales, "
        "update-shape counts, replay plan, branch-context gate, coverage floor, "
        "diversity target, recipe, verifier, and constraint-first promotion "
        "artifacts.",
    ),
    PROFILE_SCALE_COLLAPSED_PROFILE_BINDING_FRONTIER_MODE: (
        "baseline_floor_profile_scale_collapsed_profile_binding_frontier_"
        "calibrated_sequential_stabilization_screen",
        "Run records collapsed-profile binding frontier activation, outer/search "
        "scales, source-profile scale attempts, frontier anchor counts, "
        "coverage-preparation candidates, recovery retry scales, "
        "branch-stability checks, branch-stable recovery acceptances, "
        "branch-diversity recovery candidates, collapsed-profile binding "
        "candidates, attempts, acceptances, fallback acceptances, rejection "
        "reasons, target collapsed profiles, profile-diversity deltas, score "
        "deltas, coverage gain/tie/regression counts, diversity outcome counts, "
        "floor regression rejections, accepted profile scales, update-shape "
        "counts, replay plan, branch-context gate, coverage floor, diversity "
        "target, recipe, verifier, and constraint-first promotion artifacts.",
    ),
    PROFILE_SCALE_REMAINING_PROFILE_BINDING_FRONTIER_MODE: (
        "baseline_floor_profile_scale_remaining_profile_binding_frontier_"
        "calibrated_sequential_stabilization_screen",
        "Run records remaining-profile binding frontier activation, target eval "
        "profiles, prioritized source labels and source profiles, prioritized "
        "attempts, prioritized acceptances and rejections, collapsed-profile "
        "binding candidates, attempts, acceptances, fallback acceptances, "
        "rejection reasons, profile-diversity deltas, update-shape counts, "
        "replay plan, branch-context gate, coverage floor, diversity target, "
        "recipe, verifier, and constraint-first promotion artifacts.",
    ),
    PROFILE_SCALE_OWNER_PARAPHRASE_BINDING_FRONTIER_MODE: (
        "baseline_floor_profile_scale_owner_paraphrase_binding_frontier_"
        "calibrated_sequential_stabilization_screen",
        "Run records owner/paraphrase binding frontier activation, residual "
        "target eval profiles, preserved learning profile, prioritized source "
        "labels and source profiles, prioritized attempts, prioritized "
        "acceptances and rejections, preservation checks, preservation failures, "
        "collapsed-profile binding candidates, attempts, acceptances, fallback "
        "acceptances, rejection reasons, profile-diversity deltas, update-shape "
        "counts, replay plan, branch-context gate, coverage floor, diversity "
        "target, recipe, verifier, and constraint-first promotion artifacts.",
    ),
}


def profile_scale_catalog_gate(direct_answer_mode: str) -> dict[str, Any] | None:
    spec = PROFILE_SCALE_GATE_SPECS.get(direct_answer_mode)
    if spec is None:
        return None
    name, rule = spec
    return required_gate(name, rule)
