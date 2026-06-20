"""Experiment bundle contract for profile-balanced routing repair."""

from __future__ import annotations

from typing import Any


PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE = "profile-balanced-routing-repair"
PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE = (
    "profile-balanced-rank-routing-repair"
)
PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE = (
    "profile-balanced-retention-rank-routing-repair"
)
PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE = (
    "profile-balanced-topk-routing-repair"
)
PROFILE_BALANCED_ROUTING_REPAIR_MODE = (
    "branch-hidden-projection-margin-unlikelihood"
)
PROFILE_BALANCED_RANK_ROUTING_REPAIR_MODE = (
    "branch-profile-balanced-rank-margin-unlikelihood"
)
PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_MODE = (
    "branch-profile-balanced-retention-rank-margin-unlikelihood"
)
PROFILE_BALANCED_TOPK_ROUTING_REPAIR_MODE = (
    "branch-profile-balanced-topk-softmax-unlikelihood"
)
EXPERIMENT_BUNDLES = (
    PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE,
    PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE,
)

_BUNDLE_MODES = {
    PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE: PROFILE_BALANCED_ROUTING_REPAIR_MODE,
    PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE: (
        PROFILE_BALANCED_RANK_ROUTING_REPAIR_MODE
    ),
    PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE: (
        PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_MODE
    ),
    PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE: (
        PROFILE_BALANCED_TOPK_ROUTING_REPAIR_MODE
    ),
}


def routing_repair_bundle_mode(bundle: str | None) -> str | None:
    return _BUNDLE_MODES.get(bundle)


def routing_repair_bundle_supports_mode(
    bundle: str | None,
    mode: str | None,
) -> bool:
    expected = routing_repair_bundle_mode(bundle)
    return expected is not None and mode == expected


def routing_repair_bundle_hypothesis(bundle: str | None) -> str | None:
    """Return the default hypothesis for a declared routing-repair bundle."""

    if bundle != PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE:
        if bundle == PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE:
            return (
                "Profile-balanced rank-margin pressure with training-family "
                "retention anchors can improve residual branch routing while "
                "preserving already represented profile targets."
            )
        if bundle == PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE:
            return (
                "Profile-balanced top-k softmax pressure can convert lifted "
                "branch targets into stronger target-token coverage under "
                "coverage-preserving acceptance gates."
            )
        if bundle != PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE:
            return None
        return (
            "Profile-balanced hard-negative rank-margin pressure can improve "
            "branch routing when applied across failed profile families with "
            "coverage-preserving acceptance gates."
        )
    return (
        "Hidden-projection margin pressure can improve branch routing only when "
        "applied across profile-balanced replay with representation-separation "
        "evidence and coverage-preserving acceptance gates."
    )


def routing_repair_bundle_gates(bundle: str | None) -> list[dict[str, Any]]:
    """Return required acceptance gates for a declared routing-repair bundle."""

    if bundle == PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE:
        return _pressure_repair_gates(
            _gate(
                "rank_margin_pressure",
                (
                    "Apply profile-balanced hard-negative rank-margin pressure "
                    "across failed profile-family branch targets."
                ),
            ),
            _gate(
                "rank_pressure_requires_branch_response",
                (
                    "Reject rank-margin movement when target coverage and target-rank "
                    "branch-diversity score remain unchanged."
                ),
            ),
        )
    if bundle == PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE:
        return _pressure_repair_gates(
            _gate(
                "retention_rank_margin_pressure",
                (
                    "Apply profile-balanced hard-negative rank-margin pressure "
                    "with training-family retention anchors."
                ),
            ),
            _gate(
                "retention_anchors_recorded",
                (
                    "Record retention-anchor attempts that preserve already "
                    "represented training-family target tokens."
                ),
            ),
            _gate(
                "retention_rank_pressure_requires_branch_response",
                (
                    "Reject retention-anchored rank movement when target coverage "
                    "and target-rank branch-diversity score remain unchanged."
                ),
            ),
        )
    if bundle == PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE:
        return _pressure_repair_gates(
            _gate(
                "topk_softmax_pressure",
                (
                    "Apply profile-balanced hard-candidate top-k softmax pressure "
                    "across failed profile-family branch targets."
                ),
            ),
            _gate(
                "topk_pressure_requires_branch_response",
                (
                    "Reject top-k softmax movement when target coverage and "
                    "target-rank branch-diversity score remain unchanged."
                ),
            ),
        )
    if bundle != PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE:
        return []
    return [
        _gate(
            "profile_balanced_branch_batches",
            (
                "Record branch batches that cover failing multi-target profiles, "
                "zero-coverage profiles, and buried-target profiles before updates."
            ),
        ),
        _gate(
            "hidden_projection_margin_pressure",
            (
                "Apply or explicitly report hidden-projection target-margin pressure "
                "across branch targets."
            ),
        ),
        _gate(
            "representation_separation_evidence",
            (
                "Record target centroid distances and centroid margins for branch "
                "representations before accepting the repair."
            ),
        ),
        _gate(
            "coverage_preserving_update_guard",
            (
                "Reject updates that improve local scores while dropping any "
                "profile below baseline target-token coverage."
            ),
        ),
        _gate(
            "branch_diversity_acceptance_gate",
            (
                "Accept the screen only when branch-diversity evidence improves "
                "without retention, leakage, unknown-policy, or coverage regression."
            ),
        ),
        _gate(
            "hidden_advantage_requires_coverage_response",
            (
                "Reject hidden-advantage movement when target-token coverage remains "
                "unchanged for the same failed profiles."
            ),
        ),
    ]


def routing_repair_bundle_failure_criteria(bundle: str | None) -> list[str]:
    """Return bundle-specific early rejection criteria."""

    if bundle == PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE:
        return [
            "Any profile drops below baseline target-token coverage.",
            "Dominant predicted rate remains 1.0 across all measured profiles.",
            (
                "Rank-margin pressure produces no target-rank, top-k, or "
                "coverage response."
            ),
            "Representation centroid distance or margin fails to improve.",
        ]
    if bundle == PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE:
        return [
            "Any profile drops below baseline target-token coverage.",
            "Dominant predicted rate remains 1.0 across all measured profiles.",
            (
                "Retention-anchored rank pressure produces no target-rank, "
                "top-k, or coverage response."
            ),
            "Retention anchors fail to record preservation attempts.",
            "Representation centroid distance or margin fails to improve.",
        ]
    if bundle == PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE:
        return [
            "Any profile drops below baseline target-token coverage.",
            "Dominant predicted rate remains 1.0 across all measured profiles.",
            (
                "Top-k softmax pressure produces no target-rank, top-k, or "
                "coverage response."
            ),
            "Representation centroid distance or margin fails to improve.",
        ]
    if bundle != PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE:
        return []
    return [
        "Any profile drops below baseline target-token coverage.",
        "Dominant predicted rate remains 1.0 across all measured profiles.",
        "Hidden advantage improves while target-token coverage remains unchanged.",
        "Representation centroid distance or margin fails to improve.",
    ]


def routing_repair_bundle_notes(bundle: str | None) -> list[str]:
    """Return explanatory notes for the experiment intent."""

    if bundle == PROFILE_BALANCED_RANK_ROUTING_REPAIR_BUNDLE:
        return [
            "Experiment bundle: Bundle B, profile-balanced rank routing repair.",
            (
                "This bundle is a planned gate contract; it does not promote "
                "transformer language-model behavior."
            ),
        ]
    if bundle == PROFILE_BALANCED_RETENTION_RANK_ROUTING_REPAIR_BUNDLE:
        return [
            (
                "Experiment bundle: Bundle D, retention-anchored "
                "profile-balanced rank routing repair."
            ),
            (
                "This bundle is a planned gate contract; it does not promote "
                "transformer language-model behavior."
            ),
        ]
    if bundle == PROFILE_BALANCED_TOPK_ROUTING_REPAIR_BUNDLE:
        return [
            "Experiment bundle: Bundle C, profile-balanced top-k routing repair.",
            (
                "This bundle is a planned gate contract; it does not promote "
                "transformer language-model behavior."
            ),
        ]
    if bundle != PROFILE_BALANCED_ROUTING_REPAIR_BUNDLE:
        return []
    return [
        "Experiment bundle: Bundle A, profile-balanced routing repair.",
        (
            "This bundle is a planned gate contract; it does not promote "
            "transformer language-model behavior."
        ),
    ]


def _gate(name: str, rule: str) -> dict[str, Any]:
    return {"name": name, "rule": rule, "required": True}


def _pressure_repair_gates(
    pressure_gate: dict[str, Any],
    *extra_gates: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        _gate(
            "profile_balanced_branch_batches",
            (
                "Record branch batches that cover failing multi-target profiles, "
                "zero-coverage profiles, and buried-target profiles before updates."
            ),
        ),
        pressure_gate,
        _gate(
            "representation_separation_evidence",
            (
                "Record target centroid distances and centroid margins for branch "
                "representations before accepting the repair."
            ),
        ),
        *extra_gates[:-1],
        _gate(
            "coverage_preserving_update_guard",
            (
                "Reject updates that improve local scores while dropping any "
                "profile below baseline target-token coverage."
            ),
        ),
        _gate(
            "branch_diversity_acceptance_gate",
            (
                "Accept the screen only when branch-diversity evidence improves "
                "without retention, leakage, unknown-policy, or coverage regression."
            ),
        ),
        *extra_gates[-1:],
    ]
