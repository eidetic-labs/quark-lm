from __future__ import annotations


def branch_recovery_guard() -> dict[str, object]:
    return {
        "profile_scale_branch_diversity_recovery_frontier_candidates": 0,
        "profile_scale_branch_diversity_recovery_frontier_attempts": 0,
        "profile_scale_branch_diversity_recovery_frontier_records": 0,
        "profile_scale_branch_diversity_recovery_frontier_acceptances": 0,
        "profile_scale_branch_diversity_recovery_frontier_fallback_acceptances": 0,
        "profile_scale_branch_diversity_recovery_frontier_rejections": 0,
        "profile_scale_branch_diversity_recovery_frontier_rejection_reasons": {},
    }


def coverage_recovery_guard() -> dict[str, object]:
    return {
        "profile_scale_coverage_recovery_frontier_prepared_candidates": 0,
        "profile_scale_coverage_recovery_frontier_attempts": 0,
        "profile_scale_coverage_recovery_frontier_records": 0,
        "profile_scale_coverage_recovery_frontier_acceptances": 0,
        "profile_scale_coverage_recovery_frontier_rejections": 0,
        "profile_scale_coverage_recovery_frontier_rejection_reasons": {},
        "profile_scale_branch_stable_coverage_recovery_frontier_checks": 0,
        "profile_scale_branch_stable_coverage_recovery_frontier_acceptances": 0,
        "profile_scale_branch_stable_coverage_recovery_frontier_rejections": 0,
        "profile_scale_branch_stable_coverage_recovery_frontier_rejection_reasons": {},
    }


class FakeModel:
    def to_dict(self, _tokenizer: object) -> dict[str, object]:
        return {"model": "candidate"}


class FakeOptimizer:
    def to_dict(self) -> dict[str, object]:
        return {"optimizer": "candidate"}


class FakeSnapshotRecorder:
    def __init__(self) -> None:
        self.metadata: list[dict[str, object]] = []

    def record(
        self,
        _step: int,
        _loss: object,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        self.metadata.append(metadata)
        return {"snapshot": len(self.metadata)}
