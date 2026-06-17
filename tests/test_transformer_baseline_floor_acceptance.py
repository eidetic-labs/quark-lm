import unittest

from transformer_baseline_floor_acceptance import (
    BaselineFloorProfileAcceptanceAccounting,
    record_baseline_floor_profile_acceptance,
)


ZERO_KEYS = (
    "sequential_profile_acceptances",
    "profile_scale_memory_acceptances",
    "profile_scale_remaining_profile_binding_prioritized_acceptances",
    "profile_scale_owner_paraphrase_binding_prioritized_acceptances",
    "profile_scale_memory_consolidation_prioritized_acceptances",
    "profile_scale_diversity_acceptances",
    "profile_scale_diversity_score_improvements",
    "profile_scale_diversity_score_ties",
    "profile_scale_frontier_acceptances",
    "profile_scale_coverage_frontier_acceptances",
    "profile_scale_coverage_frontier_gains",
    "profile_scale_coverage_frontier_ties",
    "profile_scale_coverage_prep_frontier_acceptances",
    "profile_scale_coverage_prep_frontier_gain_acceptances",
    "profile_scale_coverage_prep_frontier_preparations",
    "profile_scale_coverage_recovery_frontier_fallback_preparations",
    "profile_scale_branch_stable_coverage_recovery_frontier_fallback_preparations",
)


MAP_KEYS = (
    "sequential_profile_acceptance_counts",
    "profile_scale_acceptance_scale_counts",
    "profile_scale_profile_acceptance_scales",
    "profile_scale_diversity_profile_acceptance_outcomes",
    "profile_scale_coverage_frontier_profile_acceptance_deltas",
    "profile_scale_coverage_prep_frontier_profile_acceptance_outcomes",
    "profile_scale_coverage_recovery_frontier_profile_acceptance_outcomes",
    "profile_scale_branch_stable_coverage_recovery_frontier_profile_acceptance_outcomes",
    "profile_scale_branch_diversity_recovery_frontier_profile_acceptance_outcomes",
    "profile_scale_branch_diversity_recovery_frontier_profile_score_deltas",
    "profile_scale_collapsed_profile_binding_frontier_profile_acceptance_outcomes",
    "profile_scale_collapsed_profile_binding_frontier_profile_deltas",
    "profile_scale_memory_consolidation_missing_first_token_profile_acceptance_outcomes",
    "profile_scale_memory_consolidation_missing_first_token_profile_deltas",
)


def empty_guard() -> dict[str, object]:
    guard: dict[str, object] = {key: 0 for key in ZERO_KEYS}
    guard.update({key: {} for key in MAP_KEYS})
    return guard


class TransformerBaselineFloorAcceptanceTest(unittest.TestCase):
    def test_acceptance_accounting_records_counters_and_maps(self) -> None:
        guard = empty_guard()
        coverage_delta = {"improved_profile_count": 1}

        record_baseline_floor_profile_acceptance(
            guard,
            BaselineFloorProfileAcceptanceAccounting(
                profile="qa:learning",
                scale_key="0.5",
                remaining_profile_binding_prioritized=True,
                owner_paraphrase_binding_prioritized=True,
                memory_consolidation_prioritized=True,
                diversity_active=True,
                diversity_outcome="improved",
                frontier_active=True,
                coverage_frontier_active=True,
                coverage_outcome="gained",
                coverage_delta=coverage_delta,
                coverage_prep_active=True,
            ),
        )

        self.assertEqual(guard["sequential_profile_acceptances"], 1)
        self.assertEqual(guard["profile_scale_memory_acceptances"], 1)
        self.assertEqual(
            guard["profile_scale_remaining_profile_binding_prioritized_acceptances"],
            1,
        )
        self.assertEqual(guard["profile_scale_diversity_score_improvements"], 1)
        self.assertEqual(guard["profile_scale_frontier_acceptances"], 1)
        self.assertEqual(guard["profile_scale_coverage_frontier_gains"], 1)
        self.assertEqual(
            guard["sequential_profile_acceptance_counts"],
            {"qa:learning": 1},
        )
        self.assertEqual(guard["profile_scale_acceptance_scale_counts"], {"0.5": 1})
        self.assertEqual(
            guard["profile_scale_profile_acceptance_scales"],
            {"qa:learning": "0.5"},
        )
        self.assertEqual(
            guard["profile_scale_coverage_frontier_profile_acceptance_deltas"],
            {"qa:learning": coverage_delta},
        )
        self.assertEqual(
            guard["profile_scale_coverage_prep_frontier_profile_acceptance_outcomes"],
            {"qa:learning": "coverage_gain"},
        )

    def test_acceptance_accounting_records_recovery_fallbacks(self) -> None:
        guard = empty_guard()

        record_baseline_floor_profile_acceptance(
            guard,
            BaselineFloorProfileAcceptanceAccounting(
                profile="qa:learning",
                scale_key="1",
                coverage_outcome="tied",
                coverage_prep_active=True,
                coverage_prep_accepted=True,
                coverage_recovery_active=True,
                coverage_recovery_attempted=True,
                coverage_recovery_accepted=False,
                branch_stable_coverage_recovery_active=True,
            ),
        )

        self.assertEqual(
            guard["profile_scale_coverage_recovery_frontier_fallback_preparations"],
            1,
        )
        self.assertEqual(
            guard[
                "profile_scale_branch_stable_coverage_recovery_frontier_fallback_preparations"
            ],
            1,
        )
        self.assertEqual(
            guard["profile_scale_coverage_recovery_frontier_profile_acceptance_outcomes"],
            {"qa:learning": "coverage_preparation_fallback"},
        )
        self.assertEqual(
            guard[
                "profile_scale_branch_stable_coverage_recovery_frontier_profile_acceptance_outcomes"
            ],
            {"qa:learning": "branch_stable_preparation_fallback"},
        )

    def test_acceptance_accounting_records_specialized_deltas(self) -> None:
        guard = empty_guard()

        record_baseline_floor_profile_acceptance(
            guard,
            BaselineFloorProfileAcceptanceAccounting(
                profile="qa:learning",
                scale_key="1",
                branch_diversity_recovery_active=True,
                branch_diversity_recovery_attempted=True,
                branch_diversity_recovery_accepted=True,
                branch_diversity_recovery_base_score=(1.0, 1.0),
                branch_diversity_recovery_score=(2.0, 1.0),
                branch_diversity_recovery_outcome="improved",
                collapsed_profile_binding_active=True,
                collapsed_profile_binding_accepted=True,
                collapsed_profile_binding_target_profiles=["owner"],
                collapsed_profile_binding_base_score=(1.0,),
                collapsed_profile_binding_score=(2.0,),
                collapsed_profile_binding_delta={"owner": 1},
                collapsed_profile_binding_outcome="binding_gain",
                missing_first_token_active=True,
                missing_first_token_accepted=True,
                missing_first_token_target_profiles=["owner"],
                missing_first_token_target_ids=[7],
                missing_first_token_base_score=(1.0,),
                missing_first_token_score=(2.0,),
                missing_first_token_delta={"owner": 1},
                missing_first_token_outcome="missing_first_token_coverage",
            ),
        )

        self.assertEqual(
            guard[
                "profile_scale_branch_diversity_recovery_frontier_profile_acceptance_outcomes"
            ],
            {"qa:learning": "branch_diversity_recovery"},
        )
        self.assertEqual(
            guard[
                "profile_scale_collapsed_profile_binding_frontier_profile_acceptance_outcomes"
            ],
            {"qa:learning": "collapsed_profile_binding"},
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_profile_acceptance_outcomes"
            ],
            {"qa:learning": "missing_first_token_coverage"},
        )
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_profile_deltas"
            ]["qa:learning"]["target_ids"],
            [7],
        )


if __name__ == "__main__":
    unittest.main()
