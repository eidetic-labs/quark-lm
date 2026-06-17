import unittest

from transformer_baseline_floor_acceptance_routing import (
    BaselineFloorProfileAcceptanceAttempt,
    record_baseline_floor_profile_attempt_acceptance,
)


ACCOUNTING_ZERO_KEYS = (
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


ACCOUNTING_MAP_KEYS = (
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


SAMPLE_KEYS = (
    "sequential_profile_probe_sample",
    "profile_scale_probe_sample",
    "profile_scale_diversity_probe_sample",
    "profile_scale_frontier_probe_sample",
    "profile_scale_coverage_frontier_probe_sample",
    "profile_scale_coverage_prep_frontier_probe_sample",
    "profile_scale_coverage_recovery_frontier_probe_sample",
    "profile_scale_branch_stable_coverage_recovery_frontier_probe_sample",
    "profile_scale_branch_diversity_recovery_frontier_probe_sample",
    "profile_scale_collapsed_profile_binding_frontier_probe_sample",
    "profile_scale_remaining_profile_binding_probe_sample",
    "profile_scale_owner_paraphrase_binding_probe_sample",
    "profile_scale_memory_consolidation_probe_sample",
    "profile_scale_memory_consolidation_missing_first_token_probe_sample",
)


def empty_guard() -> dict[str, object]:
    guard: dict[str, object] = {key: 0 for key in ACCOUNTING_ZERO_KEYS}
    guard.update({key: {} for key in ACCOUNTING_MAP_KEYS})
    guard.update({key: [] for key in SAMPLE_KEYS})
    return guard


class TransformerBaselineFloorAcceptanceRoutingTest(unittest.TestCase):
    def test_accepted_attempt_records_accounting_and_probe_sample(self) -> None:
        guard = empty_guard()
        coverage_delta = {"improved_profile_count": 1}
        recovery_delta = {"recovered_profile_count": 1}
        branch_delta = {"diversity_gain": 1}
        collapsed_delta = {"collapsed_gain": 1}
        missing_delta = {"target_gain": 1}

        record_baseline_floor_profile_attempt_acceptance(
            guard,
            BaselineFloorProfileAcceptanceAttempt(
                profile="qa:learning",
                records=3,
                frontier_records=2,
                learning_rate_scale=0.5,
                scale_key="0.5",
                remaining_profile_binding_active=True,
                remaining_profile_binding_prioritized=True,
                remaining_profile_binding_target_profiles=["qa:remaining"],
                remaining_profile_binding_source_profiles=["qa:source"],
                owner_paraphrase_binding_active=True,
                owner_paraphrase_binding_prioritized=True,
                owner_paraphrase_binding_target_profiles=["qa:owner"],
                owner_paraphrase_binding_preserved_profiles=["fact:owner"],
                owner_paraphrase_binding_preserved=True,
                owner_paraphrase_binding_preservation_delta={"fact:owner": 0},
                memory_consolidation_active=True,
                memory_consolidation_prioritized=True,
                memory_consolidation_target_profiles=["qa:memory"],
                memory_consolidation_source_plan="memory-plan.json",
                memory_consolidation_collapsed_memory_backed_profiles=["qa:memory"],
                memory_consolidation_remaining_collapsed_active=True,
                memory_consolidation_profile_specific_active=True,
                memory_consolidation_profile_specific_missing_first_token_target_map={
                    "qa:memory": [7]
                },
                diversity_active=True,
                diversity_outcome="improved",
                profile_score=(2.0, 1.0),
                profile_base_score=(1.0, 1.0),
                frontier_active=True,
                coverage_active=True,
                coverage_frontier_active=True,
                coverage_outcome="gained",
                coverage_prep_active=True,
                coverage_prep_accepted=True,
                coverage_delta=coverage_delta,
                coverage_recovery_active=True,
                coverage_recovery_attempted=True,
                coverage_recovery_accepted=True,
                coverage_recovery_outcome="coverage_recovered",
                coverage_recovery_records=1,
                coverage_recovery_learning_rate_scale=0.25,
                coverage_recovery_delta=recovery_delta,
                branch_stable_coverage_recovery_active=True,
                coverage_recovery_branch_stable_checked=True,
                coverage_recovery_branch_stable_accepted=True,
                coverage_recovery_branch_stability_preserved=True,
                coverage_recovery_prepared_score=(1.0, 1.0),
                coverage_recovery_score=(2.0, 1.0),
                branch_diversity_recovery_active=True,
                branch_diversity_recovery_attempted=True,
                branch_diversity_recovery_accepted=True,
                branch_diversity_recovery_outcome="branch_diversity_improved",
                branch_diversity_recovery_learning_rate_scale=0.25,
                branch_diversity_recovery_records=1,
                branch_diversity_recovery_base_score=(1.0, 1.0),
                branch_diversity_recovery_score=(2.0, 1.0),
                branch_diversity_recovery_delta=branch_delta,
                collapsed_profile_binding_active=True,
                collapsed_profile_binding_attempted=True,
                collapsed_profile_binding_accepted=True,
                collapsed_profile_binding_outcome="collapsed_profile_binding",
                collapsed_profile_binding_target_profiles=["qa:collapsed"],
                collapsed_profile_binding_learning_rate_scale=0.25,
                collapsed_profile_binding_records=1,
                collapsed_profile_binding_base_score=(1.0,),
                collapsed_profile_binding_score=(2.0,),
                collapsed_profile_binding_delta=collapsed_delta,
                missing_first_token_active=True,
                missing_first_token_attempted=True,
                missing_first_token_accepted=True,
                missing_first_token_outcome="missing_first_token_coverage",
                missing_first_token_target_profiles=["qa:missing"],
                missing_first_token_target_ids=[7],
                missing_first_token_profile_specific=True,
                missing_first_token_learning_rate_scale=0.25,
                missing_first_token_records=1,
                missing_first_token_base_score=(1.0,),
                missing_first_token_score=(2.0,),
                missing_first_token_delta=missing_delta,
            ),
        )

        self.assertEqual(guard["sequential_profile_acceptances"], 1)
        self.assertEqual(guard["profile_scale_memory_acceptances"], 1)
        self.assertEqual(
            guard["profile_scale_remaining_profile_binding_prioritized_acceptances"],
            1,
        )
        self.assertEqual(
            guard["profile_scale_owner_paraphrase_binding_prioritized_acceptances"],
            1,
        )
        self.assertEqual(
            guard["profile_scale_memory_consolidation_prioritized_acceptances"],
            1,
        )
        self.assertEqual(guard["profile_scale_diversity_score_improvements"], 1)
        self.assertEqual(guard["profile_scale_frontier_acceptances"], 1)
        self.assertEqual(guard["profile_scale_coverage_frontier_gains"], 1)
        self.assertEqual(
            guard["profile_scale_coverage_prep_frontier_gain_acceptances"],
            1,
        )
        self.assertEqual(
            guard["profile_scale_acceptance_scale_counts"],
            {"0.5": 1},
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

        sample = guard["profile_scale_probe_sample"][0]
        self.assertTrue(sample["accepted"])
        self.assertEqual(sample["remaining_profile_binding_target_profiles"], ["qa:remaining"])
        self.assertEqual(sample["memory_consolidation_source_plan"], "memory-plan.json")
        self.assertEqual(sample["coverage_recovery_delta"], recovery_delta)
        self.assertEqual(
            sample["branch_diversity_recovery_outcome"],
            "branch_diversity_improved",
        )
        self.assertEqual(
            sample["collapsed_profile_binding_target_profiles"],
            ["qa:collapsed"],
        )
        self.assertEqual(sample["missing_first_token_target_ids"], [7])

        for key in SAMPLE_KEYS:
            with self.subTest(key=key):
                self.assertEqual(guard[key], [sample])


if __name__ == "__main__":
    unittest.main()
