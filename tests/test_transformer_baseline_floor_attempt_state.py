import unittest
from types import SimpleNamespace

from transformer_baseline_floor_attempt_recording import (
    BaselineFloorProfileAcceptanceContext,
    baseline_floor_profile_acceptance_attempt,
)
from transformer_baseline_floor_attempt_state import (
    BaselineFloorProfileAttemptState,
)
from transformer_baseline_floor_owner_preservation import (
    check_owner_paraphrase_binding_preservation,
)
from transformer_baseline_floor_profile_outcome_types import BaselineFloorProfileOutcome


class TransformerBaselineFloorAttemptStateTest(unittest.TestCase):
    def test_attempt_state_collects_recovery_and_binding_outcomes(self) -> None:
        state = BaselineFloorProfileAttemptState.from_profile_outcome(
            BaselineFloorProfileOutcome(
                floor_preserved=True,
                diversity_outcome="tied",
                diversity_rejection_reason="",
                profile_score=(1.0, 1.0),
                coverage_outcome="tied",
                coverage_rejection_reason="",
                coverage_delta={"base": 0},
                coverage_prep_accepted=True,
            ),
            {"probe": "base"},
            (1.0, 0.5),
        )

        state.apply_coverage_recovery(
            SimpleNamespace(
                floor_preserved=True,
                profile_probe_snapshot={"probe": "coverage"},
                profile_score=(2.0, 1.0),
                diversity_outcome="improved",
                diversity_rejection_reason="",
                coverage_delta={"coverage": 1},
                coverage_outcome="gained",
                coverage_rejection_reason="",
                coverage_prep_accepted=True,
                attempted=True,
                accepted=True,
                outcome="coverage_recovered",
                rejection_reason="",
                learning_rate_scale=0.25,
                records=2,
                delta={"recovered": 1},
                prepared_score=(1.0, 1.0),
                score=(2.0, 1.0),
                branch_stable_checked=True,
                branch_stable_accepted=True,
                branch_stability_preserved=True,
            )
        )
        state.apply_branch_diversity_recovery(
            SimpleNamespace(
                floor_preserved=True,
                profile_probe_snapshot={"probe": "branch"},
                profile_score=(3.0, 1.0),
                diversity_outcome="improved",
                diversity_rejection_reason="",
                attempted=True,
                accepted=True,
                outcome="branch_diversity_improved",
                rejection_reason="",
                learning_rate_scale=0.125,
                records=1,
                base_score=(2.0, 1.0),
                score=(3.0, 1.0),
                delta={"branch": 1},
            )
        )
        state.apply_collapsed_profile_binding(
            SimpleNamespace(
                floor_preserved=True,
                profile_probe_snapshot={"probe": "binding"},
                profile_score=(4.0, 1.0),
                diversity_outcome="improved",
                diversity_rejection_reason="",
                owner_paraphrase_binding_preservation_delta={"owner": 0},
                attempted=True,
                accepted=True,
                outcome="collapsed_profile_binding",
                rejection_reason="",
                learning_rate_scale=0.125,
                records=1,
                target_profiles=["qa:collapsed"],
                base_score=(3.0, 1.0),
                score=(4.0, 1.0),
                delta={"collapsed": 1},
            )
        )
        state.apply_missing_first_token(
            SimpleNamespace(
                floor_preserved=True,
                profile_probe_snapshot={"probe": "missing"},
                profile_score=(5.0, 1.0),
                diversity_outcome="improved",
                diversity_rejection_reason="",
                coverage_delta={"missing": 1},
                coverage_outcome="gained",
                coverage_rejection_reason="",
                attempted=True,
                accepted=True,
                outcome="missing_first_token_coverage",
                rejection_reason="",
                learning_rate_scale=0.125,
                records=1,
                target_profiles=["qa:missing"],
                target_ids=[7],
                base_score=(4.0, 1.0),
                score=(5.0, 1.0),
                delta={"target_gain": 1},
            )
        )

        attempt = baseline_floor_profile_acceptance_attempt(
            state,
            BaselineFloorProfileAcceptanceContext(
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
                frontier_active=True,
                coverage_active=True,
                coverage_frontier_active=True,
                coverage_prep_active=True,
                coverage_recovery_active=True,
                branch_stable_coverage_recovery_active=True,
                branch_diversity_recovery_active=True,
                collapsed_profile_binding_active=True,
                missing_first_token_active=True,
                missing_first_token_profile_specific=True,
            ),
        )

        self.assertEqual(attempt.profile_base_score, (1.0, 0.5))
        self.assertEqual(attempt.coverage_recovery_delta, {"recovered": 1})
        self.assertEqual(
            attempt.branch_diversity_recovery_outcome,
            "branch_diversity_improved",
        )
        self.assertEqual(
            attempt.collapsed_profile_binding_target_profiles,
            ["qa:collapsed"],
        )
        self.assertEqual(attempt.missing_first_token_target_ids, [7])

    def test_owner_paraphrase_preservation_records_regression(self) -> None:
        guard = {
            "profile_scale_owner_paraphrase_binding_preservation_checks": 0,
            "profile_scale_owner_paraphrase_binding_preservation_failures": 0,
        }

        preservation = check_owner_paraphrase_binding_preservation(
            active=True,
            update_guard=guard,
            profile_probe_snapshot={"probe": "candidate"},
            profile_base_snapshot={"probe": "base"},
            preserved_profiles=["fact:owner"],
            profile_diversity_delta=lambda *_: {"regressed_profile_count": 1},
        )

        self.assertFalse(preservation.preserved)
        self.assertEqual(
            preservation.rejection_reason,
            "owner_paraphrase_preservation_regression",
        )
        self.assertEqual(
            guard["profile_scale_owner_paraphrase_binding_preservation_checks"],
            1,
        )
        self.assertEqual(
            guard["profile_scale_owner_paraphrase_binding_preservation_failures"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
