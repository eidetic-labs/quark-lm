import unittest

from support.baseline_floor_samples import empty_guard
from transformer_baseline_floor_acceptance_samples import (
    BaselineFloorProfileAcceptanceSample,
    record_baseline_floor_profile_acceptance_sample,
)
from transformer_baseline_floor_probe_samples import BaselineFloorProbeSampleStreams
from transformer_baseline_floor_rejection_samples import (
    record_baseline_floor_profile_rejection_sample,
)


class TransformerBaselineFloorProfileSamplesTest(unittest.TestCase):
    def test_profile_rejection_sample_records_counts_and_payload(self) -> None:
        guard = empty_guard()
        guard["sequential_profile_rejection_counts"] = {}
        guard["profile_scale_rejection_scale_counts"] = {}
        coverage_delta = {"improved_profile_count": 0, "regressed_profile_count": 1}

        record_baseline_floor_profile_rejection_sample(
            guard,
            profile="qa:learning",
            records=3,
            frontier_records=2,
            learning_rate_scale=0.5,
            scale_key="0.5",
            diagnostics={
                "worst_violation": {"profile": "qa:learning", "deficit": 1},
                "violating_profile_count": 1,
            },
            diversity_active=True,
            profile_score=(1.0, 2.0),
            profile_base_score=(1.0, 3.0),
            diversity_outcome="regressed",
            diversity_rejection_reason="score_regression",
            coverage_active=True,
            coverage_delta=coverage_delta,
            coverage_outcome="regressed",
            coverage_prep_accepted=False,
            coverage_rejection_reason="coverage_regression",
        )

        self.assertEqual(
            guard["sequential_profile_rejection_counts"],
            {"qa:learning": 1},
        )
        self.assertEqual(guard["profile_scale_rejection_scale_counts"], {"0.5": 1})
        sample = guard["sequential_profile_probe_sample"][0]
        self.assertEqual(sample["profile"], "qa:learning")
        self.assertFalse(sample["accepted"])
        self.assertEqual(sample["records"], 3)
        self.assertEqual(sample["frontier_records"], 2)
        self.assertEqual(sample["learning_rate_scale"], 0.5)
        self.assertEqual(sample["diversity_outcome"], "regressed")
        self.assertEqual(sample["diversity_rejection_reason"], "score_regression")
        self.assertEqual(sample["base_score"], [1.0, 3.0])
        self.assertEqual(sample["candidate_score"], [1.0, 2.0])
        self.assertEqual(sample["coverage_outcome"], "regressed")
        self.assertEqual(sample["coverage_rejection_reason"], "coverage_regression")
        self.assertEqual(sample["coverage_delta"], coverage_delta)

    def test_profile_acceptance_sample_records_optional_payload(self) -> None:
        guard = empty_guard()
        coverage_delta = {"improved_profile_count": 1}
        missing_delta = {"target_gain": 1}

        record_baseline_floor_profile_acceptance_sample(
            guard,
            BaselineFloorProfileAcceptanceSample(
                profile="qa:learning",
                records=3,
                frontier_records=2,
                learning_rate_scale=0.5,
                streams=BaselineFloorProbeSampleStreams(
                    remaining_profile_binding=True,
                    owner_paraphrase_binding=True,
                    memory_consolidation=True,
                    missing_first_token=True,
                ),
                remaining_profile_binding_active=True,
                remaining_profile_binding_prioritized=True,
                remaining_profile_binding_target_profiles=["learning"],
                remaining_profile_binding_source_profiles=["qa:learning"],
                owner_paraphrase_binding_active=True,
                owner_paraphrase_binding_prioritized=True,
                owner_paraphrase_binding_target_profiles=["owner"],
                owner_paraphrase_binding_preserved_profiles=["fact:owner"],
                owner_paraphrase_binding_preserved=True,
                owner_paraphrase_binding_preservation_delta={"fact:owner": 0},
                memory_consolidation_active=True,
                memory_consolidation_prioritized=True,
                memory_consolidation_target_profiles=["owner"],
                memory_consolidation_source_plan="source-plan.json",
                memory_consolidation_collapsed_memory_backed_profiles=["owner"],
                memory_consolidation_remaining_collapsed_active=True,
                memory_consolidation_profile_specific_active=True,
                memory_consolidation_profile_specific_missing_first_token_target_map={
                    "owner": [7]
                },
                diversity_active=True,
                diversity_outcome="improved",
                profile_score=(2.0, 1.0),
                profile_base_score=(1.0, 1.0),
                coverage_active=True,
                coverage_outcome="gained",
                coverage_prep_accepted=True,
                coverage_delta=coverage_delta,
                missing_first_token_active=True,
                missing_first_token_attempted=True,
                missing_first_token_accepted=True,
                missing_first_token_outcome="missing_first_token_coverage",
                missing_first_token_target_profiles=["owner"],
                missing_first_token_target_ids=[7],
                missing_first_token_profile_specific=True,
                missing_first_token_learning_rate_scale=0.25,
                missing_first_token_records=1,
                missing_first_token_base_score=(1.0,),
                missing_first_token_score=(2.0,),
                missing_first_token_delta=missing_delta,
            ),
        )

        sample = guard["sequential_profile_probe_sample"][0]
        self.assertTrue(sample["accepted"])
        self.assertEqual(
            sample["remaining_profile_binding_target_profiles"],
            ["learning"],
        )
        self.assertTrue(sample["owner_paraphrase_binding_preserved"])
        self.assertEqual(sample["memory_consolidation_source_plan"], "source-plan.json")
        self.assertEqual(sample["base_score"], [1.0, 1.0])
        self.assertEqual(sample["candidate_score"], [2.0, 1.0])
        self.assertEqual(sample["coverage_delta"], coverage_delta)
        self.assertEqual(sample["missing_first_token_target_ids"], [7])
        self.assertEqual(sample["missing_first_token_delta"], missing_delta)
        self.assertEqual(
            guard[
                "profile_scale_memory_consolidation_missing_first_token_probe_sample"
            ],
            [sample],
        )


if __name__ == "__main__":
    unittest.main()
