import unittest

from support.baseline_floor_samples import empty_guard
from transformer_baseline_floor_sequential_samples import (
    record_baseline_floor_sequential_profile_probe_result,
)


class TransformerBaselineFloorSequentialSamplesTest(unittest.TestCase):
    def test_sequential_profile_probe_result_records_acceptance(self) -> None:
        guard = empty_guard()
        guard["sequential_profile_acceptances"] = 0
        guard["sequential_profile_rejections"] = 0
        guard["sequential_profile_acceptance_counts"] = {}
        guard["sequential_profile_rejection_counts"] = {}

        record_baseline_floor_sequential_profile_probe_result(
            guard,
            profile="qa:learning",
            accepted=True,
            records=3,
        )

        self.assertEqual(guard["sequential_profile_acceptances"], 1)
        self.assertEqual(guard["sequential_profile_rejections"], 0)
        self.assertEqual(
            guard["sequential_profile_acceptance_counts"],
            {"qa:learning": 1},
        )
        self.assertEqual(guard["sequential_profile_rejection_counts"], {})
        self.assertEqual(
            guard["sequential_profile_probe_sample"],
            [{"profile": "qa:learning", "accepted": True, "records": 3}],
        )

    def test_sequential_profile_probe_result_records_rejection_diagnostics(self) -> None:
        guard = empty_guard()
        guard["sequential_profile_acceptances"] = 0
        guard["sequential_profile_rejections"] = 0
        guard["sequential_profile_acceptance_counts"] = {}
        guard["sequential_profile_rejection_counts"] = {}

        record_baseline_floor_sequential_profile_probe_result(
            guard,
            profile="fact:self",
            accepted=False,
            records=2,
            diagnostics={
                "worst_violation": {"profile": "fact:self", "deficit": 1},
                "violating_profile_count": 1,
            },
        )

        self.assertEqual(guard["sequential_profile_acceptances"], 0)
        self.assertEqual(guard["sequential_profile_rejections"], 1)
        self.assertEqual(guard["sequential_profile_acceptance_counts"], {})
        self.assertEqual(
            guard["sequential_profile_rejection_counts"],
            {"fact:self": 1},
        )
        self.assertEqual(
            guard["sequential_profile_probe_sample"],
            [
                {
                    "profile": "fact:self",
                    "accepted": False,
                    "records": 2,
                    "worst_violation": {
                        "profile": "fact:self",
                        "deficit": 1,
                    },
                    "violating_profile_count": 1,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
