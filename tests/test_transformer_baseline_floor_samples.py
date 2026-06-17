import unittest

from support.baseline_floor_samples import (
    BASE_SAMPLE_KEYS,
    OPTIONAL_SAMPLE_KEYS,
    SAMPLE_KEYS,
    empty_guard,
)
from transformer_baseline_floor_probe_samples import (
    SAMPLE_LIMIT,
    BaselineFloorProbeSampleStreams,
    append_baseline_floor_probe_sample,
)


class TransformerBaselineFloorProbeSamplesTest(unittest.TestCase):
    def test_default_streams_append_base_probe_samples(self) -> None:
        guard = empty_guard()
        sample = {"profile": "qa:learning", "accepted": False}

        append_baseline_floor_probe_sample(guard, sample)

        for key in BASE_SAMPLE_KEYS:
            with self.subTest(key=key):
                self.assertEqual(guard[key], [sample])
        for key in OPTIONAL_SAMPLE_KEYS:
            with self.subTest(key=key):
                self.assertEqual(guard[key], [])

    def test_active_streams_append_optional_probe_samples(self) -> None:
        guard = empty_guard()
        sample = {"profile": "qa:learning", "accepted": True}

        append_baseline_floor_probe_sample(
            guard,
            sample,
            BaselineFloorProbeSampleStreams(
                coverage_recovery=True,
                branch_stable_coverage_recovery=True,
                branch_diversity_recovery=True,
                collapsed_profile_binding=True,
                remaining_profile_binding=True,
                owner_paraphrase_binding=True,
                memory_consolidation=True,
                missing_first_token=True,
            ),
        )

        for key in SAMPLE_KEYS:
            with self.subTest(key=key):
                self.assertEqual(guard[key], [sample])

    def test_probe_samples_are_capped(self) -> None:
        guard = {
            key: [{"index": index} for index in range(SAMPLE_LIMIT)]
            for key in SAMPLE_KEYS
        }

        append_baseline_floor_probe_sample(guard, {"index": SAMPLE_LIMIT})

        for key in BASE_SAMPLE_KEYS:
            with self.subTest(key=key):
                self.assertEqual(len(guard[key]), SAMPLE_LIMIT)
                self.assertEqual(guard[key][-1], {"index": SAMPLE_LIMIT - 1})


if __name__ == "__main__":
    unittest.main()
