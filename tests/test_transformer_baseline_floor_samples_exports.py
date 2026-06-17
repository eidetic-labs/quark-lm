import unittest

import transformer_baseline_floor_samples
from transformer_baseline_floor_acceptance_samples import (
    BaselineFloorProfileAcceptanceSample,
    record_baseline_floor_profile_acceptance_sample,
)
from transformer_baseline_floor_probe_samples import (
    SAMPLE_LIMIT,
    BaselineFloorProbeSampleStreams,
    append_baseline_floor_probe_sample,
)
from transformer_baseline_floor_rejection_samples import (
    record_baseline_floor_profile_rejection_sample,
)
from transformer_baseline_floor_sequential_samples import (
    record_baseline_floor_sequential_profile_probe_result,
)


class TransformerBaselineFloorSamplesExportsTest(unittest.TestCase):
    def test_compatibility_module_reexports_focused_sample_apis(self) -> None:
        self.assertEqual(transformer_baseline_floor_samples.SAMPLE_LIMIT, SAMPLE_LIMIT)
        self.assertIs(
            transformer_baseline_floor_samples.BaselineFloorProbeSampleStreams,
            BaselineFloorProbeSampleStreams,
        )
        self.assertIs(
            transformer_baseline_floor_samples.BaselineFloorProfileAcceptanceSample,
            BaselineFloorProfileAcceptanceSample,
        )
        self.assertIs(
            transformer_baseline_floor_samples.append_baseline_floor_probe_sample,
            append_baseline_floor_probe_sample,
        )
        self.assertIs(
            transformer_baseline_floor_samples.record_baseline_floor_profile_acceptance_sample,
            record_baseline_floor_profile_acceptance_sample,
        )
        self.assertIs(
            transformer_baseline_floor_samples.record_baseline_floor_profile_rejection_sample,
            record_baseline_floor_profile_rejection_sample,
        )
        self.assertIs(
            transformer_baseline_floor_samples.record_baseline_floor_sequential_profile_probe_result,
            record_baseline_floor_sequential_profile_probe_result,
        )


if __name__ == "__main__":
    unittest.main()
