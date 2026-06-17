import unittest

import transformer_direct_modes as modes
from transformer_baseline_floor_update_shapes import baseline_floor_attempt_update_shape


class TransformerBaselineFloorUpdateShapesTest(unittest.TestCase):
    def test_update_shape_tracks_baseline_floor_mode(self) -> None:
        cases = {
            "first-error": "direct",
            modes.BASELINE_FLOOR_STABILIZATION_MODE: "stabilization",
            modes.BASELINE_FLOOR_PROFILE_TARGETED_STABILIZATION_MODE: (
                "profile_targeted_stabilization"
            ),
            modes.BASELINE_FLOOR_SEQUENTIAL_STABILIZATION_MODE: (
                "sequential_profile_stabilization"
            ),
            modes.BASELINE_FLOOR_CALIBRATED_SEQUENTIAL_STABILIZATION_MODE: (
                "calibrated_sequential_profile_stabilization"
            ),
            modes.BASELINE_FLOOR_PROFILE_SCALE_DIVERSITY_CALIBRATED_STABILIZATION_MODE: (
                "profile_scale_diversity_calibrated_sequential_profile_stabilization"
            ),
            modes.BASELINE_FLOOR_PROFILE_SCALE_BRANCH_STABLE_COVERAGE_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE: (
                "profile_scale_branch_stable_coverage_recovery_frontier_diversity_"
                "calibrated_sequential_profile_stabilization"
            ),
            modes.BASELINE_FLOOR_PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_CALIBRATED_STABILIZATION_MODE: (
                "profile_scale_memory_consolidation_remaining_collapsed_profile_specific_"
                "missing_first_token_frontier_calibrated_sequential_profile_stabilization"
            ),
        }

        for mode, update_shape in cases.items():
            with self.subTest(mode=mode):
                self.assertEqual(
                    baseline_floor_attempt_update_shape(mode),
                    update_shape,
                )


if __name__ == "__main__":
    unittest.main()
