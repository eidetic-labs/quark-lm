import unittest

from transformer_experiment_modes import (
    PROFILE_SCALE_DIVERSITY_MODE,
    PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE,
)
from transformer_profile_scale_experiment_gates import profile_scale_experiment_gates


class TransformerProfileScaleExperimentGatesTest(unittest.TestCase):
    def test_profile_scale_diversity_mode_adds_expected_gate_name(self) -> None:
        gates = profile_scale_experiment_gates(PROFILE_SCALE_DIVERSITY_MODE)

        self.assertEqual(
            [gate["name"] for gate in gates],
            [
                "baseline_floor_profile_scale_diversity_calibrated_"
                "sequential_stabilization_screen"
            ],
        )
        self.assertTrue(gates[0]["required"])

    def test_profile_specific_memory_mode_names_missing_token_surface(self) -> None:
        gates = profile_scale_experiment_gates(
            PROFILE_SCALE_REMAINING_COLLAPSED_PROFILE_SPECIFIC_MISSING_FIRST_TOKEN_CONSOLIDATION_FRONTIER_MODE
        )

        self.assertEqual(len(gates), 1)
        self.assertIn(
            "remaining_collapsed_profile_specific_missing_first_token",
            gates[0]["name"],
        )
        self.assertIn(
            "profile-specific missing first-token target map",
            gates[0]["rule"],
        )

    def test_non_profile_scale_mode_adds_no_extra_gates(self) -> None:
        self.assertEqual(profile_scale_experiment_gates("branch-basic"), [])


if __name__ == "__main__":
    unittest.main()
