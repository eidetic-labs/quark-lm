import unittest

import transformer_direct_modes as modes
from transformer_direct_mode_defaults import (
    BASELINE_FLOOR_CALIBRATED_ADAPTIVE_LEARNING_RATE_SCALES,
    BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES,
    BASELINE_FLOOR_REMAINING_PROFILE_BINDING_TARGET_SOURCE_LABELS,
)
from transformer_direct_mode_names import BASELINE_FLOOR_STABILIZATION_MODE
from transformer_direct_mode_sets import BASELINE_FLOOR_STABILIZATION_DIRECT_ANSWER_MODES
from transformer_direct_answer_guard_keys import (
    EMPTY_DICT_KEYS,
    EMPTY_LIST_KEYS,
    ZERO_INT_KEYS,
)
from transformer_direct_answer_guard_state import (
    build_direct_answer_update_guard,
    direct_answer_mode_flags as compatibility_mode_flags,
)
from transformer_direct_answer_mode_flags import direct_answer_mode_flags


class TransformerDirectAnswerGuardStateTest(unittest.TestCase):
    def test_direct_modes_reexports_baseline_floor_defaults(self) -> None:
        self.assertIs(
            modes.BASELINE_FLOOR_CALIBRATED_ADAPTIVE_LEARNING_RATE_SCALES,
            BASELINE_FLOOR_CALIBRATED_ADAPTIVE_LEARNING_RATE_SCALES,
        )
        self.assertIs(
            modes.BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES,
            BASELINE_FLOOR_OWNER_PARAPHRASE_BINDING_TARGET_PROFILES,
        )
        self.assertIs(
            modes.BASELINE_FLOOR_REMAINING_PROFILE_BINDING_TARGET_SOURCE_LABELS,
            BASELINE_FLOOR_REMAINING_PROFILE_BINDING_TARGET_SOURCE_LABELS,
        )

    def test_direct_modes_reexports_mode_names_and_sets(self) -> None:
        self.assertIs(
            modes.BASELINE_FLOOR_STABILIZATION_MODE,
            BASELINE_FLOOR_STABILIZATION_MODE,
        )
        self.assertIs(
            modes.BASELINE_FLOOR_STABILIZATION_DIRECT_ANSWER_MODES,
            BASELINE_FLOOR_STABILIZATION_DIRECT_ANSWER_MODES,
        )

    def test_mode_flags_resolve_profile_scale_recovery_mode(self) -> None:
        flags = direct_answer_mode_flags(
            modes.BASELINE_FLOOR_PROFILE_SCALE_BRANCH_DIVERSITY_RECOVERY_FRONTIER_CALIBRATED_STABILIZATION_MODE
        )

        self.assertTrue(flags["active"])
        self.assertTrue(flags["adaptive"])
        self.assertTrue(flags["sequential_stabilization_active"])
        self.assertTrue(flags["profile_scale_calibrated_stabilization_active"])
        self.assertTrue(flags["profile_scale_diversity_stabilization_active"])
        self.assertTrue(flags["profile_scale_coverage_recovery_frontier_stabilization_active"])
        self.assertTrue(
            flags[
                "profile_scale_branch_stable_coverage_recovery_frontier_stabilization_active"
            ]
        )
        self.assertTrue(
            flags[
                "profile_scale_branch_diversity_recovery_frontier_stabilization_active"
            ]
        )
        self.assertFalse(
            flags[
                "profile_scale_collapsed_profile_binding_frontier_stabilization_active"
            ]
        )

    def test_guard_state_reexports_mode_flags_for_compatibility(self) -> None:
        self.assertIs(compatibility_mode_flags, direct_answer_mode_flags)

    def test_guard_builder_initializes_key_inventory_types(self) -> None:
        guard = build_direct_answer_update_guard(
            direct_answer_mode=modes.BASELINE_FLOOR_STABILIZATION_MODE,
            memory_consolidation_max_profiles=3,
            direct_baseline_floor_learning_rate_scales=(1.0, 0.25),
            direct_baseline_floor_outer_learning_rate_scales=(1.0,),
            direct_baseline_floor_repair_anchors=[
                ([1], 2, 2, "qa:owner"),
                ([2], 3, 3, "fact:self"),
            ],
            direct_baseline_floor_frontier_anchors=[([3], 4, 4, "qa:owner")],
            direct_remaining_profile_binding_target_profiles=["qa:owner"],
            direct_remaining_profile_binding_source_labels=["qa"],
            direct_replay_plan={},
            direct_memory_consolidation_source_plan_path=None,
            direct_memory_consolidation_source_plan_summary={},
            direct_memory_consolidation_target_profiles=[],
            direct_memory_consolidation_top_priority_profiles=[],
            direct_memory_consolidation_collapsed_memory_backed_profiles=[],
            direct_memory_consolidation_missing_first_token_values={},
            direct_memory_consolidation_missing_first_token_ids={},
            direct_memory_consolidation_profile_specific_missing_first_token_target_map={},
        )

        for key in ZERO_INT_KEYS:
            self.assertEqual(guard[key], 0, key)
        for key in EMPTY_DICT_KEYS:
            self.assertEqual(guard[key], {}, key)
        for key in EMPTY_LIST_KEYS:
            self.assertEqual(guard[key], [], key)
        self.assertEqual(guard["repair_anchor_count"], 2)
        self.assertEqual(guard["frontier_anchor_count"], 1)
        self.assertTrue(guard["stabilization_active"])


if __name__ == "__main__":
    unittest.main()
