import unittest

import transformer_baseline_floor_helpers
from transformer_baseline_floor_anchor_selection import (
    BaselineFloorProfileSetup,
    baseline_floor_anchor_profile_groups,
    baseline_floor_objective_anchor_batch,
    baseline_floor_profile_attempt,
    baseline_floor_profile_setup,
)
from transformer_baseline_floor_anchor_profiles import (
    baseline_floor_anchor_profile_counts,
    baseline_floor_anchor_profile_target_count,
)
from transformer_baseline_floor_anchor_batches import (
    baseline_floor_frontier_anchor_records,
    baseline_floor_repair_anchor_records,
)
from transformer_baseline_floor_training import (
    train_direct_answer_baseline_floor_anchor_batch,
    train_direct_answer_baseline_floor_anchor_branch_diversity,
    train_direct_answer_baseline_floor_anchor_repair,
    train_direct_answer_baseline_floor_anchor_repair_stage,
    train_direct_answer_baseline_floor_stabilization_batch_stage,
)
from transformer_direct_answer_repair_selection import (
    direct_answer_balanced_repair_error,
    direct_answer_hard_branch_contrast,
)


class TransformerBaselineFloorHelpersExportsTest(unittest.TestCase):
    def test_compatibility_module_reexports_focused_helper_apis(self) -> None:
        self.assertIs(
            transformer_baseline_floor_helpers.BaselineFloorProfileSetup,
            BaselineFloorProfileSetup,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.baseline_floor_anchor_profile_counts,
            baseline_floor_anchor_profile_counts,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.baseline_floor_anchor_profile_groups,
            baseline_floor_anchor_profile_groups,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.baseline_floor_anchor_profile_target_count,
            baseline_floor_anchor_profile_target_count,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.baseline_floor_frontier_anchor_records,
            baseline_floor_frontier_anchor_records,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.baseline_floor_objective_anchor_batch,
            baseline_floor_objective_anchor_batch,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.baseline_floor_profile_attempt,
            baseline_floor_profile_attempt,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.baseline_floor_profile_setup,
            baseline_floor_profile_setup,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.baseline_floor_repair_anchor_records,
            baseline_floor_repair_anchor_records,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.train_direct_answer_baseline_floor_anchor_batch,
            train_direct_answer_baseline_floor_anchor_batch,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.train_direct_answer_baseline_floor_anchor_branch_diversity,
            train_direct_answer_baseline_floor_anchor_branch_diversity,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.train_direct_answer_baseline_floor_anchor_repair,
            train_direct_answer_baseline_floor_anchor_repair,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.train_direct_answer_baseline_floor_anchor_repair_stage,
            train_direct_answer_baseline_floor_anchor_repair_stage,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.train_direct_answer_baseline_floor_stabilization_batch_stage,
            train_direct_answer_baseline_floor_stabilization_batch_stage,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.direct_answer_balanced_repair_error,
            direct_answer_balanced_repair_error,
        )
        self.assertIs(
            transformer_baseline_floor_helpers.direct_answer_hard_branch_contrast,
            direct_answer_hard_branch_contrast,
        )


if __name__ == "__main__":
    unittest.main()
