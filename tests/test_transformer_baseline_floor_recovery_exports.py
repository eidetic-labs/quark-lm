import unittest

import transformer_baseline_floor_recovery
from transformer_baseline_floor_branch_diversity_recovery import (
    BranchDiversityRecoveryResult,
    try_baseline_floor_branch_diversity_recovery,
)
from transformer_baseline_floor_coverage_recovery import (
    CoverageRecoveryResult,
    try_baseline_floor_coverage_recovery,
)


class TransformerBaselineFloorRecoveryExportsTest(unittest.TestCase):
    def test_compatibility_module_reexports_focused_recovery_apis(self) -> None:
        self.assertIs(
            transformer_baseline_floor_recovery.BranchDiversityRecoveryResult,
            BranchDiversityRecoveryResult,
        )
        self.assertIs(
            transformer_baseline_floor_recovery.CoverageRecoveryResult,
            CoverageRecoveryResult,
        )
        self.assertIs(
            transformer_baseline_floor_recovery.try_baseline_floor_branch_diversity_recovery,
            try_baseline_floor_branch_diversity_recovery,
        )
        self.assertIs(
            transformer_baseline_floor_recovery.try_baseline_floor_coverage_recovery,
            try_baseline_floor_coverage_recovery,
        )


if __name__ == "__main__":
    unittest.main()
