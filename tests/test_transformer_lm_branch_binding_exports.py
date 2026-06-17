import unittest

from transformer_lm_branch_binding import TransformerBranchBindingMixin
from transformer_lm_branch_pairwise_binding import (
    TransformerBranchPairwiseBindingMixin,
)
from transformer_lm_branch_target_binding import TransformerBranchTargetBindingMixin
from transformer_lm_branch_target_coverage import (
    TransformerBranchTargetCoverageMixin,
)


class TransformerLMBranchBindingExportsTest(unittest.TestCase):
    def test_branch_binding_mixin_combines_focused_objective_families(self) -> None:
        self.assertTrue(
            issubclass(
                TransformerBranchBindingMixin,
                TransformerBranchPairwiseBindingMixin,
            )
        )
        self.assertTrue(
            issubclass(
                TransformerBranchBindingMixin,
                TransformerBranchTargetBindingMixin,
            )
        )
        self.assertTrue(
            issubclass(
                TransformerBranchBindingMixin,
                TransformerBranchTargetCoverageMixin,
            )
        )


if __name__ == "__main__":
    unittest.main()
