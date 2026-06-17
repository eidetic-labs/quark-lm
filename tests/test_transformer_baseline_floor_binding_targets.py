import unittest

from transformer_baseline_floor_binding_targets import (
    select_collapsed_profile_binding_targets,
)
from transformer_memory_plan_helpers import plan_missing_first_token_targets


class TransformerBaselineFloorBindingTargetsTest(unittest.TestCase):
    def test_collapsed_profile_targets_preserve_unfiltered_names(self) -> None:
        targets = select_collapsed_profile_binding_targets(
            {"collapsed": ["learning", "owner"]},
            memory_consolidation_active=False,
            memory_consolidation_target_profiles=[],
            owner_paraphrase_binding_active=False,
            owner_paraphrase_target_profiles=[],
            collapsed_profile_names=lambda snapshot: snapshot["collapsed"],
        )

        self.assertEqual(targets, ["learning", "owner"])

    def test_collapsed_profile_targets_prefer_memory_filter(self) -> None:
        targets = select_collapsed_profile_binding_targets(
            {"collapsed": ["learning", "owner", "paraphrases"]},
            memory_consolidation_active=True,
            memory_consolidation_target_profiles=["learning"],
            owner_paraphrase_binding_active=True,
            owner_paraphrase_target_profiles=["owner", "paraphrases"],
            collapsed_profile_names=lambda snapshot: snapshot["collapsed"],
        )

        self.assertEqual(targets, ["learning"])

    def test_collapsed_profile_targets_filter_owner_paraphrase(self) -> None:
        targets = select_collapsed_profile_binding_targets(
            {"collapsed": ["learning", "owner", "paraphrases"]},
            memory_consolidation_active=False,
            memory_consolidation_target_profiles=[],
            owner_paraphrase_binding_active=True,
            owner_paraphrase_target_profiles=["owner", "paraphrases"],
            collapsed_profile_names=lambda snapshot: snapshot["collapsed"],
        )

        self.assertEqual(targets, ["owner", "paraphrases"])

    def test_missing_first_token_plan_uses_all_populated_targets(self) -> None:
        plan = plan_missing_first_token_targets(
            "fact:owner",
            ["learning", "owner", "empty"],
            {"learning": [3], "owner": [7, 5], "empty": []},
            profile_specific=False,
        )

        self.assertEqual(plan.target_profiles, ["learning", "owner"])
        self.assertEqual(plan.target_ids, [3, 5, 7])
        self.assertEqual(plan.target_id_set, {3, 5, 7})

    def test_missing_first_token_plan_can_be_profile_specific(self) -> None:
        plan = plan_missing_first_token_targets(
            "fact:owner",
            ["learning", "owner", "paraphrases"],
            {"learning": [3], "owner": [7], "paraphrases": [9]},
            profile_specific=True,
        )

        self.assertEqual(plan.target_profiles, ["owner", "paraphrases"])
        self.assertEqual(plan.target_ids, [7, 9])
        self.assertEqual(plan.target_id_set, {7, 9})


if __name__ == "__main__":
    unittest.main()
