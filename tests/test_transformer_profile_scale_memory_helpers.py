from __future__ import annotations

import random
import unittest

from support.branch_diversity import (
    branch_diversity_profile_delta_has_coverage_gain,
)
from support.core import CharTokenizer
from support.memory import (
    memory_consolidation_missing_first_token_values,
    memory_consolidation_source_plan_targets,
    missing_first_token_anchor_batch,
    missing_first_token_ids_by_profile,
    profile_specific_missing_first_token_target_map,
    profile_specific_missing_first_token_targets,
    remaining_profile_binding_profile_order,
    remaining_profile_binding_source_labels,
)


class TransformerProfileScaleMemoryHelpersTest(unittest.TestCase):
    def test_remaining_profile_binding_prioritizes_source_profile_labels(
        self,
    ) -> None:
        source_labels = remaining_profile_binding_source_labels(
            ["learning", "owner", "paraphrases"]
        )
        groups = {
            "fact:self": [
                ([0], 1, 0, "fact:self"),
                ([0], 2, 0, "fact:self"),
            ],
            "fact:owner": [
                ([0], 1, 0, "fact:owner"),
                ([0], 2, 0, "fact:owner"),
            ],
            "qa:learning": [
                ([0], 3, 0, "qa:learning"),
                ([0], 4, 0, "qa:learning"),
            ],
            "unknown:place": [
                ([0], 5, 0, "unknown:place"),
                ([0], 6, 0, "unknown:place"),
            ],
        }

        ordered = remaining_profile_binding_profile_order(
            groups,
            ["learning", "owner", "paraphrases"],
        )

        self.assertEqual(
            source_labels,
            ["color", "learning", "owner", "place", "training_data"],
        )
        self.assertEqual(
            [profile for profile, _anchors in ordered[:3]],
            ["qa:learning", "fact:owner", "unknown:place"],
        )
        self.assertEqual(ordered[-1][0], "fact:self")

    def test_remaining_profile_binding_maps_memory_consolidation_targets(
        self,
    ) -> None:
        source_labels = remaining_profile_binding_source_labels(
            ["owner", "paraphrases", "heldout", "qa", "glossary"]
        )

        self.assertEqual(
            source_labels,
            ["color", "glossary", "owner", "place", "training_data"],
        )

    def test_missing_first_token_helpers_use_source_plan_targets(
        self,
    ) -> None:
        source_plan = {
            "kind": "memory_consolidation_plan",
            "profile_priorities": [
                {
                    "profile": "owner",
                    "missing_target_tokens": [
                        {"value": "u", "count": 2},
                        {"value": "a", "count": 1},
                        {"value": "u", "count": 1},
                    ],
                },
                {
                    "profile": "qa",
                    "missing_target_tokens": [{"value": "g", "count": 1}],
                },
                {
                    "profile": "self",
                    "missing_target_tokens": [{"value": "s", "count": 1}],
                },
            ],
        }
        tokenizer = CharTokenizer(["<pad>", "a", "g", "n", "u"])

        values = memory_consolidation_missing_first_token_values(
            source_plan,
            ["owner", "qa"],
        )
        ids_by_profile = missing_first_token_ids_by_profile(tokenizer, values)
        target_ids = {
            token_id
            for token_ids in ids_by_profile.values()
            for token_id in token_ids
        }
        batch = missing_first_token_anchor_batch(
            [
                ([0], tokenizer.stoi["u"], tokenizer.stoi["n"], "fact:owner"),
                ([0], tokenizer.stoi["a"], tokenizer.stoi["n"], "fact:owner"),
                ([0], tokenizer.stoi["n"], tokenizer.stoi["n"], "fact:owner"),
                ([0], tokenizer.stoi["g"], tokenizer.stoi["n"], "qa:place"),
            ],
            target_ids,
            random.Random(7),
            8,
        )

        self.assertEqual(values, {"owner": ["u", "a"], "qa": ["g"]})
        self.assertEqual(
            ids_by_profile,
            {
                "owner": [tokenizer.stoi["u"], tokenizer.stoi["a"]],
                "qa": [tokenizer.stoi["g"]],
            },
        )
        self.assertEqual(
            sorted(target for _context, target, _predicted, _profile in batch),
            [tokenizer.stoi["a"], tokenizer.stoi["g"], tokenizer.stoi["u"]],
        )
        self.assertTrue(
            branch_diversity_profile_delta_has_coverage_gain(
                {
                    "profiles": [
                        {"profile": "owner", "coverage_delta": 0.0},
                        {"profile": "qa", "coverage_delta": 0.125},
                    ]
                }
            )
        )

    def test_memory_consolidation_source_plan_can_require_collapsed_targets(
        self,
    ) -> None:
        source_plan = {
            "kind": "memory_consolidation_plan",
            "summary": {
                "top_priority_profiles": ["owner", "paraphrases"],
            },
            "profile_priorities": [
                {"profile": "owner"},
                {"profile": "paraphrases"},
            ],
        }

        _summary, targets, top_priorities, collapsed = (
            memory_consolidation_source_plan_targets(source_plan, 2)
        )

        self.assertEqual(targets, ["owner", "paraphrases"])
        self.assertEqual(top_priorities, ["owner", "paraphrases"])
        self.assertEqual(collapsed, [])
        with self.assertRaisesRegex(ValueError, "collapsed_memory_backed_profiles"):
            memory_consolidation_source_plan_targets(
                source_plan,
                2,
                require_collapsed_targets=True,
            )

    def test_profile_specific_missing_first_token_targets_follow_source_labels(
        self,
    ) -> None:
        ids_by_profile = {
            "owner": [1, 2],
            "paraphrases": [2, 3],
            "learning": [4],
        }
        target_profiles = ["owner", "paraphrases", "learning"]

        self.assertEqual(
            profile_specific_missing_first_token_targets(
                "fact:owner",
                target_profiles,
                ids_by_profile,
            ),
            ["owner", "paraphrases"],
        )
        self.assertEqual(
            profile_specific_missing_first_token_targets(
                "fact:learning",
                target_profiles,
                ids_by_profile,
            ),
            ["learning"],
        )
        self.assertEqual(
            profile_specific_missing_first_token_targets(
                "bridge:place",
                target_profiles,
                ids_by_profile,
            ),
            ["paraphrases"],
        )
        self.assertEqual(
            profile_specific_missing_first_token_target_map(
                target_profiles,
                ids_by_profile,
            ),
            {
                "color": ["paraphrases"],
                "learning": ["learning"],
                "owner": ["owner", "paraphrases"],
                "place": ["paraphrases"],
                "training_data": ["paraphrases"],
            },
        )


if __name__ == "__main__":
    unittest.main()
