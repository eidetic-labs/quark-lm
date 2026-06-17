from __future__ import annotations

import unittest

from support.branch_diversity import (
    direct_answer_branch_logit_prior_profile,
    direct_answer_branch_representation_profile,
)
from support.branch_training import branch_training_fixture
from support.core import ANSWER_TERMINATOR


class TransformerBranchDiagnosticsTest(unittest.TestCase):
    def test_branch_representation_profile_reports_hidden_distances(self) -> None:
        fixture = branch_training_fixture(seed=47)

        profile = direct_answer_branch_representation_profile(
            fixture.model,
            fixture.tokenizer,
            fixture.records,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(profile["count"], 3)
        self.assertEqual(profile["skipped"], 0)
        self.assertEqual(profile["target_unique"], 3)
        self.assertEqual(profile["different_target_pairwise_distance"]["count"], 3)
        self.assertGreater(profile["different_target_pairwise_distance"]["avg"], 0.0)
        self.assertEqual(len(profile["target_centroids"]), 3)
        self.assertEqual(profile["target_centroid_distance"]["count"], 3)
        self.assertEqual(profile["target_centroid_margin"]["count"], 3)

    def test_branch_logit_prior_profile_decomposes_dominant_bias(self) -> None:
        fixture = branch_training_fixture(seed=48)
        fixture.model.bout[fixture.tokenizer.stoi["n"]].data = 5.0

        profile = direct_answer_branch_logit_prior_profile(
            fixture.model,
            fixture.tokenizer,
            fixture.records,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        self.assertEqual(profile["count"], 3)
        self.assertEqual(profile["dominant_predicted_token"], "n")
        self.assertEqual(profile["dominant_token_bias_rank"], 1)
        missing_tokens = {item["value"] for item in profile["missing_target_tokens"]}
        self.assertIn("g", missing_tokens)
        self.assertIn("t", missing_tokens)
        failed = profile["dominant_vs_target_decomposition"]["failed_records"]
        self.assertEqual(failed["primary_pressure"], "output_bias")
        self.assertEqual(failed["count"], 2)
        self.assertGreater(failed["avg_bias_advantage"], 0.0)
        self.assertGreater(profile["dominant_vs_missing_bias_advantage"], 0.0)


if __name__ == "__main__":
    unittest.main()
