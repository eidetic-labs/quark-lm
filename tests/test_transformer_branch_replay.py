from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support.branch_diversity import branch_replay_plan
from support.core import (
    ANSWER_TERMINATOR,
    AnswerExample,
    CharTokenizer,
    TinyTransformerLM,
    TransformerConfig,
)
from support.direct_answer import direct_answer_profiled_replay_records


class TransformerBranchReplayTest(unittest.TestCase):
    def test_branch_replay_plan_tracks_profile_deficits_independently(
        self,
    ) -> None:
        replay_branches = [
            ([0], 1, 1, "qa:place"),
            ([0], 2, 1, "qa:place"),
            ([0], 2, 2, "qa:color"),
        ]

        global_plan = branch_replay_plan(
            replay_branches,
            replay_branches,
            profile_aware_targets=False,
        )
        profile_plan = branch_replay_plan(
            replay_branches,
            replay_branches,
            profile_aware_targets=True,
        )

        self.assertEqual(
            global_plan["profiles"]["__all__"]["missing_target_ids"],
            [],
        )
        self.assertEqual(
            profile_plan["profiles"]["qa:place"]["missing_target_ids"],
            [2],
        )
        self.assertEqual(
            profile_plan["profiles"]["qa:color"]["missing_target_ids"],
            [],
        )
        self.assertEqual(
            profile_plan["profiles"]["qa:place"]["coverage_floor"],
            0.5,
        )
        with tempfile.TemporaryDirectory() as temp:
            plan_path = Path(temp) / "direct_answer_replay_plan.json"
            plan_path.write_text(
                json.dumps(profile_plan, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            loaded_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertTrue(loaded_plan["profile_aware_targets"])
        self.assertEqual(
            loaded_plan["profiles"]["qa:place"]["missing_target_count"],
            1,
        )

    def test_profiled_replay_records_preserve_sources_for_shared_targets(
        self,
    ) -> None:
        near = AnswerExample(prompt="q: where?\na:", target=" near.", source="qa:place")
        nine = AnswerExample(prompt="q: number?\na:", target=" nine.", source="qa:number")
        examples = [near, nine]
        tokenizer = CharTokenizer.train(
            near.prompt
            + near.target
            + nine.prompt
            + nine.target
            + ANSWER_TERMINATOR
        )
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=8,
                embedding_dim=4,
                feedforward_dim=8,
                seed=63,
            )
        )

        records = direct_answer_profiled_replay_records(
            model,
            tokenizer,
            examples,
            branch_position=1,
            terminator=ANSWER_TERMINATOR,
        )

        target_ids_by_profile = {
            profile: target for _context, target, _predicted, profile in records
        }
        self.assertEqual(set(target_ids_by_profile), {"qa:place", "qa:number"})
        self.assertEqual(
            target_ids_by_profile["qa:place"],
            target_ids_by_profile["qa:number"],
        )

    def test_branch_replay_coverage_falls_back_to_sampled_branches(
        self,
    ) -> None:
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=3,
                context_size=4,
                embedding_dim=3,
                feedforward_dim=4,
                seed=67,
            )
        )

        loss = model.train_step_with_branch_context_replay_coverage(
            [([0, 0, 0, 0], 1, 2)],
            [],
            learning_rate=0.01,
            negative_weight=0.0,
            positive_weight=0.0,
            replay_weight=1.0,
            hard_negative_count=1,
        )

        self.assertGreater(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
