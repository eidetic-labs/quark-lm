from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_plan import (
    branch_replay_parts,
    branch_replay_plan,
    direct_answer_profile_key,
)


class ReplayPlanTest(unittest.TestCase):
    def test_legacy_replay_record_uses_global_profile(self) -> None:
        self.assertEqual(
            branch_replay_parts(([0, 1], 2, 3)),
            ([0, 1], 2, 3, "__all__"),
        )

    def test_profiled_plan_tracks_deficits_independently(self) -> None:
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

        self.assertEqual(global_plan["profiles"]["__all__"]["missing_target_ids"], [])
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

    def test_empty_replay_records_fall_back_to_branch_records(self) -> None:
        branches = [([0], 1, 0, "qa:place")]

        plan = branch_replay_plan(branches, [], profile_aware_targets=True)

        self.assertEqual(plan["replay_count"], 1)
        self.assertEqual(plan["profiles"]["qa:place"]["missing_target_ids"], [1])

    def test_profile_key_uses_unknown_for_missing_source(self) -> None:
        self.assertEqual(direct_answer_profile_key(SimpleNamespace(source=None)), "unknown")
        self.assertEqual(
            direct_answer_profile_key(SimpleNamespace(source="qa:number")),
            "qa:number",
        )

    def test_replay_plan_is_json_artifact_safe(self) -> None:
        plan = branch_replay_plan([([0], 1, 1, "qa:place")], [], True)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "direct_answer_replay_plan.json"
            path.write_text(
                json.dumps(plan, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(loaded["profile_aware_targets"])
        self.assertEqual(loaded["profiles"]["qa:place"]["coverage_floor"], 1.0)


if __name__ == "__main__":
    unittest.main()
