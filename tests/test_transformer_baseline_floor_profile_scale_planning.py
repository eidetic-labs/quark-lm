import unittest
from types import SimpleNamespace

from transformer_baseline_floor_profile_scale_planning import (
    baseline_floor_profile_scale_priorities,
    record_baseline_floor_profile_scale_remaining_sources,
)


def setup_flags(
    *,
    remaining: bool = False,
    owner: bool = False,
    memory: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "direct_answer_baseline_floor_profile_scale_remaining_profile_binding_frontier_stabilization_active": remaining,
            "direct_answer_baseline_floor_profile_scale_owner_paraphrase_binding_frontier_stabilization_active": owner,
            "direct_answer_baseline_floor_profile_scale_memory_consolidation_frontier_stabilization_active": memory,
        }
    )


class TransformerBaselineFloorProfileScalePlanningTest(unittest.TestCase):
    def test_priorities_require_active_mode_and_remaining_source_profile(self) -> None:
        priorities = baseline_floor_profile_scale_priorities(
            setup_flags(remaining=True, owner=True, memory=True),
            "qa:learning",
            ["qa:learning"],
        )

        self.assertEqual(
            priorities,
            {"remaining": True, "owner": True, "memory": True},
        )

        inactive_source_priorities = baseline_floor_profile_scale_priorities(
            setup_flags(remaining=True, owner=True, memory=True),
            "fact:self",
            ["qa:learning"],
        )

        self.assertEqual(
            inactive_source_priorities,
            {"remaining": False, "owner": False, "memory": False},
        )

    def test_remaining_source_recording_skips_inactive_mode(self) -> None:
        ctx = SimpleNamespace(
            direct_setup=setup_flags(remaining=False, owner=True),
            update_guard={},
        )

        record_baseline_floor_profile_scale_remaining_sources(ctx, ["qa:learning"])

        self.assertEqual(ctx.update_guard, {})

    def test_remaining_source_recording_includes_owner_sources_when_active(self) -> None:
        ctx = SimpleNamespace(
            direct_setup=setup_flags(remaining=True, owner=True),
            update_guard={},
        )

        record_baseline_floor_profile_scale_remaining_sources(ctx, ["qa:learning"])

        self.assertEqual(
            ctx.update_guard,
            {
                "profile_scale_remaining_profile_binding_source_profiles": [
                    "qa:learning"
                ],
                "profile_scale_owner_paraphrase_binding_source_profiles": [
                    "qa:learning"
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
