from __future__ import annotations

import unittest
from types import SimpleNamespace

from transformer_direct_answer_repair_targets import (
    direct_answer_repair_target_profiles,
    normalized_repair_target_profiles,
)


class TransformerDirectAnswerRepairTargetsTest(unittest.TestCase):
    def test_normalizes_blank_duplicate_and_unsorted_profiles(self) -> None:
        self.assertEqual(
            normalized_repair_target_profiles(["owner", " ", "learning", "owner"]),
            ["learning", "owner"],
        )

    def test_reads_declared_profiles_from_args(self) -> None:
        args = SimpleNamespace(
            direct_answer_repair_target_profile=["glossary", "learning"]
        )

        self.assertEqual(
            direct_answer_repair_target_profiles(args),
            ["glossary", "learning"],
        )


if __name__ == "__main__":
    unittest.main()
