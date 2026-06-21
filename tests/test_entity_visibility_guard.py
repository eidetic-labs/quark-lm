"""Locks the entity-visibility diagnosis: ctx=16 hides the entity, ctx>=48 shows it.

This is the falsifier behind the abstention work — if the answer-window stops
containing the query entity, no objective can teach entity-conditioned abstention.
"""

from __future__ import annotations

import unittest

import support  # noqa: F401  (inserts src/ onto sys.path)
from tokenizer import CharTokenizer
from transformer_entity_visibility_guard import entity_visible_in_window

MIA = "question: where is mia's ball?\nanswer:"
NOAH = "question: where is noah's ball?\nanswer:"


class EntityVisibilityGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharTokenizer.train(MIA + "\n" + NOAH + "\n under the box.\n")

    def test_entity_invisible_at_default_16(self) -> None:
        # The 38-char prompt is windowed to its last 16 chars ("'s ball?\nanswer:"),
        # identical for every person -> the entity is structurally absent.
        self.assertFalse(entity_visible_in_window(self.tokenizer, MIA, NOAH, 16))

    def test_entity_visible_at_48(self) -> None:
        self.assertTrue(entity_visible_in_window(self.tokenizer, MIA, NOAH, 48))


if __name__ == "__main__":
    unittest.main()
