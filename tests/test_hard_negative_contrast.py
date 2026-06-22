"""Hard-negative mining for entity-paired contrast pairs (default-off, leakage-safe).

With an opt-in scorer, build_contrast_pairs selects the HARDEST eligible non-owner
(highest score) instead of the first; without one it is byte-for-byte the prior
first-eligible pick. Selection only ranks already-eligible non-owners, so the
out-of-corpus / no-leakage guarantee is preserved. Tested with a stub scorer --
no model needed.
"""

from __future__ import annotations

import re
import unittest

import support  # noqa: F401  (puts src/ on sys.path)

from answer_contrast_pairs import build_contrast_pairs

GRAMMAR = {
    "story_facts": [
        {"id": "alice-x", "person": "alice", "object": "x", "relation": "in", "container": "box", "color": "red"},
        {"id": "bob-y", "person": "bob", "object": "y", "relation": "on", "container": "mat", "color": "blue"},
        {"id": "carol-z", "person": "carol", "object": "z", "relation": "under", "container": "bed", "color": "green"},
    ],
    "withheld_fact_ids": [],
    "unknown_facts": [],
}
EXCLUDED = {("alice", "x"), ("bob", "y"), ("carol", "z")}


def _entity_object(prompt: str) -> tuple[str, str]:
    match = re.search(r"(\w+)'s (\w+)\?", prompt)
    return match.group(1), match.group(2)


class HardNegativeContrastTest(unittest.TestCase):
    def test_default_picks_first_eligible(self) -> None:
        # alice-x eligible non-owners (sorted) = [bob, carol]; default takes bob.
        pairs = build_contrast_pairs(GRAMMAR)
        alice = [_entity_object(ooc.prompt)[0] for owner, ooc in pairs if "alice's x" in owner.prompt]
        self.assertTrue(alice and all(person == "bob" for person in alice))

    def test_scorer_picks_hardest(self) -> None:
        def scorer(prompt: str, _concrete: str) -> float:
            person, _obj = _entity_object(prompt)
            return {"carol": 10.0, "bob": 1.0}.get(person, 0.0)

        pairs = build_contrast_pairs(GRAMMAR, hard_negative_scorer=scorer)
        alice = [_entity_object(ooc.prompt)[0] for owner, ooc in pairs if "alice's x" in owner.prompt]
        # carol outscores bob -> the harder negative is chosen.
        self.assertTrue(alice and all(person == "carol" for person in alice))

    def test_every_pick_is_leakage_safe(self) -> None:
        pairs = build_contrast_pairs(GRAMMAR, hard_negative_scorer=lambda _p, _c: 1.0)
        for owner_example, ooc_example in pairs:
            owner_person, _ = _entity_object(owner_example.prompt)
            ooc_person, ooc_obj = _entity_object(ooc_example.prompt)
            self.assertNotEqual(ooc_person, owner_person)
            self.assertNotIn((ooc_person, ooc_obj), EXCLUDED)
            self.assertEqual(ooc_example.target, " unknown.")


if __name__ == "__main__":
    unittest.main()
