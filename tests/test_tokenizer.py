from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.tokenizer import CharTokenizer


class TokenizerTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        text = "the ball is red.\n"
        tokenizer = CharTokenizer.train(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.pad_id, 0)

    def test_rejects_outside_character(self) -> None:
        tokenizer = CharTokenizer.train("abc")
        with self.assertRaises(ValueError):
            tokenizer.encode("abcd")


if __name__ == "__main__":
    unittest.main()
