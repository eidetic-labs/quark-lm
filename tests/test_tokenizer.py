from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokenizer import CharTokenizer


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

    def test_extend_preserves_existing_ids_and_adds_new_characters(self) -> None:
        tokenizer = CharTokenizer.train("cab")
        before_ids = {token: tokenizer.stoi[token] for token in tokenizer.tokens}

        extended = tokenizer.extend("cad!")

        self.assertEqual(
            {token: extended.stoi[token] for token in tokenizer.tokens},
            before_ids,
        )
        self.assertEqual(extended.decode(extended.encode("bad!")), "bad!")
        self.assertNotIn("!", tokenizer.stoi)


if __name__ == "__main__":
    unittest.main()
