from __future__ import annotations

import unittest

from closed_world_subword_tokenizer import ClosedWorldSubwordTokenizer
from tokenizer import CharTokenizer
from tokenizer_io import tokenizer_from_dict
from tokenizer_protocol import TokenizerProtocol


class TokenizerProtocolTest(unittest.TestCase):
    def test_char_tokenizer_satisfies_protocol(self) -> None:
        tokenizer = CharTokenizer.train("abc")

        self.assertIsInstance(tokenizer, TokenizerProtocol)
        self.assertEqual(tokenizer.decode(tokenizer.encode("abc")), "abc")

    def test_subword_tokenizer_satisfies_protocol(self) -> None:
        tokenizer = ClosedWorldSubwordTokenizer.from_char_tokens(["<pad>", "a", "b"])

        self.assertIsInstance(tokenizer, TokenizerProtocol)
        self.assertTrue(tokenizer.extends(tokenizer))

    def test_tokenizer_from_dict_returns_protocol_instance(self) -> None:
        tokenizer = tokenizer_from_dict(
            {
                "tokenizer_type": "closed-world-subword",
                "tokens": ["<pad>", "a"],
                "merge_rules": [],
            }
        )

        self.assertIsInstance(tokenizer, TokenizerProtocol)


if __name__ == "__main__":
    unittest.main()
