"""Vocab-stability guard: the corpus character vocabulary is pinned.

A char tokenizer's vocabulary IS the model's input/output dimension. A new
character appearing in the corpus silently grows the embedding + unembedding
tables, which (a) invalidates every existing checkpoint -- saved weight shapes no
longer match -- and (b) shifts token ids out from under any frozen-init golden.
This guard makes a vocab change a deliberate, reviewed event: if the corpus grows
a new character (corpus-growth is expected; vocab-growth is not), this test goes
red and the pin below must be updated on purpose, with checkpoints regenerated.
"""

from __future__ import annotations

import unittest

import support  # noqa: F401  (inserts src/ onto sys.path)
from curriculum import build_curriculum
from tokenizer import CharTokenizer

# Pinned corpus vocabulary. tokens[0] is always the pad token; the rest are the
# exact character set emitted by the curriculum. Update ONLY with a deliberate
# corpus change -- old checkpoints become unloadable when this list changes.
PINNED_TOKENS = [
    "<pad>", "\n", " ", "'", ",", "-", ".", ":", "?", "I", "_",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
]


class VocabStabilityTest(unittest.TestCase):
    def _corpus_tokenizer(self) -> CharTokenizer:
        return CharTokenizer.train(build_curriculum(seed=3).train_text)

    def test_corpus_vocab_size_is_pinned(self) -> None:
        tokenizer = self._corpus_tokenizer()
        self.assertEqual(
            tokenizer.vocab_size,
            len(PINNED_TOKENS),
            "corpus vocab size changed -> embedding/unembedding tables resize and "
            "every saved checkpoint becomes unloadable; update the pin deliberately",
        )

    def test_corpus_tokens_match_pin(self) -> None:
        tokenizer = self._corpus_tokenizer()
        self.assertEqual(
            tokenizer.tokens,
            PINNED_TOKENS,
            "corpus character set changed (added/removed/reordered a token) -> "
            "token ids shift under existing checkpoints; update the pin deliberately",
        )

    def test_vocab_is_deterministic_across_builds(self) -> None:
        # Same seed -> same corpus -> same vocab ordering, twice. Guards a
        # nondeterministic token order that would scramble ids run-to-run.
        self.assertEqual(self._corpus_tokenizer().tokens, self._corpus_tokenizer().tokens)


if __name__ == "__main__":
    unittest.main()
