"""Fast guard for the entity-paired contrast objective.

Asserts the objective optimizes correctly and induces owner separation (the
owner comes to prefer its concrete answer over the abstain token). This runs in
a few steps so it stays cheap on the scalar engine.

NOTE: the full goal -- the answer-vs-unknown margin flipping SIGN with the entity
(non-owner prefers "unknown") -- is capacity/compute-bound. On the scalar engine
at dim=8 it reaches owner-separation but the entity-swapped non-owner only
reaches indifference (margin 0), because the owner term out-competes the OOC term
on the shared object representation. Pushing it to a true sign flip needs
position-projection and/or more capacity, which is why that measurement belongs
on the PyTorch trainer (see task: promote PyTorch trainer), not in a unit test.
"""

from __future__ import annotations

import random
import unittest

import support  # noqa: F401  (inserts src/ onto sys.path)
from answer_examples import AnswerExample
from support.core import (
    CharTokenizer,
    OptimizationConfig,
    ScalarOptimizer,
    TinyTransformerLM,
    TransformerConfig,
)
from transformer_answer_training_steps import train_answer_contrast_pair
from transformer_direct_answer_core import answer_sequence_nll

MIA_BALL = "mia ball"      # mia owns ball
NOAH_BALL = "noah ball"    # noah does NOT own ball
BALL_ANS = " a"
UNK = " u"
IN_EXAMPLE = AnswerExample(MIA_BALL, BALL_ANS, "fact")
OOC_EXAMPLE = AnswerExample(NOAH_BALL, UNK, "ooc")
ALL_TEXT = "".join([MIA_BALL, NOAH_BALL, BALL_ANS, UNK, "\n"])


def _margin(model, tokenizer, prompt: str, concrete: str) -> float:
    # nll(unknown) - nll(concrete): > 0 => concrete preferred.
    nll_unknown = answer_sequence_nll(model, tokenizer, AnswerExample(prompt, UNK, "x"))
    nll_concrete = answer_sequence_nll(model, tokenizer, AnswerExample(prompt, concrete, "x"))
    return nll_unknown - nll_concrete


class AnswerContrastPairTest(unittest.TestCase):
    def test_objective_optimizes_and_separates_owner(self) -> None:
        tokenizer = CharTokenizer.train(ALL_TEXT)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            context_size=16,
            embedding_dim=8,
            feedforward_dim=16,
            seed=11,
        )
        model = TinyTransformerLM.init_random(config)
        model.active_optimizer = ScalarOptimizer(OptimizationConfig(optimizer="adamw"))

        before_owner = _margin(model, tokenizer, MIA_BALL, BALL_ANS)
        rng = random.Random(0)
        losses = [
            train_answer_contrast_pair(model, tokenizer, IN_EXAMPLE, OOC_EXAMPLE, 0.05, rng)
            for _ in range(8)
        ]
        after_owner = _margin(model, tokenizer, MIA_BALL, BALL_ANS)

        # The contrast objective reduces its own loss (optimizes correctly).
        self.assertLess(losses[-1], losses[0])
        # The owner comes to prefer its concrete answer over the abstain token.
        self.assertGreater(after_owner, before_owner)


if __name__ == "__main__":
    unittest.main()
