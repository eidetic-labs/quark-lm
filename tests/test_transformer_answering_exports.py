import unittest

import transformer_answering
from transformer_answer_generator import (
    GENERATOR_BOS,
    GENERATOR_EOS,
    TransformerGuidedAnswerGenerator,
    build_transformer_answer_generator,
)
from transformer_answer_selector import AnswerCandidateSelector, build_answer_selector


class TransformerAnsweringExportsTest(unittest.TestCase):
    def test_compatibility_module_reexports_focused_answer_apis(self) -> None:
        self.assertIs(transformer_answering.AnswerCandidateSelector, AnswerCandidateSelector)
        self.assertIs(
            transformer_answering.TransformerGuidedAnswerGenerator,
            TransformerGuidedAnswerGenerator,
        )
        self.assertIs(transformer_answering.build_answer_selector, build_answer_selector)
        self.assertIs(
            transformer_answering.build_transformer_answer_generator,
            build_transformer_answer_generator,
        )
        self.assertEqual(transformer_answering.GENERATOR_BOS, GENERATOR_BOS)
        self.assertEqual(transformer_answering.GENERATOR_EOS, GENERATOR_EOS)


if __name__ == "__main__":
    unittest.main()
