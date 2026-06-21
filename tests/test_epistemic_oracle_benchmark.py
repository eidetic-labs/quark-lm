from __future__ import annotations

import unittest

import support  # noqa: F401  -- puts src/ on sys.path

from corpus_responder import CorpusResponder
from epistemic_oracle_benchmark import oracle_benchmark


# A tiny corpus written in the exact line format CorpusResponder's regexes
# parse: a place fact and a color fact for alice's bag.
CORPUS_TEXT = "\n".join(
    [
        "fact: alice's bag is kitchen.",
        "fact: alice's bag color is red.",
    ]
)


class EpistemicOracleBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.responder = CorpusResponder.train_from_text(CORPUS_TEXT)

    def test_in_corpus_prompt_resolves_to_real_answer(self) -> None:
        # Sanity-check the oracle so the benchmark assertions are meaningful.
        self.assertEqual(
            self.responder.answer_prompt("question: where is alice's bag?"),
            " kitchen.",
        )
        self.assertEqual(
            self.responder.answer_prompt("question: where is bob's hat?"),
            " unknown.",
        )

    def test_oracle_exact_rate_is_one_when_targets_match_oracle(self) -> None:
        records_by_set = {
            "in_corpus": [
                {"prompt": "question: where is alice's bag?", "target": " kitchen."},
                {
                    "prompt": "question: what color is alice's bag?",
                    "target": " red.",
                },
            ],
            "out_of_corpus": [
                {"prompt": "question: where is bob's hat?", "target": " unknown."},
            ],
        }

        result = oracle_benchmark(records_by_set, self.responder)

        in_corpus = result["per_set"]["in_corpus"]
        self.assertEqual(in_corpus["count"], 2)
        self.assertEqual(in_corpus["oracle_exact"], 2)
        self.assertEqual(in_corpus["oracle_exact_rate"], 1.0)
        # No predicted_candidate anywhere in this set.
        self.assertIsNone(in_corpus["agreement_rate"])

        out_of_corpus = result["per_set"]["out_of_corpus"]
        self.assertEqual(out_of_corpus["oracle_exact_rate"], 1.0)

        overall = result["overall"]
        self.assertEqual(overall["count"], 3)
        self.assertEqual(overall["oracle_exact"], 3)
        self.assertEqual(overall["oracle_exact_rate"], 1.0)

    def test_agreement_rate_computed_over_predicted_candidates(self) -> None:
        records_by_set = {
            "neural": [
                # neural matches the oracle ("kitchen") -> agreement
                {
                    "prompt": "question: where is alice's bag?",
                    "target": " kitchen.",
                    "predicted_candidate": " kitchen.",
                },
                # neural hallucinates; oracle says "red" -> disagreement,
                # and the neural answer is wrong relative to target too.
                {
                    "prompt": "question: what color is alice's bag?",
                    "target": " red.",
                    "predicted_candidate": " blue.",
                },
                # record without a predicted_candidate is ignored by agreement.
                {"prompt": "question: where is bob's hat?", "target": " unknown."},
            ],
        }

        result = oracle_benchmark(records_by_set, self.responder)

        neural = result["per_set"]["neural"]
        self.assertEqual(neural["count"], 3)
        self.assertEqual(neural["oracle_exact"], 3)
        self.assertEqual(neural["oracle_exact_rate"], 1.0)
        # 2 records carried predicted_candidate; 1 agreed with the oracle.
        self.assertEqual(neural["agreement_count"], 1)
        self.assertEqual(neural["agreement_rate"], 0.5)

        overall = result["overall"]
        self.assertEqual(overall["agreement_count"], 1)
        self.assertEqual(overall["agreement_rate"], 0.5)

    def test_empty_input_is_handled(self) -> None:
        result = oracle_benchmark({}, self.responder)
        self.assertEqual(result["per_set"], {})
        self.assertEqual(result["overall"]["count"], 0)
        self.assertEqual(result["overall"]["oracle_exact_rate"], 0.0)
        self.assertIsNone(result["overall"]["agreement_rate"])

    def test_empty_set_within_input(self) -> None:
        result = oracle_benchmark({"empty": []}, self.responder)
        empty = result["per_set"]["empty"]
        self.assertEqual(empty["count"], 0)
        self.assertEqual(empty["oracle_exact_rate"], 0.0)
        self.assertIsNone(empty["agreement_rate"])


if __name__ == "__main__":
    unittest.main()
