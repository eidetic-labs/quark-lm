from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_subword_tokenizer import ClosedWorldSubwordTokenizer, MergeRule
from tokenizer import CharTokenizer
from tokenizer_artifacts import (
    propose_closed_world_subword_tokenizer,
    stable_json_hash,
    write_tokenizer_artifacts,
)
from tokenizer_io import tokenizer_from_dict


class ClosedWorldSubwordTokenizerTest(unittest.TestCase):
    def test_round_trip_and_merge_replay(self) -> None:
        proposal = propose_closed_world_subword_tokenizer(
            "kite kitchen kind kite kitchen\n",
            max_token_chars=4,
            max_new_tokens=3,
        )
        tokenizer = proposal["tokenizer"]

        self.assertEqual(
            tokenizer.decode(tokenizer.encode("kite kitchen\n")),
            "kite kitchen\n",
        )
        self.assertLess(
            len(tokenizer.encode("kite kitchen\n")),
            len(CharTokenizer.train("kite kitchen\n").encode("kite kitchen\n")),
        )

    def test_append_only_ids_with_base_tokenizer(self) -> None:
        base = CharTokenizer.train("kite kitchen\n")
        proposal = propose_closed_world_subword_tokenizer(
            "kite kitchen kind kite kitchen\n",
            base_tokenizer=base,
            max_token_chars=4,
            max_new_tokens=2,
        )
        tokenizer = proposal["tokenizer"]

        self.assertTrue(tokenizer.extends(base))
        self.assertEqual(
            {token: tokenizer.stoi[token] for token in base.tokens},
            {token: base.stoi[token] for token in base.tokens},
        )

    def test_full_answer_token_is_rejected(self) -> None:
        proposal = propose_closed_world_subword_tokenizer(
            "ki ki\n",
            protected_answers={"ki"},
            max_token_chars=4,
            max_new_tokens=2,
        )

        self.assertNotIn("ki", proposal["tokenizer"].tokens)
        rejected = proposal["manifest"]["rejected_candidates"]
        self.assertTrue(
            any(
                item["token"] == "ki" and "full_answer_token" in item["rejection_reasons"]
                for item in rejected
            )
        )

    def test_manifest_and_tokenizer_serialization(self) -> None:
        proposal = propose_closed_world_subword_tokenizer(
            "kite kitchen kind kite kitchen\n",
            source_files=["corpus/train.txt"],
            max_token_chars=4,
            max_new_tokens=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "tokenizer_manifest.json"
            report_path = Path(tmp) / "tokenizer_report.json"
            write_tokenizer_artifacts(
                manifest_path,
                report_path,
                proposal["manifest"],
                proposal["report"],
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["tokenizer_type"], "closed-world-subword")
        self.assertFalse(manifest["purity"]["pretrained_tokenizer"])
        self.assertTrue(report["round_trip_ok"])
        self.assertTrue(report["long_answer_effect"]["measured"])
        self.assertEqual(
            report["long_answer_effect"]["scope"],
            "tokenizer_level_only",
        )
        self.assertFalse(report["long_answer_effect"]["model_effect"]["measured"])
        self.assertEqual(proposal["manifest_hash"], stable_json_hash(proposal["manifest"]))

    def test_tokenizer_io_loads_subword_payload(self) -> None:
        tokenizer = ClosedWorldSubwordTokenizer.from_char_tokens(
            CharTokenizer.train("kite\n").tokens
        ).with_merge(MergeRule("k", "i", "ki"))

        loaded = tokenizer_from_dict(tokenizer.to_dict())

        self.assertIsInstance(loaded, ClosedWorldSubwordTokenizer)
        self.assertEqual(loaded.decode(loaded.encode("kite\n")), "kite\n")


if __name__ == "__main__":
    unittest.main()
