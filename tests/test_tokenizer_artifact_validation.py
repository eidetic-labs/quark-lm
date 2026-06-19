from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tokenizer_artifact_validation import validate_tokenizer_artifacts
from tokenizer_artifacts import propose_closed_world_subword_tokenizer


class TokenizerArtifactValidationTests(unittest.TestCase):
    def test_valid_tokenizer_artifacts_pass(self) -> None:
        proposal = _proposal()

        validate_tokenizer_artifacts(
            proposal["manifest"],
            proposal["report"],
            manifest_hash=proposal["manifest_hash"],
        )

    def test_validator_rejects_stale_manifest_hash(self) -> None:
        proposal = _proposal()

        with self.assertRaisesRegex(ValueError, "manifest_hash"):
            validate_tokenizer_artifacts(
                proposal["manifest"],
                proposal["report"],
                manifest_hash="0" * 64,
            )

    def test_validator_rejects_pretrained_purity_drift(self) -> None:
        proposal = _proposal()
        manifest = copy.deepcopy(proposal["manifest"])
        manifest["purity"]["pretrained_tokenizer"] = True

        with self.assertRaisesRegex(ValueError, "pretrained_tokenizer"):
            validate_tokenizer_artifacts(manifest, proposal["report"])

    def test_validator_rejects_report_savings_drift(self) -> None:
        proposal = _proposal()
        report = copy.deepcopy(proposal["report"])
        report["token_count_savings"] += 1

        with self.assertRaisesRegex(ValueError, "token_count_savings"):
            validate_tokenizer_artifacts(proposal["manifest"], report)

    def test_validator_rejects_full_answer_tokens_by_default(self) -> None:
        proposal = _proposal()
        report = copy.deepcopy(proposal["report"])
        report["full_answer_tokens"] = [" no."]

        with self.assertRaisesRegex(ValueError, "full_answer_tokens"):
            validate_tokenizer_artifacts(proposal["manifest"], report)

    def test_validator_allows_full_answer_tokens_only_when_requested(self) -> None:
        proposal = _proposal()
        report = copy.deepcopy(proposal["report"])
        report["full_answer_tokens"] = [" no."]

        validate_tokenizer_artifacts(
            proposal["manifest"],
            report,
            require_no_full_answer_tokens=False,
        )


def _proposal() -> dict:
    return propose_closed_world_subword_tokenizer(
        "kite kitchen kind kite kitchen\n",
        source_files=["corpus/train.txt"],
        protected_answers={"kite"},
        max_token_chars=4,
        max_new_tokens=3,
    )


if __name__ == "__main__":
    unittest.main()
