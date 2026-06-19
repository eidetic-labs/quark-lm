from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformer_torch_training_parity_attempt import (
    build_torch_training_parity_attempt,
    write_torch_training_parity_attempt,
)
from transformer_torch_training_parity_attempt_cli import main, parse_args


class TransformerTorchTrainingParityAttemptCliTests(unittest.TestCase):
    def test_parse_args_accepts_verify_existing(self) -> None:
        args = parse_args(["--output-dir", "build/attempt", "--verify-existing"])

        self.assertEqual(args.output_dir, Path("build/attempt"))
        self.assertTrue(args.verify_existing)

    def test_verify_existing_returns_zero_for_valid_written_attempt(self) -> None:
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            write_torch_training_parity_attempt(Path(temp), artifacts)

            output = StringIO()
            with redirect_stdout(output):
                exit_code = main(["--output-dir", temp, "--verify-existing"])

        attempt = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(attempt["fixture_id"], artifacts["attempt"]["fixture_id"])

    def test_verify_existing_rejects_tampered_written_attempt(self) -> None:
        artifacts = _artifacts()
        with tempfile.TemporaryDirectory() as temp:
            written = write_torch_training_parity_attempt(Path(temp), artifacts)
            candidate_path = Path(written["artifacts"]["candidate"])
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            candidate["unvalidated_extra_field"] = "drift"
            candidate_path.write_text(
                json.dumps(candidate, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "artifact_hashes"):
                main(["--output-dir", temp, "--verify-existing"])


def _artifacts() -> dict:
    return build_torch_training_parity_attempt(
        corpus_dir=ROOT / "corpus",
        fixture_id="cli-verify-existing-training-parity-attempt",
        seed=53,
        context_index=4,
        context_size=4,
        embedding_dim=4,
        feedforward_dim=8,
        steps=2,
        importer=_missing_importer,
    )


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)
