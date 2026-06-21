from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import support  # noqa: F401  (inserts src/ onto sys.path)
from epistemic_eval_runner import run_epistemic_eval
from support.char_model import char_model_fixture


class _StubResponder:
    def answer_prompt(self, prompt: str) -> str:
        return " unknown." if "x" in prompt else " abc."


class EpistemicEvalRunnerTest(unittest.TestCase):
    def test_runs_and_produces_report(self) -> None:
        tokenizer, _ids, _config, model = char_model_fixture("qa unknown. abc. x\n", seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa = root / "qa.jsonl"
            unknowns = root / "unknowns.jsonl"
            qa.write_text(
                json.dumps({"id": "q1", "prompt": "qa", "target": " abc."}) + "\n",
                encoding="utf-8",
            )
            unknowns.write_text(
                json.dumps({"id": "u1", "prompt": "x", "target": " unknown."}) + "\n",
                encoding="utf-8",
            )
            report = run_epistemic_eval(
                model=model,
                tokenizer=tokenizer,
                probe_paths=[qa, unknowns],
                max_new_chars=8,
                responder=_StubResponder(),
            )

        for key in ("nll_vs_random", "abstention", "calibration", "oracle", "headline"):
            self.assertIn(key, report)
        for key in (
            "learned_all",
            "learned_any",
            "mean_nll_reduction",
            "abstention_f1",
            "calibration_ece",
            "oracle_exact_rate",
        ):
            self.assertIn(key, report["headline"])
        self.assertEqual(report["nll_vs_random"]["overall"]["sets_scored"], 2)
        self.assertIsInstance(report["calibration"]["ece"], float)
        # Stub oracle answers both probes' targets correctly.
        self.assertAlmostEqual(report["oracle"]["overall"]["oracle_exact_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
