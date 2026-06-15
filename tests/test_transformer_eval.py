from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from closed_world_lm.tokenizer import CharTokenizer
from closed_world_lm.transformer_char_model import TinyTransformerLM
from closed_world_lm.transformer_eval import (
    build_transformer_eval_report,
    eval_candidates_from_records,
    load_probe_records,
    score_transformer_evals,
    score_transformer_records,
    write_eval_report,
    write_eval_samples,
)
from closed_world_lm.transformer_model import GenerationConfig, TransformerConfig


class TransformerEvalTests(unittest.TestCase):
    def _model_and_tokenizer(self) -> tuple[TinyTransformerLM, CharTokenizer]:
        text = "q:\na: a.\n"
        tokenizer = CharTokenizer.train(text)
        model = TinyTransformerLM.init_random(
            TransformerConfig(
                vocab_size=tokenizer.vocab_size,
                context_size=4,
                embedding_dim=4,
                feedforward_dim=8,
                seed=8,
            )
        )
        return model, tokenizer

    def test_score_transformer_records_keeps_replayable_trace_shape(self) -> None:
        model, tokenizer = self._model_and_tokenizer()
        records = [{"id": "one", "prompt": "q:\na:", "target": " a."}]

        scored = score_transformer_records(
            model,
            tokenizer,
            records,
            max_new_chars=2,
            generation_config=GenerationConfig(trace_top_tokens=2),
            candidates=[" a."],
        )

        self.assertEqual(scored[0]["id"], "one")
        self.assertIn("generation_trace", scored[0])
        self.assertIn("generation_cache", scored[0])
        self.assertEqual(scored[0]["predicted_candidate"], " a.")
        self.assertEqual(scored[0]["candidate_scores"][0]["target"], " a.")

    def test_eval_report_and_writers_preserve_artifact_shape(self) -> None:
        model, tokenizer = self._model_and_tokenizer()
        probe_records = {"qa": [{"id": "one", "prompt": "q:\na:", "target": " a."}]}
        candidates = eval_candidates_from_records(probe_records)
        scored_by_eval = score_transformer_evals(
            model,
            tokenizer,
            probe_records,
            max_new_chars=2,
            generation_config=GenerationConfig(trace_top_tokens=2),
            candidates=candidates,
        )

        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "eval.json"
            samples_path = Path(temp) / "samples.jsonl"
            report = build_transformer_eval_report(
                Path("runs/demo/transformer.json"),
                [Path("evals/qa.jsonl")],
                probe_records,
                scored_by_eval,
                candidates,
                GenerationConfig(trace_top_tokens=2),
                samples_path,
            )
            write_eval_report(report_path, report)
            write_eval_samples(samples_path, scored_by_eval)

            written_report = json.loads(report_path.read_text(encoding="utf-8"))
            samples = [
                json.loads(line)
                for line in samples_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(written_report["candidate_count"], 1)
        self.assertEqual(written_report["eval_manifest"]["probe_counts"], {"qa": 1})
        self.assertEqual(samples[0]["eval"], "qa")
        self.assertEqual(samples[0]["id"], "one")

    def test_load_probe_records_uses_path_stems(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "custom.jsonl"
            path.write_text(
                json.dumps({"id": "one", "prompt": "p", "target": "t"}) + "\n",
                encoding="utf-8",
            )

            records = load_probe_records([path])

        self.assertEqual(list(records), ["custom"])
        self.assertEqual(records["custom"][0]["target"], "t")


if __name__ == "__main__":
    unittest.main()
