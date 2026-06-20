from __future__ import annotations

import unittest
from types import SimpleNamespace

from transformer_answer_direct_stage import restore_stage_state_and_rebind_recorder


class TransformerAnswerDirectStageTest(unittest.TestCase):
    def test_restore_rebinds_snapshot_recorder_to_stage_state(self) -> None:
        stage_state = _FakeStageState()
        recorder = SimpleNamespace(
            model=lambda: "old-model",
            tokenizer=lambda: "old-tokenizer",
        )

        restore_stage_state_and_rebind_recorder(
            stage_state,
            recorder,
            {"model": "restored-model"},
            {"optimizer": "restored-optimizer"},
        )

        self.assertEqual(stage_state.restore_calls, 1)
        self.assertEqual(recorder.model(), "restored-model")
        self.assertEqual(recorder.tokenizer(), "restored-tokenizer")


class _FakeStageState:
    def __init__(self) -> None:
        self.model = "old-model"
        self.tokenizer = "old-tokenizer"
        self.restore_calls = 0

    def restore(
        self,
        model_payload: dict[str, str],
        optimizer_payload: dict[str, str],
    ) -> None:
        self.restore_calls += 1
        self.model = model_payload["model"]
        self.tokenizer = "restored-tokenizer"


if __name__ == "__main__":
    unittest.main()
