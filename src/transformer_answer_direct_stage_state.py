"""Mutable state for direct-answer stage orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformer_direct_answer_update_guard import restore_direct_answer_update_state


@dataclass
class DirectAnswerStageState:
    args: Any
    model_class: type[Any]
    model: Any
    tokenizer: Any
    optimizer: Any
    params: Any

    def restore(
        self,
        model_payload: dict[str, Any],
        optimizer_payload: dict[str, Any],
    ) -> None:
        (
            self.model,
            self.tokenizer,
            self.optimizer,
            self.params,
        ) = restore_direct_answer_update_state(
            self.model_class,
            model_payload,
            optimizer_payload,
            self.tokenizer,
            self.args.direct_answer_train_top_layer_only,
            self.args.direct_answer_freeze_output_bias,
        )
