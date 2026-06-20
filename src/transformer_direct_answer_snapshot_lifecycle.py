"""Direct-answer snapshot recording and finalization lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from branch_diversity_snapshots import branch_diversity_snapshot_score
from branch_diversity_snapshot_coverage import branch_diversity_snapshot_preserves_target_coverage
from tokenizer import CharTokenizer
from transformer_direct_answer_frontier_progress import build_frontier_progress_guard
from transformer_direct_answer_snapshot_records import direct_answer_snapshot_record
from transformer_model import GenerationConfig


@dataclass
class DirectAnswerSnapshotRecorder:
    model: Callable[[], Any]
    tokenizer: Callable[[], CharTokenizer]
    eval_records: dict[str, list[dict[str, Any]]]
    branch_position: int
    max_new_chars: int
    snapshot_mode: str
    terminator: str
    generation_config: GenerationConfig
    history_writer: Any

    def record(
        self,
        step: int,
        train_loss: float | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return direct_answer_snapshot_record(
            self.model(),
            self.tokenizer(),
            self.eval_records,
            self.branch_position,
            self.max_new_chars,
            self.snapshot_mode,
            self.terminator,
            self.generation_config,
            step,
            train_loss,
            extra,
        )

    def append(
        self,
        step: int,
        train_loss: float | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.history_writer.append(self.record(step, train_loss, extra))


@dataclass
class DirectAnswerBestSnapshotTracker:
    baseline: dict[str, Any]
    step: int
    score: tuple[Any, ...]
    baseline_model_payload: dict[str, Any]
    baseline_optimizer_payload: dict[str, Any]
    model_payload: dict[str, Any]
    optimizer_payload: dict[str, Any]

    @classmethod
    def from_baseline(
        cls,
        baseline: dict[str, Any],
        model: Any,
        tokenizer: CharTokenizer,
        optimizer: Any,
    ) -> DirectAnswerBestSnapshotTracker:
        model_payload = model.to_dict(tokenizer)
        optimizer_payload = optimizer.to_dict()
        return cls(
            baseline=baseline,
            step=0,
            score=branch_diversity_snapshot_score(baseline),
            baseline_model_payload=model_payload,
            baseline_optimizer_payload=optimizer_payload,
            model_payload=model_payload,
            optimizer_payload=optimizer_payload,
        )

    def record(
        self,
        snapshot: dict[str, Any],
        model: Any,
        tokenizer: CharTokenizer,
        optimizer: Any,
    ) -> None:
        score = branch_diversity_snapshot_score(snapshot)
        if not branch_diversity_snapshot_preserves_target_coverage(
            snapshot,
            self.baseline,
        ):
            return
        if score <= self.score:
            return
        self.step = int(snapshot["step"])
        self.score = score
        self.model_payload = model.to_dict(tokenizer)
        self.optimizer_payload = optimizer.to_dict()


@dataclass
class DirectAnswerSnapshotFinalization:
    model: Any
    tokenizer: CharTokenizer
    optimizer: Any
    last_snapshot: dict[str, Any]
    last_snapshot_step: int
    restored_best_branch_snapshot: bool
    restored_frontier_progress_snapshot: bool
    frontier_progress_guard: dict[str, Any]


def finalize_direct_answer_snapshots(
    *,
    direct_answer_steps: int,
    restore_best_branch_snapshot: bool,
    model_class: Any,
    optimizer_class: Any,
    model: Any,
    tokenizer: CharTokenizer,
    optimizer: Any,
    recorder: DirectAnswerSnapshotRecorder,
    best_snapshot: DirectAnswerBestSnapshotTracker,
    last_snapshot: dict[str, Any],
    last_snapshot_step: int,
    frontier_metrics_path: Any | None = None,
    frontier_baseline_snapshot: dict[str, Any] | None = None,
) -> DirectAnswerSnapshotFinalization:
    if last_snapshot_step != direct_answer_steps:
        last_snapshot = recorder.append(direct_answer_steps, None)
        last_snapshot_step = direct_answer_steps
        best_snapshot.record(
            last_snapshot,
            model,
            tokenizer,
            optimizer,
        )

    restored_best_branch_snapshot = False
    if restore_best_branch_snapshot and best_snapshot.step != last_snapshot_step:
        restored_model, restored_tokenizer = model_class.from_dict(
            best_snapshot.model_payload
        )
        model = restored_model
        if restored_tokenizer is not None:
            tokenizer = restored_tokenizer
        optimizer = optimizer_class.from_dict(best_snapshot.optimizer_payload)
        model.active_optimizer = optimizer
        restored_best_branch_snapshot = True
        recorder.model = lambda restored_model=model: restored_model
        recorder.tokenizer = lambda restored_tokenizer=tokenizer: restored_tokenizer
        last_snapshot = recorder.append(
            direct_answer_steps,
            None,
            {
                "restored_best_branch_snapshot": True,
                "restored_from_step": best_snapshot.step,
                "restored_from_score": list(best_snapshot.score),
            },
        )

    frontier_progress_guard = build_frontier_progress_guard(
        frontier_metrics_path=frontier_metrics_path,
        baseline_snapshot=frontier_baseline_snapshot or best_snapshot.baseline,
        final_snapshot=last_snapshot,
    )
    restored_frontier_progress_snapshot = False
    if (
        frontier_progress_guard.get("active") is True
        and frontier_progress_guard.get("progress_preserved") is not True
    ):
        model, tokenizer, optimizer = _restore_snapshot_payload(
            model_class=model_class,
            optimizer_class=optimizer_class,
            model_payload=best_snapshot.baseline_model_payload,
            optimizer_payload=best_snapshot.baseline_optimizer_payload,
            fallback_tokenizer=tokenizer,
        )
        restored_frontier_progress_snapshot = True
        recorder.model = lambda restored_model=model: restored_model
        recorder.tokenizer = lambda restored_tokenizer=tokenizer: restored_tokenizer
        last_snapshot = recorder.append(
            direct_answer_steps,
            None,
            {
                "restored_frontier_progress_snapshot": True,
                "frontier_progress_restore_reason": frontier_progress_guard.get(
                    "reason"
                ),
                "frontier_progress_pre_restore": frontier_progress_guard,
            },
        )
        frontier_progress_guard = {
            **build_frontier_progress_guard(
                frontier_metrics_path=frontier_metrics_path,
                baseline_snapshot=frontier_baseline_snapshot or best_snapshot.baseline,
                final_snapshot=last_snapshot,
            ),
            "restored_frontier_progress_snapshot": True,
            "pre_restore": frontier_progress_guard,
        }

    return DirectAnswerSnapshotFinalization(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        last_snapshot=last_snapshot,
        last_snapshot_step=last_snapshot_step,
        restored_best_branch_snapshot=restored_best_branch_snapshot,
        restored_frontier_progress_snapshot=restored_frontier_progress_snapshot,
        frontier_progress_guard=frontier_progress_guard,
    )


def _restore_snapshot_payload(
    *,
    model_class: Any,
    optimizer_class: Any,
    model_payload: dict[str, Any],
    optimizer_payload: dict[str, Any],
    fallback_tokenizer: CharTokenizer,
) -> tuple[Any, CharTokenizer, Any]:
    restored_model, restored_tokenizer = model_class.from_dict(model_payload)
    if restored_tokenizer is None:
        restored_tokenizer = fallback_tokenizer
    optimizer = optimizer_class.from_dict(optimizer_payload)
    restored_model.active_optimizer = optimizer
    return restored_model, restored_tokenizer, optimizer
