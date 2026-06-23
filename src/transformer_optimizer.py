from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from autograd import Scalar
from transformer_model import OptimizationConfig, validate_optimization_config
from transformer_optimizer_gradient_evidence import (
    build_optimizer_gradient_evidence,
)


def scheduled_learning_rate(
    base_learning_rate: float,
    step: int,
    *,
    warmup_steps: int,
    decay_steps: int,
    min_learning_rate: float,
    schedule: str = "linear",
) -> float:
    """Warmup-then-decay LR schedule shared by the scalar and torch trainers.

    The default schedule="linear" reproduces the prior warmup-then-linear-decay
    values byte-for-byte (parity holds at every step, not just for constant-LR
    runs). "cosine" replaces the post-warmup linear tail with a half-cosine decay
    from peak to min over decay_steps; "wsd" holds the peak for a stable fraction
    of the tail, then cosine-decays the remainder to min.
    """

    learning_rate = base_learning_rate
    if warmup_steps > 0:
        learning_rate *= min(1.0, step / warmup_steps)
    if decay_steps > 0 and step > warmup_steps:
        decay_step = min(step - warmup_steps, decay_steps)
        decay_fraction = decay_step / decay_steps
        if schedule == "cosine":
            learning_rate = _cosine_tail(learning_rate, decay_fraction, min_learning_rate)
        elif schedule == "wsd":
            learning_rate = _wsd_tail(learning_rate, decay_fraction, min_learning_rate)
        else:
            learning_rate = learning_rate - (learning_rate - min_learning_rate) * decay_fraction
    return max(learning_rate, min_learning_rate)


# Fraction of the decay window wsd holds the peak (stable phase) before cosine-decaying.
_WSD_STABLE_FRACTION = 0.5


def _cosine_tail(peak: float, decay_fraction: float, min_learning_rate: float) -> float:
    """Half-cosine from peak (decay_fraction=0) to min_learning_rate (=1)."""

    return min_learning_rate + 0.5 * (peak - min_learning_rate) * (
        1.0 + math.cos(math.pi * decay_fraction)
    )


def _wsd_tail(peak: float, decay_fraction: float, min_learning_rate: float) -> float:
    """Hold peak for the stable fraction, then cosine-decay the remainder to min."""

    if decay_fraction <= _WSD_STABLE_FRACTION:
        return peak
    tail_fraction = (decay_fraction - _WSD_STABLE_FRACTION) / (1.0 - _WSD_STABLE_FRACTION)
    return _cosine_tail(peak, tail_fraction, min_learning_rate)


class ScalarOptimizer:
    def __init__(
        self,
        config: OptimizationConfig | None = None,
        update_count: int = 0,
        first_moment: list[float] | None = None,
        second_moment: list[float] | None = None,
        gradient_buffer: list[float] | None = None,
        pending_accumulation: int = 0,
        no_decay_mask: list[bool] | None = None,
    ) -> None:
        self.config = config or OptimizationConfig()
        self.update_count = update_count
        self.first_moment = first_moment or []
        self.second_moment = second_moment or []
        self.gradient_buffer = gradient_buffer or []
        self.pending_accumulation = pending_accumulation
        # Per-element weight-decay exclusion (True == skip decay), aligned with the
        # flat params order. Empty => uniform decay (the pre-exclusion behavior).
        self.no_decay_mask = no_decay_mask or []
        self.last_apply_evidence: dict[str, Any] | None = None
        validate_optimization_config(self.config)

    def effective_learning_rate(self, base_learning_rate: float, next_step: int | None = None) -> float:
        step = self.update_count + 1 if next_step is None else next_step
        return scheduled_learning_rate(
            base_learning_rate,
            step,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.decay_steps,
            min_learning_rate=self.config.min_learning_rate,
        )

    def apply(self, params: list[Scalar], base_learning_rate: float) -> float:
        if self.no_decay_mask and len(self.no_decay_mask) != len(params):
            raise ValueError(
                f"no_decay_mask length {len(self.no_decay_mask)} != param count {len(params)}"
            )
        self._ensure_slots(len(params))
        buffer_before = list(self.gradient_buffer)
        update_count_before = self.update_count
        pending_before = self.pending_accumulation
        raw_grads = [parameter.grad for parameter in params]
        clipped_grads = [self._clipped_grad(parameter) for parameter in params]
        for index, grad in enumerate(clipped_grads):
            self.gradient_buffer[index] += grad
        buffer_after_add = list(self.gradient_buffer)
        self.pending_accumulation += 1
        if self.pending_accumulation < self.config.gradient_accumulation_steps:
            learning_rate = self.effective_learning_rate(base_learning_rate)
            self.last_apply_evidence = build_optimizer_gradient_evidence(
                raw_gradients=raw_grads,
                clipped_gradients=clipped_grads,
                buffer_before=buffer_before,
                buffer_after_add=buffer_after_add,
                accumulated_gradients=None,
                update_applied=False,
                update_count_before=update_count_before,
                update_count_after=self.update_count,
                pending_accumulation_before=pending_before,
                pending_accumulation_after=self.pending_accumulation,
                learning_rate=learning_rate,
            )
            return learning_rate
        accumulated_grads = [
            value / self.pending_accumulation
            for value in self.gradient_buffer
        ]
        self.gradient_buffer = [0.0 for _ in params]
        self.pending_accumulation = 0
        self.update_count += 1
        learning_rate = self.effective_learning_rate(base_learning_rate, self.update_count)
        if self.config.optimizer == "sgd":
            self._apply_sgd(params, accumulated_grads, learning_rate)
        elif self.config.optimizer == "adamw":
            self._apply_adamw(params, accumulated_grads, learning_rate)
        else:
            raise ValueError(f"unsupported optimizer: {self.config.optimizer}")
        self.last_apply_evidence = build_optimizer_gradient_evidence(
            raw_gradients=raw_grads,
            clipped_gradients=clipped_grads,
            buffer_before=buffer_before,
            buffer_after_add=buffer_after_add,
            accumulated_gradients=accumulated_grads,
            update_applied=True,
            update_count_before=update_count_before,
            update_count_after=self.update_count,
            pending_accumulation_before=pending_before,
            pending_accumulation_after=self.pending_accumulation,
            learning_rate=learning_rate,
        )
        return learning_rate

    def _ensure_slots(self, param_count: int) -> None:
        if len(self.first_moment) != param_count:
            self.first_moment = [0.0 for _ in range(param_count)]
        if len(self.second_moment) != param_count:
            self.second_moment = [0.0 for _ in range(param_count)]
        if len(self.gradient_buffer) != param_count:
            self.gradient_buffer = [0.0 for _ in range(param_count)]

    def _clipped_grad(self, parameter: Scalar) -> float:
        clip = self.config.gradient_clip
        if clip <= 0.0:
            return parameter.grad
        return max(min(parameter.grad, clip), -clip)

    def _decays(self, index: int) -> bool:
        """Whether the parameter at ``index`` is subject to weight decay.

        Empty mask (the default) => every parameter decays (the uniform,
        pre-exclusion path), so weight_decay=0 / no-mask runs are bit-exact.
        """

        return not self.no_decay_mask or not self.no_decay_mask[index]

    def _apply_sgd(
        self,
        params: list[Scalar],
        grads: list[float],
        learning_rate: float,
    ) -> None:
        for index, (parameter, grad) in enumerate(zip(params, grads)):
            if self.config.weight_decay > 0.0 and self._decays(index):
                grad += self.config.weight_decay * parameter.data
            parameter.data -= learning_rate * grad

    def _apply_adamw(
        self,
        params: list[Scalar],
        grads: list[float],
        learning_rate: float,
    ) -> None:
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        beta1_correction = 1.0 - beta1**self.update_count
        beta2_correction = 1.0 - beta2**self.update_count
        for index, (parameter, grad) in enumerate(zip(params, grads)):
            self.first_moment[index] = beta1 * self.first_moment[index] + (1.0 - beta1) * grad
            self.second_moment[index] = (
                beta2 * self.second_moment[index] + (1.0 - beta2) * grad * grad
            )
            first_unbiased = self.first_moment[index] / beta1_correction
            second_unbiased = self.second_moment[index] / beta2_correction
            if self.config.weight_decay > 0.0 and self._decays(index):
                parameter.data -= learning_rate * self.config.weight_decay * parameter.data
            parameter.data -= (
                learning_rate
                * first_unbiased
                / (math.sqrt(second_unbiased) + self.config.epsilon)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": asdict(self.config),
            "update_count": self.update_count,
            "param_count": len(self.first_moment),
            "first_moment": self.first_moment,
            "second_moment": self.second_moment,
            "gradient_buffer": self.gradient_buffer,
            "pending_accumulation": self.pending_accumulation,
            "no_decay_mask": self.no_decay_mask,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScalarOptimizer":
        return cls(
            OptimizationConfig(**payload.get("config", {})),
            update_count=int(payload.get("update_count", 0)),
            first_moment=[float(value) for value in payload.get("first_moment", [])],
            second_moment=[float(value) for value in payload.get("second_moment", [])],
            gradient_buffer=[float(value) for value in payload.get("gradient_buffer", [])],
            pending_accumulation=int(payload.get("pending_accumulation", 0)),
            no_decay_mask=[bool(value) for value in payload.get("no_decay_mask", [])],
        )

    def summary(self) -> dict[str, Any]:
        return {
            "optimizer": self.config.optimizer,
            "update_count": self.update_count,
            "param_count": len(self.first_moment),
            "pending_accumulation": self.pending_accumulation,
            "last_learning_rate": (
                self.effective_learning_rate(1.0, self.update_count)
                if self.update_count
                else None
            ),
            "gradient_clip": self.config.gradient_clip,
            "weight_decay": self.config.weight_decay,
            "warmup_steps": self.config.warmup_steps,
            "decay_steps": self.config.decay_steps,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
        }


def load_optimizer_state(
    path: Path | None,
    config: OptimizationConfig,
    no_decay_mask: list[bool] | None = None,
) -> ScalarOptimizer:
    if path is None:
        return ScalarOptimizer(config, no_decay_mask=no_decay_mask)
    with path.open("r", encoding="utf-8") as handle:
        optimizer = ScalarOptimizer.from_dict(json.load(handle))
    if asdict(optimizer.config) != asdict(config):
        raise ValueError("resume optimizer config does not match requested optimizer config")
    if no_decay_mask:
        # Repopulate the authoritative mask on resume: old/mask-less checkpoints
        # would otherwise silently revert to uniform decay under weight_decay>0.
        if optimizer.no_decay_mask and optimizer.no_decay_mask != no_decay_mask:
            raise ValueError("resume optimizer no_decay_mask does not match recomputed mask")
        optimizer.no_decay_mask = no_decay_mask
    return optimizer


def save_optimizer_state(path: Path, optimizer: ScalarOptimizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(optimizer.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
