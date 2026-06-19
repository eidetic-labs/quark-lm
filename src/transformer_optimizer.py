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


class ScalarOptimizer:
    def __init__(
        self,
        config: OptimizationConfig | None = None,
        update_count: int = 0,
        first_moment: list[float] | None = None,
        second_moment: list[float] | None = None,
        gradient_buffer: list[float] | None = None,
        pending_accumulation: int = 0,
    ) -> None:
        self.config = config or OptimizationConfig()
        self.update_count = update_count
        self.first_moment = first_moment or []
        self.second_moment = second_moment or []
        self.gradient_buffer = gradient_buffer or []
        self.pending_accumulation = pending_accumulation
        self.last_apply_evidence: dict[str, Any] | None = None
        validate_optimization_config(self.config)

    def effective_learning_rate(self, base_learning_rate: float, next_step: int | None = None) -> float:
        step = self.update_count + 1 if next_step is None else next_step
        learning_rate = base_learning_rate
        if self.config.warmup_steps > 0:
            learning_rate *= min(1.0, step / self.config.warmup_steps)
        if self.config.decay_steps > 0 and step > self.config.warmup_steps:
            decay_step = min(step - self.config.warmup_steps, self.config.decay_steps)
            decay_fraction = decay_step / self.config.decay_steps
            learning_rate = learning_rate - (
                learning_rate - self.config.min_learning_rate
            ) * decay_fraction
        return max(learning_rate, self.config.min_learning_rate)

    def apply(self, params: list[Scalar], base_learning_rate: float) -> float:
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

    def _apply_sgd(
        self,
        params: list[Scalar],
        grads: list[float],
        learning_rate: float,
    ) -> None:
        for parameter, grad in zip(params, grads):
            if self.config.weight_decay > 0.0:
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
            if self.config.weight_decay > 0.0:
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
) -> ScalarOptimizer:
    if path is None:
        return ScalarOptimizer(config)
    with path.open("r", encoding="utf-8") as handle:
        optimizer = ScalarOptimizer.from_dict(json.load(handle))
    if asdict(optimizer.config) != asdict(config):
        raise ValueError("resume optimizer config does not match requested optimizer config")
    return optimizer


def save_optimizer_state(path: Path, optimizer: ScalarOptimizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(optimizer.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
