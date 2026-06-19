from __future__ import annotations

import math


def apply_fake_adamw_gradient(
    value: object,
    grad: object,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    step: int,
):
    if isinstance(value, list) and isinstance(grad, list):
        return [
            apply_fake_adamw_gradient(
                value_item,
                grad_item,
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                step,
            )
            for value_item, grad_item in zip(value, grad)
        ]
    if isinstance(value, list):
        return [
            apply_fake_adamw_gradient(
                item,
                grad,
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                step,
            )
            for item in value
        ]
    if isinstance(grad, list):
        return [
            apply_fake_adamw_gradient(
                value,
                item,
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                step,
            )
            for item in grad
        ]
    first_moment = (1.0 - beta1) * float(grad)
    second_moment = (1.0 - beta2) * float(grad) * float(grad)
    first_unbiased = first_moment / (1.0 - beta1**step)
    second_unbiased = second_moment / (1.0 - beta2**step)
    updated = float(value)
    if weight_decay > 0.0:
        updated -= learning_rate * weight_decay * updated
    return updated - learning_rate * first_unbiased / (
        math.sqrt(second_unbiased) + epsilon
    )
