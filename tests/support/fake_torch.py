from __future__ import annotations

import math
import types
from typing import Any, Callable


def fake_torch_importer(
    *,
    cuda: bool = False,
    mps: bool = False,
    training_runtime: bool = False,
    gradient_runtime: bool = False,
) -> Callable[[str], object]:
    FakeTensor.reset_grad_targets(gradient_runtime)

    fake = types.SimpleNamespace(
        __version__="fake-torch",
        float32="float32",
        float64="float64",
        tensor=lambda value, dtype=None, device=None, requires_grad=False: FakeTensor(
            value,
            requires_grad=requires_grad,
        ),
        stack=lambda values: FakeTensor([raw_value(value) for value in values]),
        tanh=lambda value: FakeTensor(_map_unary(raw_value(value), math.tanh)),
        log=lambda value: FakeTensor(_map_unary(raw_value(value), math.log)),
        softmax=lambda value, dim=0: FakeTensor(_softmax(raw_value(value))),
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps),
        ),
        nn=types.SimpleNamespace(
            utils=types.SimpleNamespace(
                clip_grad_value_=fake_clip_grad_value_,
            ),
        ),
    )
    if training_runtime:
        fake.autograd = types.SimpleNamespace()
        fake.optim = types.SimpleNamespace(AdamW=FakeAdamW)

    def importer(name: str) -> object:
        if name != "torch":
            raise ModuleNotFoundError(name)
        return fake

    return importer


class FakeTensor:
    _active_grad_targets: list["FakeTensor"] = []
    _backward_populates_grad = False

    def __init__(self, value: object, *, requires_grad: bool = False) -> None:
        self.value = _copy_raw(value)
        self.requires_grad = requires_grad
        self.grad = None
        if requires_grad:
            self._active_grad_targets.append(self)

    @classmethod
    def reset_grad_targets(cls, backward_populates_grad: bool) -> None:
        cls._active_grad_targets = []
        cls._backward_populates_grad = backward_populates_grad

    def __iter__(self):
        return (FakeTensor(item) for item in self.value)

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, key: int | slice):
        return FakeTensor(self.value[key])

    def __neg__(self):
        return FakeTensor(_map_unary(raw_value(self), lambda value: -value))

    def __add__(self, other: object):
        return FakeTensor(_binary(raw_value(self), raw_value(other), lambda left, right: left + right))

    def __radd__(self, other: object):
        return self + other

    def __sub__(self, other: object):
        return FakeTensor(_binary(raw_value(self), raw_value(other), lambda left, right: left - right))

    def __rsub__(self, other: object):
        return FakeTensor(_binary(raw_value(other), raw_value(self), lambda left, right: left - right))

    def __mul__(self, other: object):
        return FakeTensor(_binary(raw_value(self), raw_value(other), lambda left, right: left * right))

    def __rmul__(self, other: object):
        return self * other

    def __truediv__(self, other: object):
        return FakeTensor(_binary(raw_value(self), raw_value(other), lambda left, right: left / right))

    def __rtruediv__(self, other: object):
        return FakeTensor(_binary(raw_value(other), raw_value(self), lambda left, right: left / right))

    def __matmul__(self, other: object):
        vector = raw_value(self)
        matrix = raw_value(other)
        return FakeTensor(
            [
                sum(vector[row] * matrix[row][column] for row in range(len(vector)))
                for column in range(len(matrix[0]))
            ]
        )

    def pow(self, exponent: float):
        return FakeTensor(_map_unary(raw_value(self), lambda value: value**exponent))

    def sum(self):
        return FakeTensor(sum(raw_value(self)))

    def tolist(self):
        return _copy_raw(self.value)

    def backward(self) -> None:
        if not self._backward_populates_grad:
            return None
        for target in self._active_grad_targets:
            target.grad = FakeTensor(_zeros_like(target.value))
        return None


class FakeAdamW:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.param_groups = [
            {
                "params": list(args[0]) if args else [],
                "lr": kwargs.get("lr", 0.0),
            }
        ]
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1
        for group in self.param_groups:
            for parameter in group["params"]:
                if hasattr(parameter, "grad"):
                    parameter.grad = None


def raw_value(value: object) -> object:
    return value.value if isinstance(value, FakeTensor) else value


def fake_clip_grad_value_(parameters: object, clip_value: float) -> None:
    for parameter in parameters:
        grad = getattr(parameter, "grad", None)
        if isinstance(grad, FakeTensor):
            grad.value = _clip(raw_value(grad), clip_value)


def _copy_raw(value: object) -> object:
    value = raw_value(value)
    if isinstance(value, list):
        return [_copy_raw(item) for item in value]
    return float(value) if isinstance(value, int | float) else value


def _clip(value: object, clip_value: float):
    if isinstance(value, list):
        return [_clip(item, clip_value) for item in value]
    return max(min(float(value), clip_value), -clip_value)


def _binary(left: object, right: object, op: Any):
    if isinstance(left, list) and isinstance(right, list):
        return [
            _binary(left_item, right_item, op)
            for left_item, right_item in zip(left, right)
        ]
    if isinstance(left, list):
        return [_binary(item, right, op) for item in left]
    if isinstance(right, list):
        return [_binary(left, item, op) for item in right]
    return op(float(left), float(right))


def _map_unary(value: object, op: Any):
    if isinstance(value, list):
        return [_map_unary(item, op) for item in value]
    return op(float(value))


def _zeros_like(value: object):
    if isinstance(value, list):
        return [_zeros_like(item) for item in value]
    return 0.0


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]
