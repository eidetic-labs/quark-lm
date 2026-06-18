from __future__ import annotations

import math
import types
from typing import Any, Callable


def fake_torch_importer(
    *,
    cuda: bool = False,
    mps: bool = False,
) -> Callable[[str], object]:
    fake = types.SimpleNamespace(
        __version__="fake-torch",
        float32="float32",
        float64="float64",
        tensor=lambda value, dtype=None, device=None: FakeTensor(value),
        stack=lambda values: FakeTensor([raw_value(value) for value in values]),
        tanh=lambda value: FakeTensor(_map_unary(raw_value(value), math.tanh)),
        softmax=lambda value, dim=0: FakeTensor(_softmax(raw_value(value))),
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps),
        ),
    )

    def importer(name: str) -> object:
        if name != "torch":
            raise ModuleNotFoundError(name)
        return fake

    return importer


class FakeTensor:
    def __init__(self, value: object) -> None:
        self.value = _copy_raw(value)

    def __iter__(self):
        return (FakeTensor(item) for item in self.value)

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, key: int | slice):
        return FakeTensor(self.value[key])

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


def raw_value(value: object) -> object:
    return value.value if isinstance(value, FakeTensor) else value


def _copy_raw(value: object) -> object:
    value = raw_value(value)
    if isinstance(value, list):
        return [_copy_raw(item) for item in value]
    return float(value) if isinstance(value, int | float) else value


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


def _softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]
