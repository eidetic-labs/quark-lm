"""Tiny scalar autodiff engine for dependency-free model experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable


class Scalar:
    """A scalar value with reverse-mode automatic differentiation."""

    def __init__(
        self,
        data: float,
        children: Iterable["Scalar"] = (),
        op: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(children)
        self._op = op
        self._backward = lambda: None

    def __add__(self, other: float | "Scalar") -> "Scalar":
        other = ensure_scalar(other)
        out = Scalar(self.data + other.data, (self, other), "+")

        def backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other: float | "Scalar") -> "Scalar":
        return self + other

    def __mul__(self, other: float | "Scalar") -> "Scalar":
        other = ensure_scalar(other)
        out = Scalar(self.data * other.data, (self, other), "*")

        def backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other: float | "Scalar") -> "Scalar":
        return self * other

    def __neg__(self) -> "Scalar":
        return self * -1.0

    def __sub__(self, other: float | "Scalar") -> "Scalar":
        return self + (-ensure_scalar(other))

    def __rsub__(self, other: float | "Scalar") -> "Scalar":
        return ensure_scalar(other) + (-self)

    def __truediv__(self, other: float | "Scalar") -> "Scalar":
        other = ensure_scalar(other)
        return self * other.pow(-1.0)

    def __rtruediv__(self, other: float | "Scalar") -> "Scalar":
        return ensure_scalar(other) * self.pow(-1.0)

    def pow(self, exponent: float) -> "Scalar":
        out = Scalar(self.data**exponent, (self,), f"**{exponent}")

        def backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1.0)) * out.grad

        out._backward = backward
        return out

    def tanh(self) -> "Scalar":
        value = math.tanh(self.data)
        out = Scalar(value, (self,), "tanh")

        def backward() -> None:
            self.grad += (1.0 - value * value) * out.grad

        out._backward = backward
        return out

    def exp(self) -> "Scalar":
        value = math.exp(self.data)
        out = Scalar(value, (self,), "exp")

        def backward() -> None:
            self.grad += value * out.grad

        out._backward = backward
        return out

    def log(self) -> "Scalar":
        out = Scalar(math.log(self.data), (self,), "log")

        def backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = backward
        return out

    def backward(self) -> None:
        topo: list[Scalar] = []
        visited: set[Scalar] = set()

        def build(node: Scalar) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def ensure_scalar(value: float | Scalar) -> Scalar:
    if isinstance(value, Scalar):
        return value
    return Scalar(value)


def zero_grad(parameters: Iterable[Scalar]) -> None:
    for parameter in parameters:
        parameter.grad = 0.0
