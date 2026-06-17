"""Scalar conversion and parameter filtering helpers."""

from __future__ import annotations

from typing import Any

from autograd import Scalar


def matrix_to_scalars(values: list[list[float]]) -> list[list[Scalar]]:
    return [[Scalar(value) for value in row] for row in values]


def vector_to_scalars(values: list[float]) -> list[Scalar]:
    return [Scalar(value) for value in values]


def flatten_scalars(item: Any) -> list[Scalar]:
    if isinstance(item, Scalar):
        return [item]
    scalars: list[Scalar] = []
    for value in item:
        scalars.extend(flatten_scalars(value))
    return scalars


def exclude_scalars(params: list[Scalar], excluded: Any) -> list[Scalar]:
    excluded_ids = {id(value) for value in flatten_scalars(excluded)}
    return [param for param in params if id(param) not in excluded_ids]


def matrix_to_floats(values: list[list[Scalar]]) -> list[list[float]]:
    return [[value.data for value in row] for row in values]


def vector_to_floats(values: list[Scalar]) -> list[float]:
    return [value.data for value in values]
