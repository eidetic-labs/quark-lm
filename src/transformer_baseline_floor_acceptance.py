"""Compatibility facade for baseline-floor acceptance accounting."""

from __future__ import annotations

from transformer_baseline_floor_acceptance_core import (
    record_baseline_floor_profile_acceptance,
)
from transformer_baseline_floor_acceptance_types import (
    BaselineFloorProfileAcceptanceAccounting,
)


__all__ = [
    "BaselineFloorProfileAcceptanceAccounting",
    "record_baseline_floor_profile_acceptance",
]
