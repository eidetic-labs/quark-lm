"""Route baseline-floor accepted attempts into audit artifacts."""

from __future__ import annotations

from typing import Any

from transformer_baseline_floor_acceptance import (
    record_baseline_floor_profile_acceptance,
)
from transformer_baseline_floor_acceptance_accounting_builder import (
    build_baseline_floor_profile_acceptance_accounting,
)
from transformer_baseline_floor_acceptance_attempt import (
    BaselineFloorProfileAcceptanceAttempt,
)
from transformer_baseline_floor_acceptance_sample_builder import (
    build_baseline_floor_profile_acceptance_sample,
)
from transformer_baseline_floor_acceptance_samples import (
    record_baseline_floor_profile_acceptance_sample,
)


def record_baseline_floor_profile_attempt_acceptance(
    update_guard: dict[str, Any],
    attempt: BaselineFloorProfileAcceptanceAttempt,
) -> None:
    record_baseline_floor_profile_acceptance(
        update_guard,
        build_baseline_floor_profile_acceptance_accounting(attempt),
    )
    record_baseline_floor_profile_acceptance_sample(
        update_guard,
        build_baseline_floor_profile_acceptance_sample(attempt),
    )


__all__ = [
    "BaselineFloorProfileAcceptanceAttempt",
    "record_baseline_floor_profile_attempt_acceptance",
]
