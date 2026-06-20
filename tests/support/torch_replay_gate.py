from __future__ import annotations

from transformer_torch_training_replay_parity_gate import (
    build_torch_training_replay_parity_gate,
)


def matched_replay_gate() -> dict:
    return build_torch_training_replay_parity_gate(
        runtime=_runtime(),
        readiness=_readiness("ready"),
        probes=_matching_probes(),
    )


def pending_replay_gate() -> dict:
    probes = _matching_probes()
    probes["accumulation_replay_buffer_comparison"] = {
        **probes["accumulation_replay_buffer_comparison"],
        "passed": False,
        "status": "replay_buffer_signature_mismatch",
        "buffered_gradient_parity_proven": False,
    }
    return build_torch_training_replay_parity_gate(
        runtime=_runtime(),
        readiness=_readiness("ready"),
        probes=probes,
    )


def _runtime() -> dict:
    return {
        "available": True,
        "runtime_kind": "pytorch",
        "dtype_available": True,
    }


def _readiness(status: str) -> dict:
    return {"status": status}


def _matching_probes() -> dict:
    return {
        "initial_loss_probe": {"status": "matched"},
        "backward_probe": {"status": "gradients_available"},
        "optimizer_step_probe": {"status": "ready_for_optimizer_execution"},
        "optimizer_step_execution_probe": {"status": "step_control_matched"},
        "accumulation_replay_control_probe": {
            "schema_version": 1,
            "status": "accumulation_replay_control_recorded",
            "gradient_signature_match_count": 2,
            "gradient_signature_mismatch_count": 0,
            "planned_microstep_count": 2,
            "executed_microstep_count": 2,
            "backward_pass_count": 2,
            "microsteps": [{"step": 1}, {"step": 2}],
        },
        "accumulation_replay_buffer_comparison": {
            "schema_version": 1,
            "passed": True,
            "status": "replay_buffer_signature_matched",
            "buffered_gradient_parity_proven": True,
        },
        "accumulation_replay_update_comparison": {
            "schema_version": 1,
            "passed": True,
            "status": "replay_update_signature_matched",
            "optimizer_update_parity_proven": True,
        },
        "accumulation_replay_final_evaluation": {
            "schema_version": 1,
            "passed": True,
            "status": "replay_final_evaluation_matched",
            "final_logit_parity_proven": True,
            "final_loss_parity_proven": True,
        },
        "accumulation_replay_checkpoint_compatibility": {
            "schema_version": 1,
            "passed": True,
            "status": "replay_checkpoint_compatible",
            "checkpoint_parity_proven": True,
        },
    }
