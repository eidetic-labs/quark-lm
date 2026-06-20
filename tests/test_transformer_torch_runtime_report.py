from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support.fake_torch import fake_torch_importer
from transformer_torch_backend import (
    TORCH_RUNTIME_REPORT_CHECKS,
    TORCH_RUNTIME_REPORT_EVIDENCE_SCOPE,
    TORCH_RUNTIME_REPORT_KIND,
    TORCH_RUNTIME_REPORT_STATUSES,
    build_torch_runtime_report,
    validate_torch_runtime_report,
    write_torch_runtime_report,
)
from transformer_torch_runtime_report_check import build_torch_runtime_report_check
from transformer_torch_runtime_report import main


class TransformerTorchRuntimeReportTests(unittest.TestCase):
    def test_report_blocks_missing_runtime(self) -> None:
        report = build_torch_runtime_report(importer=_missing_importer)

        self.assertFalse(report["passed"])
        self.assertEqual(report["kind"], TORCH_RUNTIME_REPORT_KIND)
        self.assertEqual(report["status"], "blocked_runtime_unavailable")
        self.assertEqual(report["evidence_scope"], "runtime_preflight_only")
        self.assertFalse(report["parity_attempt_allowed"])
        self.assertFalse(report["training_evidence_allowed"])
        self.assertEqual(
            report["summary"]["failed_checks"],
            ["runtime_available", "runtime_kind", "dtype_available"],
        )
        self.assertFalse(
            report["closed_world_boundary"]["pretrained_weights_imported"]
        )
        validate_torch_runtime_report(report)

    def test_report_blocks_test_double_runtime(self) -> None:
        report = build_torch_runtime_report(importer=fake_torch_importer())

        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "blocked_test_double_runtime")
        self.assertEqual(report["summary"]["failed_checks"], ["runtime_kind"])
        self.assertIn("test doubles", report["reason"])
        validate_torch_runtime_report(report)

    def test_report_allows_real_pytorch_shaped_runtime(self) -> None:
        report = build_torch_runtime_report(importer=_real_like_importer)

        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "ready_for_pytorch_parity")
        self.assertTrue(report["parity_attempt_allowed"])
        self.assertTrue(report["training_evidence_allowed"])
        self.assertEqual(report["summary"]["failed_checks"], [])
        json.dumps(report, sort_keys=True)
        validate_torch_runtime_report(report)
        self.assertIn("ready_for_pytorch_parity", TORCH_RUNTIME_REPORT_STATUSES)
        self.assertEqual(
            TORCH_RUNTIME_REPORT_CHECKS,
            ("runtime_available", "runtime_kind", "dtype_available"),
        )
        self.assertEqual(TORCH_RUNTIME_REPORT_EVIDENCE_SCOPE, "runtime_preflight_only")

    def test_writer_and_main_emit_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "torch-runtime.json"
            write_torch_runtime_report(
                output,
                build_torch_runtime_report(importer=_real_like_importer),
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--requested-dtype",
                        "quark_missing_dtype",
                        "--output",
                        str(output),
                    ]
                )
            cli_payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["kind"], TORCH_RUNTIME_REPORT_KIND)
        self.assertEqual(exit_code, 1)
        self.assertFalse(cli_payload["passed"])
        self.assertFalse(cli_payload["parity_attempt_allowed"])
        self.assertFalse(cli_payload["training_evidence_allowed"])

    def test_validator_rejects_stale_summary(self) -> None:
        report = build_torch_runtime_report(importer=_real_like_importer)
        report["summary"]["failed_checks"] = ["runtime_kind"]

        with self.assertRaisesRegex(ValueError, "failed_checks"):
            validate_torch_runtime_report(report)

    def test_validator_rejects_extra_report_key(self) -> None:
        report = build_torch_runtime_report(importer=_real_like_importer)
        report["unvalidated_extra_field"] = "drift"

        with self.assertRaisesRegex(ValueError, "runtime_report keys"):
            validate_torch_runtime_report(report)

    def test_validator_rejects_extra_runtime_key(self) -> None:
        report = build_torch_runtime_report(importer=_real_like_importer)
        report["runtime"]["unvalidated_extra_field"] = "drift"

        with self.assertRaisesRegex(ValueError, "runtime keys"):
            validate_torch_runtime_report(report)

    def test_validator_rejects_extra_summary_key(self) -> None:
        report = build_torch_runtime_report(importer=_real_like_importer)
        report["summary"]["unvalidated_extra_field"] = "drift"

        with self.assertRaisesRegex(ValueError, "summary keys"):
            validate_torch_runtime_report(report)

    def test_validator_rejects_extra_boundary_key(self) -> None:
        report = build_torch_runtime_report(importer=_real_like_importer)
        report["closed_world_boundary"]["unvalidated_extra_field"] = "drift"

        with self.assertRaisesRegex(ValueError, "closed_world_boundary keys"):
            validate_torch_runtime_report(report)

    def test_validator_rejects_extra_check_key(self) -> None:
        report = build_torch_runtime_report(importer=_real_like_importer)
        report["checks"][0]["unvalidated_extra_field"] = "drift"

        with self.assertRaisesRegex(ValueError, "runtime_available.keys"):
            validate_torch_runtime_report(report)

    def test_validator_rejects_stale_status(self) -> None:
        report = build_torch_runtime_report(importer=_missing_importer)
        report["status"] = "ready_for_pytorch_parity"

        with self.assertRaisesRegex(ValueError, "status"):
            validate_torch_runtime_report(report)

    def test_validator_rejects_dirty_closed_world_boundary(self) -> None:
        report = build_torch_runtime_report(importer=_real_like_importer)
        report["closed_world_boundary"]["pretrained_weights_imported"] = True

        with self.assertRaisesRegex(ValueError, "pretrained_weights_imported"):
            validate_torch_runtime_report(report)

    def test_runtime_report_check_rejects_malformed_report(self) -> None:
        report = build_torch_runtime_report(importer=_real_like_importer)
        report["passed"] = False

        check = build_torch_runtime_report_check(
            runtime_report=report,
            runtime=report["runtime"],
        )

        self.assertFalse(check["passed"])
        self.assertIn("runtime_report.", check["error"])


def _missing_importer(name: str) -> object:
    raise ModuleNotFoundError(name)


def _real_like_importer(name: str) -> object:
    if name != "torch":
        raise ModuleNotFoundError(name)
    return types.SimpleNamespace(
        __version__="2.0.0",
        float32="float32",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False),
        ),
    )


if __name__ == "__main__":
    unittest.main()
