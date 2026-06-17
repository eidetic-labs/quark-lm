from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_release_gate  # noqa: E402


class ReleaseGateTest(unittest.TestCase):
    def test_srp_default_limits_match_methodology_gate(self) -> None:
        self.assertEqual(check_release_gate.DEFAULT_SRP_REVIEW_LINES, 250)
        self.assertEqual(check_release_gate.DEFAULT_SRP_P0_LINES, 500)

    def test_semver_accepts_alpha_tag_with_v_prefix(self) -> None:
        self.assertEqual(check_release_gate.validate_semver("v0.115.0-alpha.1"), "0.115.0-alpha.1")

    def test_semver_rejects_non_semver_shapes(self) -> None:
        for value in ("v0.115", "v1.00", "0.115.0-alpha.01", "0.115.0-"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    check_release_gate.validate_semver(value)

    def test_srp_finding_reports_oversized_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src" / "demo"
            source.mkdir(parents=True)
            path = source / "large_module.py"
            path.write_text("\n".join("pass" for _ in range(6)), encoding="utf-8")

            findings = check_release_gate.collect_srp_findings(
                root,
                source_limit=5,
                test_limit=100,
                p0_limit=1000,
            )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, "P1")
        self.assertEqual(findings[0].path, "src/demo/large_module.py")


if __name__ == "__main__":
    unittest.main()
