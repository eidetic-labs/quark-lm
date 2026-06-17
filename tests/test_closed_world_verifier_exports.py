import unittest

import closed_world_verifier
from closed_world_candidate_verifier import (
    verify_candidate_quarantine_manifest,
    verify_candidate_record,
)
from closed_world_training_plan_verifier import verify_training_plan
from closed_world_verifier_reports import (
    FAIL,
    PASS,
    REPORT_KIND,
    SCHEMA_VERSION,
    attach_verifier_summary,
    validate_verifier_report,
    verifier_check,
    verifier_report,
    verifier_report_summary,
    write_verifier_report,
)


class ClosedWorldVerifierExportsTest(unittest.TestCase):
    def test_compatibility_module_reexports_focused_verifier_apis(self) -> None:
        self.assertEqual(closed_world_verifier.SCHEMA_VERSION, SCHEMA_VERSION)
        self.assertEqual(closed_world_verifier.REPORT_KIND, REPORT_KIND)
        self.assertEqual(closed_world_verifier.PASS, PASS)
        self.assertEqual(closed_world_verifier.FAIL, FAIL)
        self.assertIs(
            closed_world_verifier.attach_verifier_summary,
            attach_verifier_summary,
        )
        self.assertIs(
            closed_world_verifier.validate_verifier_report,
            validate_verifier_report,
        )
        self.assertIs(closed_world_verifier.verifier_check, verifier_check)
        self.assertIs(closed_world_verifier.verifier_report, verifier_report)
        self.assertIs(
            closed_world_verifier.verifier_report_summary,
            verifier_report_summary,
        )
        self.assertIs(
            closed_world_verifier.write_verifier_report,
            write_verifier_report,
        )
        self.assertIs(
            closed_world_verifier.verify_candidate_quarantine_manifest,
            verify_candidate_quarantine_manifest,
        )
        self.assertIs(
            closed_world_verifier.verify_candidate_record,
            verify_candidate_record,
        )
        self.assertIs(
            closed_world_verifier.verify_training_plan,
            verify_training_plan,
        )


if __name__ == "__main__":
    unittest.main()
