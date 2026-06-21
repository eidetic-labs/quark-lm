"""Ledger guard: the abstention decision contract is pinned + auditable.

Reliable abstention is the closed-world LM's entire thesis, so the rule that
decides "did the model abstain?" is its load-bearing definition. Today that rule
is pure argmin over the per-type candidate menu (margin 0.0). This guard pins the
ledgered contract -- target string, fallback completion, decision rule, and the
calibration margin -- so any change to abstention semantics turns CI red and
becomes a deliberate, reviewed recalibration rather than silent drift. The same
ledger is emitted into eval provenance (epistemic_eval_runner) for auditability.
"""

from __future__ import annotations

import unittest

import support  # noqa: F401  (puts src/ on sys.path)

from epistemic_abstention import (
    ABSTAIN_COMPLETION,
    ABSTAIN_TARGET,
    ABSTENTION_DECISION,
    ABSTENTION_MARGIN,
    _model_abstained,
    abstention_ledger,
)


class AbstentionLedgerTest(unittest.TestCase):
    def test_ledger_contract_is_pinned(self) -> None:
        ledger = abstention_ledger()
        self.assertEqual(ledger["target"], " unknown.")
        self.assertEqual(ledger["completion"], "unknown.")
        self.assertEqual(
            ledger["decision"], "argmin_nll_over_per_type_menu_or_greedy_completion"
        )
        # margin 0.0 == pure argmin; a nonzero margin is a deliberate recalibration.
        self.assertEqual(ledger["margin"], 0.0)

    def test_ledger_matches_module_constants(self) -> None:
        ledger = abstention_ledger()
        self.assertEqual(ledger["target"], ABSTAIN_TARGET)
        self.assertEqual(ledger["completion"], ABSTAIN_COMPLETION)
        self.assertEqual(ledger["decision"], ABSTENTION_DECISION)
        self.assertEqual(ledger["margin"], ABSTENTION_MARGIN)

    def test_decision_rule_honors_the_ledger(self) -> None:
        # The decision the ledger documents is exactly what _model_abstained does:
        # abstain iff the argmin predicted_candidate is the target (or, with no
        # candidate set, the greedy completion is the bare abstain token).
        self.assertTrue(_model_abstained({"predicted_candidate": ABSTAIN_TARGET}))
        self.assertFalse(_model_abstained({"predicted_candidate": " in the bag."}))
        self.assertTrue(
            _model_abstained({"predicted_candidate": None, "completion": "unknown."})
        )
        self.assertFalse(
            _model_abstained({"predicted_candidate": None, "completion": " in the bag."})
        )


if __name__ == "__main__":
    unittest.main()
