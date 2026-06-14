from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SitesTest(unittest.TestCase):
    def test_shared_state_matches_marketing_current_evidence(self) -> None:
        state = json.loads((ROOT / "sites" / "shared" / "current-state.json").read_text())
        marketing = (ROOT / "sites" / "marketing" / "index.html").read_text(encoding="utf-8")

        self.assertIn(state["tagline"], marketing)
        self.assertIn(state["currentVersion"], marketing)
        self.assertIn(state["currentRun"], marketing)
        self.assertIn(state["directAdmissionProbes"], marketing)
        self.assertIn(state["admissionParaphraseProbes"], marketing)

    def test_marketing_site_is_static_not_docusaurus(self) -> None:
        package = json.loads((ROOT / "package.json").read_text())
        marketing = (ROOT / "sites" / "marketing" / "index.html").read_text(encoding="utf-8")

        self.assertEqual(package["scripts"]["marketing:build"], "node scripts/build-marketing.mjs")
        self.assertNotIn("Docusaurus", marketing)
        self.assertNotIn("__docusaurus", marketing)

    def test_docs_follow_learn_build_operate_secure_structure(self) -> None:
        docs_config = (ROOT / "sites" / "docs" / "docusaurus.config.js").read_text(encoding="utf-8")
        sidebar = (ROOT / "sites" / "docs" / "sidebars.js").read_text(encoding="utf-8")

        for section in ("Learn", "Build", "Operate", "Secure"):
            self.assertIn(section, docs_config)
            self.assertIn(section.lower(), sidebar)


if __name__ == "__main__":
    unittest.main()
