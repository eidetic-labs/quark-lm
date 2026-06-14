from __future__ import annotations

import json
import re
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

    def test_sidebar_doc_links_exist(self) -> None:
        sidebar = (ROOT / "sites" / "docs" / "sidebars.js").read_text(encoding="utf-8")
        doc_ids = [
            match
            for match in re.findall(r"'([^']+)'", sidebar)
            if "/" in match and not match.startswith("@")
        ]

        self.assertGreater(len(doc_ids), 0)
        for doc_id in doc_ids:
            md_path = ROOT / "sites" / "docs" / "docs" / f"{doc_id}.md"
            mdx_path = ROOT / "sites" / "docs" / "docs" / f"{doc_id}.mdx"
            self.assertTrue(
                md_path.exists() or mdx_path.exists(),
                f"Missing Docusaurus doc for sidebar id {doc_id!r}",
            )


if __name__ == "__main__":
    unittest.main()
