from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ProjectMetadataTests(unittest.TestCase):
    def test_pytorch_runtime_dependency_is_optional(self) -> None:
        project = _project_metadata()

        required_dependencies = project.get("dependencies", [])
        optional = project.get("optional-dependencies", {})

        self.assertNotIn("torch", " ".join(required_dependencies).lower())
        self.assertEqual(optional["pytorch"], ["torch>=2.0"])


def _project_metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]
