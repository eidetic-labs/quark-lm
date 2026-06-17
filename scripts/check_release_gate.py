from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SEMVER_RE = re.compile(
    r"^v?"
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)

PUBLIC_SURFACE_FILES = [
    "README.md",
    "STATUS.md",
    "RC_DECISION.md",
    "RC_SPEC.md",
    "RC_GAP_AUDIT.md",
    "RC_CHECKLIST.md",
    "sites/docs/docs/learn/project-overview.md",
    "sites/docs/docs/operate/release-candidate.md",
    "sites/docs/docs/learn/current-evidence.mdx",
    "sites/marketing/index.html",
    "sites/shared/current-state.json",
]

STALE_RELEASE_CLAIMS = [
    "audited ready for a tag decision",
    "prototype ready",
    "Package an honest Research Prototype RC first",
    "Research Prototype RC is audited ready",
]

IGNORED_PARTS = {
    ".git",
    ".venv",
    "node_modules",
    "runs",
    "sites/docs/build",
    "sites/marketing/build",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    path: str
    message: str


def semver_without_v(value: str) -> str:
    if value.startswith("v"):
        return value[1:]
    return value


def validate_semver(value: str) -> str:
    match = SEMVER_RE.match(value)
    if not match:
        raise ValueError(
            f"{value!r} is not SemVer. Use MAJOR.MINOR.PATCH with optional "
            "prerelease/build metadata, for example v0.115.0-alpha.1."
        )

    prerelease = match.group(4)
    if prerelease:
        for identifier in prerelease.split("."):
            if identifier.isdigit() and len(identifier) > 1 and identifier.startswith("0"):
                raise ValueError(f"{value!r} has a prerelease numeric identifier with a leading zero.")

    return semver_without_v(value)


def is_ignored(path: Path) -> bool:
    parts = set(path.parts)
    if parts & {".git", ".venv", "node_modules", "runs"}:
        return True
    rel = path.as_posix()
    return any(rel.startswith(prefix) for prefix in ("sites/docs/build", "sites/marketing/build"))


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def collect_srp_findings(
    root: Path,
    source_limit: int = 500,
    test_limit: int = 1000,
    p0_limit: int = 1000,
) -> list[Finding]:
    findings: list[Finding] = []
    suffixes = {".py", ".js", ".mjs"}
    for base in ("src", "tests", "scripts"):
        base_path = root / base
        if not base_path.exists():
            continue
        for path in base_path.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes or is_ignored(path.relative_to(root)):
                continue
            lines = count_lines(path)
            rel = path.relative_to(root).as_posix()
            if lines > p0_limit:
                findings.append(Finding("P0", rel, f"{lines} lines exceeds the {p0_limit}-line P0 SRP ceiling."))
            elif rel.startswith("tests/") and lines > test_limit:
                findings.append(Finding("P1", rel, f"{lines} lines exceeds the {test_limit}-line test ceiling."))
            elif not rel.startswith("tests/") and lines > source_limit:
                findings.append(Finding("P1", rel, f"{lines} lines exceeds the {source_limit}-line source ceiling."))
    return sorted(findings, key=lambda item: (item.severity, item.path))


def tracked_generated_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return [f"git ls-files failed: {result.stderr.strip()}"]

    generated: list[str] = []
    for line in result.stdout.splitlines():
        parts = set(Path(line).parts)
        if (
            line.endswith(".pyc")
            or line.endswith(".DS_Store")
            or "__pycache__" in parts
            or line.startswith("node_modules/")
            or line.startswith("runs/")
            or line.startswith("sites/docs/build/")
            or line.startswith("sites/marketing/build/")
        ):
            generated.append(line)
    return generated


def validate_shared_state(root: Path) -> list[str]:
    path = root / "sites" / "shared" / "current-state.json"
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact parser detail is not useful.
        return [f"{path.relative_to(root)}: {exc}"]
    return []


def stale_release_claims(root: Path) -> list[str]:
    hits: list[str] = []
    for rel in PUBLIC_SURFACE_FILES:
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for claim in STALE_RELEASE_CLAIMS:
            if claim in text:
                hits.append(f"{rel}: stale claim {claim!r}")
    return hits


def print_section(title: str, lines: list[str]) -> None:
    print(title)
    if not lines:
        print("  ok")
        return
    for line in lines:
        print(f"  - {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate QuarkLM release-gate hygiene.")
    parser.add_argument("--version", default="v0.115.0-alpha.1", help="Proposed SemVer release/tag.")
    parser.add_argument("--alpha", action="store_true", help="Fail on alpha-blocking SRP findings.")
    parser.add_argument("--source-lines", type=int, default=500)
    parser.add_argument("--test-lines", type=int, default=1000)
    parser.add_argument("--p0-lines", type=int, default=1000)
    args = parser.parse_args(argv)

    blockers: list[str] = []
    warnings: list[str] = []

    try:
        version = validate_semver(args.version)
        semver_lines = [f"{args.version} -> {version}"]
    except ValueError as exc:
        semver_lines = [str(exc)]
        blockers.append(str(exc))

    generated = tracked_generated_files(ROOT)
    blockers.extend(generated)

    state_errors = validate_shared_state(ROOT)
    blockers.extend(state_errors)

    stale_claims = stale_release_claims(ROOT)
    blockers.extend(stale_claims)

    srp_findings = collect_srp_findings(
        ROOT,
        source_limit=args.source_lines,
        test_limit=args.test_lines,
        p0_limit=args.p0_lines,
    )
    srp_lines = [f"{item.severity} {item.path}: {item.message}" for item in srp_findings]
    if args.alpha:
        blockers.extend(srp_lines)
    else:
        warnings.extend(srp_lines)

    print_section("SemVer", semver_lines)
    print_section("Tracked generated files", generated)
    print_section("Shared state JSON", state_errors)
    print_section("Stale release claims", stale_claims)
    print_section("SRP alpha blockers" if args.alpha else "SRP alpha warnings", srp_lines)

    if warnings and not args.alpha:
        print("\nWarnings are not fatal outside --alpha mode.")

    if blockers:
        print("\nRelease gate failed.")
        return 1

    print("\nRelease gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
