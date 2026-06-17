"""Generate and audit admission probes from the admitted-memory log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from admission_probe_audits import (
    audit_admission_paraphrase_probes,
    audit_admission_probes,
    audit_all_admission_probes,
)
from admission_probe_paths import (
    DEFAULT_ADMISSIONS,
    DEFAULT_OUTPUT,
    DEFAULT_PARAPHRASE_OUTPUT,
)
from admission_probe_records import (
    admission_paraphrase_probe_records,
    admission_probe_records,
)
from admission_probe_sync import (
    sync_admission_paraphrase_probes,
    sync_admission_probes,
    sync_all_admission_probes,
    write_jsonl,
)


__all__ = [
    "DEFAULT_ADMISSIONS",
    "DEFAULT_OUTPUT",
    "DEFAULT_PARAPHRASE_OUTPUT",
    "admission_paraphrase_probe_records",
    "admission_probe_records",
    "audit_admission_paraphrase_probes",
    "audit_admission_probes",
    "audit_all_admission_probes",
    "parse_args",
    "main",
    "sync_admission_paraphrase_probes",
    "sync_admission_probes",
    "sync_all_admission_probes",
    "write_jsonl",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admissions", type=Path, default=DEFAULT_ADMISSIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--paraphrases-output",
        type=Path,
        default=DEFAULT_PARAPHRASE_OUTPUT,
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        result = audit_all_admission_probes(
            args.admissions,
            args.output,
            args.paraphrases_output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    result = sync_all_admission_probes(
        args.admissions,
        args.output,
        args.paraphrases_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
