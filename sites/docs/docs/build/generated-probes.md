---
title: Generated Probes
description: Probes generated from admitted memory and glossary sources.
---

# Generated Probes

Evaluation probes that check admitted knowledge are not written by hand. They are
generated from the admitted sources themselves, so the evals cannot drift away
from the facts the corpus actually ledgers. Two generators do this work:

- `admission_probes` builds answer probes from `corpus/admissions.jsonl`, the
  admitted-memory log.
- `glossary_probes` builds definition probes from the `probe_words` list in
  `corpus/glossary.json`, the admitted glossary.

A probe is an evaluation question, not training data. These files exercise both
rails — retrieval memory and the trained weights — against the same admitted
facts. Admitting a fact through the [admission workflow](./admission-workflow.md)
is what changes what is learnable; generating probes only changes what is
measured.

## Why generate instead of hand-write

Hand-written evals rot. A fact is admitted or edited in `corpus/`, the eval that
checks it is forgotten, and the suite slowly stops describing the corpus it
claims to test. Generating probes from the admitted source removes that gap by
construction: the expected probe set is a deterministic function of the admitted
facts, so any divergence is a bug to be fixed rather than a judgement call.

This is the same anti-drift discipline the docs apply to prose. The generated
files are an artifact of the admitted corpus, not an independent input to it.

## What gets generated

Each generator reads its admitted source and writes a fixed expansion of probes
per source record.

| File | Generated from | Per source record |
| --- | --- | --- |
| `evals/admissions.jsonl` | `corpus/admissions.jsonl` | Place, color, owner, and training-data-status probes. |
| `evals/admission_paraphrases.jsonl` | `corpus/admissions.jsonl` | Alternate surface forms of the same admitted facts. |
| `evals/glossary.jsonl` | `corpus/glossary.json` probe words | A "what does X mean" probe and a "define X" probe. |

The training-data-status probe is deliberate. For each admitted fact it asks
whether that fact is part of the training data and expects the answer `yes` —
because the fact was admitted to `corpus/ledger.json`. The paraphrase file widens
coverage to alternate phrasings of the admitted facts, so a passing screen
reflects the fact rather than one memorized prompt string.

## Current counts

Counts are derived from the current admitted sources, so they move only when the
corpus does.

| Generated probes | Count | Source records |
| --- | --- | --- |
| Direct admission probes | `48` | `12` admitted facts |
| Admission paraphrase probes | `84` | `12` admitted facts |
| Glossary probes | `38` | `19` glossary probe words |

Four direct probes per admitted fact and two probes per glossary probe word
account for these totals. Adding a fact or a probe word raises the counts on the
next sync; nothing else does.

## Check that the evals match the corpus

Both generators run in two modes. With no flag they regenerate the eval files
from the admitted sources. With `--check` they regenerate the expected probe set
in memory, compare it against the files on disk by probe id, and report any
`missing`, `extra`, or `mismatched` ids without writing.

```bash
# regenerate the eval files from admitted sources
PYTHONPATH=src python3 -m admission_probes
PYTHONPATH=src python3 -m glossary_probes

# verify the checked-in files still match the corpus (no writes)
PYTHONPATH=src python3 -m admission_probes --check
PYTHONPATH=src python3 -m glossary_probes --check
```

A `--check` run exits non-zero when the files have drifted from the admitted
sources, which makes it usable as a guard. The
[admit](./admission-workflow.md) command keeps the direct and paraphrase
admission probes in sync automatically when it writes to the default
`corpus/admissions.jsonl`, unless `--no-sync-probes` is passed; `--check` then
confirms that sync after the fact rather than trusting it.

## How the audits gate promotion

The [self-improvement report](../learn/self-improvement-loop.md) carries two
audits built from these probes: a combined `admission_probe_audit` over the
direct and paraphrase results, and a separate `glossary_probe_audit`. Each audit
reports whether its generated files still match the admitted sources, and both
must pass before a run can be promoted. They sit alongside the
[closed-world verifier](../operate/closed-world-verifier.md) among the
constraint-first promotion checks, so a screen whose evals have drifted from the
corpus is rejected before any quality number is allowed to count.

## Rule

Generated probes are evaluation material, not training input. They are derived
from admitted corpus files and checked into the repo so evals stay honest, but
they are not training data unless explicitly admitted to the ledger.
