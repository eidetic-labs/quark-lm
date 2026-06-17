---
title: Admission Workflow
description: Admit new knowledge before it can be trained.
---

# Admission Workflow

Admission is how a new fact enters QuarkLM's closed world. A fact is not training
data because someone typed it in chat. It becomes training-eligible only after it
is admitted to the ledgered corpus, and it becomes learned weight behavior only
after a guarded update is evaluated and promoted. Admission is the first of those
steps and the only one this page covers.

The `admit` module appends a fact to `corpus/admissions.jsonl`, which is named in
`corpus/ledger.json` as the `admissions-v0` source and marked allowed for both
curriculum generation and training. Writing the fact through `admit` is what
makes it part of the admitted corpus; the [purity boundary](../secure/purity-boundary.md)
is the reason no other path may introduce training text.

## What admission does, and does not do

| Admission does | Admission does not |
| --- | --- |
| Append a validated fact to the admitted corpus. | Move any neural weights. |
| Reject duplicate ids before writing. | Train, evaluate, or promote anything. |
| Regenerate admission probes from the new corpus (default paths). | Regenerate `build/train.txt` or memory cards. |
| Make the fact curriculum- and training-*eligible*. | Make the fact `weight-consolidated`. |

Admission writes one log line. Everything downstream — curriculum text,
retrieval memory cards, candidate proposals, guarded updates — is run separately
and audited separately. An admitted fact is `memory-served` once curriculum and
retrieval are rebuilt; it is `weight-consolidated` only if a later run promotes a
guarded update that learned it. Those two states are kept distinct everywhere;
see [Build](./index.md).

## Admit one fact

Run from the project root with `PYTHONPATH=src` set:

```bash
PYTHONPATH=src python3 -m admit \
  --id learned-child-book \
  --person child \
  --object book \
  --color blue \
  --relation on \
  --container table
```

## Admit a batch

```bash
PYTHONPATH=src python3 -m admit \
  --batch path/to/new_admissions.jsonl
```

Each line of the batch file is one fact in the same shape as the single-fact
fields.

## The fact contract

Every admitted fact carries the same six required fields. The id may contain
lowercase letters, digits, and hyphens, and must start with a letter; the other
five fields must be lowercase letters only.

| Field | Meaning | Allowed characters |
| --- | --- | --- |
| `id` | Stable admission id, unique across the corpus. | `[a-z][a-z0-9-]*` |
| `person` | Who the fact is about. | lowercase letters |
| `object` | The thing. | lowercase letters |
| `color` | Its color. | lowercase letters |
| `relation` | Where it sits. | lowercase letters |
| `container` | What it sits on or in. | lowercase letters |

A fact missing any required field is rejected, as is any field whose characters
fall outside the pattern. This keeps admitted facts inside the nursery
vocabulary the corpus tokenizer learns from.

## Duplicate ids are rejected

Ids are the corpus's primary key. Before writing, `admit` rejects an id that
already exists in `corpus/admissions.jsonl` and rejects an id that appears more
than once within the same batch. Nothing is written when a duplicate is found,
so a rejected batch leaves the corpus unchanged.

## Probe sync

When the admission writes to the default `corpus/admissions.jsonl`, `admit` also
regenerates the direct and paraphrase admission probes so admitted-memory evals
stay aligned with admitted facts. Pass `--no-sync-probes` to skip that
regeneration, or `--path` to write to a non-default file (in which case probes
are not synced unless probe paths are given explicitly). How those probes are
generated and checked is described in [Generated probes](./generated-probes.md).

## What admission reports

The command prints the admitted record(s) and a status block. The status is
`admitted_pending_weight_update`: the fact is in the corpus and answerable once
curriculum and memory are rebuilt, but no weights have changed. The reported next
step is to run a self-improvement answer cycle, which is where a guarded update is
attempted and a promotion gate decides whether anything is kept.

```text
admit -> corpus/admissions.jsonl  (admitted, no weights moved)
      -> regenerate curriculum + memory  (memory-served)
      -> self_improve answer-cycle  (guarded update, may be rejected)
      -> promotion gate  (weight-consolidated only if promoted)
```

## After admission

Rebuild the curriculum and retrieval artifacts so the fact is answerable, then
run a self-improvement cycle that compares against the last promoted report:

```bash
PYTHONPATH=src python3 -m self_improve answer-cycle \
  --run runs/self-improve-next \
  --compare-report runs/self-improve-v0.38/self_improvement_report.json
```

The cycle does not promote a run because it completed. Candidates built from the
admitted fact are held in [candidate quarantine](../operate/candidate-quarantine.md)
until the [closed-world verifier](../operate/closed-world-verifier.md) clears
them, and any guarded update is kept only if the recorded promotion gate passes.
See [Quickstart](./quickstart.md) for the full run sequence and
[Self-improvement loop](../learn/self-improvement-loop.md) for the lifecycle
contract this admission feeds.
