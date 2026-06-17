---
title: Closed-World Verifier
description: Deterministic verifier checks for candidate and training-plan approval.
---

# Closed-World Verifier

The closed-world verifier (`src/closed_world_verifier.py`, introduced in v0.76)
is the deterministic gate that decides whether a candidate record or a training
plan is allowed to influence the next learning step. It produces a required
`closed_world_verifier.json` artifact for self-improvement answer cycles and
transformer answer-training runs.

The verifier answers one narrow question: did the proposed update stay inside
the closed-world data boundary? It does not judge whether the model answered
better. Quality is decided later, by
[constraint-first promotion](./training-recipes.md).

## What deterministic means here

The verifier does not call an external model, consult hidden training data, or
grade with a pretrained judge. Every check is a comparison against fields the
run already wrote. Each report records `uses_external_model: false` and
`verifier_type: "deterministic_closed_world"`, and the report is re-validated
when it is written, so a malformed or inconsistent report cannot be emitted.

This matters because the things the verifier protects — the boundary between
admitted and unadmitted data, and the boundary between candidates and training
data — must be checkable without trusting another model's opinion.

## Where it sits

```text
candidate quarantine ─┐
corpus hygiene ───────┼─> closed-world verifier ─> closed_world_verifier.json
training plan ────────┘        (deterministic)            (pass / fail)
                                                              │
                                                              v
                                          constraint-first promotion gate
```

The verifier reads artifacts the run produced and emits a pass/fail report. A
failing report blocks the run from being trusted; it is not discarded. See
[Candidate quarantine](./candidate-quarantine.md) for the candidate lifecycle
the verifier inspects, and [Training recipes](./training-recipes.md) for the
promotion gate downstream of it.

## What it checks

The verifier runs two kinds of check: one for the training plan as a whole, and
one for each candidate record in the quarantine manifest.

### Training-plan checks

| Check | Passes when |
| --- | --- |
| `training_plan_valid` | The plan is a structured artifact with schema version, kind, component, and run identity. |
| `closed_world_data_boundary` | The boundary flags `pretrained_weights`, `pretrained_tokenizer`, `external_embeddings`, and `unledgered_training_text` are all `false`. |
| `candidate_records_excluded` | The plan declares that candidate records are not training data. |
| `no_candidate_examples_in_training` | Training examples contain zero candidate-sourced examples. |
| `candidate_quarantine_declared` | The plan names the candidate quarantine artifact and its summary. |
| `candidate_quarantine_passes` | The candidate quarantine manifest passes its own verifier checks. |
| `protected_train_eval_overlap_passes` | Protected eval prompts do not overlap training examples or training text. |
| `verifier_artifact_planned` | A run that writes verifier evidence declared that artifact before training. |

The boundary check is the same line drawn by the
[Purity boundary](../secure/purity-boundary.md): no pretrained weights, no
pretrained tokenizer, no external embeddings, no unledgered training text.

### Candidate-record checks

Each candidate is checked for schema validity, a non-empty source label, the
payload fields its type requires, and a ledger admission link when its state is
training-eligible. The rule behind the last check is the one
[Candidate quarantine](./candidate-quarantine.md) enforces: generated material
is not training data until it is admitted to `corpus/ledger.json`.

When a prompt-target candidate is supplied together with a `CorpusResponder`,
the verifier also runs an `exact_answer_consistency` check: the candidate's
target must match the answer the deterministic responder — trained only from
admitted text — produces for that prompt. Without a responder this check is
recorded as inconclusive rather than silently passed.

The manifest-level checks confirm that the declared candidate counts match the
candidate list, that every candidate record passes, and that no
training-eligible candidate is missing an admission id.

## The report it writes

Each check records a `name`, a pass/fail `status`, the `rule` it enforces, and
the `details` it observed. The enclosing report is `passed` only when its
`failed_checks` list is empty. Because the report carries the failing check
names and their observed details, a rejected run is auditable: the report says
which boundary was crossed, not merely that something failed.

## Run artifacts

Self-improvement attempts write:

| Path | Contents |
| --- | --- |
| `attempts/attempt-###/closed_world_verifier.json` | The verifier report for that attempt. |
| `closed_world_verifier.json` | The report at the latest run level. |
| `training_plan.json` | The verifier path and summary. |
| `self_improvement_report.json` | The verifier evidence, embedded. |

Transformer answer-training runs write:

| Path | Contents |
| --- | --- |
| `closed_world_verifier.json` | The verifier report for the run. |
| `training_plan.json` | The verifier path and summary. |
| `transformer_answer_metrics.json` | The verifier evidence, embedded. |

## Boundary

Verifier approval is not a claim that the model answered better. It is a claim
that the next weight update or screen stayed inside the closed-world data
boundary. Separating these two judgments is deliberate: a run can preserve the
boundary and still be rejected on quality, and a run cannot reach the quality
judgment at all until the verifier passes.

From v0.77 onward, [training recipes](./training-recipes.md) and
constraint-first promotion reports decide whether quality metrics are eligible
to affect promotion. The verifier is the first constraint in that order.
