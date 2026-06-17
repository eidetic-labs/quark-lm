---
title: Experiment Registry
description: Run-intent artifacts for QuarkLM training evidence.
---

# Experiment Registry

v0.71 adds `src/experiment_registry.py` so a training run begins with an explicit,
validated intent rather than a loose command. The intent is written before the
weights move, and the same artifact is closed with the outcome after the
promotion gate has spoken. A run is therefore auditable against the plan it
declared, not against a description reconstructed afterward.

The registry is deliberately small: JSON artifacts, pure validation, and no
hidden promotion behavior. It records what a run intended; it does not decide
whether the run passed. Promotion is decided separately by the
[closed-world verifier](./closed-world-verifier.md) and the constraint-first
report in [training recipes](./training-recipes.md).

## The intent contract

Every experiment writes one `experiment_intent.json` record. Each field is
validated on write and on read, so a malformed or empty intent fails fast rather
than producing untrustworthy evidence.

| Field | Records |
| --- | --- |
| `version` / `run_id` | The QuarkLM version and the run the intent belongs to. |
| `component` | The component under test (for example, a transformer answer-training run). |
| `hypothesis` | The claim the run is meant to test. |
| `allowed_data_sources` | The data the run is permitted to draw from, stated up front. |
| `planned_artifacts` | The evidence files the run commits to emitting. |
| `training_recipe_id` | The reproducible recipe the run follows; see [training recipes](./training-recipes.md). |
| `acceptance_gates` | Named gates, each with a rule and a `required` flag, that the run must clear to promote. |
| `failure_criteria` | The conditions that mark the run a failure. |
| `replay_plan_id` | An optional replay-plan reference, or null. |
| `notes` | Free-text context. |
| `decision` | The closing outcome, written when the gate result is known. |

Required string fields must be non-empty; `allowed_data_sources`,
`planned_artifacts`, and `failure_criteria` must each be non-empty lists of
non-empty strings. Acceptance gate names must be unique. Declaring allowed data
sources in the intent is what lets a later audit confirm the run stayed inside
the closed-world boundary.

## Open intent, then close it

An intent has two moments. It is opened in the `planned` state before training,
and it is closed with a decision once the promotion gate has run.

```text
attempt dir exists
  -> write experiment_intent.json   (decision.status = planned, promoted = false)
  -> train under guards
  -> promotion gate decides
  -> close intent with decision     (promoted | rejected | aborted)
  -> copy final intent into the attempt report and the latest run report
```

The closing decision is the only place the intent reports promotion, and it
cannot disagree with itself. The `decision.promoted` flag must equal whether the
status is `promoted`, so an intent can never claim it was promoted while its
status says otherwise. A planned intent always starts with `promoted: false`.

| Decision status | Meaning |
| --- | --- |
| `planned` | Intent recorded before training; no outcome yet. |
| `promoted` | The run cleared its gates and the change was accepted. |
| `rejected` | The run ran but did not clear its gates. |
| `aborted` | The run did not complete to a gate decision. |

A rejected or aborted run is kept as versioned diagnostic evidence, not
discarded. The intent that opened it stays attached to the report so the
attempt remains readable.

## Who writes intents

Two run types write `experiment_intent.json`.

**Self-improvement answer cycles** write the intent as soon as an attempt
directory exists, then close it with the promotion-gate result. The final intent
is copied into both the attempt report and the latest run report.

**Transformer answer-training runs** write the intent before training. They
record baseline and final snapshot gates, closed-world data checks,
no-pretrained-weight, no-pretrained-tokenizer, and no-external-embedding checks,
and direct-answer branch-screen gates where they apply. From v0.77 onward, a
transformer run closes through the constraint-first promotion report: quality
metrics such as loss, NLL, or exact-answer counts can affect the decision only
after the closed-world constraints have passed. See
[Transformer](../build/transformer.md) for how those gates sit in the wider
answer-training stack, and [candidate quarantine](./candidate-quarantine.md) for
why generated candidates are not training data until admitted.

## What an intent does not do

The registry validates and records; it does not promote. An intent declares the
gates and the allowed data, but the gates themselves are enforced by the
verifier and the constraint-first report, and the data boundary is enforced by
the corpus ledger. The intent is the run's stated plan and its honest record of
how that plan turned out.
