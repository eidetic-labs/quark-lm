---
title: Closed-World Verifier
description: Deterministic verifier checks for candidate and training-plan approval.
---

# Closed-World Verifier

v0.76 adds `src/closed_world_lm/closed_world_verifier.py` and a required
`closed_world_verifier.json` artifact for self-improvement answer cycles and
transformer answer-training runs.

The verifier is deterministic. It does not call an external model, use hidden
training data, or grade with a pretrained judge. Its job is narrower: decide
whether a candidate record or training plan is allowed to influence the next
learning step.

## What It Checks

Training-plan approval checks:

- closed-world data boundary flags remain false;
- candidate records are excluded from training examples;
- candidate quarantine is declared and valid;
- training-eligible candidates link to admission ids;
- protected train/eval overlap checks pass;
- the verifier artifact itself was planned before training.

Candidate checks verify schema, source labels, required payload fields, and
admission links. When a `CorpusResponder` is supplied, prompt-target candidates
are also checked for exact agreement with the deterministic responder trained
from admitted text.

## Run Artifacts

Self-improvement attempts now write:

- `attempts/attempt-###/closed_world_verifier.json`
- `closed_world_verifier.json` at the latest run level
- `training_plan.json` with the verifier path and summary
- `self_improvement_report.json` with verifier evidence embedded

Transformer answer-training runs write:

- `closed_world_verifier.json`
- `training_plan.json` with the verifier path and summary
- `transformer_answer_metrics.json` with verifier evidence embedded

## Boundary

Verifier approval is not a claim that the model answered better. It is a claim
that the next weight update or screen stayed inside the closed-world data
boundary. Recipes and constraint-first promotion gates still need to decide
whether a trained snapshot should be promoted.
