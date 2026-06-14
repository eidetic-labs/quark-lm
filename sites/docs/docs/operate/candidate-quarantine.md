---
title: Candidate Quarantine
description: Keep generated lessons, probes, and repair proposals out of training until admitted.
---

# Candidate Quarantine

v0.75 adds `src/closed_world_lm/candidate_quarantine.py` and a required
`candidate_quarantine.json` artifact for self-improvement answer cycles and
transformer answer-training runs.

The artifact is empty today because QuarkLM has not yet promoted autonomous
candidate generation. That emptiness is still useful: the training plan can now
prove that the candidate lane exists and that no candidate records were silently
used as training data.

## Candidate States

Candidate records support these lifecycle states:

- `proposed`
- `quarantined`
- `needs_human_review`
- `verified`
- `rejected`
- `admitted`
- `trained`
- `promoted`

State transitions are validated. Terminal states such as `rejected` and
`promoted` cannot move again.

## Training Rule

Candidate records are not training data.

A generated lesson, probe, repair proposal, diagnosis note, or memory proposal
can train weights only after it is admitted into the ledgered corpus and
converted into curriculum lessons. The candidate record may preserve its
history, but the candidate store itself is not a training source.

## Run Artifacts

Self-improvement attempts now write:

- `attempts/attempt-###/candidate_quarantine.json`
- `candidate_quarantine.json` at the latest run level
- `training_plan.json` with the candidate quarantine path and summary
- `self_improvement_report.json` with the manifest embedded

Transformer answer-training runs write:

- `candidate_quarantine.json`
- `training_plan.json` with the candidate quarantine path and summary
- `transformer_answer_metrics.json` with the manifest embedded

## Next Step

v0.76 should add deterministic closed-world verifier checks. Once verifier
evidence exists, candidate quarantine can move from an empty lane to a real
proposal and acceptance pipeline.
