---
title: Release Discipline
description: Promotion gates for QuarkLM releases.
---

# Release Discipline

A promoted QuarkLM run must have:

- a versioned run directory
- baseline and final metrics
- responder, answer-model, and decoder evals
- generated admission-probe audit
- generated glossary-probe audit
- prompt leakage audit
- forgetting audit against the prior promoted run
- exact eval audit
- passing promotion gate
- self-diagnosis with `uses_external_model: false` unless a future release
  explicitly admits and documents a different source
- archived attempts under `attempts/attempt-###/`
- corpus snapshot and diff
- docs updated for the current state

The release is not complete until the docs are complete. If a page references
current eval counts, commands, run ids, hosting targets, or roadmap commitments,
that page must move with the version.

`closed_world_lm.self_improve answer-cycle` should return failure when the
promotion gate fails. A report can remain useful evidence when it fails, but it
is not a promoted release.

`closed_world_lm.self_diagnose` can be run directly against a report to inspect
the recommended next action. The recommendation must come from report evidence,
not from an external model.

Architecture prototypes, such as the v0.24 transformer, can be documented as
evidence only for the behavior they actually show. Lower language-model loss is
not a reliable-answer claim until answer evals pass.
