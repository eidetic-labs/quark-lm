---
title: Release Discipline
description: Promotion gates for QuarkLM releases.
---

# Release Discipline

A promoted QuarkLM run must have:

- a named RC track when tagging or announcing a release candidate
- a versioned run directory
- baseline and final metrics
- responder, answer-model, and decoder evals
- generated admission-probe audit
- generated glossary-probe audit
- prompt leakage audit
- forgetting audit against the prior promoted run
- exact eval audit
- passing promotion gate
- `experiment_intent.json` with hypothesis, allowed data, planned artifacts,
  acceptance gates, failure criteria, and final decision
- `corpus_hygiene.json` with source mixture, duplicate, train/eval overlap,
  candidate-ratio, and rare-profile evidence
- `training_plan.json` with allowed data sources, scheduled example mixture,
  replay-plan status, and planned artifacts
- `training_recipe.json` with model, tokenizer, data, objective, optimizer,
  artifact, gate, replay, and rerun details
- `candidate_quarantine.json` with candidate lifecycle state and proof that
  candidate records are not training data until admitted
- `closed_world_verifier.json` with deterministic pass/fail evidence for
  candidate checks and training-plan approval
- `constraint_first_promotion.json` with proof that quality metrics were
  blocked until closed-world constraints passed
- self-diagnosis with `uses_external_model: false` unless a future release
  explicitly admits and documents a different source
- archived attempts under `attempts/attempt-###/`
- corpus snapshot and diff
- docs updated for the current state
- RC spec, gap audit, and checklist reviewed for forbidden claims

New release identifiers use SemVer (Semantic Versioning) with
`vMAJOR.MINOR.PATCH` tags and matching run paths. The next release after
`v0.99` is `v0.100.0`, not `v1.00`; the current line then advances as
`v0.101.0`, `v0.102.0`, and so on. Do not use `XX.YY.ZZ` placeholders in
release docs. `v1.0.0` is reserved for a deliberate stable milestone.
Historical artifacts keep their existing names so provenance remains exact.

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
not a reliable-answer claim until answer evals pass. From v0.71 onward,
transformer answer-training screens also write experiment intent artifacts, but
they close as rejected screen evidence until a dedicated transformer promotion
gate exists.
