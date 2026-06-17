---
title: Release Discipline
description: Promotion gates for QuarkLM releases.
---

# Release Discipline

A QuarkLM release is not a build that finished. It is a run that passed its
gates and left behind the evidence that proves it. This page lists what a
promoted run must produce, how versions are named, and the rules that decide
when a run is allowed to be called a release.

A run that fails a gate is kept as versioned diagnostic evidence, not promoted.
A report can remain useful even when it fails; it is still not a release.

## What a promoted run must have

A promotion has three parts: the run itself, the audits that inspect it, and
the artifacts that record the decision. All three must be present.

### Run and evals

| Requirement | Purpose |
| --- | --- |
| A named RC track when tagging or announcing a candidate | Keeps alpha and release-candidate tracks separate. See [release candidate readiness](./release-candidate.md). |
| A versioned run directory | Provenance: the evidence has a fixed home under `runs/`. |
| Baseline and final metrics | Shows what changed across the run. |
| Responder, answer-model, and decoder evals | Covers the three trained answer surfaces. |
| Generated admission-probe and glossary-probe audits | Confirms evals are generated from admitted text, not hand-seeded. |
| Prompt leakage audit | Confirms eval prompts did not leak into training. |
| Forgetting audit against the prior promoted run | Confirms the update did not lose previously held behavior. |
| Exact eval audit | Records exact pass/fail counts rather than averaged scores. |

### Gates that must pass

| Gate | Meaning |
| --- | --- |
| Promotion gate | The run cleared the constraints and is eligible to be promoted. |
| `npm run check` | Repository and docs checks pass. |
| `npm run alpha:gate` | Required before any alpha tag. |

`self_improve answer-cycle` returns failure when the promotion gate fails, so a
failed run cannot be mistaken for a promoted one.

### Required artifacts

Each promoted run carries machine-checkable JSON so the run can be audited
rather than trusted. The same artifacts appear in the
[operating surfaces](./index.md) table and in the
[transformer](../build/transformer.md) evidence stack.

| Artifact | Records |
| --- | --- |
| `experiment_intent.json` | Hypothesis, allowed data, planned artifacts, acceptance gates, failure criteria, and final decision. See [experiment registry](./experiment-registry.md). |
| `corpus_hygiene.json` | Source mixture, duplicates, train/eval overlap, candidate ratio, and rare-profile evidence. See [corpus hygiene](./corpus-hygiene.md). |
| `training_plan.json` | Allowed data sources, scheduled example mixture, replay-plan status, and planned artifacts. |
| `training_recipe.json` | Model, tokenizer, data, objective, optimizer, artifact, gate, replay, and rerun details. See [training recipes](./training-recipes.md). |
| `candidate_quarantine.json` | Candidate lifecycle state, with proof that candidate records are not training data until admitted. See [candidate quarantine](./candidate-quarantine.md). |
| `closed_world_verifier.json` | Deterministic pass/fail evidence for candidate checks and training-plan approval. See [closed-world verifier](./closed-world-verifier.md). |
| `constraint_first_promotion.json` | Proof that quality metrics were blocked until closed-world constraints passed. |

### Provenance and docs

| Requirement | Purpose |
| --- | --- |
| Self-diagnosis with `uses_external_model: false` | Recommendations come from report evidence, not an external model, unless a future release explicitly admits and documents a different source. |
| Archived attempts under `attempts/attempt-###/` | Every attempt is kept, not only the accepted one. |
| Corpus snapshot and diff | Records the ledger state and how it changed from the prior promoted run. |
| Docs updated for the current state | The release is not complete until the docs are. |
| Alpha gate, RC spec, gap audit, and checklist reviewed for forbidden claims | No surface overstates what the run shows. |

## Version naming

New release identifiers use SemVer (Semantic Versioning) with
`MAJOR.MINOR.PATCH` versions and optional prerelease labels.

| Rule | Example |
| --- | --- |
| The first proposed alpha is a SemVer prerelease, tagged with a `v` prefix. | `0.115.0-alpha.1`, Git tag `v0.115.0-alpha.1` |
| Minor versions roll over numerically, not as decimals. | The release after `0.99.0` is `0.100.0`, not `v1.00`; the line then advances `0.101.0`, `0.102.0`. |
| `1.0.0` is reserved for a deliberate stable milestone. | Not `1.0.0` by accident. |
| Placeholders are not allowed in release docs. | No `XX.YY.ZZ`. |
| Historical artifacts keep their existing names. | Run paths such as `runs/transformer-answer-v0.42/` stay as named, so provenance remains exact. |

## Single-responsibility blocker

Single Responsibility Principle is part of release discipline, not a separate
code-quality concern. Alpha cannot be tagged while `ALPHA_GATE.md` records P0
source or test files that combine many responsibilities in thousands of lines.
This is the current blocker on the alpha tag; see
[release candidate readiness](./release-candidate.md).

## Docs move with the version

The release is not complete until the docs are complete. A page that references
current eval counts, commands, run ids, hosting targets, or roadmap commitments
must move with the version that changes them. Docs are a promotion gate and an
anti-drift discipline, not training input; see [docs drift](./docs-drift.md).

## Diagnosis without external models

`self_diagnose` can be run directly against a report to inspect the recommended
next action. The recommendation must come from report evidence, not from an
external model. This keeps the closed-world boundary intact even when reading a
failed run; see [purity boundary](../secure/purity-boundary.md).

## Prototypes are documented for what they show

Architecture prototypes, such as the v0.24 transformer, can be documented as
evidence only for the behavior they actually demonstrate. Lower language-model
loss is not a reliable-answer claim until answer evals pass.

From v0.71 onward, transformer answer-training screens also write experiment
intent artifacts, but those screens close as rejected screen evidence until a
dedicated transformer promotion gate exists. The from-scratch transformer is not
promoted; it is blocked on `branch_diversity_target`. Retrieval memory answering
its eval probes exactly is `memory-served`, not `weight-consolidated`, and
counts as memory evidence rather than neural promotion. See
[transformer](../build/transformer.md).
