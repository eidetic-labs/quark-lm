---
title: Transformer Responsibilities
description: The v0.78-v0.101.0 transformer responsibility, objective, and screen surfaces.
---

# Transformer Responsibilities

The from-scratch transformer (`transformer_char_model`, see
[Transformer](./transformer.md)) began as a single broad module. Starting at
v0.78, the answer-training stack moved its work into a set of narrow, separately
tested surfaces. This page is the durable map of those surfaces: what each one
owns, why the split exists, and the contract the split preserves.

:::note
This page explains the surface layout. The version-by-version screen log that
created and exercised these surfaces (v0.78 onward, and every evidence table)
lives in [Transformer screen history](./transformer-screen-history.md).
:::

## Why the monolith was split

Before v0.78, model and optimizer config validation, checkpoint loading,
eval-report assembly, the direct-answer objective catalog, and trainer
utilities all lived inside the model class and the CLI command body. That made
each transformer repair a broad patch over a large surface, and made it hard to
test one concern without exercising all of them.

The split has one purpose: make later repair work smaller and more auditable. It
does **not** claim better answer quality, and it does not change the public CLI
or the artifacts a run writes. The transformer remains unpromoted, blocked on
`branch_diversity_target`; the split is about how that work is organized, not
about clearing the gate.

## The surfaces

Each surface owns one concern and is tested on its own:

| Surface | File | Responsibility |
| --- | --- | --- |
| Model and checkpoint metadata | `src/transformer_model.py` | Model, optimizer, and generation configs; validation; checkpoint identity; closed-world dataset metadata; run metadata. |
| Checkpoint loading | `src/transformer_checkpoint.py` | Checkpoint payload loading, identity validation, and checkpoint summaries. |
| Eval reports | `src/transformer_eval.py` | Probe loading, candidate collection, generic transformer scoring, report assembly, samples JSONL writing, and eval JSON writing. |
| Experiment and artifacts | `src/transformer_experiment.py` | Run artifact paths, intent gates, recipe construction, and promotion decisions. |
| Trainer utilities | `src/transformer_training.py` | JSONL snapshot writing, shuffled training cursors, and loss averaging. |
| Objective catalog | `src/transformer_objectives.py` | Direct-answer objective names and small objective-selection primitives. |
| Replay planning | `src/replay_plan.py` | Branch replay records, profile grouping, replay summaries, and coverage floors. |
| Verifier | `src/closed_world_verifier.py` | Deterministic closed-world data-boundary checks before evidence is trusted. |
| Recipes and gates | `src/training_recipe.py` | Reproducible recipe artifacts and constraint-first promotion reports. |

The model class and the direct-answer eval helpers still live in
`transformer_char_model.py`, which continues to export the older names for
compatibility. The narrow surfaces own the logic; the old module remains the
stable public entry point.

## The contract the split preserves

Moving code into these surfaces did not change what an `answer-train` run looks
like from the outside:

- `answer-train` keeps writing the same public artifacts.
- `training_recipe.json` still binds model, tokenizer, data, objective,
  optimizer, replay, artifacts, gates, and rerun details.
- `constraint_first_promotion.json` still blocks any quality metric until the
  closed-world constraints pass first.
- The public `eval` CLI and artifact shapes are unchanged.

What changed is ownership. Config validation is no longer owned by the
transformer monolith; checkpoint payload loading and identity checks are no
longer hidden inside the model class; eval report assembly and file writing are
no longer owned by the CLI command body; direct-answer objective names are no
longer owned by the CLI parser. Each of those concerns now sits in a surface
with its own focused tests.

## How the surfaces serve the boundary

The split keeps the evidence rails QuarkLM depends on cleanly separated:

- Replay planning (`replay_plan.py`) supplies profile grouping and coverage
  floors, so a repair objective can apply balanced target-share pressure across
  a profile's replay targets without one represented target dominating a
  multi-target profile.
- Transformer answer metrics explicitly declare the closed-world embedding
  boundary (`external_embeddings: false`) before constraint-first promotion
  reads them.
- The deterministic [closed-world verifier](../operate/closed-world-verifier.md)
  must pass before a screen's evidence is trusted.

This matters because the catalog of direct-answer objectives is exactly where a
repair could quietly cross a line. Generated candidates are not training data
until verified against admitted sources and admitted to the ledger; the surfaces
keep that discipline in the recipe and verifier artifacts rather than in the
objective code that proposes the pressure. None of these surfaces import
external weights, tokenizers, embeddings, datasets, or training text — see
[Purity boundary](../secure/purity-boundary.md).

## The screen record

From v0.78 onward, every direct-answer repair objective and the screen that
tested it was recorded against these surfaces. The objectives moved coverage and
diagnostics forward — through profile target-share pressure, prompt-ownership
margins, baseline-floor anchors and retries, calibrated sub-`0.01` source-profile
updates, and coverage-recovery frontiers — but promotion stayed blocked on
branch diversity throughout. Each objective name, the screen that exercised it,
and its evidence table are catalogued in
[Transformer screen history](./transformer-screen-history.md). That page is the
authoritative log; this page is the durable map.

## Forward rule

Future objective-repair work should use these narrow surfaces rather than adding
another broad patch to the monolith. The open repair direction is to turn the
expanded, calibrated, floor-preserving source-profile movement these surfaces
now allow into genuinely branch-diverse behavior, before adding branch-diversity
pressure back. Until a run preserves the boundary, passes the gates, and updates
the docs that describe current state, the transformer stays unpromoted and these
surfaces hold versioned diagnostic evidence rather than a promotion claim.
