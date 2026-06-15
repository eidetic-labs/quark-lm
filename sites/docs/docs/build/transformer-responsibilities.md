---
title: Transformer Responsibilities
description: The v0.78-v0.86 transformer responsibility, objective, and screen surfaces.
---

# Transformer Responsibilities

v0.78 starts splitting transformer answer-training behind the recipe and
verifier surfaces without changing the public CLI. v0.79 adds the
model/config/checkpoint metadata surface. v0.80 adds checkpoint-load and eval
report surfaces. v0.81 uses those surfaces to add profile target-share
anti-collapse pressure to the direct-answer objective path. v0.82 screens that
objective under the modern artifact stack, rejects it on branch diversity, and
fixes the transformer purity metrics so `external_embeddings: false` is
declared for constraint-first checks. v0.83 adds prompt-specific ownership
margins on top of profile target-share pressure and rejects the screen because
trained snapshots still lose target-token coverage. v0.84 adds baseline replay
anchors and rejects the screen because trained snapshots preserve only half of
the baseline QA/heldout coverage floor. v0.85 adds baseline-floor update gating
and rejects the screen because all attempted updates fall below the floor. v0.86
adds adaptive baseline-floor retries across smaller learning-rate scales and
rejects the screen because all `200/200` attempted retry updates still fall
below the floor.

The current surfaces are:

| Surface | File | Responsibility |
| --- | --- | --- |
| Model and checkpoint metadata | `src/closed_world_lm/transformer_model.py` | Model, optimizer, and generation configs; validation; checkpoint identity; closed-world dataset metadata; run metadata. |
| Checkpoint loading | `src/closed_world_lm/transformer_checkpoint.py` | Checkpoint payload loading, identity validation, and checkpoint summaries. |
| Eval reports | `src/closed_world_lm/transformer_eval.py` | Probe loading, candidate collection, generic transformer scoring, report assembly, samples JSONL writing, and eval JSON writing. |
| Experiment and artifacts | `src/closed_world_lm/transformer_experiment.py` | Run artifact paths, intent gates, recipe construction, and promotion decisions. |
| Trainer utilities | `src/closed_world_lm/transformer_training.py` | JSONL snapshot writing, shuffled training cursors, and loss averaging. |
| Objective catalog | `src/closed_world_lm/transformer_objectives.py` | Direct-answer objective names, including the v0.81 profile target-share mode, and small objective-selection primitives. |
| Replay planning | `src/closed_world_lm/replay_plan.py` | Branch replay records, profile grouping, replay summaries, and coverage floors. |
| Verifier | `src/closed_world_lm/closed_world_verifier.py` | Deterministic closed-world data-boundary checks before evidence is trusted. |
| Recipes and gates | `src/closed_world_lm/training_recipe.py` | Reproducible recipe artifacts and constraint-first promotion reports. |

This version does not claim better answer quality. It makes later transformer
repair work smaller and more auditable:

- `answer-train` keeps writing the same public artifacts.
- `training_recipe.json` still binds model, tokenizer, data, objective,
  optimizer, replay, artifacts, gates, and rerun details.
- `constraint_first_promotion.json` still blocks quality metrics until
  closed-world constraints pass.
- Model, optimizer, and generation config validation is no longer owned by the
  transformer monolith.
- Checkpoint architecture, format, tokenizer identity, and run metadata are
  centralized in a tested surface.
- Checkpoint payload loading and identity checks are no longer hidden inside
  the model class.
- Generic eval report assembly and file writing are no longer owned by the CLI
  command body.
- Direct-answer objective names are no longer owned by the CLI parser.
- Profile-aware preserving-deficit replay can add balanced target-share
  pressure across each profile's replay targets.
- Transformer answer metrics explicitly declare the closed-world embedding
  boundary before constraint-first promotion reads them.
- The v0.82 profile target-share screen shows rank lift is not enough when it
  depends on prompt-collapse and lost target-token coverage.
- The v0.83 prompt-ownership screen shows prompt-specific margins still need a
  coverage-preserving training constraint before rank gains can promote.
- The v0.84 baseline-anchor screen records active baseline replay prediction
  anchors and shows the next repair must preserve the full coverage floor.
- The v0.85 baseline-floor update guard preserves that floor by rejecting every
  unsafe attempted update, showing the next repair must produce accepted safe
  updates.
- The v0.86 adaptive baseline-floor retry guard shows smaller direct-answer
  learning-rate scales are not enough; the next repair must change update shape
  while staying under the full baseline coverage floor.
- Training cursors and history writing have focused tests outside the model.

The model class and direct-answer eval helpers still live in
`transformer_char_model.py`. Future objective-repair work should use the
narrower surfaces rather than adding another broad monolith patch.
