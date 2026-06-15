---
title: Transformer Responsibilities
description: The v0.78-v0.79 transformer responsibility surfaces.
---

# Transformer Responsibilities

v0.78 starts splitting transformer answer-training behind the recipe and
verifier surfaces without changing the public CLI. v0.79 adds the
model/config/checkpoint metadata surface.

The current surfaces are:

| Surface | File | Responsibility |
| --- | --- | --- |
| Model and checkpoint metadata | `src/closed_world_lm/transformer_model.py` | Model, optimizer, and generation configs; validation; checkpoint identity; closed-world dataset metadata; run metadata. |
| Experiment and artifacts | `src/closed_world_lm/transformer_experiment.py` | Run artifact paths, intent gates, recipe construction, and promotion decisions. |
| Trainer utilities | `src/closed_world_lm/transformer_training.py` | JSONL snapshot writing, shuffled training cursors, and loss averaging. |
| Objective catalog | `src/closed_world_lm/transformer_objectives.py` | Direct-answer objective names and small objective-selection primitives. |
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
- Direct-answer objective names are no longer owned by the CLI parser.
- Training cursors and history writing have focused tests outside the model.

The model class, checkpoint loading internals, and several eval helpers still
live in `transformer_char_model.py`. Those are the next transformer boundaries
to extract before another objective-repair screen.
