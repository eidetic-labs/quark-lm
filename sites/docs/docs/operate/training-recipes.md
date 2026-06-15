---
title: Training Recipes
description: Reproducible recipes and constraint-first promotion reports for QuarkLM runs.
---

# Training Recipes

v0.77 adds `src/closed_world_lm/training_recipe.py` and two required run
artifacts for self-improvement answer cycles and transformer answer-training
runs:

- `training_recipe.json`
- `constraint_first_promotion.json`

The recipe records how to reproduce a run. The constraint-first report records
whether a run is even allowed to consider quality metrics for promotion.

## Recipe Contents

`training_recipe.json` records:

- recipe id and version;
- component and run id;
- model configuration;
- tokenizer provenance;
- data sources;
- objective settings;
- optimizer settings;
- replay status;
- planned artifacts;
- required gates;
- rerun surface.

This keeps recipe identity out of hidden argparse memory. A future screen
should be reconstructable from the recipe artifact and the admitted project
state.

## Constraint-First Promotion

`constraint_first_promotion.json` separates constraints from quality checks.

Constraints must pass before quality metrics are considered. For transformer
runs, those constraints include closed-world verifier approval, closed-world
training data, no pretrained weights, no pretrained tokenizer, no external
embeddings, direct-answer evidence, branch-context coverage, branch diversity,
and target-coverage preservation.

Only after those constraints pass can exact direct-answer quality checks affect
promotion. Loss, NLL, rank, and top-k movement remain advisory until the
constraint report says quality metrics are eligible.

## Status

Self-improvement runs can still promote through their existing exact-eval
discipline, now with a constraint-first report attached. Transformer runs now
have a promotion gate, but current evidence is still expected to reject until
the model satisfies the branch-context, diversity, coverage, and exact-answer
quality checks.
