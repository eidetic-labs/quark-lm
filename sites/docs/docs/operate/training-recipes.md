---
title: Training Recipes
description: Reproducible recipes and constraint-first promotion reports for QuarkLM runs.
---

# Training Recipes

v0.77 adds `src/training_recipe.py` and two required artifacts that every
self-improvement answer cycle and transformer answer-training run must emit:

- `training_recipe.json` records how to reproduce a run.
- `constraint_first_promotion.json` records whether a run is even allowed to let
  quality metrics affect promotion.

The two split a single question into its two honest halves. The recipe answers
"what was run, and how would it be run again." The constraint-first report
answers "did the run stay inside the closed-world boundary before any score was
allowed to count." Neither file promotes a run. They make the run auditable
against its own declared plan rather than a description reconstructed afterward.

These artifacts sit downstream of the [experiment
registry](./experiment-registry.md), which opens a run's intent, and downstream
of the [closed-world verifier](./closed-world-verifier.md), which approves the
training plan. The recipe and the constraint-first report close the loop: one
preserves reproducibility, the other gates promotion.

## The recipe artifact

`training_recipe.json` captures the run's configuration so a later screen can be
reconstructed from the artifact and the admitted project state, not from hidden
argparse memory.

| Field | Records |
| --- | --- |
| `recipe_id` / `version` | The recipe identity and the QuarkLM version it belongs to. |
| `component` / `run_id` | The component under test and the run the recipe describes. |
| `model` | Model configuration for the screen. |
| `tokenizer` | Tokenizer provenance — corpus-trained, never pretrained. |
| `data` | The admitted data sources the run is permitted to draw from. |
| `objective` | The training objective and its settings. |
| `optimizer` | Optimizer settings. |
| `replay` | Replay status and any replay-plan reference. |
| `artifacts` | The evidence files the run commits to emitting. |
| `gates` | The required gates the run must clear. |
| `rerun` | The rerun surface — enough to reproduce the run. |

The artifact also carries `uses_external_model: false` alongside the same
no-pretrained-weights, no-pretrained-tokenizer, and no-external-embeddings
posture enforced everywhere else. A recipe that named an external weight,
tokenizer, embedding, or dataset would cross the [purity
boundary](../secure/purity-boundary.md); recipes are reproduction records for
closed-world runs only.

## Constraint-first promotion

`constraint_first_promotion.json` separates closed-world *constraints* from
quality *checks*, and enforces the order between them. Constraints are
pass/fail facts about the data boundary. Quality checks are scores. The report
will not let a score count until every required constraint has passed.

```text
constraints (boundary facts)  ->  must all pass
        |
        v  (only then)
quality checks (scores)        ->  may affect promotion
        |
        v
status: blocked_before_quality_metrics  |  considered
```

For a transformer answer-training run the constraints are:

| Constraint | Asserts |
| --- | --- |
| `baseline_snapshot_recorded` / `final_snapshot_recorded` | The run captured comparable before/after evidence. |
| `closed_world_training_data` | Training drew only from admitted sources. |
| `closed_world_verifier` | The deterministic [verifier](./closed-world-verifier.md) approved the plan. |
| `no_pretrained_weights` | No imported weights. |
| `no_pretrained_tokenizer` | No imported tokenizer. |
| `no_external_embeddings` | No imported embeddings. |
| `direct_answer_evidence_present` | The run recorded direct-answer evidence to judge. |
| `branch_context_gate` | Branch-context coverage holds. |
| `branch_diversity_target` | Predictions route across distinct branch tokens. |
| `target_coverage_preserved` | Trained snapshots did not lose target-token coverage. |

Only after those constraints pass can the exact direct-answer quality check
(`direct_greedy_exact`) affect promotion. Loss, NLL, rank, and top-k movement
stay advisory throughout: the report states plainly that these metrics are
advisory until all closed-world constraints pass. When a required constraint
fails, the report status is `blocked_before_quality_metrics` and
`quality_metrics_considered` is false — the scores are recorded, but they are
not allowed to influence the decision.

This is the mechanism behind the constraint-first rule stated on
[Operate](./index.md): a run can produce better numbers and still be rejected.
A good score behind a failed constraint is not evidence of promotion.

## Current status

Self-improvement answer cycles promote through their existing exact-eval
discipline, now with a constraint-first report attached to each attempt.

Transformer answer-training runs now have a promotion gate, and it is rejecting.
Recent screens block on `branch_diversity_target`: the constraints report marks
that constraint failed, leaves quality metrics unconsidered, and closes with
status `blocked_before_quality_metrics`. The transformer is therefore not
promoted, and the docs say so — see [Transformer](../build/transformer.md) for
why branch diversity is the blocker and how it differs from the retrieval
memory that answers admitted probes exactly. Exact retrieval is
`memory-served`, not `weight-consolidated`; the gate does not confuse the two.
