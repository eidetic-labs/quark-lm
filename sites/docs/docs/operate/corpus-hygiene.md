---
title: Corpus Hygiene
description: Corpus hygiene and training-plan artifacts for QuarkLM runs.
---

# Corpus Hygiene

Corpus hygiene is the first evidence surface a run produces. Before any metric is
read, it makes the *shape of the data* visible: where the training text came
from, whether examples are duplicated, and whether the eval set overlaps the
training set. A run whose numbers look good for the wrong reason — leaked eval
prompts, a lopsided source mixture — is caught here, not after promotion.

`src/corpus_hygiene.py` (added in v0.73) writes two required run artifacts:

- `corpus_hygiene.json` — what the data looks like;
- `training_plan.json` — what the run is permitted to do with it.

Self-improvement answer cycles and transformer answer-training runs write both
artifacts before their metrics are treated as evidence. The artifacts do not
promote or reject a model. They make data risk legible so the gates downstream —
the [closed-world verifier](./closed-world-verifier.md) and constraint-first
promotion — can act on it.

## Where hygiene sits in the chain

```text
admitted corpus (ledger.json)
  -> corpus_hygiene.json     describe the data: sources, duplicates, overlap, coverage
  -> training_plan.json      declare what the run may train on, inside the boundary
  -> closed_world_verifier   decide whether the plan may influence the next step
  -> constraint-first gate   only then are quality metrics allowed to count
```

Hygiene is descriptive; the verifier and the gate are decisive. Keeping the two
separate is deliberate: a report that could itself promote a run would be a
quality shortcut, not a check.

## Corpus hygiene report

`corpus_hygiene.json` records the measurable properties of the data the run drew
from:

| Field | Records |
| --- | --- |
| Corpus source counts | How many records came from each admitted source. |
| Training text path and character count | The exact text the run trained on, and its size. |
| Training-example source mixture | The proportion of examples drawn from each source family. |
| Duplicate training examples | Repeated training examples that could inflate apparent learning. |
| Duplicate admission and eval ids | Repeated identifiers across admitted and eval records. |
| Train/eval prompt overlap | Eval prompts that also appear in training — the contamination check. |
| Protected heldout prompt overlap | Overlap against the protected heldout set that may never be trained on. |
| Candidate-example ratio | The share of examples that originate from generated candidates. |
| Rare-profile coverage | Whether low-frequency answer profiles are represented at all. |

The report does not promote or reject a model by itself. It makes data risk
visible before promotion gates or transformer screens interpret metrics. A high
train/eval overlap, for example, does not fail the run on its own — it tells the
verifier and the reader why an exact-answer count should be distrusted.

## Training plan

`training_plan.json` is the run's declared scope. Where the hygiene report
describes the data, the plan states what the run is permitted to do with it,
inside the closed-world boundary.

| Field | Records |
| --- | --- |
| Component and run id | Which component is training, and the run it belongs to. |
| Allowed data sources | The admitted sources the run may draw from, stated up front. |
| Closed-world data boundary | The flags asserting no external weights, tokenizer, embeddings, or text. |
| Hygiene report path | The `corpus_hygiene.json` this plan was built against. |
| Eval-set counts | The size of each eval set the run will be scored on. |
| Base and scheduled example mixture | The intended source mixture, before and after scheduling. |
| Candidate policy status | Whether generated candidates are excluded from training. |
| Training recipe path and summary | The reproducible [recipe](./training-recipes.md), when written. |
| Replay-plan path and summary | The profile-aware replay plan, when one is written. |
| Closed-world verifier path and summary | The [verifier](./closed-world-verifier.md) approval, when written. |
| Planned artifacts | The evidence files the run commits to emitting. |

The candidate ratio is the load-bearing field. Generated or proposed examples
are reported here, but reporting them is not the same as admitting them: a
candidate cannot become training data without a later admission to the ledger
and a verification path. The plan records the lane; the
[candidate quarantine](./candidate-quarantine.md) enforces it.

## How the surfaces grew, and what they prove

The hygiene and training-plan artifacts were extended across many versions, but
the additions follow one shape: each new surface attaches more evidence to the
same plan without renaming the existing artifacts or letting any of them double
as a promotion shortcut.

| Version | Addition |
| --- | --- |
| v0.73 | Hygiene report and training plan; candidate ratio reported. |
| v0.75 | `candidate_quarantine.json`, linked from the plan, so generated examples cannot enter training without admission and verification. |
| v0.76 | `closed_world_verifier.json`, so the plan can be approved or rejected before its evidence influences the next version. |
| v0.77 | `training_recipe.json`, so the plan links a recipe that can reconstruct the run. |
| v0.78–v0.80 | Transformer responsibility, model/config, checkpoint, and eval surfaces that consume the same plan, recipe, and verifier without changing their names. |

From v0.81 onward, successive versions used these surfaces to attempt
profile-targeted and baseline-floor repairs of the transformer's
target-routing problem. The decisive outcome is consistent: the same artifact
trail kept rejecting promotion whenever a trained snapshot lost target-token
coverage or missed the baseline floor — including every `200/200` retry, which
is a memory-served count, not learned weights. Only small, guarded
source-profile updates that preserved the baseline floor (first one
`bridge:owner` update, then several profile-scale updates) were allowed to
become trusted model state.

This is the point of the report. The corpus-hygiene trail is part of the
evidence bundle, not a quality promotion shortcut. It demonstrates, run after
run, that broader coverage or a better-looking number cannot become trusted
model state by itself — only a guarded update that keeps the data boundary and
the baseline floor intact can. The from-scratch transformer remains unpromoted
on `branch_diversity_target`; the hygiene trail is part of why that claim is
honest. See [Transformer](../build/transformer.md) for the routing problem
itself, and [Build](../build/index.md) for the `memory-served` versus
`weight-consolidated` distinction these artifacts protect.
