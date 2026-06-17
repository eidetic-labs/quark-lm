---
title: Transformer
description: How the from-scratch QuarkLM transformer works, and its current status.
---

# Transformer

`transformer_char_model` is QuarkLM's from-scratch neural model: a tiny
decoder-only transformer introduced in v0.24, built without PyTorch, JAX, Hugging
Face, pretrained checkpoints, or pretrained tokenizers. It starts from random
weights and trains with a small standard-library scalar autodiff engine.

It is **not** the reliable answering path. Retrieval memory and the deterministic
responder already answer admitted probes exactly (see [Build](./index.md)). The
transformer is the *weight-consolidation* path — the component meant to gradually
learn language behavior from admitted candidates after memory has made the
knowledge available and evaluation can reject harmful updates.

:::note
The full version-by-version screen log (v0.24 to current) and every evidence
table live in [Transformer screen history](./transformer-screen-history.md).
This page is the durable explanation.
:::

## Architecture

The model is intentionally small, so cause and effect stay inspectable:

- a corpus-trained character tokenizer (`tokenizer.CharTokenizer`) that learns
  its vocabulary from admitted text and rejects out-of-vocabulary characters;
- learned token and position embeddings;
- one causal self-attention block;
- one feed-forward block;
- a next-character language-model head;
- dependency-free scalar autodiff;
- random initialization only.

Every part is corpus-derived or randomly initialized. A pretrained vocabulary
would cross the same boundary as pretrained weights — see
[Purity boundary](../secure/purity-boundary.md).

## Train a checkpoint

Run from the project root with `PYTHONPATH=src` set:

```bash
# next-character language-model pretraining on the corpus
PYTHONPATH=src python3 -m transformer_char_model train \
  --run runs/transformer-smoke \
  --steps 40 --context-size 8 --embedding-dim 6 --feedforward-dim 12

# evaluate answer probes against a checkpoint
PYTHONPATH=src python3 -m transformer_char_model eval \
  --checkpoint runs/transformer-smoke/transformer.json \
  --json runs/transformer-smoke/transformer_eval.json
```

`answer-train` trains on corpus-derived answer lessons. It carries a large
catalog of direct-answer objectives aimed at the branch-diversity problem below;
each objective name and the screen that tested it is recorded in
[Transformer screen history](./transformer-screen-history.md). See
[Quickstart](./quickstart.md) for a representative `answer-train` invocation.

## The answer-training stack writes its own evidence

Every `answer-train` run emits machine-checkable artifacts, so a screen can be
audited rather than trusted:

| Artifact | Records |
| --- | --- |
| `experiment_intent.json` / `transformer_answer_metrics.json` | The screen's hypothesis, acceptance gate, and closing decision. |
| `training_plan.json` / `corpus_hygiene.json` | Source mixture, duplicate and train/eval overlap checks, candidate ratio, allowed sources. |
| `candidate_quarantine.json` | Candidate lifecycle state; candidates are not training data until admitted to the ledger. |
| `closed_world_verifier.json` | Deterministic check that the data boundary, candidate exclusion, quarantine, and protected train/eval overlap pass. |
| `training_recipe.json` / `constraint_first_promotion.json` | Model, tokenizer, data, objective, optimizer, replay, and gates; blocks any loss, NLL, rank, or exact-quality number until constraints pass first. |
| `retrieval_memory_report.json` | Retrieval-memory evidence, kept separate from neural weight metrics. |

See [Transformer responsibilities](./transformer-responsibilities.md) for how
these surfaces are divided across modules.

## Current status: branch diversity is the blocker

The transformer is not promoted, and the docs say so plainly. Direct-answer
snapshots emit `branch_diversity_target`, which fails when multi-target eval
profiles collapse to too few predicted branch tokens — the model learns to
predict one dominant token instead of routing each prompt to its own answer.

From v0.112 onward the failure is classified as a critical `target_routing_gap`:
`9/9` multi-target profiles fail, representation separation across profiles is
low, and dominant-token wins are hidden-projection driven. Dozens of
direct-answer objectives (catalogued in the screen history) have moved coverage
and diagnostics forward without clearing the gate.

Retrieval memory answers `219/219` eval probes exactly, with provenance and **no
weight updates**. That is evidence for the memory-first rail, not neural
promotion: `memory-served` is not `weight-consolidated`.

## Foundation-stack options

From v0.51, audited GPT-style components are available as opt-in flags before the
next repair objective: `--optimizer adamw`, gradient accumulation, warmup/decay
schedules, `--attention-heads`, `--use-rms-norm`, `--use-gated-mlp`,
`--tie-output-embeddings`, `--use-rotary-positions`, `--use-kv-cache-path`, and
`--use-pre-layer-norm`. Per `STRUCTURE_AUDIT.md`, QuarkLM may study open-source
model, trainer, tokenizer, and checkpoint *structure*, but must not import
external weights, tokenizers, embeddings, datasets, or training text.

## Current evidence

| Run | Signal | Value |
| --- | --- | --- |
| `runs/transformer-v0.25/` | Validation NLL | `3.5885 -> 3.4382` |
| `runs/transformer-v0.25/` | Answer exact eval | `0/28` |
| `runs/transformer-answer-v0.42/` | Direct-answer transformer exact | `0/219` |
| `runs/transformer-answer-v0.42/` | Selector / generator exact | `219/219` |
| `runs/transformer-answer-v0.42/` | Direct-answer target loss | `3.4278 -> 2.2708` |
| latest screens (through v0.115) | Promotion gate | rejected on `branch_diversity_target` |
| all runs | Pretrained weights / tokenizer / external embeddings | `false` |

The selector and generator reaching `219/219` while the direct transformer stays
at `0/219` is exactly why evidence states are kept separate: the system can
*serve* every answer while the neural weights have not yet *learned* to route
them. Full run-by-run detail is in
[Transformer screen history](./transformer-screen-history.md).
