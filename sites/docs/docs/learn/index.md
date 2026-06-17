---
title: Learn
description: Understand the QuarkLM model, goal, and current evidence.
slug: /learn/
---

# Learn

QuarkLM is a closed-world language model experiment. It starts without
pretrained weights, without a pretrained tokenizer, and without external
embeddings. Knowledge enters only through the admitted corpus and the
corpus-derived lessons generated from it. The neural weights learn only from
the admitted, ledgered corpus named in `corpus/ledger.json`.

The current prototype is intentionally small. Its value is not scale. Its value
is discipline: every claim about learning has a corpus source, a retrieval
record when memory can serve it, a training candidate when weights should learn
from it, a guarded update, an evaluation artifact, and a docs update.

This section carries the why. For the mechanical orientation — modules, data
flow, and which path actually changes weights — see [Build](../build/index.md).

## Core Learning Loop

```text
new lesson -> corpus -> retrieval memory -> training candidates -> guarded weight update -> evaluation -> accepted or rejected
```

This is the central Learn concept. A new lesson is not learned just because a
model or tool generated it. Generated material is a candidate, not training
data, until it is verified against admitted sources and admitted to the ledger.
Only then can it enter retrieval memory, produce source-backed training
candidates, move weights through a guarded update, and survive evaluation. The
final state is explicit: accepted into the current model evidence, or rejected
as diagnostic evidence for the next repair.

| Stage | What must be true |
| --- | --- |
| New lesson | The proposed fact, rule, probe, or repair has a declared source. |
| Corpus | The lesson is admitted into the closed-world ledger before training can use it. |
| Retrieval memory | Memory can serve the admitted knowledge without pretending the transformer learned it. |
| Training candidates | Candidate examples are built from admitted data, retrieval evidence, and failure reports. |
| Guarded weight update | The update is bounded, auditable, and allowed to fail without promoting. |
| Evaluation | Closed-world, retention, branch-diversity, target-coverage, leakage, and quality checks run. |
| Accepted or rejected | Passing evidence can promote; failing evidence remains useful but unpromoted. |

The loop produces three distinct evidence states, and QuarkLM keeps them
separate on purpose: `corpus-known` (the lesson is admitted), `memory-served`
(retrieval can answer it from the admitted corpus, with provenance), and
`weight-consolidated` (the transformer learned the behavior and passed the
promotion gates). A correct retrieved answer proves the corpus contains the
answer; it does not prove the weights learned it. The states are defined in
[Language model](./language-model.md).

## Large Models vs QuarkLM

Large language models usually learn broad world knowledge by absorbing massive
mixed corpora into weights, then specialize through prompting, retrieval,
fine-tuning, or adapters. QuarkLM explores the opposite direction. Knowledge is
first admitted into a tiny owned world, served by exact retrieval when possible,
and only later considered for weight consolidation if the guarded trainer proves
the update improves behavior without breaking older evidence.

| Conventional large-model path | QuarkLM path |
| --- | --- |
| Broad pretraining creates a fluent base before a user ever adds data. | Random weights start with no world knowledge; admitted data creates the world. |
| Retrieval often augments an already capable model at inference time. | Retrieval memory is the first auditable expression of admitted knowledge. |
| Fine-tuning or adapters may update behavior after the base already knows many things. | Guarded updates are the only route from admitted corpus evidence into weights. |
| Success is often measured through aggregate benchmark movement. | Promotion requires closed-world constraints, retention, target coverage, and current evidence. |

## Read First

| Page | Covers |
| --- | --- |
| [Project overview](./project-overview.md) | The repository front door: release posture, public surfaces, and where the long evidence trail lives. |
| [Language model](./language-model.md) | The memory-native model philosophy, closed-world boundaries, and the three evidence states. |
| [Self-improvement loop](./self-improvement-loop.md) | The corpus-to-memory-to-weight-update lifecycle and its promotion rule. |
| [Research grounding](./research-grounding.md) | The paper-backed design rules for closed-world self-improvement. |
| [Open-source mechanics audit](./open-source-mechanics-audit.md) | The gap matrix from studying comparable open-source mechanics without copying code or data. |
| [Branch diversity research](./branch-diversity-research.md) | The v0.115 hidden-projection candidate evidence and external research on branch collapse. |
| [Forward research plan](./forward-research-plan.md) | The v0.69 strategy sequence through the v0.115.0 hidden-projection candidate screen. |
| [Deep research review](./deep-research-review.md) | The v0.70 cross-referenced literature, open-source mechanics, gap review, and v0.115.0 routing-repair handoff. |
| [Research implementation map](./research-implementation-map.md) | The v0.74 source-to-gap-to-version map and the v0.115.0 hidden-projection candidate evidence. |
| [Current evidence](./current-evidence.mdx) | The latest promoted metrics and audits. |
| [Historical evidence archive](./historical-evidence.md) | Older evidence that was moved out of `GOAL.md`. |

## What "docs" mean here

In QuarkLM, docs are not training input. They are a promotion gate and an
anti-drift discipline: a run is promoted only if it updates the docs that
describe current state, so the docs and the released evidence move together. The
neural weights never read these pages; they learn only from the admitted corpus.

## Where the project stands

The honest current status follows the same evidence discipline these pages
describe. Retrieval memory answers admitted probes exactly, with provenance and
no weight updates — that is `memory-served` evidence for the memory-first rail,
not a claim about learned weights. The from-scratch transformer is the
`weight-consolidation` path, and it is not promoted: its screens are rejected on
`branch_diversity_target`. See [Transformer](../build/transformer.md) for the
current evidence and [Current evidence](./current-evidence.mdx) for promoted
metrics.

That makes the project more like a lab organism than a product assistant. The
important question is not yet "what can it answer?" The important question is
whether a model can grow only from its own admitted dataset while leaving
trustworthy evidence of each step.
