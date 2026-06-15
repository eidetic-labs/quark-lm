---
title: Learn
description: Understand the QuarkLM model, goal, and current evidence.
slug: /learn/
---

# Learn

QuarkLM is a closed-world language model experiment. It starts without
pretrained weights, without a pretrained tokenizer, and without external
embeddings. Knowledge enters only through the admitted corpus and the
corpus-derived lessons generated from it.

The current prototype is intentionally small. Its value is not scale. Its value
is discipline: every claim about learning has a corpus source, a retrieval
record when memory can serve it, a training candidate when weights should learn
from it, a guarded update, an evaluation artifact, and a docs update.

## Core Learning Loop

```text
new lesson -> corpus -> retrieval memory -> training candidates -> guarded weight update -> evaluation -> accepted or rejected
```

This is the central Learn concept. A new lesson is not considered learned just
because a model or tool generated it. It must enter the ledgered corpus, become
available to retrieval memory, produce source-backed training candidates, move
weights only through a guarded update, and then survive evaluation. The final
state is explicit: accepted into the current model evidence or rejected as
diagnostic evidence for the next repair.

| Stage | What must be true |
| --- | --- |
| New lesson | The proposed fact, rule, probe, or repair has a declared source. |
| Corpus | The lesson is admitted into the closed-world ledger before training can use it. |
| Retrieval memory | Memory can serve the admitted knowledge without pretending the transformer learned it. |
| Training candidates | Candidate examples are built from admitted data, retrieval evidence, and failure reports. |
| Guarded weight update | The update is bounded, auditable, and allowed to fail without promoting. |
| Evaluation | Closed-world, retention, branch-diversity, target-coverage, leakage, and quality checks run. |
| Accepted or rejected | Passing evidence can promote; failing evidence remains useful but unpromoted. |

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

| Page | Use it when |
| --- | --- |
| [Language model](./language-model.md) | You want the memory-native model philosophy and closed-world boundaries. |
| [Self-improvement loop](./self-improvement-loop.md) | You want the corpus-to-memory-to-weight-update lifecycle. |
| [Research grounding](./research-grounding.md) | You want the paper-backed design rules for closed-world self-improvement. |
| [Open-source mechanics audit](./open-source-mechanics-audit.md) | You want the gap matrix from studying comparable open-source mechanics without copying code or data. |
| [Branch diversity research](./branch-diversity-research.md) | You want the v0.115 hidden-projection candidate evidence and external research on branch collapse. |
| [Forward research plan](./forward-research-plan.md) | You want the v0.69 strategy sequence through the v0.115.0 hidden-projection candidate screen. |
| [Deep research review](./deep-research-review.md) | You want the v0.70 cross-referenced literature, open-source mechanics, QuarkLM gap review, and v0.115.0 routing-repair handoff. |
| [Research implementation map](./research-implementation-map.md) | You want the v0.74 source-to-gap-to-version map and the v0.115.0 hidden-projection candidate evidence. |
| [Current evidence](./current-evidence.mdx) | You want the latest promoted metrics and audits. |

## Core Idea

Most language models learn broadly first and specialize later. QuarkLM explores
the opposite direction: learn from a tiny admitted world, preserve the boundary,
answer from memory when the world already contains the knowledge, and gradually
consolidate only the updates that survive evaluation.

That makes the project more like a lab organism than a product assistant. The
important question is not "what can it answer?" yet. The important question is:
can a model grow only from its own admitted dataset while leaving trustworthy
evidence of each step?
