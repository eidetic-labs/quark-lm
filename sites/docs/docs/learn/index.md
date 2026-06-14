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
is discipline: every claim about learning has a corpus source, a training run,
an evaluation artifact, and a docs update.

## Read First

| Page | Use it when |
| --- | --- |
| [Language model](./language-model.md) | You want the model philosophy and boundaries. |
| [Self-improvement loop](./self-improvement-loop.md) | You want the release and training loop. |
| [Research grounding](./research-grounding.md) | You want the paper-backed design rules for closed-world self-improvement. |
| [Open-source mechanics audit](./open-source-mechanics-audit.md) | You want the gap matrix from studying comparable open-source mechanics without copying code or data. |
| [Forward research plan](./forward-research-plan.md) | You want the v0.69 implementation sequence for experiment registry, replay extraction, verifier checks, and candidate quarantine. |
| [Current evidence](./current-evidence.mdx) | You want the latest promoted metrics and audits. |

## Core Idea

Most language models learn broadly first and specialize later. QuarkLM explores
the opposite direction: learn from a tiny admitted world, preserve the boundary,
then grow through measured self-improvement.

That makes the project more like a lab organism than a product assistant. The
important question is not "what can it answer?" yet. The important question is:
can a model grow only from its own admitted dataset while leaving trustworthy
evidence of each step?
