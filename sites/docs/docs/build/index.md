---
title: Build
description: Build and extend QuarkLM.
slug: /build/
---

# Build

Build docs cover the local commands and extension points for the prototype.
The current import path is still `closed_world_lm`, even though the package and
product name are now QuarkLM / `quark-lm`.

## Common Tasks

| Task | Page |
| --- | --- |
| Run the prototype | [Quickstart](./quickstart.md) |
| Teach a new fact | [Admission workflow](./admission-workflow.md) |
| Keep evals generated | [Generated probes](./generated-probes.md) |
| Train the transformer prototype | [Transformer](./transformer.md) |
| Understand transformer surfaces | [Transformer responsibilities](./transformer-responsibilities.md) |

## Rule

New training data must be admitted or generated from admitted corpus files.
Evaluation probes can be checked into the repo, but they are not training data
unless explicitly allowed by the ledger.
