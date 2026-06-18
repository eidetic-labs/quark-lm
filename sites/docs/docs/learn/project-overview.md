---
title: Project Overview
description: Repository-level orientation for QuarkLM.
---

# Project Overview

QuarkLM is a closed-world language model research prototype. It asks whether a
language model can grow from a tiny owned dataset while keeping every learning
claim tied to admitted sources and promotion evidence.

This page carries the durable project orientation that used to make the README
hard to scan. The README should stay short. The docs carry the model
philosophy, evidence trail, operating rules, and release-candidate boundaries.

## What QuarkLM Is

QuarkLM starts from a constrained world:

- human-authored seed glossary, grammar, stories, self facts, and admitted
  memories;
- deterministic curriculum generated from ledgered corpus files;
- a character tokenizer baseline and guarded append-only subword tokenizer path,
  both trained only on admitted text;
- tiny learned components initialized from random weights;
- corpus-only retrieval memory;
- a tiny decoder-only transformer initialized from random weights.

The project does not claim to be a useful assistant yet. It is a research
system for testing whether self-improvement can remain bounded, inspectable,
and honest about failures.

## What QuarkLM Does Not Use

The current prototype does not use:

- pretrained weights;
- pretrained tokenizers;
- external embeddings;
- unledgered training text;
- external model outputs as training authority.

Generated material can propose lessons, probes, or repairs, but it is not
training data until it is verified against admitted sources and included in the
ledgered curriculum.

## Current Release Posture

QuarkLM separates release-candidate readiness into two tracks:

| Track | Current posture |
| --- | --- |
| Research Prototype RC | Near. The closed-world self-improvement system is reproducible, auditable, documented, and clear about what is not promoted. |
| Language Model RC | Not ready. The from-scratch transformer still fails `branch_diversity_target` after v0.115. |

The current promoted responder evidence is `runs/self-improve-v0.42/`. The
latest transformer screen is
`runs/transformer-answer-v0.115.0-hidden-projection-margin-candidate-step1-dim4-context80/`.
That screen is diagnostic evidence, not promoted neural model evidence.

Use [Release Candidate Readiness](../operate/release-candidate.md),
`RC_SPEC.md`, `RC_GAP_AUDIT.md`, and `RC_CHECKLIST.md` before tagging or
announcing an RC.

## Where The Long Evidence Trail Lives

The historical version narrative belongs in docs, not in README:

| Topic | Canonical docs |
| --- | --- |
| Model philosophy and closed-world boundaries | [Language model](./language-model.md) |
| Learning lifecycle | [Self-improvement loop](./self-improvement-loop.md) |
| Paper-backed control matrix | [Research grounding](./research-grounding.md) |
| Open-source architecture/mechanics comparison | [Open-source mechanics audit](./open-source-mechanics-audit.md) |
| Branch-diversity root cause and v0.115 evidence | [Branch diversity research](./branch-diversity-research.md) |
| Source-to-gap implementation trail | [Research implementation map](./research-implementation-map.md) |
| Latest metrics and run history | [Current evidence](./current-evidence.mdx) |
| Promotion, release, and docs-drift rules | [Operate](../operate/index.md) |

## Public Surfaces

QuarkLM has two public surfaces with separate hosts:

| Surface | Host | Target |
| --- | --- | --- |
| Docusaurus docs | Read the Docs | `docs.quark-lm.eidetic-labs.com` |
| Standalone marketing page | GitHub Pages | `quark-lm.eidetic-labs.com` |

See `sites/DEPLOYMENT.md` for deployment details.

## Repository Orientation

| Path | Purpose |
| --- | --- |
| `corpus/` | Ledgered source files allowed to influence training or evaluation. |
| `src/` | Curriculum, models, responder, retrieval memory, verifier, trainer, and eval surfaces. |
| `tests/` | Regression coverage for core mechanics. |
| `runs/` | Local run evidence and checkpoints; ignored by git. |
| `sites/docs/` | Docusaurus source for Learn, Build, Operate, and Secure docs. |
| `sites/marketing/` | Standalone marketing page source. |
| `sites/shared/current-state.json` | Shared state consumed by docs and marketing. |

## Verification

Use these commands before release-candidate packaging or upload prep:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
npm run sites:build
python3 -m json.tool sites/shared/current-state.json >/dev/null
```

For day-to-day local use, start with [Quickstart](../build/quickstart.md).
