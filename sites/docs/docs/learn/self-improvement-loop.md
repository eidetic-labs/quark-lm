---
title: Self-Improvement Loop
description: How QuarkLM improves without leaving the admitted dataset.
---

# Self-Improvement Loop

QuarkLM improves by changing the system that teaches and audits it:

1. Admit or refine corpus data.
2. Regenerate curriculum files.
3. Train learned components from random initialization.
4. Evaluate responder, classifier, and decoder.
5. Audit generated probes, prompt leakage, provenance, forgetting, and exact
   eval coverage.
6. Diagnose the report and name the next action without using an external model.
7. Archive the attempt before updating the latest report pointer.
8. Promote only when the promotion gate passes and docs are current.

## Components

| Component | Role |
| --- | --- |
| `closed_world_lm.curriculum` | Builds `build/train.txt`, `build/valid.txt`, and manifest data. |
| `closed_world_lm.respond` | Reliable corpus-only responder used as a grounded rail. |
| `closed_world_lm.answer_model` | Learned answer classifier trained from random softmax weights. |
| `closed_world_lm.answer_decoder` | Generative answer decoder trained from random prompt-conditioned weights. |
| `closed_world_lm.transformer_char_model` | Experimental decoder-only transformer trained from random weights on the corpus tokenizer. |
| `closed_world_lm.self_improve` | Orchestrates training, evaluation, audits, and run reports. |
| `closed_world_lm.self_diagnose` | Reads a run report and emits deterministic repair recommendations with `uses_external_model: false`. |

## Promotion Rule

A run is not promoted because it completed. A run is promoted only when it
preserves the purity boundary, records baseline and final metrics, passes the
required audits, passes the recorded promotion gate, and updates the docs that
describe current state.

The docs are part of the loop. If README, Docusaurus, or the marketing page
references a current release, that surface must move with the release.

The current diagnosis layer is intentionally rule-based. It is not the final
form of autonomous improvement, but it establishes the interface: QuarkLM should
learn from its own reports, name what changed, and propose the next repair
without another model shaping that decision.

## Research Guardrails

The self-improvement loop is guided by continual learning, lifelong
pretraining, replay, self-generated reasoning, and model-editing research, but
QuarkLM applies those ideas under a stricter data boundary. A generated repair
or lesson is only a candidate until a deterministic verifier accepts it against
admitted sources. Weight updates still come from versioned corpus-derived
curriculum, and every admitted batch must preserve prior accepted behavior
through forgetting checks or replay.

See [Research grounding](./research-grounding.md) for the current paper map and
the design rules it implies.
