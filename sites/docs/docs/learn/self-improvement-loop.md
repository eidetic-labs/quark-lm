---
title: Self-Improvement Loop
description: How QuarkLM improves without leaving the admitted dataset.
---

# Self-Improvement Loop

QuarkLM improves by changing the system that teaches and audits it:

```text
new lesson -> corpus -> retrieval memory -> tokenizer candidates -> training candidates -> guarded weight update -> evaluation -> accepted or rejected
```

That lifecycle is the difference between QuarkLM and a conventional large-model
workflow. A large pretrained model usually starts with world knowledge already
encoded in its weights, then uses a smaller supervised or retrieval layer to
shape behavior. QuarkLM starts with no world knowledge in weights. A lesson is
first ledgered into the corpus, memory can answer it exactly, tokenizer
candidates are proposed from admitted text, the trainer builds closed-world
candidates from that evidence, and only guarded updates are allowed to modify
weights.

## Lifecycle Contract

| Step | System responsibility | Evidence artifact |
| --- | --- | --- |
| New lesson | Receive a proposed fact, rule, probe, or repair with source context. | Candidate record or admission request. |
| Corpus | Admit only verified material into the ledgered closed world. | Ledger, corpus diff, curriculum manifest. |
| Retrieval memory | Make admitted knowledge answerable without weight movement. | Retrieval memory cards and exact retrieval evals. |
| Tokenizer candidates | Propose corpus-only compression without changing the active tokenizer silently. | Tokenizer manifest, tokenizer report, tokenizer candidate guard. |
| Training candidates | Convert admitted sources and failure reports into bounded examples. | Training plan, replay plan, candidate quarantine, source map. |
| Guarded weight update | Apply only constrained pressure to random-initialized or closed-world checkpointed weights. | Update guard, accepted/rejected attempt records, checkpoint metadata. |
| Evaluation | Test current behavior before promotion is allowed. | Constraint-first promotion, forgetting audit, probe audits, branch metrics. |
| Accepted or rejected | Promote only passing evidence; keep failed runs as diagnostics. | Current-state docs, run report, release notes, archived attempt. |

This is why QuarkLM can say "I learned something new" only after the admission
and evidence chain is visible. A retrieved answer means the corpus can serve the
knowledge. A promoted guarded update means the learned model consolidated
behavior from that knowledge without breaking the boundary.

Release loop:

1. Admit or refine corpus data.
2. Regenerate curriculum files and retrieval memory artifacts.
3. Generate tokenizer candidates and reject unsafe vocabulary before model claims.
4. Build training candidates from admitted sources and current failure reports.
5. Train learned components from random initialization or a declared closed-world
   checkpoint.
6. Evaluate responder, classifier, decoder, transformer, and retrieval memory.
7. Audit tokenizer candidates, generated probes, prompt leakage, provenance, forgetting, and exact
   eval coverage.
8. Diagnose the report and name the next action without using an external model.
9. Archive the attempt before updating the latest report pointer.
10. Promote only when the promotion gate passes and docs are current.

## Components

| Component | Role |
| --- | --- |
| `curriculum` | Builds `build/train.txt`, `build/valid.txt`, and manifest data. |
| `respond` | Reliable corpus-only responder used as a grounded rail. |
| `answer_model` | Learned answer classifier trained from random softmax weights. |
| `answer_decoder` | Generative answer decoder trained from random prompt-conditioned weights. |
| `transformer_char_model` | Experimental decoder-only transformer trained from random weights on the corpus tokenizer. |
| `self_improve` | Orchestrates training, evaluation, audits, and run reports. |
| `self_improvement_tokenizer` | Writes tokenizer candidate manifests, reports, and guard evidence for each answer cycle. |
| `self_diagnose` | Reads a run report and emits deterministic repair recommendations with `uses_external_model: false`. |

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

## Memory Before Consolidation

QuarkLM treats memory and weights as separate evidence rails. Retrieval memory
can prove that admitted knowledge is available without pretending the neural
model has learned it. Tokenizer candidates sit on the same evidence side of the
boundary: they can prove corpus-only compression and round-trip safety, but they
do not prove the weights learned anything. Training candidates are the bridge:
they decide what parts of retrieved or ledgered knowledge deserve pressure on
the transformer. Guarded weight updates are accepted only if they improve the
targeted profile while preserving closed-world constraints, prior coverage, and
promotion gates.

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
