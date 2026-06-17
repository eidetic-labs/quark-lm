---
title: Build
description: How QuarkLM is put together, and where to change it.
slug: /build/
---

# Build

QuarkLM is a small set of Python modules under `src/`, run as top-level modules
with `PYTHONPATH=src` set. This page is the mechanical orientation: what the
components are, how data moves through them, and which path actually changes
neural weights. For the philosophy behind the design, see
[Language model](../learn/language-model.md); for the lifecycle contract, see
[Self-improvement loop](../learn/self-improvement-loop.md).

## Two paths, kept separate

QuarkLM deliberately separates *answering* from *learning*. A reader who keeps
these two paths distinct will understand the rest of the docs.

The **answer path** serves admitted knowledge without moving any weights:

```text
question -> retrieval memory (corpus-only) -> exact answer + provenance
         \-> respond (deterministic responder) -> grounded answer or `unknown`
```

The **learning path** is the only path that changes neural weights, and only
under gates that can reject the change:

```text
lesson -> corpus (ledger.json) -> curriculum -> training candidates
      -> guarded weight update -> evaluation / promotion gate -> accepted or rejected
```

Retrieval answering a probe correctly proves the corpus *contains* the answer.
It does not prove the transformer *learned* it. That distinction —
`memory-served` versus `weight-consolidated` — is enforced everywhere; see the
three evidence states in [Language model](../learn/language-model.md).

## Component map

| Module | Role | Changes weights? |
| --- | --- | --- |
| `curriculum` | Builds `build/train.txt`, `build/valid.txt`, and manifest data from ledgered corpus files. | no |
| `respond` | Deterministic corpus-only responder; the grounded rail that answers or returns `unknown`. | no |
| `memory_retrieval` | Deterministic closed-world retrieval memory; serves admitted knowledge with provenance. | no |
| `answer_model` | Learned answer classifier, trained from random softmax weights. | yes — gated |
| `answer_decoder` | Generative answer decoder, trained from random prompt-conditioned weights. | yes — gated |
| `transformer_char_model` | The from-scratch decoder-only transformer; the weight-consolidation path. | yes — gated |
| `self_improve` | Orchestrates training, evaluation, audits, and run reports. | drives the trained components |
| `self_diagnose` | Reads a run report and emits deterministic repair recommendations (`uses_external_model: false`). | no |

Every module that can change weights does so only through guarded updates a
promotion gate can reject, and the deterministic
[closed-world verifier](../operate/closed-world-verifier.md) must pass before a
screen's evidence is trusted. Nothing imports external weights, tokenizers,
embeddings, or training text — see [Purity boundary](../secure/purity-boundary.md).

## How a lesson becomes (maybe) learned behavior

1. **Admit.** A fact, rule, probe, or repair is ledgered into `corpus/` with
   source context. Until it is named in `corpus/ledger.json`, it is not training
   data. See [Admission workflow](./admission-workflow.md).
2. **Serve.** `curriculum` regenerates training and validation text and
   `memory_retrieval` builds memory cards, so the knowledge is answerable
   immediately — with provenance, and without moving any weights.
3. **Propose.** Training candidates are built from admitted sources and current
   failure reports, then held in
   [candidate quarantine](../operate/candidate-quarantine.md) until the verifier
   clears them.
4. **Consolidate under guard.** The transformer receives only constrained
   pressure from those candidates. An update is *attempted*, not assumed: the
   guard can reject it and restore prior weights.
5. **Evaluate.** Constraint-first promotion runs the verifier, contamination,
   branch-context, coverage, and diversity checks *before* any loss, NLL, or
   exact-quality number is allowed to count.
6. **Promote or keep as diagnostic.** A run is promoted only if it preserves the
   boundary, passes the gates, and updates the docs that describe current state.
   Failed runs stay as versioned diagnostic evidence; they are not discarded.

This is why QuarkLM only says it learned something new after the
admission-and-evidence chain is visible. A run is not promoted because it
completed.

## Where to change things

| Task | Page |
| --- | --- |
| Run the prototype | [Quickstart](./quickstart.md) |
| Teach a new fact | [Admission workflow](./admission-workflow.md) |
| Keep evals generated from admitted text | [Generated probes](./generated-probes.md) |
| Train the transformer prototype | [Transformer](./transformer.md) |
| Understand the transformer surfaces | [Transformer responsibilities](./transformer-responsibilities.md) |

## Rule

New training data must be admitted or generated from admitted corpus files.
Evaluation probes can be checked into the repo, but they are not training data
unless explicitly allowed by the ledger.
