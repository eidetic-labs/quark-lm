---
title: Learn
description: Understand the QuarkLM model, goal, and current evidence.
slug: /learn/
---

# Learn

<p className="qlm-meta"><span>6 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- Why QuarkLM is a closed-world experiment where knowledge enters only through the admitted corpus
- The core learning loop, and why a generated lesson is a candidate — not training data
- The three evidence states QuarkLM keeps separate: `corpus-known`, `memory-served`, and `weight-consolidated`
- How QuarkLM's direction inverts the conventional large-model path
- Where the project honestly stands today, and which pages to read next

</div>

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

```text title="The core learning loop"
new lesson -> corpus -> retrieval memory -> training candidates -> guarded weight update -> evaluation -> accepted or rejected
```

This is the central Learn concept. A new lesson is not learned just because a
model or tool generated it. Generated material is a candidate, not training
data, until it is verified against admitted sources and admitted to the ledger.
Only then can it enter retrieval memory, produce source-backed training
candidates, move weights through a guarded update, and survive evaluation. The
final state is explicit: accepted into the current model evidence, or rejected
as diagnostic evidence for the next repair.

Each stage in the loop carries a condition that must hold before the next stage
may run:

<div className="qlm-grid">
<div><h4>New lesson</h4><p>The proposed fact, rule, probe, or repair has a declared source.</p></div>
<div><h4>Corpus</h4><p>The lesson is admitted into the closed-world ledger before training can use it.</p></div>
<div><h4>Retrieval memory</h4><p>Memory can serve the admitted knowledge without pretending the transformer learned it.</p></div>
<div><h4>Training candidates</h4><p>Candidate examples are built from admitted data, retrieval evidence, and failure reports.</p></div>
<div><h4>Guarded weight update</h4><p>The update is bounded, auditable, and allowed to fail without promoting.</p></div>
<div><h4>Evaluation</h4><p>Closed-world, retention, branch-diversity, target-coverage, leakage, and quality checks run.</p></div>
<div><h4>Accepted or rejected</h4><p>Passing evidence can promote; failing evidence remains useful but unpromoted.</p></div>
</div>

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

The loop produces three distinct evidence states, and QuarkLM keeps them
separate on purpose: `corpus-known` (the lesson is admitted), `memory-served`
(retrieval can answer it from the admitted corpus, with provenance), and
`weight-consolidated` (the transformer learned the behavior and passed the
promotion gates). A correct retrieved answer proves the corpus contains the
answer; it does not prove the weights learned it. The states are defined in
[Language model](./language-model.md).

</div>

## Large Models vs QuarkLM

Large language models usually learn broad world knowledge by absorbing massive
mixed corpora into weights, then specialize through prompting, retrieval,
fine-tuning, or adapters. QuarkLM explores the opposite direction. Knowledge is
first admitted into a tiny owned world, served by exact retrieval when possible,
and only later considered for weight consolidation if the guarded trainer proves
the update improves behavior without breaking older evidence.

| Conventional large model | QuarkLM |
| --- | --- |
| Broad pretraining creates a fluent base before a user ever adds data. | Random weights start with no world knowledge; admitted data creates the world. |
| Retrieval often augments an already capable model at inference time. | Retrieval memory is the first auditable expression of admitted knowledge. |
| Fine-tuning or adapters may update behavior after the base already knows many things. | Guarded updates are the only route from admitted corpus evidence into weights. |
| Success is often measured through aggregate benchmark movement. | Promotion requires closed-world constraints, retention, target coverage, and current evidence. |

## Read First

<div className="qlm-grid">
<div><h4><a href="./project-overview.md">Project overview</a></h4><p>The repository front door: release posture, public surfaces, and where the long evidence trail lives.</p></div>
<div><h4><a href="./language-model.md">Language model</a></h4><p>The memory-native model philosophy, closed-world boundaries, and the three evidence states.</p></div>
<div><h4><a href="./self-improvement-loop.md">Self-improvement loop</a></h4><p>The corpus-to-memory-to-weight-update lifecycle and its promotion rule.</p></div>
<div><h4><a href="./research-grounding.md">Research grounding</a></h4><p>The paper-backed design rules for closed-world self-improvement.</p></div>
<div><h4><a href="./open-source-mechanics-audit.md">Open-source mechanics audit</a></h4><p>The gap matrix from studying comparable open-source mechanics without copying code or data.</p></div>
<div><h4><a href="./branch-diversity-research.md">Branch diversity research</a></h4><p>The v0.115 hidden-projection candidate evidence and external research on branch collapse.</p></div>
<div><h4><a href="./forward-research-plan.md">Forward research plan</a></h4><p>The v0.69 strategy sequence through the v0.115.0 hidden-projection candidate screen.</p></div>
<div><h4><a href="./deep-research-review.md">Deep research review</a></h4><p>The v0.70 cross-referenced literature, open-source mechanics, gap review, and v0.115.0 routing-repair handoff.</p></div>
<div><h4><a href="./research-implementation-map.md">Research implementation map</a></h4><p>The v0.74 source-to-gap-to-version map and the v0.115.0 hidden-projection candidate evidence.</p></div>
<div><h4><a href="./current-evidence.mdx">Current evidence</a></h4><p>The latest promoted metrics and audits.</p></div>
<div><h4><a href="./historical-evidence.md">Historical evidence archive</a></h4><p>Older evidence that was moved out of <code>GOAL.md</code>.</p></div>
</div>

## What "docs" mean here

<div className="qlm-keypoint">

**Docs are a promotion gate, not training input**

In QuarkLM, docs are not training input. They are a promotion gate and an
anti-drift discipline: a run is promoted only if it updates the docs that
describe current state, so the docs and the released evidence move together. The
neural weights never read these pages; they learn only from the admitted corpus.

</div>

## Where the project stands

The honest current status follows the same evidence discipline these pages
describe. Retrieval memory answers admitted probes exactly, with provenance and
no weight updates — that is `memory-served` evidence for the memory-first rail,
not a claim about learned weights. The from-scratch transformer is the
`weight-consolidation` path, and it is not promoted: its screens are rejected on
`branch_diversity_target`. See [Transformer](../build/transformer.md) for the
current evidence and [Current evidence](./current-evidence.mdx) for promoted
metrics.

:::note

That makes the project more like a lab organism than a product assistant. The
important question is not yet "what can it answer?" The important question is
whether a model can grow only from its own admitted dataset while leaving
trustworthy evidence of each step.

:::

<div className="qlm-next">
<a href="./language-model.md"><strong>Read next</strong><span>Language model</span><small>The memory-native philosophy and the three evidence states.</small></a>
<a href="./self-improvement-loop.md"><strong>Go deeper</strong><span>Self-improvement loop</span><small>The corpus-to-memory-to-weight-update lifecycle and its promotion rule.</small></a>
<a href="../build/index.md"><strong>Switch to mechanics</strong><span>Build</span><small>Modules, data flow, and which path actually changes weights.</small></a>
</div>
