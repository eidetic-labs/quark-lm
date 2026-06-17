---
title: Candidate Quarantine
description: Keep generated lessons, probes, and repair proposals out of training until admitted.
---

# Candidate Quarantine

<p className="qlm-meta"><span>5 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- What the candidate quarantine holds, and why none of it is training data.
- Why a declared-and-empty quarantine is still evidence.
- The two-step rule a candidate must clear before weights can train on it.
- The validated lifecycle a candidate record moves through, and the run artifacts that record it.

</div>

Candidate quarantine is the holding area for material QuarkLM generates about
itself: lessons, probes, repair proposals, diagnosis notes, and memory
proposals. None of it is training data. The quarantine exists so that generated
material can be tracked through a lifecycle without ever leaking into the
ledgered corpus the neural weights learn from.

`src/candidate_quarantine.py` and the required `candidate_quarantine.json`
artifact were added in v0.75 for self-improvement answer cycles and transformer
answer-training runs. The artifact is part of every such run's evidence bundle,
so a screen can be audited rather than trusted.

## Why an empty artifact is still evidence

The quarantine is empty today. QuarkLM has not promoted autonomous candidate
generation, so no candidate records exist to track. That emptiness is recorded
on purpose: it lets the training plan prove the candidate lane exists and that no
candidate record was silently used as training data.

<div className="qlm-keypoint">

**A declared-and-empty control answers the question an absent one leaves open**

An absent control would leave open whether a candidate was ever used as training
data. A declared-and-empty control answers it: the lane exists, and it holds
nothing.

</div>

## The training rule

<div className="qlm-keypoint">

**Candidate records are not a training source**

This is the same boundary the rest of the system enforces: generated text is not
learned behavior, and memory availability is not weight consolidation.

</div>

A generated lesson, probe, repair proposal, diagnosis note, or memory proposal
can train weights only after two things happen, in order:

<ol className="qlm-steps">
<li><strong>Admission into the ledger</strong><p>The material is admitted into the ledgered corpus (<code>corpus/ledger.json</code>) against admitted sources.</p></li>
<li><strong>Conversion to curriculum</strong><p>The curriculum converts the admitted source into training lessons.</p></li>
</ol>

Until both hold, the material stays in quarantine. The candidate record may keep
its own history, but the candidate store itself never feeds the trainer. See
[Self-improvement loop](../learn/self-improvement-loop.md) for the full
lifecycle this rule sits inside.

## Candidate lifecycle

A candidate record moves through a set of validated states. Transitions are
checked, and terminal states cannot move again.

```text title="Validated state transitions"
proposed -> quarantined -> needs_human_review -> verified -> admitted -> trained -> promoted
                                              \-> rejected   (terminal)
                                                                 promoted (terminal)
```

<div className="qlm-grid">
<div><h4><code>proposed</code></h4><p>The record has been generated but not yet placed under quarantine control.</p></div>
<div><h4><code>quarantined</code></h4><p>The record is held out of training while its claims are checked.</p></div>
<div><h4><code>needs_human_review</code></h4><p>A human decision is required before the record can advance.</p></div>
<div><h4><code>verified</code></h4><p>The record passed deterministic checks against admitted sources.</p></div>
<div><h4><code>rejected</code></h4><p>The record was refused. This is a terminal state.</p></div>
<div><h4><code>admitted</code></h4><p>The record was admitted into the ledgered corpus and now has a ledger admission id.</p></div>
<div><h4><code>trained</code></h4><p>The admitted material was converted to curriculum and used in a guarded weight update.</p></div>
<div><h4><code>promoted</code></h4><p>The resulting behavior cleared the promotion gate. This is a terminal state.</p></div>
</div>

Only an `admitted` record carries a ledger admission id, and only material that
has reached the ledger can become curriculum. The gap between `verified` and
`admitted` is deliberate: passing a deterministic check does not by itself put
material in the corpus.

## Run artifacts

Self-improvement attempts write:

| Artifact | Records |
| --- | --- |
| `attempts/attempt-###/candidate_quarantine.json` | Per-attempt candidate lifecycle state. |
| `candidate_quarantine.json` | Candidate lifecycle state at the latest run level. |
| `training_plan.json` | The candidate quarantine path and a summary. |
| `self_improvement_report.json` | The manifest, embedded. |
| `closed_world_verifier.json` | Deterministic quarantine and plan checks. |

Transformer answer-training runs write:

| Artifact | Records |
| --- | --- |
| `candidate_quarantine.json` | Candidate lifecycle state. |
| `training_plan.json` | The candidate quarantine path and a summary. |
| `transformer_answer_metrics.json` | The manifest, embedded. |
| `closed_world_verifier.json` | Deterministic quarantine and plan checks. |

The quarantine path is linked from `training_plan.json`, so generated or
proposed examples cannot become training data without a later admission and
verification path. See [Corpus hygiene](./corpus-hygiene.md) for the rest of the
training-plan trail.

## Verifier link

The deterministic [closed-world verifier](./closed-world-verifier.md), added in
v0.76, reads the quarantine without using an external model. Two of its checks
turn the quarantine into an enforceable contract:

- non-admitted candidates must remain excluded from training examples, and
- any training-eligible candidate must link to a ledger admission id.

A run whose quarantine declaration is missing or invalid does not pass approval.

<div className="qlm-keypoint">

**Verifier approval is narrow by design**

It certifies that the next weight update stayed inside the closed-world data
boundary — not that the model answered any question better.

</div>

:::note
The same boundary applies to everything QuarkLM ingests. Nothing imports
external weights, tokenizers, embeddings, datasets, or training text — see
[Purity boundary](../secure/purity-boundary.md).
:::

## What is next

<div className="qlm-next">

<a href="../closed-world-verifier/"><strong>Read next</strong><span>Closed-world verifier</span><small>The deterministic checks that turn the quarantine into a contract.</small></a>

<a href="../corpus-hygiene/"><strong>Read</strong><span>Corpus hygiene</span><small>The rest of the training-plan trail and source-mixture checks.</small></a>

<a href="../../learn/self-improvement-loop/"><strong>Concept</strong><span>Self-improvement loop</span><small>The full lifecycle this quarantine rule sits inside.</small></a>

</div>
