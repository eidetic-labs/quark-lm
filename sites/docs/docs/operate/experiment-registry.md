---
title: Experiment Registry
description: Run-intent artifacts for QuarkLM training evidence.
---

# Experiment Registry

<p className="qlm-meta"><span>6 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- What an `experiment_intent.json` record declares, and which fields are validated on write and on read
- How an intent is opened in the `planned` state and closed with a `promoted`, `rejected`, or `aborted` decision
- Which two run types write intents, and what each records
- Why the registry validates and records but never decides promotion itself

</div>

v0.71 adds `src/experiment_registry.py` so a training run begins with an explicit,
validated intent rather than a loose command. The intent is written before the
weights move, and the same artifact is closed with the outcome after the
promotion gate has spoken. A run is therefore auditable against the plan it
declared, not against a description reconstructed afterward.

<div className="qlm-keypoint">

**The registry records intent; it does not grant promotion**

The registry is deliberately small: JSON artifacts, pure validation, and no
hidden promotion behavior. It records what a run intended; it does not decide
whether the run passed. Promotion is decided separately by the
[closed-world verifier](./closed-world-verifier.md) and the constraint-first
report in [training recipes](./training-recipes.md).

</div>

## The intent contract

Every experiment writes one `experiment_intent.json` record. Each field is
validated on write and on read, so a malformed or empty intent fails fast rather
than producing untrustworthy evidence.

<div className="qlm-grid">
<div><h4><code>version</code> / <code>run_id</code></h4><p>The QuarkLM version and the run the intent belongs to.</p></div>
<div><h4><code>component</code></h4><p>The component under test (for example, a transformer answer-training run).</p></div>
<div><h4><code>hypothesis</code></h4><p>The claim the run is meant to test.</p></div>
<div><h4><code>allowed_data_sources</code></h4><p>The data the run is permitted to draw from, stated up front.</p></div>
<div><h4><code>planned_artifacts</code></h4><p>The evidence files the run commits to emitting.</p></div>
<div><h4><code>training_recipe_id</code></h4><p>The reproducible recipe the run follows; see <a href="../training-recipes/">training recipes</a>.</p></div>
<div><h4><code>acceptance_gates</code></h4><p>Named gates, each with a rule and a <code>required</code> flag, that the run must clear to promote.</p></div>
<div><h4><code>failure_criteria</code></h4><p>The conditions that mark the run a failure.</p></div>
<div><h4><code>replay_plan_id</code></h4><p>An optional replay-plan reference, or null.</p></div>
<div><h4><code>notes</code></h4><p>Free-text context.</p></div>
<div><h4><code>decision</code></h4><p>The closing outcome, written when the gate result is known.</p></div>
</div>

Required string fields must be non-empty; `allowed_data_sources`,
`planned_artifacts`, and `failure_criteria` must each be non-empty lists of
non-empty strings. Acceptance gate names must be unique. Declaring allowed data
sources in the intent is what lets a later audit confirm the run stayed inside
the closed-world boundary.

## Open intent, then close it

An intent has two moments. It is opened in the `planned` state before training,
and it is closed with a decision once the promotion gate has run.

```text title="Intent lifecycle, open to close"
attempt dir exists
  -> write experiment_intent.json   (decision.status = planned, promoted = false)
  -> train under guards
  -> promotion gate decides
  -> close intent with decision     (promoted | rejected | aborted)
  -> copy final intent into the attempt report and the latest run report
```

<div className="qlm-keypoint">

**An intent cannot disagree with itself**

The closing decision is the only place the intent reports promotion. The
`decision.promoted` flag must equal whether the status is `promoted`, so an
intent can never claim it was promoted while its status says otherwise. A
planned intent always starts with `promoted: false`.

</div>

The decision status records which of four outcomes the run reached.

<div className="qlm-grid">
<div><h4><code>planned</code></h4><p>Intent recorded before training; no outcome yet.</p></div>
<div><h4><code>promoted</code></h4><p>The run cleared its gates and the change was accepted.</p></div>
<div><h4><code>rejected</code></h4><p>The run ran but did not clear its gates.</p></div>
<div><h4><code>aborted</code></h4><p>The run did not complete to a gate decision.</p></div>
</div>

A rejected or aborted run is kept as versioned diagnostic evidence, not
discarded. The intent that opened it stays attached to the report so the
attempt remains readable.

## Who writes intents

Two run types write `experiment_intent.json`.

<div className="qlm-grid">
<div><h4>Self-improvement answer cycles</h4><p>Write the intent as soon as an attempt directory exists, then close it with the promotion-gate result. The final intent is copied into both the attempt report and the latest run report.</p></div>
<div><h4>Transformer answer-training runs</h4><p>Write the intent before training. They record baseline and final snapshot gates, closed-world data checks, no-pretrained-weight, no-pretrained-tokenizer, and no-external-embedding checks, and direct-answer branch-screen gates where they apply.</p></div>
</div>

From v0.77 onward, a transformer run closes through the constraint-first
promotion report: quality metrics such as loss, NLL, or exact-answer counts can
affect the decision only after the closed-world constraints have passed. See
[Transformer](../build/transformer.md) for how those gates sit in the wider
answer-training stack, and [candidate quarantine](./candidate-quarantine.md) for
why generated candidates are not training data until admitted.

## What an intent does not do

<div className="qlm-keypoint">

**Declaring a gate is not enforcing it**

The registry validates and records; it does not promote. An intent declares the
gates and the allowed data, but the gates themselves are enforced by the
verifier and the constraint-first report, and the data boundary is enforced by
the corpus ledger.

</div>

The intent is the run's stated plan and its honest record of how that plan turned
out.

<div className="qlm-next">
<a href="../closed-world-verifier/"><strong>Read next</strong><span>Closed-world verifier</span><small>The deterministic gate that enforces the data boundary the intent declares.</small></a>
<a href="../training-recipes/"><strong>Go deeper</strong><span>Training recipes</span><small>The constraint-first promotion report that closes a transformer intent.</small></a>
<a href="../candidate-quarantine/"><strong>Reference</strong><span>Candidate quarantine</span><small>Why generated candidates are not training data until admitted.</small></a>
</div>
