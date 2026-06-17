---
title: Prompt Leakage
description: Prevent held-out prompts from becoming training lessons.
---

# Prompt Leakage

<p className="qlm-meta"><span>5 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- Why a held-out probe is only a fair test if the model never trained on its exact prompt form.
- How the `audit_prompt_leakage` exact-string scan finds leaks, including missing lesson files.
- Which two protected prompt sets the audit covers and how the ownership set is filtered.
- What promotion requires: both leakage checks must report zero leaked prompts.

</div>

A held-out probe is only a fair test if the model has never trained on its exact
prompt form. The prompt-leakage audit enforces that separation: it checks that no
training lesson reuses the verbatim prompt of a protected held-out evaluation. It
is one of the promotion-gate audits a run must pass before any weight update is
promoted (see [Self-improvement loop](../learn/self-improvement-loop.md)).

<div className="qlm-keypoint">

**The held-out prompt is the measuring instrument**

The protected prompt string is the test, not the knowledge. A lesson may teach the
*fact*, but copying the *exact prompt* turns a later correct answer from evidence of
generalization into memorization of the test — and the held-out number stops meaning
anything.

</div>

## Facts may be learned; protected prompts may not

QuarkLM is allowed to learn a held-out *fact* — through ordinary fact-style
lessons admitted to the corpus, the knowledge can be consolidated like any other.
What it must not train on is the *exact prompt string* used to evaluate that fact
in a held-out set. The held-out prompt form is the measuring instrument. If a
lesson copies it verbatim, a later "correct" answer no longer distinguishes
generalization from memorization of the test, and the held-out number stops
meaning anything.

The audit therefore operates on exact prompt strings, not on facts or topics. A
paraphrased lesson about the same fact is fine; a byte-for-byte copy of the
evaluation prompt is a leak.

## How the audit works

`audit_prompt_leakage` (in `self_improvement_audits`) compares lesson files
against one protected evaluation file:

<ol className="qlm-steps">
<li><strong>Collect protected prompts</strong><p>Read the protected eval file and collect the set of its <code>prompt</code> strings.</p></li>
<li><strong>Scan each lesson file</strong><p>Walk every lesson file line by line. A lesson record whose <code>prompt</code> exactly matches a protected eval prompt is recorded as leaked.</p></li>
<li><strong>Treat a missing file as a leak</strong><p>If a named lesson file does not exist, that absence is itself recorded as a leak (<code>&lt;missing lesson file&gt;</code>), so a missing source cannot silently pass.</p></li>
</ol>

The audit reports the eval source, the lesson sources, the list of leaked
prompts, and `passed` — which is true only when no leak was found.

```text title="Exact-match scan: leaked_prompts must be empty"
protected eval prompts ─┐
                        ├─ exact-match scan ─> leaked_prompts (must be empty)
lesson file prompts ────┘
```

## Protected prompt sets

`audit_all_protected_prompts` runs the audit over two protected sets:

<div className="qlm-grid">
<div><h4>Held-out facts</h4><p>Source <code>evals/heldout.jsonl</code>. All records in the file are protected.</p></div>
<div><h4>Held-out ownership</h4><p>Source <code>evals/owner.jsonl</code>. Only records whose <code>id</code> contains <code>-heldout-</code> are protected.</p></div>
</div>

The ownership file mixes known and held-out probes, so the audit filters it to
the held-out portion (`protected_id_contains="-heldout-"`). The known ownership
prompts are answerable and may appear in lessons; only the held-out ones are
protected.

:::note

These two audits surface in the run report as the named checks
`heldout_prompt_leakage` and `owner_heldout_prompt_leakage`.

:::

## What promotion requires

Both checks must report `passed: true` — zero leaked protected prompts — for a
run to clear the promotion gate. They sit alongside the generated-probes,
closed-world verifier, constraint-first, forgetting, and exact-eval audits; a run
is not promoted because it completed, only because every required audit passed
(see [Self-improvement loop](../learn/self-improvement-loop.md)).

## Why it exists

This audit protects the integrity of the evidence, not the weights themselves.

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

QuarkLM keeps `memory-served` and `weight-consolidated` as separate rails, and it
reports held-out numbers as out-of-sample evidence about the learning path. If the
held-out prompt could leak into training, that evidence would quietly collapse
into memorization of the test.

</div>

The leakage audit keeps the held-out sets honest, in the same spirit as the
[purity boundary](./purity-boundary.md): training may only consume admitted
material, and the test set is not it.

## What's next

<div className="qlm-next">
<a href="../learn/self-improvement-loop.md"><strong>Read next</strong><span>Self-improvement loop</span><small>The promotion gate this audit feeds, and the other required audits beside it.</small></a>
<a href="./purity-boundary.md"><strong>Related</strong><span>Purity boundary</span><small>Why training may only consume admitted material — and why probes are not it.</small></a>
<a href="../learn/language-model.md"><strong>Concept</strong><span>The language model</span><small>The memory-served versus weight-consolidated distinction this audit protects.</small></a>
</div>
