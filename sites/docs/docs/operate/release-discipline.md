---
title: Release Discipline
description: Promotion gates for QuarkLM releases.
---

# Release Discipline

<p className="qlm-meta"><span>6 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- Why a finished build is not a release, and what a promoted run must leave behind
- The run, eval, gate, and artifact evidence every promotion has to produce
- How QuarkLM versions are named under SemVer, and what the docs gate requires
- Why the from-scratch transformer is documented but not promoted

</div>

A QuarkLM release is not a build that finished. It is a run that passed its
gates and left behind the evidence that proves it. This page lists what a
promoted run must produce, how versions are named, and the rules that decide
when a run is allowed to be called a release.

<div className="qlm-keypoint">

**A failed run is still evidence**

A run that fails a gate is kept as versioned diagnostic evidence, not promoted.
A report can remain useful even when it fails; it is still not a release.

</div>

## What a promoted run must have

A promotion has three parts: the run itself, the audits that inspect it, and
the artifacts that record the decision. All three must be present.

### Run and evals

<div className="qlm-grid">
<div><h4>Named RC track</h4><p>A named release-candidate track when tagging or announcing a candidate, keeping alpha and RC tracks separate. See <a href="./release-candidate.md">release candidate readiness</a>.</p></div>
<div><h4>Versioned run directory</h4><p>Provenance: the evidence has a fixed home under <code>runs/</code>.</p></div>
<div><h4>Baseline and final metrics</h4><p>Shows what changed across the run.</p></div>
<div><h4>Three responder evals</h4><p>Responder, answer-model, and decoder evals, covering the three trained answer surfaces.</p></div>
<div><h4>Generated-probe audits</h4><p>Admission-probe and glossary-probe audits, confirming evals are generated from admitted text, not hand-seeded.</p></div>
<div><h4>Prompt leakage audit</h4><p>Confirms eval prompts did not leak into training.</p></div>
<div><h4>Forgetting audit</h4><p>Run against the prior promoted run, confirming the update did not lose previously held behavior.</p></div>
<div><h4>Exact eval audit</h4><p>Records exact pass/fail counts rather than averaged scores.</p></div>
</div>

### Gates that must pass

<div className="qlm-grid">
<div><h4>Promotion gate</h4><p>The run cleared the constraints and is eligible to be promoted.</p></div>
<div><h4><code>npm run check</code></h4><p>Repository and docs checks pass.</p></div>
<div><h4><code>npm run alpha:gate</code></h4><p>Required before any alpha tag.</p></div>
</div>

<div className="qlm-keypoint">

**A failed gate cannot pass as a promotion**

`self_improve answer-cycle` returns failure when the promotion gate fails, so a
failed run cannot be mistaken for a promoted one.

</div>

### Required artifacts

Each promoted run carries machine-checkable JSON so the run can be audited
rather than trusted. The same artifacts appear in the
[operating surfaces](./index.md) overview and in the
[transformer](../build/transformer.md) evidence stack.

<div className="qlm-grid">
<div><h4><code>experiment_intent.json</code></h4><p>Hypothesis, allowed data, planned artifacts, acceptance gates, failure criteria, and final decision. See <a href="./experiment-registry.md">experiment registry</a>.</p></div>
<div><h4><code>corpus_hygiene.json</code></h4><p>Source mixture, duplicates, train/eval overlap, candidate ratio, and rare-profile evidence. See <a href="./corpus-hygiene.md">corpus hygiene</a>.</p></div>
<div><h4><code>training_plan.json</code></h4><p>Allowed data sources, scheduled example mixture, replay-plan status, and planned artifacts.</p></div>
<div><h4><code>training_recipe.json</code></h4><p>Model, tokenizer, data, objective, optimizer, artifact, gate, replay, and rerun details. See <a href="./training-recipes.md">training recipes</a>.</p></div>
<div><h4><code>candidate_quarantine.json</code></h4><p>Candidate lifecycle state, with proof that candidate records are not training data until admitted. See <a href="./candidate-quarantine.md">candidate quarantine</a>.</p></div>
<div><h4><code>closed_world_verifier.json</code></h4><p>Deterministic pass/fail evidence for candidate checks and training-plan approval. See <a href="./closed-world-verifier.md">closed-world verifier</a>.</p></div>
<div><h4><code>constraint_first_promotion.json</code></h4><p>Proof that quality metrics were blocked until closed-world constraints passed.</p></div>
</div>

### Provenance and docs

<div className="qlm-grid">
<div><h4>External-model-free diagnosis</h4><p>Self-diagnosis with <code>uses_external_model: false</code>: recommendations come from report evidence, not an external model, unless a future release explicitly admits and documents a different source.</p></div>
<div><h4>Archived attempts</h4><p>Every attempt is kept under <code>attempts/attempt-###/</code>, not only the accepted one.</p></div>
<div><h4>Corpus snapshot and diff</h4><p>Records the ledger state and how it changed from the prior promoted run.</p></div>
<div><h4>Docs updated for current state</h4><p>The release is not complete until the docs are.</p></div>
<div><h4>Forbidden-claim review</h4><p>Alpha gate, RC spec, gap audit, and checklist reviewed so no surface overstates what the run shows.</p></div>
</div>

## Version naming

New release identifiers use SemVer (Semantic Versioning) with
`MAJOR.MINOR.PATCH` versions and optional prerelease labels.

<div className="qlm-grid">
<div><h4>Alpha prereleases carry a <code>v</code></h4><p>The first proposed alpha is a SemVer prerelease, tagged with a <code>v</code> prefix: <code>0.115.0-alpha.1</code>, Git tag <code>v0.115.0-alpha.1</code>.</p></div>
<div><h4>Minors roll over numerically</h4><p>The release after <code>0.99.0</code> is <code>0.100.0</code>, not <code>v1.00</code>; the line then advances <code>0.101.0</code>, <code>0.102.0</code>.</p></div>
<div><h4><code>1.0.0</code> is deliberate</h4><p>It is reserved for a stable milestone, not reached <code>1.0.0</code> by accident.</p></div>
<div><h4>No placeholders</h4><p>Placeholders are not allowed in release docs. No <code>XX.YY.ZZ</code>.</p></div>
<div><h4>Historical names are frozen</h4><p>Historical artifacts keep their existing names, so run paths such as <code>runs/transformer-answer-v0.42/</code> stay as named and provenance remains exact.</p></div>
</div>

## Single-responsibility blocker

Single Responsibility Principle is part of release discipline, not a separate
code-quality concern. Alpha cannot be tagged while `ALPHA_GATE.md` records P0
source or test files that combine many responsibilities in thousands of lines.
This is the current blocker on the alpha tag; see
[release candidate readiness](./release-candidate.md).

## Docs move with the version

The release is not complete until the docs are complete. A page that references
current eval counts, commands, run ids, hosting targets, or roadmap commitments
must move with the version that changes them.

<div className="qlm-keypoint">

**Docs are a promotion gate, not training input**

Updating the docs is part of clearing a release and an anti-drift discipline. It
is not a source the weights learn from; see [docs drift](./docs-drift.md).

</div>

## Diagnosis without external models

`self_diagnose` can be run directly against a report to inspect the recommended
next action.

```bash title="Inspect the recommended next action for a report"
self_diagnose <report>
```

The recommendation must come from report evidence, not from an external model.
This keeps the closed-world boundary intact even when reading a failed run; see
[purity boundary](../secure/purity-boundary.md).

## Prototypes are documented for what they show

Architecture prototypes, such as the v0.24 transformer, can be documented as
evidence only for the behavior they actually demonstrate. Lower language-model
loss is not a reliable-answer claim until answer evals pass.

From v0.71 onward, transformer answer-training screens also write experiment
intent artifacts, but those screens close as rejected screen evidence until a
dedicated transformer promotion gate exists. The from-scratch transformer is not
promoted; it is blocked on `branch_diversity_target`.

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

Retrieval memory answering its eval probes exactly is `memory-served`, not
`weight-consolidated`, and counts as memory evidence rather than neural
promotion. See [transformer](../build/transformer.md).

</div>

:::tip
Reading the evidence in order — intent, hygiene, plan, recipe, quarantine,
verifier — is the fastest way to tell whether a run earned its tag or was kept
as a diagnostic.
:::

## What is next

<div className="qlm-next">

<a href="./release-candidate.md"><strong>Read next</strong><span>Release candidate readiness</span><small>The RC track and the single-responsibility blocker on the alpha tag.</small></a>

<a href="./docs-drift.md"><strong>Read</strong><span>Docs drift</span><small>Why docs are a promotion gate and an anti-drift discipline.</small></a>

<a href="../build/transformer.md"><strong>Concept</strong><span>The transformer</span><small>Why the from-scratch model is documented but not promoted.</small></a>

</div>
