---
title: Build
description: How QuarkLM is put together, and where to change it.
slug: /build/
---

# Build

<p className="qlm-meta"><span>6 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- What the QuarkLM modules are, and which ones can move neural weights
- Why the answer path and the learning path are kept strictly separate
- How a lesson travels from admission to (maybe) consolidated behavior
- Where to change things, and the rule that governs new training data

</div>

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

```text title="Answer path — no weight movement"
question -> retrieval memory (corpus-only) -> exact answer + provenance
         \-> respond (deterministic responder) -> grounded answer or `unknown`
```

The **learning path** is the only path that changes neural weights, and only
under gates that can reject the change:

```text title="Learning path — gated weight updates"
lesson -> corpus (ledger.json) -> curriculum -> training candidates
      -> guarded weight update -> evaluation / promotion gate -> accepted or rejected
```

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

Retrieval answering a probe correctly proves the corpus *contains* the answer.
It does not prove the transformer *learned* it. That distinction —
`memory-served` versus `weight-consolidated` — is enforced everywhere; see the
three evidence states in [Language model](../learn/language-model.md).

</div>

## Component map

<div className="qlm-grid">
<div><h4>curriculum</h4><p>Builds <code>build/train.txt</code>, <code>build/valid.txt</code>, and manifest data from ledgered corpus files. No weight movement.</p></div>
<div><h4>respond</h4><p>Deterministic corpus-only responder; the grounded rail that answers or returns <code>unknown</code>. No weight movement.</p></div>
<div><h4>memory_retrieval</h4><p>Deterministic closed-world retrieval memory; serves admitted knowledge with provenance. No weight movement.</p></div>
<div><h4>answer_model</h4><p>Learned answer classifier, trained from random softmax weights. Changes weights — gated.</p></div>
<div><h4>answer_decoder</h4><p>Generative answer decoder, trained from random prompt-conditioned weights. Changes weights — gated.</p></div>
<div><h4>transformer_char_model</h4><p>The from-scratch decoder-only transformer; the weight-consolidation path. Changes weights — gated.</p></div>
<div><h4>self_improve</h4><p>Orchestrates training, evaluation, audits, and run reports. Drives the trained components.</p></div>
<div><h4>self_diagnose</h4><p>Reads a run report and emits deterministic repair recommendations (<code>uses_external_model: false</code>). No weight movement.</p></div>
</div>

Every module that can change weights does so only through guarded updates a
promotion gate can reject, and the deterministic
[closed-world verifier](../operate/closed-world-verifier.md) must pass before a
screen's evidence is trusted. Nothing imports external weights, tokenizers,
embeddings, or training text — see [Purity boundary](../secure/purity-boundary.md).

## How a lesson becomes (maybe) learned behavior

<ol className="qlm-steps">
<li><strong>Admit</strong><p>A fact, rule, probe, or repair is ledgered into <code>corpus/</code> with source context. Until it is named in <code>corpus/ledger.json</code>, it is not training data. See <a href="./admission-workflow.md">Admission workflow</a>.</p></li>
<li><strong>Serve</strong><p><code>curriculum</code> regenerates training and validation text and <code>memory_retrieval</code> builds memory cards, so the knowledge is answerable immediately — with provenance, and without moving any weights.</p></li>
<li><strong>Propose</strong><p>Training candidates are built from admitted sources and current failure reports, then held in <a href="../operate/candidate-quarantine.md">candidate quarantine</a> until the verifier clears them.</p></li>
<li><strong>Consolidate under guard</strong><p>The transformer receives only constrained pressure from those candidates. An update is <em>attempted</em>, not assumed: the guard can reject it and restore prior weights.</p></li>
<li><strong>Evaluate</strong><p>Constraint-first promotion runs the verifier, contamination, branch-context, coverage, and diversity checks <em>before</em> any loss, NLL, or exact-quality number is allowed to count.</p></li>
<li><strong>Promote or keep as diagnostic</strong><p>A run is promoted only if it preserves the boundary, passes the gates, and updates the docs that describe current state. Failed runs stay as versioned diagnostic evidence; they are not discarded.</p></li>
</ol>

<div className="qlm-keypoint">

**A run is not promoted because it completed**

QuarkLM only says it learned something new after the admission-and-evidence
chain is visible. Completion is not promotion: a run earns promotion by
preserving the boundary, passing the gates, and updating the docs — not by
finishing.

</div>

## Where to change things

<div className="qlm-grid">
<div><h4><a href="./quickstart.md">Quickstart</a></h4><p>Run the prototype end to end, and read what each command produces.</p></div>
<div><h4><a href="./admission-workflow.md">Admission workflow</a></h4><p>Teach a new fact by ledgering it into the closed-world corpus.</p></div>
<div><h4><a href="./generated-probes.md">Generated probes</a></h4><p>Keep evaluation probes generated from admitted text, not hand-written.</p></div>
<div><h4><a href="./transformer.md">Transformer</a></h4><p>Train the from-scratch transformer prototype and read its screens.</p></div>
<div><h4><a href="./transformer-responsibilities.md">Transformer responsibilities</a></h4><p>Understand the transformer surfaces and what each one owns.</p></div>
</div>

## Rule

<div className="qlm-keypoint">

**New training data must be admitted or generated from admitted corpus files**

Evaluation probes can be checked into the repo, but they are not training data
unless explicitly allowed by the ledger.

</div>

<div className="qlm-next">
<a href="./quickstart.md"><strong>Read next</strong><span>Quickstart</span><small>Run the prototype and read what each command produces.</small></a>
<a href="./admission-workflow.md"><strong>Go deeper</strong><span>Admission workflow</span><small>Ledger a new fact into the closed-world corpus.</small></a>
<a href="../learn/language-model.md"><strong>Switch to the why</strong><span>Language model</span><small>The memory-native philosophy and the three evidence states.</small></a>
</div>
