---
title: Deep Research Review
description: The v0.70 cross-referenced research and implementation-gap review for QuarkLM.
---

# Deep Research Review

<p className="qlm-meta"><span>6 min read</span><span>For contributors</span><span>Updated 2026-06-14</span></p>

<div className="qlm-lead">

**What you will learn**

- Why v0.70 is a research checkpoint, not a model improvement
- The eight operating-system pieces a new training mechanic must sit inside
- Which sources the review cross-referenced, and why none of them is a source of weights or data
- The structural codebase gap the review found, and the checkpoint sequence it set in motion
- Why the transformer remains unpromoted and blocked on branch diversity

</div>

The full review lives in the repository root at `DEEP_RESEARCH_REVIEW.md`.

v0.70 is a research checkpoint, not a model improvement. The earlier forward
plan named the right direction, but the project needed a deeper cross-reference
pass — primary papers, official open-source mechanics, and the current codebase
read together — before adding more training mechanics. This page records the
decision that pass produced.

## The decision

QuarkLM should stop adding direct-answer objectives until the learning loop has
an operating system around it. A new mechanic is justified only when it sits
inside that system, not when it moves a loss number on its own.

The required pieces are:

<div className="qlm-grid">
<div><h4>Experiment registry</h4><p>A recorded hypothesis, acceptance gate, and decision for every screen.</p></div>
<div><h4>Training recipes</h4><p>An explicit model, data, objective, optimizer, and replay specification.</p></div>
<div><h4>Corpus hygiene</h4><p>Duplicate and train/eval overlap checks over the admitted corpus.</p></div>
<div><h4>Candidate quarantine</h4><p>Generated material held outside training until verified and admitted.</p></div>
<div><h4>Deterministic closed-world verifier</h4><p>A pass/fail check on the data boundary before evidence is trusted.</p></div>
<div><h4>Extracted replay planner</h4><p>Replay planning moved out of the model module into its own surface.</p></div>
<div><h4>Constraint-first promotion</h4><p>Gates that run before any loss, rank, or quality number can count.</p></div>
<div><h4>Transformer module boundaries</h4><p>The model split into inspectable surfaces rather than one file.</p></div>
</div>

That is the path that keeps "I learned something new" from meaning
"I generated something new." Inside this system the phrase means proposed,
quarantined, verified, admitted, trained, evaluated, and promoted — in that
order. Generated material is not training data until it is verified against
admitted sources and admitted to `corpus/ledger.json`.

<div className="qlm-keypoint">

**A mechanic earns its place inside the system, not on a loss number**

A new objective is justified only when it sits inside the operating system above.
Moving a loss number on its own is not justification — that is how "I learned
something new" quietly degrades into "I generated something new."

</div>

## Sources cross-referenced

The review reads the design literature and official open-source mechanics
together. The clusters are:

- continual-learning surveys for staged updates and catastrophic forgetting;
- replay systems such as Deep Generative Replay, Avalanche, and Reverb;
- self-generated-data methods such as Self-Instruct, STaR, Self-Refine, and
  Reflexion;
- self-feedback and self-judgment risks, including self-bias and reward
  hacking;
- verifiable-reward systems such as Tulu 3 and DeepSeek-R1;
- model-collapse studies on recursive synthetic training;
- small-model and data-centric work such as TinyStories, BabyLM, and SmolLM2;
- transparent open-model practice from Pythia, LLM360, OLMo, OLMo 2, and Dolma;
- implementation references from nanoGPT, minGPT, GPT-NeoX, LLM Foundry,
  LitGPT, Avalanche, Open-Instruct, and Hugging Face tokenizers.

:::note

All of these are design references only. None of them is a source of weights or
data. QuarkLM still forbids pretrained weights, pretrained tokenizers, external
embeddings, external datasets, copied code, and external-model-shaped training
data.

:::

The version-by-version map from each source cluster to the mechanic it
motivated is kept in the
[Research implementation map](./research-implementation-map.md); the gap matrix
that compares QuarkLM against those mechanics is in the
[Open-source mechanics audit](./open-source-mechanics-audit.md).

## The codebase gap

The strongest local finding is structural rather than about model quality.

QuarkLM has serious transformer mechanics, but at the time of the review
`src/transformer_char_model.py` owned too many responsibilities in one
9,494-line module: the model, the optimizer, the direct-answer objectives,
replay planning, snapshot scoring, CLI parsing, checkpoint writing, and run
reporting. A module that wide cannot be audited screen by screen.

The self-improvement path was already cleaner. It records prompt leakage,
forgetting, exact eval, promotion gates, corpus snapshots, corpus diffs,
attempt archives, and deterministic self-diagnosis. The conclusion is that the
transformer path needs the same discipline — separable surfaces and recorded
evidence — before another large repair screen runs.

## What followed

The review set a sequence of research-control checkpoints, not direct-answer
knobs. Each one builds part of the operating system above, and each is recorded
where its evidence belongs rather than restated here:

<ol className="qlm-steps">
<li><strong>v0.71 — Experiment registry</strong><p>Experiment registry and run-intent schemas.</p></li>
<li><strong>v0.72 — Standalone replay planner</strong><p>Replay planning extracted into `src/replay_plan.py`.</p></li>
<li><strong>v0.73 — Corpus hygiene and training plan</strong><p>Hygiene and training-plan artifacts (`corpus_hygiene.json`, `training_plan.json`).</p></li>
<li><strong>v0.74 — Research implementation map</strong><p>The <a href="./research-implementation-map.md">Research implementation map</a>, tying mechanics to sources, gaps, and acceptance evidence.</p></li>
<li><strong>v0.75 — Candidate quarantine</strong><p>Candidate quarantine artifacts and lifecycle state.</p></li>
<li><strong>v0.76 — Closed-world verifier</strong><p>Deterministic closed-world verifier checks.</p></li>
<li><strong>v0.77 — Recipes and promotion gates</strong><p>Recipes and constraint-first promotion gates.</p></li>
<li><strong>v0.78–v0.80 — Transformer module split</strong><p>Transformer experiment, artifact, model/config, checkpoint, and eval surfaces split out of the model module.</p></li>
</ol>

The per-screen direct-answer history that followed — every objective name, its
attempt and acceptance counts, and its rejection evidence — is the job of the
[Transformer screen history](../build/transformer-screen-history.md). The
forward strategy through the current screen is in the
[Forward research plan](./forward-research-plan.md). This page does not
duplicate either; it records why that history is structured as a chain of
rejected diagnostics rather than a sequence of promotions.

## Operating rule

The review leaves one durable rule for every later screen:

<div className="qlm-keypoint">

**Metrics are not promotion criteria**

No larger transformer screen runs without an experiment-intent artifact, a
corpus plan, a replay plan, verifier checks, and explicit promotion
constraints. Loss, rank, top-k, and NLL are useful metrics, but they are not
promotion criteria. A snapshot is promoted only after retention, leakage,
unknown-policy, coverage, diversity, and contamination gates pass first.

</div>

This is why the screens after v0.70 read as a long list of accepted guarded
updates that still reject promotion. Many of them improve coverage or
diagnostics; none has cleared the gate.

## Routing-repair handoff

The open blocker is `branch_diversity_target`. Multi-target eval profiles
collapse to too few predicted branch tokens: the weights learn to emit one
dominant token instead of routing each prompt to its own answer. From v0.112
the failure is classified as a critical `target_routing_gap`, and the screens
through v0.115.0 instrument it rather than clear it — branch routing audits,
logit-prior and centroid-separation summaries, and a hidden-projection margin
candidate. The current candidate and the external research behind it are
detailed in [Branch diversity research](./branch-diversity-research.md).

Retrieval memory answers `219/219` eval probes exactly, with provenance and no
weight updates. That is evidence for the memory-served rail, not for neural
promotion. The transformer remains unpromoted and blocked on branch diversity:
the system can serve every admitted answer while the weights have not yet
learned to route them.

<div className="qlm-keypoint">

**`memory-served` is not `weight-consolidated`**

Retrieval answering every admitted probe proves the corpus contains the answer.
It does not prove the weights learned to route to it. The two rails stay
labeled separately so a green memory rail can never be read as a promoted
transformer.

</div>

<div className="qlm-next">
<a href="./branch-diversity-research.md"><strong>Read next</strong><span>Branch diversity research</span><small>The open `branch_diversity_target` blocker and the candidate repair.</small></a>
<a href="./forward-research-plan.md"><strong>Read next</strong><span>Forward research plan</span><small>The forward strategy through the current screen.</small></a>
<a href="./research-implementation-map.md"><strong>Read next</strong><span>Research implementation map</span><small>Each source cluster tied to the mechanic it motivated.</small></a>
</div>
