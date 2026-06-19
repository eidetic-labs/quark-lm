---
title: Purity Boundary
description: What QuarkLM may and may not train on.
---

# Purity Boundary

<p className="qlm-meta"><span>5 min read</span><span>For contributors</span><span>Updated 2026-06-18</span></p>

<div className="qlm-lead">

**What you will learn**

- The four inputs that may never reach QuarkLM's neural weights, and why each is excluded.
- How `corpus/ledger.json` acts as the single admission gate, with two independent per-source permissions.
- Why a ledgered source can still be barred from training, and why generated material is not yet trainable.
- The one distinction that is easy to miss: studying open-source structure is allowed; importing open-source weights, tokenizers, embeddings, vocabulary, or text is not.

</div>

The purity boundary is the line between knowledge QuarkLM's neural weights are
allowed to learn from and everything else. Its neural weights learn only from the
admitted, ledgered corpus. Nothing arrives pretrained, and no text becomes
training input by being typed, retrieved, or generated. This page states the
prohibitions, the ledger that enforces them, and the one distinction that is easy
to miss: studying open-source structure is allowed; importing open-source weights,
tokenizers, embeddings, or text is not.

<div className="qlm-keypoint">

**The corpus is the only training input**

Every input that touches neural weights — training, validation, tokenizer
fitting, and repair data — must trace back to a source admitted in
`corpus/ledger.json`. Truth is not sufficient; admission is.

</div>

## What may not enter

Four things are excluded from every part of the model that touches neural
weights: training, validation, tokenizer fitting, and repair data.

<div className="qlm-grid">
<div><h4>Pretrained weights</h4><p>They carry knowledge from outside the corpus that cannot be traced to an admitted source.</p></div>
<div><h4>Pretrained tokenizers</h4><p>A pretrained vocabulary encodes a world the corpus never admitted; it crosses the same boundary as pretrained weights.</p></div>
<div><h4>External embeddings</h4><p>They inject learned representations the corpus did not produce.</p></div>
<div><h4>Unledgered training text</h4><p>Any text not named and allowed in <code>corpus/ledger.json</code> is not training data, regardless of how true it is.</p></div>
</div>

These four are not a code-comment policy. They are the same four boundary flags
the deterministic [closed-world verifier](../operate/closed-world-verifier.md)
checks before a training plan is trusted: `pretrained_weights`,
`pretrained_tokenizer`, `external_embeddings`, and `unledgered_training_text`
must all be `false`.

## The ledger is the gate

`corpus/ledger.json` is the single source that decides what is admitted and what
each admitted source may be used for. It carries a `purity_boundary` block that
records the four prohibitions above and a `training_data_policy` stating that only
curriculum generated from allowed source files in the ledger may be used for
training. It then lists every source with two independent permissions:

<div className="qlm-grid">
<div><h4><code>allowed_for_curriculum_generation</code></h4><p>Whether <code>curriculum</code> may derive training and validation text from the source.</p></div>
<div><h4><code>allowed_for_training</code></h4><p>Whether the source's derived material may reach a weight update.</p></div>
</div>

Both permissions default to closed. A source is allowed only when its ledger
entry says so explicitly. The human-authored glossary, the grammar and story
facts, and the admitted-memory log are marked allowed for both; everything else
in the ledger is not.

## Admitted is not the same as trainable

A source can be named in the ledger and still be excluded from training. The
evaluation probe sets — the known, unknown, held-out, paraphrase, ownership,
self, learning, admission, and glossary probes — are all ledgered sources, but
each carries `allowed_for_curriculum_generation: false` and
`allowed_for_training: false`. They are admitted so their provenance is recorded,
not so the model can learn from them. Letting an eval probe into training would
teach the model the test, which is the leak the
[prompt leakage](./prompt-leakage.md) discipline guards against.

```text title="corpus/ledger.json — permission summary"
corpus/ledger.json
  glossary-v0     curriculum: yes   training: yes
  grammar-v0      curriculum: yes   training: yes
  admissions-v0   curriculum: yes   training: yes
  *-probes-v0     curriculum: no    training: no   (admitted for provenance only)
```

## Generated material is not yet trainable

The boundary also holds for material QuarkLM produces itself. A curriculum
lesson, a candidate, or a repair proposal derived from admitted sources is not
training data until it has been verified against admitted sources and admitted to
the ledger. Until then it sits in
[candidate quarantine](../operate/candidate-quarantine.md), where the verifier
confirms a ledger admission link before the material can become training-eligible.
Generation does not bypass admission; it feeds it.

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

Retrieval memory answering a probe is not evidence that the weights learned it.
Retrieval serves admitted knowledge with provenance and moves no weights — it is
`memory-served`, not `weight-consolidated`. The two states are kept separate
everywhere; see [Language model](../learn/language-model.md).

</div>

## Structure may be studied; substance may not be imported

The boundary restricts imported content, not knowledge of how transformers are
built. Per `STRUCTURE_AUDIT.md`, QuarkLM may study open-source model, trainer,
tokenizer, and checkpoint *structure* — config layout, attention and residual
shapes, tokenizer pipeline stages, evaluation artifact formats — and adopt those
shapes when each is recorded in docs, tests, and run evidence. It must not import
external weights, tokenizers, vocabularies, embeddings, datasets, or training text.

The character tokenizer (`tokenizer.CharTokenizer`) is the reference example:
its pipeline structure can follow open-source references, but its vocabulary is
trained from admitted corpus text and rejects out-of-vocabulary characters.

The subword tokenizer (`closed_world_subword_tokenizer.ClosedWorldSubwordTokenizer`)
follows the same rule. It may use known tokenizer ideas such as merge scoring,
append-only vocabulary growth, and round-trip checks, but every accepted token
must come from admitted text. Its `tokenizer_manifest.json` records source
files, corpus hash, accepted and rejected candidates, and explicit
`pretrained_tokenizer: false` / `external_vocabulary: false` evidence. See
[Tokenizer manifests](../operate/tokenizer-manifests.md).
Manifest/report validation also checks the stable manifest hash, round-trip
status, full-answer-token audit, and report math before self-improvement treats
the tokenizer candidate as evidence.

Self-improvement attempts add one more check: `tokenizer_candidate_guard`.
That guard fails if a candidate silently changes the active tokenizer, fails
round-trip, imports pretrained or external vocabulary, or creates a protected
full-answer token. Passing it means the tokenizer candidate is safe evidence,
not that the neural model has learned more.

:::note

Studying structure is a docs-gated activity, not a training input. The structure
audit is recorded in `STRUCTURE_AUDIT.md`; adopting a shape requires evidence in
docs, tests, and run records — it never admits external content into the corpus.

:::

## Rule

To admit new training data, follow the ledger path: a source is either admitted
directly, or generated from admitted corpus files and then admitted in turn.

<ol className="qlm-steps">
<li><strong>Admit the source to the ledger</strong><p>New training data must be named in <code>corpus/ledger.json</code>, or generated from admitted corpus files and then admitted in turn.</p></li>
<li><strong>Open the two permissions explicitly</strong><p>A source's <code>allowed_for_curriculum_generation</code> and <code>allowed_for_training</code> flags decide whether it may reach curriculum and weights; both stay closed until the ledger opens them.</p></li>
<li><strong>Keep evaluation probes out of training</strong><p>Evaluation probes may be ledgered for provenance, but they are not training data unless the ledger explicitly allows it.</p></li>
</ol>

## What is next

<div className="qlm-next">

<a href="../../operate/closed-world-verifier/"><strong>Read next</strong><span>Closed-world verifier</span><small>The deterministic check for the four boundary flags before a training plan is trusted.</small></a>

<a href="../prompt-leakage/"><strong>Read next</strong><span>Prompt leakage</span><small>How held-out evaluation prompts are kept out of training lessons.</small></a>

<a href="../../operate/candidate-quarantine/"><strong>Read next</strong><span>Candidate quarantine</span><small>Where generated material waits for a ledger admission link before it becomes trainable.</small></a>

</div>
