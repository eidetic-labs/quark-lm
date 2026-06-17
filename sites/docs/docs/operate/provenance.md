---
title: Provenance
description: Corpus snapshots and diffs.
---

# Provenance

<p className="qlm-meta"><span>5 min read</span><span>For contributors</span><span>Updated 2026-06-16</span></p>

<div className="qlm-lead">

**What you will learn**

- What the two corpus-provenance artifacts record, and why a snapshot and a diff answer different questions.
- What the snapshot captures, and why hashes and record counts make it tamper-evident.
- How an empty diff separates a training change from a data change at the promotion gate.
- Why retrieval answering a probe is `memory-served`, not `weight-consolidated`.

</div>

Provenance is the record of *what the corpus was* at each promoted run, so a
weight change or eval change can be read against the exact data it came from. A
metric that improved is only interpretable if the corpus behind it is pinned.
QuarkLM pins it: every self-improvement report carries two corpus-provenance
artifacts, `corpus_snapshot.json` and `corpus_diff.json`, written from the
ledgered corpus and validated deterministically.

<div className="qlm-keypoint">

**Provenance records the corpus; it does not promote a run.**

These artifacts describe the admitted corpus only. They do not promote a run and
they do not move any weights. They make corpus changes visible next to the weight
and eval changes in the same report, which is what lets an audit ask whether a
result followed from new data, new training, or both.

</div>

## Two artifacts

<div className="qlm-grid">
<div><h4>corpus_snapshot.json</h4><p>The absolute record: the state of the ledgered corpus at this run. Compared against nothing — it stands on its own.</p></div>
<div><h4>corpus_diff.json</h4><p>The relative record: what changed in the corpus since the last promoted run, compared against the previous promoted snapshot.</p></div>
</div>

The snapshot is the absolute record; the diff is the relative one. Reading them
together answers a single question for each release: did the corpus move, and if
so, exactly where.

## What the snapshot captures

`corpus_snapshot.json` is built from `corpus/ledger.json` and the files it names.
For each admitted source it records:

- ledger source ids;
- file paths;
- training permissions;
- curriculum-generation permissions;
- file hashes;
- JSONL record counts;
- admitted memory ids.

The hashes and record counts make the snapshot tamper-evident: the same ledger
and the same files reproduce the same snapshot, and any silent edit to an
admitted file changes its hash. The two permission columns are the same boundary
the [admission workflow](../build/admission-workflow.md) sets when a fact is
ledgered — a source marked for curriculum generation but not for training is
recorded that way, so the snapshot also shows *which* admitted sources were even
eligible to influence weights.

## What the diff captures

`corpus_diff.json` compares the current snapshot to the snapshot of the previous
promoted run. It is the line in the report that says, in machine-checkable form,
whether the corpus changed and how. A run that kept its corpus sources unchanged
records that; a run that admitted new memories or regenerated a probe source
records the delta.

```text title="How the diff is computed"
previous promoted snapshot ──┐
                             ├─► corpus_diff.json  (added / changed sources,
current run snapshot ────────┘                      new admitted memory ids)
```

This is why a promotion decision can separate two questions that are easy to
confuse: a better eval number that arrives alongside an empty corpus diff came
from training, not from new data; a better number alongside newly admitted
memories has a data change that must be accounted for before the metric is
credited. The constraint-first promotion gate (see
[Corpus hygiene](./corpus-hygiene.md)) depends on that separation.

## How provenance reads against the rest of the report

Corpus provenance sits next to the other per-run evidence, and the distinctions
QuarkLM enforces elsewhere hold here too.

<div className="qlm-grid">
<div><h4>Eval up, diff empty</h4><p>The change came from training under guard, not from new admitted data.</p></div>
<div><h4>Eval up, new memory ids in diff</h4><p>The corpus moved; the metric must be read against that data change.</p></div>
<div><h4>Retrieval answers a new probe</h4><p>The corpus <em>contains</em> the answer — <code>memory-served</code>, not <code>weight-consolidated</code>.</p></div>
</div>

<div className="qlm-keypoint">

**Memory-served is not weight-consolidated**

Retrieval answering a probe exactly proves only that the admitted corpus holds
it. It does not prove the transformer learned it; the snapshot records what is
*admitted*, never what is consolidated into weights. That boundary is the same
one drawn in [Build](../build/index.md) and [Transformer](../build/transformer.md).

</div>

## Generated material is not provenance until admitted

Probes, paraphrases, and other generated material appear in provenance only once
they are admitted to the ledger. A source generated from admitted text and marked
eval-only is recorded with curriculum-generation permission but not training
permission, so it can shape evals without ever counting as training input.
Candidates that have not been admitted are not in the snapshot at all; they live
in [candidate quarantine](./candidate-quarantine.md) until the verifier clears
and the ledger admits them. The snapshot is therefore a record of the closed
world as it actually stands, not of everything a run considered.

## Run history is kept, not rewritten

Promoted self-improvement runs and unpromoted transformer screens keep their
original directory names so provenance stays exact, and failed runs are retained
as versioned diagnostic evidence rather than discarded (see
[Experiment registry](./experiment-registry.md)). Each run's snapshot and diff
stay attached to its report, so the corpus state behind any past metric remains
recoverable.

The version-by-version corpus and transformer-screen changes — which sources
moved at each release, which memories were admitted, and how the unpromoted
direct-answer transformer screens progressed — are catalogued in
[Transformer screen history](../build/transformer-screen-history.md). The
transformer remains unpromoted, blocked on `branch_diversity_target`; provenance
records the corpus behind each of those screens without claiming any of them
learned the answers.

:::note

Provenance is evidence, not a gate. It makes a corpus change visible in the same
report as the weight and eval changes; the decision to credit or reject a metric
is made by the constraint-first promotion gate in
[Corpus hygiene](./corpus-hygiene.md).

:::

<div className="qlm-next">
<a href="../corpus-hygiene/"><strong>Read next</strong><span>Corpus hygiene</span><small>The constraint-first promotion gate that depends on the corpus diff.</small></a>
<a href="../../build/admission-workflow/"><strong>Read</strong><span>Admission workflow</span><small>How a fact is ledgered and its training and curriculum permissions set.</small></a>
<a href="../../build/transformer/"><strong>Concept</strong><span>Transformer</span><small>Why memory-served is kept apart from weight-consolidated.</small></a>
</div>
